"""
prepare_dataset_deepwriting.py

Liest die DeepWriting/IAM-OnDB-JSON-Dateien (Format siehe extended_dataset/readme,
per --sample verifiziert) und baut daraus den Trainingsdatensatz
(dataset/data.npz + dataset/labels.txt).

Format pro Datei (eine JSON-Datei je Formular/Schreibprobe):
    { "sample0": {...}, "sample1": {...}, ... }
Jedes "sampleN" ist eine ganze Zeile/ein ganzer Satz mit u. a.:
    - "word_stroke": flache Liste ALLER Punkte der Zeile, je
      {"x": "224", "y": "115", "ev": "0"|"1"|"2", "ts": "1471352798684", ...}
      (x, y, ts sind Strings – müssen konvertiert werden; ts = Unix-Millisekunden)
    - "wholeword_segments": Liste der Wörter dieser Zeile, je
      {"ocr_label": "Summons", "ranges": [[0,1,2,...,19,21,...]], "recognition_is_correct": True, ...}
      -> ranges[0] ist die Liste der Punkt-INDIZES in word_stroke, die zu
         diesem Wort gehören (exakte Ground-Truth-Segmentierung – anders als
         beim rohen IAM-OnDB-Pfad in prepare_dataset_iam.py wird hier NICHTS
         geometrisch geraten).
    - "is_word_segmentation_valid" (bool), "is_sentence_misspelled" (STRING
      "true"/"false", kein echter bool!) – zur Qualitätsfilterung genutzt.

AUGMENTATION (Vereinfachung + Interpolation): Das Plugin speichert Striche
NICHT roh, sondern nach Douglas-Peucker-Vereinfachung (StrokeSimplify.ts),
und interpoliert sie vor der OCR-Erkennung wieder (resampleNormalizedPoints
in OcrStrokeCollector.ts). Damit das Modell auf genau dieser Datenverteilung
trainiert wird, wendet dieses Skript denselben zweistufigen Prozess (siehe
ink_utils.py: simplify_stroke_random_tolerance + build_normalized_ink) mit
einer zufälligen, größenrelativen Vereinfachungstoleranz auf jedes
Trainingsbeispiel an, BEVOR normalisiert wird.

LIZENZ-HINWEIS: Die "Deepwriting Dataset"-Unterordner stehen unter der
mitgelieferten CC-BY-NC-SA-artigen Lizenz (license.pdf). Die "Iamondb
Dataset"-Unterordner enthalten IAM-OnDB-Rohdaten und unterliegen weiterhin
IAM-OnDBs eigenen Nutzungsbedingungen (https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database),
auch wenn der Download hier über den ETH-Mirror lief. Prüft beide, bevor ihr
ein daraus trainiertes Modell veröffentlicht.

Nutzung:
    python prepare_dataset_deepwriting.py --data-dir extended_dataset --sample 5
    python prepare_dataset_deepwriting.py --data-dir extended_dataset --out dataset
"""
import argparse
import json
import random
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from tqdm import tqdm

from ink_utils import (
    build_normalized_ink as ink_build_normalized_ink,
    simplify_stroke_random_tolerance,
    stroke_bounds,
    merge_bounds,
    normalize_text_for_training,
)

MAX_SEQ_LEN = 256  # MUSS zu settings.ocrMaxSequenceLength im Plugin passen
VAL_SPLIT = 0.1
DEFAULT_RESAMPLE_SPACING = 0.02        # MUSS in der Größenordnung zu settings.ocrResampleSpacing im Plugin passen
DEFAULT_MAX_SIMPLIFY_FRACTION = 0.05   # deckt settings.strokeSimplifyTolerance (0-3pt) für typische Wortgrößen ab


def iter_json_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*-segmented.json")
    yield from root.rglob("*-segmented-nonorm.json")


def is_truthy(value) -> bool:
    """Die JSON-Dateien nutzen teils echte bools, teils die Strings "true"/"false"."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def split_word_points_into_strokes(points: List[dict]) -> List[List[dict]]:
    """
    Trennt die flache Punktliste eines Worts (bereits als float-Dicts mit
    'x','y','t','ev') anhand des 'ev'-Feldes in einzelne Pen-Down/Up-
    Teilstriche, analog zu den einzelnen FreehandObject-Strichen im Plugin.

    ACHTUNG: Die genaue Bedeutung von "ev" ist nicht zweifelsfrei
    dokumentiert (siehe README, Abschnitt "Hinweis zum Prüfstand"). Diese
    Funktion nimmt die gängige Konvention ev == 0 -> Stift-Aufsetzen (Beginn
    eines neuen Teilstrichs) an. Liefert das keine sinnvolle Mehrfach-
    Segmentierung, fallen alle Punkte auf einen einzigen Teilstrich zurück –
    das entspricht exakt dem Verhalten dieses Skripts VOR dieser Erweiterung
    (ein Wort = ein einziger Polygonzug).
    """
    strokes: List[List[dict]] = []
    current: List[dict] = []
    for p in points:
        if str(p.get("ev")) == "0" and current:
            strokes.append(current)
            current = []
        current.append(p)
    if current:
        strokes.append(current)
    return strokes if len(strokes) > 1 else [points]


def build_normalized_ink(
    points: List[dict],
    max_seq_len: int,
    rng: random.Random,
    max_simplify_fraction: float,
    resample_spacing: float,
) -> Tuple[List[List[float]], int]:
    """
    Wandelt die rohen {"x","y","ts",...}-Punkt-Dicts dieses Datensatz-Schemas
    (Strings -> float) in Teilstriche um, wendet darauf -- als Data
    Augmentation -- Ramer-Douglas-Peucker mit einer zufälligen, relativ zur
    Wort-Diagonale skalierten Toleranz an (simuliert die im Plugin bereits
    vor dem Speichern vereinfachten Striche, siehe StrokeSimplify.ts) und
    interpoliert danach wieder auf die erwartete Punktdichte (siehe
    resample_normalized_points in ink_utils.py / OcrStrokeCollector.ts).

    max_simplify_fraction == 0 deaktiviert die Vereinfachung (rohe
    Originaldaten). resample_spacing == 0 deaktiviert die Interpolation.
    """
    float_points = [
        {"x": float(p["x"]), "y": float(p["y"]), "t": float(p["ts"]), "ev": p.get("ev")}
        for p in points
    ]
    strokes = split_word_points_into_strokes(float_points)
    bounds = merge_bounds([stroke_bounds(s) for s in strokes])

    if max_simplify_fraction > 0:
        strokes = [
            simplify_stroke_random_tolerance(s, bounds, rng, max_simplify_fraction)
            for s in strokes
        ]

    return ink_build_normalized_ink(strokes, bounds, max_seq_len, resample_spacing)


def process_file(
    path: Path,
    rng: random.Random,
    max_simplify_fraction: float,
    resample_spacing: float,
):
    """Generator über (word_text, ink_sequence, valid_length) für eine JSON-Datei."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    for sample in data.values():
        if not isinstance(sample, dict):
            continue
        if not is_truthy(sample.get("is_word_segmentation_valid", True)):
            continue
        if is_truthy(sample.get("is_sentence_misspelled", False)):
            continue

        word_stroke = sample.get("word_stroke")
        segments = sample.get("wholeword_segments")
        if not word_stroke or not segments:
            continue

        for word in segments:
            if not is_truthy(word.get("recognition_is_correct", True)):
                continue
            raw_text = word.get("ocr_label") or word.get("recognized_label")
            if not raw_text:
                continue
            text = normalize_text_for_training(raw_text)
            if not text:
                continue
            ranges = word.get("ranges")
            if not ranges or not ranges[0]:
                continue
            indices = ranges[0]

            try:
                points = [word_stroke[i] for i in indices]
            except IndexError:
                continue
            if len(points) < 2:
                continue

            seq, valid_len = build_normalized_ink(
                points, MAX_SEQ_LEN, rng, max_simplify_fraction, resample_spacing
            )
            yield text, seq, valid_len


def process_all(
    root: Path,
    rng: random.Random,
    max_simplify_fraction: float,
    resample_spacing: float,
):
    for path in iter_json_files(root):
        yield from process_file(path, rng, max_simplify_fraction, resample_spacing)


def run_sample(
    root: Path,
    n: int,
    rng: random.Random,
    max_simplify_fraction: float,
    resample_spacing: float,
) -> None:
    samples = []
    for text, _seq, valid_len in process_all(root, rng, max_simplify_fraction, resample_spacing):
        samples.append((text, valid_len))
        if len(samples) >= n * 20:
            break

    if not samples:
        print("Keine Beispiele gefunden. Pfad zu --data-dir korrekt gesetzt?")
        return

    print(f"{len(samples)} Beispiele gesammelt (zeige {min(n, len(samples))} zufällige):\n")
    for text, valid_len in random.sample(samples, min(n, len(samples))):
        print(f"  Text: {text!r:20}  Punkte: {valid_len:4d}")


def run_full(
    root: Path,
    out_dir: Path,
    rng: random.Random,
    max_simplify_fraction: float,
    resample_spacing: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    sequences: List[List[List[float]]] = []
    valid_lengths: List[int] = []
    charset = set()

    for text, seq, valid_len in tqdm(
        process_all(root, rng, max_simplify_fraction, resample_spacing), desc="Verarbeite Wörter"
    ):
        texts.append(text)
        sequences.append(seq)
        valid_lengths.append(valid_len)
        charset.update(text)

    if not texts:
        raise RuntimeError("Kein einziges Beispiel extrahiert – siehe --sample zur Fehlersuche.")

    labels = sorted(charset)
    label_to_index = {c: i for i, c in enumerate(labels)}

    sequences_np = np.array(sequences, dtype=np.float32)
    valid_lengths_np = np.array(valid_lengths, dtype=np.int32)
    label_indices = [np.array([label_to_index[c] for c in t], dtype=np.int32) for t in texts]

    n = len(texts)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    n_val = max(1, int(n * VAL_SPLIT))
    val_idx = set(indices[:n_val])
    split = np.array(["val" if i in val_idx else "train" for i in range(n)])

    np.savez(
        out_dir / "data.npz",
        sequences=sequences_np,
        valid_lengths=valid_lengths_np,
        label_indices=np.array(label_indices, dtype=object),
        texts=np.array(texts, dtype=object),
        split=split,
    )
    (out_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

    n_train = n - n_val
    print(f"Fertig: {n} Wort-Beispiele ({n_train} train / {n_val} val), {len(labels)} Zeichen.")
    print(f"-> {out_dir / 'data.npz'}")
    print(f"-> {out_dir / 'labels.txt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help='Entpackter extended_dataset-Ordner (enthält "Deepwriting Dataset/" und "Iamondb Dataset/")',
    )
    parser.add_argument("--out", type=Path, help="Ausgabeordner für data.npz + labels.txt")
    parser.add_argument("--sample", type=int, default=0, help="Nur N Beispiele zur Kontrolle anzeigen, nichts speichern.")
    parser.add_argument(
        "--max-simplify-fraction",
        type=float,
        default=DEFAULT_MAX_SIMPLIFY_FRACTION,
        help=(
            "Obergrenze (relativ zur Wort-Diagonale) für die zufällige Douglas-Peucker-"
            "Vereinfachungs-Toleranz je Trainingsbeispiel (Data Augmentation, simuliert "
            "settings.strokeSimplifyTolerance im Plugin). 0 deaktiviert die Vereinfachung."
        ),
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=DEFAULT_RESAMPLE_SPACING,
        help=(
            "Ziel-Punktabstand (relativ zur Wortgröße) für die Interpolation nach der "
            "Vereinfachung, siehe settings.ocrResampleSpacing im Plugin. 0 deaktiviert "
            "die Interpolation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed für die Augmentation (Toleranz-Ziehung).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.sample > 0:
        run_sample(args.data_dir, args.sample, rng, args.max_simplify_fraction, args.resample_spacing)
    else:
        if not args.out:
            raise SystemExit("--out ist erforderlich, wenn --sample nicht gesetzt ist.")
        run_full(args.data_dir, args.out, rng, args.max_simplify_fraction, args.resample_spacing)


if __name__ == "__main__":
    main()