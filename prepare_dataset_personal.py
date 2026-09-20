"""
prepare_dataset_personal.py

Wandelt die von OcrTrainingCaptureModal.ts exportierten JSON-Dateien (rohe,
unvereinfachte Stift-Punkte je Wort) in ein zu train.py kompatibles
dataset/data.npz um. Nutzt BEWUSST die vorhandene labels.txt des
Basismodells weiter (KEINE neue Zeichentabelle!), damit die Klassenindizes
zu training_weights.weights.h5 passen.

Da die Rohdaten hier höchste Punktdichte haben, wird -- wie beim
DeepWriting-Pfad -- pro Beispiel eine zufällige Douglas-Peucker-
Vereinfachung + anschließende Interpolation angewendet, damit die
Trainingsverteilung der zur Laufzeit im Plugin erzeugten entspricht.

Nutzung:
    python prepare_dataset_personal.py \
        --input ocr-training-1234567890.json [weitere --input ...] \
        --labels dataset/labels.txt --out dataset_personal
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ink_utils import build_normalized_ink, simplify_stroke_random_tolerance, stroke_bounds, merge_bounds, normalize_text_for_training

MAX_SEQ_LEN = 256
VAL_SPLIT = 0.15
DEFAULT_MAX_SIMPLIFY_FRACTION = 0.05
DEFAULT_RESAMPLE_SPACING = 0.02


def load_samples(paths: List[Path]) -> List[dict]:
    samples = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples.extend(data["samples"])
    return samples


def build_example(
    sample: dict, rng: random.Random, max_simplify_fraction: float, resample_spacing: float
) -> Tuple[List[List[float]], int]:
    strokes = [
        [{"x": p["x"], "y": p["y"], "t": p["t"]} for p in stroke]
        for stroke in sample["strokes"]
    ]
    bounds = merge_bounds([stroke_bounds(s) for s in strokes])

    if max_simplify_fraction > 0:
        strokes = [simplify_stroke_random_tolerance(s, bounds, rng, max_simplify_fraction) for s in strokes]

    return build_normalized_ink(strokes, bounds, MAX_SEQ_LEN, resample_spacing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--labels", type=Path, required=True, help="labels.txt des BESTEHENDEN Modells.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-simplify-fraction", type=float, default=DEFAULT_MAX_SIMPLIFY_FRACTION)
    parser.add_argument("--resample-spacing", type=float, default=DEFAULT_RESAMPLE_SPACING)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels = [l for l in args.labels.read_text(encoding="utf-8").splitlines() if l != ""]
    label_to_index = {c: i for i, c in enumerate(labels)}

    raw_samples = load_samples(args.input)
    print(f"{len(raw_samples)} exportierte Wort-Beispiele gefunden.")

    texts, sequences, valid_lengths = [], [], []
    skipped = 0

    for sample in tqdm(raw_samples, desc="Verarbeite Beispiele"):
        text = normalize_text_for_training(sample["text"])
        if not text:                     # <-- neue Zeile
            skipped += 1                 # <-- neue Zeile
            continue                     # <-- neue Zeile
        if not sample["strokes"]:
            continue
        unknown = [c for c in text if c not in label_to_index]
        if unknown:
            print(f"  Übersprungen ({unknown!r} nicht in labels.txt): {sample['text']!r}")
            skipped += 1
            continue

        seq, valid_len = build_example(sample, rng, args.max_simplify_fraction, args.resample_spacing)
        texts.append(text)
        sequences.append(seq)
        valid_lengths.append(valid_len)

    if not texts:
        raise RuntimeError("Kein gültiges Beispiel übrig – Eingabedateien und labels.txt prüfen.")

    args.out.mkdir(parents=True, exist_ok=True)
    sequences_np = np.array(sequences, dtype=np.float32)
    valid_lengths_np = np.array(valid_lengths, dtype=np.int32)
    label_indices = [np.array([label_to_index[c] for c in t], dtype=np.int32) for t in texts]

    n = len(texts)
    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    n_val = max(1, int(n * VAL_SPLIT))
    val_idx = set(indices[:n_val])
    split = np.array(["val" if i in val_idx else "train" for i in range(n)])

    np.savez(
        args.out / "data.npz",
        sequences=sequences_np, valid_lengths=valid_lengths_np,
        label_indices=np.array(label_indices, dtype=object),
        texts=np.array(texts, dtype=object), split=split,
    )
    # WICHTIG: unveraendert kopieren, NICHT aus eigenem Charset neu erzeugen.
    (args.out / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

    print(f"Fertig: {n} Beispiele ({n - n_val} train / {n_val} val), {skipped} übersprungen.")


if __name__ == "__main__":
    main()