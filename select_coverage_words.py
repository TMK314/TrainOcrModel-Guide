"""
select_coverage_words.py

Waehlt aus einer grossen Wortliste (z. B. eine deutsche Frequenzliste, ein
Wort pro Zeile) greedy eine moeglichst kleine Teilmenge aus, die jedes
Zeichen aus labels.txt mindestens --min-count mal abdeckt.

Nutzung:
    python select_coverage_words.py --wordlist de_words.txt --labels dataset/labels.txt \
        --min-count 30 --out my_training_words.txt
"""
import argparse
from collections import Counter
from pathlib import Path
from ink_utils import normalize_text_for_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordlist", type=Path, required=True, help="Grosse Wortliste, ein Wort pro Zeile.")
    parser.add_argument("--labels", type=Path, required=True, help="labels.txt des vorhandenen Modells.")
    parser.add_argument("--min-count", type=int, default=30, help="Ziel-Mindestanzahl je Zeichen.")
    parser.add_argument("--max-words", type=int, default=800, help="Obergrenze, um die Liste handhabbar zu halten.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = [l for l in args.labels.read_text(encoding="utf-8").splitlines() if l != ""]
    target_chars = set(labels)

    candidates = [
        w.strip() for w in args.wordlist.read_text(encoding="utf-8").splitlines()
        if w.strip() and set(w.strip()) & target_chars
    ]
    candidates.sort(key=len)  # kürzere Wörter leicht bevorzugen

    counts: Counter = Counter()
    selected: list[str] = []

    while len(selected) < args.max_words:
        need = {c: max(0, args.min_count - counts[c]) for c in target_chars}
        if all(v == 0 for v in need.values()):
            break

        best_word, best_score = None, 0
        for word in candidates:
            # Abdeckung anhand der NORMALISIERTEN Zeichen zählen (Umlaute
            # zählen auf ihre Basisform ein, Satzzeichen zählen gar nicht),
            # da genau diese Zeichen später tatsächlich als CTC-Ziel dienen.
            normalized_chars = normalize_text_for_training(word)
            score = sum(1 for c in normalized_chars if need.get(c, 0) > 0)
            if score > best_score:
                best_word, best_score = word, score

        if not best_word or best_score == 0:
            break

        selected.append(best_word)
        counts.update(normalize_text_for_training(best_word))
        candidates.remove(best_word)

    args.out.write_text("\n".join(selected), encoding="utf-8")

    missing = [c for c in sorted(target_chars) if counts[c] < args.min_count]
    print(f"{len(selected)} Wörter ausgewählt -> {args.out}")
    if missing:
        print(f"WARNUNG: {len(missing)} Zeichen erreichen --min-count nicht: {missing}")
        print("Ergänzt diese Zeichen ggf. manuell (Zahlen, Satzzeichen, seltene Buchstaben).")
    else:
        print("Alle Zeichen erreichen die Ziel-Mindestanzahl.")


if __name__ == "__main__":
    main()