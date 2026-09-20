"""
merge_datasets.py

Mischt den kleinen personalisierten Datensatz mit einem Teil des grossen
Basis-Datensatzes, um "catastrophic forgetting" beim Fine-Tuning zu
vermeiden. Beide Datensaetze muessen dieselbe labels.txt verwenden.

Nutzung:
    python merge_datasets.py --base dataset --personal dataset_personal \
        --base-fraction 0.25 --oversample 8 --out dataset_finetune
"""
import argparse
import random
from pathlib import Path

import numpy as np


def load(dataset_dir: Path):
    data = np.load(dataset_dir / "data.npz", allow_pickle=True)
    labels = [l for l in (dataset_dir / "labels.txt").read_text(encoding="utf-8").splitlines() if l != ""]
    return data, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--personal", type=Path, required=True)
    parser.add_argument("--base-fraction", type=float, default=0.25)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_data, base_labels = load(args.base)
    personal_data, personal_labels = load(args.personal)
    if base_labels != personal_labels:
        raise SystemExit("labels.txt stimmen nicht überein – prepare_dataset_personal.py mit --labels auf die Basis-labels.txt aufrufen.")

    rng = random.Random(args.seed)

    def subset(data, split_name, fraction, oversample):
        idx = np.where(data["split"] == split_name)[0].tolist()
        if split_name == "train":
            rng.shuffle(idx)
            if fraction < 1.0:
                idx = idx[: int(len(idx) * fraction)]
            idx = idx * oversample
        return idx

    base_train = subset(base_data, "train", args.base_fraction, 1)
    base_val = subset(base_data, "val", 1.0, 1)
    pers_train = subset(personal_data, "train", 1.0, args.oversample)
    pers_val = subset(personal_data, "val", 1.0, 1)

    def gather(data, idx):
        return (
            data["sequences"][idx],
            data["valid_lengths"][idx],
            data["label_indices"][idx].tolist(),   # Holt alle Label‑Indices auf einmal
            data["texts"][idx]
        )

    b_seq, b_len, b_lab, b_txt = gather(base_data, base_train + base_val)
    p_seq, p_len, p_lab, p_txt = gather(personal_data, pers_train + pers_val)

    sequences = np.concatenate([b_seq, p_seq], axis=0)
    valid_lengths = np.concatenate([b_len, p_len], axis=0)
    label_indices = np.array(b_lab + p_lab, dtype=object)
    texts = np.concatenate([b_txt, p_txt], axis=0)
    split = np.array(
        ["train"] * len(base_train) + ["val"] * len(base_val) +
        ["train"] * len(pers_train) + ["val"] * len(pers_val)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "data.npz", sequences=sequences, valid_lengths=valid_lengths,
             label_indices=label_indices, texts=texts, split=split)
    (args.out / "labels.txt").write_text("\n".join(base_labels), encoding="utf-8")

    n_train = int((split == "train").sum())
    n_val = int((split == "val").sum())
    print(f"Gemischt: {n_train} train ({len(base_train)} Basis + {len(pers_train)} personalisiert) / {n_val} val")


if __name__ == "__main__":
    main()