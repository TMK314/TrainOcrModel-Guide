"""
train.py

Trainiert model.build_model() mit CTC-Loss auf dem von prepare_dataset.py
erzeugten Datensatz. Folgt dem klassischen Keras-CTC-Pattern (eigene
Loss-Layer mit self.add_loss, siehe offizielles Keras-Beispiel
"Handwriting recognition using CTC"): die Trainings-Loss wird intern über
add_loss() berechnet, das Modell wird ohne explizites `loss=` kompiliert.

Nutzung:
    python train.py --dataset dataset --epochs 60 --batch-size 64 --out runs/v1
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

# Globale Absicherung zusätzlich zu jit_compile=False beim compile() weiter
# unten: manche TF/Keras-Versionen versuchen auch bei evaluate()/predict()
# (z. B. während der Validierung in fit()) automatisch XLA zu nutzen, was
# für CTCLoss auf der GPU fehlschlägt (kein XLA-Kernel registriert).
tf.config.optimizer.set_jit(False)

from model import build_model, MAX_SEQ_LEN

PAD_LABEL = -1  # Padding-Wert für Label-Sequenzen unterschiedlicher Länge innerhalb eines Batches


class CTCLossLayer(tf.keras.layers.Layer):
    """Berechnet den CTC-Loss und reicht y_pred unveraendert durch (add_loss-Pattern)."""

    def call(self, y_pred, labels, input_length, label_length):
        input_length = tf.expand_dims(input_length, axis=-1)
        label_length = tf.expand_dims(label_length, axis=-1)
        # Padding-Wert (-1) durch 0 ersetzen; label_length blendet die
        # ungültigen Positionen ohnehin korrekt aus ctc_batch_cost aus.
        safe_labels = tf.where(labels < 0, tf.zeros_like(labels), labels)
        loss = tf.keras.backend.ctc_batch_cost(safe_labels, y_pred, input_length, label_length)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


def load_split(dataset_dir: Path, split_name: str):
    data = np.load(dataset_dir / "data.npz", allow_pickle=True)
    mask = data["split"] == split_name
    return data["sequences"][mask], data["label_indices"][mask]


def make_dataset(sequences, label_indices, max_label_len, batch_size, shuffle):
    def gen():
        for seq, labels in zip(sequences, label_indices):
            padded_labels = np.full(max_label_len, PAD_LABEL, dtype=np.int32)
            padded_labels[: len(labels)] = labels
            yield {
                "ink": seq.astype(np.float32),
                "labels": padded_labels,
                "input_length": np.int32(MAX_SEQ_LEN),
                "label_length": np.int32(len(labels)),
            }

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "ink": tf.TensorSpec(shape=(MAX_SEQ_LEN, 3), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(max_label_len,), dtype=tf.int32),
            "input_length": tf.TensorSpec(shape=(), dtype=tf.int32),
            "label_length": tf.TensorSpec(shape=(), dtype=tf.int32),
        },
    )
    if shuffle:
        ds = ds.shuffle(4096)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_training_model(num_labels: int, max_label_len: int):
    base_model = build_model(num_labels)

    labels_in = tf.keras.Input(shape=(max_label_len,), dtype="int32", name="labels")
    input_length_in = tf.keras.Input(shape=(), dtype="int32", name="input_length")
    label_length_in = tf.keras.Input(shape=(), dtype="int32", name="label_length")

    output = CTCLossLayer(name="ctc_loss")(
        base_model.output, labels_in, input_length_in, label_length_in
    )

    training_model = tf.keras.Model(
        inputs={
            "ink": base_model.input,
            "labels": labels_in,
            "input_length": input_length_in,
            "label_length": label_length_in,
        },
        outputs=output,
    )
    return training_model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help=(
            "Optional: Pfad zu einer bestehenden training_weights.weights.h5 "
            "(Fine-Tuning statt Training von Grund auf). labels.txt des "
            "Datensatzes MUSS dieselbe Zeichentabelle wie beim ursprünglichen "
            "Lauf sein, sonst passt die Ausgabeschicht nicht."
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = (args.dataset / "labels.txt").read_text(encoding="utf-8").splitlines()
    labels = [l for l in labels if l != ""]
    num_labels = len(labels)
    print(f"{num_labels} Zeichen in labels.txt geladen.")

    train_seq, train_labels = load_split(args.dataset, "train")
    val_seq, val_labels = load_split(args.dataset, "val")
    max_label_len = max(
        max((len(l) for l in train_labels), default=1),
        max((len(l) for l in val_labels), default=1),
    )
    print(f"Train: {len(train_seq)}  Val: {len(val_seq)}  max_label_len: {max_label_len}")

    train_ds = make_dataset(train_seq, train_labels, max_label_len, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_seq, val_labels, max_label_len, args.batch_size, shuffle=False)

    training_model, base_model = build_training_model(num_labels, max_label_len)

    if args.init_weights:
        # Architektur ist bei gleicher labels.txt identisch zum urspruenglichen
        # Lauf (CTCLossLayer hat keine eigenen trainierbaren Parameter) -
        # direktes, positionsbasiertes Laden funktioniert daher unveraendert.
        training_model.load_weights(str(args.init_weights))
        print(f"Vortrainierte Gewichte geladen aus: {args.init_weights}")

    training_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), jit_compile=False)
    training_model.summary()

    weights_path = args.out / "training_weights.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path), monitor="val_loss",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    training_model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    training_model.load_weights(weights_path)
    best_model_path = args.out / "best_model.keras"
    base_model.save(best_model_path)
    print(f"Bestes Inferenz-Modell gespeichert unter: {best_model_path}")


if __name__ == "__main__":
    main()