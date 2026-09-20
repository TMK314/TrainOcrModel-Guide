"""
convert_to_tflite.py

Konvertiert das trainierte Inferenz-Modell (ink -> softmax, aus train.py)
nach TFLite mit fixer Eingabeform [1, MAX_SEQ_LEN, 3], kompatibel zu
OcrModelRunner.ts. Nutzt ausschliesslich TFLITE_BUILTINS (keine SELECT_TF_OPS
noetig, da model.py bewusst nur Conv1D/Dense/BatchNorm verwendet).

Nutzung:
    python convert_to_tflite.py --model runs/v1/best_model.keras --labels dataset/labels.txt --out runs/v1/digitalink.tflite
"""
import argparse
from pathlib import Path

import tensorflow as tf

from model import MAX_SEQ_LEN, FEATURES_PER_POINT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Pfad zu best_model.keras")
    parser.add_argument("--labels", type=Path, required=True, help="Pfad zu labels.txt (nur zur Konsistenzprüfung)")
    parser.add_argument("--out", type=Path, required=True, help="Ausgabepfad für die .tflite-Datei")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    labels = [l for l in args.labels.read_text(encoding="utf-8").splitlines() if l != ""]
    expected_classes = len(labels) + 1  # + Blank
    actual_classes = model.output_shape[-1]
    if actual_classes != expected_classes:
        raise ValueError(
            f"Modell hat {actual_classes} Ausgabeklassen, labels.txt + Blank ergibt aber "
            f"{expected_classes}. Wurden Modell und Zeichentabelle mit demselben Datensatz-Lauf erzeugt?"
        )

    # Eingabeform fixieren: [1, MAX_SEQ_LEN, FEATURES_PER_POINT] – wichtig,
    # damit OcrModelRunner.ts einen festen Tensor derselben Form erzeugen kann.
    fixed_input = tf.keras.Input(shape=(MAX_SEQ_LEN, FEATURES_PER_POINT), batch_size=1, name="ink")
    fixed_output = model(fixed_input)
    fixed_model = tf.keras.Model(fixed_input, fixed_output)

    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite_model)
    print(f"Geschrieben: {args.out} ({len(tflite_model) / 1024:.1f} KB)")
    print(f"Eingabeform: [1, {MAX_SEQ_LEN}, {FEATURES_PER_POINT}]  Ausgabeklassen: {actual_classes} (Blank = Index {actual_classes - 1})")


if __name__ == "__main__":
    main()
