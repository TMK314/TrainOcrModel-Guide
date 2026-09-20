"""
test_inference.py

Lädt das exportierte .tflite direkt über tf.lite.Interpreter und dekodiert
mit EXAKT derselben Greedy-CTC-Logik wie OcrModelRunner.decodeCtc() im
Plugin (Python-Nachbau, siehe Kommentare) – bewusst NICHT über Kerasʼ
eigene CTC-Decode-Hilfsfunktionen, damit ihr euch auf das reale
Laufzeitverhalten im Plugin verlassen könnt.

Nutzung:
    python test_inference.py --model runs/v1/digitalink.tflite --labels dataset/labels.txt --dataset dataset --num-samples 20
"""
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def greedy_ctc_decode(frames: np.ndarray, labels: list) -> str:
    """1:1-Nachbau von OcrModelRunner.decodeCtc() (Greedy-CTC, Blank = letzte Klasse)."""
    blank_index = len(labels)
    prev_class = -1
    text = ""
    for frame in frames:
        best_idx = int(np.argmax(frame))
        if best_idx != blank_index and best_idx != prev_class:
            text += labels[best_idx]
        prev_class = best_idx
    return text


def character_error_rate(pred: str, target: str) -> float:
    """Levenshtein-Distanz / Zeichenanzahl des Zielworts."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = list(range(len(pred) + 1))
    for i in range(1, len(target) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(pred) + 1):
            cur = dp[j]
            dp[j] = prev if target[i - 1] == pred[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(pred)] / len(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    labels = [l for l in args.labels.read_text(encoding="utf-8").splitlines() if l != ""]

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    data = np.load(args.dataset / "data.npz", allow_pickle=True)
    mask = data["split"] == "val"
    sequences = data["sequences"][mask]
    valid_lengths = data["valid_lengths"][mask]
    texts = data["texts"][mask]

    indices = random.sample(range(len(sequences)), min(args.num_samples, len(sequences)))

    total_cer = 0.0
    for i in indices:
        seq = sequences[i][np.newaxis, ...].astype(np.float32)
        interpreter.set_tensor(input_details["index"], seq)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]  # [T, num_classes]

        # Wie in OcrModelRunner.ts: nur bis zur tatsächlichen (ungepaddeten)
        # Punktanzahl dekodieren, da das Modell keine Zeitachsen-Downsampling
        # vornimmt (T_out == T_in).
        valid_len = int(valid_lengths[i])
        predicted = greedy_ctc_decode(output[:valid_len], labels)

        target = str(texts[i])
        cer = character_error_rate(predicted, target)
        total_cer += cer
        print(f'  Ziel: {target!r:20}  Erkannt: {predicted!r:20}  CER: {cer:.2f}')

    print(f"\nDurchschnittliche CER über {len(indices)} Beispiele: {total_cer / len(indices):.3f}")


if __name__ == "__main__":
    main()
