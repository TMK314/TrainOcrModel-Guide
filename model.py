"""
model.py

Definiert die CTC-Handschrift-Erkennungsarchitektur: rein Faltungs-basiert
(keine RNN-/LSTM-/GRU-Layer), damit die Konvertierung nach TFLite und die
Ausführung über die WASM-Runtime von @tensorflow/tfjs-tflite unproblematisch
bleibt (RNN-Operatoren sind in der WASM-Runtime nicht zuverlässig
unterstützt, Conv1D dagegen schon).

Bewusst KEIN zeitliches Downsampling (durchgängig stride=1): die Anzahl der
Ausgabe-Zeitschritte T_out entspricht dadurch immer exakt der
Eingabe-Sequenzlänge T_in. Das erlaubt eine einfache und robuste
Korrespondenz zur Greedy-CTC-Dekodierung in OcrModelRunner.decodeCtc() im
Plugin (siehe dortiger Patch: es wird nur bis zur tatsächlichen,
ungepaddeten Punktanzahl dekodiert).

Eingabe:  [batch, MAX_SEQ_LEN, 3]   ([x, y, t] pro Punkt, wie
          OcrStrokeCollector.buildNormalizedInk() im Plugin erzeugt)
Ausgabe:  [batch, MAX_SEQ_LEN, NUM_CLASSES]  Softmax-Wahrscheinlichkeiten
          pro Zeitschritt. Die LETZTE Klasse (Index NUM_CLASSES-1) ist der
          CTC-Blank (siehe OcrModelRunner.decodeCtc: blankIndex = labels.length).
"""
import tensorflow as tf

MAX_SEQ_LEN = 256  # MUSS zu settings.ocrMaxSequenceLength im Plugin passen
FEATURES_PER_POINT = 3


def build_model(num_labels: int) -> tf.keras.Model:
    """num_labels = Anzahl Zeichen in labels.txt (OHNE Blank).
    Der Blank-Index wird als zusaetzliche, letzte Klasse angehaengt."""
    num_classes = num_labels + 1  # + Blank

    inputs = tf.keras.Input(shape=(MAX_SEQ_LEN, FEATURES_PER_POINT), name="ink")

    x = inputs
    # Anfangs-Blöcke: reine Merkmalsextraktion, KEIN Downsampling (stride=1).
    for filters in (64, 96, 128):
        x = tf.keras.layers.Conv1D(filters, kernel_size=5, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

    # Dilated-Conv-Residualblöcke für größeren zeitlichen Kontext, ebenfalls
    # ohne Downsampling.
    for dilation in (1, 2, 4, 8, 4, 2):
        residual = x
        x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", dilation_rate=dilation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Add()([x, residual])

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="ink_ctc_model")
