# Workflow to Train Your Own OCR Model
## Requirements
This workflow uses Google Colab, so you'll need a Google account. To train the model on your own handwriting, you first need to create a dataset. You can do this directly in the [PDF-Composer](https://github.com/TMK314/pdf-composer) plugin using the command **"Gather OCR training data."** You'll also need the Python scripts from this repository — just upload them to Colab alongside your training data.
## Preparation

### DeepWriting Dataset (ETH Zürich)
If you'd rather start from an existing dataset, you can download the DeepWriting dataset with:
```bash

!wget https://files.ait.ethz.ch/projects/deepwriting/extended_dataset.zip
!unzip extended_dataset.zip -d extended_dataset
```
Please observe the license terms — by downloading the dataset you agree to them.
### Set Up the Environment

Once you're logged into Google Colab, switch the runtime type to **"Python 3" with a "T4 GPU."** Then upload all your training data and the Python scripts.

```bash
!pip install -r requirements.txt
```
## Prepare the Dataset
You can adjust the simplification and interpolation parameters to match your setup:
```bash
!python prepare_dataset_deepwriting.py --data-dir extended_dataset --out dataset \
    --max-simplify-fraction 0.05 --resample-spacing 0.02
```
**To sanity-check the dataset before committing to a full run**, use the `--sample` flag:
```bash
!python prepare_dataset_deepwriting.py --data-dir extended_dataset --sample 5
```
This prints a handful of random (word, point-count) pairs so you can verify by eye that the words look plausible (no empty strings, no encoding artifacts). If everything looks good, drop the `--sample` flag to build the full dataset.
### Build the Full Dataset

```bash
!python prepare_dataset_deepwriting.py --data-dir extended_dataset --out dataset
```
This searches `extended_dataset/` recursively and creates:
- `dataset/data.npz` — ink sequences + text, with a train/val split
- `dataset/labels.txt` — character table, one token per line
### Umlauts and Special Characters
Umlauts are projected to their base form (e.g. `Ä → A`), and punctuation/special characters are removed from the target text. The corresponding pen strokes remain in the input, teaching the model to emit **Blank** for these shapes instead of guessing.

---
## Your Own Dataset (Fine-Tuning on Personal Handwriting)
Once you have a base model (`runs/v1/`), you can fine-tune it on your own handwriting to make it noticeably more accurate for you:
1. **Collect training data**  
    In the plugin, run **"Gather OCR training data"**, paste in a word list, and write the words by hand. Export the result as `ocr-training-*.json`.  
2. **Prepare the personal dataset**
```bash
    !python prepare_dataset_personal.py --input ocr-training-*.json --labels dataset/labels.txt --out dataset_personal
```
    Uses the base model's `labels.txt` so class indices stay compatible.
3. **Merge with the base dataset** to avoid catastrophic forgetting during fine-tuning:
```bash
    !python merge_datasets.py --base dataset --personal dataset_personal --base-fraction 0.25 --oversample 8 --out dataset_finetune
```    
4. **Fine-tune from the previous weights** (see note below):

```bash
    !python train.py --dataset dataset_finetune --out runs/v2 \
        --init-weights runs/v1/training_weights.weights.h5 --lr 1e-4 --epochs 30
```
---
## Training (from scratch)

```bash

!python train.py --dataset dataset --epochs 60 --batch-size 64 --out runs/v1
```
`train.py` saves the best model (lowest validation CTC loss) under `runs/v1/best_model.keras`, plus a reusable checkpoint at `runs/v1/training_weights.weights.h5`.
## Convert to TFLite

```bash
!python convert_to_tflite.py --model runs/v1/best_model.keras --labels dataset/labels.txt --out runs/v1/digitalink.tflite
```
## Sanity Check

```bash
!python test_inference.py --model runs/v1/digitalink.tflite --labels dataset/labels.txt --dataset dataset --num-samples 20
```
Shows 20 random validation examples with the recognized vs. actual text and their Character Error Rate (CER). This replicates the plugin's exact Greedy-CTC decoding logic, so what you see here is what the plugin will do at runtime.
## Download

Download the final model from the path printed by `convert_to_tflite.py` (e.g. `runs/v1/digitalink.tflite`).