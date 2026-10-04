# Brain Tumor MRI Binary Classification

This repository reorganizes `Code_brain_tumor.ipynb` into a reproducible Python project for binary MRI classification:

- `positive`: glioma, meningioma, or pituitary tumor
- `negative`: no-tumor images

It includes two models from the original notebook: a custom convolutional neural network and a frozen ResNet50V2 transfer-learning model.

## Important medical-use statement

This is a research and educational image-classification project. It is not a clinical diagnostic system and must not be used for patient diagnosis or treatment decisions. Performance must be independently validated on external clinical data before any clinical interpretation.

## Repository structure

```text
brain-tumor-repository/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── notebooks/Code_brain_tumor_original.ipynb
├── scripts/download_data.py
├── src/brain_tumor/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/raw/
├── data/processed/
├── models/
├── results/
└── reports/report.md
```

The image datasets are not committed to GitHub because of size, licensing, and reproducibility concerns. They are downloaded from Kaggle by the user.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download the data

Create a Kaggle API token at https://www.kaggle.com/settings/account. Place the downloaded `kaggle.json` in:

- Linux/macOS: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

Then run:

```bash
python scripts/download_data.py
```

The script downloads the two datasets used in the notebook into `data/raw/` and extracts them without overwriting existing files.

## Train and evaluate

Run the complete pipeline:

```bash
python run_pipeline.py --model resnet --epochs 30
```

Run the custom CNN:

```bash
python run_pipeline.py --model cnn --epochs 30
```

Outputs are saved to `models/` and `results/`. The pipeline creates a stratified 80/10/10 train/validation/test split, rescales images to 224 × 224, trains the selected model, saves the best checkpoint by validation loss, and writes a classification report and confusion matrix.

## Predict one image

```bash
python -m src.brain_tumor.predict \
  --model models/resnet_best.keras \
  --image path/to/image.jpg
```

The prediction is a research classification score. It is not a medical diagnosis.

## Reproduce the original notebook

The original notebook is retained in `notebooks/Code_brain_tumor_original.ipynb`. The refactored scripts are recommended because they remove Google Colab paths, avoid duplicate image copying, use deterministic splitting, and save evaluation artifacts consistently.

## Citation and data provenance

The original notebook references the Kaggle datasets `ahmedhamada0/brain-tumor-detection` and `masoudnickparvar/brain-tumor-mri-dataset`. Review each dataset's current license and citation requirements before redistribution or publication.
