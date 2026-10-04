# Brain Tumor MRI Binary Classification Report

## Objective

The project classifies MRI images into tumor-present and no-tumor classes. Tumor-present images combine glioma, meningioma, and pituitary categories. The original notebook implements a custom CNN and a ResNet50V2 transfer-learning model.

## Data

The source data are the two Kaggle datasets listed in `data/README.md`. The project keeps the raw images outside version control. The preprocessing stage discovers class folders recursively, assigns tumor-present images label 1, assigns no-tumor images label 0, and creates a deterministic stratified train/validation/test split.

## Method

Images are resized to 224 × 224 pixels and rescaled to the [0, 1] interval. The custom CNN uses convolution, max-pooling, batch normalization, flattening, and dense layers. The transfer-learning model uses ImageNet-pretrained ResNet50V2 with the base layers frozen and a small binary classification head.

The best checkpoint is selected using validation loss. Evaluation includes accuracy, precision, recall, F1-score, a confusion matrix, and training-history plots.

## Reproducibility

```bash
python scripts/download_data.py
python run_pipeline.py --model resnet --epochs 30
```

The command writes `results/resnet_classification_report.txt`, `results/resnet_confusion_matrix.png`, and `results/resnet_history.png`.

## Results

Results are intentionally not pre-filled in this report. They must be generated after the user downloads the datasets and runs the pipeline. Report the test-set sample count, class distribution, accuracy, precision, recall, F1-score, confusion matrix, and decision threshold after execution.

## Limitations

- The binary label combines three biologically different tumor types.
- Image-level data may contain dataset-specific acquisition and preprocessing artifacts.
- Random image splits may overestimate generalisation when related images or subjects occur across splits.
- No external clinical validation is included.
- Accuracy alone is insufficient for medical assessment; sensitivity, specificity, calibration, and subgroup performance should also be reported.
- The model is a research prototype and is not a regulated medical device.
