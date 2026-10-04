# Data

Place downloaded image data under `data/raw/`. The repository does not redistribute the Kaggle images.

Expected source datasets:

1. `ahmedhamada0/brain-tumor-detection`
2. `masoudnickparvar/brain-tumor-mri-dataset`

The downloader accepts the Kaggle API token and extracts the source archives. The training code searches recursively for folders named `glioma`, `meningioma`, `pituitary`, and `notumor`.
