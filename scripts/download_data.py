from pathlib import Path
import shutil
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "ahmedhamada0/brain-tumor-detection",
    "masoudnickparvar/brain-tumor-mri-dataset",
]

def main():
    api = KaggleApi()
    api.authenticate()
    for dataset in DATASETS:
        slug = dataset.split("/")[-1]
        archive = RAW / f"{slug}.zip"
        print(f"Downloading {dataset}")
        api.dataset_download_files(dataset, path=str(RAW), unzip=False, quiet=False)
        downloaded = RAW / f"{slug}.zip"
        if downloaded.exists():
            with zipfile.ZipFile(downloaded) as zf:
                zf.extractall(RAW / slug)
            print(f"Extracted to {RAW / slug}")

if __name__ == "__main__":
    main()
