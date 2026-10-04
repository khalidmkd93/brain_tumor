from pathlib import Path
import random
import shutil

from sklearn.model_selection import train_test_split

CLASS_FOLDERS = {"glioma": 1, "meningioma": 1, "pituitary": 1, "notumor": 0}

def find_images(raw_dir: Path):
    records = []
    for folder, label in CLASS_FOLDERS.items():
        for path in raw_dir.rglob(folder):
            if path.is_dir():
                for image in path.iterdir():
                    if image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        records.append((image, label))
    if not records:
        raise FileNotFoundError("No class folders found. Run scripts/download_data.py first.")
    return records

def build_split(raw_dir: Path, processed_dir: Path, seed: int = 25):
    records = find_images(raw_dir)
    random.seed(seed)
    random.shuffle(records)
    paths = [str(p) for p, _ in records]
    labels = [y for _, y in records]
    train_p, temp_p, train_y, temp_y = train_test_split(paths, labels, test_size=.2, random_state=seed, stratify=labels)
    val_p, test_p, val_y, test_y = train_test_split(temp_p, temp_y, test_size=.5, random_state=seed, stratify=temp_y)
    splits = {"training": (train_p, train_y), "validation": (val_p, val_y), "test": (test_p, test_y)}
    for split, (items, ys) in splits.items():
        for label in (0, 1):
            (processed_dir / split / str(label)).mkdir(parents=True, exist_ok=True)
        for i, (item, y) in enumerate(zip(items, ys)):
            src = Path(item)
            shutil.copy2(src, processed_dir / split / str(y) / f"{i:06d}_{src.name}")
    return {split: len(items) for split, (items, _) in splits.items()}
