import argparse
from pathlib import Path
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .data import build_split
from .models import custom_cnn, resnet_transfer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet"], default="resnet")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    processed = root / "data" / "processed"
    counts = build_split(root / "data" / "raw", processed, args.seed)
    train_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(processed / "training", target_size=(224,224), batch_size=64, class_mode="binary", seed=args.seed)
    val_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(processed / "validation", target_size=(224,224), batch_size=64, class_mode="binary", seed=args.seed)
    model = custom_cnn() if args.model == "cnn" else resnet_transfer()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model_dir = root / "models"; result_dir = root / "results"
    model_dir.mkdir(exist_ok=True); result_dir.mkdir(exist_ok=True)
    ckpt = model_dir / f"{args.model}_best.keras"
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=[tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)])
    with open(result_dir / f"{args.model}_history.json", "w") as f: json.dump({k: [float(x) for x in v] for k,v in history.history.items()}, f)
    with open(result_dir / f"{args.model}_split_counts.json", "w") as f: json.dump(counts, f, indent=2)
    print(f"Saved best model to {ckpt}")

if __name__ == "__main__": main()
