import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model', required=True); args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]; result_dir=root/'results'; test_dir=root/'data'/'processed'/'test'
    gen=ImageDataGenerator(rescale=1/255).flow_from_directory(test_dir,target_size=(224,224),batch_size=64,class_mode='binary',shuffle=False)
    model=tf.keras.models.load_model(args.model); prob=model.predict(gen,verbose=0).ravel(); pred=(prob>=.5).astype(int)
    report=classification_report(gen.classes,pred,target_names=['negative','positive'])
    name=Path(args.model).stem; (result_dir/f'{name}_classification_report.txt').write_text(report)
    ConfusionMatrixDisplay.from_predictions(gen.classes,pred,display_labels=['negative','positive'],cmap='Blues'); plt.tight_layout(); plt.savefig(result_dir/f'{name}_confusion_matrix.png',dpi=180); plt.close()
    print(report)

if __name__=='__main__': main()
