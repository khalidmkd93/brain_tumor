import argparse
import cv2
import numpy as np
import tensorflow as tf

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--image',required=True); args=ap.parse_args()
    image=cv2.imread(args.image)
    if image is None: raise FileNotFoundError(args.image)
    image=cv2.cvtColor(cv2.resize(image,(224,224)),cv2.COLOR_BGR2RGB).astype('float32')/255
    probability=float(tf.keras.models.load_model(args.model).predict(image[None,...],verbose=0)[0][0])
    print({'class':'positive' if probability>=.5 else 'negative','tumor_probability':probability})

if __name__=='__main__': main()
