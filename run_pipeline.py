import subprocess, sys
import argparse

ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['cnn','resnet'],default='resnet'); ap.add_argument('--epochs',type=int,default=30); args=ap.parse_args()
subprocess.run([sys.executable,'-m','src.brain_tumor.train','--model',args.model,'--epochs',str(args.epochs)],check=True)
subprocess.run([sys.executable,'-m','src.brain_tumor.evaluate','--model',f'models/{args.model}_best.keras'],check=True)
