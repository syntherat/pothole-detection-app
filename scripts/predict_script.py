# scripts/predict_script.py

from ultralytics import YOLO
import os
import sys

model = YOLO("model/best.pt")

if len(sys.argv) < 2:
    print("Usage: python predict_script.py <image_path>")
    sys.exit()

image_path = sys.argv[1]
results = model.predict(source=image_path, show=True, save=True)

print(f"Saved output to: {results[0].save_dir}")
