"""
Adim 2 — Baseline egitim (Kaggle'da calisir).

YOLOv8n (en kucuk model) ile kisa bir egitim: "hat calisiyor mu?" denemesi.
Kaggle notebook'unda hucre hucre calistirilir. GPU: Settings -> T4.

Not: Bu script Kaggle'in /kaggle/input ve /kaggle/working yollarini kullanir.
"""

import glob
import os

import yaml
from ultralytics import YOLO


def veri_yaml_hazirla():
    # dataset Kaggle'a eklenince /kaggle/input altina baglanir; data.yaml'i bul
    src = glob.glob("/kaggle/input/**/data.yaml", recursive=True)[0]
    root = os.path.dirname(src)
    print("Dataset kok:", root)

    # ultralytics'in bulabilmesi icin mutlak yollu yeni bir data.yaml yaz
    cfg = {
        "path": root,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 5,
        "names": ["bicycle", "bus", "car", "motorbike", "person"],
    }
    hedef = "/kaggle/working/data.yaml"
    with open(hedef, "w") as f:
        yaml.dump(cfg, f)
    return hedef


def main():
    veri = veri_yaml_hazirla()

    model = YOLO("yolov8n.pt")   # COCO'da onceden egitilmis nano model
    model.train(
        data=veri,
        epochs=20,        # baseline icin kisa
        imgsz=640,        # veri zaten 640x640
        batch=16,         # T4 16GB icin rahat
        name="baseline",  # sonuclar runs/detect/baseline/ altina
    )


if __name__ == "__main__":
    main()
