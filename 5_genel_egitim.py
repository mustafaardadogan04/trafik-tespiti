"""
Adim 5 — Genel (cok acili) model egitimi (Kaggle'da calisir).

Demo'nun kullandigi ASIL modeli (genel_best.pt) egitir: 'birlesik' veri seti
(sokak-CCTV archive + MIO-TCD alt kumesi, ~18k goruntu) uzerinde YOLOv8s.
2_egitim.py'deki baseline'dan farki: daha buyuk model (s vs n), daha uzun
egitim (40 vs 20 epoch) ve cok acili birlesik veri.

Sonuc (40. epoch, dogrulama):
    precision 0.928 · recall 0.881 · mAP50 0.929 · mAP50-95 0.702
Egitim egrileri ve confusion matrix: ciktilar/ klasorunde.

Calistirma:
  - Kaggle'da (onerilen, T4 GPU): birlesik veri setini notebook'a ekle,
    hucre olarak calistir. Yollar /kaggle/input otomatik bulunur.
  - Yerelde: python 5_genel_egitim.py <data.yaml yolu>

Egitim bitince runs/detect/genel/weights/best.pt dosyasini
model/genel_best.pt olarak kopyala — demo (7_demo.py) onu arar.
"""

import glob
import os
import sys

import yaml
from ultralytics import YOLO


def veri_yaml_hazirla():
    # dataset Kaggle'a eklenince /kaggle/input altina baglanir; kokunu bul
    cand = glob.glob("/kaggle/input/**/train/images", recursive=True)[0]
    root = os.path.dirname(os.path.dirname(cand))
    print("Dataset kok:", root)

    # ultralytics'in bulabilmesi icin mutlak yollu yeni bir data.yaml yaz
    cfg = {
        "path": root,
        "train": "train/images",
        "val": "valid/images",
        "nc": 5,
        "names": ["bicycle", "bus", "car", "motorbike", "person"],
    }
    hedef = "/kaggle/working/data.yaml"
    with open(hedef, "w") as f:
        yaml.dump(cfg, f)
    return hedef


def main():
    # yerelde calistiriliyorsa data.yaml yolunu argumandan al
    veri = sys.argv[1] if len(sys.argv) > 1 else veri_yaml_hazirla()

    model = YOLO("yolov8s.pt")   # baseline'daki 'n' yerine 's': daha isabetli
    model.train(
        data=veri,
        epochs=40,        # baseline'in iki kati — egri 40'ta oturuyor
        imgsz=640,        # veri 640x640 hazirlandi
        batch=16,         # T4 16GB icin rahat
        name="genel",     # sonuclar runs/detect/genel/ altina
    )


if __name__ == "__main__":
    main()
