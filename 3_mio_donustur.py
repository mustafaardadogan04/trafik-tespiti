"""
Adim 3 — MIO-TCD Localization -> YOLO donusturucu.

Ne yapar:
  - gt_train.csv'yi okur (piksel kutulari)
  - Cesitli bir ALT KUME secer (cok kameradan, rastgele)
  - 11 sinifi bizim 5 sinifa esler (digerlerini eler)
  - Piksel kutulari YOLO formatina (normalize cx,cy,w,h) cevirir
  - train/valid olarak yeni bir YOLO dataset'i yazar (+ data.yaml)

Calistirma:  python 3_mio_donustur.py --mio <MIO-TCD klasoru> --cikti mio_yolo
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

# --- Ayarlar (yollar --mio / --cikti ile degistirilebilir) -------------------
MIO = Path("MIO-TCD-Localization")   # gt_train.csv iceren klasor
CIKTI = Path("mio_yolo")             # uretilecek YOLO dataset klasoru
ALT_KUME = 12000      # kac goruntu alinsin (cesitli alt kume)
VAL_ORAN = 0.10       # dogrulama payi
random.seed(42)

# 11 sinif -> bizim 5 sinif (indeks). Listedeki digerleri ELENIR.
ESLEME = {
    "bicycle": 0,
    "bus": 1,
    "car": 2, "pickup_truck": 2,   # pickup'i car'a katiyoruz
    "motorcycle": 3,               # bizde 'motorbike'
    "pedestrian": 4,               # bizde 'person'
}
SINIFLAR = ["bicycle", "bus", "car", "motorbike", "person"]


def kutulari_oku():
    # image_id -> [(sinif_idx, x1,y1,x2,y2), ...]  (sadece eslenen siniflar)
    kutular = defaultdict(list)
    with open(MIO / "gt_train.csv", newline="") as f:
        for row in csv.reader(f):
            img_id, cls = row[0], row[1]
            if cls in ESLEME:
                x1, y1, x2, y2 = map(int, row[2:6])
                kutular[img_id].append((ESLEME[cls], x1, y1, x2, y2))
    return kutular


def yaz(bolum, idler, kutular):
    (CIKTI / bolum / "images").mkdir(parents=True, exist_ok=True)
    (CIKTI / bolum / "labels").mkdir(parents=True, exist_ok=True)
    for i, img_id in enumerate(idler, 1):
        src = MIO / "train" / f"{img_id}.jpg"
        if not src.exists():
            continue
        W, H = Image.open(src).size
        satirlar = []
        for cls, x1, y1, x2, y2 in kutular[img_id]:
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            satirlar.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        shutil.copy(src, CIKTI / bolum / "images" / f"{img_id}.jpg")
        (CIKTI / bolum / "labels" / f"{img_id}.txt").write_text("\n".join(satirlar))
        if i % 2000 == 0:
            print(f"  {bolum}: {i}/{len(idler)}")


def main():
    global MIO, CIKTI
    p = argparse.ArgumentParser(description="MIO-TCD -> YOLO donusturucu")
    p.add_argument("--mio", default=str(MIO), help="MIO-TCD koku (gt_train.csv iceren)")
    p.add_argument("--cikti", default=str(CIKTI), help="uretilecek YOLO dataset klasoru")
    a = p.parse_args()
    MIO, CIKTI = Path(a.mio), Path(a.cikti)

    if CIKTI.exists():           # eski cikti kalmasin, temiz uret
        shutil.rmtree(CIKTI)

    print("gt_train.csv okunuyor...")
    kutular = kutulari_oku()

    gecerli = [i for i in kutular if kutular[i]]

    # NADIR sinifli goruntuleri oncele (car disi) -> hem denge hem aci cesitliligi
    NADIR = {0, 1, 3, 4}  # bicycle, bus, motorbike, person
    nadir_img = [i for i in gecerli if any(c in NADIR for c, *_ in kutular[i])]
    nadir_set = set(nadir_img)
    car_img = [i for i in gecerli if i not in nadir_set]
    random.shuffle(nadir_img)
    random.shuffle(car_img)

    secilen = nadir_img[:ALT_KUME]
    if len(secilen) < ALT_KUME:                       # kalani car ile doldur
        secilen += car_img[: ALT_KUME - len(secilen)]
    random.shuffle(secilen)

    n_val = int(len(secilen) * VAL_ORAN)
    print(f"Nadir-sinifli goruntu: {len(nadir_img)} | car-only: {len(car_img)}")
    print(f"Secilen: {len(secilen)} (train {len(secilen)-n_val} / valid {n_val})")

    yaz("valid", secilen[:n_val], kutular)
    yaz("train", secilen[n_val:], kutular)

    (CIKTI / "data.yaml").write_text(
        f"path: {CIKTI.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 5\n"
        f"names: {SINIFLAR}\n"
    )
    print(f"\nBitti -> {CIKTI}")
    print("Sinif dagilimini gormek icin YOLO egitiminde otomatik cikar.")


if __name__ == "__main__":
    main()
