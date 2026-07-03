"""
Adim 4 — archive (sokak-CCTV) + mio_yolo (cok acili) -> tek 'birlesik' dataset.

Ikisi de ayni sinif sirasina sahip (bicycle,bus,car,motorbike,person),
o yuzden sadece dosyalari birlestiriyoruz. Isim cakismasin diye onek koyuyoruz.

Calistirma:  python 4_birlestir.py --archive <archive yolu> --mio mio_yolo
"""

import argparse
import shutil
from pathlib import Path

# yollar --archive / --mio / --cikti ile degistirilebilir
ARCHIVE = Path("archive")
MIO = Path("mio_yolo")
CIKTI = Path("birlesik")

SINIFLAR = ["bicycle", "bus", "car", "motorbike", "person"]


def main():
    global CIKTI
    p = argparse.ArgumentParser(description="Iki YOLO veri setini birlestirir")
    p.add_argument("--archive", default=str(ARCHIVE), help="sokak-CCTV veri seti koku")
    p.add_argument("--mio", default=str(MIO), help="mio_yolo veri seti koku")
    p.add_argument("--cikti", default="birlesik", help="uretilecek birlesik klasoru")
    a = p.parse_args()
    CIKTI = Path(a.cikti)
    kaynaklar = {"arch": Path(a.archive), "mio": Path(a.mio)}   # onek -> kok

    if CIKTI.exists():
        shutil.rmtree(CIKTI)

    for bolum in ("train", "valid"):
        (CIKTI / bolum / "images").mkdir(parents=True, exist_ok=True)
        (CIKTI / bolum / "labels").mkdir(parents=True, exist_ok=True)

        for onek, kok in kaynaklar.items():
            img_dir = kok / bolum / "images"
            lbl_dir = kok / bolum / "labels"
            if not img_dir.exists():
                print(f"  ! {kok.name}/{bolum} yok, atlaniyor")
                continue

            sayac = 0
            for img in img_dir.glob("*.jpg"):
                yeni_ad = f"{onek}_{img.name}"
                shutil.copy(img, CIKTI / bolum / "images" / yeni_ad)
                lbl = lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, CIKTI / bolum / "labels" / f"{onek}_{img.stem}.txt")
                sayac += 1
            print(f"  {bolum}/{onek}: {sayac} kopyalandi")

    (CIKTI / "data.yaml").write_text(
        f"path: {CIKTI.as_posix()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"nc: 5\n"
        f"names: {SINIFLAR}\n"
    )

    print("\n=== ozet ===")
    for bolum in ("train", "valid"):
        n = len(list((CIKTI / bolum / "images").glob("*.jpg")))
        print(f"  {bolum}: {n} goruntu")
    print(f"Bitti -> {CIKTI}")


if __name__ == "__main__":
    main()
