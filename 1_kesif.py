"""
Adim 1 — Veri kesfi (EDA).

Trafik tespiti veri setini taniyoruz: bolunme sayilari, sinif dagilimi
(dengesizlik) ve ornek goruntuler uzerine ciziilmis kutucuklar.

Cikti: class_dagilimi.png + ornek_goruntuler.png
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# veri yolu --veri ile verilir; varsayilan proje yanindaki 'archive' klasoru
VERI = Path("archive")
CIKTI = Path(__file__).parent / "ciktilar"   # uretilen gorseller ciktilar/ altina
CIKTI.mkdir(exist_ok=True)
random.seed(42)

# Her sinifa bir renk
RENKLER = ["#e24b4a", "#ba7517", "#185fa5", "#1d9e75", "#534ab7"]


def yukle_siniflar():
    with open(VERI / "data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


def say_goruntu(bolum):
    klasor = VERI / bolum / "images"
    return len([p for p in klasor.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def sinif_dagilimi(bolum="train"):
    # her etiket dosyasinin ilk sutunu = sinif id; hepsini say
    sayac = Counter()
    for txt in (VERI / bolum / "labels").glob("*.txt"):
        for satir in txt.read_text().splitlines():
            if satir.strip():
                sayac[int(satir.split()[0])] += 1
    return sayac


def grafik_dagilim(siniflar, sayac):
    ids = sorted(sayac, key=lambda k: -sayac[k])
    isimler = [siniflar[i] for i in ids]
    degerler = [sayac[i] for i in ids]
    toplam = sum(degerler)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(isimler, degerler, color=[RENKLER[i] for i in ids])
    ax.set_title("Sinif dagilimi (train — kutu sayisi)")
    ax.set_ylabel("Kutu (instance) sayisi")
    for bar, d in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width() / 2, d, f"{d:,}\n%{d/toplam*100:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(CIKTI / "class_dagilimi.png", dpi=120)
    print(f"  -> {CIKTI / 'class_dagilimi.png'}")


def grafik_ornekler(siniflar, n=6):
    goruntuler = list((VERI / "train" / "images").glob("*.jpg"))
    secilen = random.sample(goruntuler, n)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, img_yol in zip(axes.flat, secilen):
        img = Image.open(img_yol)
        W, H = img.size
        ax.imshow(img)
        ax.axis("off")

        etiket = (VERI / "train" / "labels" / (img_yol.stem + ".txt"))
        if etiket.exists():
            for satir in etiket.read_text().splitlines():
                if not satir.strip():
                    continue
                sid, xc, yc, w, h = satir.split()
                sid = int(sid)
                xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
                dikdortgen = patches.Rectangle(
                    (xc - w / 2, yc - h / 2), w, h,
                    linewidth=1.5, edgecolor=RENKLER[sid], facecolor="none")
                ax.add_patch(dikdortgen)
    # ortak lejant
    eller = [patches.Patch(color=RENKLER[i], label=siniflar[i]) for i in range(len(siniflar))]
    fig.legend(handles=eller, loc="lower center", ncol=5)
    fig.suptitle("Ornek goruntuler + kutucuklar", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(CIKTI / "ornek_goruntuler.png", dpi=110)
    print(f"  -> {CIKTI / 'ornek_goruntuler.png'}")


def main():
    global VERI
    p = argparse.ArgumentParser(description="Trafik veri seti kesfi (EDA)")
    p.add_argument("--veri", default=str(VERI),
                   help="YOLO formatli veri seti koku (data.yaml iceren klasor)")
    VERI = Path(p.parse_args().veri)

    siniflar = yukle_siniflar()
    print(f"Siniflar ({len(siniflar)}): {siniflar}\n")

    print("Bolunme:")
    for b in ("train", "valid", "test"):
        print(f"  {b:6}: {say_goruntu(b):5} goruntu")

    sayac = sinif_dagilimi("train")
    toplam = sum(sayac.values())
    print(f"\nSinif dagilimi (train, toplam {toplam:,} kutu):")
    for i in sorted(sayac, key=lambda k: -sayac[k]):
        print(f"  {siniflar[i]:10}: {sayac[i]:6,}  (%{sayac[i]/toplam*100:.1f})")

    print("\nGrafikler kaydediliyor...")
    grafik_dagilim(siniflar, sayac)
    grafik_ornekler(siniflar)


if __name__ == "__main__":
    main()
