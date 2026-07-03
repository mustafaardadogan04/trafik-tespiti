"""
Adim 1b — Etiket kalitesi kontrolu.

Her sinif icin ornek kutulari goruntuden KIRPIP yan yana gosterir.
Amac: "bicycle denen sey gercekten bisiklet mi?" gozle dogrulamak.

Cikti: etiket_kontrol.png  (5 satir = 5 sinif, her satirda ornekler)
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from PIL import Image

# veri yolu --veri ile verilir; varsayilan proje yanindaki 'archive' klasoru
VERI = Path("archive")
CIKTI = Path(__file__).parent / "ciktilar"   # uretilen gorseller ciktilar/ altina
CIKTI.mkdir(exist_ok=True)
random.seed(0)

N = 8            # her sinif icin kac ornek
MIN_ALAN = 0.004  # cok kucuk kutular kirpinca taninmaz, ele


def yukle_siniflar():
    with open(VERI / "data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


def kutulari_topla():
    # her sinif id -> o sinifa ait (goruntu, kutu) ornekleri
    havuz = defaultdict(list)
    img_dir = VERI / "train" / "images"
    for txt in (VERI / "train" / "labels").glob("*.txt"):
        img = img_dir / (txt.stem + ".jpg")
        if not img.exists():
            continue
        for satir in txt.read_text().splitlines():
            if not satir.strip():
                continue
            sid, xc, yc, w, h = satir.split()
            if float(w) * float(h) < MIN_ALAN:
                continue
            havuz[int(sid)].append((img, float(xc), float(yc), float(w), float(h)))
    return havuz


def kirp(img_yol, xc, yc, w, h, pad=0.15):
    # normalize kutuyu piksele cevir, biraz pay birak, goruntuden kes
    im = Image.open(img_yol)
    W, H = im.size
    bw, bh = w * W, h * H
    sol = max(0, (xc - w / 2) * W - bw * pad)
    ust = max(0, (yc - h / 2) * H - bh * pad)
    sag = min(W, (xc + w / 2) * W + bw * pad)
    alt = min(H, (yc + h / 2) * H + bh * pad)
    return im.crop((sol, ust, sag, alt))


def main():
    global VERI
    p = argparse.ArgumentParser(description="Etiket kalitesi goz kontrolu")
    p.add_argument("--veri", default=str(VERI),
                   help="YOLO formatli veri seti koku (data.yaml iceren klasor)")
    VERI = Path(p.parse_args().veri)

    siniflar = yukle_siniflar()
    havuz = kutulari_topla()

    fig, axes = plt.subplots(len(siniflar), N, figsize=(N * 1.8, len(siniflar) * 1.9))
    for r, ad in enumerate(siniflar):
        ornekler = havuz.get(r, [])
        secilen = random.sample(ornekler, min(N, len(ornekler)))
        for c in range(N):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(secilen):
                ax.imshow(kirp(*secilen[c]))
        axes[r][0].set_ylabel(f"{ad}\n(n={len(ornekler)})",
                              rotation=0, ha="right", va="center", fontsize=11)

    fig.suptitle("Etiket kontrolu — her satir bir sinif (kirpilmis kutular)", fontsize=14)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    fig.savefig(CIKTI / "etiket_kontrol.png", dpi=110)
    print(f"  -> {CIKTI / 'etiket_kontrol.png'}")
    for r, ad in enumerate(siniflar):
        print(f"  {ad:10}: {len(havuz.get(r, [])):5} uygun kutu")


if __name__ == "__main__":
    main()
