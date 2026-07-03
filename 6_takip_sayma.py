"""
Adim 6 — Takip + Sayma.

Cok-acili model (genel_best.pt) + ByteTrack ile videodaki araclari takip eder,
yatay bir sayim cizgisini gecen her araci (sinifiyla) BIR kez sayar ve
kutulu + ID'li + sayacli bir cikti videosu yazar.

Calistirma:  py -3.12 6_takip_sayma.py
"""

from collections import defaultdict

import cv2
from ultralytics import YOLO

MODEL = "model/genel_best.pt"
VIDEO = "videolar/kavsak.mp4"          # giris videolari videolar/ klasorunde
CIKTI = "ciktilar/kavsak_sayim.mp4"    # uretilen dosyalar ciktilar/ klasorunde

IMGSZ = 1920          # yuksek = daha cok tespit, daha yavas (CPU)
CONF = 0.25
KIRP_UST = 0.45       # ust %45'i at (gokyuzu/bina) — yol bolgesi kalsin
CIZGI = 0.55          # sayim cizgisinin kirpik icindeki dikey konumu
CIKTI_YUKSEKLIK = 1080  # cikti videosunu bu yukseklige olcekle

SINIFLAR = ["bicycle", "bus", "car", "motorbike", "person"]
RENK = {0: (230, 80, 80), 1: (30, 120, 230), 2: (230, 150, 30),
        3: (30, 200, 120), 4: (180, 80, 230)}  # BGR


def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    y0 = int(H * KIRP_UST)
    cW, cH = W, H - y0                 # kirpilmis boyut
    cizgi_y = int(cH * CIZGI)

    olcek = CIKTI_YUKSEKLIK / cH
    oW, oH = int(cW * olcek), CIKTI_YUKSEKLIK
    out = cv2.VideoWriter(CIKTI, cv2.VideoWriter_fourcc(*"mp4v"), fps, (oW, oH))

    onceki_y = {}                      # id -> onceki merkez y
    sayildi = set()                    # sayilan id'ler (cift sayma yok)
    sayac = defaultdict(int)           # sinif -> gecen sayisi

    kare = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kare += 1
        alt = frame[y0:, :]            # yol bolgesi

        r = model.track(alt, persist=True, conf=CONF, imgsz=IMGSZ,
                        tracker="bytetrack.yaml", verbose=False)[0]

        if r.boxes.id is not None:
            for box, cls, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.cls.int().cpu().tolist(),
                                     r.boxes.id.int().cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) // 2
                renk = RENK.get(cls, (200, 200, 200))
                cv2.rectangle(alt, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(alt, f"{SINIFLAR[cls]} {tid}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)

                # cizgi gecisi: onceki y ile simdiki y cizginin iki yaninda mi?
                if tid in onceki_y:
                    oy = onceki_y[tid]
                    if (oy < cizgi_y <= cy or oy > cizgi_y >= cy) and tid not in sayildi:
                        sayildi.add(tid)
                        sayac[cls] += 1
                onceki_y[tid] = cy

        # sayim cizgisi
        cv2.line(alt, (0, cizgi_y), (cW, cizgi_y), (0, 0, 255), 3)

        # sayac paneli
        toplam = sum(sayac.values())
        cv2.rectangle(alt, (0, 0), (330, 150), (0, 0, 0), -1)
        cv2.putText(alt, f"TOPLAM: {toplam}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for i, ad in enumerate(SINIFLAR):
            cv2.putText(alt, f"{ad}: {sayac[i]}", (10, 72 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RENK[i], 2)

        out.write(cv2.resize(alt, (oW, oH)))
        if kare % 30 == 0:
            print(f"  {kare} kare islendi | toplam gecen: {toplam}")

    cap.release()
    out.release()
    print(f"\nBitti -> {CIKTI}")
    print("Sinif bazli gecen:", {SINIFLAR[k]: v for k, v in sorted(sayac.items())})
    print("TOPLAM gecen arac:", sum(sayac.values()))


if __name__ == "__main__":
    main()
