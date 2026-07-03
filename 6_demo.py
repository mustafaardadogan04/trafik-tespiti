"""
Adim 6 — Streamlit demo (goruntu + video). Arayuz Turkce/Ingilizce.

Sade akis:
- Goruntu modu: ornek sec ya da yukle -> tespit + sayim
- Video modu: video sec/yukle -> ByteTrack takip + (istege bagli) cizgi-gecis sayimi
    * Cizgi OTOMATIK yerlestirilir (en yogun trafigin oldugu yer); istersen
      kapatip elle iki noktaya tiklayarak koyabilirsin
    * Cizgi kapali: model sadece tespit + takip yapar, benzersiz nesneleri sayar
    * Takip ayarli (track_buffer=90) -> nesne bir an kaybolsa da ayni ID'de kalir,
      tekrar sayim azalir
    * Sayim islem bitince gosterilir (rerun'larda kaybolmaz); panelde cizgiden
      gecen (yon ayrimiyla) + o ana kadarki toplam yazar
- Ayarlar: guven esigi + kirpma + cizgi videodan otomatik cikarilir (3 cikarim);
  imgsz Isle'ye basinca secilir. Istege bagli elle ayar. Isleme hizi secilebilir
  (kare atlama), ilerleme cubugunda tahmini kalan sure gosterilir.
- Dil: kenar cubugundaki 🌐 secimi tum arayuz metnini degistirir (METIN sozlugu).

Calistirma:            py -3.12 -m streamlit run 6_demo.py
Tiklayarak cizim icin: pip install streamlit-image-coordinates
"""

import os
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    TIKLA_VAR = True
except ImportError:
    TIKLA_VAR = False

KOK = Path(__file__).parent
MODEL = KOK / "model" / "genel_best.pt"
MODEL_URL = ("https://github.com/mustafaardadogan04/trafik-tespiti"
             "/releases/download/v1.0/genel_best.pt")
TAKIP = KOK / "bytetrack_takip.yaml"   # track_buffer buyutulmus config
ORNEK = KOK / "ornekler"
ORNEK_VID = KOK / "ornek_videolar"
SINIFLAR = ["bicycle", "bus", "car", "motorbike", "person"]
RENK = {0: (230, 80, 80), 1: (30, 120, 230), 2: (230, 150, 30),
        3: (30, 200, 120), 4: (180, 80, 230)}  # BGR

# --- Arayuz metinleri (tr/en) ------------------------------------------------
METIN = {
    "tr": {
        "baslik": "🚦 Trafik Nesne Tespiti + Sayma",
        "alt": "YOLOv8 · çok-açılı veriyle (MIO-TCD + Street View, ~18k görüntü) eğitildi. "
               "Sınıflar: bicycle, bus, car, motorbike, person.",
        "model_hata": "Model yüklenemedi ve otomatik indirme başarısız oldu — internet "
                      "bağlantısını kontrol et. Elle kurulum: [`genel_best.pt`]({url}) "
                      "dosyasını indirip `model/` klasörüne koy.",
        "ayarlar": "Ayarlar",
        "oto_ayar": "Ayarları otomatik seç",
        "oto_ayar_help": "Açıkken çözünürlük (imgsz) ve güven eşiği videoya göre otomatik "
                         "seçilir. Kapatırsan ikisini de elle ayarlarsın.",
        "oto_cap": "⚙️ imgsz ve güven eşiği otomatik.",
        "imgsz": "Çözünürlük (imgsz)",
        "guven": "Güven eşiği",
        "manuel_cap": "Küçük/uzak nesne varsa imgsz'yi artır. CPU'da büyük değer yavaştır.",
        "mod": "Mod",
        "mod_goruntu": "📷 Görüntü",
        "mod_video": "🎬 Video (takip + sayma)",
        "ornek_sec": "Örnek seç",
        "kendi_goruntu": "(Kendi görüntümü yükle)",
        "goruntu_yukle": "veya bir görüntü yükle",
        "tespit_spinner": "Tespit ediliyor...",
        "oto_lbl": "⚙️ Otomatik",
        "man_lbl": "✋ Manuel",
        "ayar_cap": "{lbl} → çözünürlük {imgsz} · güven {conf}",
        "tespit_sonuc": "Tespit sonucu",
        "sayim": "Sayım",
        "toplam_nesne": "Toplam nesne",
        "ornek_ya_yukle": "Bir örnek seç ya da görüntü yükle.",
        "video_cap": "Yukarıdan/açılı yol trafik videosu en iyi sonucu verir. "
                     "⚠️ CPU'da işlem birkaç dakika sürebilir.",
        "ornek_video_sec": "Örnek video seç",
        "kendi_video": "(Kendi videomu yükle)",
        "video_yukle": "veya video yükle (mp4 / avi / mov)",
        "cizgi_kullan": "Sayım çizgisi kullan",
        "cizgi_kullan_help": "Kapatırsan çizgi çizilmez; model sadece tespit + takip yapıp "
                             "benzersiz nesneleri sayar.",
        "iki_kaynak": "ℹ️ Hem örnek seçili hem video yüklü — **yüklenen video** kullanılıyor.",
        "video_analiz": "Video analiz ediliyor...",
        "oto_video_cap": "⚙️ Otomatik → güven {conf} · üst kırpma {kirp} "
                         "(çözünürlük İşle'ye basınca seçilir)",
        "man_video_cap": "✋ Manuel → çözünürlük {imgsz} · güven {conf} (kırpma yok)",
        "cizgi_oto": "Çizgiyi otomatik yerleştir (önerilen)",
        "cizgi_oto_help": "En yoğun trafiğin olduğu yere yatay çizgi koyar. Kapatırsan "
                          "çizgiyi elle koyabilirsin.",
        "cizgi_bul": "En iyi çizgi konumu bulunuyor...",
        "cizgi_onizle": "Otomatik çizgi — istersen üstteki kutuyu kapatıp elle koy",
        "tikla_cap": "👉 Kare üstünde **iki noktaya tıkla**: 1. başlangıç, 2. bitiş "
                     "(yatay/dikey/eğik). Yeniden koymak için 🗑 Sıfırla.",
        "sifirla": "🗑 Çizgiyi sıfırla",
        "cizgi_ok": "Çizgi ayarlandı ✔ — İşle'ye basabilirsin.",
        "nokta_bilgi": "{n}/2 nokta seçildi. Seçmezsen ortadan yatay çizgi kullanılır.",
        "tikla_uyari": "Elle çizim için: `pip install streamlit-image-coordinates`. "
                       "Şimdilik kaydırıcı kullanılıyor.",
        "cizgi_yon": "Çizgi yönü",
        "yatay": "Yatay",
        "dikey": "Dikey",
        "cizgi_konum": "Çizgi konumu",
        "onizleme": "Önizleme",
        "islem_hizi": "İşleme hızı",
        "hiz_hizli": "🐇 Hızlı (2 karede 1) — önerilen",
        "hiz_tam": "🐢 Tam (her kare)",
        "hiz_cok": "⚡ Çok hızlı (3 karede 1)",
        "hiz_cap": "Kare atlama süreyi o oranda kısaltır; takip tamponu geniş olduğu "
                   "için sayım doğruluğu pek etkilenmez.",
        "isle": "▶ İşle ve say",
        "imgsz_spinner": "En uygun çözünürlük seçiliyor (video başına bir kez)...",
        "secilen_imgsz": "⚙️ Seçilen çözünürlük: {imgsz}",
        "kirp_bilgi": "ℹ️ Üst kırpma, sayım çizgisi görünür kalsın diye {oran:.2f} oranına çekildi.",
        "yazici_hata": "Video yazıcı açılamadı (ffmpeg gerekli): `pip install imageio-ffmpeg`\n\n{e}",
        "progress_ilk": "İşleniyor... (sonuç işlem bitince gösterilecek)",
        "progress": "İşleniyor... {i}/{n} kare · kalan ~{k} sn",
        "progress_belirsiz": "İşleniyor... {i}. kare (toplam bilinmiyor)",
        "bitti": "Bitti",
        "ov_gecen": "GECEN",
        "ov_toplam": "TOPLAM",
        "ov_alt": "(gecen / toplam)",
        "sonuc_cizgi": "✅ Çizgiden geçen: {g}  ·  Görülen toplam (benzersiz): {t}",
        "gecen_md": "**Çizgiden geçen**",
        "gecen_lbl": "Geçen",
        "yon_cap": "↕ Yön ayrımı: bir yönden **{y1}** · karşı yönden **{y2}**",
        "toplam_md": "**Toplam görülen (benzersiz)**",
        "toplam_lbl": "Toplam",
        "sonuc_toplam": "✅ Toplam görülen (benzersiz): {t}",
        "indir": "📥 Sonuç videosunu indir",
    },
    "en": {
        "baslik": "🚦 Traffic Object Detection + Counting",
        "alt": "YOLOv8 · trained on a multi-angle dataset (MIO-TCD + Street View, ~18k images). "
               "Classes: bicycle, bus, car, motorbike, person.",
        "model_hata": "Model could not be loaded and auto-download failed — check your "
                      "internet connection. Manual setup: download [`genel_best.pt`]({url}) "
                      "and place it in the `model/` folder.",
        "ayarlar": "Settings",
        "oto_ayar": "Auto-select settings",
        "oto_ayar_help": "When on, resolution (imgsz) and confidence threshold are picked "
                         "automatically from the video. Turn off to set both manually.",
        "oto_cap": "⚙️ imgsz and confidence are automatic.",
        "imgsz": "Resolution (imgsz)",
        "guven": "Confidence threshold",
        "manuel_cap": "Increase imgsz for small/far objects. Large values are slow on CPU.",
        "mod": "Mode",
        "mod_goruntu": "📷 Image",
        "mod_video": "🎬 Video (tracking + counting)",
        "ornek_sec": "Pick a sample",
        "kendi_goruntu": "(Upload my own image)",
        "goruntu_yukle": "or upload an image",
        "tespit_spinner": "Detecting...",
        "oto_lbl": "⚙️ Auto",
        "man_lbl": "✋ Manual",
        "ayar_cap": "{lbl} → resolution {imgsz} · confidence {conf}",
        "tespit_sonuc": "Detection result",
        "sayim": "Counts",
        "toplam_nesne": "Total objects",
        "ornek_ya_yukle": "Pick a sample or upload an image.",
        "video_cap": "Top-down / angled road traffic videos work best. "
                     "⚠️ Processing may take a few minutes on CPU.",
        "ornek_video_sec": "Pick a sample video",
        "kendi_video": "(Upload my own video)",
        "video_yukle": "or upload a video (mp4 / avi / mov)",
        "cizgi_kullan": "Use counting line",
        "cizgi_kullan_help": "If off, no line is drawn; the model only detects + tracks "
                             "and counts unique objects.",
        "iki_kaynak": "ℹ️ Both a sample and an upload are set — the **uploaded video** is used.",
        "video_analiz": "Analyzing video...",
        "oto_video_cap": "⚙️ Auto → confidence {conf} · top crop {kirp} "
                         "(resolution is picked when you press Process)",
        "man_video_cap": "✋ Manual → resolution {imgsz} · confidence {conf} (no crop)",
        "cizgi_oto": "Place the line automatically (recommended)",
        "cizgi_oto_help": "Puts a horizontal line where traffic is densest. Turn off to "
                          "place the line yourself.",
        "cizgi_bul": "Finding the best line position...",
        "cizgi_onizle": "Automatic line — turn off the box above to place it manually",
        "tikla_cap": "👉 **Click two points** on the frame: 1st start, 2nd end "
                     "(horizontal/vertical/diagonal). Use 🗑 Reset to redo.",
        "sifirla": "🗑 Reset line",
        "cizgi_ok": "Line set ✔ — you can press Process.",
        "nokta_bilgi": "{n}/2 points selected. If you skip, a horizontal mid-line is used.",
        "tikla_uyari": "For click-to-draw: `pip install streamlit-image-coordinates`. "
                       "Using a slider for now.",
        "cizgi_yon": "Line direction",
        "yatay": "Horizontal",
        "dikey": "Vertical",
        "cizgi_konum": "Line position",
        "onizleme": "Preview",
        "islem_hizi": "Processing speed",
        "hiz_hizli": "🐇 Fast (every 2nd frame) — recommended",
        "hiz_tam": "🐢 Full (every frame)",
        "hiz_cok": "⚡ Very fast (every 3rd frame)",
        "hiz_cap": "Frame skipping shortens processing proportionally; the wide tracking "
                   "buffer keeps counting accuracy mostly unaffected.",
        "isle": "▶ Process and count",
        "imgsz_spinner": "Picking the best resolution (once per video)...",
        "secilen_imgsz": "⚙️ Selected resolution: {imgsz}",
        "kirp_bilgi": "ℹ️ Top crop pulled up to {oran:.2f} so the counting line stays visible.",
        "yazici_hata": "Video writer failed (ffmpeg required): `pip install imageio-ffmpeg`\n\n{e}",
        "progress_ilk": "Processing... (results will appear when done)",
        "progress": "Processing... {i}/{n} frames · ~{k} s left",
        "progress_belirsiz": "Processing... frame {i} (total unknown)",
        "bitti": "Done",
        "ov_gecen": "CROSSED",
        "ov_toplam": "TOTAL",
        "ov_alt": "(crossed / total)",
        "sonuc_cizgi": "✅ Crossed the line: {g}  ·  Total unique seen: {t}",
        "gecen_md": "**Crossed the line**",
        "gecen_lbl": "Crossed",
        "yon_cap": "↕ Direction split: **{y1}** one way · **{y2}** the other",
        "toplam_md": "**Total unique seen**",
        "toplam_lbl": "Total",
        "sonuc_toplam": "✅ Total unique seen: {t}",
        "indir": "📥 Download result video",
    },
}


@st.cache_resource
def model_yukle():
    if not MODEL.exists():
        # agirlik repoda yok (buyuk dosya) — ilk calistirmada Release'ten indirilir
        try:
            MODEL.parent.mkdir(exist_ok=True)
            with st.spinner("Model indiriliyor / downloading (~22 MB)..."):
                urllib.request.urlretrieve(MODEL_URL, MODEL)
        except Exception:
            if MODEL.exists():
                MODEL.unlink()   # yarim kalan dosya sonraki denemeyi bozmasin
            return None
    return YOLO(str(MODEL))


@st.cache_data(show_spinner=False)
def orta_kare(video_yol):
    """Videonun orta karesini okur (cizim onizlemesi icin). BGR ya da None."""
    cap = cv2.VideoCapture(video_yol)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, kare = cap.read()
    cap.release()
    return kare if ok else None


def taraf(px, py, ax, ay, bx, by):
    """P noktasinin A->B cizgisine gore hangi tarafta oldugu (isaretli alan)."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def kesisir(p, q, a, b):
    """P->Q hareket parcasi, A-B cizgi PARCASINI kesiyor mu (uzantisini degil).

    Iki parcali kesisim testi: uclar birbirinin zit taraflarinda olmali.
    Boylece kisa cizilen cizginin uzantisindan gecenler SAYILMAZ.
    """
    d1 = taraf(p[0], p[1], a[0], a[1], b[0], b[1])
    d2 = taraf(q[0], q[1], a[0], a[1], b[0], b[1])
    d3 = taraf(a[0], a[1], p[0], p[1], q[0], q[1])
    d4 = taraf(b[0], b[1], p[0], p[1], q[0], q[1])
    return d1 * d2 < 0 and d3 * d4 < 0


@st.cache_data(show_spinner=False)
def oto_analiz(video_yol):
    """3 ornek kareyi TEK gecişte tarar: guven esigi + ust kirpma + cizgi onerisi.

    Sabit 1280'de kosar (hizli); agir imgsz taramasi Isle'ye ertelenir. Boylece
    video secilir secilmez 9 yerine 3 cikarim yapilir, arayuz bekletmez.
    Donen: (conf, kirp, cizgi) — cizgi (ax, ay, bx, by) 0-1 orani, yatay.
    """
    cap = cv2.VideoCapture(video_yol)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(4))
    skorlar, ustler, merkezler = [], [], []
    for oran in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(N * oran))
        ok, fr = cap.read()
        if not ok:
            continue
        r = model.predict(fr, conf=0.10, imgsz=1280, verbose=False)[0]
        skorlar += r.boxes.conf.cpu().tolist()
        for b in r.boxes.xyxy.cpu().numpy():
            ustler.append(float(b[1]) / H)              # kutu ust kenari (kirpma icin)
            merkezler.append(float(b[1] + b[3]) / 2 / H)  # merkez y (cizgi icin)
    cap.release()
    # recall icin esigi dusuk tut; [0.12, 0.25]
    conf = float(min(0.25, max(0.12, np.percentile(skorlar, 15) - 0.05))) if skorlar else 0.15
    # en ust tespitin biraz ustune kirp (gokyuzu/binayi at)
    kirp = float(max(0.0, min(0.6, np.percentile(ustler, 5) - 0.05))) if ustler else 0.0
    # cizgi: tespit merkezlerinin ortancasi (trafigin yogun oldugu serit)
    y = float(np.median(merkezler)) if merkezler else 0.5
    return round(conf, 2), round(kirp, 2), (0.0, y, 1.0, y)


def en_iyi_imgsz(goruntu):
    """Bir kare uzerinde birkac cozunurluk deneyip en cok tespiti vereni secer.

    Aday cozunurlukler 1280'den baslar; trafikte 640/960 cok kucuk (uzak araci
    kacirir). Yuksek cozunurluk genelde daha cok yakaladigi icin biraz daha kolay
    yukselir (esik 1.10). CPU'da 2560 yavastir ama recall'i belirgin artirir.
    """
    en_iyi, secim = -1, 1280
    for sz in (1280, 1920, 2560):
        n = len(model.predict(goruntu, conf=0.15, imgsz=sz, verbose=False)[0].boxes)
        if n > en_iyi * 1.10:
            en_iyi, secim = n, sz
    return secim


@st.cache_data(show_spinner=False)
def oto_imgsz_video(video_yol):
    """Videonun orta karesinde en iyi cozunurlugu secer (Isle'ye basinca cagrilir)."""
    kare = orta_kare(video_yol)
    return en_iyi_imgsz(kare) if kare is not None else 1280


@st.cache_data(show_spinner=False)
def en_iyi_imgsz_gorsel(goruntu):
    """Goruntu modu icin cache'li sarmal: ayni goruntude her rerun'da (kenar
    cubuguna dokunmak dahil) 3 cikarimin tekrar kosmasini engeller."""
    return en_iyi_imgsz(goruntu)


st.set_page_config(page_title="Trafik Tespiti · Traffic Detection", page_icon="🚦", layout="wide")

# --- Dil secimi (tum arayuz metinleri T uzerinden gelir) --------------------
dil = st.sidebar.radio("🌐 Dil / Language", ["Türkçe", "English"], horizontal=True)
T = METIN["tr" if dil == "Türkçe" else "en"]

st.title(T["baslik"])
st.caption(T["alt"])

model = model_yukle()
if model is None:
    st.error(T["model_hata"].format(url=MODEL_URL))
    st.stop()

# --- Ayarlar (kenar cubugu) -----------------------------------------------
st.sidebar.header(T["ayarlar"])
oto_ayar_ac = st.sidebar.checkbox(T["oto_ayar"], value=True, help=T["oto_ayar_help"])
if oto_ayar_ac:
    st.sidebar.caption(T["oto_cap"])
    imgsz_manuel, conf_manuel = 1280, 0.25
else:
    imgsz_manuel = st.sidebar.select_slider(T["imgsz"], [640, 960, 1280, 1920, 2560], value=1280)
    conf_manuel = st.sidebar.slider(T["guven"], 0.10, 0.90, 0.25, 0.05)
    st.sidebar.caption(T["manuel_cap"])

mod = st.radio(T["mod"], [T["mod_goruntu"], T["mod_video"]], horizontal=True)


# --- GÖRÜNTÜ MODU ----------------------------------------------------------
if mod == T["mod_goruntu"]:
    ornekler = sorted(ORNEK.glob("*.jpg")) if ORNEK.exists() else []
    secim = st.selectbox(T["ornek_sec"], [T["kendi_goruntu"]] + [p.name for p in ornekler])
    dosya = st.file_uploader(T["goruntu_yukle"], type=["jpg", "jpeg", "png"])

    kaynak = None
    if dosya is not None:
        kaynak = Image.open(dosya).convert("RGB")
    elif secim != T["kendi_goruntu"]:
        kaynak = Image.open(ORNEK / secim).convert("RGB")

    if kaynak is not None:
        goruntu = np.array(kaynak)
        with st.spinner(T["tespit_spinner"]):
            if oto_ayar_ac:
                imgsz_g, conf_g = en_iyi_imgsz_gorsel(goruntu), 0.25   # otomatik + cache'li
            else:
                imgsz_g, conf_g = imgsz_manuel, conf_manuel     # elle
            r = model.predict(goruntu, conf=conf_g, imgsz=imgsz_g, verbose=False)[0]
        st.caption(T["ayar_cap"].format(lbl=T["oto_lbl"] if oto_ayar_ac else T["man_lbl"],
                                        imgsz=imgsz_g, conf=conf_g))
        annotated = r.plot()[:, :, ::-1]  # BGR -> RGB
        c1, c2 = st.columns([3, 1])
        with c1:
            st.image(annotated, caption=T["tespit_sonuc"], use_container_width=True)
        with c2:
            st.subheader(T["sayim"])
            sc = Counter(int(x) for x in r.boxes.cls.tolist())
            st.metric(T["toplam_nesne"], sum(sc.values()))
            for i, ad in enumerate(SINIFLAR):
                st.write(f"{ad}: **{sc.get(i, 0)}**")
    else:
        st.info(T["ornek_ya_yukle"])


# --- VIDEO MODU ------------------------------------------------------------
else:
    st.caption(T["video_cap"])

    ornek_vids = sorted(ORNEK_VID.glob("*.mp4")) if ORNEK_VID.exists() else []
    vsecim = st.selectbox(T["ornek_video_sec"], [T["kendi_video"]] + [p.name for p in ornek_vids])
    vdosya = st.file_uploader(T["video_yukle"], type=["mp4", "avi", "mov"])

    cizgi_acik = st.checkbox(T["cizgi_kullan"], value=True, help=T["cizgi_kullan_help"])

    # video yolu (upload'i tek sefer diske yaz) + kimlik degisince cizgiyi sifirla
    video_yol, kimlik = None, None
    if vdosya is not None:
        if vsecim != T["kendi_video"]:
            st.caption(T["iki_kaynak"])
        kimlik = f"upload:{vdosya.name}:{vdosya.size}"
        if st.session_state.get("vid_kimlik") != kimlik:
            eski = st.session_state.get("vid_yol")      # onceki upload'in temp'ini sil
            if eski and Path(eski).parent == Path(tempfile.gettempdir()):
                try:
                    os.remove(eski)
                except OSError:
                    pass
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vdosya.getvalue())
            tmp.close()
            st.session_state["vid_yol"] = tmp.name
        video_yol = st.session_state.get("vid_yol")
    elif vsecim != T["kendi_video"]:
        kimlik = f"ornek:{vsecim}"
        video_yol = str(ORNEK_VID / vsecim)

    if kimlik and st.session_state.get("vid_kimlik") != kimlik:
        st.session_state["vid_kimlik"] = kimlik
        st.session_state.pop("cizgi_noktalar", None)   # yeni video -> eski cizgiyi unut
        st.session_state.pop("son_tik", None)
        st.session_state.pop("sonuc", None)            # eski videonun sonucunu da temizle

    # --- guven esigi + kirpma + cizgi onerisi: tek hafif analiz (3 cikarim) ---
    # imgsz secimi agir oldugu icin Isle'ye ertelenir; secim aninda arayuz bekletmez
    imgsz_oto, conf_oto, kirp_oto = imgsz_manuel, conf_manuel, 0.0
    cizgi_oner = None
    if video_yol and oto_ayar_ac:
        with st.spinner(T["video_analiz"]):
            conf_oto, kirp_oto, cizgi_oner = oto_analiz(video_yol)
        st.caption(T["oto_video_cap"].format(conf=conf_oto, kirp=kirp_oto))
    elif video_yol:
        st.caption(T["man_video_cap"].format(imgsz=imgsz_oto, conf=conf_oto))

    # --- Cizgi: varsayilan OTOMATIK; istege bagli elle ---
    cizgi = None   # (ax, ay, bx, by) 0-1 orani; None ise ortadan yatay
    if cizgi_acik and video_yol:
        kare = orta_kare(video_yol)
        if kare is not None:
            H, W = kare.shape[:2]
            disp_W = 720
            disp_H = int(H * disp_W / W)

            def onizle(c, altyazi):
                onp = cv2.resize(kare, (disp_W, disp_H)).copy()
                cv2.line(onp, (int(c[0] * disp_W), int(c[1] * disp_H)),
                         (int(c[2] * disp_W), int(c[3] * disp_H)), (0, 0, 255), 3)
                st.image(onp[:, :, ::-1], caption=altyazi, use_container_width=True)

            oto = st.checkbox(T["cizgi_oto"], value=True, help=T["cizgi_oto_help"])

            if oto:
                if cizgi_oner is None:      # manuel ayar modunda da cizgi istenebilir
                    with st.spinner(T["cizgi_bul"]):
                        _, _, cizgi_oner = oto_analiz(video_yol)
                cizgi = cizgi_oner
                onizle(cizgi, T["cizgi_onizle"])
            elif TIKLA_VAR:
                st.caption(T["tikla_cap"])
                if st.button(T["sifirla"]):
                    st.session_state.pop("cizgi_noktalar", None)
                    st.session_state.pop("son_tik", None)

                noktalar = st.session_state.get("cizgi_noktalar", [])
                goster = cv2.resize(kare, (disp_W, disp_H)).copy()
                for (px, py) in noktalar:
                    cv2.circle(goster, (int(px), int(py)), 6, (0, 0, 255), -1)
                if len(noktalar) == 2:
                    cv2.line(goster, tuple(map(int, noktalar[0])),
                             tuple(map(int, noktalar[1])), (0, 0, 255), 3)

                deger = streamlit_image_coordinates(
                    Image.fromarray(goster[:, :, ::-1]), key="tik_kare")
                if deger is not None:
                    yeni = (deger["x"], deger["y"])
                    if st.session_state.get("son_tik") != yeni:
                        st.session_state["son_tik"] = yeni
                        nk = st.session_state.get("cizgi_noktalar", [])
                        nk = ([] if len(nk) >= 2 else nk) + [yeni]   # 3. tik -> bastan
                        st.session_state["cizgi_noktalar"] = nk
                        st.rerun()

                noktalar = st.session_state.get("cizgi_noktalar", [])
                if len(noktalar) == 2:
                    (ax, ay), (bx, by) = noktalar
                    cizgi = (ax / disp_W, ay / disp_H, bx / disp_W, by / disp_H)
                    st.success(T["cizgi_ok"])
                else:
                    st.info(T["nokta_bilgi"].format(n=len(noktalar)))
            else:
                st.warning(T["tikla_uyari"])
                yon = st.radio(T["cizgi_yon"], [T["yatay"], T["dikey"]], horizontal=True)
                pos = st.slider(T["cizgi_konum"], 0.05, 0.95, 0.5, 0.01)
                cizgi = (0.0, pos, 1.0, pos) if yon == T["yatay"] else (pos, 0.0, pos, 1.0)
                onizle(cizgi, T["onizleme"])

    if video_yol:
        hiz_map = {T["hiz_hizli"]: 2, T["hiz_tam"]: 1, T["hiz_cok"]: 3}
        hiz = st.radio(T["islem_hizi"], list(hiz_map), horizontal=True)
        stride = hiz_map[hiz]
        st.caption(T["hiz_cap"])
    else:
        stride = 1

    if video_yol and st.button(T["isle"]):
        if oto_ayar_ac:
            with st.spinner(T["imgsz_spinner"]):
                imgsz_oto = oto_imgsz_video(video_yol)
            st.caption(T["secilen_imgsz"].format(imgsz=imgsz_oto))
        cap = cv2.VideoCapture(video_yol)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W, H = int(cap.get(3)), int(cap.get(4))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # cizgi ucları (piksel). Cizilmediyse ortadan yatay.
        if cizgi is not None:
            ax, ay, bx, by = int(cizgi[0] * W), int(cizgi[1] * H), int(cizgi[2] * W), int(cizgi[3] * H)
        else:
            ax, ay, bx, by = 0, H // 2, W, H // 2

        y0 = int(H * kirp_oto)   # tespit girdisi icin ust kirpma (recall'i artirir)
        if cizgi_acik:
            # kirpma cizginin ustunu kapatirsa oradaki nesne tespit edilemez ve
            # gecis hic sayilmazdi -> kirpmayi cizginin biraz ustune cek
            sinir = max(0, min(ay, by) - int(0.05 * H))
            if y0 > sinir:
                y0 = sinir
                st.caption(T["kirp_bilgi"].format(oran=y0 / H))

        # cikti cozunurlugu: kaynagi koru (en fazla 1080p), cift boyut (H264 sart)
        oH = (min(H, 1080) // 2) * 2
        oW = (int(W * (oH / H)) // 2) * 2
        kucult = (oW, oH) != (W, H)      # kaynak <=1080 ise yeniden boyutlandirma yok
        cikti_yol = str(Path(tempfile.gettempdir()) / "trafik_demo_cikti.mp4")
        try:
            # CRF 20 + yuv420p: net goruntu, tarayicida oynar, makul dosya boyutu.
            # fps stride'a bolunur ki cikti video ayni surede oynasin
            writer = imageio.get_writer(
                cikti_yol, fps=fps / stride, codec="libx264", macro_block_size=None,
                pixelformat="yuv420p", output_params=["-crf", "20", "-preset", "medium"])
        except Exception as e:
            st.error(T["yazici_hata"].format(e=e))
            st.stop()

        vmodel = YOLO(str(MODEL))        # taze tracker durumu
        onceki_poz, sayildi = {}, set()  # tid -> onceki merkez; sayilan tid'ler
        omur = defaultdict(int)          # tid -> kac karede goruldu
        MIN_OMUR = 5                     # toplam sayaca girmek icin asgari omur (kare);
                                         # <=3-4 karelik izler genelde gurultu/ID-atlama
        gecen = defaultdict(int)         # cizgiden gecen (cizgi acikken)
        gorulen = defaultdict(set)       # benzersiz nesne (MIN_OMUR filtresiyle)

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        prog = st.progress(0.0, text=T["progress_ilk"])
        islenecek = (N // stride) if N > 0 else 0   # N=0: bozuk metadata olabilir
        yon1 = yon2 = 0                              # cizgi gecis yonleri
        t0 = time.time()
        islenen = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            islenen += 1
            alt = frame[y0:, :] if y0 else frame        # tespit kirpilmis karede
            r = vmodel.track(alt, persist=True, conf=conf_oto, imgsz=imgsz_oto,
                             tracker=str(TAKIP), verbose=False)[0]
            if r.boxes.id is not None:
                for box, cls, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                         r.boxes.cls.int().cpu().tolist(),
                                         r.boxes.id.int().cpu().tolist()):
                    x1, y1, x2, y2 = map(int, box)
                    y1, y2 = y1 + y0, y2 + y0           # kutuyu tam-kareye geri kaydir
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    renk = RENK.get(cls, (200, 200, 200))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                    cv2.putText(frame, f"{SINIFLAR[cls]} {tid}", (x1, y1 - 4),
                                FONT, 0.5, renk, 1)
                    omur[tid] += 1
                    if omur[tid] >= MIN_OMUR:             # kisa izler toplami sisirmesin
                        gorulen[cls].add(tid)
                    if cizgi_acik:                        # hareket, cizgi PARCASINI kesti mi
                        if tid in onceki_poz and tid not in sayildi and \
                                kesisir(onceki_poz[tid], (cx, cy), (ax, ay), (bx, by)):
                            sayildi.add(tid)
                            gecen[cls] += 1
                            # hangi taraftan geldi -> yon ayrimi
                            if taraf(onceki_poz[tid][0], onceki_poz[tid][1],
                                     ax, ay, bx, by) > 0:
                                yon1 += 1
                            else:
                                yon2 += 1
                        onceki_poz[tid] = (cx, cy)

            if cizgi_acik:
                cv2.line(frame, (ax, ay), (bx, by), (0, 0, 255), 3)

            # --- Panel: HEM cizgiden gecen HEM o ana kadarki toplam ---
            gecen_t = sum(gecen.values())
            toplam_t = sum(len(v) for v in gorulen.values())
            if cizgi_acik:
                cv2.rectangle(frame, (0, 0), (380, 190), (0, 0, 0), -1)
                cv2.putText(frame, f"{T['ov_gecen']}: {gecen_t} ({yon1}|{yon2})", (10, 34),
                            FONT, 0.9, (60, 60, 255), 2)
                cv2.putText(frame, f"{T['ov_toplam']}: {toplam_t}", (10, 68), FONT, 0.9, (255, 255, 255), 2)
                for i, ad in enumerate(SINIFLAR):
                    cv2.putText(frame, f"{ad}: {gecen[i]} / {len(gorulen[i])}",
                                (10, 98 + i * 17), FONT, 0.5, RENK[i], 1)
                cv2.putText(frame, T["ov_alt"], (10, 185), FONT, 0.4, (180, 180, 180), 1)
            else:
                cv2.rectangle(frame, (0, 0), (330, 150), (0, 0, 0), -1)
                cv2.putText(frame, f"{T['ov_toplam']}: {toplam_t}", (10, 38), FONT, 1.0, (255, 255, 255), 2)
                for i, ad in enumerate(SINIFLAR):
                    cv2.putText(frame, f"{ad}: {len(gorulen[i])}", (10, 72 + i * 16),
                                FONT, 0.5, RENK[i], 2)

            cikti_kare = cv2.resize(frame, (oW, oH), interpolation=cv2.INTER_AREA) if kucult else frame
            writer.append_data(cikti_kare[:, :, ::-1])   # BGR -> RGB

            if islenecek:
                gecti = time.time() - t0
                kalan = int(gecti / islenen * (islenecek - islenen))
                prog.progress(min(islenen / islenecek, 1.0),
                              text=T["progress"].format(i=islenen, n=islenecek, k=kalan))
            else:   # kare sayisi bilinmiyor (bozuk metadata) — yine de ilerledigi gorunsun
                prog.progress(0.0, text=T["progress_belirsiz"].format(i=islenen))

            # aradaki kareleri decode etmeden atla (grab, read'den cok daha ucuz)
            for _ in range(stride - 1):
                if not cap.grab():
                    break

        cap.release()
        writer.close()
        prog.progress(1.0, text=T["bitti"])

        # sonucu session_state'e yaz: indirme tiklamak gibi her etkilesim scripti
        # bastan kosturur (rerun); buton blogu icinde kalsaydi sonuc kaybolurdu
        st.session_state["sonuc"] = {
            "cizgi_acik": cizgi_acik,
            "gecen": {i: gecen[i] for i in range(len(SINIFLAR))},
            "gorulen": {i: len(gorulen[i]) for i in range(len(SINIFLAR))},
            "yon": (yon1, yon2),
            "cikti_yol": cikti_yol,
        }

    # --- SONUC: session_state'ten goster (rerun'larda da ekranda kalir) ---
    snc = st.session_state.get("sonuc")
    if snc and Path(snc["cikti_yol"]).exists():
        gecen_t = sum(snc["gecen"].values())
        toplam_t = sum(snc["gorulen"].values())
        if snc["cizgi_acik"]:
            st.success(T["sonuc_cizgi"].format(g=gecen_t, t=toplam_t))
            st.markdown(T["gecen_md"])
            cols = st.columns(6)
            cols[0].metric(T["gecen_lbl"], gecen_t)
            for i, ad in enumerate(SINIFLAR):
                cols[i + 1].metric(ad, snc["gecen"][i])
            y1, y2 = snc.get("yon", (0, 0))
            st.caption(T["yon_cap"].format(y1=y1, y2=y2))
            st.markdown(T["toplam_md"])
            cols = st.columns(6)
            cols[0].metric(T["toplam_lbl"], toplam_t)
            for i, ad in enumerate(SINIFLAR):
                cols[i + 1].metric(ad, snc["gorulen"][i])
        else:
            st.success(T["sonuc_toplam"].format(t=toplam_t))
            cols = st.columns(6)
            cols[0].metric(T["toplam_lbl"], toplam_t)
            for i, ad in enumerate(SINIFLAR):
                cols[i + 1].metric(ad, snc["gorulen"][i])

        st.video(snc["cikti_yol"])
        with open(snc["cikti_yol"], "rb") as f:
            st.download_button(T["indir"], f, "sonuc_sayim.mp4", "video/mp4")
