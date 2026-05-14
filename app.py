import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fish Freshness Detector",
    page_icon="🐟",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 820px; }

.app-header {
    text-align: center; padding: 2rem 1.5rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    margin-bottom: 1.4rem; position: relative; overflow: hidden;
}
.app-header::after {
    content:''; position:absolute; bottom:-30px; right:-30px;
    width:130px; height:130px;
    background:rgba(255,255,255,.05); border-radius:50%;
}
.app-header h1  { color:#fff; font-size:1.8rem; font-weight:800; margin:0 0 .35rem; letter-spacing:-.5px; }
.app-header p   { color:rgba(255,255,255,.65); font-size:.88rem; margin:0; }
.app-chip {
    display:inline-block; background:rgba(255,255,255,.12); color:#6ee7b7;
    font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:3px 10px; border-radius:20px; margin-bottom:.7rem;
}
[data-testid="stFileUploader"] > div:first-child {
    border:2px dashed rgba(16,185,129,.45) !important;
    border-radius:14px !important; background:rgba(16,185,129,.04) !important;
    padding:1rem !important;
}
.box { border-radius:14px; padding:1.1rem 1.3rem; margin:.5rem 0 .8rem; }
.box-fresh    { background:rgba(16,185,129,.1);  border:1.5px solid rgba(16,185,129,.4); }
.box-notfresh { background:rgba(239,68,68,.1);   border:1.5px solid rgba(239,68,68,.35); }
.box-warn     { background:rgba(234,179,8,.1);   border:1.5px solid rgba(234,179,8,.35); }
.box-title    { font-size:1.25rem; font-weight:800; letter-spacing:-.3px;
                display:flex; align-items:center; gap:8px; margin-bottom:.25rem; }
.txt-fresh    { color:#059669; }
.txt-notfresh { color:#dc2626; }
.txt-warn     { color:#b45309; }
.box-sub      { font-size:.83rem; color:#6b7280; }
.bar-wrap { margin:.55rem 0; }
.bar-head { display:flex; justify-content:space-between; font-size:.78rem;
            font-weight:600; margin-bottom:4px; color:#374151; }
.bar-bg   { background:rgba(0,0,0,.07); border-radius:99px; height:9px; overflow:hidden; }
.bar-fill-g { height:100%; border-radius:99px; background:linear-gradient(90deg,#059669,#34d399); }
.bar-fill-r { height:100%; border-radius:99px; background:linear-gradient(90deg,#dc2626,#f87171); }
.info-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:.6rem; }
.info-cell { background:rgba(0,0,0,.04); border-radius:10px; padding:.6rem .9rem;
             font-size:.78rem; color:#6b7280; line-height:1.65; }
.info-cell b { color:#111827; font-weight:600; }
[data-testid="stImage"] img { border-radius:12px !important; }
hr { border-color:rgba(0,0,0,.08) !important; margin:.9rem 0 !important; }
@media (prefers-color-scheme: dark) {
    .bar-head  { color:#d1d5db; }
    .bar-bg    { background:rgba(255,255,255,.1); }
    .info-cell { background:rgba(255,255,255,.06); color:#9ca3af; }
    .info-cell b { color:#f9fafb; }
    .box-sub   { color:#9ca3af; }
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)
THRESHOLD = 0.5

# Kata kunci IKAN — kalau ada ini di top-20 ImageNet → langsung lolos
FISH_ALLOW = {
    'fish', 'tench', 'goldfish', 'salmon', 'eel', 'shark', 'ray', 'stingray',
    'puffer', 'blowfish', 'lionfish', 'coho', 'carp', 'mackerel', 'tuna',
    'cod', 'trout', 'snapper', 'grouper', 'flounder', 'sole', 'halibut',
    'pike', 'barracuda', 'marlin', 'swordfish', 'anchovy', 'herring',
    'sardine', 'catfish', 'tilapia', 'perch', 'sturgeon', 'mullet',
    'shrimp', 'lobster', 'crab', 'squid', 'octopus',
}

# Kata kunci BUKAN IKAN — semua single-word (tanpa underscore) agar
# cocok dengan lbl.replace('_', ' ')
NON_FISH_BLOCK = {
    # Manusia & atribut
    'person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'head',
    'human', 'baby', 'child', 'portrait',
    # Pakaian (penting untuk foto orang berpakaian)
    'shirt', 'dress', 'suit', 'jacket', 'coat', 'trousers', 'jeans',
    'skirt', 'blouse', 'uniform', 'jersey', 'robe', 'kimono', 'abaya',
    'hijab', 'veil', 'scarf', 'tie', 'sock', 'shoe', 'boot', 'sandal',
    'hat', 'cap', 'helmet', 'glasses', 'sunglasses', 'bag', 'backpack',
    'handbag', 'purse', 'wallet', 'umbrella', 'watch', 'sari', 'sarong',
    'poncho', 'cloak', 'gown', 'pajama', 'swimsuit', 'mitten', 'brassiere',
    # Hewan darat / udara
    'cat', 'dog', 'horse', 'cow', 'elephant', 'lion', 'tiger', 'bear',
    'rabbit', 'bird', 'chicken', 'duck', 'goose', 'penguin', 'parrot',
    'snake', 'lizard', 'frog', 'insect', 'butterfly', 'bee', 'monkey',
    'gorilla', 'deer', 'fox', 'wolf', 'squirrel', 'hamster', 'pig',
    'sheep', 'goat', 'camel', 'zebra', 'giraffe', 'hippo',
    # Kendaraan
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane',
    'helicopter', 'ambulance', 'tractor', 'tank',
    # Bangunan / tempat
    'building', 'house', 'church', 'mosque', 'tower', 'bridge', 'castle',
    'street', 'road', 'mountain', 'volcano', 'forest', 'tree', 'flower',
    'grass', 'sky', 'cloud',
    # Elektronik / perangkat
    'phone', 'computer', 'keyboard', 'television', 'camera', 'microphone',
    'monitor', 'laptop', 'screen', 'display', 'tablet',
    # Furnitur / benda
    'book', 'bottle', 'cup', 'chair', 'table', 'lamp', 'clock', 'sofa',
    'bed', 'door', 'window',
    # Ilustrasi / media / poster
    'comic', 'cartoon', 'illustration', 'drawing', 'anime', 'poster',
    'banner', 'envelope', 'jigsaw', 'puzzle',
    # Musik / panggung
    'piano', 'guitar', 'violin', 'drum', 'stage', 'theater', 'microphone',
    # Diagram / teknis
    'web', 'diagram', 'chart', 'graph', 'plot', 'abacus',
}


# ── Cache model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    import tensorflow as tf
    for path in [
        'model_final.keras', 'output/model_final.keras',
        'model_final.h5',    'output/model_final.h5',
    ]:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    return None, None


@st.cache_resource(show_spinner=False)
def load_validator():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    m = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    return m, preprocess_input, decode_predictions


# ── Layer 1: Deteksi wajah manusia (OpenCV) ───────────────────────────────────
def detect_face(pil_img: Image.Image) -> bool:
    """
    Return True kalau terdeteksi wajah manusia.
    Menggunakan dua cascade (frontal + profile) agar lebih akurat.
    """
    try:
        import cv2
        img_np = np.array(pil_img.convert('RGB'))
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Equalise histogram agar kontras lebih jelas
        gray = cv2.equalizeHist(gray)

        front = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        h, w = gray.shape
        min_face = max(20, int(min(h, w) * 0.08))   # min 8% ukuran gambar

        faces = front.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_face, min_face)
        )
        if len(faces) > 0:
            return True

        faces_p = profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_face, min_face)
        )
        return len(faces_p) > 0

    except Exception:
        return False   # cv2 tidak tersedia → skip layer ini


# ── Layer 2: Validasi ImageNet ────────────────────────────────────────────────
def imagenet_check(pil_img: Image.Image):
    """
    Return (lolos: bool, label: str)

    BUG-FIX dari versi lama:
      1. Tidak ada 'break' dini — semua top-20 selalu dicek blocklist
      2. Semua keyword dibandingkan sebagai string tanpa underscore
      3. MIN_BLOCK_PROB diturunkan ke 0.01 (lebih sensitif)
    """
    MIN_FISH_PROB  = 0.02   # ikan harus ≥ 2% baru dianggap lolos
    MIN_BLOCK_PROB = 0.01   # non-ikan ≥ 1% sudah cukup untuk blokir

    val_model, preprocess_input, decode_predictions = load_validator()

    img  = pil_img.convert('RGB').resize((224, 224))
    arr  = np.array(img, dtype=np.float32)
    arr  = preprocess_input(arr)
    arr  = np.expand_dims(arr, 0)

    preds = val_model.predict(arr, verbose=0)
    top20 = decode_predictions(preds, top=20)[0]

    top_label = top20[0][1].lower().replace('_', ' ')

    # ── Cek allowlist ikan (hentikan lebih awal kalau prob sudah kecil) ───────
    for _, lbl, prob in top20:
        if prob < MIN_FISH_PROB:
            break
        lbl_clean = lbl.lower().replace('_', ' ')
        if any(kw in lbl_clean for kw in FISH_ALLOW):
            return True, lbl_clean

    # ── Cek blocklist non-ikan — TIDAK ada break, cek semua top-20 ───────────
    # PERBAIKAN: keyword di-replace underscore juga agar cocok dengan lbl_clean
    for _, lbl, prob in top20:
        if prob < MIN_BLOCK_PROB:
            continue   # skip yang terlalu kecil, tapi JANGAN break
        lbl_clean = lbl.lower().replace('_', ' ')
        for kw in NON_FISH_BLOCK:
            kw_clean = kw.replace('_', ' ')   # FIX: samakan format
            if kw_clean in lbl_clean:
                return False, lbl_clean

    # Ambigu (makanan, objek tak dikenal, dll) → lolos, biar classifier yg memutuskan
    return True, top_label


# ── Fungsi validasi utama (gabungan semua layer) ──────────────────────────────
def validate(pil_img: Image.Image):
    """
    Return (lolos: bool, pesan: str)

    Pipeline:
      Layer 1 → deteksi wajah (OpenCV)  → paling cepat, paling akurat untuk manusia
      Layer 2 → ImageNet dual-check     → tangkap non-ikan lainnya
    """
    # Layer 1
    if detect_face(pil_img):
        return False, "Terdeteksi wajah manusia pada gambar"

    # Layer 2
    lolos, label = imagenet_check(pil_img)
    if not lolos:
        return False, f'Terdeteksi sebagai: <b style="color:#92400e">{label}</b>'

    return True, ""


# ── Preprocessing & prediksi ──────────────────────────────────────────────────
def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def predict(model, arr: np.ndarray):
    prob_not = float(model.predict(arr, verbose=0)[0][0])
    prob_ok  = 1.0 - prob_not
    if prob_not >= THRESHOLD:
        return 'not_fresh', prob_not, prob_ok, prob_not
    return 'fresh', prob_ok, prob_ok, prob_not


def bar(label, dot_color, pct, css):
    return f"""
    <div class="bar-wrap">
      <div class="bar-head">
        <span style="display:flex;align-items:center;gap:6px">
          <span style="width:8px;height:8px;border-radius:50%;
                background:{dot_color};display:inline-block"></span>
          {label}
        </span>
        <span>{pct:.1f}%</span>
      </div>
      <div class="bar-bg">
        <div class="{css}" style="width:{min(pct,100):.1f}%"></div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  TAMPILAN UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <div class="app-chip">CNN · MobileNetV2 · Skripsi 2025</div>
  <h1>🐟 Fish Freshness Detector</h1>
  <p>Upload foto ikan — model akan mendeteksi tingkat kesegaran secara otomatis</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model klasifikasi…"):
    model, model_path = load_classifier()

if model is None:
    st.error(
        "**Model tidak ditemukan!**  \n"
        "Letakkan `model_final.keras` atau `model_final.h5` "
        "di folder yang sama dengan `app.py`."
    )
    st.stop()

with st.spinner("Memuat validator gambar…"):
    load_validator()

uploaded = st.file_uploader(
    "Pilih foto ikan (JPG · PNG · BMP · WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded:
    pil_img = Image.open(uploaded)
    col_foto, col_hasil = st.columns([1, 1], gap="large")

    with col_foto:
        st.image(pil_img, use_container_width=True,
                 caption=f"{pil_img.width} × {pil_img.height} px")

    with col_hasil:
        st.markdown("#### Hasil Analisis")

        with st.spinner("Memvalidasi gambar…"):
            lolos, pesan_tolak = validate(pil_img)

        if not lolos:
            st.markdown(f"""
            <div class="box box-warn">
              <div class="box-title txt-warn">⚠️ Bukan Gambar Ikan</div>
              <div class="box-sub">{pesan_tolak}</div>
            </div>
            <p style="font-size:.83rem;color:#6b7280;margin-top:.4rem;line-height:1.7">
              Sistem hanya dapat menganalisis foto <b>ikan</b>.<br>
              Silakan upload foto ikan yang jelas.
            </p>
            """, unsafe_allow_html=True)

        else:
            with st.spinner("Menganalisis kesegaran…"):
                arr = preprocess(pil_img)
                status, conf, prob_fresh, prob_not = predict(model, arr)

            if status == 'fresh':
                st.markdown(f"""
                <div class="box box-fresh">
                  <div class="box-title txt-fresh">✅ Segar (Fresh)</div>
                  <div class="box-sub">Tingkat keyakinan:
                    <b style="color:#059669">{conf*100:.2f}%</b>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="box box-notfresh">
                  <div class="box-title txt-notfresh">❌ Tidak Segar (Not Fresh)</div>
                  <div class="box-sub">Tingkat keyakinan:
                    <b style="color:#dc2626">{conf*100:.2f}%</b>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                "<p style='font-size:.8rem;color:#6b7280;margin:.6rem 0 .2rem;"
                "font-weight:600'>PROBABILITAS PER KELAS</p>" +
                bar("Segar (Fresh)",           "#059669", prob_fresh * 100, "bar-fill-g") +
                bar("Tidak Segar (Not Fresh)", "#dc2626", prob_not   * 100, "bar-fill-r"),
                unsafe_allow_html=True,
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            with st.expander("🔍 Detail Teknis"):
                st.markdown(f"""
                <div class="info-grid">
                  <div class="info-cell">Arsitektur<br><b>MobileNetV2</b></div>
                  <div class="info-cell">Ukuran Input<br><b>224 × 224 px</b></div>
                  <div class="info-cell">Prob. Segar<br><b>{prob_fresh*100:.4f}%</b></div>
                  <div class="info-cell">Prob. Tidak Segar<br><b>{prob_not*100:.4f}%</b></div>
                  <div class="info-cell">Threshold<br><b>0.5 (sigmoid)</b></div>
                  <div class="info-cell">Prediksi<br>
                    <b>{"FRESH ✅" if status=="fresh" else "NOT FRESH ❌"}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;color:#9ca3af;padding:2rem 0;font-size:.9rem">
      ⬆️ &nbsp; Upload foto ikan di atas untuk memulai analisis
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr>
<div style="text-align:center;color:#9ca3af;font-size:.75rem;padding:.2rem 0 .8rem">
  Klasifikasi Kesegaran Ikan · CNN MobileNetV2 · Skripsi 2025
</div>
""", unsafe_allow_html=True)