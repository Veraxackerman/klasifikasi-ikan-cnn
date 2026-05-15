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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 820px;
}

.app-header {
    text-align: center;
    padding: 2rem 1.5rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}

.app-header::after {
    content:'';
    position:absolute;
    bottom:-30px;
    right:-30px;
    width:130px;
    height:130px;
    background:rgba(255,255,255,.05);
    border-radius:50%;
}

.app-header h1 {
    color:#fff;
    font-size:1.8rem;
    font-weight:800;
    margin:0 0 .35rem;
}

.app-header p {
    color:rgba(255,255,255,.7);
    font-size:.9rem;
    margin:0;
}

.app-chip {
    display:inline-block;
    background:rgba(255,255,255,.12);
    color:#6ee7b7;
    font-size:.7rem;
    font-weight:700;
    letter-spacing:.08em;
    text-transform:uppercase;
    padding:3px 10px;
    border-radius:20px;
    margin-bottom:.7rem;
}

[data-testid="stFileUploader"] > div:first-child {
    border:2px dashed rgba(16,185,129,.45) !important;
    border-radius:14px !important;
    background:rgba(16,185,129,.04) !important;
    padding:1rem !important;
}

.box {
    border-radius:14px;
    padding:1.1rem 1.3rem;
    margin:.5rem 0 .8rem;
}

.box-fresh {
    background:rgba(16,185,129,.1);
    border:1.5px solid rgba(16,185,129,.4);
}

.box-notfresh {
    background:rgba(239,68,68,.1);
    border:1.5px solid rgba(239,68,68,.35);
}

.box-warn {
    background:rgba(234,179,8,.1);
    border:1.5px solid rgba(234,179,8,.35);
}

.box-title {
    font-size:1.25rem;
    font-weight:800;
    display:flex;
    align-items:center;
    gap:8px;
    margin-bottom:.25rem;
}

.txt-fresh {
    color:#059669;
}

.txt-notfresh {
    color:#dc2626;
}

.txt-warn {
    color:#b45309;
}

.box-sub {
    font-size:.83rem;
    color:#6b7280;
}

.bar-wrap {
    margin:.55rem 0;
}

.bar-head {
    display:flex;
    justify-content:space-between;
    font-size:.78rem;
    font-weight:600;
    margin-bottom:4px;
    color:#374151;
}

.bar-bg {
    background:rgba(0,0,0,.07);
    border-radius:99px;
    height:9px;
    overflow:hidden;
}

.bar-fill-g {
    height:100%;
    border-radius:99px;
    background:linear-gradient(90deg,#059669,#34d399);
}

.bar-fill-r {
    height:100%;
    border-radius:99px;
    background:linear-gradient(90deg,#dc2626,#f87171);
}

.info-grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:8px;
    margin-top:.6rem;
}

.info-cell {
    background:rgba(0,0,0,.04);
    border-radius:10px;
    padding:.6rem .9rem;
    font-size:.78rem;
    color:#6b7280;
    line-height:1.65;
}

.info-cell b {
    color:#111827;
}

[data-testid="stImage"] img {
    border-radius:12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

FISH_ALLOW = {
    'fish', 'salmon', 'eel', 'shark', 'stingray',
    'tuna', 'cod', 'trout', 'snapper', 'catfish',
    'tilapia', 'sardine', 'anchovy', 'mackerel',
    'shrimp', 'lobster', 'crab', 'squid', 'octopus'
}

NON_FISH_BLOCK = {
    'person', 'man', 'woman', 'human',
    'cat', 'dog', 'bird',
    'car', 'motorcycle',
    'phone', 'computer', 'laptop',
    'chair', 'table',
    'building', 'house',
    'cartoon', 'illustration',
    'logo', 'symbol', 'banner'
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():

    import tensorflow as tf

    for path in [
        'model_final.keras',
        'output/model_final.keras',
        'model_final.h5',
    ]:

        if os.path.exists(path):

            model = tf.keras.models.load_model(
                path,
                compile=False
            )

            return model, path

    return None, None

# ── Validator ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_validator():

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions
    )

    model = MobileNetV2(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )

    return model, preprocess_input, decode_predictions

# ── Validasi gambar ───────────────────────────────────────────────────────────
def imagenet_check(pil_img):

    MIN_FISH_PROB = 0.08
    MIN_BLOCK_PROB = 0.10

    val_model, preprocess_input, decode_predictions = load_validator()

    img = pil_img.convert('RGB').resize((224, 224))

    arr = np.array(img, dtype=np.float32)

    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, 0)

    preds = val_model.predict(arr, verbose=0)

    top10 = decode_predictions(preds, top=10)[0]

    top_label = top10[0][1].lower().replace('_', ' ')

    for _, lbl, prob in top10:

        lbl_clean = lbl.lower().replace('_', ' ')

        if prob >= MIN_FISH_PROB:

            if any(x in lbl_clean for x in FISH_ALLOW):

                return True, lbl_clean

    for _, lbl, prob in top10:

        lbl_clean = lbl.lower().replace('_', ' ')

        if prob >= MIN_BLOCK_PROB:

            if any(x in lbl_clean for x in NON_FISH_BLOCK):

                return False, lbl_clean

    return True, top_label

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(pil_img):

    img = pil_img.convert("RGB").resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

# ── Prediksi ──────────────────────────────────────────────────────────────────
def predict(model, arr):

    prob_notfresh = float(model.predict(arr, verbose=0)[0][0])

    prob_fresh = 1.0 - prob_notfresh

    if prob_notfresh >= THRESHOLD:
        return "not_fresh", prob_notfresh, prob_fresh, prob_notfresh

    return "fresh", prob_fresh, prob_fresh, prob_notfresh

# ── Progress Bar ──────────────────────────────────────────────────────────────
def progress_bar(label, value, css_class):

    return f"""
    <div class="bar-wrap">
        <div class="bar-head">
            <span>{label}</span>
            <span>{value:.1f}%</span>
        </div>

        <div class="bar-bg">
            <div class="{css_class}" style="width:{value:.1f}%"></div>
        </div>
    </div>
    """

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="app-chip">CNN · MobileNetV2 · Skripsi 2025</div>
    <h1>🐟 Fish Freshness Detector</h1>
    <p>Upload foto ikan untuk mendeteksi tingkat kesegaran secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Memuat model..."):

    model, model_path = load_classifier()

if model is None:

    st.error(
        "Model tidak ditemukan. "
        "Pastikan file model_final.keras tersedia."
    )

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOADER
# ═══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Upload foto ikan",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# HASIL
# ═══════════════════════════════════════════════════════════════════════════════
if uploaded:

    pil_img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:

        st.image(
            pil_img,
            use_container_width=True,
            caption=f"{pil_img.width} × {pil_img.height} px"
        )

    with col2:

        st.markdown("## Hasil Prediksi")

        with st.spinner("Memvalidasi gambar..."):

            lolos, label = imagenet_check(pil_img)

        if not lolos:

            st.markdown(f"""
            <div class="box box-warn">
                <div class="box-title txt-warn">
                    ⚠️ Objek Tidak Didukung
                </div>

                <div class="box-sub">
                    Terdeteksi sebagai:
                    <b>{label}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            arr = preprocess(pil_img)

            with st.spinner("Menganalisis gambar..."):

                status, conf, prob_fresh, prob_notfresh = predict(model, arr)

            if status == "fresh":

                st.markdown(f"""
                <div class="box box-fresh">
                    <div class="box-title txt-fresh">
                        ✅ Fresh
                    </div>

                    <div class="box-sub">
                        Tingkat keyakinan:
                        <b>{conf*100:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="box box-notfresh">
                    <div class="box-title txt-notfresh">
                        ❌ Not Fresh
                    </div>

                    <div class="box-sub">
                        Tingkat keyakinan:
                        <b>{conf*100:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Probabilitas")

            st.markdown(
                progress_bar(
                    "Fresh",
                    prob_fresh * 100,
                    "bar-fill-g"
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                progress_bar(
                    "Not Fresh",
                    prob_notfresh * 100,
                    "bar-fill-r"
                ),
                unsafe_allow_html=True
            )

            with st.expander("🔍 Detail Teknis"):

                st.markdown(f"""
                <div class="info-grid">

                    <div class="info-cell">
                        Arsitektur<br>
                        <b>MobileNetV2</b>
                    </div>

                    <div class="info-cell">
                        Input Size<br>
                        <b>224 × 224</b>
                    </div>

                    <div class="info-cell">
                        Prob Fresh<br>
                        <b>{prob_fresh*100:.4f}%</b>
                    </div>

                    <div class="info-cell">
                        Prob Not Fresh<br>
                        <b>{prob_notfresh*100:.4f}%</b>
                    </div>

                    <div class="info-cell">
                        Threshold<br>
                        <b>0.5</b>
                    </div>

                    <div class="info-cell">
                        Model<br>
                        <b>{os.path.basename(model_path)}</b>
                    </div>

                </div>
                """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div style="
        text-align:center;
        color:#9ca3af;
        padding:2rem 0;
        font-size:.9rem;
    ">
        ⬆️ Upload gambar ikan untuk memulai analisis
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<hr>

<div style="
    text-align:center;
    color:#9ca3af;
    font-size:.75rem;
    padding-top:.3rem;
">
    Klasifikasi Kesegaran Ikan · CNN MobileNetV2 · Skripsi 2025
</div>
""", unsafe_allow_html=True)
