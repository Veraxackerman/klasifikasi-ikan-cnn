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
}

.app-header h1 {
    color: white;
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: .3rem;
}

.app-header p {
    color: rgba(255,255,255,.7);
    font-size: .9rem;
}

.result-box {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}

.fresh {
    background: rgba(16,185,129,.1);
    border: 1.5px solid rgba(16,185,129,.4);
}

.notfresh {
    background: rgba(239,68,68,.1);
    border: 1.5px solid rgba(239,68,68,.35);
}

.bar-bg {
    background: rgba(0,0,0,.08);
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin-top: 5px;
}

.bar-fill-green {
    background: linear-gradient(90deg,#059669,#34d399);
    height: 100%;
}

.bar-fill-red {
    background: linear-gradient(90deg,#dc2626,#f87171);
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():

    import tensorflow as tf

    for path in [
        'model_final.keras',
        'output/model_final.keras',
    ]:
        if os.path.exists(path):

            model = tf.keras.models.load_model(
                path,
                compile=False
            )

            return model, path

    return None, None

# ── Preprocessing ─────────────────────────────────────────────────────────────
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
    <p style="margin-bottom:4px;font-size:.85rem;font-weight:600">
        {label} ({value:.2f}%)
    </p>

    <div class="bar-bg">
        <div class="{css_class}" style="width:{value:.2f}%"></div>
    </div>
    """

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1>🐟 Fish Freshness Detector</h1>
    <p>Klasifikasi kesegaran ikan menggunakan CNN MobileNetV2</p>
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
        "Pastikan file model_final.keras ada."
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

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            pil_img,
            use_container_width=True,
            caption=f"{pil_img.width} × {pil_img.height} px"
        )

    with col2:

        st.markdown("### Hasil Prediksi")

        arr = preprocess(pil_img)

        with st.spinner("Menganalisis gambar..."):

            status, conf, prob_fresh, prob_notfresh = predict(model, arr)

        if status == "fresh":

            st.markdown(f"""
            <div class="result-box fresh">
                <h3 style="color:#059669;">✅ Fresh</h3>
                <p>Tingkat keyakinan: <b>{conf*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-box notfresh">
                <h3 style="color:#dc2626;">❌ Not Fresh</h3>
                <p>Tingkat keyakinan: <b>{conf*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Probabilitas")

        st.markdown(
            progress_bar(
                "Fresh",
                prob_fresh * 100,
                "bar-fill-green"
            ),
            unsafe_allow_html=True
        )

        st.markdown(
            progress_bar(
                "Not Fresh",
                prob_notfresh * 100,
                "bar-fill-red"
            ),
            unsafe_allow_html=True
        )

        with st.expander("Detail Teknis"):

            st.write(f"Probabilitas Fresh: {prob_fresh:.4f}")
            st.write(f"Probabilitas Not Fresh: {prob_notfresh:.4f}")
            st.write(f"Threshold: {THRESHOLD}")
            st.write(f"Model: {model_path}")

else:

    st.markdown("""
    <div style="text-align:center;color:#9ca3af;padding:2rem 0">
        ⬆️ Upload gambar ikan untuk memulai analisis
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<hr>
<div style="text-align:center;color:#9ca3af;font-size:.75rem">
    CNN MobileNetV2 • Streamlit • Skripsi 2025
</div>
""", unsafe_allow_html=True)
