import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Fish Freshness Detector",
    page_icon="🐟",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 900px;
}

.app-header {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    color: white;
}

.app-header h1 {
    font-size: 2rem;
    margin-bottom: .4rem;
    font-weight: 800;
}

.app-header p {
    color: rgba(255,255,255,.8);
    margin: 0;
}

.result-box {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}

.fresh-box {
    background: rgba(16,185,129,.1);
    border: 1px solid rgba(16,185,129,.3);
}

.notfresh-box {
    background: rgba(239,68,68,.1);
    border: 1px solid rgba(239,68,68,.3);
}

.warning-box {
    background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.3);
}

.result-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: .3rem;
}

.green {
    color: #059669;
}

.red {
    color: #dc2626;
}

.orange {
    color: #d97706;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Kata kunci IKAN
FISH_ALLOW = {
    'fish', 'tench', 'goldfish', 'salmon', 'eel', 'shark', 'ray', 'stingray',
    'puffer', 'blowfish', 'lionfish', 'coho', 'carp', 'mackerel', 'tuna',
    'cod', 'trout', 'snapper', 'grouper', 'flounder', 'sole', 'halibut',
    'pike', 'barracuda', 'marlin', 'swordfish', 'anchovy', 'herring',
    'sardine', 'catfish', 'tilapia',
}

# Kata kunci BUKAN IKAN
NON_FISH_BLOCK = {

    # Manusia & atribut
    'person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'head',
    'human', 'baby', 'child', 'portrait',

    # Pakaian
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

    # Ilustrasi / media
    'comic', 'cartoon', 'illustration', 'drawing', 'anime', 'poster',
    'banner', 'envelope', 'jigsaw', 'puzzle',

    # Musik / panggung
    'piano', 'guitar', 'violin', 'drum', 'stage', 'theater',

    # Diagram / teknis
    'web', 'diagram', 'chart', 'graph', 'plot', 'abacus',

    # Seafood non ikan
    'shrimp', 'lobster', 'crab', 'squid', 'octopus', 'seafood',
}

@st.cache_resource
def load_classifier():

    model_paths = [
        "model_final.keras",
        "output/model_final.keras",
        "model_final.h5"
    ]

    for path in model_paths:

        if os.path.exists(path):

            model = tf.keras.models.load_model(
                path,
                compile=False
            )

            return model, path

    return None, None

@st.cache_resource
def load_validator():

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions
    )

    validator = MobileNetV2(
        weights="imagenet",
        include_top=True
    )

    return validator, preprocess_input, decode_predictions

def validate_image(pil_img):

    validator, preprocess_input, decode_predictions = load_validator()

    img = pil_img.convert("RGB").resize((224, 224))

    arr = np.array(img, dtype=np.float32)

    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    preds = validator.predict(arr, verbose=0)

    top20 = decode_predictions(preds, top=20)[0]

    for _, label, prob in top20:

        label = label.lower().replace("_", " ")

        if any(word in label for word in FISH_ALLOW):

            return True, label

    for _, label, prob in top20:

        label = label.lower().replace("_", " ")

        if any(word in label for word in NON_FISH_BLOCK):

            return False, label

    return False, top20[0][1]

def preprocess_image(pil_img):

    img = pil_img.convert("RGB").resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr

def predict(model, img_array):

    pred = float(model.predict(img_array, verbose=0)[0][0])

    prob_notfresh = pred
    prob_fresh = 1 - pred

    if pred >= THRESHOLD:
        label = "Not Fresh"
        confidence = prob_notfresh
    else:
        label = "Fresh"
        confidence = prob_fresh

    return label, confidence, prob_fresh, prob_notfresh

st.markdown("""
<div class="app-header">
    <h1>🐟 Fish Freshness Detector</h1>
    <p>Klasifikasi kesegaran ikan menggunakan CNN MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat model..."):

    model, model_path = load_classifier()

if model is None:

    st.error(
        "Model tidak ditemukan. "
        "Pastikan file model tersedia."
    )

    st.stop()

uploaded_file = st.file_uploader(
    "Upload foto ikan",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("## Hasil Klasifikasi")

        with st.spinner("Menganalisis gambar..."):

            valid, detected_label = validate_image(image)

        if not valid:
              st.warning(
                  f"Objek tidak didukung.\n\n"
                  f"Terdeteksi sebagai: {detected_label}"
              )
                 st.info(
                     "Silakan upload gambar ikan sesuai dataset penelitian."
                 )
        else:

            img_array = preprocess_image(image)

            label, confidence, prob_fresh, prob_notfresh = predict(
                model,
                img_array
            )

            if label == "Fresh":
                st.success(
                    f"Fresh\n\n"
                    f"Confidence Score: {confidence*100:.2f}%"

            else:
                st.error(
                    f"Not Fresh\n\n"
                    f"Confidence Score: {confidence*100:.2f}%"
            )
            st.markdown("### Probabilitas Klasifikasi")
            
            st.write(f"Fresh : {prob_fresh*100:.2f}%")
            st.progress(float(prob_fresh))
    
            st.write(f"Not Fresh : {prob_notfresh*100:.2f}%")
            st.progress(float(prob_notfresh))
            )

            st.write("Not Fresh")
            st.progress(float(prob_notfresh))
            st.caption(f"{prob_notfresh*100:.2f}%")

            with st.expander("Detail Teknis"):

                st.write(f"Model : {os.path.basename(model_path)}")
                st.write("Ukuran Input : 224 x 224")

else:

    st.info(
        "Upload gambar ikan untuk memulai klasifikasi."
    )

st.markdown("""
<hr>

<div style="
text-align:center;
font-size:.8rem;
color:#9ca3af;
">
CNN MobileNetV2 • Streamlit • Skripsi 2025
</div>
""", unsafe_allow_html=True)
