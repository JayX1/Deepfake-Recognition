# cd C:\Users\blend\PycharmProjects\Deepfake_recognition
# streamlit run main.py

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

# --- paths ---
BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = (256, 256)

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Detekcja Deepfake")
st.caption("Drag & drop an image.")

WEIGHTS_PATH = Path(r"C:\Users\blend\Desktop\deepfake_model_weights.weights.h5")



def build_model():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(256, 256, 3))
    base.trainable = False  # inference

    model = tf.keras.Sequential([
        tf.keras.layers.Input((256, 256, 3)),
        tf.keras.layers.Lambda(preprocess_input),  # tak jak w treningu
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return model


@st.cache_resource
def load_model_from_weights():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")

    m = build_model()
    _ = m(tf.zeros((1, 256, 256, 3), dtype=tf.float32))  # tworzy zmienne wag
    m.load_weights(str(WEIGHTS_PATH))
    return m


model = load_model_from_weights()

thr = st.slider("Próg decyzyjny)", 0.0, 1.0, 0.5, 0.01)

uploaded = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    x = np.array(img.resize(IMG_SIZE), dtype=np.float32)  # 0..255
    x = np.expand_dims(x, 0)

    p_real = float(model.predict(x, verbose=0).reshape(-1)[0])
    verdict = "REAL" if p_real >= thr else "DEEPFAKE"

    st.subheader(f"Verdict: {verdict}")
    st.write(f"Prawdopodobieństwo prawdziwości = **{p_real:.4f}**")
    st.write(f"Prawdopodobieństwo AI = **{1 - p_real:.4f}**")




