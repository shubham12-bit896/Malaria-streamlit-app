import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown

# ------------------------ CONFIG ------------------------
IMG_SIZE = 128
MODEL_PATH = "model/malaria_model_fixed.keras"   # NEW
MODEL_URL = "https://drive.google.com/uc?id=1bt9bLaCa5VBetWEtuaTzrzaJDIqWmZH7"
DISPLAY_SIZE = 250


# ------------------------ MODEL LOADING ------------------------
@st.cache_resource
def load_model():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

    return tf.keras.models.load_model(MODEL_PATH)  # new format loads cleanly


model = load_model()


# ------------------ Grad-CAM Helper -------------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("❌ No Conv2D layer found.")

LAST_CONV = get_last_conv_layer(model)


# ------------------------ PREPROCESS ------------------------
def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, 0)


# ------------------------ GRAD-CAM ------------------------
def gradcam(img_tensor):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(LAST_CONV).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_tensor)
        prob = prediction[0][0]  # sigmoid output

    grads = tape.gradient(prob, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-10

    return heatmap, float(prob)


# ------------------------ PREDICTION ------------------------
def predict(image):
    img_tensor = preprocess(image)
    heatmap, prob = gradcam(img_tensor)

    if prob < 0.5:
        label = "🦠 Infected (Parasitized)"
        confidence = round((1 - prob) * 100, 2)
        css = "infected"
    else:
        label = "🧪 Uninfected"
        confidence = round(prob * 100, 2)
        css = "uninfected"

    orig = np.array(image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_resized, 0.4, 0)

    return label, confidence, overlay, css


# ------------------------ STREAMLIT UI ------------------------
st.set_page_config(page_title="Malaria Detection AI", page_icon="🦠")

st.markdown("""
<style>
    .title {text-align:center; font-size:32px; font-weight:700;}
    .subtitle {text-align:center; color:#777; margin-bottom:20px;}
    .infected {background:#ffe0e0; padding:10px; border-radius:10px; border:2px solid #cc0000; color:#a30000;}
    .uninfected {background:#e6ffe9; padding:10px; border-radius:10px; border:2px solid #009933; color:#005c1c;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🦠 Malaria Cell Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning + Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload blood smear image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)

    col1, col2 = st.columns(2)
    with col1:
        st.write("📌 Uploaded Image")
        st.image(img, width=DISPLAY_SIZE)

    with st.spinner("Analyzing..."):
        label, confidence, grad_img, css_class = predict(img)

    st.markdown(f'<div class="{css_class}">{label}<br><b>Confidence: {confidence}%</b></div>', unsafe_allow_html=True)

    with col2:
        st.write("🔥 Model Attention (Grad-CAM)")
        st.image(grad_img, width=DISPLAY_SIZE)

else:
    st.info("Upload a microscope image to begin diagnosis.")