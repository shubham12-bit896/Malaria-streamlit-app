import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import streamlit as st
import gdown  

# ------------------------ CONFIG ------------------------
IMG_SIZE = 128
MODEL_PATH = "model/malaria_model_fixed.h5"
DISPLAY_SIZE = 260   
MODEL_URL = "https://drive.google.com/file/d/1HUdTj4PLBDuKOpPBNAhDDF_Mq49UgtPc/view?usp=drive_link"  


# ------------------------ MODEL LOADING -------------------
@st.cache_resource
def load_model():
    os.makedirs("model", exist_ok=True)

    # 👉 Automatically download model if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Model (first time only)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ------------------- STREAMLIT PAGE SETUP ----------------
st.set_page_config(page_title="Malaria Detection AI", page_icon="🦠", layout="centered")

# UI Theme Styling
st.markdown("""
<style>
    .main {max-width: 750px; margin: auto;}
    .title {text-align:center; font-size:30px; font-weight:700; margin-bottom:6px;}
    .subtitle {text-align:center; font-size:15px; color:#666; margin-bottom:25px;}
    .result-box {
        text-align:center; padding:12px; border-radius:10px;
        font-size:18px; font-weight:600; margin-top:15px;
    }
    .infected {background:#ffe0e0; border:2px solid #cc0000; color:#b30000;}
    .uninfected {background:#e6ffe9; border:2px solid #009933; color:#005c1c;}
</style>
""", unsafe_allow_html=True)

# ------------------------ MODEL LOADING -------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Add it to `/model/` folder.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Identify last Conv layer for Grad-CAM
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("❌ No Conv2D layer found.")

LAST_CONV_LAYER = get_last_conv_layer(model)

# ------------------- IMAGE PREPROCESSING ------------------
def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# ------------------------ GRAD-CAM ------------------------
def generate_gradcam(img_tensor):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_tensor)
        prediction = tf.squeeze(prediction)  # sigmoid output
        loss = prediction

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    return heatmap, float(prediction)

# ------------------------ PREDICTION -----------------------
def predict(image):
    arr = preprocess(image)
    heatmap, prob = generate_gradcam(arr)

    if prob < 0.5:
        label = "🦠 Infected"
        confidence = round((1 - prob) * 100, 2)
        css = "infected"
    else:
        label = "🧪 Uninfected"
        confidence = round(prob * 100, 2)
        css = "uninfected"

    # Fix RGBA and resizing mismatch
    orig = np.array(image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    return label, confidence, overlay, css


# ------------------------- UI -----------------------------

st.markdown('<div class="title">🦠 Malaria Cell Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Diagnosis • Explainable Deep Learning (Grad-CAM)</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if file:
    img = Image.open(file)

    col1, col2 = st.columns(2)

    with col1:
        st.write("📌 Uploaded")
        st.image(img, width=DISPLAY_SIZE)

    with st.spinner("🔍 Analyzing..."):
        label, confidence, grad_img, css_class = predict(img)

    st.markdown(
        f'<div class="result-box {css_class}">{label}<br>Confidence: {confidence}%</div>',
        unsafe_allow_html=True
    )

    with col2:
        st.write("🔥 Model Attention")
        st.image(grad_img, width=DISPLAY_SIZE)

else:
    st.info("Upload a cell microscopy image to begin analysis.")
