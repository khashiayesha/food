import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Konfigurasi Model 
MODEL_REPO = "ayeshaca/food101"         # repo Hugging Face
MODEL_FILE = "cnn_best_final (1).h5"    # nama file model di repo

@st.cache_resource
def load_model_and_labels():
    # Download model dari Hugging Face Hub
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Cek ukuran input otomatis
    input_shape = model.input_shape
    target_size = input_shape[1:3]

    # Label 
    labels = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
        "beef_tartare", "beet_salad", "beignets", "bibimbap",
        "bread_pudding", "breakfast_burrito"
    ]
    return model, labels, target_size

#  Load model 
model, labels, target_size = load_model_and_labels()

#  Streamlit UI 
st.set_page_config(page_title="Food101 Classifier")
st.title("🍱 Food101 Image Classifier")
uploaded_file = st.file_uploader("Upload gambar makanan (jpg/jpeg/png):", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Input", use_column_width=True)

    # Preprocessing sesuai ukuran model
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediksi
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    st.subheader(f"Prediksi: **{labels[idx]}** ({preds[idx]*100:.2f}%)")
else:
    st.info("Silakan upload gambar terlebih dahulu.")
