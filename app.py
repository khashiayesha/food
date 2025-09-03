import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

MODEL_REPO = "ayeshaca/food101"
MODEL_FILE = "cnn_best_final (1).h5"

@st.cache_resource
def load_model_and_labels():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = tf.keras.models.load_model(model_path, compile=False)
    # labels di-hardcode
    labels = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
        "beef_tartare", "beet_salad", "beignets", "bibimbap",
        "bread_pudding", "breakfast_burrito"
    ]
    return model, labels

model, labels = load_model_and_labels()

st.set_page_config(page_title="Food101 Classifier")
st.title("Food101 Image Classifier")
uploaded_file = st.file_uploader("Upload gambar…", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True, caption="Gambar Input")
    arr = np.expand_dims(np.array(img.resize((224,224))) / 255.0, axis=0)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    st.subheader(f"Prediksi: {labels[idx]} — {preds[idx]*100:.2f}%")
else:
    st.info("Upload gambar untuk melihat prediksi.")
