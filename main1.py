import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import gdown

# Google Drive model URL and save path
MODEL_URL = "https://drive.google.com/uc?id=1EgQFY9lGliSDX03BOJ4sv8nAgupmhONP"
MODEL_PATH = "model.h5"

# Function to download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model... Please wait."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")

# Download the model
download_model()

# Load model only if file is valid
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
    st.error("❌ Model file is missing or corrupted. Please check the download.")
    st.stop()
else:
    model = load_model(MODEL_PATH)

# Constants
IMAGE_SIZE = 150

# App configuration
st.set_page_config(page_title="Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image and let the model predict if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_tumor(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence_score = prediction

    if prediction < 0.5:
        return "No Tumor", (1 - confidence_score)
    else:
        return "Tumor Detected", confidence_score

# Handle prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        result, confidence = predict_tumor(image)

    st.success(f"🧾 *Result:* **{result}**")
    st.info(f"📊 *Confidence:* **{confidence * 100:.2f}%**")
