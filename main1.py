import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import requests
from PIL import Image
import os

# Google Drive file ID (replace this with your actual file ID)
MODEL_URL = "https://drive.google.com/uc?id=1EgQFY9lGliSDX03BOJ4sv8nAgupmhONP&export=download"
MODEL_PATH = "model.h5"

# Function to download the model dynamically if it's not present locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")

# Download the model if not already downloaded
download_model()

# Load the trained model
model = load_model(MODEL_PATH)

# Set image size
IMAGE_SIZE = 150

# Set page configuration
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

# Display prediction when image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        result, confidence = predict_tumor(image)

    st.success(f"*Result:* {result}")
    st.info(f"*Confidence:* {confidence * 100:.2f}%")
