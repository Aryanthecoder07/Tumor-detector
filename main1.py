import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Load the trained model
model = load_model("model.h5")  # Make sure model.h5 is in the same folder

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

    st.success(f"**Result:** {result}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
