import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set title
st.title("Brain Tumor MRI Classification")
st.subheader("Upload an MRI image to predict tumor type")

# Define model path and download from Google Drive if needed
model_url = "https://drive.google.com/uc?id=1QfHAz5mdWBmHTmcCPNbfxN_S9mYQpdYa"
model_path = "cnn_model.h5"

if not os.path.exists(model_path):
    with st.spinner('Downloading model...'):
        gdown.download(model_url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Class names (update if different in your project)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# File uploader
uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display prediction
    st.markdown(f"### 🧠 Predicted Tumor Type: `{predicted_class.upper()}`")
    st.markdown(f"### 🔍 Confidence Score: `{confidence:.2f}%`")

    # Show all class probabilities
    st.markdown("#### Confidence per class:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name.capitalize()}: {predictions[i]*100:.2f}%")
