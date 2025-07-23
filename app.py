import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = load_model("Brain_Tumor_CNN_Model.h5")

# Define class names (based on your dataset)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# App title
st.title("🧠 Brain Tumor MRI Classification App")
st.write("Upload an MRI image and the model will predict the tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Show prediction
    st.markdown(f"### 🧾 Prediction: **{predicted_class.upper()}**")
    st.markdown(f"#### 🎯 Confidence: **{confidence:.2f}%**")

    # Optional: Show all class probabilities
    st.markdown("### 🔍 Class-wise Confidence:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")
