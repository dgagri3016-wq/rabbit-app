import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os
import requests

# --- CONFIGURATION ---
MODEL_URL = "https://github.com/dgagri3016-wq/rabbit-weight-app/releases/download/keras_model/rabbit_weight_model.keras"
MODEL_PATH = "rabbit_weight_model.keras"
SCALER_PATH = "weight_scaler.pkl"

st.set_page_config(page_title="Rabbit Weight Predictor", page_icon="🐇", layout="centered")

st.markdown("<h2 style='text-align: center;'>🐇 Rabbit Weight Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered rabbit weight estimation using deep learning</p>", unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from GitHub Releases (This happens once)..."):
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
            else:
                st.error("Failed to download the model. Please check the MODEL_URL.")
                st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()
        
    return model, scaler

model, scaler = load_assets()

# --- INITIALIZE VARIABLE FOR UI ---
# This prevents the "NameError" by ensuring the variable exists before the button is clicked!
final_weight = None

# --- UI INTERFACE ---
st.write("### 📷 Choose Input Method")
input_mode = st.radio("Select Image Input Method:", ("Upload Image", "Take Picture"), label_visibility="collapsed")
image_data = None

if input_mode == "Upload Image":
    st.write("Upload a rabbit image (JPG, PNG)")
    image_data = st.file_uploader("Choose a rabbit image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
elif input_mode == "Take Picture":
    image_data = st.camera_input("Take a picture of the rabbit")

if image_data is not None:
    # Open the image
    image = Image.open(image_data)
    
    if input_mode == "Upload Image":
        st.image(image, caption="Image for Prediction", use_container_width=True)

    if st.button("Predict Weight", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # --- PREPROCESSING (Corrected for your specific model) ---
                img = image.convert('RGB')
                img = img.resize((192, 264))
                
                # 1. Convert to float32
                img_array = np.array(img, dtype="float32")
                
                # 2. Scale to [-1, 1] exactly like your other working app
                img_array = (img_array / 127.5) - 1.0
                
                # 3. Add batch dimension -> (1, 264, 192, 3)
                img_array = np.expand_dims(img_array, axis=0)

                # --- PREDICTION ---
                pred_scaled = model.predict(img_array)
                pred_weight = scaler.inverse_transform(pred_scaled)
                
                # Update the final_weight variable!
                final_weight = pred_weight[0][0]
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- CUSTOM HTML RESULTS DISPLAY ---
st.divider()
st.write("### Predicted Weight")

if final_weight is not None:
    # If the model has run successfully, show the real weight
    st.markdown(f'<div class="result-value" style="font-size: 24px; font-weight: bold; color: #4CAF50;">{final_weight:.2f} kg</div>', unsafe_allow_html=True)
else:
    # If no prediction has been made yet, show the placeholder
    st.markdown('<div class="result-value" style="font-size: 24px; font-weight: bold; color: gray;">-- kg</div>', unsafe_allow_html=True)
    if image_data is None:
        st.info("Upload an image or take a picture to begin prediction.")
