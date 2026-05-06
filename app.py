import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import urllib.request
import pickle

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================

# 🚨 GITHUB RELEASE DIRECT DOWNLOAD LINKS 🚨
BREED_MODEL_URL = "https://github.com/Rahman-Lone/rabbit-app/releases/download/v1.1/rabbit_breed_final_model.keras"
WEIGHT_MODEL_URL = "https://github.com/Rahman-Lone/rabbit-app/releases/download/v1.1/rabbit_weight_model.keras"

BREED_MODEL_PATH = "rabbit_breed_final_model.keras"
WEIGHT_MODEL_PATH = "rabbit_weight_model.keras"
LABELS_PATH = "tlabels.txt"

BREED_IMG_SIZE = (299, 299)
WEIGHT_IMG_SIZE = (192, 264) # As defined in your Jupyter Notebook

# Set up the Streamlit UI Page
st.set_page_config(page_title="Rabbit AI Predictor", page_icon="🐇")
st.title("🐇 Rabbit Breed & Weight AI Predictor")
st.write("Upload a picture or take a photo to analyze the rabbit!")

# ==========================================
# 2. MODEL & FILE LOADERS
# ==========================================

@st.cache_resource
def load_breed_model():
    # Self-healing check: delete broken/HTML files smaller than 1MB
    if os.path.exists(BREED_MODEL_PATH) and os.path.getsize(BREED_MODEL_PATH) < 1000000:
        os.remove(BREED_MODEL_PATH)
            
    if not os.path.exists(BREED_MODEL_PATH):
        st.info("Downloading Breed AI Model (this only happens once)...")
        urllib.request.urlretrieve(BREED_MODEL_URL, BREED_MODEL_PATH)
        
    return tf.keras.models.load_model(BREED_MODEL_PATH)

@st.cache_resource
def load_weight_model():
    # Self-healing check: delete broken/HTML files smaller than 1MB
    if os.path.exists(WEIGHT_MODEL_PATH) and os.path.getsize(WEIGHT_MODEL_PATH) < 1000000:
        os.remove(WEIGHT_MODEL_PATH)
            
    if not os.path.exists(WEIGHT_MODEL_PATH):
        st.info("Downloading Weight Estimator Model (this only happens once)...")
        urllib.request.urlretrieve(WEIGHT_MODEL_URL, WEIGHT_MODEL_PATH)
        
    return tf.keras.models.load_model(WEIGHT_MODEL_PATH)

@st.cache_resource
def load_scaler():
    # Load the MinMaxScaler to convert predictions back to KG
    with open("weight_scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        return None
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def preprocess_image_for_breed(image):
    image = image.convert("RGB")
    image = image.resize(BREED_IMG_SIZE)
    img_array = np.array(image, dtype="float32")
    img_array = img_array / 255.0 # Standard normalization (adjust if you used MobileNet etc.)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_for_weight(image):
    image = image.convert("RGB")
    image = image.resize(WEIGHT_IMG_SIZE) 
    img_array = np.array(image, dtype="float32")
    img_array = (img_array / 127.5) - 1.0 # Your exact notebook scaling [-1, 1]
    return np.expand_dims(img_array, axis=0)

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

# Initialize labels early so it crashes gracefully if missing
class_labels = load_labels()
if not class_labels:
    st.error(f"Missing {LABELS_PATH}. Please make sure it is uploaded to your GitHub repository.")
    st.stop()

# User Input Menu
option = st.radio("Choose how to provide an image:", ("Upload Image", "Camera Capture", "ESP32-CAM URL (Coming Soon)"))
image_to_process = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Select a rabbit photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

elif option == "Camera Capture":
    camera_file = st.camera_input("Take a clear picture of the rabbit")
    if camera_file is not None:
        image_to_process = Image.open(camera_file)

elif option == "ESP32-CAM URL (Coming Soon)":
    st.info("We will configure the Ngrok ESP32-CAM stream here next!")

# ==========================================
# 5. PREDICTION & RESULTS DISPLAY
# ==========================================

if image_to_process is not None:
    st.image(image_to_process, caption="Image ready for analysis", use_container_width=True)
    
    if st.button("Predict Breed & Weight"):
        with st.spinner("Analyzing rabbit features..."):
            
            # --- 1. BREED PREDICTION ---
            breed_model = load_breed_model()
            breed_processed = preprocess_image_for_breed(image_to_process)
            breed_predictions = breed_model.predict(breed_processed)
            
            predicted_index = np.argmax(breed_predictions[0])
            confidence = breed_predictions[0][predicted_index]
            predicted_label = class_labels[predicted_index]
            
            st.success(f"**Breed Prediction:** {predicted_label}")
            st.info(f"**Breed Confidence:** {confidence * 100:.2f}%")
            
            # --- 2. WEIGHT PREDICTION ---
            try:
                weight_model = load_weight_model()
                scaler = load_scaler()
                
                weight_processed = preprocess_image_for_weight(image_to_process)
                scaled_weight_pred = weight_model.predict(weight_processed)
                
                real_weight = scaler.inverse_transform(scaled_weight_pred)
                
                # Real weight prediction display
                st.warning(f"⚖️ **Estimated Weight:** {real_weight[0][0]:.2f} kg")
            except Exception as e:
                st.error(f"Could not calculate weight: {e}")

            # --- 3. PROBABILITY BREAKDOWN ---
            st.write("### Top Breed Probabilities:")
            top_indices = np.argsort(breed_predictions[0])[-3:][::-1]
            for i in top_indices:
                st.write(f"- **{class_labels[i]}:** {breed_predictions[0][i] * 100:.2f}%")
