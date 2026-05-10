import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import urllib.request
import pickle
import requests
import io

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================

BREED_MODEL_URL = "https://github.com/Rahman-Lone/rabbit-app/releases/download/v1.1/rabbit_breed_final_model.keras"
WEIGHT_MODEL_URL = "https://github.com/Rahman-Lone/rabbit-app/releases/download/v1.1/rabbit_weight_model.keras"

BREED_MODEL_PATH = "rabbit_breed_final_model.keras"
WEIGHT_MODEL_PATH = "rabbit_weight_model.keras"
LABELS_PATH = "tlabels.txt"

BREED_IMG_SIZE = (299, 299)
WEIGHT_IMG_SIZE = (192, 264)

# Set up the Streamlit UI Page (Centered layout is best for mobile readability)
st.set_page_config(page_title="Rabbit AI Predictor", page_icon="🐇", layout="centered")

# Use a slightly smaller, more compact title for mobile
st.markdown("<h2 style='text-align: center;'>🐇 Rabbit AI Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload or take a photo to analyze your rabbit</p>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL & FILE LOADERS
# ==========================================

@st.cache_resource
def load_breed_model():
    if os.path.exists(BREED_MODEL_PATH) and os.path.getsize(BREED_MODEL_PATH) < 1000000:
        os.remove(BREED_MODEL_PATH)
            
    if not os.path.exists(BREED_MODEL_PATH):
        print("Downloading Breed Model (once)...") 
        urllib.request.urlretrieve(BREED_MODEL_URL, BREED_MODEL_PATH)
        
    return tf.keras.models.load_model(BREED_MODEL_PATH)

@st.cache_resource
def load_weight_model():
    if os.path.exists(WEIGHT_MODEL_PATH) and os.path.getsize(WEIGHT_MODEL_PATH) < 1000000:
        os.remove(WEIGHT_MODEL_PATH)
            
    if not os.path.exists(WEIGHT_MODEL_PATH):
        print("Downloading Weight Model (once)...")
        urllib.request.urlretrieve(WEIGHT_MODEL_URL, WEIGHT_MODEL_PATH)
        
    return tf.keras.models.load_model(WEIGHT_MODEL_PATH)

@st.cache_resource
def load_scaler():
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
    img_array = np.array(image)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_for_weight(image):
    image = image.convert("RGB")
    image = image.resize(WEIGHT_IMG_SIZE) 
    img_array = np.array(image, dtype="float32")
    img_array = (img_array / 127.5) - 1.0 
    return np.expand_dims(img_array, axis=0)

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

class_labels = load_labels()
if not class_labels:
    st.error(f"Missing {LABELS_PATH}.")
    st.stop()

# MOBILE TWEAK: Use a selectbox instead of radio buttons to save vertical screen space
option = st.selectbox(
    "📸 Choose image source:", 
    ("Upload Image", "Camera Capture", "ESP32-CAM via Ngrok")
)

image_to_process = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Select a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

elif option == "Camera Capture":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image_to_process = Image.open(camera_file)

elif option == "ESP32-CAM via Ngrok":
    st.info("Ensure your ESP32 is running and Ngrok is forwarding the tunnel.")
    ngrok_url = st.text_input("https://pureblood-kinfolk-cod.ngrok-free.dev ", placeholder="https://1234-abcd.ngrok-free.app")
    
    if st.button("📸 Capture from ESP32", use_container_width=True):
        if ngrok_url:
            capture_url = f"{ngrok_url.rstrip('/')}/capture"
            try:
                with st.spinner("Snapping photo from ESP32..."):
                    # Give it 15 seconds to connect and download the image
                    response = requests.get(capture_url, timeout=15)
                    
                if response.status_code == 200:
                    image_to_process = Image.open(io.BytesIO(response.content))
                    st.success("Photo captured!")
                else:
                    st.error("Failed to capture image. Is the camera streaming?")
            except Exception as e:
                st.error(f"Error connecting to camera: {e}")
        else:
            st.warning("Please paste your Ngrok URL first!")

# ==========================================
# 5. PREDICTION & RESULTS DISPLAY
# ==========================================

if image_to_process is not None:
    # use_container_width=True ensures the image scales down perfectly on mobile screens
    st.image(image_to_process, use_container_width=True)
    
    # MOBILE TWEAK: Make the button full width so it's easy to tap with a thumb
    if st.button("🔮 Predict Breed & Weight", use_container_width=True):
        with st.spinner("Analyzing..."):
            
            # --- BREED PREDICTION ---
            breed_model = load_breed_model()
            breed_processed = preprocess_image_for_breed(image_to_process)
            breed_predictions = breed_model.predict(breed_processed)
            
            predicted_index = np.argmax(breed_predictions[0])
            confidence = breed_predictions[0][predicted_index]
            predicted_label = class_labels[predicted_index]
            
            st.divider() # Visual break
            
            # MOBILE TWEAK: Use st.metric for large, readable numbers on small screens
            col1, col2 = st.columns(2)
            col1.metric("Primary Breed", predicted_label)
            col2.metric("Confidence", f"{confidence * 100:.1f}%")
            
            # --- WEIGHT PREDICTION ---
            try:
                weight_model = load_weight_model()
                scaler = load_scaler()
                
                weight_processed = preprocess_image_for_weight(image_to_process)
                scaled_weight_pred = weight_model.predict(weight_processed)
                real_weight = scaler.inverse_transform(scaled_weight_pred)
                
                st.metric("⚖️ Estimated Weight", f"{real_weight[0][0]:.2f} kg")
            except Exception as e:
                st.error(f"Could not calculate weight: {e}")

            # --- PROBABILITY BREAKDOWN ---
            with st.expander("📊 View Top 3 Breed Probabilities"):
                top_indices = np.argsort(breed_predictions[0])[-3:][::-1]
                for i in top_indices:
                    # Uses a progress bar for a nice visual representation of confidence
                    st.write(f"**{class_labels[i]}**")
                    st.progress(float(breed_predictions[0][i]))
