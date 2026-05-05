import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
import urllib.request

# --- Constants ---
MODEL_PATH = "rabbit_breed_final_model.keras"
LABELS_PATH = "tlabels.txt"
IMG_SIZE = (299, 299)

# --- Caching to prevent reloading on every click ---
@st.cache_resource
def load_model():
    # PASTE YOUR COPIED GITHUB RELEASE LINK HERE:
    MODEL_URL = "https://github.com/Rahman-Lone/rabbit-app/releases/download/v1.1/rabbit_breed_final_model.keras"
  
    # If the model isn't already downloaded, download it from the release
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI Model for the first time. This might take a minute...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        return []
    with open(LABELS_PATH, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def preprocess_image(image):
    # Standardize image format and size
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    # Expand dimensions to match model input (1, 299, 299, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Main Web App UI ---
st.set_page_config(page_title="Rabbit Breed Classifier", page_icon="🐰")
st.title("🐰 Rabbit Breed Classifier")
st.write("Upload, capture, or use an ESP32-CAM to predict the breed!")

# Load model and labels
model = load_model()
class_labels = load_labels()

if not class_labels:
    st.error(f"Error: {LABELS_PATH} not found. Please ensure it is uploaded to your repository.")

# --- INPUT METHODS: 3 Tabs ---
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📸 Web Camera", "📡 ESP32 Camera"])

image_to_process = None

with tab1:
    uploaded_file = st.file_uploader("Choose a rabbit image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

with tab2:
    camera_image = st.camera_input("Take a picture of the rabbit")
    if camera_image:
        image_to_process = Image.open(camera_image)

with tab3:
    st.info("Ensure your ESP32 is running the CameraWebServer sketch and is exposed to the internet via Ngrok (if deploying publicly).")
    esp_url = st.text_input("ESP32 Capture URL:", value="http://192.168.1.100/capture")
    
    if st.button("Capture from ESP32"):
        try:
            with st.spinner("Fetching image from ESP32..."):
                response = requests.get(esp_url, timeout=10)
                if response.status_code == 200:
                    image_to_process = Image.open(BytesIO(response.content))
                    st.success("Successfully captured image!")
                else:
                    st.error(f"Failed to fetch. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to ESP32: {e}. Check the IP address or URL.")

# --- Classification Logic ---
if image_to_process is not None:
    # Display the image
    st.image(image_to_process, caption="Image for Classification", use_container_width=True)
    st.write("Classifying...")
    
    # Process and predict
    processed_image = preprocess_image(image_to_process)
    predictions = model.predict(processed_image)
    
    # Extract results
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    if class_labels:
        predicted_label = class_labels[predicted_index]
        
        # Display Results
        st.success(f"**Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
        
        # Display top 3 probabilities
        st.write("### Top Probabilities:")
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for i in top_indices:
            st.write(f"- **{class_labels[i]}:** {predictions[0][i] * 100:.2f}%")
