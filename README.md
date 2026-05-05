# Rabbit Recognition System

> **A Streamlit-based web application utilizing deep learning to automatically identify rabbit breeds and estimate their weight within a controlled environment.**

### 📖 Overview
The Rabbit Recognition System is designed to streamline the monitoring and cataloging of rabbits. By operating in a controlled environment (consistent lighting, specific camera angles, and background), the system analyzes uploaded images to accurately classify the specific breed of the rabbit and provide a non-invasive estimation of its weight. 

### ✨ Key Features
* **Breed Identification:** Utilizes a Convolutional Neural Network (CNN) to automatically classify the rabbit's breed based on visual characteristics.
* **Weight Estimation:** Calculates an estimated weight without requiring physical scales, minimizing stress on the animals.
* **Interactive Web UI:** Built with Streamlit, allowing users to easily upload images (or provide image URLs) and get real-time predictions.
* **Controlled Environment Optimization:** Tuned for high accuracy under specific, predefined environmental conditions.

### 🛠️ Technology Stack
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (CPU version)
* **Core Model Architecture:** InceptionV3 (Transfer Learning)
* **Frontend UI:** Streamlit
* **Image Processing & Data Handling:** Pillow (PIL), NumPy, Requests

### 🚀 Getting Started

#### Prerequisites
Ensure you have Python installed on your machine (Python 3.8 or higher is recommended). 

#### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rahman-Lone/rabbit-app.git](https://github.com/Rahman-Lone/rabbit-app.git)
   cd rabbit-app
