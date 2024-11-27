import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64

# Load the pre-trained model
model = load_model(r'model_mobilenet.h5')  # Update with your model path
class_names = ['Matang', 'Mentah']

# Function to preprocess and classify image
def classify_image(image_path):
    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Apply softmax for probability
        
        # Get class with highest confidence
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Function to create a custom progress bar
def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100  # Confidence for class 0 (Matang)
    percentage2 = confidence[1] * 100  # Confidence for class 1 (Mentah)
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}% Matang
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}% Mentah
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

# Add custom Christmas-themed background with snow animation
christmas_background = """
<style>
/* Set full-page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #FF0000, #00FF00); /* Red to Green */
    background-size: cover;
    background-attachment: fixed;
    color: white; /* Text color */
    font-family: Arial, sans-serif;
}

/* Hide default Streamlit styling for a cleaner look */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.7);
    color: white;
}

/* Perbaiki warna teks judul agar tetap terlihat */
h1, h2, h3, h4, h5, h6, p {
    color: white; /* Warna default */
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); /* Efek bayangan */
}

/* Create snowflake animation */
.snowflake {
    color: white;
    font-size: 1.5em;
    position: absolute;
    top: -10%;
    animation: snow 10s linear infinite;
    opacity: 0.8;
    z-index: 10;
}

@keyframes snow {
    0% { transform: translateY(0); }
    100% { transform: translateY(100vh); }
}

.snowflake:nth-child(1) { left: 10%; animation-delay: 0s; }
.snowflake:nth-child(2) { left: 20%; animation-delay: 2s; }
.snowflake:nth-child(3) { left: 30%; animation-delay: 4s; }
.snowflake:nth-child(4) { left: 40%; animation-delay: 6s; }
.snowflake:nth-child(5) { left: 50%; animation-delay: 8s; }
.snowflake:nth-child(6) { left: 60%; animation-delay: 1s; }
.snowflake:nth-child(7) { left: 70%; animation-delay: 3s; }
.snowflake:nth-child(8) { left: 80%; animation-delay: 5s; }
.snowflake:nth-child(9) { left: 90%; animation-delay: 7s; }
.snowflake:nth-child(10) { left: 100%; animation-delay: 9s; }
</style>

<div class="snowflake">❄</div>
<div class="snowflake">❅</div>
<div class="snowflake">❆</div>
<div class="snowflake">❄</div>
<div class="snowflake">❅</div>
<div class="snowflake">❆</div>
<div class="snowflake">❄</div>
<div class="snowflake">❅</div>
<div class="snowflake">❆</div>
<div class="snowflake">❄</div>
"""

# Display the background animation and snowflakes
st.markdown(christmas_background, unsafe_allow_html=True)

# Menambahkan audio autoplay menggunakan HTML
try:
    with open(r"natal_lagu.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode()

    audio_html = f"""
    <audio autoplay loop>
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("File audio tidak ditemukan. Pastikan 'natal_lagu.mp3' sudah ada di direktori project.")

# Streamlit UI
st.title("🎄 Prediksi Kematangan Buah Naga 🎅")

# Quotes Natal dengan tampilan menarik
quotes_html = """
<div style="background: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
    <h2 style="color: #8B0000; font-family: 'Georgia', serif; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        ✨ Selamat Natal! ✨
    </h2>
    <p style="color: #4B0082; font-size: 18px; font-family: 'Arial', sans-serif;">
        "Natal bukanlah tentang hadiah yang kita terima, tetapi tentang cinta yang kita bagi.
        Dalam setiap senyum dan kebaikan yang kita berikan, di situlah makna Natal sesungguhnya."
    </p>
    <p style="color: #4B0082; font-size: 18px; font-family: 'Arial',
