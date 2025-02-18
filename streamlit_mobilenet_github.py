import streamlit as st
from streamlit_extras.let_it_rain import rain
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
import time
import base64
from pathlib import Path
from st_social_media_links import SocialMediaIcons

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

def get_christmas_countdown():
    today = datetime.now()
    christmas_date = datetime(today.year, 12, 25)
    
    # Jika sudah melewati Natal tahun ini, hitung mundur ke Natal tahun depan
    if today > christmas_date:
        christmas_date = datetime(today.year + 1, 12, 25)
    
    delta = christmas_date - today
    
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return days, hours, minutes, seconds

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def Social(sidebarPos = False,heading = None):
    
    if heading != None:
        st.title(f":rainbow[{heading}]")
        
    social_media_links = [
            "https://www.linkedin.com/in/atanasius-surya-656a91207",
            "https://github.com/AtanasiusSuryaGunadharma",
            "https://www.instagram.com/ata.sur_"]

    social_media_icons = SocialMediaIcons(social_media_links) 

    social_media_icons.render(sidebar=sidebarPos, justify_content="center")
    
# Function to apply snowfall effect
def run_snow_animation():
    rain(emoji="❄️", font_size=20, falling_speed=10, animation_length="infinite")

st.balloons()
# Run snowfall animation
run_snow_animation()

# Change Background Streamlit
set_background(r"background1.gif")

# Add custom Christmas-themed background with snow animation
christmas_background = """
<style>
/* Set full-page background */
# [data-testid="stAppViewContainer"] {
#     background: linear-gradient(to bottom, #FF0000, #00FF00); /* Red to Green */
#     background-size: cover;
#     background-attachment: fixed;
#     color: white; /* Text color */
#     font-family: Arial, sans-serif;
# }

/* Hide default Streamlit styling for a cleaner look */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.8);
    color: black;
}

/* Perbaiki warna teks judul agar tetap terlihat */
h1, h2, h3, h4, h5, h6, p {
    color: blue; 
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); /* Efek bayangan */
}

/* Create snowflake animation */
.snowflake {
    color: #1caeed;
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
    with open(r"natal_lagu5.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode()

    audio_html = f"""
    <audio autoplay loop>
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("File audio tidak ditemukan. Pastikan 'natal_lagu3.mp3' sudah ada di direktori project.")

title_html = """
<div style="text-align: center; color: black; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); font-size: 50px; font-weight: bold;">
    🎄 Prediksi Kematangan Buah Naga - XXXX 🎅
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

# Upload multiple files in the main page
uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Mengambil countdown ke Natal
days, hours, minutes, seconds = get_christmas_countdown()

# Quotes Natal dengan tampilan menarik
quotes_html = f"""
<div style="background: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; margin-top: 20px; text-align: center;">
    <h2 style="color: #8B0000; font-family: 'Georgia', serif; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        ✨ Countdown Christmas! ✨
    </h2>
    <p style="color: #4B0082; font-size: 24px; font-weight: bold; font-family: 'Arial', sans-serif;">
        {days} Hari {hours} Jam {minutes} Menit
    </p>
    <p style="color: #4B0082; font-size: 16px; font-family: 'Arial', sans-serif;">
        "Natal bukanlah tentang hadiah yang kita terima, tetapi tentang cinta yang kita bagi.
        Dalam setiap senyum dan kebaikan yang kita berikan, di situlah makna Natal sesungguhnya."
    </p>
</div>
"""

# Display quotes on the main page
st.markdown(quotes_html, unsafe_allow_html=True)

st.sidebar.image(r"treeChristmas.png")
    
# Style for the prediction button
style_button = """
<style>
.button-prediksi {
    display: block;
    margin: 0 auto;
    text-align: center;
}
</style>
"""
st.markdown(style_button, unsafe_allow_html=True)

# Sidebar for prediction button and results
col1, col2, col3 = st.sidebar.columns([1, 1, 1])
with col2:
    Social(sidebarPos=True)
    if st.button("Prediksi"):
        st.snow()
        if uploaded_files:
            st.sidebar.write("### 🎁 Hasil Prediksi")
            with st.spinner('Memprediksi...'):
                for uploaded_file in uploaded_files:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Perform prediction
                    label, confidence = classify_image(uploaded_file.name)
                    
                    if label != "Error":
                        # Define colors for the bar and label
                        primary_color = "#00FF00"  # Green for "Matang"
                        secondary_color = "#FF0000"  # Red for "Mentah"
                        label_color = primary_color if label == "Matang" else secondary_color
                        
                        # Display prediction results
                        st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                        st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)
                        
                        # Display confidence scores
                        st.sidebar.write("**Confidence:**")
                        for i, class_name in enumerate(class_names):
                            st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")
                        
                        # Display custom progress bar
                        custom_progress_bar(confidence, primary_color, secondary_color)
                        
                        st.sidebar.write("---")
                    else:
                        st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
        else:
            st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

# Preview images in the main page
if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
        
# Tambahkan copyright di bagian bawah
copyright_html = """
<div style="text-align: center; margin-top: 5px; font-size: 14px; color: #000000; opacity: 0.8;">
    © 2024 Atanasius Surya. All Rights Reserved.
</div>
"""
st.markdown(copyright_html, unsafe_allow_html=True)

