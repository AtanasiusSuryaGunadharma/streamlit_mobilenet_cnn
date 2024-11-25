import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model(r'D:\SURYA\UAJY\Semester 5\Asdos Machine Learning\Pemegang Modul\Modul CNN\Notebook\model3.h5')  # Path harus sesuai
class_names = ['Matang', 'Mentah']

# Function to preprocess and classify image
def classify_image(image_path):
    try:
        # Load and preprocess the image (consistent with notebook)
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Apply softmax for probability
        
        # Get class with highest confidence
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Streamlit UI
st.title("Prediksi Kematangan Buah Naga")

# Upload multiple files in the main page
uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Sidebar for prediction button and results
if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Perform prediction
            label, confidence = classify_image(uploaded_file.name)
            
            if label != "Error":
                # Display prediction results
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.write(f"**Prediksi:** {label}")
                
                # Display confidence scores
                st.sidebar.write("**Confidence:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")
                
                # Confidence bar for highest prediction
                highest_confidence = np.max(confidence) * 100
                st.sidebar.progress(int(highest_confidence))
                
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
