import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Animal Classification", page_icon="🐾")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'animal_classification_model.h5')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# Class indices based on alphabetical sorting of the folders
class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

st.title("Animal Classification 🐾")
st.write("Upload an image of an animal, and the model will predict its species from 15 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Predict'):
        if model is None:
            st.error("Model file (animal_classification_model.h5) not found. Please ensure the model is trained and saved in this directory.")
        else:
            with st.spinner('Predicting...'):
                img = image.resize((224, 224))
                img_array = np.array(img)
                if len(img_array.shape) == 2:
                    img_array = np.stack((img_array,)*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:,:,:3]
                    
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
                
                if predicted_class_idx < len(class_names):
                    predicted_class = class_names[predicted_class_idx]
                else:
                    predicted_class = f"Class Index: {predicted_class_idx}"
                
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")
