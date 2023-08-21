import streamlit as st
import tempfile
from PIL import Image
import tensorflow as tf
# import keras

st.title("Fashion MNIST")

model = tf.keras.models.load_model('model.h5')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open('trousejpg')

    st.image(image, caption='Sunrise by the mountains')