import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title("Image Upload and Classification")

uploaded_file = st.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model_choice = st.selectbox('Choose a model', ["CNN", "VGG16"])

    if model_choice == "CNN":
        model_path = 'models/fashion_mnist_cnn_model.h5'

        resized_image = ImageOps.grayscale(image)
        resized_image = resized_image.resize((28, 28))
        input_array = np.array(resized_image) / 255.0
        input_array = np.expand_dims(input_array, axis=(0, -1))
    else:
        model_path = 'models/vgg16_fashion_mnist_fine_tuned.h5'

        resized_image = image.resize((32, 32))
        input_array = np.array(resized_image) / 255.0
        if input_array.shape[-1] == 4:
            input_array = input_array[:, :, :3]
        elif input_array.shape[-1] == 1:
            input_array = np.repeat(input_array, 3, axis=-1)
        input_array = np.expand_dims(input_array, axis=0)

    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model {model_choice} successfully loaded!")
    except Exception as e:
        st.error(f"Error loading model {model_choice}: {e}")
        model = None

    if model is not None:
        predictions = model.predict(input_array)
        class_probabilities = predictions[0]
        predicted_class = np.argmax(class_probabilities)

        st.subheader("Probabilities for Each Class")
        for i, prob in enumerate(class_probabilities):
            st.write(f"{class_names[i]}: {prob:.2f}")

        st.subheader("Classification Result")
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Probability: {predictions[0][predicted_class]:.2f}")

        st.subheader("Probability Chart for Each Class")
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(class_probabilities)), class_probabilities, tick_label=class_names)
        plt.title('Probabilities for Each Class')
        plt.xlabel('Class Name')
        plt.ylabel('Probability')
        st.pyplot(plt)
