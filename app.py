import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["Control", "Depression"]
IMAGE_DIM = 64

st.set_page_config(
    page_title="Depression Detection from MRI",
    layout="centered"
)

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("tuned_3d_cnn_model.keras")

def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((IMAGE_DIM, IMAGE_DIM))

    image_array = np.array(image).astype("float32") / 255.0

    volume = np.repeat(image_array[np.newaxis, :, :], IMAGE_DIM, axis=0)

    volume = np.expand_dims(volume, axis=-1)
    volume = np.expand_dims(volume, axis=0)

    return image, volume

st.title("Depression Detection from Structural MRI")
st.write("Upload an MRI scan image.")

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    original_image, input_volume = preprocess_uploaded_image(uploaded_file)

    st.success("MRI image uploaded successfully")

    st.subheader("Uploaded MRI Image")
    st.image(original_image, caption="Uploaded MRI Scan", use_container_width=True)

    st.write("Model input shape:", input_volume.shape)

    model = load_cnn_model()

    if st.button("Predict"):
        prediction = model.predict(input_volume)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        st.subheader("Prediction Result")
        st.write("Prediction:", CLASS_NAMES[predicted_class])
        st.write(f"Confidence: {confidence:.2%}")

        st.subheader("Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prediction[0][i]:.2%}")
