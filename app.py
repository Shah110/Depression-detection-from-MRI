import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CLASS_NAMES = ["Control", "Depression"]

st.set_page_config(
    page_title="Depression Detection from MRI",
    layout="centered"
)

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("tuned_3d_cnn_model.keras")

def prepare_for_prediction(volume):
    volume = volume.astype("float32")

    if np.max(volume) > 0:
        volume = volume / np.max(volume)

    if volume.ndim == 3:
        volume = np.expand_dims(volume, axis=-1)

    if volume.ndim == 4:
        volume = np.expand_dims(volume, axis=0)

    return volume

st.title("Depression Detection from Structural MRI")
st.write("Upload a preprocessed MRI volume file.")

model = load_cnn_model()

uploaded_file = st.file_uploader("Upload MRI file", type=["npy"])

if uploaded_file is not None:
    volume = np.load(uploaded_file)

    st.success("File uploaded successfully")
    st.write("Original MRI volume shape:", volume.shape)

    display_volume = volume[:, :, :, 0] if volume.ndim == 4 else volume

    if display_volume.ndim == 3:
        slice_index = display_volume.shape[0] // 2

        fig, ax = plt.subplots()
        ax.imshow(display_volume[slice_index, :, :], cmap="gray")
        ax.axis("off")

        st.subheader("Middle MRI Slice")
        st.pyplot(fig)

    input_volume = prepare_for_prediction(volume)

    st.write("Model input shape:", input_volume.shape)

    if st.button("Predict"):
        prediction = model.predict(input_volume)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        st.subheader("Prediction Result")
        st.write("Class:", CLASS_NAMES[predicted_class])
        st.write(f"Confidence: {confidence:.2%}")

        st.subheader("Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {prediction[0][i]:.2%}")
