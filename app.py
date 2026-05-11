import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Depression Detection from MRI",
    layout="centered"
)

st.title("Depression Detection from Structural MRI")
st.write("Upload a preprocessed MRI volume file.")

uploaded_file = st.file_uploader("Upload MRI file", type=["npy"])

if uploaded_file is not None:
    volume = np.load(uploaded_file)

    st.success("File uploaded successfully")
    st.write("MRI volume shape:", volume.shape)

    if volume.ndim == 3:
        slice_index = volume.shape[0] // 2

        fig, ax = plt.subplots()
        ax.imshow(volume[slice_index, :, :], cmap="gray")
        ax.axis("off")

        st.subheader("Middle MRI Slice")
        st.pyplot(fig)

    st.subheader("Prediction")
    st.warning("Model prediction is not connected yet.")
