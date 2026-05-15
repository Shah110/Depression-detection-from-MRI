
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import tempfile
import os


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def resize_volume(volume, target_shape=(128, 128, 128)):
    factors = [
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2]
    ]
    return zoom(volume, factors, order=1)


def normalize_volume(volume):
    volume = volume.astype(np.float32)

    min_val = np.min(volume)
    max_val = np.max(volume)

    if max_val - min_val == 0:
        return volume

    return (volume - min_val) / (max_val - min_val)


def preprocess_mri(file_path):
    img = nib.load(file_path)
    volume = img.get_fdata()

    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]

    volume = resize_volume(volume)
    volume = normalize_volume(volume)

    volume = np.expand_dims(volume, axis=0)
    volume = np.expand_dims(volume, axis=0)

    return torch.tensor(volume, dtype=torch.float32), volume[0, 0]


@st.cache_resource
def load_model():
    model = Simple3DCNN()
    model_path = "simple_3dcnn.pth"

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


st.set_page_config(
    page_title="Depressive Disorder Detection from Structural MRI",
    layout="centered"
)

st.title("Depressive Disorder Detection Using Structural MRI")

st.write(
    "Upload a T1-weighted structural MRI scan in `.nii` or `.nii.gz` format. "
    "The model will preprocess the scan and predict whether it belongs to "
    "Healthy Control or MDD class."
)

st.warning(
    "This app is a research prototype. It is not a medical diagnosis tool."
)

uploaded_file = st.file_uploader(
    "Upload MRI scan",
    type=["nii", "gz"]
)

model = load_model()

if uploaded_file is not None:
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        st.success("MRI uploaded successfully.")

        image_tensor, processed_volume = preprocess_mri(temp_path)

        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()

        if probability >= 0.5:
            prediction = "MDD"
        else:
            prediction = "Healthy Control"

        st.subheader("Prediction Result")
        st.write(f"Predicted Class: {prediction}")
        st.write(f"MDD Probability: {probability:.4f}")

        st.subheader("Processed MRI Slice")
        middle_slice = processed_volume[:, :, processed_volume.shape[2] // 2]
        st.image(
            middle_slice,
            caption="Middle slice after preprocessing",
            clamp=True
        )

    except Exception as error:
        st.error(f"Error processing MRI file: {error}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
