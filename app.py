import tensorflow as tf

CLASS_NAMES = ["Control", "Depression"]

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("tuned_3d_cnn_model.keras")

model = load_cnn_model()

st.subheader("Prediction")

if st.button("Predict"):
    input_volume = volume.astype("float32")

    if np.max(input_volume) > 0:
        input_volume = input_volume / np.max(input_volume)

    if input_volume.ndim == 3:
        input_volume = np.expand_dims(input_volume, axis=-1)

    if input_volume.ndim == 4:
        input_volume = np.expand_dims(input_volume, axis=0)

    prediction = model.predict(input_volume)

    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    st.write("Prediction:", CLASS_NAMES[predicted_class])
    st.write(f"Confidence: {confidence:.2%}")
