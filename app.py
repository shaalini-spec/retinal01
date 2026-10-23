
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "outputs/best_model.keras"
CLASSES_PATH = "outputs/class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Retinal OCT Classifier",
    page_icon="👁️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as file:
        return json.load(file)

st.title("Retinal OCT Disease Classification")
st.write(
    "Upload an OCT retinal scan to classify it as "
    "CNV, DME, DRUSEN, or NORMAL."
)

st.info(
    "Educational research project only. "
    "This tool does not provide a medical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload an OCT image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded OCT scan", use_container_width=True)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            model = load_model()
            class_names = load_classes()

            resized_image = image.resize(IMG_SIZE)
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            probabilities = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100

        st.success(f"Predicted class: {predicted_class}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Class probabilities")
        probability_data = {
            name: float(probability)
            for name, probability in zip(class_names, probabilities)
        }
        for name, probability in sorted(
            probability_data.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            st.write(f"**{name}** — {probability * 100:.2f}%")
            st.progress(float(probability))
