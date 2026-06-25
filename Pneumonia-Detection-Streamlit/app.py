import streamlit as st
import numpy as np
import os
import tensorflow as tf
from PIL import Image

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="X-Ray Pneumonia Classifier",
    page_icon="🫁",
    layout="centered",
)

IMG_SIZE = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "custom_pre_trained_model_10.keras")
CATEGORIES = ["NORMAL", "PNEUMONIA"]


# ----------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(file, img_size: int) -> np.ndarray:
    """Load an uploaded file into a (1, size, size, 3) normalized array."""
    image = Image.open(file).convert("RGB")  # forces 3 channels regardless of input mode
    image = image.resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = arr.reshape(1, img_size, img_size, 3)
    return arr


# ----------------------------
# UI
# ----------------------------
st.title("🫁 X-Ray Pneumonia Classifier")
st.write(
    "Upload a chest X-ray image and the model will predict whether it shows "
    "signs of **Pneumonia** or appears **Normal**."
)

with st.expander("ℹ️ About this project"):
    st.write(
        "This model uses a CNN built on transfer learning (VGG16) trained on "
        "chest X-ray images to classify scans as Normal or Pneumonia. "
        "Upload an image below to get a prediction."
    )
    st.warning(
        "⚠️ This is a personal/academic project and **not a medical diagnostic tool**. "
        "Do not use this for real medical decisions — consult a qualified radiologist or physician."
    )

model, load_error = load_model()

if load_error:
    st.error(
        "Couldn't load the model file. Make sure "
        f"`{MODEL_PATH}` is in the same directory as this app and was committed to your repo."
    )
    st.caption(f"Details: {load_error}")
    st.stop()

st.divider()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpeg", "jpg", "png"],
    help="Accepted formats: JPEG, JPG, PNG",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

    with col2:
        st.write("**Ready to analyze this image.**")
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing X-ray..."):
            try:
                processed = preprocess_image(uploaded_file, IMG_SIZE)
                prediction = model.predict(processed, verbose=0)
                raw_score = float(prediction[0][0])
                predicted_idx = int(round(raw_score))
                label = CATEGORIES[predicted_idx]

                # Confidence = how far the score sits toward the predicted class
                confidence = raw_score if predicted_idx == 1 else (1 - raw_score)
                confidence_pct = round(confidence * 100, 2)

            except Exception as e:
                st.error("Something went wrong while processing this image.")
                st.caption(f"Details: {e}")
            else:
                st.divider()
                if label == "PNEUMONIA":
                    st.error(f"**Prediction: {label}**")
                else:
                    st.success(f"**Prediction: {label}**")

                st.metric("Confidence", f"{confidence_pct}%")
                st.progress(confidence)

                st.caption(
                    "Confidence reflects the model's certainty in its predicted class, "
                    "not a clinical probability."
                )

st.divider()
st.caption("Built with Streamlit & TensorFlow • For educational/demo purposes only.")
