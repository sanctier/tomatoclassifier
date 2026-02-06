import os
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


# --- App Config ---
st.set_page_config(
    page_title="Tomato Disease Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Styling ---
st.markdown(
    """
<style>
:root {
  --bg: #ffffff;
  --fg: #0b0b0b;
  --muted: #6b6b6b;
  --accent: #0f766e;
  --card: #f7f7f7;
  --border: #e6e6e6;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--fg);
}

h1, h2, h3, h4 { color: var(--fg); }

.stButton > button {
  background: var(--fg);
  color: var(--bg);
  border: 1px solid var(--fg);
  border-radius: 10px;
  padding: 0.5rem 1.2rem;
}

.stButton > button:hover {
  background: var(--accent);
  border-color: var(--accent);
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--fg);
  color: var(--bg);
  font-size: 12px;
}

.muted { color: var(--muted); }
</style>
""",
    unsafe_allow_html=True,
)


# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "MobileNetV2_hypertuned_final.tflite"
DATASET_TRAIN_DIR = ROOT_DIR / "tomato_dataset" / "train"


@st.cache_resource(show_spinner=False)
def load_labels():
    if DATASET_TRAIN_DIR.exists():
        class_names = sorted([d.name for d in DATASET_TRAIN_DIR.iterdir() if d.is_dir()])
        return class_names
    return []


@st.cache_resource(show_spinner=False)
def load_tflite_interpreter():
    if not MODEL_PATH.exists():
        return None
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: Image.Image, target_size=224):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_tflite(interpreter: tf.lite.Interpreter, input_data: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure dtype matches model expectation
    input_dtype = input_details[0]["dtype"]
    if input_data.dtype != input_dtype:
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data[0]


# --- UI ---
st.title("Tomato Disease Classifier")
st.caption("Upload a tomato leaf image and get predictions using your best TFLite model.")

with st.sidebar:
    st.markdown("## Model Settings")
    st.write(f"**Model:** {MODEL_PATH.name}")
    st.write(f"**Labels source:** {DATASET_TRAIN_DIR}")
    st.markdown("---")
    st.markdown("## Analysis Tools")
    show_top3 = st.checkbox("Show Top-3 predictions", value=True)
    show_meta = st.checkbox("Show image metadata", value=True)


labels = load_labels()
interpreter = load_tflite_interpreter()

if interpreter is None:
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

if not labels:
    st.error(f"No labels found in {DATASET_TRAIN_DIR}")
    st.stop()


col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop a tomato leaf image (JPG/PNG).",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Prediction")

    if uploaded_file is None:
        st.info("Upload an image to see predictions.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        input_data = preprocess_image(image, target_size=224)
        preds = predict_tflite(interpreter, input_data)

        top_idx = int(np.argmax(preds))
        top_label = labels[top_idx]
        top_conf = float(preds[top_idx])

        st.markdown(
            f"<h3>Prediction: <span class='badge'>{top_label}</span></h3>",
            unsafe_allow_html=True,
        )
        st.write(f"**Confidence:** {top_conf:.4f}")

        if show_top3:
            st.markdown("#### Top-3 Predictions")
            top3_idx = np.argsort(preds)[-3:][::-1]
            for rank, idx in enumerate(top3_idx, start=1):
                st.write(f"{rank}. {labels[idx]} — {preds[idx]:.4f}")

        if show_meta:
            st.markdown("#### Image Metadata")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]}")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {image.format}")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    "<div class='muted'>Tip: Use a clear leaf image with the disease region centered for best results.</div>",
    unsafe_allow_html=True,
)
