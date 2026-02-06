from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


# --- App Config ---
st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Styling ---
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #ffffff;
    --fg: #0a0a0a;
    --muted: #666666;
    --accent: #0b7a75;
    --card: #f5f6f7;
    --border: #e3e5e8;
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--fg);
    font-family: 'Space Grotesk', system-ui, -apple-system, Segoe UI, sans-serif;
}

h1, h2, h3, h4 { color: var(--fg); letter-spacing: -0.02em; }

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 24px 28px;
    border-radius: 18px;
    background: radial-gradient(1200px 600px at 0% 0%, #f0f7f7 0%, #ffffff 50%) ,
                            radial-gradient(900px 500px at 100% 0%, #f6f6f6 0%, #ffffff 55%);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.hero h1 {
    margin-bottom: 4px;
}

.hero p {
    color: var(--muted);
    margin: 0;
}

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
    box-shadow: var(--shadow);
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--fg);
  color: var(--bg);
    font-size: 12px;
    font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

.muted { color: var(--muted); }

.kpi {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
}

.kpi .item {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
}

.kpi .label {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 2px;
}

.kpi .value {
    font-weight: 600;
    font-size: 16px;
}

/* File uploader contrast */
[data-testid="stFileUploader"] button {
    background: var(--fg) !important;
    color: var(--bg) !important;
    border: 1px solid var(--fg) !important;
    border-radius: 10px;
}

[data-testid="stFileUploader"] button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

[data-testid="stFileUploader"] * {
    color: var(--fg);
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "MobileNetV2_hypertuned_final.tflite"


CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


@st.cache_resource(show_spinner=False)
def load_labels():
    return CLASS_NAMES


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
st.markdown(
        """
<div class="hero">
    <h1>Tomato Disease Classifier</h1>
    <p>Drop a leaf image to get an instant, high-confidence prediction.</p>
</div>
""",
        unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Model Settings")
    st.write(f"**Model:** {MODEL_PATH.name}")
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
    st.error("No labels configured in the app.")
    st.stop()


col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Upload Image")
    st.markdown("<div class='muted'>JPEG or PNG, clear leaf close-up works best.</div>", unsafe_allow_html=True)
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
        st.progress(min(max(top_conf, 0.0), 1.0))
        st.write(f"**Confidence:** {top_conf:.4f}")

        st.markdown(
            """
<div class="kpi">
    <div class="item">
        <div class="label">Model</div>
        <div class="value">TFLite</div>
    </div>
    <div class="item">
        <div class="label">Input Size</div>
        <div class="value">224×224</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

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
    "<div class='muted'>Tip: Center the disease region and avoid blurry photos for best accuracy.</div>",
    unsafe_allow_html=True,
)
