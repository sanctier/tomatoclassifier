from pathlib import Path
import json
import time

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

/* Slimmer centered content */
[data-testid="stAppViewContainer"] .block-container {
        max-width: 980px;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
}

h1, h2, h3, h4 { color: var(--fg); letter-spacing: -0.02em; }

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 24px 28px;
    border-radius: 18px;
    background: #0a0a0a;
    border: 1px solid #0a0a0a;
    box-shadow: var(--shadow);
}

.hero h1 {
    margin-bottom: 4px;
    color: #ffffff;
}

.hero p {
    color: #ffffff;
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

/* Ensure all buttons keep white text on dark background */
button[kind],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"],
.stDownloadButton > button,
.stForm button,
[data-testid="stFileUploader"] button {
    background: var(--fg) !important;
    color: var(--bg) !important;
    border-color: var(--fg) !important;
}

button[kind]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover,
.stDownloadButton > button:hover,
.stForm button:hover,
[data-testid="stFileUploader"] button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

button[kind] *,
[data-testid="baseButton-secondary"] *,
[data-testid="baseButton-primary"] *,
.stDownloadButton > button *,
.stForm button *,
[data-testid="stFileUploader"] button * {
    color: var(--bg) !important;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
    box-shadow: var(--shadow);
}

/* Column cards */
[data-testid="column"] > div {
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

.align-after-image {
    margin-top: 260px;
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

/* File uploader dropzone text */
[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0a0a0a !important;
    border-color: #0a0a0a !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "MobileNetV2_hypertuned_final.tflite"
LABELS_PATH = ROOT_DIR / "models" / "labels.json"


@st.cache_resource(show_spinner=False)
def load_labels():
    if not LABELS_PATH.exists():
        return []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return [label_map[str(i)] for i in range(len(label_map))]


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


def format_label(label: str) -> str:
    if "___" in label:
        return label.split("___", 1)[1].replace("_", " ")
    return label.replace("_", " ")


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
    show_chart = st.checkbox("Show confidence chart", value=True)


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
    st.markdown("### Upload Image")
    st.markdown("<div class='muted'>JPEG or PNG, clear leaf close-up works best.</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop a tomato leaf image (JPG/PNG).",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

image = None
preds = None
elapsed_ms = None

if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File too large. Please upload an image under 10MB.")
        st.stop()

    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Could not read the image. Please upload a valid JPG/PNG file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        start_time = time.perf_counter()
        input_data = preprocess_image(image, target_size=224)
        preds = predict_tflite(interpreter, input_data)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

with col_left:
    if preds is not None:
        st.markdown("<div class='align-after-image'>", unsafe_allow_html=True)
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
                st.write(f"{rank}. {format_label(labels[idx])} — {preds[idx]:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("### Prediction")

    if uploaded_file is None:
        st.info("Upload an image to see predictions.")
    else:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        top_idx = int(np.argmax(preds))
        top_label = labels[top_idx]
        top_conf = float(preds[top_idx])

        st.markdown(
            f"<h3>Prediction: <span class='badge'>{format_label(top_label)}</span></h3>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(top_conf, 0.0), 1.0))
        st.write(f"**Confidence:** {top_conf:.4f}")
        st.caption(f"Inference time: {elapsed_ms:.1f} ms")

        if show_chart:
            st.markdown("#### Confidence Chart")
            order = np.argsort(preds)[::-1]
            chart_labels = [format_label(labels[i]) for i in order]
            chart_values = [float(preds[i]) for i in order]
            chart_df = {
                "Class": chart_labels,
                "Confidence": chart_values,
            }
            st.bar_chart(chart_df, x="Class", y="Confidence")

        if show_meta:
            st.markdown("#### Image Metadata")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]}")
            st.write(f"**Mode:** {image.mode}")
            st.write(f"**Format:** {image.format}")



st.markdown("---")
st.markdown(
    "<div class='muted'>Tip: Center the disease region and avoid blurry photos for best accuracy.</div>",
    unsafe_allow_html=True,
)
