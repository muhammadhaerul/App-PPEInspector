# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Dashboard Deteksi APD K3",
    page_icon="🧑‍🚒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("PPE Detection Dashboard")

# Sidebar
st.sidebar.header("Model Configurations")

# Model Options
model_type = st.sidebar.radio(
    "Select Model", ['YOLOv8 Object Detection', 'YOLOv8 Instance Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'YOLOv8 Object Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'YOLOv8 Instance Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Source Configuration")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)


# Source Options
source_img = None
source_vid = None

if source_radio == settings.IMAGE:
    helper.detect_image(confidence, model)

elif source_radio == settings.VIDEO:
    helper.detect_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.detect_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.detect_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.detect_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
