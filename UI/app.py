import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import os
from datetime import datetime

SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define the video processor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Title
st.title("Customer KYC")

# Webcam streamer
ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Capture image on button click
if ctx.video_processor and st.button("📷 Capture Image"):
    frame = ctx.video_processor.frame
    if frame is not None:
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        st.success(f"Image saved at {filepath}")
        st.image(frame, channels="BGR", caption="Captured Image")
    else:
        st.warning("No frame available yet. Please try again.")
