import io
import subprocess
import time

import av
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


# Load YOLOv5n
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="yolov5n.pt", force_reload=True)


model = load_model()

st.set_page_config(page_title="Deteksi Objek Raspi", layout="wide")
st.title("🎯 Deteksi Objek Realtime dengan YOLOv5n - Raspberry Pi")


# Fungsi Deteksi Gambar dari PIL
def detect_image_pil(pil_img):
    results = model(pil_img)
    return Image.fromarray(np.squeeze(results.render()))


# WebRTC Video Processor
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(img[:, :, ::-1])  # BGR to RGB
        results = model(pil_img)
        result_img = np.squeeze(results.render())
        return result_img[:, :, ::-1]  # Convert back to BGR


# UI Mode Pilihan
mode = st.radio("Pilih Mode:", ["📷 Upload Gambar dari HP", "📡 Live RaspiCam", "📱 Kamera HP Live Langsung (Browser)"])

# 📷 MODE 1: Upload Gambar
if mode == "📷 Upload Gambar dari HP":
    uploaded = st.file_uploader("Upload gambar dari HP:", type=["jpg", "jpeg", "png"])
    if uploaded:
        col1, col2 = st.columns(2)
        img = Image.open(uploaded)
        col1.image(img, caption="📤 Gambar Asli", use_container_width=True)
        result_img = detect_image_pil(img)
        col2.image(result_img, caption="✅ Hasil Deteksi", use_container_width=True)

# 📡 MODE 2: RaspiCam Realtime
elif mode == "📡 Live RaspiCam":
    st.warning("⏹ Klik tombol STOP untuk menghentikan stream kamera.")
    show_area = st.empty()
    stop_btn = st.button("🛑 STOP Stream")

    while True:
        if stop_btn:
            st.success("✅ Stream kamera dihentikan.")
            break

        try:
            # Ambil frame dari kamera
            proc = subprocess.run(
                ["libcamera-jpeg", "-n", "-t", "300", "--width", "320", "--height", "240", "-o", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )

            if proc.returncode != 0 or not proc.stdout:
                show_area.error("❌ Gagal membaca frame dari RaspiCam.")
                break

            # Buka & flip horizontal
            # img = Image.open(io.BytesIO(proc.stdout)).transpose(Image.FLIP_LEFT_RIGHT)
            # img = Image.open(io.BytesIO(proc.stdout)).transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            img = Image.open(io.BytesIO(proc.stdout)).transpose(Image.FLIP_TOP_BOTTOM)

            # Deteksi objek
            hasil = detect_image_pil(img)

            # Tampilkan gambar dengan lebar tetap (480px)
            show_area.image(hasil, caption="📡 Deteksi RaspiCam", width=480)

        except subprocess.TimeoutExpired:
            show_area.error("❌ Timeout kamera.")
            break
        except Exception as e:
            show_area.error(f"❌ Error: {e}")
            break

        time.sleep(0.1)

# 📱 MODE 3: Kamera HP Langsung (via browser)
elif mode == "📱 Kamera HP Live Langsung (Browser)":
    st.markdown("""
    📱 Kamera akan menyala jika kamu membuka halaman ini dari **HP** dan mengizinkan akses kamera.
    """)
    webrtc_streamer(
        key="yolo-live",
        video_processor_factory=YOLOTransformer,
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        async_processing=True,
    )
