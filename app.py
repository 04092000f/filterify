import streamlit as st
import numpy as np
import io
import base64
import cv2
from PIL import Image
from filters import *

def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">📥 {text}</a>'
    return href

st.set_page_config(page_title="🎨 Artistic Filters", layout="wide")
st.title("🖌️ Artistic Image Filters Playground")

# Sidebar
st.sidebar.header("🧪 Choose Filter")
option = st.sidebar.radio(
    "Pick a filter to apply:",
    (
        "None",
        "Black and White",
        "Sepia / Vintage",
        "Vignette Effect",
        "Pencil Sketch",
        "Cartoonify",
        "HDR Effect",
        "Color Invert",
        "Emboss"
    ),
)

# Upload section & processing logic
uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Immediately show results without needing scroll
    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("🖼️ Original")
        st.image(img, channels="BGR", use_column_width=True)

    output_flag = 1
    color = "BGR"

    # Apply filter
    if option == "None":
        output_flag = 0
    elif option == "Black and White":
        output = bw_filter(img)
        color = "GRAY"
    elif option == "Sepia / Vintage":
        output = sepia(img)
    elif option == "Vignette Effect":
        level = st.sidebar.slider("🔆 Vignette Level", 0, 5, 2)
        output = vignette(img, level)
    elif option == "Pencil Sketch":
        ksize = st.sidebar.slider("🌀 Blur Kernel Size", 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = "GRAY"
    elif option == "Cartoonify":
        output = cartoonify(img)
    elif option == "HDR Effect":
        output = hdr_effect(img)
    elif option == "Color Invert":
        output = invert_colors(img)
    elif option == "Emboss":
        output = emboss_effect(img)

    with output_col:
        if output_flag == 1:
            st.subheader("✨ Filtered Output")
            st.image(output, channels=color, use_column_width=True)

            if color == "BGR":
                result = Image.fromarray(output[:, :, ::-1])
            else:
                result = Image.fromarray(output)

            st.markdown(
                f"<div style='margin-top:20px;'>{get_image_download_link(result, 'output.jpg', 'Download Filtered Image')}</div>",
                unsafe_allow_html=True,
            )

else:
    # Only show uploader if no image is uploaded
    st.info("👈 Upload an image from the sidebar to begin editing.")
