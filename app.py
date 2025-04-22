import streamlit as st
import numpy as np
import io
import base64
import cv2
from PIL import Image
from filters import *

# Function to convert PIL image to byte format for download
def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im

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

uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])

# Display sample images for each filter if no image is uploaded
if not uploaded_file:
    st.info("👈 Upload an image or try one of the sample filters below.")
    
    # Show sample images for each filter
    st.subheader("🎨 Filter Samples")

    filter_images = {
        "Black and White": "sample1.jpg",
        "Sepia / Vintage": "sample2.jpg",
        "Vignette Effect": "sample3.jpg",
        "Pencil Sketch": "sample4.jpg",
        "Cartoonify": "sample5.jpg",
        "HDR Effect": "sample6.jpg",
        "Color Invert": "sample7.jpg",
        "Emboss": "sample8.jpg"
    }

    cols = st.columns(4)  # Divide the screen into 4 columns for better alignment

    # Loop through each filter and display its sample
    for i, (filter_name, image_path) in enumerate(filter_images.items()):
        with cols[i % 4]:  # Distribute evenly across columns
            st.image(image_path, caption=filter_name, use_container_width=True)

# Processing the uploaded file
if uploaded_file:
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("🖼️ Original")
        st.image(img, channels="BGR", use_container_width=True)

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
        level = st.sidebar.slider("🔆 Vignette Level", 1, 5, 2)  # Avoid zero
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
            st.image(output, channels=color, use_container_width=True)

            if color == "BGR":
                result = Image.fromarray(output[:, :, ::-1])
            else:
                result = Image.fromarray(output)

            st.download_button(
                label="📥 Download Filtered Image",
                data=pil_to_bytes(result),
                file_name="filtered_output.jpg",
                mime="image/jpeg",
            )
