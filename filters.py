import cv2
import numpy as np
import streamlit as st


@st.cache_data
def bw_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

@st.cache_data
def sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, kernel)
    sepia_img = np.clip(sepia_img, 0, 255)
    return sepia_img.astype(np.uint8)

@st.cache_data
def vignette(img, level=2):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/level)
    kernel_y = cv2.getGaussianKernel(rows, rows/level)
    kernel = kernel_y @ kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask
    return output

@st.cache_data
def pencil_sketch(img, ksize=5):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_img = 255 - gray_img
    blur_img = cv2.GaussianBlur(inv_img, (ksize, ksize), 0)
    sketch = cv2.divide(gray_img, 255 - blur_img, scale=256)
    return sketch

@st.cache_data
def cartoonify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

@st.cache_data
def hdr_effect(img):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

@st.cache_data
def invert_colors(img):
    return cv2.bitwise_not(img)

@st.cache_data
def emboss_effect(img):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    embossed = cv2.filter2D(img, -1, kernel) + 128
    return embossed