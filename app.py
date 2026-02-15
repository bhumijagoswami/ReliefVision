import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

st.set_page_config(page_title="ReliefVision", layout="centered")

st.title("ReliefVision")
st.subheader("Satellite-Based Disaster Damage Assessment System")

before = st.file_uploader("Upload Pre-Disaster Image", type=["jpg", "png"])
after = st.file_uploader("Upload Post-Disaster Image", type=["jpg", "png"])
sensitivity = st.slider("Damage Sensitivity", 0, 100, 50)


if before and after:

    img1 = np.array(Image.open(before).convert("RGB"))
    img2 = np.array(Image.open(after).convert("RGB"))

    img1 = cv2.resize(img1, (600, 600))
    img2 = cv2.resize(img2, (600, 600))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Remove bright cloud regions
    _, cloud_mask1 = cv2.threshold(gray1, 220, 255, cv2.THRESH_BINARY)
    _, cloud_mask2 = cv2.threshold(gray2, 220, 255, cv2.THRESH_BINARY)

    gray1[cloud_mask1 == 255] = 0
    gray2[cloud_mask2 == 255] = 0


    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    threshold_value = 255 - (sensitivity * 2)
_, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = img2.copy()
    total_area = 600 * 600
    changed_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            changed_area += area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y),
                          (x+w, y+h), (255, 0, 0), 2)

    percent = (changed_area / total_area) * 100

    st.image(output, caption="Detected Damage Regions")
    st.success(f"Structural Similarity Score: {score:.4f}")
    st.warning(f"Estimated Damage Area: {percent:.2f}%")


