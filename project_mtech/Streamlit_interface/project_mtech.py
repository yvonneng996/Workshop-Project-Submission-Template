# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:01:43 2026

@author: yvonneng
"""

import streamlit as st
import cv2
import numpy as np
import import_ipynb 
import preprocessing
import inference
import stitching
import metrics_calculation
import classification_llm_API
import torch
import os
st.set_page_config(page_title="Conjunctiva AI Assistant", layout="wide")

st.markdown("""
    <h1 style='text-align: center; font-size:42px; margin-bottom:5px;'>
        Conjunctiva Vessel Analysis AI Assistant
    </h1>
    <h3 style='text-align: center; color:#0d6efd; margin-top: -20px;'>
        Upload your image below
    </h3>
    """, unsafe_allow_html=True)
import streamlit as st

import streamlit as st
from PIL import Image

# ---- Initialize session states ----
if "image1_uploaded" not in st.session_state:
    st.session_state.image1_uploaded = False

if "image2_uploaded" not in st.session_state:
    st.session_state.image2_uploaded = False

if "image1" not in st.session_state:
    st.session_state.image1 = None

if "image2" not in st.session_state:
    st.session_state.image2 = None
    
#model = "BestModel_iter136080_iou0.4913.pth"
# ---- Buttons ----
if st.button("Upload Image 1"):
    st.session_state.image1_uploaded = True

if st.button("Upload Image 2"):
    st.session_state.image2_uploaded = True


# ---- Upload Image 1 ----
if st.session_state.image1_uploaded:
    st.session_state.image1 = st.file_uploader(
        "Choose Image 1",
        type=["png", "jpg", "jpeg"],
        key="image_uploader_1"
    )

if st.session_state.image1 is not None:
    st.image(st.session_state.image1, caption="Uploaded Image 1")
    st.success("Image 1 uploaded successfully!")


# ---- Upload Image 2 ----
if st.session_state.image2_uploaded:
    st.session_state.image2 = st.file_uploader(
        "Choose Image 2",
        type=["png", "jpg", "jpeg"],
        key="image_uploader_2"
    )

if st.session_state.image2 is not None:
    st.image(st.session_state.image2, caption="Uploaded Image 2")
    st.success("Image 2 uploaded successfully!")

#Fmodelsave_dir = 'D:\\newly vessel train\\test newly preprocess\\prediction train with only mina\\try\\save binary\\'

# ---- Process Image ----
if st.session_state.image1 is not None and st.session_state.image2 is not None:
    if st.button("Segmentation Process"):
        with st.spinner("Processing..."):
        # Example of processing images (Convert to grayscale here)
            st.session_state.image1.seek(0)
            st.session_state.image2.seek(0)

            file_bytes1 = np.asarray(
                bytearray(st.session_state.image1.read()),
                dtype=np.uint8
            )
            file_bytes2 = np.asarray(
                bytearray(st.session_state.image2.read()),
                dtype=np.uint8
            )

            img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            result = preprocessing.process_images(img1, img2)
           
            infer = [inference.run_unet_inference(patch, img_size=224, device="cpu")
                    for patch in result]
            num_images = len(infer)
            st.write(f"Generated {num_images} predicted patches")
            ids = [f"Patch_{i}.png" for i in range(num_images)]
            stitched_image = stitching.stitch_patches(infer, ids)
            st.image(stitched_image, caption="Final Stitched Vessel Map", use_column_width=True)
            vessel_density = metrics_calculation.calculate_vessel_density(img2, stitched_image)
            st.success("Segmentation and stitching completed!")
            st.write("Vessel Density", round(vessel_density, 4))
            
            output = classification_llm_API.classify_with_claude(vessel_density,img1)

            st.subheader("Prediction with LLM explanation based on Conjunctiva image and vesseldensity")
            st.write(output)
      