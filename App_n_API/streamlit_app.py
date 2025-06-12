#change to file directory and then : streamlit run streamlit_app.py

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

API_ENDPOINT = 'http://localhost:8000/predict/'

st.title("Image Segmentation Demo")
st.subheader("Select an image to see the ground truth and predicted mask")

# --- Load Test Images and Masks ---
TEST_DATA_DIR = 'test_data'
IMAGE_DIR = os.path.join(TEST_DATA_DIR, 'images')
MASK_DIR = os.path.join(TEST_DATA_DIR, 'gen_masks')

available_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
available_masks = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_common_id(filename, is_mask=False):
    """Extracts the common identifier from the filename."""
    base, ext = os.path.splitext(filename)
    parts = base.split('_')
    final = '_'.join(parts[:-1])
    if is_mask:
        final = '_'.join(parts[:-2])
    return final 

# Create dictionaries with the common ID as the key
image_names = {get_common_id(img): os.path.join(IMAGE_DIR, img) for img in available_images}
mask_names = {get_common_id(mask, is_mask=True): os.path.join(MASK_DIR, mask) for mask in available_masks}

common_image_ids = set(image_names.keys()) & set(mask_names.keys())
test_samples = sorted(list(common_image_ids)) # Sort for consistent order

print(f"Test Samples: {test_samples}")

# --- Color Map ---
colors = np.array([[128, 0, 128],   # Violet #"void"
                   [255, 0, 255],   # Magenta #"flat"
                   [0, 0, 255],     # Bleu #"construction"
                   [255, 255, 0],   # Jaune #"object"
                   [0, 255, 0],     # Vert #"nature"
                   [0, 255, 255],   # Cyan #"sky"
                   [255, 0, 0],     # Rouge #"human"
                   [255, 165, 0]],  # Orange #"vehicle"
                  dtype=np.uint8)

def load_image_from_bytes(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def display_pil_image(image_pil, caption):
    st.image(image_pil, caption=caption, use_container_width=True)

if not test_samples:
    st.error(f"No matching images and masks found in '{IMAGE_DIR}' and '{MASK_DIR}'.")
else:
    selected_image_id = st.selectbox("Choose a test image:", test_samples)
    selected_image_path = image_names[selected_image_id]
    selected_mask_path = mask_names[selected_image_id]

    # --- Load and Display Ground Truth ---
    ground_truth_image = cv2.imread(selected_image_path)
    ground_truth_image = cv2.resize(ground_truth_image, (512, 512))
    ground_truth_mask = cv2.imread(selected_mask_path, cv2.IMREAD_GRAYSCALE) # Assuming grayscale indices
    ground_truth_mask = cv2.resize(ground_truth_mask, (512, 512))

    if ground_truth_image is not None and ground_truth_mask is not None:
        st.subheader("Selected Image:")
        st.image(cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.subheader("Ground Truth Mask:")
        ground_truth_mask_color = colors[ground_truth_mask] # Colorize the Ground Truth Mask using the color map
        st.image(ground_truth_mask_color, use_container_width=True)

    else:
        st.error("Could not load selected image or mask.")

    # --- Prediction ---
    if st.button("Predict with API"):
        try:
            with open(selected_image_path, 'rb') as f:
                files = {'image_file': (os.path.basename(selected_image_path), f, 'image/jpeg')} # Adjust type if needed
                response = requests.post(API_ENDPOINT, files=files)
                response.raise_for_status()
                data = response.json()

                if 'mask_indices' in data:
                    mask_indices = np.array(data['mask_indices'], dtype=np.uint8)

                    predicted_mask_resized = mask_indices.reshape((512, 512))
                    predicted_mask_color = colors[predicted_mask_resized]

                    st.subheader("Predicted Mask:")
                    st.image(predicted_mask_color, use_container_width=True)
                else:
                    st.error(f"Error: No 'mask_indices' found in the API response: {data}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")