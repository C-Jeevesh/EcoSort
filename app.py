import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog

# --- PAGE SETUP ---
st.set_page_config(page_title="EcoSort Local", page_icon="♻️")
st.title("♻️ EcoSort: Smart Waste Sorter")
st.markdown("**Powered by SVM & HOG (Classical Machine Learning)**")

# --- LOAD MODEL ---
# Since running locally, we just look for the file in the same folder
MODEL_PATH = 'ecosort_svm_model.pkl'

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error(f"⚠️ Model not found! Please ensure '{MODEL_PATH}' is in this folder.")
    st.stop()

# --- PREPROCESSING ---
def process_image(img_array):
    IMG_HEIGHT = 128
    IMG_WIDTH = 64
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features.reshape(1, -1)

# --- INPUT METHOD ---
input_method = st.radio("Select Input:", ["📸 Camera", "📂 Upload File"])

image_input = None
if input_method == "📸 Camera":
    image_input = st.camera_input("Take a photo")
else:
    image_input = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image_input is not None:
    # Convert input to array
    file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.image(opencv_image, caption="Input Image", width=300, channels="BGR")
    
    if st.button("Classify Waste"):
        with st.spinner('Analyzing...'):
            features = process_image(opencv_image)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features).max()
            
            if prediction == 'R':
                st.success(f"### ♻️ RECYCLABLE (Confidence: {probability*100:.1f}%)")
                st.info("Instructions: Rinse and place in Blue Bin.")
            else:
                st.warning(f"### 🍎 ORGANIC / TRASH (Confidence: {probability*100:.1f}%)")
                st.info("Instructions: Compost or Green Bin.")