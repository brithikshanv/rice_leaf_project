import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_mobilenetv2.keras")

model = load_model()

# Class labels (must match training order)
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut"
]

# -------------------------------
# Sidebar Content
# -------------------------------
st.sidebar.title("Project Overview")

st.sidebar.markdown("### Project Intent")
st.sidebar.markdown("""
This project demonstrates the use of **deep learning** techniques
to automatically identify common rice leaf diseases from images.

The objective is to assist in early disease detection by
classifying rice leaves into three major disease categories:
- Leaf Smut 
- Brown Spot
- Bacterial Leaf Blight

""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Model Details")
st.sidebar.markdown("""
- Architecture: MobileNetV2  
- Technique: Transfer Learning  
- Input Size: 224 × 224 RGB images  
- Output: 3 disease classes  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### Further Improvements")
st.sidebar.markdown("""
- Train on a larger and more diverse dataset.
- Include healthy leaf classification.
- Add real-time image capture support.
- Provide disease treatment recommendations.
- Deploy the application on cloud platforms.
""")

# -------------------------------
# Main Page Content
# -------------------------------
st.markdown(
    "<h1 style='text-align: center;'>🌾 Rice Leaf Disease Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Upload a rice leaf image to identify the disease</p>",
    unsafe_allow_html=True
)

st.info(
    "This application uses a trained deep learning model to classify "
    "rice leaf diseases based on visual symptoms."
)


# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image of a rice leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    if st.button(" Predict Disease"):
        with st.spinner("Analyzing image..."):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = float(np.max(predictions))

        st.markdown("---")

        # Results
        st.subheader(" Prediction Result")
        st.success(f"**Predicted Disease:** {class_names[predicted_class]}")

        st.write("**Prediction Confidence:**")
        st.progress(confidence)
        st.write(f"{confidence * 100:.2f}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool is intended for educational and research purposes only. "
    "For real-world agricultural decisions, consult crop protection experts."
)
