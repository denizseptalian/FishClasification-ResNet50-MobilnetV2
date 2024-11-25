import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained models
@st.cache(allow_output_mutation=True)
def load_resnet50_model():
    return load_model("ResNet50_model.keras")

@st.cache(allow_output_mutation=True)
def load_mobilenet_model():
    return load_model("MobileNet_model.keras")

# Function to preprocess image
def preprocess_image(uploaded_image, target_size):
    img = uploaded_image.resize(target_size)  # Resize image to target size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Map of class indices to labels
class_labels = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

# Streamlit app interface
def main():
    st.title("Fish Classification Deeplearning A - Kelompok 4")
    st.write("Upload Gambar Untuk Uji Coba Model")
    
    # Add dataset download link
    st.markdown(
        "[📥 Download Dataset Uji Coba](https://drive.google.com/drive/folders/1YLJyr2jc7BGZlfAzYM0yegT4SCe7UB02?usp=sharing)",
        unsafe_allow_html=True
    )

    # Upload image section
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Load models
        resnet_model = load_resnet50_model()
        mobilenet_model = load_mobilenet_model()

        # Preprocess image for both models
        resnet_input = preprocess_image(uploaded_image, target_size=(256, 256))
        mobilenet_input = preprocess_image(uploaded_image, target_size=(256, 256))

        # Detection Buttons
        if st.button("Detect with ResNet50"):
            with st.spinner("Classifying with ResNet50..."):
                predictions = resnet_model.predict(resnet_input)
                predicted_class = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)

            predicted_label = class_labels[predicted_class[0]]
            st.success(f"[ResNet50] Predicted Class: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}")

        if st.button("Detect with MobileNet50"):
            with st.spinner("Classifying with MobileNet50..."):
                predictions = mobilenet_model.predict(mobilenet_input)
                predicted_class = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)

            predicted_label = class_labels[predicted_class[0]]
            st.success(f"[MobileNet50] Predicted Class: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}")

        # Compare Results
        if st.button("Bandingkan 2 Model"):
            with st.spinner("Comparing ResNet50 and MobileNet50..."):
                # ResNet50 Prediction
                resnet_predictions = resnet_model.predict(resnet_input)
                resnet_predicted_class = np.argmax(resnet_predictions, axis=-1)
                resnet_confidence = np.max(resnet_predictions)

                # MobileNet50 Prediction
                mobilenet_predictions = mobilenet_model.predict(mobilenet_input)
                mobilenet_predicted_class = np.argmax(mobilenet_predictions, axis=-1)
                mobilenet_confidence = np.max(mobilenet_predictions)

                # Display Results
                st.write("### Comparison Results")
                st.write(f"[ResNet50] Predicted Class: {class_labels[resnet_predicted_class[0]]}, Confidence: {resnet_confidence:.2f}")
                st.write(f"[MobileNet50] Predicted Class: {class_labels[mobilenet_predicted_class[0]]}, Confidence: {mobilenet_confidence:.2f}")

                # Bar Chart
                st.write("### Confidence Comparison")
                fig, ax = plt.subplots()
                models = ["ResNet50", "MobileNet50"]
                confidences = [resnet_confidence, mobilenet_confidence]
                ax.bar(models, confidences)
                ax.set_ylim([0, 1])  # Confidence is between 0 and 1
                ax.set_ylabel("Confidence")
                ax.set_title("Confidence Comparison between Models")
                st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
