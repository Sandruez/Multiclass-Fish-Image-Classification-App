import streamlit as st
import os
import uuid
from model_utils import load_model, preprocess_image, predict_fish

# Set title
st.title("🐟 Fish Classifier - ResNet50")
st.write("Upload an image of a fish and let the model predict its category.")

# Load the model
model = load_model("resnet50_fish.h5")

# Define class names in the same order as training
class_names = ['animal fish', 'animal fish bass',
               'fish sea_food black_sea_sprat',
               'fish sea_food gilt_head_bream',
               'fish sea_food hourse_mackerel',
               'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
               'fish sea_food sea_bass', 'fish sea_food shrimp',
               'fish sea_food striped_red_mullet',
               'fish sea_food trout']

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create uploads folder if not exists
    os.makedirs("uploads", exist_ok=True)
    
    # Save file with a unique name
    file_path = os.path.join("uploads", f"{uuid.uuid4()}.jpg")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show uploaded image
    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    # Preprocess & predict
    img_array = preprocess_image(file_path)
    predicted_class, confidence, all_confidences = predict_fish(model, img_array, class_names)

    # Display prediction
    st.subheader(f"Predicted Fish: {predicted_class}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Show confidence for all classes
    st.bar_chart({class_names[i]: float(all_confidences[i]) * 100 for i in range(len(class_names))})
