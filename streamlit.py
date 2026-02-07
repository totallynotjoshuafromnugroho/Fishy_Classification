import streamlit as st
from fastai.vision.all import *

st.title("Fish Species Identification")
st.text("Built by Joshua")

def extract_images(file_name):
    p = Path(file_name)
    species_name_parts = p.stem.split("_")

    final_species_name = " "
    length_parts = len(species_name_parts)-1
    for i in range(length_parts):
        final_species_name += species_name_parts[i]
        if i != length_parts:
            final_species_name += "_"

    return final_species_name

fish_model = load_learner("fish_species_prediction_model_fastai284.pkl")



def predict(image):
    real_img = PILImage.create(image)
    resized_img = real_img.resize((224, 224), Image.NEAREST)
    pred_class, pred_idx, outputs = fish_model.predict(resized_img)
    label = f"{pred_class}, {outputs[pred_idx]*100:.2f}%"
    return label

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", )
    st.subheader(f"Prediction Result: {predict(uploaded_file)}")

st.text("Built with Streamlit and Fastai")