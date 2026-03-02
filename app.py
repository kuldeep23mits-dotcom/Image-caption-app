import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# Load model
model = load_model("image_caption_model")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Load feature extractor
base_model = InceptionV3(weights='imagenet')
feature_extractor = Model(base_model.input, base_model.layers[-2].output)

max_length = 34  # use your actual max_length

def extract_features(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    feature = feature_extractor.predict(image)
    return feature

def generate_caption(photo):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text.replace("startseq","").replace("endseq","").strip()

st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    feature = extract_features(image)
    caption = generate_caption(feature)
    st.success(caption)
