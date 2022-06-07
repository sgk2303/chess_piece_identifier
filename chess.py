import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Image Classifier")
st.text("Provide URL of flower Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('./Chess Piece Prediction')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']


def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0 # external data norm while training
  # in our case norm is part of sequential model

  return tf.image.resize(image,[224,224])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg')
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(np.array(classes))
      st.write(model.predict(decode_img(content)))
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Bean Image', use_column_width=True)    
