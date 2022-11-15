import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Chess Piece Identification")
st.text("Provide URL of Chess Piece Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('./Chess Piece Prediction')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']


def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return tf.image.resize(image,[224,224])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','https://as2.ftcdn.net/v2/jpg/02/29/00/27/1000_F_229002770_Vl4N0hLmkgXnh9foZgLB1B5hIUAzB6gX.jpg')

if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('Classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(np.array(classes))
      st.write("Accuracy Measure:")
      st.write(model.predict(decode_img(content)))
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Chess Pieces', use_column_width=True)
