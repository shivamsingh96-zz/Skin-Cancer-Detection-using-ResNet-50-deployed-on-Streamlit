import streamlit as st
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_classfication_model():
  model = load_model('Skin_Cancer_Classification_with ResNet.hdf5')
  return model

model = load_classfication_model()

st.write("""
          # Image Classification
          """)

file = st.file_uploader("Please upload an IMAGE",type = ["jpeg","jpg","png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data,model):

  size = (224, 224)
  image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  #img = img_to_array(image)
  #img = img.reshape(1, 32, 32, 3)

  #img = img.astype('float32')
  #img = img / 255.0

  img = np.asarray(image)
  #img_reshape = img.reshape(1 , 32 ,32 ,3)
  img_reshape = img[np.newaxis,...]
  prediction = model.predict(img_reshape)

  return prediction

if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_and_predict(image,model)
  class_names = ["Benign", "Malignant"]
  string = "This particular image most likely is :"+class_names[np.argmax(predictions)]
  st.success(string)