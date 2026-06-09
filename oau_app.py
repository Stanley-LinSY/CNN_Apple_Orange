import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model("C:/Python/2120795_StanleyLin_Orange_2112422_WaiYanAung_Apple/oau_model_400_e10.hdf5") #load a trained model
st.write("""
        # Orange-Apple-Unknown Prediction
        """)
st.write("This is a simple image classification web app to predict Orange-Apple-Unknown")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image (image, use_column_width=True)
    prediction = import_and_predict(image, model)
    if np.argmax(prediction) == 0:
        st.write("It is an Apple!")
    elif np.argmax (prediction) == 1:
        st.write("It is an Orange!")
    else:
        st.write("It is unknown!")
        
    st.text("Probability (0: Apple, 1: Orange, 2: Unknown)")
    st.write(prediction)
