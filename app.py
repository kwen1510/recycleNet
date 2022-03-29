# Main libraries
import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np


### Define functions ###

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction # return position of the highest probability
#     return np.argmax(prediction) # return position of the highest probability


### Streamlit code ###

st.title("RecycleNet")

st.write("Welcome to RecycleNet. Upload a photo to see what material it is.")


uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	
	with st.spinner('Analysing...'):
		
		label = teachable_machine_classification(image, 'keras_model.h5')
		
		st.write(label)

		if label == 0:
		  material = "cardboard"
		elif label == 1:
		  material = "glass"
		elif label == 2:
		  material = "metal"
		elif label == 3:
		  material = "paper"
		elif label == 4:
		  material = "plastic"
		elif label == 5:
		  material = "trash"
		else:
		  material = "No idea"

		st.header(f"Material: {material}")
		st.image(image, use_column_width=True)
