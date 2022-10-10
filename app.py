import streamlit as st
st.title("Classify American Sign Language Digits")
st.header("Classify American Sign Language Digits")
st.text("Upload a digit sign image for classification between 0 to 9")
from img_classification import sign_image_classification
from PIL import Image
import io
uploaded_file = st.file_uploader("Choose a sign image ...")
if uploaded_file is not None:
    buffer = io.BytesIO()     # create file in memory
    image = Image.open(uploaded_file)
    image.save(buffer, 'jpeg') # save in file in memory - it has to be `jpeg`, not `jpg`
    buffer.seek(0)            # move to the beginning of file
    bg_image = buffer 
    st.image(bg_image, caption='Uploaded sign image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = sign_image_classification(bg_image, 'CNN_Model_ResNet50.h5')
    st.write("This sign image is of digit {0}".format(label))
    # success
    st.success("Success")
