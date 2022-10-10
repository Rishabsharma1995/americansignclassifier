import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def sign_image_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    img = image.load_img(img, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    y_class = predictions[0:].argmax()
    return y_class 
