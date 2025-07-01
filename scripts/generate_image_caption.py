import tensorflow as tf
from tensorflow.keras.models import load_model
from scripts.utilities import greedy_generator,preprocess_image,cnn_output_dim,max_caption_length,inception_v3_model,tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tensorflow.keras.layers import Add # Import the Add layer

# Load the saved model, specifying custom objects
loaded_caption_model = load_model('models/caption_model.keras')

# Assume tokenizer, inception_v3_model, max_caption_length, and cnn_output_dim are already defined in the notebook
# If not, you would need to load/define them here.
# Example (replace with your actual loading/definition if needed):


def predict_caption(image_path):
    """
    Predicts a caption for a given image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated caption.
    """
    # Preprocess the image
    preprocess_image
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    # Extract image features using the InceptionV3 model
    image_features = inception_v3_model.predict(img, verbose=0)
    image_features = image_features.flatten()

    # Generate caption using the greedy search method (assuming greedy_generator is defined)
    # If you want to use beam search, call beam_search_generator instead.
    predicted_caption = greedy_generator(image_features)

    return predicted_caption

# Example usage:
if __name__ == "__main__":
    image_path_to_predict = 'examples\ElleVet_Peny_92-1024x717.jpg' # Replace with your image path
    generated_caption = predict_caption(image_path_to_predict)
    print("Predicted Caption:", generated_caption)


#predicted outputs: Predicted Caption:  a basketball player in a white uniform is playing a game ,Predicted Caption:  a brown dog is playing with a red ball in its mouth

