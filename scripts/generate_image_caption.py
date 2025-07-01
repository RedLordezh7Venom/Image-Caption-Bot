from tensorflow.keras.models import load_model
from scripts.utilities import (
    greedy_generator,
    beam_search_generator,
    extract_image_features,
    inception_v3_model,
)
# Load the saved model, specifying custom objects
loaded_caption_model = load_model('models/caption_model.keras')


def predict_caption(image_path):
    """
    Predicts a caption for a given image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated caption.
    """
    # Preprocess the image
    image_features = extract_image_features(inception_v3_model,image_path)
    # Generate caption using the greedy search method (assuming greedy_generator is defined)
    # If you want to use beam search, call beam_search_generator instead.
    greedy_caption = greedy_generator(image_features)
    beam_caption = beam_search_generator(image_features)


    return greedy_caption,beam_caption

# Example usage:
if __name__ == "__main__":
    image_path_to_predict = 'examples\ElleVet_Peny_92-1024x717.jpg' # Replace with your image path
    generated_caption = predict_caption(image_path_to_predict)
    print("Predicted Caption:", generated_caption)


#predicted outputs: Predicted Caption:  a basketball player in a white uniform is playing a game ,Predicted Caption:  a brown dog is playing with a red ball in its mouth

