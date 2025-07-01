import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import pickle
import argparse

# Load the InceptionV3 model for feature extraction
inception_v3_model = InceptionV3(weights='imagenet')
inception_v3_model = tf.keras.models.Model(inception_v3_model.input, inception_v3_model.layers[-2].output)


caption_model = tf.keras.models.load_model('models/caption_model.h5')

# Load the tokenizer
with open('models/preprocessing/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_caption_length = 34
cnn_output_dim = 2048

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def greedy_generator(image_features):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_caption_length).reshape((1,max_caption_length))
        prediction = caption_model.predict([image_features.reshape(1,cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word[idx]
        in_text += ' ' + word
        if word == 'end':
            break
    in_text = in_text.replace('start ', '')
    in_text = in_text.replace(' end', '')
    return in_text

def predict_caption(image_path):
    processed_image = preprocess_image(image_path)
    image_features = inception_v3_model.predict(processed_image, verbose=0)
    caption = greedy_generator(image_features)
    return caption

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a caption for an image.')
    parser.add_argument('image_path', type=str, help='The path to the image file.')
    args = parser.parse_args()

    caption = predict_caption(args.image_path)
    print("Generated Caption:", caption)
