# prompt: create gradio app to load the model and run it 

import gradio as gr
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
import re

# Load the model
model = tf.keras.models.load_model('caption_model.h5')

# Load tokenizer (you'll need to adapt this to your actual tokenizer loading)
# Replace with your actual tokenizer loading
# Example using pickle
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1
max_caption_length = 34 # Replace with your actual max_caption_length
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
        prediction = model.predict([image_features.reshape(1,cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word[idx]
        in_text += ' ' + word
        if word == 'end':
            break
    in_text = in_text.replace('start ', '')
    in_text = in_text.replace(' end', '')
    return in_text


def predict(image):
    processed_image = preprocess_image(image)
    image_features = model.layers[2].predict(processed_image, verbose = 0) # assuming InceptionV3 is the second layer
    image_features = image_features.flatten()

    caption = greedy_generator(image_features)
    return caption

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Image Captioning",
    description="Upload an image and get a caption!"
)

iface.launch()
