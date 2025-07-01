from scripts.sequence_generators import beam_search_generator,greedy_generator
def predict_image_caption(image_path):
    # Step 2: Preprocess the image
    processed_image = preprocess_image(image_path)

    # Step 3: Extract image features using the InceptionV3 model
    image_features = inception_v3_model.predict(processed_image, verbose=0)
    image_features = image_features.flatten()

    # Step 4: Generate captions using both methods
    greedy_caption = greedy_generator(image_features)
    beam_search_caption = beam_search_generator(image_features)

    # Print captions
    print("Greedy Caption:", greedy_caption)
    print("Beam Search Caption:", beam_search_caption)

    # Return captions
    return greedy_caption, beam_search_caption

if __name__ == "__main__":
    predict_image_caption('examples/fight.jpg')
