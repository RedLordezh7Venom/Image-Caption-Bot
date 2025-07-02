from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_blip_caption(image_path):
    """
    Generates a caption for a given image using the BLIP model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The generated caption.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image and generate the caption
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    
    # Decode the generated caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption
