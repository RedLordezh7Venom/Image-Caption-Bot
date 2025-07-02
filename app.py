import gradio as gr
from scripts.generate_image_caption import predict_caption
from scripts.blip_model import generate_blip_caption

def get_caption(image_path, model_choice):
    if model_choice == "BLIP":
        return generate_blip_caption(image_path)
    elif model_choice == "CNN_LSTM":
        greedy_caption, beam_caption = predict_caption(image_path)
        return f"Greedy Search: {greedy_caption}\nBeam Search: {beam_caption}"

# Create the Gradio interface
iface = gr.Interface(
    fn=get_caption,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Dropdown(
            ["BLIP", "CNN_LSTM"], label="Choose Model", value="BLIP"
        ),
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning with BLIP and CNN-LSTM",
    description="Upload an image and choose a model to generate a caption.",
    examples=[
        ["examples/fight.jpg"],
        ["examples/101669240_b2d3e7f17b.jpg"],
    ],
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
