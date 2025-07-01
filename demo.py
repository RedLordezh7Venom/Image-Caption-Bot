import gradio as gr
from scripts.generate_image_caption import predict_caption

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=[
        gr.Textbox(label="Greedy Search Caption"),
        gr.Textbox(label="Beam Search Caption"),
    ],
    title="Image Captioning with Greedy and Beam Search",
    description="Upload an image to generate two different captions using Greedy Search and Beam Search.",
    examples=[["examples/fight.jpg"],["examples/101669240_b2d3e7f17b.jpg"]],
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
