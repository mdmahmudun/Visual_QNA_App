import gradio as gr
from transformers.utils import logging
from transformers import BlipForQuestionAnswering, AutoProcessor
from PIL import Image
logging.set_verbosity_error()

# Load the model and processor
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

# Define a function to process inputs and generate outputs
def predict(image, question):
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

# Create the Gradio interface with custom Markdown and HTML formatting
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Question", placeholder="Ask a question about the image")
    ],
    outputs=gr.Textbox(label="Answer"),
    description="<h1 style='text-align: center; font-family: Times New Roman;'>Visual Question Answering</h1> \
                 <p style='text-align: center; font-family: Times New Roman;'><strong>Model name:</strong> Salesforce/blip-vqa-base</p> \
                 <p style='text-align: center; font-family: Times New Roman;'><strong>Made by:</strong> MD MAHMUDUN NABI</p>"
)

# Launch the Gradio interface
demo.launch()
