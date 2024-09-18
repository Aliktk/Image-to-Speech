import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from datasets import load_dataset
import torch
import soundfile as sf
import openai
from PIL import Image
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import OpenAI
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Initialize models and API keys
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# Function to process the image and generate a caption
def generate_caption(image):
    try:
        print("[INFO]: Processing image for caption...")
        text = "a photography of"
        inputs = processor(image, text, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"[INFO]: Generated caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error in generating caption: {e}")
        return "Error generating caption."


# Create a Prompt Template and Chain in LangChain for generating the story
def generate_story(caption):
    try:
        print("[INFO]: Generating story from caption...")
        prompt = f"""
        You are a story teller; 
        You can generate a very short story based on a simple narrative, the story should two sentances with simple vocabulary and don't mentioned any name of the image related;
        CONTEXT: {caption}
        STORY:
        """
        print("[INFO]: Crafting prompt for OpenAI...")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        print("[INFO]: OpenAI response received.")
        
        story = response.choices[0].message.content
        print(f"Generated Story: {story}")
        return story
    except Exception as e:
        print(f"Error in generating story: {e}")
        return "Error generating story."
# Function to convert story to speech
def story_to_speech(story):
    try:
        print("[INFO]: Converting story to speech...")
        speech = synthesiser(story, forward_params={"speaker_embeddings": speaker_embedding})
        sf.write("temp_dir_tts/generated_speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
        print("[INFO]: Speech file created: speech.wav")
        print("[INFO]: Returning speech file path.")
        return "temp_dir_tts/generated_speech.wav"
    except Exception as e:
        print(f"Error in converting story to speech: {e}")
        return None

# Function to handle submit button
def on_submit(image):
    caption = generate_caption(image)
    story = generate_story(caption)
    audio_file = story_to_speech(story)
    
    return caption, story, audio_file

# Function to clear the outputs
def on_clear():
    return "", "", None

# Custom CSS for better styling
custom_css = """
    body {
        background-color: #f4f4f9;
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .gradio-container {
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    
    .gr-button {
        background-color: #4CAF50;  /* Default button color */
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
    }
    .gr-button:hover {
        background-color: orange;  /* Color on hover */
    }
    .gr-button {
        background-color: orange;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
    }
    .gr-textbox, .gr-audio {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .gr-column {
        padding: 10px;
    }
    h2 {
        font-size: 24px;
        color: #333;
        text-align: center;
        margin-bottom: 10px;
    }
    h3 {
        font-size: 20px;
        color: #555;
        margin-bottom: 10px;
    }
"""

# Gradio Interface with visual improvements
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h2 style='color:white;'>🎨 Image-to-Speech Application 🎨</h2>")


    with gr.Row():
        # Left column for image upload and buttons
        with gr.Column():
            gr.Markdown("<h3 style='color:white;'>🖼️ Upload Image</h3>")
            
            # Image upload component
            image_input = gr.Image(label="Upload an image", type="pil")
            
            # Buttons (Clear and Submit) in a row
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
        
        # Right column for caption, story, and audio outputs
        with gr.Column():
            gr.Markdown("<h3 style='color:white;'>📜 Generated Results</h3>")
            
            # Output components for caption, story, and audio file
            caption_output = gr.Textbox(label="Text Caption", lines=2, placeholder="Caption will appear here...", interactive=False)
            story_output = gr.Textbox(label="Story", lines=5, placeholder="Story will appear here...", interactive=False)
            audio_output = gr.Audio(label="Generated Audio", type="filepath", interactive=False)

    # Button click handlers
    submit_button.click(on_submit, inputs=image_input, outputs=[caption_output, story_output, audio_output])
    clear_button.click(on_clear, outputs=[caption_output, story_output, audio_output])
    
    # Launch the app
    demo.launch()
