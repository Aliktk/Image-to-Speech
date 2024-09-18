# **Image-to-Speech Application: AI-Powered Image Captioning and Storytelling**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ali-nawaz-khattak/)

## Overview

The **Image-to-Speech Application** transforms images into spoken stories through an AI-driven pipeline. Using advanced models from Hugging Face, OpenAI, and Gradio, this application takes an image as input, generates a caption, creates a short narrative, and then converts the story to speech. The application provides a user-friendly interface to bridge the gap between image processing, natural language generation, and text-to-speech synthesis.

---

## 📝 Table of Contents

- [**Image-to-Speech Application: AI-Powered Image Captioning and Storytelling**](#image-to-speech-application-ai-powered-image-captioning-and-storytelling)
  - [Overview](#overview)
  - [📝 Table of Contents](#-table-of-contents)
  - [📁 Project Structure](#-project-structure)
  - [✨ Features](#-features)
  - [💻 Installation](#-installation)
  - [🚀 Usage](#-usage)
  - [⚙️ Configuration](#️-configuration)
  - [📊 Models \& Data](#-models--data)
  - [🔬 Src](#-src)
  - [🙏 Acknowledgements](#-acknowledgements)

---

## 📁 Project Structure

```bash
Image-to-Speech/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
└── src/
    ├── __init__.py
    ├── image_captioning.py
    └── story_creator.py
    └── text2speech.py
└── tempdir_tts
```

## ✨ Features
**Image Captioning:** Extracts meaningful captions from images using the 'Salesforce/blip-image-captioning-large model'.
Story Generation: Converts the caption into a creative short story using 'OpenAI's GPT-4' model.

**Text-to-Speech:** Generates speech from the story using Hugging Face's 'microsoft/speecht5_tts' model with custom speaker embeddings.

**Interactive UI:** Intuitive user interface built with **Gradio** for easy image upload and result presentation.
Real-time Audio Output: Automatically generates and plays the audio for the generated story.


## 💻 Installation
To set up the project locally, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/Image-to-Speech.git
cd Image-to-Speech
```

**2. Install the required dependencies:**

```bash
pip install -r requirements.txt
```

**2. Ensure you have your OpenAI API key set up as an environment variable::**

```bash
export OPENAI_API_KEY='your-api-key' #or
api_key = os.getenv("OPENAI_API_KEY")
```

## 🚀 Usage
To run the Image-to-Speech application, execute the following command:

```bash
python image2speech_app.py
```


<p>Once launched, you can upload an image, generate a caption, create a story, and listen to the audio output.</p>

## ⚙️ Configuration
<p>You can modify the behavior of the application by updating the src/utils.py file to change how captions or stories are generated.
The TTS configuration and speaker embeddings are managed in the src/tts_model.py file.</p>

## 📊 Models & Data
The following pre-trained models and datasets are utilized:

1. **Image Captioning:** Salesforce/blip-image-captioning-large
2. **Text-to-Speech:** microsoft/speecht5_tts with speaker embeddings from Matthijs/cmu-arctic-xvectors
3. **Story Generation:** GPT-4 from OpenAI

You can provide custom image samples by placing them in the experiments/image_samples folder for testing and experimentation.

## 🔬 Src
The experiments and seprate files is in Src folder :

A python files for testing and tweaking the application.
Image samples for validating the model’s captioning, storytelling, and speech synthesis.

## 🙏 Acknowledgements
Gratitude to the Hugging Face, OpenAI, and Gradio teams for their amazing APIs and models. Special thanks to the community for providing invaluable resources and support.