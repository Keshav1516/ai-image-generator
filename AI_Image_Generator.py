!pip install --upgrade pip
!pip install diffusers transformers accelerate safetensors

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="AI Image Generator", layout="wide")

# Title
st.title("🎨 AI Image Generator")
st.write("Generate stunning images from text prompts using Stable Diffusion.")

# Sidebar inputs
st.sidebar.header("⚙️ Settings")
prompt = st.text_area("Enter your prompt:", "A futuristic city skyline at sunset")
num_images = st.sidebar.slider("Number of images", 1, 4, 1)
image_size = st.sidebar.selectbox("Image size", ["512x512", "768x768", "1024x1024"])
guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.5)
steps = st.sidebar.slider("Inference steps", 10, 100, 50)

# Load model once
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Generate button
if st.button("🚀 Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating images... ⏳"):
            images = []
            for _ in range(num_images):
                img = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]
                images.append(img)

        # Show images
        cols = st.columns(num_images)
        for i, img in enumerate(images):
            with cols[i]:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name=f"generated_image_{i+1}.png",
                    mime="image/png"
                )
