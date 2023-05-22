import argparse
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Stable Diffusion Pipeline')
parser.add_argument('image_path', type=str, help='Path to the input image')
args = parser.parse_args()

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Use the image path provided as a command-line argument
init_image = Image.open(args.image_path).convert("RGB")
init_image.show()
init_image = init_image.resize((768, 512))

prompt = "green shrek"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("giga_variation.png")
