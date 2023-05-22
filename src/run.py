import argparse
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description='Image Captioning and Image Generation')
parser.add_argument('image_path1', type=str, help='Path to the first input image')
parser.add_argument('image_path2', type=str, help='Path to the second input image')
args = parser.parse_args()

# Load the vision transformer model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the stable diffusion model
sd_model_id_or_path = "runwayml/stable-diffusion-v1-5"
sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_id_or_path, torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sd_pipe.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return pred.strip()

def generate_image(prompt, image_path):
    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((768, 512))

    images = sd_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
    images[0].save("generated_image.png")

# Generate a caption for the first image
caption = predict_caption(args.image_path1)
print("Caption: ", caption)

# Use the caption as a prompt to generate an image from the second image
generate_image(caption, args.image_path2)


