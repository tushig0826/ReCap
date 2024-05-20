import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


class ImageGenerator:
    def __init__(self, checkpoint="CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = StableDiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch_dtype)
        self.model = self.model.to(self.device)

    def generate_image(self, prompt):
        image = self.model(prompt).images[0]
        return image

    @staticmethod
    def save_image_to_drive(image, filename, save_dir='/content/drive/MyDrive/OOD_dataset'):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)
