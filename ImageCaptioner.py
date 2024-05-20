from transformers import AutoProcessor, AutoModelForCausalLM
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re

class ImageCaptioner:
    def __init__(self, checkpoint="microsoft/git-base-coco"):
        self.inputs = None
        self.outputs = None
        self.checkpoint = checkpoint
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.create_processor()
        self.create_model()
        self.tokenizer = self.create_tokenizer()
        self.generated_caption = None

    def create_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)

    def create_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint)

    def create_tokenizer(self):
        auto_tokenizer = AutoTokenizer if 'vit' in self.checkpoint else AutoProcessor
        return auto_tokenizer.from_pretrained(self.checkpoint)

    def process_input(self, image):
        return self.processor(images=image, return_tensors="pt").to(self.device)

    def open_image(self, image):
        url = re.compile(
            r'^(?:http|ftp)s?://*', re.IGNORECASE)

        image_pattern = re.compile(r'.*\.(jpg|jpeg|png|gif|bmp|svg)$', re.IGNORECASE)
        if re.match(url, image):
            return Image.open(requests.get(image, stream=True).raw)
        elif re.match(image_pattern, image):
            return Image.open(image)
        else:
            raise ValueError("Neither URL nor Image")

    def generate_caption(self, image_url, max_length=50):
        image = self.open_image(image_url)
        self.inputs = self.process_input(image)
        pixel_values = self.inputs.pixel_values
        self.outputs = self.model.generate(pixel_values=pixel_values.to(self.device), max_length=max_length, return_dict_in_generate=True, output_scores=True)
        self.generated_caption = self.tokenizer.batch_decode(self.outputs.sequences, skip_special_tokens=True)[0]
        return [self.generated_caption, self.outputs]

    def display_image_with_caption(self, image_url):
        image = self.open_image(image_url)
        generated_caption = self.generate_caption(image_url)
        plt.imshow(image)
        plt.axis('off')
        plt.title(generated_caption[0])
        plt.show()

    def transition_scores(self):
        caption_id = 0
        generated_outputs = self.outputs
        generated_outputs.scores[0].shape, generated_outputs.scores[0][caption_id].shape
        return self.model.compute_transition_scores(generated_outputs.sequences, generated_outputs.scores, normalize_logits=True)

    def print_outputs(self):
        input_length = 1 if self.model.config.is_encoder_decoder else self.inputs.pixel_values.shape[1]
        input_length = 1

        for idx, output in enumerate(self.outputs.sequences):
            print(self.generated_caption[idx])
            caption_np = output.numpy()

            try:
                pad_index = np.where(caption_np==0)[0][0]
            except:
                pad_index = -1

            generated_tokens = output[input_length:pad_index]
            print(generated_tokens)
            for tok, score in zip(generated_tokens, self.transition_scores()[idx]):
                # | token | token string | logits | probability
                print(f"| {tok:5d} | {self.processor.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

    def process_generated_outputs(self):
        input_length = 1 if self.model.config.is_encoder_decoder else self.inputs.pixel_values.shape[1]
        input_length = 1

        processed_outputs = []

        for idx, output in enumerate(self.outputs.sequences):
            caption = self.generated_caption[idx]
            caption_np = output.numpy()

            try:
                pad_index = np.where(caption_np==0)[0][0]
            except:
                pad_index = -1

            generated_tokens = output[input_length:pad_index]

            tokens_info = []
            for tok, score in zip(generated_tokens, self.transition_scores()[idx]):
                token_info = {
                    'token': int(tok),
                    'token_string': self.processor.decode(tok),
                    'logits': float(score.numpy()),
                    'probability': float(np.exp(score.numpy()))
                }
                tokens_info.append(token_info)

            processed_outputs.append({'caption': caption, 'tokens_info': tokens_info})

        return processed_outputs


