import numpy as np
import torch
from pkg_resources import packaging
import clip
import cv2
from PIL import Image
import os
import json
import pickle

class RelevanceEvaluator:
    def __init__(self, checkpoint="ViT-B/32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(checkpoint, device=self.device)


    def extract_image_features(self, image_path):
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
        return image_features

    def extract_text_features(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif isinstance(texts, list):
            texts = [str(text) for text in texts]

        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        return text_features

    def measure_similarity(self, image_path, text):
        text_features = self.extract_text_features(text)
        image_features = self.extract_image_features(image_path)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ image_features.T).cpu().numpy()
        return similarity
