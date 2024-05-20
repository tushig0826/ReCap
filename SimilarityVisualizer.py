import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class SimilarityVisualizer:
    def __init__(self, texts, similarity, image_directory):
        self.texts = texts
        self.similarity = similarity
        self.image_files = self._get_image_files(image_directory)
        self.count = len(texts)

    def _get_image_files(self, image_directory):
        image_files = []
        for file in os.listdir(image_directory):
            if file.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
                image_files.append(os.path.join(image_directory, file))
        return image_files

    def _load_images(self):
        images = []
        for image_file in self.image_files:
            images.append(np.array(Image.open(image_file)))
        return images

    def _plot_similarity_matrix(self):
        plt.imshow(self.similarity, vmin=0.1, vmax=0.3)
        plt.yticks(range(self.count), self.texts, fontsize=18)
        plt.xticks([])
        for i, image in enumerate(self._load_images()):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(self.similarity.shape[1]):
            for y in range(self.similarity.shape[0]):
                plt.text(x, y, f"{self.similarity[y, x]:.2f}", ha="center", va="center", size=12)

    def _customize_plot(self):
        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        plt.xlim([-0.5, self.count - 0.5])
        plt.ylim([self.count + 0.5, -2])
        plt.title("Cosine similarity between text and image features", size=20)

    def visualize_similarity(self):
        plt.figure(figsize=(20, 14))
        self._plot_similarity_matrix()
        self._customize_plot()
        plt.show()
