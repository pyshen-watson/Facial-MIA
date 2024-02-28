import numpy as np
from PIL import Image
from deepface import DeepFace

import cv2
import torch
from matplotlib import pyplot as plt


def extract_face(img: Image, size=(112, 112)) -> np.ndarray:
    img = np.array(img)
    face_obj = DeepFace.extract_faces(img, target_size=size, enforce_detection=False)
    face = face_obj[0]["face"] * 255
    return face.astype(np.uint8)

def draw_rows(img_ori, img_rec, n_pair=16, save_path=""):
    
    def tensor2img(tensor: torch.Tensor):
        tensor = tensor.permute(1, 2, 0) * 255
        ndarray = tensor.cpu().detach().numpy().astype(np.uint8)
        img = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
        return img
    
    # Draw two rows of images, upper row for original images and lower row for reconstructed images
    fig, axes = plt.subplots(2, n_pair, figsize=(n_pair, 2))
    for col, tensors in enumerate(zip(img_ori[:n_pair], img_rec[:n_pair])):
        for row, tensor in enumerate(tensors):
            img = tensor2img(tensor)
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_axis_off()

    fig.savefig(save_path)
