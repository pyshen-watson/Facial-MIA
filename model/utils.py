import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from typing import List


def tensor2img(tensor: torch.Tensor) -> cv2.Mat:
    tensor = tensor.permute(1, 2, 0) * 255
    ndarray = tensor.cpu().detach().numpy().astype(np.uint8)
    img = cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB)
    return img


def draw_rows(img_rows: List[torch.Tensor], row_titles: List[str], n_pair=20, save_path=""):

    n_rows = len(img_rows)
    fig, axes = plt.subplots(n_rows, n_pair, figsize=(n_pair*0.75, n_rows*0.5))
    fig.subplots_adjust(wspace=0, hspace=0)

    for row, (img_row, title) in enumerate(zip(img_rows, row_titles)):

        fig.text(0.05, (n_rows-row) / (n_rows+1), title, ha='left', va='center')

        for col, img in enumerate(img_row[:n_pair]):
            img = tensor2img(img)
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_axis_off()

    fig.savefig(save_path)
