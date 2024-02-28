import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from configs import get_config
from model import get_backbone
from dataset import get_raw_dataset


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', default='base',
                        type=str, help='[base/lfw/ms1mv3]')
    args = parser.parse_args()
    return args


def main():

    # Prepare components
    arg = get_args()
    cfg = get_config(arg.config)
    backbone = get_backbone(cfg)
    dataset = get_raw_dataset(cfg)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For cache
    img_data = []
    feat_data = []

    # Evaluate
    backbone.to(device)
    backbone.eval()

    for img, label in tqdm(dataloader, desc='Extracting features'):

        # Inference
        img = img.to(device)
        with torch.no_grad():
            feature = backbone(img)

        # Record
        img = (img.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        img_data.append(img)
        feat_data.append(feature.cpu().numpy())

    # Save
    data_dict = {
        'img': np.concatenate(img_data, axis=0),
        'feat': np.concatenate(feat_data, axis=0)
    }

    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cfg.cache_path, data_dict)


if __name__ == '__main__':
    main()
