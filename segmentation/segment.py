import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from cellpose import models as cellpose_models
from tqdm import tqdm


class MyoblastDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.volume = io.imread(self.file_path)

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        img = self.volume[idx]
        data = {}
        data["image"] = img[1]
        data["first_channel"] = img[0]

        return data


def inference_loop(dataloader, model):
    reconstructed_volume = []
    with torch.inference_mode():
        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = data['image'].numpy()
            pred_mask, _, _ = model.eval(image, channels=[0,0], diameter=49.03, normalize=True, net_avg=False)
            image_uint8 = (np.squeeze(image) * 255).astype(np.uint16)
            grooves_uint8 = np.squeeze(data['first_channel'].numpy()).astype(np.uint16)
            reconstructed_volume.append(np.stack([grooves_uint8, image_uint8, pred_mask], axis=0))

    # Stack the masks along a new dimension to get the 3D volume
    reconstructed_volume = np.stack(reconstructed_volume, axis=0)

    return reconstructed_volume


def segment_cells(volumes, model_path):
    model_cp = cellpose_models.CellposeModel(
        gpu=True if torch.cuda.is_available() else False,
        pretrained_model=model_path,
    )
    dataset = MyoblastDataset(volumes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    res_image = inference_loop(dataloader, model_cp)
    return res_image