import numpy as np
from pathlib import Path

import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import random_split
from monai.data import ThreadDataLoader
from monai.transforms import Compose, RandCropByPosNegLabeld, RandFlipd, RandRotated, RandRotate90d

class MicrogroovesDataset(Dataset):

    def __init__(self, root_folder, is_train, transform=None):
        self.root_folder = Path(root_folder)
        self.is_train = is_train
        self.transform = transform
        self.image_folder = "train" if is_train else "test"
        self.image_dir = self.root_folder / f"{self.image_folder}/image"
        self.mask_dir = self.root_folder / f"{self.image_folder}/mask"

        # Assuming image and mask filenames are the same in both directories
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        self.mask_files = sorted(list(self.mask_dir.glob("*.tif")))

    def __len__(self):
        return len(self.image_files)

    def per_channel_scaling(self, image):
        image = image.astype(np.float32)
        min_vals = np.min(image)
        max_vals = np.max(image)
        scaled_image = (image - min_vals) / (max_vals - min_vals + 1e-8)

        return scaled_image

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = io.imread(image_path)
        mask = io.imread(mask_path)
        image = self.per_channel_scaling(image)
        mask = mask.astype(np.float32)
        data = {}
        data["image"] = torch.tensor(image).unsqueeze(0)
        data["mask"] = torch.tensor(mask).unsqueeze(0)
        if self.transform:
            data = self.transform(data)

        return data


def build_dataloaders(num_workers, batch_size_train, batch_size_val):

    train_transforms = Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(512, 512),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "mask"],
                spatial_axis=[0],
                prob=0.15,
            ),
            RandRotated(
                keys=["image", "mask"],
                range_x=0.4,
                range_y=0.4,
                prob=0.15,
            ),
            RandRotate90d(keys=["image", "mask"],
                          prob=0.15,
                          max_k=3, ),
        ]
    )

    train_data = MicrogroovesDataset(root_folder='', is_train=True, transform=train_transforms)
    test_data = MicrogroovesDataset(root_folder='', is_train=False, transform=None)
    test_size = len(test_data) // 2
    val_size = test_size
    test_size = len(test_data) - val_size
    val_data, test_data = random_split(test_data, [val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = ThreadDataLoader(train_data, num_workers=num_workers, batch_size=batch_size_train, shuffle=True)
    val_loader = ThreadDataLoader(val_data, num_workers=num_workers, batch_size=batch_size_val, shuffle=True)
    test_loader = ThreadDataLoader(test_data, num_workers=num_workers, batch_size=batch_size_val, shuffle=True)

    return train_loader, val_loader, test_loader