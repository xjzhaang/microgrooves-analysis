import numpy as np

import torch
from skimage import io
from monai.networks.nets import UNet

from preprocessing.preprocess_utils import (per_channel_scaling, apply_clahe, apply_intensity_clipping_and_denoising,
                              detect_and_rotate_angle, filter_microgrooves, filter_microgrooves_with_model)


def preprocess_image(image_path, model_path, keep_grooves=True, filter_with_model=True):
    image = io.imread(image_path)
    image = per_channel_scaling(image)
    image = apply_clahe(image)
    image = apply_intensity_clipping_and_denoising(image)
    if "Flat_" not in image_path.name and "FlatPos" not in image_path.name:
        image = detect_and_rotate_angle(image)
        if not filter_with_model:
            image = filter_microgrooves(image)
        else:
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),)
            model.load_state_dict(torch.load(model_path)["model"])
            image = filter_microgrooves_with_model(image, model)
        if not keep_grooves:
                image = np.flip(image[:, 1:, :, :], axis=1)
    return image.astype(np.float32)


