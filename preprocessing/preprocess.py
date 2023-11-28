import numpy as np

import torch
from skimage import io, util
from monai.networks.nets import UNet

from preprocessing.preprocess_utils import (per_channel_scaling, apply_clahe,
                                            apply_intensity_clipping_and_denoising,
                                            detect_and_rotate_angle, filter_microgrooves,
                                            filter_microgrooves_with_model)


def preprocess_image(image_path, model_path, keep_grooves=True, filter_grooves=True, filter_with_model=True):
    image = io.imread(image_path)
    image = per_channel_scaling(image)
    image = apply_clahe(image)
    image = apply_intensity_clipping_and_denoising(image)
    if "Flat_" not in image_path.name and "FlatPos" not in image_path.name and image.ndim != 3:
        image = detect_and_rotate_angle(image, rho=4, sigma=25)
    if image.ndim == 3 or not filter_grooves:
        return util.img_as_ubyte(image)
    elif not filter_with_model:
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
    image = util.img_as_ubyte(image)
    return image



