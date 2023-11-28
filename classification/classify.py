import numpy as np

from skimage import io, util
from classification.classification_utils import create_object_labels, find_caged_nucleus


def classify_volume(volume, preprocessed_volume, filter_grooves):
    video = io.imread(volume)
    preprocessed = io.imread(preprocessed_volume)
    df_all = create_object_labels(video)
    final_label_image = find_caged_nucleus(df_all, video, preprocessed, filter_grooves)
    channel_0_2_normalized = (final_label_image * (255 / 2)).astype(np.uint8)
    if preprocessed.ndim == 3:
        preprocessed = np.expand_dims(preprocessed, axis=1)
    final_image = np.concatenate([preprocessed, np.expand_dims(channel_0_2_normalized, axis=1)], axis=1)
    return final_image

