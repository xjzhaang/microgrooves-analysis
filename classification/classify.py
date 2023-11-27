import numpy as np

from skimage import io, util
from classification.classification_utils import create_object_labels, find_caged_nucleus


def classify_volume(volume):
    video = io.imread(volume)
    video = np.transpose(video, (0, 3, 1, 2))
    df_all = create_object_labels(video)
    final_label_image = find_caged_nucleus(df_all, video)
    channel_0_2_normalized = (final_label_image * (255 / 2)).astype(np.uint8)
    final_image = np.concatenate([video[:, :-1, :, :], np.expand_dims(channel_0_2_normalized, axis=1)], axis=1)
    return final_image

