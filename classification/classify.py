import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from classification.classification_utils import create_object_labels, find_caged_nucleus, sensitivity_analysis
from kneed import KneeLocator


def classify_volume(volume, preprocessed_volume, filter_grooves):
    video = io.imread(volume)
    preprocessed = io.imread(preprocessed_volume)
    df_all = create_object_labels(video)
    final_label_image = find_caged_nucleus(df_all, video, preprocessed, filter_grooves)
    channel_0_2_normalized = (final_label_image * (255 / 2)).astype(np.uint8)
    # if preprocessed.ndim == 3:
    #     preprocessed = np.expand_dims(preprocessed, axis=1)
    # final_image = np.concatenate([preprocessed, np.expand_dims(channel_0_2_normalized, axis=1)], axis=1)
    return channel_0_2_normalized


def sensitivities_from_paths(video_paths, filter_grooves, remove_one=False):
    param_dict = {"minor_axis_length": np.arange(15, 45, 1),
                  'orientation': np.arange(1.2, 1.61, 0.01),
                  'extent': np.arange(0.5, 0.91, 0.01),
                  'major_minor_ratio': np.arange(1, 4.1, 0.1),
                  'grooves_area': np.arange(0.1, 0.825, 0.025),
                  }

    mean_std_dict = {key: [] for key in param_dict.keys()}
    if remove_one:
        full_dict = {key: [] for key in param_dict.keys()}
        mean_std_dict["keep_all"] = []
        full_dict["keep_all"] = []
    else:
        full_dict = {key: np.zeros((len(video_paths), param.shape[0])) for key, param in param_dict.items()}

    for idx, video_path in enumerate(video_paths):
        video = io.imread(video_path)
        preprocessed = io.imread(str(video_path).replace("segmentations", "preprocessed"))
        label = create_object_labels(video)
        result_dict = sensitivity_analysis(label, preprocessed, param_dict, filter_grooves, remove_one)

        for key in result_dict.keys():
            if remove_one:
                full_dict[key].append(result_dict[key])
            else:
                full_dict[key][idx] = result_dict[key]

    for key in full_dict.keys():
        mean_std_dict[key].append(np.mean(full_dict[key], axis=0))
        mean_std_dict[key].append(np.std(full_dict[key], axis=0))

    return full_dict, param_dict, mean_std_dict


def plot_sensitivity(save_path, param_dict, full_dict, mean_std_dict, remove_one=False):
    num_variables = len(param_dict)
    if remove_one:
        fig, ax = plt.subplots()
        ax.boxplot([full_dict[key] for key in full_dict.keys()], labels=mean_std_dict.keys())
        ax.set_xticklabels(labels=mean_std_dict.keys(), rotation=45)
        ax.set_xlabel('Removed criterion')
        ax.set_ylabel('caging percentage')
        ax.set_title('% caging when removing one criterion')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_variables,
                                 figsize=(num_variables * 5, num_variables))
        list_of_keys = list(param_dict.keys())
        for i in range(num_variables):
            x = param_dict[list_of_keys[i]]
            y = mean_std_dict[list_of_keys[i]][0]
            axes[i].errorbar(x, y, yerr=mean_std_dict[list_of_keys[i]][1],
                             fmt='o', markersize=3, capsize=0.3, label=list_of_keys[i])
            if i == 0:
                axes[i].invert_xaxis()
            axes[i].set_title(f' % caging against {list_of_keys[i]}')
            axes[i].set_xlabel(list_of_keys[i])
            axes[i].set_ylabel('caging percentage')
            axes[i].legend()
            axes[i].set_ylim([0, 0.8])

        plt.suptitle("Caging criteria sensitivity")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()



