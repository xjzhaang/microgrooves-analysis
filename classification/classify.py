import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io
from kneed import KneeLocator
from classification.classification_utils import create_object_labels, find_caged_nucleus


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


def sensitivity_analysis(dataframe, grooves, param_dict, filter_grooves):
    total_objects = len(dataframe)
    fixed_params = {'minor_axis_length': 30, 'orientation': 1.47, 'extent': 0.74, 'major_minor_ratio': 1.5,
                    'grooves_area': 0.4}

    label_value_2_count = {key: np.zeros_like(param) for key, param in param_dict.items()}

    for frame_id in tqdm(range(dataframe['frame_index'].max() + 1)):
        frame_0_data = dataframe[dataframe['frame_index'] == frame_id]

        for index, row in frame_0_data.iterrows():
            if filter_grooves:
                coords = row["coords"]
                values_at_coords = grooves[frame_id, 0][coords[:, 0], coords[:, 1]]
                on_grooves = np.mean(values_at_coords == 255) >= 0.4
            else:
                on_grooves = True
            for varying_param, param_values in param_dict.items():
                for idx, param_value in enumerate(param_values):
                    conditions = False
                    if varying_param == "minor_axis_length":
                        conditions = (
                            (row[varying_param] <= param_value and
                             np.abs(row["orientation"]) >= fixed_params['orientation'] and
                             row["extent"] >= fixed_params['extent'] and
                             row["major_axis_length"] / row["minor_axis_length"] >= fixed_params[
                                 'major_minor_ratio'] and
                             on_grooves)
                        )
                    elif varying_param == "orientation":
                        conditions = (
                            (row["minor_axis_length"] <= fixed_params["minor_axis_length"] and
                             np.abs(row[varying_param]) >= param_value and
                             row["extent"] >= fixed_params['extent'] and
                             row["major_axis_length"] / row["minor_axis_length"] >= fixed_params[
                                 'major_minor_ratio'] and
                             on_grooves)
                        )
                    elif varying_param == "extent":
                        conditions = (
                            (row["minor_axis_length"] <= fixed_params["minor_axis_length"] and
                             np.abs(row["orientation"]) >= fixed_params['orientation'] and
                             row[varying_param] >= param_value and
                             row["major_axis_length"] / row["minor_axis_length"] >= fixed_params[
                                 'major_minor_ratio'] and
                             on_grooves)
                        )
                    elif varying_param == "major_minor_ratio":
                        conditions = (
                            (row["minor_axis_length"] <= fixed_params["minor_axis_length"] and
                             np.abs(row["orientation"]) >= fixed_params['orientation'] and
                             row["extent"] >= fixed_params['extent'] and
                             row["major_axis_length"] / row["minor_axis_length"] >= param_value and
                             on_grooves)
                        )
                    elif varying_param == "grooves_area":
                        if filter_grooves:
                            on_grooves = np.mean(values_at_coords == 255) >= param_value
                        conditions = (
                            (row["minor_axis_length"] <= fixed_params["minor_axis_length"] and
                             np.abs(row["orientation"]) >= fixed_params['orientation'] and
                             row["extent"] >= fixed_params['extent'] and
                             row["major_axis_length"] / row["minor_axis_length"] >= fixed_params[
                                 'major_minor_ratio'] and
                             on_grooves)
                        )
                    if conditions:
                        label_value_2_count[varying_param][idx] += 1

    for key in label_value_2_count.keys():
        label_value_2_count[key] = label_value_2_count[key] / total_objects

    return label_value_2_count


def sensitivities_from_paths(video_paths, filter_grooves):
    param_dict = {"minor_axis_length": np.arange(15, 45, 1),
                  'orientation': np.arange(1.2, 1.61, 0.01),
                  'extent': np.arange(0.5, 0.91, 0.01),
                  'major_minor_ratio': np.arange(1, 4.1, 0.1),
                  'grooves_area': np.arange(0.1, 0.825, 0.025),
                  }

    full_dict = {key: np.zeros((len(video_paths), param.shape[0])) for key, param in param_dict.items()}
    mean_std_dict = {key: [] for key in param_dict.keys()}

    for idx, video_path in enumerate(video_paths):
        video = io.imread(video_path)
        preprocessed = io.imread(str(video_path).replace("segmentations", "preprocessed"))
        label = create_object_labels(video)
        result_dict = sensitivity_analysis(label, preprocessed, param_dict, filter_grooves)
        for key in result_dict.keys():
            full_dict[key][idx] = result_dict[key]

    for key in full_dict.keys():
        mean_std_dict[key].append(np.mean(full_dict[key], axis=0))
        mean_std_dict[key].append(np.std(full_dict[key], axis=0))

    return full_dict, param_dict, mean_std_dict


def plot_sensitivity(save_path, param_dict, mean_std_dict):
    num_variables = len(param_dict)
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
        #     kneedle = KneeLocator(x[5:-5], y[5:-5], S=5, curve="concave", direction="increasing", interp_method="polynomial",
        #                           polynomial_degree=4)
        # else:
        #     kneedle = KneeLocator(x[5:-5], y[5:-5], S=5, curve="concave", direction="decreasing", interp_method="polynomial",
        #                           polynomial_degree=4)
        # knee_point = round(kneedle.knee, 3)
        # target_y_value = round(np.interp(knee_point, x, y), 3)
        # axes[i].scatter(knee_point, target_y_value, color='red', label=f'Knee (x={knee_point},y={target_y_value})')
        axes[i].set_title(f' % caging against {list_of_keys[i]}')
        axes[i].set_xlabel(list_of_keys[i])
        axes[i].set_ylabel('caging percentage')
        axes[i].legend()
        axes[i].set_ylim([0, 0.8])

    plt.suptitle("Caging criteria sensitivity")
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
    plt.close()
