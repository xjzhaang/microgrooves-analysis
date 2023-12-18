import numpy as np

import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm




def randomly_select_frames(video_data, num_frames=10):
    """
    Randomly select a specified number of frames from a video.

    Parameters:
    - video_data (np.ndarray): Video data of shape (t, x, y).
    - num_frames (int): Number of frames to randomly select.

    Returns:
    - selected_frames (np.ndarray): Randomly selected frames.
    """
    t, x, y = video_data.shape

    if num_frames > t:
        print("Error: Number of frames to select exceeds the total number of frames in the video.")
        return None

    selected_indices = np.random.choice(t, num_frames, replace=False)
    selected_frames = video_data[selected_indices]

    return selected_frames


def is_near_border(x, y, distance, image_shape):
    """
    Check if a point (x, y) is near the image border.

    Parameters:
    - x (int): x-coordinate of the point.
    - y (int): y-coordinate of the point.
    - distance (int): Minimum distance from the border to consider the point not near the border.
    - image_shape (tuple): Shape of the image (height, width).

    Returns:
    - bool: True if the point is near the border, False otherwise.
    """
    border_distance_x = min(x, image_shape[1] - x)
    border_distance_y = min(y, image_shape[0] - y)
    return border_distance_x < distance or border_distance_y < distance


def create_object_labels(video_data):
    """
    Create labels for objects in every frame of a video using scikit-image's regionprops.

    Parameters:
    - video_data (np.ndarray): Video data of shape (t, x, y).

    Returns:
    - frame_labels (list): List of labels for each frame.
        Each label is a list of dictionaries with 'label' and 'regionprops' fields.
    """
    t, x, y = video_data.shape
    all_dataframe = []
    unique_label_counter = 1

    for frame_index in range(t):
        frame = video_data[frame_index]

        # Assuming your images are already labeled
        labeled_frame = frame.copy()

        # Calculate region properties using regionprops
        props = regionprops(labeled_frame)

        # frame_properties = []
        dataframe_properties = []

        for prop in props:
            region_label = prop.label

            # Create a unique label across frames
            unique_label = f"{region_label}_{unique_label_counter}"
            unique_label_counter += 1

            # Append the unique label to the properties
            # prop_dict = {'label': unique_label, 'frame_index': frame_index, 'regionprops': prop}
            # frame_properties.append(prop_dict)
        #     centroid_x, centroid_y = prop.centroid
        # if not is_near_border(centroid_x, centroid_y, 20, (x, y)):
            #Extract relevant properties
            properties_dict = {
                'label': unique_label,
                'frame_index': frame_index,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'bbox': prop.bbox,
                'area_bbox': prop.area_bbox,
                'coords': prop.coords,
                'extent': prop.extent,
                'orientation': prop.orientation,
            }
            if properties_dict['minor_axis_length'] > 1:
                dataframe_properties.append(properties_dict)

        all_dataframe.extend(dataframe_properties)

    df = pd.DataFrame(all_dataframe)

    return df


def find_caged_nucleus(dataframe, video, grooves, filter_grooves):
    caged_video = np.zeros_like(video)

    for frame_id in tqdm(range(dataframe['frame_index'].max() + 1)):
        frame_0_data = dataframe[dataframe['frame_index'] == frame_id]

        # Create a blank labeled image for frame 0
        labeled_image = np.zeros_like(video[frame_id], dtype=np.uint8)

        # Assign cluster labels to each object in the labeled image
        for index, row in frame_0_data.iterrows():
            coords = row["coords"]
            if filter_grooves:
                values_at_coords = grooves[frame_id, 0][coords[:, 0], coords[:, 1]]
                on_grooves = np.mean(values_at_coords == 255) >= 0.47
                on_grooves2 = np.mean(values_at_coords == 255) >= 0.55
            else:
                on_grooves = True
                on_grooves2 = True
            conditions = ((row["minor_axis_length"] <= 26.5 and
                          np.abs(row["orientation"]) >= 1.47 and
                          row["extent"] >= 0.745 and
                          row["major_axis_length"] / row["minor_axis_length"] >= 2.0 and
                          on_grooves)
                          )
                          # or
                          # (row["minor_axis_length"] <= 25 and
                          # np.abs(row["orientation"]) >= 1.45 and
                          # row["extent"] >= 0.74 and
                          # row["major_axis_length"] / row["minor_axis_length"] >= 2.5 and
                          # on_grooves2)
                          # )

            label_value = 1 + int(conditions)

            original_label = int(row['label'].split("_")[0])

            #replace labeled_image regions with label_value when video's pixels are equal to original_label
            labeled_image[video[frame_id] == original_label] = label_value

        caged_video[frame_id, :, :] = labeled_image

    return caged_video


def sensitivity_analysis(dataframe, grooves, param_dict, filter_grooves, remove_one=False):
    total_objects = len(dataframe)
    fixed_params = {'minor_axis_length': 27, 'orientation': 1.48, 'extent': 0.75, 'major_minor_ratio': 2.0,
                    'grooves_area': 0.47}

    if remove_one:
        label_value_2_count = {key: 0 for key, param in param_dict.items()}
        label_value_2_count["keep_all"] = 0
    else:
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

            if remove_one:
                conditions = [
                    row["minor_axis_length"] <= fixed_params["minor_axis_length"],
                    np.abs(row["orientation"]) >= fixed_params['orientation'],
                    row["extent"] >= fixed_params['extent'],
                    row["major_axis_length"] / row["minor_axis_length"] >= fixed_params['major_minor_ratio'],
                    on_grooves
                ]
                label_value_2_count["keep_all"] += int(all(conditions))
                for i in range(len(conditions)):
                    new_conditions = conditions[:i] + conditions[i + 1:]
                    label_value_2_count[list(fixed_params.keys())[i]] += int(all(new_conditions))
            else:
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
                        label_value_2_count[varying_param][idx] += int(conditions)

    for key in label_value_2_count.keys():
        label_value_2_count[key] = label_value_2_count[key] / total_objects
    return label_value_2_count



