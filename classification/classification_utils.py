import numpy as np

import pandas as pd
from skimage.measure import label, regionprops
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

        frame_properties = []
        dataframe_properties = []

        for prop in props:
            region_label = prop.label

            # Create a unique label across frames
            unique_label = f"{region_label}_{unique_label_counter}"
            unique_label_counter += 1

            # Append the unique label to the properties
            prop_dict = {'label': unique_label, 'frame_index': frame_index, 'regionprops': prop}
            frame_properties.append(prop_dict)
            centroid_x, centroid_y = prop.centroid
            if not is_near_border(centroid_x, centroid_y, 40, (x, y)):
                # Extract relevant properties
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
                on_grooves = np.mean(values_at_coords == 255) >= 0.4
            else:
                on_grooves = True
            conditions = (row["minor_axis_length"] <= 27 and
                          np.abs(row["orientation"]) >= 1.47 and
                          row["extent"] >= 0.73 and
                          row["major_axis_length"] / row["minor_axis_length"] >= 1.2 and
                          on_grooves
                          )

            label_value = 1 + int(conditions)

            original_label = int(row['label'].split("_")[0])

            #replace labeled_image regions with label_value when video's pixels are equal to original_label
            labeled_image[video[frame_id] == original_label] = label_value

        caged_video[frame_id, :, :] = labeled_image

    return caged_video


