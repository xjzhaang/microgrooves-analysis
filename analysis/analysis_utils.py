import pandas as pd
import numpy as np
from tracking.trackmate_utils import read_trackmate_xml


def analyze_caging(spots, label):
    bool_list = list(spots[spots["label"] == label]["caged"])
    true_count = 0
    true_lengths = []
    false_count = 0
    false_lengths = []
    current_length = 0
    current_value = bool_list[0] if bool_list else None

    for value in bool_list:
        if value == current_value:
            current_length += 1
        else:
            if current_value:
                true_count += 1
                true_lengths.append(current_length)
            else:
                false_count += 1
                false_lengths.append(current_length)
            current_value = value
            current_length = 1

    # Check if the last group extends to the end of the list
    if current_value:
        true_count += 1
        true_lengths.append(current_length)
    else:
        false_count += 1
        false_lengths.append(current_length)

    # Calculate min, max, and average for true groups
    true_lengths = [elem for elem in true_lengths if elem > 1]
    false_lengths = [elem for elem in false_lengths if elem > 1]
    min_true_length = min(true_lengths) if true_lengths else 0
    max_true_length = max(true_lengths) if true_lengths else 0
    avg_true_length = sum(true_lengths) / len(true_lengths) if true_lengths else 0
    tot_true_length = sum(true_lengths) if true_lengths else 0

    # Calculate min, max, and average for false groups
    min_false_length = min(false_lengths) if false_lengths else 0
    max_false_length = max(false_lengths) if false_lengths else 0
    avg_false_length = sum(false_lengths) / len(false_lengths) if false_lengths else 0
    tot_false_length = sum(false_lengths) if false_lengths else 0

    return (true_count, min_true_length, max_true_length, avg_true_length, tot_true_length,
            false_count, min_false_length, max_false_length, avg_false_length, tot_false_length, true_lengths, false_lengths)


def analyze_shape(spots, label):
    track_spots = spots[spots["label"] == label]
    area_caged = list(track_spots.loc[track_spots['caged'], 'AREA'])
    average_area_caged = track_spots.loc[track_spots['caged'], 'AREA'].mean()
    average_area_uncaged = track_spots.loc[~track_spots['caged'], 'AREA'].mean()
    aspect_ratio = 1 / (track_spots.loc[track_spots['caged'], 'ELLIPSE_MAJOR'] /
                        track_spots.loc[track_spots['caged'], 'ELLIPSE_MINOR'])
    aspect_ratio_caged = list(aspect_ratio)
    mean_aspect_ratio_caged = aspect_ratio.mean()
    major_length = track_spots.loc[track_spots['caged'], 'ELLIPSE_MAJOR'].mean()
    std_aspect_ratio_caged = aspect_ratio.std()
    solidity_uncaged = list(track_spots.loc[~track_spots['caged'], 'SOLIDITY'])
    mean_solidity_uncaged = track_spots.loc[~track_spots['caged'], 'SOLIDITY'].mean()
    std_solidity_uncaged = track_spots.loc[~track_spots['caged'], 'SOLIDITY'].std()
    major_axis_length_list = list(track_spots.loc[track_spots['caged'], 'ELLIPSE_MAJOR'])

    return (area_caged, average_area_caged, average_area_uncaged, aspect_ratio_caged, mean_aspect_ratio_caged, std_aspect_ratio_caged,
            solidity_uncaged, mean_solidity_uncaged, std_solidity_uncaged, major_length, major_axis_length_list)


def analyze_edges(spots, edges):

    spot_states_dict = dict(zip(spots['ID'], spots['caged']))

    spot_x_dict = dict(zip(spots['ID'], spots['POSITION_X']))
    spot_y_dict = dict(zip(spots['ID'], spots['POSITION_Y']))

    # Use NumPy vectorized operations to create 'spot_states' column
    edges['spot_states'] = list(zip(
        np.vectorize(spot_states_dict.get)(edges['SPOT_SOURCE_ID']),
        np.vectorize(spot_states_dict.get)(edges['SPOT_TARGET_ID'])
    ))

    edges['spot_x'] = list(zip(
        np.vectorize(spot_x_dict.get)(edges['SPOT_SOURCE_ID']),
        np.vectorize(spot_x_dict.get)(edges['SPOT_TARGET_ID'])
    ))

    edges['spot_y'] = list(zip(
        np.vectorize(spot_y_dict.get)(edges['SPOT_SOURCE_ID']),
        np.vectorize(spot_y_dict.get)(edges['SPOT_TARGET_ID'])
    ))
    state_mapping = {
        (False, False): 'uncaged',
        (True, True): 'caged',
        (False, True): 'transition',
        (True, False): 'transition'
    }
    edges['state_type'] = edges['spot_states'].map(state_mapping)

    frame_dict = dict(zip(spots['ID'], spots['FRAME']))

    # Use NumPy vectorized operations to create 'frame' column containing tuples
    edges['FRAMES'] = list(zip(
        np.vectorize(frame_dict.get)(edges['SPOT_SOURCE_ID']),
        np.vectorize(frame_dict.get)(edges['SPOT_TARGET_ID'])
    ))

    # Assign label using vectorized operations
    edges['label'] = spots.set_index('ID')['label'].loc[edges['SPOT_SOURCE_ID'].values].values

    return edges


def analyze_caging_periods(label_edges):
    # Initialize lists to store information for each caging period
    caging_periods = []

    # Initialize variables to track the current caging period
    current_caging_period = None
    current_uncaging_period = None

    # Iterate over rows in the dataframe
    for index, row in label_edges.iterrows():
        # Check if the state_type is 'caged', 'uncaged', or 'transition'
        if row['state_type'] == 'caged':
            if current_caging_period is not None:
                current_caging_period["speed"].append(row["SPEED"])
                current_caging_period["frames"].append(row["FRAMES"])
                current_caging_period["displacement"].append(row["DISPLACEMENT"])
                current_caging_period["x_coord"].append(row["spot_x"])
                current_caging_period["y_coord"].append(row["spot_y"])
            else:
                current_caging_period = {
                    "state": "caged",
                    "frames": [row["FRAMES"]],
                    "speed": [row["SPEED"]],
                    "displacement": [row["DISPLACEMENT"]],
                    "x_coord": [row["spot_x"]],
                    "y_coord": [row["spot_y"]],

                }
        if row['state_type'] == 'uncaged':
            if current_uncaging_period is not None:
                current_uncaging_period["speed"].append(row["SPEED"])
                current_uncaging_period["frames"].append(row["FRAMES"])
                current_uncaging_period["displacement"].append(row["DISPLACEMENT"])
                current_uncaging_period["x_coord"].append(row["spot_x"])
                current_uncaging_period["y_coord"].append(row["spot_y"])
            else:
                current_uncaging_period = {
                    "state": "uncaged",
                    "frames": [row["FRAMES"]],
                    "speed": [row["SPEED"]],
                    "displacement": [row["DISPLACEMENT"]],
                    "x_coord": [row["spot_x"]],
                    "y_coord": [row["spot_y"]],
                }
        if row['state_type'] == 'transition':
            if current_caging_period is not None:
                caging_periods.append(current_caging_period)
                current_caging_period = None
            if current_uncaging_period is not None:
                caging_periods.append(current_uncaging_period)
                current_uncaging_period = None

    if current_caging_period is not None:
        caging_periods.append(current_caging_period)
        current_caging_period = None
    if current_uncaging_period is not None:
        caging_periods.append(current_uncaging_period)
        current_uncaging_period = None

    return caging_periods


def get_speeds_from_list(caging_list, fps):
    caged_min_speeds = []
    caged_max_speeds = []
    caged_avg_speeds = []
    uncaged_min_speeds = []
    uncaged_max_speeds = []
    uncaged_avg_speeds = []

    caged_speeds = []
    uncaged_speeds = []
    caged_displacements = []
    uncaged_displacements = []
    caged_x_coords = []
    uncaged_x_coords = []
    caged_y_coords = []
    uncaged_y_coords = []

    for l in caging_list:
        if l["state"] == "caged":
            for val in l["speed"]:
                caged_speeds.append(val)
            for val in l["displacement"]:
                caged_displacements.append(val)
            for val in l["x_coord"]:
                caged_x_coords.append(val)
            for val in l["y_coord"]:
                caged_y_coords.append(val)

        if l["state"] == "uncaged":
            for val in l["speed"]:
                uncaged_speeds.append(val)
            for val in l["displacement"]:
                uncaged_displacements.append(val)
            for val in l["x_coord"]:
                uncaged_x_coords.append(val)
            for val in l["y_coord"]:
                uncaged_y_coords.append(val)

    if caged_speeds:
        caged_speeds = [i * 0.325 / (fps / 60 / 60) for i in caged_speeds]
        caged_speeds = np.array(caged_speeds)
        caged_speed_vals = [np.min(caged_speeds), np.max(caged_speeds), np.mean(caged_speeds), np.median(caged_speeds),
                            np.std(caged_speeds)]
    else:
        caged_speed_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]

    if uncaged_speeds:
        uncaged_speeds = [i * 0.325 / (fps / 60 / 60) for i in uncaged_speeds]
        uncaged_speeds = np.array(uncaged_speeds)
        uncaged_speed_vals = [np.min(uncaged_speeds), np.max(uncaged_speeds), np.mean(uncaged_speeds),
                              np.median(uncaged_speeds), np.std(uncaged_speeds)]
    else:
        uncaged_speed_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return (caged_speed_vals, uncaged_speed_vals, list(caged_speeds), list(uncaged_speeds),
            list(caged_displacements), list(uncaged_displacements), list(caged_x_coords), list(caged_y_coords),
            list(uncaged_x_coords), list(uncaged_y_coords),)


def analyze_data(spots, tracks, edges, fps):
    spots.drop(
        columns=["QUALITY", "MANUAL_SPOT_COLOR", "MEAN_INTENSITY_CH1", "MEDIAN_INTENSITY_CH1", "MIN_INTENSITY_CH1",
                 "MAX_INTENSITY_CH1", "TOTAL_INTENSITY_CH1", "STD_INTENSITY_CH1", "CONTRAST_CH1", "SNR_CH1",
                 "VISIBILITY", "ELLIPSE_X0", "ELLIPSE_Y0", "POSITION_Z", "POSITION_T"], inplace=True)

    tracks.drop(
        columns=["TRACK_INDEX", "NUMBER_SPOTS", "NUMBER_GAPS", "NUMBER_SPLITS", "NUMBER_MERGES", "NUMBER_COMPLEX",
                 "LONGEST_GAP", "TRACK_Z_LOCATION", "TRACK_X_LOCATION", "TRACK_Y_LOCATION", "TRACK_MEAN_QUALITY"],
        inplace=True)
    edges.drop(columns=["MANUAL_EDGE_COLOR", "EDGE_Z_LOCATION", "EDGE_TIME", "EDGE_X_LOCATION", "EDGE_Y_LOCATION"],
               inplace=True)

    filtered_spots = spots.groupby('label').apply(lambda group: group.sort_values(by='FRAME')).reset_index(drop=True)
    filtered_spots['caged'] = filtered_spots['caged'] == 255.0  # Convert to boolean

    # creation of track features
    track_features = pd.DataFrame(index=filtered_spots['label'].unique())
    track_features['has_caged_spots'] = filtered_spots.groupby('label')['caged'].any()
    track_features["caged_speed_list"] = None
    track_features["uncaged_speed_list"] = None

    track_features["caged_displacements"] = None
    track_features["uncaged_displacements"] = None
    track_features["caged_x_coords"] = None
    track_features["caged_y_coords"] = None
    track_features["uncaged_x_coords"] = None
    track_features["uncaged_y_coords"] = None
    track_features["area_caged_list"] = None
    track_features["caging_periods"] = None
    track_features["caging_periods_s"] = None
    track_features["uncaging_periods"] = None
    track_features["aspect_ratio_caged"] = None
    track_features["solidity_uncaged"] = None
    track_features["major_axis_length_list"] = None

    # compute caging period and area data
    for label in filtered_spots["label"].unique():
        (true_count, min_true_length, max_true_length, avg_true_length,
         tot_true_length, false_count, min_false_length, max_false_length,
         avg_false_length, tot_false_length, true_lengths, false_lengths) = analyze_caging(filtered_spots, label)
        (area_caged, average_area_caged, average_area_uncaged, aspect_ratio_caged, mean_aspect_ratio_caged,
         std_aspect_ratio_caged, solidity_uncaged, mean_solidity_uncaged,
         std_solidity_uncaged, major_length, major_axis_length_list) = analyze_shape(filtered_spots, label)
        track_features.at[label, "number_of_caging"] = int(true_count)
        track_features.at[label, "total_period"] = int(len(list(filtered_spots[filtered_spots["label"] == label]["caged"])))
        track_features.at[label, "caging_periods"] = true_lengths
        track_features.at[label, "total_period_s"] = int(len(list(filtered_spots[filtered_spots["label"] == label]["caged"]))) * fps / 60 / 60
        track_features.at[label, "caging_periods_s"] = [i * fps / 60 / 60 for i in true_lengths]
        track_features.at[label, "uncaging_periods"] = false_lengths
        track_features.at[label, "min_caging_period"] = min_true_length
        track_features.at[label, "max_caging_period"] = max_true_length
        track_features.at[label, "avg_caging_period"] = avg_true_length
        track_features.at[label, "tot_caging_period"] = tot_true_length
        track_features.at[label, "major_length"] = major_length
        track_features.at[label, "min_uncaging_period"] = min_false_length
        track_features.at[label, "max_uncaging_period"] = max_false_length
        track_features.at[label, "avg_uncaging_period"] = avg_false_length
        track_features.at[label, "tot_uncaging_period"] = tot_false_length
        track_features.at[label, "area_caged_list"] = area_caged
        track_features.at[label, "average_area_caged"] = average_area_caged
        track_features.at[label, "average_area_uncaged"] = average_area_uncaged
        track_features.at[label, "aspect_ratio_caged"] = aspect_ratio_caged
        track_features.at[label, "std_aspect_ratio_caged"] = std_aspect_ratio_caged
        track_features.at[label, "mean_aspect_ratio_caged"] = mean_aspect_ratio_caged
        track_features.at[label, "solidity_uncaged"] = solidity_uncaged
        track_features.at[label, "mean_solidity_uncaged"] = mean_solidity_uncaged
        track_features.at[label, "std_solidity_uncaged"] = std_solidity_uncaged
        track_features.at[label, "major_axis_length_list"] = major_axis_length_list


    # merging with track dataframe
    track_features = pd.merge(track_features, tracks, left_index=True, right_on='TRACK_ID')
    track_features["TRACK_MEAN_SPEED"] = track_features["TRACK_MEAN_SPEED"] * 0.325 / (fps / 60 / 60)
    track_features.drop(columns=["TRACK_DURATION", "TRACK_START", "TRACK_STOP"], inplace=True)

    edges = analyze_edges(filtered_spots, edges)
    edges = edges.sort_values(['label', 'FRAMES']).reset_index(drop=True)

    for idx, label in enumerate(edges["label"].unique()):
        (caged_speed_vals, uncaged_speed_vals, caged_speeds, uncaged_speeds,
         caged_displacements, uncaged_displacements, caged_x_coords, caged_y_coords,
         uncaged_x_coords, uncaged_y_coords) \
            = get_speeds_from_list(analyze_caging_periods(edges[edges["label"] == label]), fps)
        track_features.at[idx, "caged_min_speed"] = caged_speed_vals[0]
        track_features.at[idx, "caged_max_speed"] = caged_speed_vals[1]
        track_features.at[idx, "caged_mean_speed"] = caged_speed_vals[2]
        track_features.at[idx, "caged_median_speed"] = caged_speed_vals[3]
        track_features.at[idx, "caged_std_speed"] = caged_speed_vals[4]
        track_features.at[idx, "uncaged_min_speed"] = uncaged_speed_vals[0]
        track_features.at[idx, "uncaged_max_speed"] = uncaged_speed_vals[1]
        track_features.at[idx, "uncaged_mean_speed"] = uncaged_speed_vals[2]
        track_features.at[idx, "uncaged_median_speed"] = uncaged_speed_vals[3]
        track_features.at[idx, "uncaged_std_speed"] = uncaged_speed_vals[4]
        track_features.at[idx, "caged_speed_list"] = caged_speeds
        track_features.at[idx, "uncaged_speed_list"] = uncaged_speeds
        track_features.at[idx, "caged_displacements"] = caged_displacements
        track_features.at[idx, "uncaged_displacements"] = uncaged_displacements

        track_features.at[idx, "caged_x_coords"] = caged_x_coords
        track_features.at[idx, "caged_y_coords"] = caged_y_coords
        track_features.at[idx, "uncaged_x_coords"] = uncaged_x_coords
        track_features.at[idx, "uncaged_y_coords"] = uncaged_y_coords

    return track_features


def read_and_analyze(xml_file, fps):
    spots, tracks, edges = read_trackmate_xml(xml_file, get_tracks=True, get_edges=True)
    track_features = analyze_data(spots, tracks, edges, fps)

    track_features.to_csv(str(xml_file).replace("trackings", "analysis").replace("xml", "csv"))