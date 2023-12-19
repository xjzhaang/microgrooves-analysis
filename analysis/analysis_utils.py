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
            false_count, min_false_length, max_false_length, avg_false_length, tot_false_length)


def analyze_area(spots, label):
    track_spots = spots[spots["label"] == label]
    average_area_caged = track_spots.loc[track_spots['caged'], 'AREA'].mean()
    average_area_uncaged = track_spots.loc[~track_spots['caged'], 'AREA'].mean()

    return average_area_caged, average_area_uncaged


def analyze_edges(spots, edges):
    spot_states_dict = dict(zip(spots['ID'], spots['caged']))

    # Use NumPy vectorized operations to create 'spot_states' column
    edges['spot_states'] = list(zip(
        np.vectorize(spot_states_dict.get)(edges['SPOT_SOURCE_ID']),
        np.vectorize(spot_states_dict.get)(edges['SPOT_TARGET_ID'])
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
            else:
                current_caging_period = {
                    "state": "caged",
                    "frames": [row["FRAMES"]],
                    "speed": [row["SPEED"]]
                }
        if row['state_type'] == 'uncaged':
            if current_uncaging_period is not None:
                current_uncaging_period["speed"].append(row["SPEED"])
                current_uncaging_period["frames"].append(row["FRAMES"])
            else:
                current_uncaging_period = {
                    "state": "uncaged",
                    "frames": [row["FRAMES"]],
                    "speed": [row["SPEED"]]
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


def get_speeds_from_list(caging_list):
    caged_min_speeds = []
    caged_max_speeds = []
    caged_avg_speeds = []
    uncaged_min_speeds = []
    uncaged_max_speeds = []
    uncaged_avg_speeds = []

    caged_speeds = []
    uncaged_speeds = []

    for l in caging_list:
        if l["state"] == "caged":
            # caged_min_speeds.append(np.min(l["speed"]))
            # caged_max_speeds.append(np.max(l["speed"]))
            # caged_avg_speeds.append(np.mean(l["speed"]))
            for val in l["speed"]:
                caged_speeds.append(val)
        if l["state"] == "uncaged":
            # uncaged_min_speeds.append(np.min(l["speed"]))
            # uncaged_max_speeds.append(np.max(l["speed"]))
            # uncaged_avg_speeds.append(np.mean(l["speed"]))
            for val in l["speed"]:
                uncaged_speeds.append(val)

    if caged_speeds:
        caged_speeds = np.array(caged_speeds)
        caged_speed_vals = [np.min(caged_speeds), np.min(caged_speeds), np.mean(caged_speeds), np.median(caged_speeds),
                            np.std(caged_speeds)]
    else:
        caged_speed_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]

    if uncaged_speeds:
        uncaged_speeds = np.array(uncaged_speeds)
        uncaged_speed_vals = [np.min(uncaged_speeds), np.max(uncaged_speeds), np.mean(uncaged_speeds),
                              np.median(uncaged_speeds), np.std(uncaged_speeds)]
    else:
        uncaged_speed_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return caged_speed_vals, uncaged_speed_vals


def analyze_data(spots, tracks, edges):
    spots.drop(
        columns=["QUALITY", "MANUAL_SPOT_COLOR", "MEAN_INTENSITY_CH1", "MEDIAN_INTENSITY_CH1", "MIN_INTENSITY_CH1",
                 "MAX_INTENSITY_CH1", "TOTAL_INTENSITY_CH1", "STD_INTENSITY_CH1", "CONTRAST_CH1", "SNR_CH1",
                 "VISIBILITY", "ELLIPSE_X0", "ELLIPSE_Y0", "POSITION_Z", "POSITION_T"], inplace=True)
    tracks.drop(
        columns=["TRACK_INDEX", "NUMBER_SPOTS", "NUMBER_GAPS", "NUMBER_SPLITS", "NUMBER_MERGES", "NUMBER_COMPLEX",
                 "LONGEST_GAP", "TRACK_Z_LOCATION", "TRACK_X_LOCATION", "TRACK_Y_LOCATION", "TRACK_MEAN_QUALITY"],
        inplace=True)
    edges.drop(columns=["MANUAL_EDGE_COLOR", "EDGE_Z_LOCATION", "EDGE_X_LOCATION", "EDGE_Y_LOCATION", "EDGE_TIME"],
               inplace=True)

    filtered_spots = spots.groupby('label').apply(lambda group: group.sort_values(by='FRAME')).reset_index(drop=True)
    filtered_spots['caged'] = filtered_spots['caged'] == 255.0  # Convert to boolean

    # creation of track features
    track_features = pd.DataFrame(index=filtered_spots['label'].unique())
    track_features['has_caged_spots'] = filtered_spots.groupby('label')['caged'].any()

    # compute caging period and area data
    for label in filtered_spots["label"].unique():
        (true_count, min_true_length, max_true_length, avg_true_length, tot_true_length,
         false_count, min_false_length, max_false_length, avg_false_length, tot_false_length) = analyze_caging(filtered_spots, label)
        average_area_caged, average_area_uncaged = analyze_area(filtered_spots, label)
        track_features.at[label, "number_of_caging"] = int(true_count)
        track_features.at[label, "total_period"] = int(
            len(list(filtered_spots[filtered_spots["label"] == label]["caged"])))
        track_features.at[label, "min_caging_period"] = min_true_length
        track_features.at[label, "max_caging_period"] = max_true_length
        track_features.at[label, "avg_caging_period"] = avg_true_length
        track_features.at[label, "tot_caging_period"] = tot_true_length
        track_features.at[label, "min_uncaging_period"] = min_false_length
        track_features.at[label, "max_uncaging_period"] = max_false_length
        track_features.at[label, "avg_uncaging_period"] = avg_false_length
        track_features.at[label, "tot_uncaging_period"] = tot_false_length
        track_features.at[label, "average_area_caged"] = average_area_caged
        track_features.at[label, "average_area_uncaged"] = average_area_uncaged

    # merging with track dataframe
    track_features = pd.merge(track_features, tracks, left_index=True, right_on='TRACK_ID')
    track_features.drop(columns=["TRACK_DURATION", "TRACK_START", "TRACK_STOP"], inplace=True)

    edges = analyze_edges(filtered_spots, edges)
    edges = edges.sort_values(['label', 'FRAMES']).reset_index(drop=True)

    for idx, label in enumerate(edges["label"].unique()):
        caged_speed_vals, uncaged_speed_vals = get_speeds_from_list(
            analyze_caging_periods(edges[edges["label"] == label]))
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

    return track_features


def read_and_analyze(xml_file):
    spots, tracks, edges = read_trackmate_xml(xml_file, get_tracks=True, get_edges=True)
    track_features = analyze_data(spots, tracks, edges)

    track_features.to_csv(str(xml_file).replace("trackings", "analysis").replace("xml", "csv"))