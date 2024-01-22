# Code for function read_trackmate_xml adapted from https://github.com/hadim/pytrackmate
import executing.executing
import numpy as np

import xml.etree.ElementTree as et
import pandas as pd
from skimage import io


def read_trackmate_xml(trackmate_xml_path, get_tracks=False, get_edges=False):
    """Import detected peaks with TrackMate Fiji plugin.

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    objects = []
    object_labels = {'FRAME': 't_stamp',
                     'POSITION_T': 't',
                     'POSITION_X': 'x',
                     'POSITION_Y': 'y',
                     'POSITION_Z': 'z',
                     'QUALITY': 'q',
                     'ID': 'spot_id',
                     "RADIUS": "Radius",
                     "ELLIPSE_X0": "Ellipse center x0",
                     "ELLIPSE_Y0": "Ellipse center y0",
                     "ELLIPSE_MAJOR": "Ellipse long axis",
                     "ELLIPSE_MINOR": "Ellipse short axis",
                     "ELLIPSE_THETA": "Ellipse angle",
                     "ELLIPSE_ASPECTRATIO": "Ellipse aspect ratio",
                     "AREA": "Area",
                     "PERIMETER": "Perimeter",
                     "CIRCULARITY": "Circularity",
                     "SOLIDITY": "Solidity",
                     "SHAPE_INDEX": "Shape index",
                     }

    features = root.find('Model').find('FeatureDeclarations').find('SpotFeatures')
    features = [c.get('feature') for c in list(features)] + ['ID']

    spots = root.find('Model').find('AllSpots')
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall('SpotsInFrame'):
        for spot in frame.findall('Spot'):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)
    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(float)

    tracks = []
    track_labels = {"TRACK_ID": "Track ID",
                    "NUMBER_SPOTS": "Number of spots in track",
                    "NUMBER_GAPS": "Number of gaps",
                    "NUMBER_SPLITS": "Number of split events",
                    "NUMBER_MERGES": "Number of merge events",
                    "NUMBER_COMPLEX": "Number of complex points",
                    "LONGEST_GAP": "Longest gap",
                    "TRACK_DURATION": "Track duration",
                    "TRACK_START": "Track start",
                    "TRACK_STOP": "Track stop",
                    "TRACK_DISPLACEMENT": "Track displacement",
                    "TRACK_X_LOCATION": "Track mean X",
                    "TRACK_Y_LOCATION": "Track mean Y",
                    "TRACK_Z_LOCATION": "Track mean Z",
                    "TRACK_MEAN_SPEED": "Track mean speed",
                    "TRACK_MAX_SPEED": "Track max speed",
                    "TRACK_MIN_SPEED": "Track min speed",
                    "TRACK_MEDIAN_SPEED": "Track median speed",
                    "TRACK_STD_SPEED": "Track std speed",
                    "TRACK_MEAN_QUALITY": "Track mean quality",
                    "TOTAL_DISTANCE_TRAVELED": "Total distance traveled",
                    "MAX_DISTANCE_TRAVELED": "Max distance traveled",
                    "CONFINEMENT_RATIO": "Confinement ratio",
                    "MEAN_STRAIGHT_LINE_SPEED": "Mean straight line speed",
                    "LINEARITY_OF_FORWARD_PROGRESSION": "Linearity of forward progression",
                    "MEAN_DIRECTIONAL_CHANGE_RATE": "Mean directional change rate"}

    features = root.find('Model').find('FeatureDeclarations').find('TrackFeatures')
    features = [c.get('feature') for c in list(features)]

    spots = root.find('Model').find('AllTracks')
    tracks_ = pd.DataFrame([])
    tracks = []
    for frame in spots.findall('Track'):
        single_object = []
        for label in features:
            single_object.append(frame.get(label))
        tracks.append(single_object)

    tracks_ = pd.DataFrame(tracks, columns=features)
    tracks_ = tracks_.astype(float)

    # tracks_ = tracks_.loc[:, track_labels.keys()]
    # tracks_.columns = [track_labels[k] for k in track_labels.keys()]

    # trajs = trajs.loc[:, object_labels.keys()]
    # trajs.columns = [object_labels[k] for k in object_labels.keys()]
    trajs['label'] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in
                              root.find('Model').find('FilteredTracks').findall('TrackID')]

        label_id = 0
        trajs['label'] = np.nan

        tracks = root.find('Model').find('AllTracks')
        for track in tracks.findall('Track'):

            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:
                spot_ids = [(edge.get('SPOT_SOURCE_ID'), edge.get('SPOT_TARGET_ID'), edge.get('EDGE_TIME')) for edge in
                            track.findall('Edge')]
                spot_ids = np.array(spot_ids).astype('float')[:, :2]
                spot_ids = set(spot_ids.flatten())

                trajs.loc[trajs["ID"].isin(spot_ids), "label"] = track_id
                label_id += 1

    if get_edges:
        edge_features = root.find('Model').find('FeatureDeclarations').find('EdgeFeatures')
        edge_features = [c.get('feature') for c in list(edge_features)]
        all_tracks = root.find('Model').find('AllTracks')
        edges = []
        for track in all_tracks.findall('Track'):
            for frame in track.findall("Edge"):
                single_object = []
                for label in edge_features:
                    single_object.append(frame.get(label))
                edges.append(single_object)
        edges_ = pd.DataFrame(edges, columns=edge_features)
        edges_ = edges_.astype(float)

        return trajs, tracks_, edges_

    return trajs, tracks_


def add_caged_feature(row, image):
    x, y, z = row['POSITION_X'], row['POSITION_Y'], row['POSITION_T']
    pixel_value = image[int(z), int(y), int(x)]
    if pixel_value == 0:
        pixel_value = 127
    return pixel_value


def post_process_caging(row, df):
    if row['caged'] == 127.0 and (df['caged'].shift(1)[row.name] == 255.0 and df['caged'].shift(-1)[row.name] == 255.0):
        if (abs(df['POSITION_Y'].shift(-1)[row.name] - row['POSITION_Y']) < 5
                and abs(df['POSITION_Y'].shift(1)[row.name] - row['POSITION_Y']) < 5):
            return 255
        else:
            return 127

    elif row['caged'] == 127.0 and (np.isnan(df['caged'].shift(1)[row.name]) and df['caged'].shift(-1)[row.name] == 255.0 and df['caged'].shift(-2)[row.name] == 255.0):
        if abs(df['POSITION_Y'].shift(-1)[row.name] - row['POSITION_Y']) < 5:
            return 255
        else:
            return 127

    elif (row['caged'] == 255
            and df['caged'].shift(1)[row.name] == 127
            and df['caged'].shift(2)[row.name] == 127
            and df['caged'].shift(-1)[row.name] == 127
            and df['caged'].shift(-2)[row.name] == 127
        ) or (
            row['caged'] == 255
            and df['caged'].shift(1)[row.name] == 127.0
            and df['caged'].shift(-2)[row.name] == 127.0
            and df['caged'].shift(-1)[row.name] == 127.0
            and np.isnan(df['caged'].shift(2)[row.name])
         ) or (
            row['caged'] == 255
            and np.isnan(df['caged'].shift(1)[row.name])
            and df['caged'].shift(-1)[row.name] == 127.0):
        return 127

    elif np.isnan(df['caged'].shift(-1)[row.name]):
        if row['caged'] == 255 and df['caged'].shift(1)[row.name] == 127.0:
            return 127
        elif row['caged'] == 127 and df['caged'].shift(1)[row.name] == 255.0:
            return 255
        else:
            return row['caged']
    else:
        return row["caged"]

def post_process_caging1(row, df):
    if np.isnan(df['caged'].shift(1)[row.name]):
        if row['caged'] == 255 and df['caged'].shift(-1)[row.name] == 127.0:
            print(row["FRAME"], df['FRAME'].shift(-1)[row.name])
            print(df.head(6))
    return row["caged"]


def remove_outlier_tracks_and_spots(tracks, spots):
    filtered_tracks = tracks[
        (tracks['TRACK_DURATION'] >= 10) &
        (tracks['TOTAL_DISTANCE_TRAVELED'] >= 100) &
        (2030 >= tracks['TRACK_Y_LOCATION']) &
        (tracks['TRACK_Y_LOCATION'] >= 10) &
        (tracks['NUMBER_GAPS'] <= 3) &
        (tracks['NUMBER_SPLITS'] <= 1) &
        (tracks['NUMBER_MERGES'] <= 1) &
        (tracks['MAX_DISTANCE_TRAVELED'] >= 100)
        ]

    filtered_spot_ids = filtered_tracks['TRACK_ID'].unique()

    # Filter the spots DataFrame based on the 'label' column
    filtered_spots = spots[spots['label'].isin(filtered_spot_ids)]
    filtered_spots = filtered_spots.groupby('label').apply(lambda group: group.sort_values(by='FRAME')).reset_index(
        drop=True)
    for track in filtered_spots["label"].unique():
        track_mask = filtered_spots["label"] == track
        updated_values = filtered_spots[track_mask].apply(
            lambda row: post_process_caging(row, filtered_spots[track_mask]), axis=1)
        filtered_spots.loc[track_mask, 'caged'] = updated_values

    # for track in filtered_spots["label"].unique():
    #     track_mask = filtered_spots["label"] == track
    #     updated_values1 = filtered_spots[track_mask].apply(
    #         lambda row: post_process_caging1(row, filtered_spots[track_mask]), axis=1)
    #     filtered_spots.loc[track_mask, 'caged'] = updated_values1

    return filtered_tracks, filtered_spots


def spots_preprocessing(spots, image):
    df_no_nan = spots.dropna(subset=['label'])
    res = df_no_nan.copy()
    res.loc[:, 'caged'] = res.apply(add_caged_feature, axis=1, image=image)
    return res


def add_feature_to_xml(root, feature_name, feature_shortname, feature_dimension, feature_isint):
    spot_features_element = root.find(".//FeatureDeclarations/SpotFeatures")

    # Create a new Feature element
    feature_element = et.Element("Feature")
    feature_element.set("feature", feature_name)
    feature_element.set("name", feature_name)
    feature_element.set("shortname", feature_shortname)
    feature_element.set("dimension", feature_dimension)
    feature_element.set("isint", str(feature_isint).lower())  # Convert boolean to lowercas

    spot_features_element.append(feature_element)


def process_trackmate_xml(trackmate_file):
    spots_dataframe, tracks = read_trackmate_xml(trackmate_file, get_tracks=True)
    tree = et.parse(trackmate_file)
    root = tree.getroot()
    image_data_element = root.find(".//Settings/ImageData")

    # Extract filename and folder attributes
    filename = image_data_element.get("filename")
    folder = image_data_element.get("folder").replace("segmentations", "classifications")
    image = io.imread(folder + filename)
    caged_dataframe = spots_preprocessing(spots_dataframe, image)
    add_feature_to_xml(root, 'caged', 'Caged', 'NONE', True)

    final_tracks, final_spots = remove_outlier_tracks_and_spots(tracks, caged_dataframe)

    track_filters = root.findall(".//FilteredTracks/TrackID")
    spot_ids_to_filter = final_spots['ID'].tolist()
    track_ids_to_filter = final_tracks['TRACK_ID'].tolist()

    # Filter spots, tracks, and edges
    filtered_track_ids = [track for track in track_filters if float(track.get('TRACK_ID')) not in track_ids_to_filter]

    for allspot in root.findall('.//AllSpots'):
        for spots_in_frame in allspot.findall('.//SpotsInFrame'):
            for spot in spots_in_frame.findall('.//Spot'):
                if float(spot.get('ID')) not in spot_ids_to_filter:
                    spots_in_frame.remove(spot)

    for all_tracks in root.findall('.//AllTracks'):
        for track in all_tracks.findall('.//Track'):
            if float(track.get('TRACK_ID')) not in track_ids_to_filter:
                all_tracks.remove(track)

    for all_tracks in root.findall(".//FilteredTracks"):
        for track in filtered_track_ids:
            all_tracks.remove(track)

    image_data_element = root.find('.//Settings/ImageData')
    spots_count = root.find(".//AllSpots")
    spots_count.set('nspots', str(len(spot_ids_to_filter)))

    if image_data_element is not None:
        image_data_element.set('folder', image_data_element.get('folder').replace("segmentations", "preprocessed"))

    for index, row in final_spots.iterrows():
        spot_id = int(row['ID'])
        caged_value = row['caged']
        spot_element = root.find(f".//Spot[@ID='{spot_id}']")
        spot_element.set('caged', str(caged_value))

    # Save the updated XML file
    tree.write(str(trackmate_file).replace("exportModel", "filtered"))
