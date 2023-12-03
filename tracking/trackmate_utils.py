# Code for function read_trackmate_xml adapted from https://github.com/hadim/pytrackmate
import numpy as np

import xml.etree.ElementTree as et
import pandas as pd
from skimage import io


def read_trackmate_xml(trackmate_xml_path, get_tracks=False):
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
    #
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

                trajs.loc[trajs["ID"].isin(spot_ids), "label"] = label_id
                label_id += 1

    return trajs, tracks_


def add_caged_feature(row, image):
    x, y, z = row['POSITION_X'], row['POSITION_Y'], row['POSITION_T']
    pixel_value = image[int(z), int(y), int(x)]
    if pixel_value == 0:
        pixel_value = 127
    return pixel_value


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
    dataframe, _ = read_trackmate_xml(trackmate_file)
    tree = et.parse(trackmate_file)
    root = tree.getroot()
    image_data_element = root.find(".//Settings/ImageData")

    # Extract filename and folder attributes
    filename = image_data_element.get("filename")
    folder = image_data_element.get("folder").replace("segmentations", "classifications")
    image = io.imread(folder + filename)
    caged_dataframe = spots_preprocessing(dataframe, image)
    add_feature_to_xml(root, 'caged', 'Caged', 'NONE', True)

    for index, row in caged_dataframe.iterrows():
        spot_id = int(row['ID'])
        caged_value = row['caged']

        # Find the <Spot> element with the matching ID
        spot_element = root.find(f".//Spot[@ID='{spot_id}']")

        # Add/update the 'caged' attribute in the <Spot> element
        spot_element.set('caged', str(caged_value))

    # Save the updated XML file
    tree.write(str(trackmate_file).replace("exportModel", "filtered"))
