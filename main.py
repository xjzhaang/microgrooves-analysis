import argparse
import numpy as np
from pathlib import Path

import torch
from skimage import io
from preprocessing.preprocess import preprocess_image
from segmentation.segment import segment_cells
from classification.classify import classify_volume, plot_sensitivity, sensitivities_from_paths
from tracking.init_fiji import track
from tracking.trackmate_utils import process_trackmate_xml
from analysis.analysis_utils import read_and_analyze


def preprocess(directory_path, filter_grooves=True):
    preprocessed_directory = Path('./output') / directory_path.relative_to('../data') / 'preprocessed'
    preprocessed_directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory_path.glob('*.tif'):
        destination_path = preprocessed_directory / file_path.relative_to(directory_path)
        preprocessed_image = preprocess_image(file_path,
                                              model_path="./preprocessing/grooves_segmentation/models/best.pth",
                                              keep_grooves=True, filter_grooves=filter_grooves, filter_with_model=True)
        if preprocessed_image.shape[1] == 2 or preprocessed_image.ndim == 3:
            io.imsave(destination_path, preprocessed_image, imagej=True, check_contrast=False)
        else:
            groove_seg_and_nuclei = preprocessed_image[:, -2:, :, :][:, ::-1, :, :]
            io.imsave(destination_path, groove_seg_and_nuclei, imagej=True, check_contrast=False)
            groove_path = destination_path.with_name(destination_path.stem + '_grooves').with_suffix('.tif')
            io.imsave(groove_path, preprocessed_image[:, 0, :, :], imagej=True, check_contrast=False)
        del preprocessed_image
        print(f"Processed {file_path} and saved to {destination_path}")


def segment(directory_path):
    preprocessed_directory = Path('./output') / directory_path.relative_to('../data') / 'preprocessed'
    volume_list = list(preprocessed_directory.glob('*[!_grooves].tif'))
    segmentation_directory = Path('./output') / directory_path.relative_to('../data') / 'segmentations'
    segmentation_directory.mkdir(exist_ok=True)
    model_type = "myoblast" if "myoblastes" in str(segmentation_directory).lower() else "endothelial"
    model_path = str(list(Path(f"./segmentation/cellpose_segmentation/{model_type}/models/").glob("CP*"))[0])
    print(model_path)
    for i in range(len(volume_list)):
        res_image = segment_cells(volume_list[i], model_path=model_path)
        io.imsave(segmentation_directory / volume_list[i].name, res_image, imagej=True, check_contrast=False)
        torch.cuda.empty_cache()
        print(f"Segmented {volume_list[i].name} and saved to {segmentation_directory / volume_list[i].name}")


def classify(directory_path, filter_grooves=True):
    segmentation_directory = Path('./output') / directory_path.relative_to('../data') / 'segmentations'
    classification_directory = Path('./output') / directory_path.relative_to('../data') / 'classifications'
    classification_directory.mkdir(exist_ok=True)
    volume_list = list(segmentation_directory.glob('*[!_grooves].tif'))
    for volume in volume_list:
        final_vol = classify_volume(volume, str(volume).replace("segmentations", "preprocessed"),
                                    filter_grooves=filter_grooves)
        io.imsave(classification_directory / volume.name, final_vol, imagej=True, check_contrast=False)
        print(f"Classified {volume.name} and saved to {classification_directory / volume.name}")
        del final_vol


def process_xml(directory_path):
    trackings_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'trackings'
    xml_list = list(trackings_directory.glob('*exportModel.xml'))
    for xml_file in xml_list:
        process_trackmate_xml(xml_file, )
        print(f"Filtered {xml_file}!")


def measure_sensitivity(directory_path, filter_grooves=True, remove_one=False):
    segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
    if "myoblastes" in str(segmentation_directory).lower():
        subtypes = ['WT', 'Mut']
    else:
        subtypes = ['5h1', '5h5', '5h4,5', 'HD', 'LD', 'merged']
    for subtype in subtypes:
        plot_path = f'remove_one_criterion_{subtype}.png' if remove_one else f'caging_sensitivity_{subtype}.png'
        save_path = (Path(str(segmentation_directory).replace("segmentations", "classifications"))
                     / plot_path)
        volume_list = [path for path in segmentation_directory.glob(f'**/*{subtype}*') if 'Flat' not in str(path)]
        if not volume_list:
            pass
        else:
            full_dict, param_dict, mean_std_dict = sensitivities_from_paths(volume_list, filter_grooves, remove_one)
            plot_sensitivity(save_path, param_dict, full_dict, mean_std_dict, remove_one)
            print(f"Saved {save_path}!")


def analyze_xml(directory_path):
    trackings_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'trackings'
    analysis_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'analysis'
    analysis_directory.mkdir(exist_ok=True)
    xml_list = list(trackings_directory.glob('*filtered.xml'))
    for xml_file in xml_list:
        read_and_analyze(xml_file)
        print(f"Analyzed {xml_file} and saved to csv!")

def select_segmentation_samples(directory_path):
    preprocessed_directory = Path('./output') / directory_path.relative_to('../data') / 'preprocessed'
    volume_list = list(preprocessed_directory.glob('*[!_grooves].tif'))

    selected_volumes = volume_list

    # Create destination folder if it doesn't exist
    destination_folder_path = Path("./segmentation/cellpose_segmentation/myoblast")
    destination_folder_path.mkdir(parents=True, exist_ok=True)

    # Select 10 evenly spaced slices from each volume
    for volume_path in selected_volumes:
        volume = io.imread(volume_path)

        # Determine the number of slices (t)
        num_slices = volume.shape[0]
        num_channel = len(volume.shape)

        # Select 10 evenly spaced indices
        selected_slices_indices = np.linspace(0, num_slices - 1, num=3, dtype=int)

        # Save the selected slices
        for idx in selected_slices_indices:
            # Construct the destination filename
            filename = f"{volume_path.parent.parent.parent.name}_{volume_path.parent.parent.name}_{volume_path.stem}_slice{idx}.tif"
            destination_path = destination_folder_path / filename

            # Save the slice as an image (assuming grayscale)
            if num_channel > 3:
                slice_to_save = volume[idx, 1, :, :]
            else:
                slice_to_save = volume[idx, :, :]

            io.imsave(destination_path, slice_to_save)

    print(f"Saved 10 slices from each of the {len(selected_volumes)} volumes to {destination_folder_path}")


def main():
    parser = argparse.ArgumentParser(description='Cell analysis.')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing')
    parser.add_argument('--grooves', action='store_true', help='Segment groves or not')
    parser.add_argument('--segment', action='store_true', help='Enable segmentation')
    parser.add_argument('--classify', action='store_true', help='Enable classification')
    parser.add_argument('--track', action='store_true', help='Enable Tracking with Fiji')
    parser.add_argument('--filter_xml', action='store_true', help='Filtered xml tracks')
    parser.add_argument('--analyze', action='store_true', help='Analyze xml tracks')
    parser.add_argument('--sens', action='store_true', help='Plot caging percentages as functions of criteria')
    parser.add_argument('-fiji', type=str, default="/home/z/Fiji.app", help='Fiji.app path')
    parser.add_argument('-d', type=str, default=None, help='directory')

    args = parser.parse_args()

    if args.d is None:
        directories = {
            # # '../data/preprocessing_test': True,
            '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J1 6h culture': True,
            '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J2 24h culture': True,
            '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J1 6h culture': True,
            '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J2 24h culture': True,
            '../data/220208 Film myoblastes WT K32 5-5-5/J1 6h culture': True,
            '../data/220208 Film myoblastes WT K32 5-5-5/J2 24h culture': True,
            '../data/200121 hoechst': True,
            '../data/191219 HOECHST 3D 5-5': True,
            '../data/191015 HOECHST 3D 5-5': True,
            '../data/200911 Diff density and BB': True,
            '../data/200916 Diff densities h4,5': True,
            '../data/210910 Grooves dapi 48h': False,
            '../data/231003 HUVEC grooves h5,2 CellMask Dapi': False,
            '../data/231005 Grooves h5,4 CellMask Dapi': False,
        }
    else:
        directories = {args.d: args.grooves}
    full_list = []
    for directory, filter_grooves in directories.items():
        directory_path = Path(directory)
        if args.preprocess:
            print(f"Begin preprocessing {directory_path.name}")
            preprocess(directory_path, filter_grooves=filter_grooves)
        if args.segment:
            print(f"Begin segmenting {directory_path.name}")
            segment(directory_path)
        if args.classify:
            print(f"Begin classifying {directory_path.name}")
            classify(directory_path, filter_grooves=filter_grooves)
        if args.track:
            trackings_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'trackings'
            trackings_directory.mkdir(exist_ok=True)
            segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
            volume_list = list(segmentation_directory.glob('*[!_grooves].tif'))
            full_list.extend(volume_list)
        if args.sens:
            print(f"Begin plotting {directory_path.name}")
            measure_sensitivity(directory_path, filter_grooves, remove_one=False)
            measure_sensitivity(directory_path, filter_grooves, remove_one=True)

    abs_paths = [path.resolve() for path in full_list]
    if args.track:
        print(f"Begin tracking")
        track(abs_paths, fiji_path=args.fiji)
        print("Tracking finished")

    if args.filter_xml:
        for directory in directories.keys():
            directory_path = Path(directory)
            process_xml(directory_path)

    if args.analyze:
        for directory in directories.keys():
            directory_path = Path(directory)
            analyze_xml(directory_path)

    # for directory in directories.keys():
    #     directory_path = Path(directory)
    #     select_segmentation_samples(directory_path)

if __name__ == "__main__":
    main()
