import argparse
from pathlib import Path

import torch
from skimage import io
from preprocessing.preprocess import preprocess_image
from segmentation.segment import segment_cells
from classification.classify import classify_volume, plot_sensitivity, sensitivities_from_paths
from tracking.init_fiji import track
from tracking.trackmate_utils import process_trackmate_xml



def preprocess(directory_path, filter_grooves=True):
    preprocessed_directory = Path('./output') / directory_path.relative_to('../data') / 'preprocessed'
    preprocessed_directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory_path.glob('*.tif'):
        destination_path = preprocessed_directory / file_path.relative_to(directory_path)
        preprocessed_image = preprocess_image(file_path, model_path="./preprocessing/grooves_segmentation/models/best.pth", keep_grooves=True, filter_grooves=filter_grooves, filter_with_model=True)
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
    for i in range(len(volume_list)):
        res_image = segment_cells(volume_list[i], model_path="./segmentation/cellpose_segmentation/data/models/CP_20231205_001809")
        io.imsave(segmentation_directory / volume_list[i].name, res_image, imagej=True, check_contrast=False)
        torch.cuda.empty_cache()
        print(f"Segmented {volume_list[i].name} and saved to {segmentation_directory / volume_list[i].name}")


def classify(directory_path, filter_grooves=True):
    segmentation_directory = Path('./output') / directory_path.relative_to('../data') / 'segmentations'
    classification_directory = Path('./output') / directory_path.relative_to('../data') / 'classifications'
    classification_directory.mkdir(exist_ok=True)
    volume_list = list(segmentation_directory.glob('*[!_grooves].tif'))
    for volume in volume_list:
        final_vol = classify_volume(volume, str(volume).replace("segmentations", "preprocessed"), filter_grooves=filter_grooves)
        io.imsave(classification_directory / volume.name, final_vol, imagej=True, check_contrast=False)
        print(f"Classified {volume.name} and saved to {classification_directory / volume.name}")
        del final_vol


def process_xml(directory_path):
    trackings_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'trackings'
    xml_list = list(trackings_directory.glob('*exportModel.xml'))
    for xml_file in xml_list:
        process_trackmate_xml(xml_file)
        print(f"Filtered {xml_file}!")


def main():
    parser = argparse.ArgumentParser(description='Cell analysis.')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing')
    parser.add_argument('--grooves', action='store_true', help='Segment groves or not')
    parser.add_argument('--segment', action='store_true', help='Enable segmentation')
    parser.add_argument('--classify', action='store_true', help='Enable classification')
    parser.add_argument('--track', action='store_true', help='Enable Tracking with Fiji')
    parser.add_argument('--filter_xml', action='store_true', help='Filtered xml tracks')
    parser.add_argument('--sens',action='store_true', help='Plot caging percentages as functions of criteria')
    parser.add_argument('-fiji', type=str, default="/home/z/Fiji.app", help='Fiji.app path')
    parser.add_argument('-d', type=str, default=None, help='directory')

    args = parser.parse_args()

    if args.d is None:
        directories = {
                        # '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J1 6h culture': True,
                        # '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J2 24h culture': True,
                        # '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J1 6h culture': True,
                        # '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J2 24h culture': True,
                        # '../data/200121 hoechst': True,
                        # '../data/191219 HOECHST 3D 5-5': True,
                        '../data/191015 HOECHST 3D 5-5': True,
                        # '../data/200911 Diff density and BB': True,
                        # '../data/200916 Diff densities h4,5': True,
                        # '../data/210910 Grooves dapi 48h': False,
                        # '../data/231003 HUVEC grooves h5,2 CellMask Dapi': False,
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

    abs_paths = [path.resolve() for path in full_list]
    if args.track:
        print(f"Begin tracking")
        track(abs_paths, fiji_path=args.fiji)
        print("Tracking finished")

    if args.filter_xml:
        for directory in directories.keys():
            directory_path = Path(directory)
            process_xml(directory_path)

    if args.sens:
        for directory in directories.keys():
            directory_path = Path(directory)
            segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
            if "Myoblastes" in directory:
                for subtype in ['WT', 'Mut']:
                    volume_list = list(segmentation_directory.glob(f'**/*{subtype}_*'))
                    full_dict, param_dict, mean_std_dict = sensitivities_from_paths(volume_list, directories[directory])
                    save_path = (Path(str(segmentation_directory).replace("segmentations", "classifications"))
                                 / f'caging_sensitivity_{subtype}.png')
                    plot_sensitivity(save_path, param_dict, mean_std_dict)
                    print(f"Saved {save_path}!")
            else:
                for subtype in ['5h1', '5h5', '5h4,5', 'HD', 'LD', 'merged']:
                    volume_list = list(segmentation_directory.glob(f'**/*{subtype}*'))
                    if not volume_list:
                        pass
                    else:
                        full_dict, param_dict, mean_std_dict = sensitivities_from_paths(volume_list, directories[directory])
                        save_path = (Path(str(segmentation_directory).replace("segmentations", "classifications"))
                                     / f'caging_sensitivity_{subtype}.png')
                        plot_sensitivity(save_path, param_dict, mean_std_dict)
                        print(f"Saved {save_path}!")


if __name__ == "__main__":
    main()
