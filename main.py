import argparse
from pathlib import Path

import torch
from skimage import io
from preprocessing.preprocess import preprocess_image
from segmentation.segment import segment_cells
from classification.classify import classify_volume


def preprocess(directory_path, filter_grooves=True):
    preprocessed_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'preprocessed'
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
    preprocessed_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'preprocessed'
    volume_list = list(preprocessed_directory.glob('*[!_grooves].tif'))
    segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
    segmentation_directory.mkdir(exist_ok=True)
    for i in range(len(volume_list)):
        res_image = segment_cells(volume_list[i], model_path="./segmentation/cellpose_segmentation/data/models/CP_20231126_192858")
        io.imsave(segmentation_directory / volume_list[i].name, res_image, imagej=True, check_contrast=False)
        torch.cuda.empty_cache()
        print(f"Segmented {volume_list[i].name} and saved to {segmentation_directory / volume_list[i].name}")


def classify(directory_path, filter_grooves=True):
    segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
    classification_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'classifications'
    classification_directory.mkdir(exist_ok=True)
    volume_list = list(segmentation_directory.glob('*[!_grooves].tif'))
    for volume in volume_list:
        final_vol = classify_volume(volume, str(volume).replace("segmentations", "preprocessed"), filter_grooves=filter_grooves)
        io.imsave(classification_directory / volume.name, final_vol, imagej=True, check_contrast=False)
        print(f"Classified {volume.name} and saved to {classification_directory / volume.name}")
        del final_vol


def main():
    parser = argparse.ArgumentParser(description='Cell analysis.')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing')
    parser.add_argument('--grooves', action='store_true', help='Segment groves or not')
    parser.add_argument('--segment', action='store_true', help='Enable segmentation')
    parser.add_argument('--classify', action='store_true', help='Enable classification')
    parser.add_argument('--d', type=str, default=None, help='directory')

    args = parser.parse_args()

    if args.d is None:
        directories = {
                        # '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J1 6h culture': True,
                        # '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J2 24h culture': True,
                        '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J1 6h culture': True,
                        # '../data/220202 Film Myoblastes WT-K32 Tr 5-5-5/J2 24h culture': True,
                        # # '../data/200121 hoechst': True,
                        # # '../data/191219 HOECHST 3D 5-5': True,
                        # # '../data/191015 HOECHST 3D 5-5': True,
                        # '../data/200911 Diff density and BB': True,
                        # '../data/200916 Diff densities h4,5': True,
                        # '../data/210910 Grooves dapi 48h': False,
                        #'../data/231003 HUVEC grooves h5,2 CellMask Dapi': False,
        }
    else:
        directories = {args.d: args.grooves}

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


if __name__ == "__main__":
    main()
