import argparse
from pathlib import Path

import torch
from skimage import io, util
from preprocessing.preprocess import preprocess_image
from segmentation.segment import segment_cells
from classification.classify import classify_volume

def preprocess(directory_path):
    preprocessed_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'preprocessed'
    preprocessed_directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory_path.glob('*.tif'):
        destination_path = preprocessed_directory / file_path.relative_to(directory_path)
        preprocessed_image = preprocess_image(file_path, model_path="./preprocessing/grooves_segmentation/models/best.pth", keep_grooves=True, filter_with_model=True)
        if preprocessed_image.shape[1] == 2:
            io.imsave(destination_path, preprocessed_image, imagej=True, check_contrast=False)
        else:
            groove_seg_and_nuclei = preprocessed_image[:, -2:, :, :][:, ::-1, :, :]
            io.imsave(destination_path, util.img_as_ubyte(groove_seg_and_nuclei), imagej=True, check_contrast=False)
            groove_path = destination_path.with_name(destination_path.stem + '_grooves').with_suffix('.tif')
            io.imsave(groove_path, util.img_as_ubyte(preprocessed_image[:, 0, :, :]), imagej=True, check_contrast=False)
        del preprocessed_image
        print(f"Processed {file_path} and saved to {destination_path}")


def segment(directory_path):
    preprocessed_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'preprocessed'
    volume_list = list(preprocessed_directory.glob('*[!_grooves].tif'))
    segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
    segmentation_directory.mkdir(exist_ok=True)
    for i in range(len(volume_list)):
        res_image = segment_cells(volume_list[i], model_path="./segmentation/cellpose_segmentation/data/models/CP_20231126_192858")
        io.imsave(segmentation_directory / volume_list[i].name, util.img_as_ubyte(res_image), imagej=True, check_contrast=False)
        torch.cuda.empty_cache()
        print(f"Segmented {volume_list[i].name} and saved to {segmentation_directory / volume_list[i].name}")


def classify(directory_path):
    segmentation_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'segmentations'
    classification_directory = Path('./output') / directory_path.relative_to(Path('../data')) / 'classifications'
    classification_directory.mkdir(exist_ok=True)
    volume_list = list(segmentation_directory.glob('*[!_grooves].tif'))
    for volume in volume_list:
        final_vol = classify_volume(volume)
        io.imsave(classification_directory / volume.name, util.img_as_ubyte(final_vol), imagej=True, check_contrast=False)
        print(f"Segmented {volume.name} and saved to {classification_directory / volume.name}")
        del final_vol


def main():
    parser = argparse.ArgumentParser(description='Cell analysis.')
    parser.add_argument('--preprocess', action='store_true', help='Enable preprocessing')
    parser.add_argument('--segment', action='store_true', help='Enable segmentation')
    parser.add_argument('--classify', action='store_true', help='Enable classification')
    parser.add_argument('--d', type=str, default=None, help='directory')

    args = parser.parse_args()

    if args.d is None:
        directories = [ '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J1 6h culture',
                        '../data/220127 Film Myoblastes WT - K32 tranchees 5-5-5/J2 24h culture',
                        '../data/200121 hoechst',
                        '../data/191219 HOECHST 3D 5-5',
                        '../data/191015 HOECHST 3D 5-5',
                        '../data/200911 Diff density and BB',
                        '../data/200916 Diff densities h4,5',
                        ]
    else:
        directories = [args.d]

    for directory in directories:
        directory_path = Path(directory)
        if args.preprocess:
            print(f"Begin preprocessing {directory_path.name}")
            preprocess(directory_path)
        if args.segment:
            print(f"Begin segmenting {directory_path.name}")
            segment(directory_path)
        if args.classify:
            print(f"Begin classifying {directory_path.name}")
            classify(directory_path)


if __name__ == "__main__":
    main()