import os

import nibabel as nib
import numpy as np


def predict_folder_pixel_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):

        source_file = os.path.join(input_folder, f)
        target_file = os.path.join(target_folder, f)

        nimg = nib.load(source_file)
        nimg_array = nimg.get_fdata()

        nimg_array[nimg_array < 0.01] = 0.5

        abnomal_score_array = np.abs(nimg_array - 0.5)

        final_nimg = nib.Nifti1Image(abnomal_score_array, affine=nimg.affine)
        nib.save(final_nimg, target_file)


def predict_folder_sample_abs(input_folder, target_folder):
    for f in os.listdir(input_folder):
        abnomal_score = np.random.rand()

        with open(os.path.join(target_folder, f + ".txt"), "w") as write_file:
            write_file.write(str(abnomal_score))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode

    if mode == "pixel":
        predict_folder_pixel_abs(input_dir, output_dir)
    elif mode == "sample":
        predict_folder_sample_abs(input_dir, output_dir)
    else:
        print("Mode not correctly defined. Either choose 'pixel' oder 'sample'")

    # predict_folder_sample_abs("/home/david/data/datasets_slow/mood_brain/toy", "/home/david/data/datasets_slow/mood_brain/target_sample")
