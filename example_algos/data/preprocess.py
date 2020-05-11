import argparse
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm


def nifti_to_numpy(input_folder: str, output_folder: str):
    """Converts all nifti files in a input folder to numpy and saves the data and affine matrix into the output folder

    Args:
        input_folder (str): Folder to read the nifti files from
        output_folder (str): Folder to write the numpy arrays to
    """

    for fname in tqdm(sorted(os.listdir(input_folder))):

        if not fname.endswith("nii.gz"):
            continue

        n_file = os.path.join(input_folder, fname)
        nifti = nib.load(n_file)

        np_data = nifti.get_fdata()
        np_affine = nifti.affine

        f_basename = fname.split(".")[0]

        np.save(os.path.join(output_folder, f_basename + "_data.npy"), np_data.astype(np.float16))
        np.save(os.path.join(output_folder, f_basename + "_aff.npy"), np_affine)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=False, type=str)

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    nifti_to_numpy(input_dir, output_dir)
