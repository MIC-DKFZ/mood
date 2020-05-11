

def ni_load(f_path):
    """Loads a nifti file from a given path and returns the data and affine matrix

    Args:
        f_path ([str]): [Path to the nifti file]

    Returns:
        data [np.ndarray]: [Nifti file image data]
        affine [np.ndarray]: [Nifti file affine matrix]
    """
    import nibabel

    nimg = nibabel.load(f_path)
    return nimg.get_fdata(), nimg.affine


def ni_save(f_path, ni_data, ni_affine):
    """Saves image data and a affine matrix as a new nifti file

    Args:
        f_path ([str]): [Path to the nifti file]
        ni_data ([np.ndarray]): [Image data]
        ni_affine ([type]): [Affine matrix]
    """
    import nibabel

    nimg = nibabel.Nifti1Image(ni_data, ni_affine)
    nibabel.save(nimg, f_path)
