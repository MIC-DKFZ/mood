import random

import numpy as np
import torch


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value


def get_square_mask(data_shape, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False, square_pos=None):
    """Returns a 'mask' with the same size as the data, where random squares are != 0

    Args:
        data_shape ([tensor]): [data_shape to determine the shape of the returned tensor]
        square_size ([tuple]): [int/ int tuple (min_size, max_size), determining the min and max squear size]
        n_squares ([type]): [int/ int tuple (min_number, max_number), determining the min and max number of squares]
        noise_val (tuple, optional): [int/ int tuple (min_val, max_val), determining the min and max value given in the 
                                        squares, which habe the value != 0 ]. Defaults to (0, 0).
        channel_wise_n_val (bool, optional): [Use a different value for each channel]. Defaults to False.
        square_pos ([type], optional): [Square position]. Defaults to None.
    """

    def mask_random_square(img_shape, square_size, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks (sets = 0) a random square in an image"""

        img_h = img_shape[-2]
        img_w = img_shape[-1]

        img = np.zeros(img_shape)

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        if img.ndim == 2:
            rnd_n_val = get_range_val(n_val)
            img[h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 3:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 4:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[:, i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, :, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val

        return img

    def mask_random_squares(img_shape, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks a given number of squares in an image"""
        img = np.zeros(img_shape)
        for i in range(n_squares):
            img = mask_random_square(
                img_shape, square_size, n_val, channel_wise_n_val=channel_wise_n_val, square_pos=square_pos
            )
        return img

    ret_data = np.zeros(data_shape)
    for sample_idx in range(data_shape[0]):
        # rnd_n_val = get_range_val(noise_val)
        rnd_square_size = get_range_val(square_size)
        rnd_n_squares = get_range_val(n_squares)

        ret_data[sample_idx] = mask_random_squares(
            data_shape[1:],
            square_size=rnd_square_size,
            n_squares=rnd_n_squares,
            n_val=noise_val,
            channel_wise_n_val=channel_wise_n_val,
            square_pos=square_pos,
        )

    return ret_data


def smooth_tensor(tensor, kernel_size=8, sigma=3, channels=1):

    # Set these to whatever you want for your gaussian filter

    if kernel_size % 2 == 0:
        kernel_size -= 1

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    import math

    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=kernel_size // 2,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    gaussian_filter.to(tensor.device)

    return gaussian_filter(tensor)


def normalize(tensor):

    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta
