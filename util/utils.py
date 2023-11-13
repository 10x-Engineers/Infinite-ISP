"""File: utils.py
Description: Common helper functions for all algorithms
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import os
from datetime import datetime
import random
import warnings
from pathlib import Path
import numpy as np
import yaml
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from modules.demosaic.malvar_he_cutler import Malvar as MAL


# Infinite-ISP output directory
OUTPUT_DIR = "out_frames/"

# Output directory for Module output
OUTPUT_ARRAY_DIR = "./module_output/"


def introduce_defect(img, total_defective_pixels, padding):

    """
    This function randomly replaces pixels values with extremely high or low
    pixel values to create dead pixels (Dps).
    Note that the defective pixel values are never introduced on the periphery
    of the image to ensure that there are no adjacent DPs.
    Parameters
    ----------
    img: 2D ndarray
    total_defective_pixels: number of Dps to introduce in img.
    padding: bool value (set to True to add padding)
    Returns
    -------
    defective image: image/padded img containing specified (by TOTAL_DEFECTIVE_PIXELS)
    number of dead pixels.
    orig_val: ndarray of size same as img/padded img containing original pixel values
    in place of introduced DPs and zero elsewhere.
    """

    if padding:
        padded_img = np.pad(img, ((2, 2), (2, 2)), "reflect")
    else:
        padded_img = img.copy()

    orig_val = np.zeros((padded_img.shape[0], padded_img.shape[1]))

    while total_defective_pixels:
        defect = [
            random.randrange(1, 15),
            random.randrange(4081, 4095),
        ]  # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0, 1)]
        random_row, random_col = random.randint(2, img.shape[0] - 3), random.randint(
            2, img.shape[1] - 3
        )
        left, right = (
            orig_val[random_row, random_col - 2],
            orig_val[random_row, random_col + 2],
        )
        top, bottom = (
            orig_val[random_row - 2, random_col],
            orig_val[random_row + 2, random_col],
        )
        neighbours = [left, right, top, bottom]

        if (
            not any(neighbours) and orig_val[random_row, random_col] == 0
        ):  # if all neighbouring values in orig_val are 0 and the pixel itself is not defective
            orig_val[random_row, random_col] = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels -= 1

    return padded_img, orig_val


def gauss_kern_raw(size, std_dev, stride):
    """
    This function takes in size, standard deviation and spatial stride required for adjacet
    weights to output a gaussian kernel of size NxN
    Parameters
    ----------
    size:   size of gaussian kernel, odd
    std_dev: standard deviation of the gaussian kernel
    stride: spatial stride between to be considered for adjacent gaussian weights
    Returns
    -------
    outKern: an output gaussian kernel of size NxN
    """

    if size % 2 == 0:
        warnings.warn("kernel size (N) cannot be even, setting it as odd value")
        size = size + 1

    if size <= 0:
        warnings.warn("kernel size (N) cannot be <= zero, setting it as 3")
        size = 3

    out_kern = np.zeros((size, size), dtype=np.float32)

    for i in range(0, size):
        for j in range(0, size):
            out_kern[i, j] = np.exp(
                -1
                * (
                    (stride * (i - ((size - 1) / 2))) ** 2
                    + (stride * (j - ((size - 1) / 2))) ** 2
                )
                / (2 * (std_dev**2))
            )

    sum_kern = np.sum(out_kern)
    out_kern[0:size:1, 0:size:1] = out_kern[0:size:1, 0:size:1] / sum_kern

    return out_kern


def crop(img, rows_to_crop=0, cols_to_crop=0):

    """
    Crop 2D array.
    Parameter:
    ---------
    img: image (2D array) to be cropped.
    rows_to_crop: Number of rows to crop. If it is an even integer,
                    equal number of rows are cropped from either side of the image.
                    Otherwise the image is cropped from the extreme right/bottom.
    cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
    Output: cropped image
    """

    if rows_to_crop:
        if rows_to_crop % 2 == 0:
            img = img[rows_to_crop // 2 : -rows_to_crop // 2, :]
        else:
            img = img[0:-1, :]
    if cols_to_crop:
        if cols_to_crop % 2 == 0:
            img = img[:, cols_to_crop // 2 : -cols_to_crop // 2]
        else:
            img = img[:, 0:-1]
    return img


def stride_convolve2d(matrix, kernel):
    """2D convolution"""
    return correlate2d(matrix, kernel, mode="valid")[
        :: kernel.shape[0], :: kernel.shape[1]
    ]


def display_ae_statistics(ae_feedback, awb_gains):
    """
    Print AE Stats for current frame
    """
    # Logs for AWB
    if awb_gains is None:
        print("   - 3A Stats    - AWB is Disable")
    else:
        print("   - 3A Stats    - AWB Rgain = ", awb_gains[0])
        print("   - 3A Stats    - AWB Bgain = ", awb_gains[1])

    # Logs for AE
    if ae_feedback is None:
        print("   - 3A Stats    - AE is Disable")
    else:
        if ae_feedback < 0:
            print("   - 3A Stats    - AE Feedback = Underexposed")
        elif ae_feedback > 0:
            print("   - 3A Stats    - AE Feedback = Overexposed")
        else:
            print("   - 3A Stats    - AE Feedback = Correct Exposure")


def reconstruct_yuv_from_422_custom(yuv_422_custom, width, height):
    """
    Reconstruct a YUV from YUV 422 format
    """
    # Create an empty 3D YUV image (height, width, channels)
    yuv_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Rearrange the flattened 4:2:2 YUV data back to 3D YUV format
    yuv_img[:, 0::2, 0] = yuv_422_custom[0::4].reshape(height, -1)
    yuv_img[:, 0::2, 1] = yuv_422_custom[1::4].reshape(height, -1)
    yuv_img[:, 1::2, 0] = yuv_422_custom[2::4].reshape(height, -1)
    yuv_img[:, 0::2, 2] = yuv_422_custom[3::4].reshape(height, -1)

    # Replicate the U and V (chroma) channels to the odd columns
    yuv_img[:, 1::2, 1] = yuv_img[:, 0::2, 1]
    yuv_img[:, 1::2, 2] = yuv_img[:, 0::2, 2]

    return yuv_img


def reconstruct_yuv_from_444_custom(yuv_444_custom, width, height):
    """
    Reconstruct a YUV from YUV 444 format
    """
    # Create an empty 3D YUV image (height, width, channels)
    yuv_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Rearrange the flattened 4:2:2 YUV data back to 3D YUV format
    yuv_img[:, 0::1, 0] = yuv_444_custom[0::3].reshape(height, -1)
    yuv_img[:, 0::1, 1] = yuv_444_custom[1::3].reshape(height, -1)
    yuv_img[:, 0::1, 2] = yuv_444_custom[2::3].reshape(height, -1)

    return yuv_img


def get_image_from_yuv_format_conversion(yuv_img, height, width, yuv_custom_format):
    """
    Convert YUV image into RGB based on its format & Conversion Standard
    """

    # Reconstruct the 3D YUV image from the custom given format YUV data
    if yuv_custom_format == "422":
        yuv_img = reconstruct_yuv_from_422_custom(yuv_img, width, height)
    else:
        yuv_img = reconstruct_yuv_from_444_custom(yuv_img, width, height)

    return yuv_img


def save_pipeline_output(img_name, output_img, config_file):
    """
    Saves the output image (png) and config file in OUTPUT_DIR
    """

    # Time Stamp for output filename
    dt_string = datetime.now().strftime("_%Y%m%d_%H%M%S")

    # Set list format to flowstyle to dump yaml file
    yaml.add_representer(list, represent_list)

    # Storing configuration file for output image
    with open(
        OUTPUT_DIR + img_name + dt_string + ".yaml", "w", encoding="utf-8"
    ) as file:
        yaml.dump(
            config_file,
            file,
            sort_keys=False,
            Dumper=CustomDumper,
            width=17000,
        )

    # Save Image as .png
    plt.imsave(OUTPUT_DIR + img_name + dt_string + ".png", output_img)


# utilities to save the config_automate exactly as config.yml
class CustomDumper(yaml.Dumper):

    """This class is a custom YAML dumper that overrides the default behavior
    of the increase_indent and write_line_break methods. It ensures that indentations
    and line breaks are inserted correctly in the output YAML file."""

    def increase_indent(self, flow=False, indentless=False):
        """For indentation"""
        return super(CustomDumper, self).increase_indent(flow, False)

    def write_line_break(self, data=None):
        """For line break"""
        super().write_line_break(data)
        if len(self.indents) == 1:
            self.stream.write("\n")


def represent_list(self, data):
    """This function ensures that the lookup table are stored on flow style
    to keep the saved yaml file readable."""
    return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def save_output_array(
    img_name, output_array, module_name, platform, bitdepth, bayer_pattern
):
    """
    Saves output array [raw/rgb] for pipline modules
    """

    # create directory to save array
    if not os.path.exists(OUTPUT_ARRAY_DIR):
        Path(OUTPUT_ARRAY_DIR).mkdir(parents=True, exist_ok=False)

    # filename identifies input image and isp pipeline module for which testing
    # vector is generated
    filename = OUTPUT_ARRAY_DIR + module_name + img_name.split(".")[0]

    if platform["save_format"] == "npy" or platform["save_format"] == "both":

        # save image as npy array
        np.save(filename, output_array.astype("uint16"))

    if platform["save_format"] == "png" or platform["save_format"] == "both":

        # for 1-channel raw: convert raw image to 8-bit rgb image
        if len(output_array.shape) == 2:
            output_array = apply_cfa(output_array, bitdepth, bayer_pattern)

        # convert image to 8-bit image if required
        if output_array.dtype != np.uint8:
            shift_by = bitdepth - 8
            output_array = (output_array >> shift_by).astype("uint8")

        # save Image as .png
        plt.imsave(filename + ".png", output_array)


def save_output_array_yuv(img_name, output_array, module_name, platform, conv_std):

    """
    Saves output array [yuv] for pipline modules
    """
    # create directory to save array
    if not os.path.exists(OUTPUT_ARRAY_DIR):
        Path(OUTPUT_ARRAY_DIR).mkdir(parents=True, exist_ok=False)

    # filename identifies input image and isp pipeline module for which testing
    # vector is generated
    filename = OUTPUT_ARRAY_DIR + module_name + img_name.split(".")[0]

    # save image as .npy array
    if platform["save_format"] == "npy" or platform["save_format"] == "both":
        np.save(filename, output_array.astype("uint16"))

    # save image as .png
    if platform["save_format"] == "png" or platform["save_format"] == "both":
        # convert the yuv image to RGB image
        output_array = yuv_to_rgb(output_array, conv_std)
        plt.imsave(filename + ".png", output_array)


def yuv_to_rgb(yuv_img, conv_std):
    """
    YUV-to-RGB Colorspace conversion 8bit
    """

    # make nx3 2d matrix of image
    mat_2d = yuv_img.reshape((yuv_img.shape[0] * yuv_img.shape[1], 3))

    # convert to 3xn for matrix multiplication
    mat2d_t = mat_2d.transpose()

    # subract the offsets
    mat2d_t = mat2d_t - np.array([[16, 128, 128]]).transpose()

    if conv_std == 1:
        # for BT. 709
        yuv2rgb_mat = np.array([[74, 0, 114], [74, -13, -34], [74, 135, 0]])
    else:
        # for BT.601/407
        # conversion metrix with 8bit integer co-efficients - m=8
        yuv2rgb_mat = np.array([[64, 87, 0], [64, -44, -20], [61, 0, 105]])

    # convert to RGB
    rgb_2d = np.matmul(yuv2rgb_mat, mat2d_t)
    rgb_2d = rgb_2d >> 6

    # reshape the image back
    rgb2d_t = rgb_2d.transpose()
    yuv_img = rgb2d_t.reshape(yuv_img.shape).astype(np.float32)

    # clip the resultant img as it can have neg rgb values for small Y'
    yuv_img = np.float32(np.clip(yuv_img, 0, 255))

    # convert the image to [0-255]
    yuv_img = np.uint8(yuv_img)
    return yuv_img


def masks_cfa_bayer(img, bayer):
    """
    Generating masks for the given bayer pattern
    """

    # dict will be creating 3 channel boolean type array of given shape with the name
    # tag like 'r_channel': [False False ....] , 'g_channel': [False False ....] ,
    # 'b_channel': [False False ....]
    channels = dict((channel, np.zeros(img.shape, dtype=bool)) for channel in "rgb")

    # Following comment will create boolean masks for each channel r_channel,
    # g_channel and b_channel
    for channel, (y_channel, x_channel) in zip(bayer, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y_channel::2, x_channel::2] = True

    # tuple will return 3 channel boolean pattern for r_channel,
    # g_channel and b_channel with True at corresponding value
    # For example in rggb pattern, the r_channel mask would then be
    # [ [ True, False, True, False], [ False, False, False, False]]
    return tuple(channels[c] for c in "rgb")


def apply_cfa(img, bit_depth, bayer):
    """
    Demosaicing the given raw image using given algorithm
    """
    # 3D masks according to the given bayer
    masks = masks_cfa_bayer(img, bayer)
    mal = MAL(img, masks)
    demos_out = mal.apply_malvar()

    # Clipping the pixels values within the bit range
    demos_out = np.clip(demos_out, 0, 2**bit_depth - 1)
    demos_out = np.uint16(demos_out)
    return demos_out
