"""
File: joint_bf.py
Description: Noise reduction in bayer domain uding joint bilateral filter
Code / Paper  Reference:
https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image
Author: 10xEngineers
------------------------------------------------------------
"""

import warnings
import numpy as np
from scipy import ndimage
from tqdm import tqdm


class JointBF:
    """
    Bayer noise reduction using joint bilateral filer technique
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform

    def apply_jbf(self):
        """
        Apply bnr to the input image and return the output image
        """
        in_img = self.img
        bayer_pattern = self.sensor_info["bayer_pattern"]
        width, height = self.sensor_info["width"], self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        # extract BNR parameters
        filt_size = self.parm_bnr["filter_window"]
        # s stands for spatial kernel parameters, r stands for range kernel parameters
        stddev_s_red, stddev_r_red = (
            self.parm_bnr["r_std_dev_s"],
            self.parm_bnr["r_std_dev_r"],
        )
        stddev_s_green, stddev_r_green = (
            self.parm_bnr["g_std_dev_s"],
            self.parm_bnr["g_std_dev_r"],
        )
        stddev_s_blue, stddev_r_blue = (
            self.parm_bnr["b_std_dev_s"],
            self.parm_bnr["b_std_dev_r"],
        )

        # assuming image is in 12-bit range, converting to [0 1] range
        in_img = np.float32(in_img) / (2**bit_depth - 1)

        interp_g = np.zeros((height, width), dtype=np.float32)
        in_img_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )
        in_img_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )

        # convert bayer image into sub-images for filtering each colour ch
        in_img_raw = in_img.copy()
        if bayer_pattern == "rggb":
            in_img_r = in_img_raw[0:height:2, 0:width:2]
            in_img_b = in_img_raw[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            in_img_r = in_img_raw[1:height:2, 1:width:2]
            in_img_b = in_img_raw[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            in_img_r = in_img_raw[0:height:2, 1:width:2]
            in_img_b = in_img_raw[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            in_img_r = in_img_raw[1:height:2, 0:width:2]
            in_img_b = in_img_raw[0:height:2, 1:width:2]

        # define the G interpolation kernel
        interp_kern_g_at_r = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )

        interp_kern_g_at_r = interp_kern_g_at_r / np.sum(interp_kern_g_at_r)

        interp_kern_g_at_b = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=np.float32,
        )

        interp_kern_g_at_b = interp_kern_g_at_b / np.sum(interp_kern_g_at_b)

        # convolve the kernel with image and mask the result based on given bayer pattern
        kern_filt_g_at_r = ndimage.convolve(in_img, interp_kern_g_at_r, mode="reflect")
        kern_filt_g_at_b = ndimage.convolve(in_img, interp_kern_g_at_b, mode="reflect")

        # clip any interpolation overshoots to [0 1] range
        kern_filt_g_at_r = np.clip(kern_filt_g_at_r, 0, 1)
        kern_filt_g_at_b = np.clip(kern_filt_g_at_b, 0, 1)

        interp_g = in_img.copy()
        interp_g_at_r = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )
        interp_g_at_b = np.zeros(
            (np.uint32(height / 2), np.uint32(width / 2)), dtype=np.float32
        )

        if bayer_pattern == "rggb":
            # extract R and B location Green pixels to form interpG image
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_b[1:height:2, 1:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 1:width:2]

        elif bayer_pattern == "bggr":
            # extract R and B location Green pixels to form interpG image
            interp_g[1:height:2, 1:width:2] = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g[0:height:2, 0:width:2] = kern_filt_g_at_b[0:height:2, 0:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 0:width:2]

        elif bayer_pattern == "grbg":
            # extract R and B location Green pixels to form interpG image
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_b[1:height:2, 0:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[0:height:2, 1:width:2]
            interp_g_at_b = kern_filt_g_at_b[1:height:2, 0:width:2]

        elif bayer_pattern == "gbrg":
            # extract R and B location Green pixels to form interpG image
            interp_g[1:height:2, 0:width:2] = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g[0:height:2, 1:width:2] = kern_filt_g_at_b[0:height:2, 1:width:2]

            # extract interpG ch sub-images for R sub-image and B sub-image guidance signals
            interp_g_at_r = kern_filt_g_at_r[1:height:2, 0:width:2]
            interp_g_at_b = kern_filt_g_at_b[0:height:2, 1:width:2]

        # BNR window / filter size will be the same for full image and smaller for sub-image
        filt_size_g = filt_size
        filt_size_r = int((filt_size + 1) / 2)
        filt_size_b = int((filt_size + 1) / 2)

        # apply joint bilateral filter to the image with G channel as guidance signal
        out_img_r = self.fast_joint_bilateral_filter(
            in_img_r,
            interp_g_at_r,
            filt_size_r,
            stddev_s_red,
            filt_size_r,
            stddev_r_red,
            2,
        )
        out_img_g = self.fast_joint_bilateral_filter(
            interp_g,
            interp_g,
            filt_size_g,
            stddev_s_green,
            filt_size_g,
            stddev_r_green,
            1,
        )
        out_img_b = self.fast_joint_bilateral_filter(
            in_img_b,
            interp_g_at_b,
            filt_size_b,
            stddev_s_blue,
            filt_size_b,
            stddev_r_blue,
            2,
        )

        # join the colour pixel images back into the bayer image
        bnr_out_img = np.zeros(in_img.shape)
        bnr_out_img = out_img_g.copy()

        if bayer_pattern == "rggb":
            bnr_out_img[0:height:2, 0:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[1:height:2, 1:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "bggr":
            bnr_out_img[1:height:2, 1:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[0:height:2, 0:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "grbg":
            bnr_out_img[0:height:2, 1:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[1:height:2, 0:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        elif bayer_pattern == "gbrg":
            bnr_out_img[1:height:2, 0:width:2] = out_img_r[
                0 : np.size(out_img_r, 0) : 1, 0 : np.size(out_img_r, 1) : 1
            ]
            bnr_out_img[0:height:2, 1:width:2] = out_img_b[
                0 : np.size(out_img_b, 0) : 1, 0 : np.size(out_img_b, 1) : 1
            ]

        # convert normalized image to 12-bit range
        bnr_out_img = np.uint16(np.clip(bnr_out_img, 0, 1) * ((2**bit_depth) - 1))
        return bnr_out_img

    def gauss_kern_raw(self, kern, std_dev, stride):
        """
        Applying Gaussian Filter
        """
        if kern % 2 == 0:
            warnings.warn("kernel size (kern) cannot be even, setting it as odd value")
            kern = kern + 1

        if kern <= 0:
            warnings.warn("kernel size (kern) cannot be <= zero, setting it as 3")
            kern = 3

        out_kern = np.zeros((kern, kern), dtype=np.float32)

        for i in range(0, kern):
            for j in range(0, kern):
                # stride is used to adjust the gaussian weights for neighbourhood
                # pixel that are 'stride' spaces apart in a bayer image
                out_kern[i, j] = np.exp(
                    -1
                    * (
                        (stride * (i - ((kern - 1) / 2))) ** 2
                        + (stride * (j - ((kern - 1) / 2))) ** 2
                    )
                    / (2 * (std_dev**2))
                )

        sum_kern = np.sum(out_kern)
        out_kern[0:kern:1, 0:kern:1] = out_kern[0:kern:1, 0:kern:1] / sum_kern

        return out_kern

    def joint_bilateral_filter(
        self, in_img, guide_img, spatial_kern, stddev_s, range_kern, stddev_r, stride
    ):
        """
        Applying Joint Bilateral Filter
        """

        # check if filter window sizes spatial_kern and range_kern greater
        # than zero and are odd
        if spatial_kern <= 0:
            spatial_kern = 3
            warnings.warn(
                "spatial kernel size (spatial_kern) cannot be <= zero, setting it as 3"
            )
        elif spatial_kern % 2 == 0:
            warnings.warn(
                "range kernel size (spatial_kern) cannot be even, "
                "assigning it an odd value"
            )
            spatial_kern = spatial_kern + 1

        # check if range_kern > spatial_kern
        if range_kern > spatial_kern:
            warnings.warn(
                "range kernel size (range_kern) cannot be more "
                "than spatial kernel size (spatial_kern)"
            )
            range_kern = spatial_kern

        # spawn a NxN gaussian kernel
        s_kern = self.gauss_kern_raw(spatial_kern, stddev_s, stride)

        # pad the image with half arm length of the kernel;
        # padType='constant' => pad value = 0; 'reflect' is more suitable
        pad_len = int((spatial_kern - 1) / 2)
        kern_arm = pad_len
        in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect")
        guide_img_ext = np.pad(
            guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
        )

        filt_out = np.zeros(in_img.shape, dtype=np.float32)

        for i in tqdm(
            range(kern_arm, np.size(in_img, 0) + kern_arm),
            disable=self.is_progress,
            leave=self.is_leave,
        ):
            for j in range(kern_arm, np.size(in_img, 1) + kern_arm):
                guide_img_ext_center_pix = guide_img_ext[i, j]
                guide_img_ext_filt_window = guide_img_ext[
                    i - kern_arm : i + kern_arm + 1, j - kern_arm : j + kern_arm + 1
                ]
                in_img_ext_filt_window = in_img_ext[
                    i - kern_arm : i + kern_arm + 1, j - kern_arm : j + kern_arm + 1
                ]

                # normalization fact of a filter window = sum(matrix multiplication of
                # spatial kernel and range kernel weights) = sum of filter weights
                norm_fact = np.sum(
                    s_kern[0:spatial_kern, 0:spatial_kern]
                    * np.exp(
                        -1
                        * (guide_img_ext_center_pix - guide_img_ext_filt_window) ** 2
                        / (2 * stddev_r**2)
                    )
                )

                # filter output for a window = sum(spatial kernel weights x range kernel weights x
                # windowed input image) / normalization factor
                filt_out[i - kern_arm, j - kern_arm] = np.sum(
                    s_kern[0:spatial_kern, 0:spatial_kern]
                    * np.exp(
                        -1
                        * (guide_img_ext_center_pix - guide_img_ext_filt_window) ** 2
                        / (2 * stddev_r**2)
                    )
                    * in_img_ext_filt_window
                )
                filt_out[i - kern_arm, j - kern_arm] = (
                    filt_out[i - kern_arm, j - kern_arm] / norm_fact
                )

        return filt_out

    def fast_joint_bilateral_filter(
        self, in_img, guide_img, spatial_kern, stddev_s, range_kern, stddev_r, stride
    ):
        """
        Applying Joint Bilateral Filter
        """

        # check if filter window sizes spatial_kern and range_kern greater than zero and are odd
        if spatial_kern <= 0:
            spatial_kern = 3
            warnings.warn(
                "spatial kernel size (spatial_kern) cannot be <= zero, setting it as 3"
            )
        elif spatial_kern % 2 == 0:
            warnings.warn(
                "range kernel size (spatial_kern) cannot be even, assigning it an odd value"
            )
            spatial_kern = spatial_kern + 1

        if range_kern <= 0:
            range_kern = 3
            warnings.warn(
                "range kernel size (range_kern) cannot be <= zero, setting it as 3"
            )
        elif range_kern % 2 == 0:
            warnings.warn(
                "range kernel size (range_kern) cannot be even, assigning it an odd value"
            )
            range_kern = range_kern + 1

        # check if range_kern > spatial_kern
        if range_kern > spatial_kern:
            warnings.warn(
                "range kernel size (range_kern) cannot be more than..."
                "spatial kernel size (spatial_kern)"
            )
            range_kern = spatial_kern

        # spawn a NxN gaussian kernel
        s_kern = self.gauss_kern_raw(spatial_kern, stddev_s, stride)

        # pad the image with half arm length of the kernel;
        # padType='constant' => pad value = 0; 'reflect' is more suitable
        pad_len = int((spatial_kern - 1) / 2)
        in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect")
        guide_img_ext = np.pad(
            guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
        )

        filt_out = np.zeros(in_img.shape, dtype=np.float32)
        norm_fact = np.zeros(in_img.shape)
        sum_filt_out = np.zeros(in_img.shape)

        for i in range(spatial_kern):
            for j in range(spatial_kern):
                # Creating shifted arrays for processing each pixel in the window
                in_img_ext_array = in_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]
                guide_img_ext_array = guide_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]

                # Adding normalization factor for each pixel needed to average out the
                # final result
                norm_fact += s_kern[i, j] * np.exp(
                    -1 * (guide_img - guide_img_ext_array) ** 2 / (2 * stddev_r**2)
                )

                # Summing up the final result
                sum_filt_out += (
                    s_kern[i, j]
                    * np.exp(
                        -1
                        * (guide_img - guide_img_ext_array) ** 2
                        / (2 * stddev_r**2)
                    )
                    * in_img_ext_array
                )

        filt_out = sum_filt_out / norm_fact

        return filt_out
