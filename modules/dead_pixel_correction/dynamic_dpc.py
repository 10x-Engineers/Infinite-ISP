"""
File: dead_pixel_correction.py
Description: Corrects the hot or dead pixels
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, correlate


class DynamicDPC:
    """
    Dynamic dead pixel correction scans the entire image for potential
    dead pixels and apply correction using the neighboring pixels.
    """

    def __init__(self, img, sensor_info, parm_dpc):
        self.img = img
        self.sensor_info = sensor_info
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = parm_dpc["dp_threshold"]
        self.is_debug = parm_dpc["is_debug"]

    def dynamic_dpc(self):
        """This function detects and corrects Dead pixels using numpy
        array opertaions."""

        height, width = self.sensor_info["height"], self.sensor_info["width"]

        dpc_img = np.empty((height, width), np.float32)

        # Get 3x3 neighbourhood of each pixel.
        # 5x5 matrix is defined as this window is extarcted from raw image.
        window = np.array(
            [
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
            ]
        )

        # The maximum and minimum filters automatically pad the input image internally,
        # eliminating the need for manual padding.
        max_value = maximum_filter(self.img, footprint=window, mode="mirror")
        min_value = minimum_filter(self.img, footprint=window, mode="mirror")

        # Condition 1: center_pixel needs to be corrected if it lies outside the
        # interval(min_value,max) of the 3x3 neighbourhood.
        # min_value < center_pixel < max_value--> no correction needed
        mask_cond1 = (
            np.where((min_value > self.img) | (self.img > max_value), True, False)
        ).astype("int32")

        # Condition 2:
        # center_pixel is corrected only if the difference of center_pixel and every
        # neighboring pixel is greater than the specified threshold.
        # The two if conditions are used in combination to reduce False positives.

        # Kernels to compute the difference between center pixel and
        # each of the 8 neighbours.
        ker_top_left = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_mid = np.array(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_top_right = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_mid_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        ker_bottom_left = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_bottom_mid = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0],
            ]
        )
        ker_bottom_right = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        # convolve each kernel over image to compute differences
        # The correlate function automatically pads the input image internally,
        # eliminating the need for manual padding.

        diff_top_left = np.abs(correlate(self.img, ker_top_left, mode="mirror"))
        diff_top_mid = np.abs(correlate(self.img, ker_top_mid, mode="mirror"))
        diff_top_right = np.abs(correlate(self.img, ker_top_right, mode="mirror"))
        diff_mid_left = np.abs(correlate(self.img, ker_mid_left, mode="mirror"))
        diff_mid_right = np.abs(correlate(self.img, ker_mid_right, mode="mirror"))
        diff_bottom_left = np.abs(correlate(self.img, ker_bottom_left, mode="mirror"))
        diff_bottom_mid = np.abs(correlate(self.img, ker_bottom_mid, mode="mirror"))
        diff_bottom_right = np.abs(correlate(self.img, ker_bottom_right, mode="mirror"))

        del (
            ker_top_left,
            ker_top_mid,
            ker_top_right,
            ker_mid_left,
            ker_mid_right,
            ker_bottom_left,
            ker_bottom_mid,
            ker_bottom_right,
        )

        # Stack all arrays
        diff_array = np.stack(
            [
                diff_top_left,
                diff_top_mid,
                diff_top_right,
                diff_mid_left,
                diff_mid_right,
                diff_bottom_left,
                diff_bottom_mid,
                diff_bottom_right,
            ],
            axis=2,
        )

        del (
            diff_top_left,
            diff_top_mid,
            diff_top_right,
            diff_mid_left,
            diff_mid_right,
            diff_bottom_left,
            diff_bottom_mid,
            diff_bottom_right,
        )

        # all gradients must be greater than the threshold for a pixel to be a DP.
        mask_cond2 = np.all(np.where(diff_array > self.threshold, True, False), axis=2)

        # mask with 1 for DPs and 0 for good pixels (dead pixels are the ones for which
        # both conditions are true)
        detection_mask = mask_cond1 * mask_cond2

        # Compute gradients
        ker_v = np.array([[-1, 0, 2, 0, -1]]).T
        ker_h = np.array([[-1, 0, 2, 0, -1]])
        ker_left_dia = np.array(
            [
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
            ]
        )
        ker_right_dia = np.array(
            [
                [-1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        # Convole the kernels to compute respective gradients
        vertical_grad = np.abs(correlate(self.img, ker_v, mode="mirror"))
        horizontal_grad = np.abs(correlate(self.img, ker_h, mode="mirror"))
        left_diagonal_grad = np.abs(correlate(self.img, ker_left_dia, mode="mirror"))
        right_diagonal_grad = np.abs(correlate(self.img, ker_right_dia, mode="mirror"))

        # Delete temporary variables
        del ker_v, ker_h, ker_left_dia, ker_right_dia

        # compute the direction of the minimum gradient
        min_grad = np.min(
            np.stack(
                [
                    vertical_grad,
                    horizontal_grad,
                    left_diagonal_grad,
                    right_diagonal_grad,
                ],
                axis=2,
            ),
            axis=2,
        )

        # corrected value is computed as the mean of the neighbours
        # in the direction of min_value gadient.
        ker_mean_v = np.array([[1, 0, 0, 0, 1]]).T / 2
        ker_mean_h = np.array([[1, 0, 0, 0, 1]]) / 2
        ker_mean_ldia = (
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]
            )
            / 2
        )
        ker_mean_rdia = (
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            / 2
        )

        # Convolve kernels to compute mean for each direction
        mean_v = correlate(self.img, ker_mean_v, mode="mirror")
        mean_h = correlate(self.img, ker_mean_h, mode="mirror")
        mean_ldia = correlate(self.img, ker_mean_ldia, mode="mirror")
        mean_rdia = correlate(self.img, ker_mean_rdia, mode="mirror")

        del ker_mean_v, ker_mean_h, ker_mean_ldia, ker_mean_rdia

        # Corrected image has the corrected pixel values in place of a detected dead pixel
        # and 0 elsewhere
        corrected_img = np.zeros(self.img.shape)

        # compile all pixels that can be corrected using the neighbors in the same direction
        corrected_v = np.where(min_grad == vertical_grad, mean_v, 0) * detection_mask
        corrected_h = np.where(min_grad == horizontal_grad, mean_h, 0) * detection_mask
        corrected_ldia = (
            np.where(min_grad == left_diagonal_grad, mean_ldia, 0) * detection_mask
        )
        corrected_rdia = (
            np.where(min_grad == right_diagonal_grad, mean_rdia, 0) * detection_mask
        )

        # In most cases, the corrected masks created above will not overlap, as each pixel will
        # have a unique gradient direction. However, in rare cases where two or more directions
        # have the same gradient magnitude, it is necessary to specify which neighboring pixels
        # should be used to calculate the corrected value. To resolve this issue, the next block
        # of code ensures that each pixel is only corrected once, giving priority to the vertical
        # direction, then horizontal, left diagonal, and finally the right diagonal.
        corrected_img = corrected_img + corrected_v
        corrected_img = np.where(corrected_img == 0, corrected_h, corrected_img)
        corrected_img = np.where(corrected_img == 0, corrected_ldia, corrected_img)
        corrected_img = np.where(corrected_img == 0, corrected_rdia, corrected_img)

        del mean_h, mean_v, mean_ldia, mean_rdia
        del corrected_v, corrected_h, corrected_ldia, corrected_rdia

        # Insert correct value of the detected dead pixels
        dpc_img = np.where(detection_mask, corrected_img, self.img)

        # Remove padding
        self.img = np.uint16(np.clip(dpc_img, 0, (2**self.bpp) - 1))

        if self.is_debug:
            print(
                "   - Number of corrected pixels = ",
                np.count_nonzero(detection_mask),
            )
            print("   - Threshold = ", self.threshold)
        return self.img
