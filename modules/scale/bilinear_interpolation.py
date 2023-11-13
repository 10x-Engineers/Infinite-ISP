"""
File: scale.py
Description: Implements both hardware friendly and non hardware freindly scaling
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import numpy as np
from util.utils import stride_convolve2d

################################################################################
class BilinearInterpolation:
    """Scale 2D image to given size."""

    def __init__(self, img, new_size):
        self.single_channel = np.float32(img)
        self.new_size = new_size

    def bilinear_interpolation(self):

        """
        Upscale/Downscale 2D array by any scale factor using Bilinear method.
        """

        old_height, old_width = (
            self.single_channel.shape[0],
            self.single_channel.shape[1],
        )
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height, scale_width = new_height / old_height, new_width / old_width

        scaled_img = np.zeros((new_height, new_width), dtype="float32")
        old_coor = lambda a, scale_fact: (a + 0.5) / scale_fact - 0.5

        for row in range(new_height):
            for col in range(new_width):

                # Coordinates in old image
                old_row, old_col = old_coor(row, scale_height), old_coor(
                    col, scale_width
                )

                x_1 = (
                    0
                    if np.floor(old_col) < 0
                    else min(int(np.floor(old_col)), old_width - 1)
                )
                y_1 = (
                    0
                    if np.floor(old_row) < 0
                    else min(int(np.floor(old_row)), old_height - 1)
                )
                x_2 = (
                    0
                    if np.ceil(old_col) < 0
                    else min(int(np.ceil(old_col)), old_width - 1)
                )
                y_2 = (
                    0
                    if np.ceil(old_row) < 0
                    else min(int(np.ceil(old_row)), old_height - 1)
                )

                # Get four neghboring pixels
                q_11 = self.single_channel[y_1, x_1]
                q_12 = self.single_channel[y_1, x_2]
                q_21 = self.single_channel[y_2, x_1]
                q_22 = self.single_channel[y_2, x_2]

                # Interpolating p_1 and p_2
                weight_right = old_col - np.floor(old_col)
                weight_left = 1 - weight_right
                p_1 = weight_left * q_11 + weight_right * q_12
                p_2 = weight_left * q_21 + weight_right * q_22

                # The case where the new pixel lies between two pixels
                if x_1 == x_2:
                    p_1 = q_11
                    p_2 = q_22

                # Interpolating p
                weight_bottom = old_row - np.floor(old_row)
                weight_top = 1 - weight_bottom
                pixel_val = weight_top * p_1 + weight_bottom * p_2

                scaled_img[row, col] = pixel_val
        return scaled_img.astype("float32")

    def downscale_by_int_factor(self):

        """
        Downscale a 2D array by an integer scale factor using Bilinear method with 2D convolution.
        Parameters
        ----------
        new_size: Required output size.
        Output: 16 bit scaled image in which each pixel is an average of box nxm
        determined by the scale factors.
        """

        scale_height = self.new_size[0] / self.single_channel.shape[0]
        scale_width = self.new_size[1] / self.single_channel.shape[1]

        box_height = int(np.ceil(1 / scale_height))
        box_width = int(np.ceil(1 / scale_width))

        scaled_img = np.zeros((self.new_size[0], self.new_size[1]), dtype="float32")
        kernel = np.ones((box_height, box_width)) / (box_height * box_width)

        scaled_img = stride_convolve2d(self.single_channel, kernel)
        return scaled_img.astype("float32")
