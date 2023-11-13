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
class NearestNeighbor:
    """Scale 2D image to given size."""

    def __init__(self, img, new_size):
        self.single_channel = np.float32(img)
        self.new_size = new_size

    def downscale_nearest_neighbor(self):

        """Down scale by an integer factor using NN method using convolution."""
        # print("Downscaling with Nearest Neighbor.")

        old_height, old_width = (
            self.single_channel.shape[0],
            self.single_channel.shape[1],
        )
        new_height, new_width = self.new_size[0], self.new_size[1]

        # As new_size is less than old_size, scale factor is defined s.t it is >1 for downscaling
        scale_height, scale_width = old_height / new_height, old_width / new_width

        kernel = np.zeros((int(scale_height), int(scale_width)))
        kernel[0, 0] = 1

        scaled_img = stride_convolve2d(self.single_channel, kernel)
        return scaled_img.astype("float32")

    def scale_nearest_neighbor(self):

        """
        Upscale/Downscale 2D array by any scale factor using Nearest Neighbor (NN) algorithm.
        """

        old_height, old_width = (
            self.single_channel.shape[0],
            self.single_channel.shape[1],
        )
        new_height, new_width = self.new_size[0], self.new_size[1]
        scale_height, scale_width = new_height / old_height, new_width / old_width

        scaled_img = np.zeros((new_height, new_width), dtype="float32")

        for row in range(new_height):
            for col in range(new_width):
                row_nearest = int(np.floor(row / scale_height))
                col_nearest = int(np.floor(col / scale_width))
                scaled_img[row, col] = self.single_channel[row_nearest, col_nearest]
        return scaled_img
