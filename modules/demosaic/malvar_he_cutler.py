"""
File: malvar_he_cutler.py
Description: Implements the Malvar-He-Cutler algorithm for cfa interpolation
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
from scipy.signal import correlate2d


class Malvar:
    """
    CFA interpolation or Demosaicing
    """

    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks

    def apply_malvar(self):
        """
        Demosaicing the given raw image using Malvar-He-Cutler
        """
        # 3D masks accoridng to the given bayer
        mask_r, mask_g, mask_b = self.masks
        raw_in = np.float32(self.img)

        # Declaring 3D Demosaiced image
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))

        # 5x5 2D Filter coefficients for linear interpolation of
        # r_channel,g_channel and b_channel channels
        # These filters helps to retain corresponding pixels information using
        # laplacian while interpolation

        # g_channel at r_channel & b_channel location,
        g_at_r_and_b = (
            np.float32(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            * 0.125
        )

        # r_channel at green in r_channel row & b_channel column --
        # b_channel at green in b_channel row & r_channel column
        r_at_gr_and_b_at_gb = (
            np.float32(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            * 0.125
        )

        # r_channel at green in b_channel row & r_channel column --
        # b_channel at green in r_channel row & b_channel column
        r_at_gb_and_b_at_gr = np.transpose(r_at_gr_and_b_at_gb)

        # r_channel at blue in b_channel row & b_channel column --
        # b_channel at red in r_channel row & r_channel column
        r_at_b_and_b_at_r = (
            np.float32(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            * 0.125
        )

        # Creating r_channel, g_channel & b_channel channels from raw_in
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b

        # Creating g_channel channel first after applying g_at_r_and_b filter
        g_channel = np.where(
            np.logical_or(mask_r == 1, mask_b == 1),
            correlate2d(raw_in, g_at_r_and_b, mode="same", boundary="symm"),
            g_channel,
        )

        # Applying other linear filters
        rb_at_g_rbbr = correlate2d(
            raw_in, r_at_gr_and_b_at_gb, mode="same", boundary="symm"
        )
        rb_at_g_brrb = correlate2d(
            raw_in, r_at_gb_and_b_at_gr, mode="same", boundary="symm"
        )
        rb_at_gr_bbrr = correlate2d(
            raw_in, r_at_b_and_b_at_r, mode="same", boundary="symm"
        )

        # After convolving the input raw image with rest of the filters,
        # now we have the respective interpolated data, now we just have
        # to extract the updated pixels values according to the
        # position they are meant to be updated

        # Extracting Red rows.
        r_rows = np.transpose(np.any(mask_r == 1, axis=1)[np.newaxis]) * np.ones(
            r_channel.shape, dtype=np.float32
        )

        # Extracting Red columns.
        r_col = np.any(mask_r == 1, axis=0)[np.newaxis] * np.ones(
            r_channel.shape, dtype=np.float32
        )

        # Extracting Blue rows.
        b_rows = np.transpose(np.any(mask_b == 1, axis=1)[np.newaxis]) * np.ones(
            b_channel.shape, dtype=np.float32
        )

        # Extracting Blue columns
        b_col = np.any(mask_b == 1, axis=0)[np.newaxis] * np.ones(
            b_channel.shape, dtype=np.float32
        )

        # For R channel we have to update pixels at [r_channel rows
        # and b_channel cols] & at [b_channel rows and r_channel cols]
        # 3 pixels need to be updated near one given r_channel
        r_channel = np.where(
            np.logical_and(r_rows == 1, b_col == 1), rb_at_g_rbbr, r_channel
        )
        r_channel = np.where(
            np.logical_and(b_rows == 1, r_col == 1), rb_at_g_brrb, r_channel
        )

        # Similarly for B channel we have to update pixels at
        # [r_channel rows and b_channel cols]
        # & at [b_channel rows and r_channel cols] 3 pixels need
        # to be updated near one given b_channel
        b_channel = np.where(
            np.logical_and(b_rows == 1, r_col == 1), rb_at_g_rbbr, b_channel
        )
        b_channel = np.where(
            np.logical_and(r_rows == 1, b_col == 1), rb_at_g_brrb, b_channel
        )

        # Final r_channel & b_channel channels
        r_channel = np.where(
            np.logical_and(b_rows == 1, b_col == 1), rb_at_gr_bbrr, r_channel
        )
        b_channel = np.where(
            np.logical_and(r_rows == 1, r_col == 1), rb_at_gr_bbrr, b_channel
        )

        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel

        return demos_out
