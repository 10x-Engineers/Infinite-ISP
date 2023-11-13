"""
File: clahe.py
Description: Implements the contrast adjustment using contrast limited adaptive
histogram equalization (CLAHE) approach.
Code / Paper  Reference:
https://arxiv.org/ftp/arxiv/papers/2108/2108.12818.pdf#:~:text
=The%20technique%20to%20equalize%20the,a%20linear%20trend%20(CDF).
Implementation inspired from: MATLAB &
Fast Open ISP Author: Qiu Jueqin (qiujueqin@gmail.com)
Author: x10xEngineers
------------------------------------------------------------
"""
import math
import numpy as np


class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci):
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.wind = parm_ldci["wind"]
        self.clip_limit = parm_ldci["clip_limit"]
        self.is_save = parm_ldci["is_save"]
        self.platform = platform

    def pad_array(self, array, pads, mode="reflect"):
        """
        Pad an array with the given margins on left, right, top and bottom
        """
        if isinstance(pads, (list, tuple, np.ndarray)):
            if len(pads) == 2:
                pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (
                    array.ndim - 2
                )
            elif len(pads) == 4:
                pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (
                    array.ndim - 2
                )
            else:
                raise NotImplementedError

        return np.pad(array, pads, mode)

    def crop(self, array, crops):
        """
        Crop an array within the given margins
        """
        if isinstance(crops, (list, tuple, np.ndarray)):
            if len(crops) == 2:
                top_crop = bottom_crop = crops[0]
                left_crop = right_crop = crops[1]
            elif len(crops) == 4:
                top_crop, bottom_crop, left_crop, right_crop = crops
            else:
                raise NotImplementedError
        else:
            top_crop = bottom_crop = left_crop = right_crop = crops

        height, width = array.shape[:2]
        return array[
            top_crop : height - bottom_crop, left_crop : width - right_crop, ...
        ]

    def get_tile_lut(self, tiled_array):
        """
        Generating LUT using histogram equalization
        """
        # Computing histograms--hist will have bincounts
        hist, _ = np.histogram(tiled_array, bins=256, range=(0, 255))
        clip_limit = self.clip_limit

        # Clipping each bin counts within the range of window size to avoid artifacts
        # Applying a check to keep the clipping limit within the appropriate range
        if clip_limit >= self.wind:
            clip_limit = 0.08 * self.wind

        clipped_hist = np.clip(hist, 0, clip_limit)
        num_clipped_pixels = (hist - clipped_hist).sum()

        # Adding clipped pixels to each bin and getting its sum for normalization
        hist = clipped_hist + num_clipped_pixels / 256 + 1
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)

        # Computing cdf and getting the LUT for the array
        look_up_table = (cdf * 255).astype(np.uint8)

        return look_up_table

    def interp_blocks(self, weights, block, first_block_lut, second_block_lut):
        """
        Interpolating blocks using alpha blending weights
        """
        # alpha blending = weights
        first = weights * first_block_lut[block].astype(np.int32)
        second = (1024 - weights) * second_block_lut[block].astype(np.int32)

        # Interpolating both the LUTs
        return np.right_shift(first + second, 10).astype(np.uint8)

    def interp_top_bottom_block(self, left_lut_weights, block, left_lut, current_lut):
        """
        Interpolating blocks present at top and bottom of the arrays
        """
        return self.interp_blocks(left_lut_weights, block, left_lut, current_lut)

    def interp_left_right_block(self, top_lut_weights, block, top_lut, current_lut):
        """
        Interpolating blocks present at left and right of the arrays
        """
        return self.interp_blocks(top_lut_weights, block, top_lut, current_lut)

    def interp_neighbor_block(
        self,
        left_lut_weights,
        top_lut_weights,
        block,
        tl_lut,
        top_lut,
        left_lut,
        current_lut,
    ):
        """
        Interpolating blocks present in the middle of the arrays
        """
        interp_top_blocks = self.interp_blocks(left_lut_weights, block, tl_lut, top_lut)
        interp_current_blocks = self.interp_blocks(
            left_lut_weights, block, left_lut, current_lut
        )

        interp_final = np.right_shift(
            top_lut_weights * interp_top_blocks
            + (1024 - top_lut_weights) * interp_current_blocks,
            10,
        ).astype(np.uint8)
        return interp_final

    def is_corner_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in a corner region
        """
        return (
            (i_row == 0 and i_col == 0)
            or (i_row == 0 and i_col == x_tiles)
            or (i_row == y_tiles and i_col == 0)
            or (i_row == y_tiles and i_col == x_tiles)
        )

    def is_top_or_bottom_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in teh top or bottom region
        """
        return (i_row == 0 or i_row == y_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def is_left_or_right_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in the left or right region
        """
        return (i_col == 0 or i_col == x_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def apply_clahe(self):
        """
        Applying clahe algorithm for contrast enhancement
        """
        wind = self.wind
        in_yuv = self.yuv

        # Extracting Luminance channel from yuv as LDCI will be applied to Y channel only

        yuv = in_yuv[:, :, 0]
        img_height, img_width = yuv.shape

        # pipeline specific: if input is in analog yuv
        if in_yuv.dtype == "float32":
            yuv = np.round(255 * yuv).astype(np.uint8)

        # output clipped equalized histogram
        out_ceh = np.empty(shape=(img_height, img_width, 3), dtype=np.uint8)

        # computing number of tiles (tiles = block = window).
        vert_tiles = math.ceil(img_height / wind)
        horiz_tiles = math.ceil(img_width / wind)
        tile_height = wind
        tile_width = wind

        # Computing number of columns and rows to be padded in the image
        # for getting proper block/tile
        row_pads = tile_height * vert_tiles - img_height
        col_pads = tile_width * horiz_tiles - img_width
        pads = (
            row_pads // 2,
            row_pads - row_pads // 2,
            col_pads // 2,
            col_pads - col_pads // 2,
        )

        # Assigning linearized LUT weights to top and left blocks
        left_lut_weights = np.linspace(1024, 0, tile_width, dtype=np.int32).reshape(
            (1, -1)
        )
        top_lut_weights = np.linspace(1024, 0, tile_height, dtype=np.int32).reshape(
            (-1, 1)
        )

        # Declaring an empty 3D (x,y,z) array of LUTs for each tile, where x,y are
        # the coordinates of the tile and z a linear array of 256
        luts = np.empty(shape=(vert_tiles, horiz_tiles, 256), dtype=np.uint8)

        # Creating a copy of yuv image
        y_padded = yuv
        y_padded = self.pad_array(y_padded, pads=pads)

        # for loops for getting LUT for each tile (row wise iterations)
        for rows in range(vert_tiles):
            for colm in range(horiz_tiles):
                # Extracting tile
                start_row = rows * tile_height
                end_row = (rows + 1) * tile_height
                start_col = colm * tile_width
                end_col = (colm + 1) * tile_width

                # Extracting each tile
                y_tile = y_padded[start_row:end_row, start_col:end_col]

                # Getting LUT for each tile using HE
                luts[rows, colm] = self.get_tile_lut(y_tile)

        # Declaring an empty array for output array after padding is done
        y_ceh = np.empty_like(y_padded)

        # For loops for processing image array tile by tile
        for i_row in range(vert_tiles + 1):
            for i_col in range(horiz_tiles + 1):
                # Extracting tile/block
                start_row_index = i_row * tile_height - tile_height // 2
                end_row_index = min(start_row_index + tile_height, y_padded.shape[0])
                start_col_index = i_col * tile_width - tile_width // 2
                end_col_index = min(start_col_index + tile_width, y_padded.shape[1])
                start_row_index = max(start_row_index, 0)
                start_col_index = max(start_col_index, 0)

                # Extracting the tile for processing
                y_block = (
                    y_padded[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ]
                ).astype(
                    np.uint8
                )  # tile/block

                # checking the position of the block and applying interpolation accordingly
                if self.is_corner_block(horiz_tiles, vert_tiles, i_col, i_row):
                    # if the block is present at the corner, no need of interpolation,
                    # just apply the LUT
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    lut = luts[lut_y_idx, lut_x_idx]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = (lut[y_block]).astype(np.float32)

                elif self.is_top_or_bottom_block(horiz_tiles, vert_tiles, i_col, i_row):
                    # if the block is present at the top or bottom region,
                    # current block is updated after interpolating with its left block
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    left_lut = luts[lut_y_idx, i_col - 1]
                    current_lut = luts[lut_y_idx, i_col]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = (
                        (
                            self.interp_top_bottom_block(
                                left_lut_weights, y_block, left_lut, current_lut
                            )
                        )
                    ).astype(
                        np.float32
                    )

                elif self.is_left_or_right_block(horiz_tiles, vert_tiles, i_col, i_row):
                    # if the block is present at the left or right region, current block is
                    # updated after interpolating with its top block
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    top_lut = luts[i_row - 1, lut_x_idx]
                    current_lut = luts[i_row, lut_x_idx]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = (
                        (
                            self.interp_left_right_block(
                                top_lut_weights, y_block, top_lut, current_lut
                            )
                        )
                    ).astype(
                        np.float32
                    )

                else:
                    # check to see if the block is present in the middle region of the image
                    # the block needs to be updated after getting interpolated with its
                    # neighboring blocks
                    tl_lut = luts[i_row - 1, i_col - 1]
                    top_lut = luts[i_row - 1, i_col]
                    left_lut = luts[i_row, i_col - 1]
                    current_lut = luts[i_row, i_col]
                    y_ceh[
                        start_row_index:end_row_index, start_col_index:end_col_index
                    ] = (
                        (
                            self.interp_neighbor_block(
                                left_lut_weights,
                                top_lut_weights,
                                y_block,
                                tl_lut,
                                top_lut,
                                left_lut,
                                current_lut,
                            )
                        )
                    ).astype(
                        np.float32
                    )

        y_padded = self.crop(y_ceh, pads)

        # pipeline specific: if input is in analog yuv
        if in_yuv.dtype == "float32":
            y_padded = np.float32((y_padded) / 255.0)
            out_ceh = out_ceh.astype("float32")

        out_ceh[:, :, 0] = y_padded
        out_ceh[:, :, 1] = in_yuv[:, :, 1]
        out_ceh[:, :, 2] = in_yuv[:, :, 2]

        return out_ceh
