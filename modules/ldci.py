import numpy as np
import math

class LDCI:
    'Local Dynamic Contrast Enhancement'

    def __init__(self, YUV, sensor_info, parm_ldci):
        self.YUV = YUV
        self.enable = parm_ldci['isEnable']
        self.sensor_info = sensor_info
        self.wind = parm_ldci['wind']
        self.clip_limit = parm_ldci['clip_limit']

    def pad_array(self, array, pads, mode='reflect'):
        
        # -------Pad an array with the given margins on left, right, top and bottom--------
        if isinstance(pads, (list, tuple, np.ndarray)):
            if len(pads) == 2:
                pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
            elif len(pads) == 4:
                pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
            else:
                raise NotImplementedError

        return np.pad(array, pads, mode)

    def crop(self, array, crops):
        
        # ---------crop an array within the given margins-----------
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
        return array[top_crop: height - bottom_crop, left_crop: width - right_crop, ...]

    def get_tile_lut(self, tiled_array):
        
        # ---------generating LUT using histogram equalization----
        
        # Computing histograms--hist will have bincounts
        hist, bins = np.histogram(tiled_array, bins=256, range=(0, 255))
        clip_limit = self.clip_limit

        # Clipping each bin counts within the range of window size to avoid artifacts
        # Applying a check to keep the clipping limit within the appropriate range
        if clip_limit >= self.wind:
            clip_limit = 0.08*self.wind

        clipped_hist = np.clip(hist, 0, clip_limit)
        num_clipped_pixels = (hist - clipped_hist).sum()
        
        # Adding clipped pixels to each bin and getting its sum for normalization
        hist = clipped_hist + num_clipped_pixels / 256 + 1
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)
        
        # Computing cdf and getting the LUT for the array
        LUT = (cdf * 255).astype(np.uint8)

        return LUT

    def interp_blocks(self, weights, block, first_block_lut, second_block_lut):
        
        # alpha blending = weights
        first = weights * first_block_lut[block].astype(np.int32)
        second = (1024 - weights) * second_block_lut[block].astype(np.int32)
        
        # Interpolating both the LUTs
        return np.right_shift(first + second, 10).astype(np.uint8)

    def interp_top_bottom_block(self, left_lut_weights, block, left_lut, current_lut):
        
        # ---------interpolating blocks present at top and bottom of the arrays-----
        return self.interp_blocks(left_lut_weights, block, left_lut, current_lut )


    def interp_left_right_block(self, top_lut_weights, block, top_lut, current_lut):
        
        return self.interp_blocks(top_lut_weights, block, top_lut, current_lut )


    def interp_neighbor_block(self, left_lut_weights, top_lut_weights, block, tl_lut, top_lut, left_lut, current_lut):
        
        # ---------interpolating blocks present in the middle of the arrays-----
        interp_top_blocks = self.interp_blocks(left_lut_weights, block, tl_lut, top_lut)
        interp_current_blocks = self.interp_blocks(left_lut_weights, block, left_lut, current_lut)
        
        interp_final = np.right_shift(top_lut_weights * interp_top_blocks + (1024 - top_lut_weights) * interp_current_blocks, 10).astype(np.uint8)
        return interp_final


    def is_corner_block(self, x_tiles, y_tiles, i_col, i_row):
        
        #----Checking if the current image block is locating in a corner region """
        return ((i_row == 0 and i_col == 0) or
                (i_row == 0 and i_col == x_tiles) or
                (i_row == y_tiles and i_col == 0) or
                (i_row == y_tiles and i_col == x_tiles))

    def is_top_or_bottom_block(self, x_tiles, y_tiles, i_col, i_row):

        #------Checking if the current image block is locating in teh top or bottom region
        return (i_row == 0 or i_row == y_tiles) and not self.is_corner_block(x_tiles, y_tiles, i_col, i_row)

    def is_left_or_right_block(self, x_tiles, y_tiles, i_col, i_row):
        
        # ----- Checking if the current image block is locating in the left or right region
        return (i_col == 0 or i_col == x_tiles) and not self.is_corner_block(x_tiles, y_tiles, i_col, i_row)

    def apply_ldci(self):

        # ----------applying LDCI module to the given image-------
        w = self.wind
        in_YUV = self.YUV
        
        # Extracting Luminance channel from YUV as LDCI will be applied to Y channel only
        
        YUV = in_YUV[:, :, 0]
        img_height, img_width = YUV.shape

        # pipeline specific: if input is in analog yuv
        if(in_YUV.dtype == 'float32'):
            YUV = np.round(255*YUV).astype(np.uint8)

        # output clipped equalized histogram
        out_CEH = np.empty(shape=(img_height, img_width, 3), dtype=np.uint8)

        # computing number of tiles (tiles = block = window). 
        vert_tiles = math.ceil(img_height / w)
        horiz_tiles = math.ceil(img_width / w)
        tile_height = w
        tile_width = w

        # Computing number of columns and rows to be padded in the image for getting proper block/tile
        row_pads = tile_height * vert_tiles - img_height
        col_pads = tile_width * horiz_tiles - img_width
        pads = (row_pads // 2, row_pads - row_pads // 2, col_pads // 2, col_pads - col_pads // 2)

        # Assigning linearized LUT weights to top and left blocks
        left_lut_weights = np.linspace(1024, 0, tile_width, dtype=np.int32).reshape((1, -1))
        top_lut_weights = np.linspace(1024, 0, tile_height, dtype=np.int32).reshape((-1, 1))

        # Declaring an empty 3D (x,y,z) array of LUTs for each tile, where x,y are
        # the coordinates of the tile and z a linear array of 256
        luts = np.empty(shape=(vert_tiles, horiz_tiles, 256), dtype=np.uint8)

        # Creating a copy of YUV image
        Y_padded = YUV
        Y_padded = self.pad_array(Y_padded, pads=pads)

        # for loops for getting LUT for each tile (row wise iterations)
        for rows in range(vert_tiles):
            for colm in range(horiz_tiles):
                # Extracting tile
                start_row = rows * tile_height
                end_row = (rows + 1) * tile_height
                start_col = colm * tile_width
                end_col = (colm + 1) * tile_width

                # Extracting each tile
                Y_tile = Y_padded[start_row : end_row , start_col : end_col]
                
                # Getting LUT for each tile using HE
                luts[rows, colm] = self.get_tile_lut(Y_tile)

        # Declaring an empty array for output array after padding is done
        Y_ceh = np.empty_like(Y_padded)

        # For loops for processing image array tile by tile
        for i_row in range(vert_tiles + 1):
            for i_col in range(horiz_tiles + 1):

                # Extracting tile/block
                start_row_index = i_row * tile_height - tile_height // 2
                end_row_index = min(start_row_index + tile_height, Y_padded.shape[0])
                start_col_index = i_col * tile_width - tile_width // 2
                end_col_index = min(start_col_index + tile_width, Y_padded.shape[1])
                start_row_index = max(start_row_index, 0)
                start_col_index = max(start_col_index, 0)

                # Extracting the tile for processing
                y_block = (Y_padded[start_row_index:end_row_index, start_col_index:end_col_index]).astype(np.uint8)  # tile/block

                # checking the position of the block and applying interpolation accordingly
                if self.is_corner_block(horiz_tiles, vert_tiles, i_col, i_row):

                    # if the block is present at the corner, no need of interpolation, just apply the LUT
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    lut = luts[lut_y_idx, lut_x_idx]
                    Y_ceh[start_row_index:end_row_index, start_col_index:end_col_index] = (lut[y_block]).astype(np.float32)

                elif self.is_top_or_bottom_block(horiz_tiles, vert_tiles, i_col, i_row):

                    # if the block is present at the top or bottom region, current block is updated after
                    # interpolating with its left block
                    lut_y_idx = 0 if i_row == 0 else vert_tiles - 1
                    left_lut = luts[lut_y_idx, i_col - 1]
                    current_lut = luts[lut_y_idx, i_col]
                    Y_ceh[start_row_index:end_row_index, start_col_index:end_col_index] = ((self.interp_top_bottom_block(left_lut_weights, y_block, left_lut,
                                                                           current_lut))).astype(np.float32)

                elif self.is_left_or_right_block(horiz_tiles, vert_tiles, i_col, i_row):

                    # if the block is present at the left or right region, current block is updated after
                    # interpolating with its top block
                    lut_x_idx = 0 if i_col == 0 else horiz_tiles - 1
                    top_lut = luts[i_row - 1, lut_x_idx]
                    current_lut = luts[i_row, lut_x_idx]
                    Y_ceh[start_row_index:end_row_index, start_col_index:end_col_index] = (
                                (self.interp_left_right_block(top_lut_weights, y_block, top_lut, current_lut))).astype(
                        np.float32)

                else:
                    # check to see if the block is present in the middle region of the image
                    # the block needs to be updated after getting interpolated with its neighboring blocks
                    tl_lut = luts[i_row - 1, i_col - 1]
                    top_lut = luts[i_row - 1, i_col]
                    left_lut = luts[i_row, i_col - 1]
                    current_lut = luts[i_row, i_col]
                    Y_ceh[start_row_index:end_row_index, start_col_index:end_col_index] = (
                                (self.interp_neighbor_block(left_lut_weights, top_lut_weights, y_block, tl_lut,
                                                        top_lut, left_lut, current_lut))).astype(np.float32)

        Y_padded = self.crop(Y_ceh, pads)

        # pipeline specific: if input is in analog yuv
        if(in_YUV.dtype == 'float32'):
            Y_padded = np.float32((Y_padded) / 255.0)
            out_CEH = out_CEH.astype('float32')

        out_CEH[:, :, 0] = Y_padded
        out_CEH[:, :, 1] = in_YUV[:, :, 1]
        out_CEH[:, :, 2] = in_YUV[:, :, 2]

        return out_CEH

    def execute(self):
        print('LDCI = ' + str(self.enable))

        if self.enable == False:
            return self.YUV
        else:
            return self.apply_ldci()