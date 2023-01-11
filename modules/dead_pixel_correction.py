# File: dead_pixel_correction.py
# Description: Corrects the hot or dead pixels
# Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921                   
# Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from scipy import ndimage

class DeadPixelCorrection:
    'Dead Pixel Correction'

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc['isEnable']
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform['disable_progress_bar']
        self.is_leave = platform['leave_pbar_string']

    def padding(self):
        """Return a mirror padded copy of image."""

        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def apply_dead_pixel_correction(self):
        """This function detects and corrects Dead pixels."""

        height, width = self.sensor_info["height"], self.sensor_info["width"]
        self.bpp = self.sensor_info["bitdep"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["isDebug"]

        # Mirror padding is applied to self.img.
        img_padded = np.float32(self.padding())
        dpc_img = np.empty((height, width), np.float32)
        corrected_pv_count = 0

        fp = np.array([[1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1]])

        max_filt = ndimage.maximum_filter(img_padded, footprint=fp)
        min_filt = ndimage.minimum_filter(img_padded, footprint=fp)

        possible_dead_pixels = np.logical_or(img_padded > max_filt, img_padded < min_filt)

        tmp_shape = (img_padded.shape[0], img_padded.shape[1], 1)

        cdif0 = ndimage.correlate(img_padded, np.array([[1, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif1 = ndimage.correlate(img_padded, np.array([[0, 0, 1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif2 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 1],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif3 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [1, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif4 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 1],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif5 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [1, 0, 0, 0, 0]])).reshape(tmp_shape)

        cdif6 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 1, 0, 0]])).reshape(tmp_shape)
                                                
        cdif7 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, -1, 0, 0],
                                                        [0, 0, 0, 0, 0],
                                                        [0, 0, 0, 0, 1]])).reshape(tmp_shape)

        cdiff = np.abs(np.concatenate((cdif0, cdif1, cdif2, cdif3, cdif4, cdif5, cdif6, cdif7), axis=2))
        del cdif0, cdif1, cdif2, cdif3, cdif4, cdif5, cdif6, cdif7

        correct_at = np.logical_and(possible_dead_pixels, np.where(cdiff > self.threshold, True, False).any(axis=2))

        gradH = np.abs(ndimage.correlate(img_padded, np.array([[-1, 0, 2, 0, -1]]))).reshape(tmp_shape)

        gradV = np.abs(ndimage.correlate(img_padded, np.array([[-1, 0, 2, 0, -1]]).T)).reshape(tmp_shape)

        grad45 = np.abs(ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, -1],
                                                                [0, 0, 0, 0, 0],
                                                                [0, 0, 2, 0, 0],
                                                                [0, 0, 0, 0, 0],
                                                                [-1, 0, 0, 0, 0]]))).reshape(tmp_shape)

        grad135 = np.abs(ndimage.correlate(img_padded, np.array([[-1, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 2, 0, 0],
                                                                 [0, 0, 0, 0, 0],
                                                                 [0, 0, 0, 0, -1]]))).reshape(tmp_shape)

        min_grads = np.min(np.concatenate((gradH, gradV, grad45, grad135), axis=2), axis=2)

        corrected = np.zeros(img_padded.shape)

        meanH = ndimage.correlate(img_padded, np.array([[1, 0, 0, 0, 1]])) / 2

        meanV = ndimage.correlate(img_padded, np.array([[1, 0, 0, 0, 1]]).T) / 2

        mean45 = ndimage.correlate(img_padded, np.array([[0, 0, 0, 0, 1],
                                                         [0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0],
                                                         [0, 0, 0, 0, 0],
                                                         [1, 0, 0, 0, 0]])) / 2

        mean135 = ndimage.correlate(img_padded, np.array([[1, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 0],
                                                          [0, 0, 0, 0, 1]])) / 2

        corrected = np.where(min_grads == gradH, meanH, 
                    np.where(min_grads == gradV, meanV, 
                    np.where(min_grads == grad45, mean45, 
                    np.where(min_grads == grad135, mean135, img_padded))))

        del gradH, gradV, grad45, grad135
        del meanH, meanV, mean45, mean135

        corrected = np.where(correct_at, corrected, img_padded)

        dpc_img = corrected[2:-2, 2:-2]
        
        self.img = np.uint16(np.clip(dpc_img, 0, (2**self.bpp)-1))
        if self.is_debug:
            corrected_pv_count = np.where(correct_at, 1, 0).sum()
            print('   - Number of corrected pixels = ', corrected_pv_count)
            print('   - Threshold = ', self.threshold)
        return self.img

    def execute(self):
        print('Dead Pixel Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_dead_pixel_correction()