"""
File: sharpen.py
Description: Simple unsharp masking with frequency and strength control.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
"""
import numpy as np
from scipy import ndimage


class UnsharpMasking:
    """
    Implements Unsharp Masking Algorithm
    """

    def __init__(self, img, sharpen_sigma, sharpen_strength):
        self.img = img
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength

    def apply_sharpen(self):
        """
        Applying sharpening to the input image
        """
        luma = np.float32(self.img[:, :, 0])

        # if self.img.dtype == "float32":
        #     luma = np.round(255 * luma).astype(np.uint8)

        # Filter the luma component of the image with a Gaussian LPF
        # Smoothing magnitude can be controlled with the sharpen_sigma parameter
        smoothened = ndimage.gaussian_filter(luma, self.sharpen_sigma)
        # Sharpen the image with upsharp mask
        # Strength is tuneable with the sharpen_strength parameter
        sharpened = luma + ((luma - smoothened) * self.sharpen_strength)

        if self.img.dtype == "float32":
            # sharpened = np.float32((sharpened) / 255.0)
            # out = sharpened.astype("float32")
            self.img[:, :, 0] = np.clip(sharpened, 0, 1)
        else:
            self.img[:, :, 0] = np.uint8(np.clip(sharpened, 0, 255))
        return self.img
