"""
File: norm_2.py
Description: Implementation of Norm GrayWorld - an AWB Algorithm
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
Author: 10xEngineers

"""
import numpy as np


class NormGrayWorld:
    """
    Norm 2 Gray World White Balance:

    Gray world algorithm calculates white balance (G/R and G/B)
    by average values of RGB channels. Average values for each channel
    are calculated by norm-2
    """

    def __init__(self, flatten_img):
        self.flatten_img = flatten_img

    def calculate_gains(self):
        """
        Calulate WB gains using normed average values of R, G and B channels
        """
        avg_rgb = np.linalg.norm(self.flatten_img, axis=0)

        # white balance gains G/R and G/B are calculated from RGB returned from AWB Algorithm
        # 0 if nan is encountered
        rgain = np.nan_to_num(avg_rgb[1] / avg_rgb[0])
        bgain = np.nan_to_num(avg_rgb[1] / avg_rgb[2])

        return rgain, bgain
