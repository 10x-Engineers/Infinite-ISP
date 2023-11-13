"""
File: gray_world.py
Description: Implementation of Gray_World - an AWB Algorithm
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
Author: 10xEngineers

"""
import numpy as np


class GrayWorld:
    """
    Gray World White Balance:
    Gray world algorithm calculates white balance (G/R and G/B)
    by average values of RGB channels
    """

    def __init__(self, flatten_img):
        self.flatten_img = flatten_img

    def calculate_gains(self):
        """
        Calulate WB gains using average values of R, G and B channels
        """
        avg_rgb = np.mean(self.flatten_img, axis=0)

        # white balance gains G/R and G/B are calculated from RGB returned from AWB Algorithm
        # 0 if nan is encountered
        rgain = np.nan_to_num(avg_rgb[1] / avg_rgb[0])
        bgain = np.nan_to_num(avg_rgb[1] / avg_rgb[2])

        return rgain, bgain
