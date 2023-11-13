"""
File: norm_2.py
Description: Implementation of Norm 2 GrayWorld - an AWB Algorithm
Code / Paper  Reference: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
Author: 10xEngineers

"""
import numpy as np


class PCAIlluminEstimation:
    """
    PCA Illuminant Estimation:

    This algorithm gets illuminant estimation directly from the color distribution
    The method that chooses bright and dark pixels using a projection distance in
    the color distribution and then applies principal component analysis to estimate
    the illumination direction
    """

    def __init__(self, flatten_img, pixel_percentage):
        self.flatten_img = flatten_img
        self.pixel_percentage = pixel_percentage

    def calculate_gains(self):
        """
        Calulate WB gains using normed average values of R, G and B channels
        """

        # Img flattened to Nx3 numpy array where N = heightxwidth to get only the color dist
        flat_img = self.flatten_img  # .flatten().reshape(-1,3)
        size = len(flat_img)

        # mean_vector is the direction vector for mean RGB obtained by dividing mean RBG vector
        # by its magnitude.
        mean_rgb = np.mean(flat_img, axis=0)
        mean_vector = mean_rgb / np.linalg.norm(mean_rgb)

        # To obtain dark and light pixels first distance projected of data on mean direction vector
        # is calculated
        data_p = np.sum(flat_img * mean_vector, axis=1)

        # Projected distance array is sorted in the ascending order to obtain light and dark pixels
        sorted_data = np.argsort(data_p)

        # Number of dark and light pixels are  calculated from pixel_percentage parameter
        index = int(np.ceil(size * (self.pixel_percentage / 100)))

        # Index of selective pixels (dark and light) is obtained
        filtered_index = np.concatenate(
            (sorted_data[0:index], sorted_data[-index:None])
        )
        # Selective pixels are retreived on the basis of index from 'data' array
        filtered_data = flat_img[filtered_index, :].astype(np.float32)

        # For first PCA a dot product of selected pixels data matrix with itself is calculated
        # and 3x3 matrix is obtained
        sigma = np.dot(filtered_data.transpose(), filtered_data)

        # Eigenvalues and vectors of the 3x3 matrix (sigma) are calculated
        eig_value, eig_vector = np.linalg.eig(sigma)

        # Eigenvector with maximum eigen value is the direction of iluminated estimation
        eig_vector = eig_vector[:, np.argsort(eig_value)]
        avg_rgb = np.abs(eig_vector[:, 2])

        # white balance gains G/R and G/B are calculated from RGB returned from AWB Algorithm
        # # 0 if nan is encountered
        rgain = np.nan_to_num(avg_rgb[1] / avg_rgb[0])
        bgain = np.nan_to_num(avg_rgb[1] / avg_rgb[2])
        return rgain, bgain
