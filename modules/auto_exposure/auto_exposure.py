"""
File: auto_exposure.py
Description: 3A-AE Runs the Auto exposure algorithm in a loop
Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
                         http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np


class AutoExposure:
    """
    Auto Exposure Module
    """

    def __init__(self, img, sensor_info, parm_ae):
        self.img = img
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.center_illuminance = parm_ae["center_illuminance"]
        self.histogram_skewness_range = parm_ae["histogram_skewness"]
        self.sensor_info = sensor_info
        self.param_ae = parm_ae
        self.bit_depth = sensor_info["bit_depth"]

        # Pipeline modules included in AE Feedback Loop
        # (White Balance) wb module is renamed to wbc (white balance correction)
        # gc (Gamma Correction) module is renamed to gcm (Gamma Correction Module)

    def get_exposure_feedback(self):
        """
        Get Correct Exposure by Adjusting Digital Gain
        """
        # Convert Image into 8-bit for AE Calculation
        self.img = self.img >> (self.bit_depth - 8)
        self.bit_depth = 8

        # calculate the exposure metric
        return self.determine_exposure()

    def determine_exposure(self):
        """
        Image Exposure Estimation using Skewness Luminance of Histograms
        """

        # plt.imshow(self.img)
        # plt.show()

        # For Luminance Histograms, Image is first converted into greyscale image
        # Function also returns average luminance of image which is used as AE-Stat
        grey_img, avg_lum = self.get_greyscale_image(self.img)
        print("Average luminance is = ", avg_lum)

        # Histogram skewness Calculation for AE Stats
        skewness = self.get_luminance_histogram_skewness(grey_img)

        # get the ranges
        upper_limit = self.histogram_skewness_range
        lower_limit = -1 * upper_limit

        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)

        # see if skewness is within range
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def get_greyscale_image(self, img):
        """
        Conversion of an Image into Greyscale Image
        """
        # Each RGB pixels with [0.299, 0.587, 0.144] to get its luminance
        grey_img = np.clip(
            np.dot(img[..., :3], [0.299, 0.587, 0.144]), 0, (2**self.bit_depth)
        ).astype(np.uint16)
        return grey_img, np.average(grey_img, axis=(0, 1))

    def get_luminance_histogram_skewness(self, img):
        """
        Skewness Calculation in reference to:
        Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics
        Tables and Formulae. Chapman & Hall: New York. 2000. Section 2.2.24.1
        """

        # First subtract central luminance to calculate skewness around it
        img = img.astype(np.float64) - self.center_illuminance

        # The sample skewness is computed as the Fisher-Pearson coefficient of
        # skewness, i.e. (m_3 / m_2**(3/2)) * g_1
        # where m_2 is 2nd moment (variance) and m_3 is third moment skewness

        img_size = img.size
        m_2 = np.sum(np.power(img, 2)) / img_size
        m_3 = np.sum(np.power(img, 3)) / img_size

        g_1 = np.sqrt(img_size * (img_size - 1)) / (img_size - 2)
        skewness = np.nan_to_num((m_3 / abs(m_2) ** (3 / 2)) * g_1)

        if self.is_debug:
            print("   - AE - Histogram Skewness = ", skewness)

        return skewness

    def execute(self):
        """
        Execute Auto Exposure
        """
        print("Auto Exposure= " + str(self.enable))

        if self.enable is False:
            return None
        else:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  Execution time: {time.time()-start:.3f}s")
            return ae_feedback
