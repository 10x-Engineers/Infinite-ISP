"""
File: gamma_correction.py
Description: Implements the gamma look up table provided in the config file
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class GammaCorrection:
    """
    Gamma Correction
    """

    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.bit_depth = sensor_info["bit_depth"]
        self.parm_gmm = parm_gmm
        self.is_save = parm_gmm["is_save"]
        self.platform = platform

    def generate_gamma_8bit_lut(self):
        """
        Generates Gamma LUT for 8bit Image
        """
        lut = np.linspace(0, 255, 256)
        lut = np.uint8(np.round(255 * ((lut / 255) ** (1 / 2.2))))
        return lut

    def apply_gamma(self):
        """
        Apply Gamma LUT on n-bit Image
        """
        # load gamma table
        if self.bit_depth == 8:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_8"]))
        elif self.bit_depth == 10:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_10"]))
        elif self.bit_depth == 12:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_12"]))
        elif self.bit_depth == 14:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_14"]))
        else:
            print("LUT is not available for the given bit depth.")

        # apply LUT
        gamma_img = lut[self.img]
        return gamma_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_gamma_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Exceute Gamma Correction
        """
        print("Gamma Correction = " + str(self.enable))
        if self.enable is True:
            start = time.time()
            gc_out = self.apply_gamma()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = gc_out

        self.save()
        return self.img
