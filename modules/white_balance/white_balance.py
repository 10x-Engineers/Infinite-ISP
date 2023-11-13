"""
File: white_balance.py
Description: Applies the white balance gains from the config file
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class WhiteBalance:
    """
    White balance Module
    """

    def __init__(self, img, platform, sensor_info, parm_wbc):
        """
        Class Constructor
        """
        self.img = img.copy()
        self.enable = parm_wbc["is_enable"]
        self.is_save = parm_wbc["is_save"]
        self.is_auto = parm_wbc["is_auto"]
        self.is_debug = parm_wbc["is_debug"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_wbc = parm_wbc
        self.bayer = self.sensor_info["bayer_pattern"]
        self.bpp = self.sensor_info["bit_depth"]
        self.raw = None

    def apply_wb_parameters(self):
        """
        Applies white balance gains from config file to raw images
        """

        # get config params
        redgain = self.parm_wbc["r_gain"]
        bluegain = self.parm_wbc["b_gain"]
        self.raw = np.float32(self.img)

        if self.is_debug:
            print("   - WB  - red gain : ", redgain)
            print("   - WB  - blue gain: ", bluegain)

        if self.bayer == "rggb":
            self.raw[::2, ::2] = self.raw[::2, ::2] * redgain
            self.raw[1::2, 1::2] = self.raw[1::2, 1::2] * bluegain
        elif self.bayer == "bggr":
            self.raw[::2, ::2] = self.raw[::2, ::2] * bluegain
            self.raw[1::2, 1::2] = self.raw[1::2, 1::2] * redgain
        elif self.bayer == "grbg":
            self.raw[1::2, ::2] = self.raw[1::2, ::2] * bluegain
            self.raw[::2, 1::2] = self.raw[::2, 1::2] * redgain
        elif self.bayer == "gbrg":
            self.raw[1::2, ::2] = self.raw[1::2, ::2] * redgain
            self.raw[::2, 1::2] = self.raw[::2, 1::2] * bluegain

        raw_whitebal = np.uint16(np.clip(self.raw, 0, (2**self.bpp) - 1))

        return raw_whitebal

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_white_balance_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute White Balance Module
        """

        if self.enable is True:
            print("White balancing = " + "True")
            start = time.time()
            wb_out = self.apply_wb_parameters()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = wb_out

        self.save()
        return self.img
