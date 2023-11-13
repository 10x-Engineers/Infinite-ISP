"""
File: black_level_correction.py
Description: Implements black level correction and image linearization based on config file params
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class BlackLevelCorrection:
    """
    Black Level Correction
    """

    def __init__(self, img, platform, sensor_info, parm_blc):
        self.img = img
        self.enable = parm_blc["is_enable"]
        self.sensor_info = sensor_info
        self.param_blc = parm_blc
        self.is_linearize = self.param_blc["is_linear"]
        self.is_save = parm_blc["is_save"]
        self.platform = platform

    def apply_blc_parameters(self):
        """
        Apply BLC parameters provided in config file
        """

        # get config parm
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]
        r_offset = self.param_blc["r_offset"]
        gb_offset = self.param_blc["gb_offset"]
        gr_offset = self.param_blc["gr_offset"]
        b_offset = self.param_blc["b_offset"]

        r_sat = self.param_blc["r_sat"]
        gr_sat = self.param_blc["gr_sat"]
        gb_sat = self.param_blc["gb_sat"]
        b_sat = self.param_blc["b_sat"]

        raw = np.float32(self.img)

        if bayer == "rggb":

            # implementing this formula with condition
            # ((img - blc) / (sat_level-blc)) * bitRange

            raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )

        elif bayer == "bggr":
            raw[0::2, 0::2] = raw[0::2, 0::2] - b_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gb_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gr_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - r_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )

        elif bayer == "grbg":
            raw[0::2, 0::2] = raw[0::2, 0::2] - gr_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - r_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - b_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gb_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )

        elif bayer == "gbrg":
            raw[0::2, 0::2] = raw[0::2, 0::2] - gb_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - b_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - r_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gr_offset

            if self.is_linearize is True:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )

        raw_blc = np.uint16(np.clip(raw, 0, (2**bpp) - 1))
        return raw_blc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Black Level Correction
        """
        print("Black Level Correction = " + str(self.enable))

        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img
