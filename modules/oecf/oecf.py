"""
File: oecf.py
Description: Implements the opto electronic conversion function as a LUT
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class OECF:
    "Optical Electronic Conversion Function - correction"

    def __init__(self, img, platform, sensor_info, parm_oecf):
        self.img = img
        self.enable = parm_oecf["is_enable"]
        self.sensor_info = sensor_info
        self.parm_oecf = parm_oecf
        self.is_save = parm_oecf["is_save"]
        self.platform = platform

    def apply_oecf(self):
        """Execute OECF."""
        raw = self.img

        # get config parm
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]

        # duplicating r_lut here - when correcting add LUTs for each channel
        # in config.yml and load here
        rd_lut = np.uint16(np.array(self.parm_oecf["r_lut"]))
        gr_lut = np.uint16(np.array(self.parm_oecf["r_lut"]))
        gb_lut = np.uint16(np.array(self.parm_oecf["r_lut"]))
        bl_lut = np.uint16(np.array(self.parm_oecf["r_lut"]))

        raw_oecf = np.zeros(raw.shape)

        if bayer == "rggb":

            raw_oecf[0::2, 0::2] = rd_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = gr_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = gb_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = bl_lut[raw[1::2, 1::2]]

        elif bayer == "bggr":
            raw_oecf[0::2, 0::2] = bl_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = gb_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = gr_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = rd_lut[raw[1::2, 1::2]]

        elif bayer == "grbg":
            raw_oecf[0::2, 0::2] = gr_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = rd_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = bl_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = gb_lut[raw[1::2, 1::2]]

        elif bayer == "gbrg":
            raw_oecf[0::2, 0::2] = gb_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = bl_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = rd_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = gr_lut[raw[1::2, 1::2]]

        raw_oecf = np.uint16(np.clip(raw_oecf, 0, (2**bpp) - 1))
        return raw_oecf

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_oecf_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute OECF if enabled."""
        print("Optical Electronic Conversion Function = " + str(self.enable))

        if self.enable:
            start = time.time()
            oecf_out = self.apply_oecf()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = oecf_out
        self.save()
        return self.img
