"""
File: digital_gain.py
Description: Applies the digital gain gain_array[current_gain].  With is_auto, the
auto-exposure control (EV law) chooses current_gain for the next frame - digital gain
stands in for the sensor exposure that the firmware AE moves.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class DigitalGain:
    """
    Digital Gain
    """

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img.copy()
        self.is_save = parm_dga["is_save"]
        self.is_debug = parm_dga["is_debug"]
        self.gains_array = parm_dga["gain_array"]
        self.current_gain = parm_dga["current_gain"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.param_dga = parm_dga

    def apply_digital_gain(self):
        """
        Apply Digital Gain gain_array[current_gain] - current_gain from the config,
        or chosen by the AE control when is_auto
        """

        # get desired param from config
        bpp = self.sensor_info["bit_depth"]

        # converting to float image
        self.img = np.float32(self.img)

        # Gain_Array is an array of pre-defined digital gains for ISP
        self.img = self.gains_array[self.current_gain] * self.img

        if self.is_debug:
            print("   - DG  - Applied Gain = ", self.gains_array[self.current_gain])

        # np.uint16 bit to contain the bpp bit raw
        self.img = np.uint16(np.clip(self.img, 0, ((2**bpp) - 1)))
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_digital_gain_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Digital Gain Module
        """
        print("Digital Gain (default) = True ")

        start = time.time()
        dg_out = self.apply_digital_gain()
        print(f"  Execution time: {time.time() - start:.3f}s")
        self.img = dg_out
        self.save()
        return self.img, self.current_gain
