"""
File: demosaic.py
Description: Implements the cfa interpolation algorithms
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array
from modules.demosaic.malvar_he_cutler import Malvar as MAL


class Demosaic:
    "CFA Interpolation"

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img
        self.bayer = sensor_info["bayer_pattern"]
        self.bit_depth = sensor_info["bit_depth"]
        self.is_save = parm_dga["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform

    def masks_cfa_bayer(self):
        """
        Generating masks for the given bayer pattern
        """
        pattern = self.bayer
        # dict will be creating 3 channel boolean type array of given shape with the name
        # tag like 'r_channel': [False False ....] , 'g_channel': [False False ....] ,
        # 'b_channel': [False False ....]
        channels = dict(
            (channel, np.zeros(self.img.shape, dtype=bool)) for channel in "rgb"
        )

        # Following comment will create boolean masks for each channel r_channel,
        # g_channel and b_channel
        for channel, (y_channel, x_channel) in zip(
            pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]
        ):
            channels[channel][y_channel::2, x_channel::2] = True

        # tuple will return 3 channel boolean pattern for r_channel,
        # g_channel and b_channel with True at corresponding value
        # For example in rggb pattern, the r_channel mask would then be
        # [ [ True, False, True, False], [ False, False, False, False]]
        return tuple(channels[c] for c in "rgb")

    def apply_cfa(self):
        """
        Demosaicing the given raw image using given algorithm
        """
        # 3D masks according to the given bayer
        masks = self.masks_cfa_bayer()
        mal = MAL(self.img, masks)
        demos_out = mal.apply_malvar()

        # Clipping the pixels values within the bit range
        demos_out = np.clip(demos_out, 0, 2**self.bit_depth - 1)
        demos_out = np.uint16(demos_out)
        return demos_out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Applying demosaicing to bayer image
        """
        print("CFA interpolation (default) = True")
        start = time.time()
        cfa_out = self.apply_cfa()
        print(f"  Execution time: {time.time() - start:.3f}s")
        self.img = cfa_out
        self.save()
        return self.img
