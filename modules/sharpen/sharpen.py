"""
File: sharpen.py
Description: Implements sharpening for Infinite-ISP.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
"""

import time
from modules.sharpen.unsharp_masking import UnsharpMasking as USM

from util.utils import save_output_array_yuv


class Sharpening:
    """
    Sharpening
    """

    def __init__(self, img, platform, sensor_info, parm_sha, conv_std):
        self.img = img
        self.enable = parm_sha["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sha = parm_sha
        self.is_save = parm_sha["is_save"]
        self.platform = platform
        self.conv_std = conv_std

    def apply_unsharp_masking(self):
        """
        Apply function for Shapening Algorithm - Unsharp Masking
        """
        usm = USM(
            self.img, self.parm_sha["sharpen_sigma"], self.parm_sha["sharpen_strength"]
        )
        return usm.apply_sharpen()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_Shapening_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Applying sharpening to input image
        """
        print("Sharpening = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            s_out = self.apply_unsharp_masking()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
