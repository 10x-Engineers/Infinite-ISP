"""
File: dead_pixel_correction.py
Description: Corrects the hot or dead pixels
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array
from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC as DynDPC



class DeadPixelCorrection:
    "Dead Pixel Correction"

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["is_debug"]
        self.is_save = parm_dpc["is_save"]
        self.platform = platform

    def padding(self):
        """Return a mirror padded copy of image."""

        img_pad = np.pad(self.img, (2, 2), "reflect")
        return img_pad

    def apply_dynamic_dpc(self):
        """Apply DPC"""
        dpc = DynDPC(self.img, self.sensor_info, self.parm_dpc)
        return dpc.dynamic_dpc()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_dead_pixel_correction_",
                self.platform,
                self.bpp,
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute DPC Module"""

        print("Dead Pixel Correction = " + str(self.enable))

        if self.enable:
            start = time.time()
            self.img = np.float32(self.img)
            dpc_out = self.apply_dynamic_dpc()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = dpc_out

        self.save()
        return self.img
