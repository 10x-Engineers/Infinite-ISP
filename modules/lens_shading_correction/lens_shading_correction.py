"""
File: lens_shading_correction.py
Description: Corrects lens shading (vignetting and color shading) with per-channel
gain grids from flat-field calibration, bilinearly interpolated to every pixel
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import (
    save_output_array,
    get_bayer_channel_offsets,
    bilinear_interp_weights,
)


class LensShadingCorrection:
    """
    Lens Shading Correction
    """

    def __init__(self, img, platform, sensor_info, parm_lsc):
        self.img = img
        self.enable = parm_lsc["is_enable"]
        self.is_save = parm_lsc["is_save"]
        self.sensor_info = sensor_info
        self.parm_lsc = parm_lsc
        self.platform = platform

    def apply_lsc_gain_grids(self):
        """
        Multiply each bayer channel by its gain grid from the config file. The grid
        nodes span the channel from its first to its last pixel and the gain between
        nodes is bilinearly interpolated.
        """

        # get config parm
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]

        raw = np.float32(self.img)

        for channel, (row, col) in get_bayer_channel_offsets(bayer).items():
            gain_grid = np.array(self.parm_lsc[channel + "_gain_grid"], dtype=np.float64)
            plane = raw[row::2, col::2]

            # Bilinear interpolation of the grid to the channel size:
            # (height x rows) @ (rows x cols) @ (cols x width)
            weight_y = bilinear_interp_weights(plane.shape[0], gain_grid.shape[0])
            weight_x = bilinear_interp_weights(plane.shape[1], gain_grid.shape[1])
            gain_map = weight_y @ gain_grid @ weight_x.T

            raw[row::2, col::2] = plane * np.float32(gain_map)

        raw_lsc = np.uint16(np.clip(raw, 0, (2**bpp) - 1))
        return raw_lsc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_lens_shading_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Lens Shading Correction
        """
        print("Lens Shading Correction = " + str(self.enable))

        if self.enable:
            start = time.time()
            lsc_out = self.apply_lsc_gain_grids()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = lsc_out
        self.save()
        return self.img
