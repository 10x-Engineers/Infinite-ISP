"""
File: saturation_enhancement.py
Description: Saturation enhancement in YUV. Runs after 2D noise reduction, before
             RGB conversion. Y passes through; each chroma sample is scaled about
             the chroma pedestal of the 8-bit YUV that colour space conversion
             produces and rgb_conversion undoes:

                 C' = clip((C - 128) * saturation_gain + 128)

             This used to run inside color_space_conversion, applied before the
             pedestal was added there (so no explicit subtract/re-add was needed).
             It's now its own module, like every other pipeline stage, and runs
             later - on the pedestal-carrying 8-bit YUV - so the pedestal is
             subtracted and re-added explicitly instead.

             Algorithm-design counterpart of Infinite-ISP_ReferenceModel's
             saturation_enhancement module ("global" algorithm) - same formula,
             without the fixed-point register modeling.
Code / Paper Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array_yuv

CSC_OFFSET = 128  # chroma pedestal of the 8-bit YUV that CSC produces


class SaturationEnhancement:
    """
    Saturation Enhancement
    """

    def __init__(self, img, platform, sensor_info, parm_se, conv_std):
        self.img = img.copy()
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_se["is_enable"]
        self.is_save = parm_se["is_save"]
        self.saturation_gain = float(parm_se["saturation_gain"])
        self.conv_std = conv_std

    def apply_saturation(self):
        """
        Scale Cb / Cr about the chroma pedestal; Y is unchanged
        """
        out = self.img.astype(np.float32)
        out[:, :, 1] = (out[:, :, 1] - CSC_OFFSET) * self.saturation_gain + CSC_OFFSET
        out[:, :, 2] = (out[:, :, 2] - CSC_OFFSET) * self.saturation_gain + CSC_OFFSET
        return np.clip(out, 0, 255).astype(np.uint8)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_saturation_enhancement_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute Saturation Enhancement
        """
        print("Saturation Enhancement = " + str(self.enable))
        if self.enable:
            start = time.time()
            print(f"   - SE  - saturation_gain = {self.saturation_gain}")
            self.img = self.apply_saturation()
            print(f"  Execution time: {time.time() - start:.3f}s")
        self.save()
        return self.img
