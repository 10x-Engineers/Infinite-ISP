"""
File: noise_reduction_2d.py
Description: Apply denoising algorithms on luminance channel
Author: 10xEngineers
------------------------------------------------------------
"""
import time
from util.utils import save_output_array_yuv
from modules.noise_reduction_2d.non_local_means import NLM


class NoiseReduction2d:
    """
    2D Noise Reduction
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform, conv_std):
        self.img = img
        self.enable = parm_2dnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.conv_std = conv_std
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_2dnr["is_save"]
        self.platform = platform

    def apply_2dnr(self):
        """
        Applying noise reduction algorithms (EBF, NLM, Mean, BF)
        """
        nlm = NLM(self.img, self.sensor_info, self.parm_2dnr, self.platform)
        return nlm.apply_nlm()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_2d_noise_reduction_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Executing 2D noise reduction module
        """
        print("Noise Reduction 2d = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            s_out = self.apply_2dnr()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
