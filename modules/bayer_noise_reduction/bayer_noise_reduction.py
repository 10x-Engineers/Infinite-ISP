"""
File: bayer_noise_reduction.py
Description: Noise reduction in bayer domain
Author: 10xEngineers
------------------------------------------------------------
"""


import time
from modules.bayer_noise_reduction.joint_bf import JointBF as JBF
from util.utils import save_output_array


class BayerNoiseReduction:
    """
    Noise Reduction in Bayer domain
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform

    def apply_bnr(self):
        """
        Apply bnr to the input image and return the output image
        """
        jbf = JBF(self.img, self.sensor_info, self.parm_bnr, self.platform)
        bnr_out_img = jbf.apply_jbf()
        return bnr_out_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_bayer_noise_reduction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Appling BNR to input RAW image and returns the output image
        """
        print("Bayer Noise Reduction = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            bnr_out = self.apply_bnr()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = bnr_out

        self.save()
        return self.img
