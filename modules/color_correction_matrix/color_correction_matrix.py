"""
File: color_correction_matrix.py
Description: Applies the 3x3 correction matrix on the image
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class ColorCorrectionMatrix:
    "Apply the color correction 3x3 matrix"

    def __init__(self, img, platform, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.bit_depthth = sensor_info["bit_depth"]
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform

    def apply_ccm(self):
        """
        Apply CCM Params
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.float32([r_1, r_2, r_3])

        # normalize nbit to 0-1 img
        self.img = np.float32(self.img) / (2**self.bit_depthth - 1)

        # convert to nx3
        img1 = self.img.reshape(((self.img.shape[0] * self.img.shape[1], 3)))

        # keeping imatest convention of colum sum to 1 mat. O*A => A = ccm
        out = np.matmul(img1, self.ccm_mat.transpose())

        # clipping after ccm is must to eliminate neg values
        out = np.float32(np.clip(out, 0, 1))

        # convert back
        out = out.reshape(self.img.shape).astype(self.img.dtype)
        out = np.uint16(out * (2**self.bit_depthth - 1))

        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute ccm if enabled."""
        print("Color Correction Matrix = " + str(self.enable))

        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img
