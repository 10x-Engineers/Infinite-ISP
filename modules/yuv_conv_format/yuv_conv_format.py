"""
File: yuv_conv_format.py
Description:
Code / Paper  Reference:
- https://web.archive.org/web/20190220164028/http://www.sunrayimage.com/examples.html
- https://en.wikipedia.org/wiki/YUV
- https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-
  for-video-rendering
- https://www.flir.com/support-center/iis/machine-vision/knowledge-base/understanding-yuv-
  data-formats/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import re
import numpy as np

from util.utils import save_output_array


class YUVConvFormat:
    "YUV Conversion Formats - 444, 442"

    def __init__(self, img, platform, sensor_info, parm_yuv):  # parm_csc):
        self.img = img
        self.shape = img.shape
        self.enable = parm_yuv["is_enable"]
        # self.is_csc_enable = parm_csc['is_enable']
        self.sensor_info = sensor_info
        self.param_yuv = parm_yuv
        self.is_save = parm_yuv["is_save"]
        self.platform = platform
        self.in_file = self.platform["in_file"]

    def convert2yuv_format(self):
        """Execute YUV conversion."""
        conv_type = self.param_yuv["conv_type"]

        if conv_type == "422":
            y_0 = self.img[:, 0::2, 0].reshape(-1, 1)
            u_ch = self.img[:, 0::2, 1].reshape(-1, 1)
            v_ch = self.img[:, 0::2, 2].reshape(-1, 1)
            y_1 = self.img[:, 1::2, 0].reshape(-1, 1)
            yuv = np.concatenate([y_0, u_ch, y_1, v_ch], axis=1)

        elif conv_type == "444":
            y_0 = self.img[:, :, 0].reshape(-1, 1)
            u_0 = self.img[:, :, 1].reshape(-1, 1)
            v_0 = self.img[:, :, 2].reshape(-1, 1)
            yuv = np.concatenate([y_0, u_0, v_0], axis=1)

        out_path = "./out_frames/out_" + self.in_file + ".yuv"

        raw_wb = open(out_path, "wb")
        yuv.flatten().tofile(raw_wb)
        raw_wb.close()

        return yuv.flatten()

    def save(self):
        """
        Function to save module output
        """
        # update size of array in filename
        self.in_file = re.sub(
            r"\d+x\d+", f"{self.shape[1]}x{self.shape[0]}", self.in_file
        )
        if self.is_save:
            # save format for yuv_conversion_format is .npy only
            save_format = self.platform["save_format"]
            self.platform["save_format"] = "npy"

            save_output_array(
                self.in_file,
                self.img,
                f"Out_yuv_conversion_format_{self.param_yuv['conv_type']}_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )
            # restore the original save format
            self.platform["save_format"] = save_format

    def execute(self):
        """Execute YUV conversion if enabled."""
        print(
            "YUV conversion format "
            + self.param_yuv["conv_type"]
            + " = "
            + str(self.enable)
        )

        if self.enable:
            if self.platform["rgb_output"]:
                print("Invalid input for YUV conversion: RGB image format.")
                self.param_yuv["is_enable"] = False
            else:
                start = time.time()
                yuv = self.convert2yuv_format()
                print(f"  Execution time: {time.time() - start:.3f}s")
                self.img = yuv

        self.save()
        return self.img
