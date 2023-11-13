"""
File: color_space_conversion.py
Description: Converts RGB to YUV or YCbCr
Code / Paper  Reference: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
                         https://www.itu.int/rec/R-REC-BT.601/
                         https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
                         https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-
                         8-bit-yuv-formats-for-video-rendering
                         https://web.archive.org/web/20180423091842/http://www.equasys.de/
                         colorconversion.html
                         Author: 10xEngineers Pvt Ltd
------------------------------------------------------------------------------
"""

import time
import numpy as np
from util.utils import save_output_array_yuv, save_output_array


class RGBConversion:
    """
    YUV to RGB Conversion
    """

    def __init__(self, img, platform, sensor_info, parm_rgb, parm_csc):
        self.img = img.copy()
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_rgb = parm_rgb
        self.enable = self.parm_rgb["is_enable"]
        self.is_save = self.parm_rgb["is_save"]
        self.bit_depth = sensor_info["bit_depth"]
        self.conv_std = parm_csc["conv_standard"]
        self.yuv_img = img
        self.yuv2rgb_mat = None

    def yuv_to_rgb(self):
        """
        YUV-to-RGB Colorspace conversion 8bit
        """

        # make nx3 2d matrix of image
        mat_2d = self.yuv_img.reshape(
            (self.yuv_img.shape[0] * self.yuv_img.shape[1], 3)
        )

        # convert to 3xn for matrix multiplication
        mat2d_t = mat_2d.transpose()

        # subract the offsets
        mat2d_t = mat2d_t - np.array([[16, 128, 128]]).transpose()

        if self.conv_std == 1:
            # for BT. 709
            self.yuv2rgb_mat = np.array([[74, 0, 114], [74, -13, -34], [74, 135, 0]])
        else:
            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.yuv2rgb_mat = np.array([[64, 87, 0], [64, -44, -20], [61, 0, 105]])

        # convert to RGB
        rgb_2d = np.matmul(self.yuv2rgb_mat, mat2d_t)
        rgb_2d = rgb_2d >> 6

        # reshape the image back
        rgb2d_t = rgb_2d.transpose()
        self.yuv_img = rgb2d_t.reshape(self.yuv_img.shape).astype(np.float32)

        # clip the resultant img as it can have neg rgb values for small Y'
        self.yuv_img = np.float32(np.clip(self.yuv_img, 0, 255))

        # convert the image to [0-255]
        self.yuv_img = np.uint8(self.yuv_img)
        return self.yuv_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            if self.enable:
                save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_rgb_conversion_",
                    self.platform,
                    self.bit_depth,
                    self.sensor_info["bayer_pattern"],
                )
            else:
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_rgb_conversion_",
                    self.platform,
                    self.conv_std,
                )

    def execute(self):
        """
        Execute RGB Conversion
        """
        print("RGB Conversion" + " = " + str(self.enable))
        if self.enable:
            start = time.time()
            rgb_out = self.yuv_to_rgb()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = rgb_out
        self.save()
        return self.img
