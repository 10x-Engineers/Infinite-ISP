# File: sharpen.py
# Description: 
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)

import numpy as np
from scipy import ndimage 

class Sharpening:
    def __init__(self, img, sensor_info, parm_sha):
         self.img = img
         self.enable = parm_sha['isEnable']
         self.sensor_info = sensor_info
         self.parm_sha = parm_sha

    def execute(self):
        print('Sharpening = ' + str(self.enable))

        if not self.enable:
            return self.img
        else:
            luma = np.float32(self.img[:, :, 0])
            smoothened = ndimage.gaussian_filter(luma, self.parm_sha['sharpen_sigma'])
            sharpened = luma + ((luma - smoothened) * self.parm_sha['sharpen_strength'])
            self.img[:, :, 0] = np.clip(sharpened, 0, 255)
            return np.uint8(self.img)