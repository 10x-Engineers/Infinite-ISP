# File: sharpen.py
# Description: 
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)

import numpy as np

class Sharpening:
    def __init__(self, img, sensor_info, parm_sha):
         self.img = img
         self.enable = parm_sha['isEnable']
         self.sensor_info = sensor_info
         self.parm_sha = parm_sha

    def execute(self):
        print('Sharpening = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img