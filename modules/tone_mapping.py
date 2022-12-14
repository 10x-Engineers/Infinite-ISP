# File: tone_mapping.py
# Description: 
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np


class ToneMapping:
    'Tone Mapping'

    def __init__(self, img, sensor_info, parm_tmp):
        self.img = img
        self.enable = parm_tmp['isEnable']
        self.sensor_info = sensor_info
        self.parm_tmp = parm_tmp

    def execute(self):
        print('Tone Mapping = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img
