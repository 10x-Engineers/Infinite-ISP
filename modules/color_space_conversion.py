import numpy as np


class ColorSpaceConv:

    def __init__(self, img, sensor_info, parm_csc):
        self.img = img
        self.enable = parm_csc['isEnable']
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc

    def execute(self):
        print('Color Space Conversion = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img
