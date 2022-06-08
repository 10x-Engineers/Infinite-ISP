import numpy as np


class DeadPixelCorrection:
    'Dead Pixel Correction'

    def __init__(self, img, sensor_info, parm_dpc):
        self.img = img
        self.enable = parm_dpc['isEnable']
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc

    def execute(self):
        print('Dead Pixel Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            # Write here

            return self.img
