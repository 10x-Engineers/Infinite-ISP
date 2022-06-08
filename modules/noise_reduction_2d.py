import numpy as np


class NoiseReduction2d:
    def __init__(self, img, sensor_info, parm_2dn):
        self.img = img
        self.enable = parm_2dn['isEnable']
        self.sensor_info = sensor_info
        self.parm_2dn = parm_2dn

    def execute(self):
        print('Noise Reduction 2d = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img
