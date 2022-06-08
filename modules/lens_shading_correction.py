import numpy as np


class LensShadingCorrection:
    'Lens Shading Correction'

    def __init__(self, img, sensor_info, parm_lsc):
        self.img = img
        self.enable = parm_lsc['isEnable']
        self.sensor_info = sensor_info
        self.parm_lsc = parm_lsc

    def execute(self):
        print('Lens Shading Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img
