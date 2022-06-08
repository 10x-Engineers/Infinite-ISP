import numpy as np


class GammaCorrection:
    'Gamma Correction'

    def __init__(self, img, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm['isEnable']
        self.sensor_info = sensor_info
        self.parm_gmm = parm_gmm

    def execute(self):
        print('Gamma Correction = ' + str(self.enable))
        
        if self.enable == False:
            return self.img
        else:
            return self.img
