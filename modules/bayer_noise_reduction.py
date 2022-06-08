import numpy as np


class BayerNoiseReduction:
    'Noise Reduction in Bayer domain'
    def __init__(self, img, sensor_info, parm_bnr):
        self.img = img
        self.enable = parm_bnr['isEnable']
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr

    def execute(self):
        print('Bayer Noise Reduction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.img
