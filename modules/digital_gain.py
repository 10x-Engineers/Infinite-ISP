import numpy as np


class DigitalGain:
    'Digital Gain'

    def __init__(self, img, sensor_info, parm_dga):
        self.img = img
        self.enable = parm_dga['isEnable']
        self.sensor_info = sensor_info
        self.param_dga = parm_dga

    def apply_digitalGain(self):

        # get desired param from config
        bpp = self.sensor_info['bitdep']
        dg = self.param_dga['dg_gain']

        #converting to float image
        self.img = np.float32(self.img)

        #apply gain
        self.img = dg * self.img

        #np.uint16bit to contain the bpp bit raw
        self.img = np.uint16(np.clip(self.img, 0, ((2**bpp)-1)))
        return self.img

    def execute(self):
        print('Digital Gain = ' + str(self.enable))

        if self.enable == False:
            return self.img
        return self.apply_digitalGain()
