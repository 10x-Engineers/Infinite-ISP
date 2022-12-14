import numpy as np


class GammaCorrection:
    'Gamma Correction'

    def __init__(self, img, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm['isEnable']
        self.sensor_info = sensor_info
        self.parm_gmm = parm_gmm

    def generate_gamma8bit_LUT(self):
        lut = np.linspace(0, 255, 256)
        lut = np.uint8(np.round(255 * ((lut/255) ** (1/2.2))))
        return lut

    def apply_gamma(self):
        # load gamma uint8 table
        lut = np.uint8(np.array(self.parm_gmm['gammaLut']))
        
        #apply LUT
        gamma_img = lut[self.img]
        return gamma_img

    def execute(self):
        print('Gamma Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_gamma()
