import numpy as np
import colour_demosaicing as cd
from matplotlib import pyplot as plt


class CFAInterpolation:
    'CFA interpolation (demosaicing)'

    def __init__(self, img, sensor_info, parm_dem):
        self.img = img
        self.enable = parm_dem['isEnable']
        self.sensor_info = sensor_info
        self.parm_dem = parm_dem

    def demosaic_raw(self):

        #get config parm
        bpp = self.sensor_info['bitdep']
        bayer = self.sensor_info['bayer_pattern']
        
        #normalize img
        self.img = np.float32(self.img) / ((2**bpp)-1)

        # convert to 8bit raw for input compatibility
        hs_raw = np.uint8(self.img*255)
        img = np.uint8(cd.demosaicing_CFA_Bayer_bilinear(hs_raw, bayer))
        return img

    def execute(self):
        print('CFA interpolation (demosaicing) = ' + str(self.enable))
        
        if self.enable == False:
            return self.img
        return self.demosaic_raw()
