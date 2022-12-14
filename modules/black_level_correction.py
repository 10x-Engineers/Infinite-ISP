import numpy as np

class BlackLevelCorrection:
    'Black Level Correction'

    def __init__(self, img, sensor_info, parm_blc):
        self.img = img
        self.enable = parm_blc['isEnable']
        self.sensor_info = sensor_info
        self.param_blc = parm_blc

    def apply_blc_parameters(self):

        #get config parm
        bayer = self.sensor_info['bayer_pattern']
        bpp = self.sensor_info['bitdep']
        r_offset = self.param_blc['r_offset']
        gb_offset = self.param_blc['gb_offset']
        gr_offset = self.param_blc['gr_offset']
        b_offset = self.param_blc['b_offset']
        raw = np.float32(self.img)

        if bayer == 'rggb':
            raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset

        elif bayer == 'bggr':
            raw[0::2, 0::2] = raw[0::2, 0::2] - b_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gb_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gr_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - r_offset
        
        elif bayer == 'grbg':
            raw[0::2, 0::2] = raw[0::2, 0::2] - gr_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - r_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - b_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gb_offset
        
        elif bayer == 'gbrg':
            raw[0::2, 0::2] = raw[0::2, 0::2] - gb_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - b_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - r_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gr_offset

        raw_blc = np.uint16(np.clip(raw, 0, (2**bpp)-1))
        return raw_blc

    def execute(self):
        print('Black Level Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_blc_parameters()
