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

        self.isLinearize = self.param_blc['isLinear']
        r_sat = self.param_blc['r_sat']
        gr_sat = self.param_blc['gr_sat']
        gb_sat = self.param_blc['gb_sat']
        b_sat = self.param_blc['b_sat']


        raw = np.float32(self.img)

        if bayer == 'rggb':
            
            #implementing this formula with condition 
            # ((img - blc) / (sat_level-blc)) * bitRange

            raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset

            if self.isLinearize == True:
                raw[0::2, 0::2] = raw[0::2, 0::2] / (r_sat - r_offset) * ((2**bpp)-1)
                raw[0::2, 1::2] = raw[0::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp)-1)
                raw[1::2, 0::2] = raw[1::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp)-1)
                raw[1::2, 1::2] = raw[1::2, 1::2] / (b_sat - b_offset) * ((2**bpp)-1) 

        elif bayer == 'bggr':
            raw[0::2, 0::2] = raw[0::2, 0::2] - b_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - gb_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - gr_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - r_offset

            if self.isLinearize == True:
                raw[0::2, 0::2] = raw[0::2, 0::2] / (b_sat - b_offset) * ((2**bpp)-1)
                raw[0::2, 1::2] = raw[0::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp)-1)
                raw[1::2, 0::2] = raw[1::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp)-1)
                raw[1::2, 1::2] = raw[1::2, 1::2] / (r_sat - r_offset) * ((2**bpp)-1) 

        elif bayer == 'grbg':
            raw[0::2, 0::2] = raw[0::2, 0::2] - gr_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - r_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - b_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gb_offset

            if self.isLinearize == True:
                raw[0::2, 0::2] = raw[0::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp)-1)
                raw[0::2, 1::2] = raw[0::2, 1::2] / (r_sat - r_offset) * ((2**bpp)-1)
                raw[1::2, 0::2] = raw[1::2, 0::2] / (b_sat - b_offset) * ((2**bpp)-1)
                raw[1::2, 1::2] = raw[1::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp)-1) 

        elif bayer == 'gbrg':
            raw[0::2, 0::2] = raw[0::2, 0::2] - gb_offset
            raw[0::2, 1::2] = raw[0::2, 1::2] - b_offset
            raw[1::2, 0::2] = raw[1::2, 0::2] - r_offset
            raw[1::2, 1::2] = raw[1::2, 1::2] - gr_offset

            if self.isLinearize == True:
                raw[0::2, 0::2] = raw[0::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp)-1)
                raw[0::2, 1::2] = raw[0::2, 1::2] / (b_sat - b_offset) * ((2**bpp)-1)
                raw[1::2, 0::2] = raw[1::2, 0::2] / (r_sat - r_offset) * ((2**bpp)-1)
                raw[1::2, 1::2] = raw[1::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp)-1) 

        raw_blc = np.uint16(np.clip(raw, 0, (2**bpp)-1))
        return raw_blc

    def execute(self):
        print('Black Level Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_blc_parameters()
