import numpy as np

class OECF:
    'Optical Electronic Conversion Function - correction'
    def __init__(self, img, sensor_info, parm_oecf):
         self.img = img
         self.enable = parm_oecf['isEnable']
         self.sensor_info = sensor_info
         self.parm_oecf = parm_oecf

    def execute(self):
        print('Optical Electronic Conversion Function = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_OECF()
    
    def apply_OECF(self):
        
        raw = self.img
        
        #get config parm
        bayer = self.sensor_info['bayer_pattern']
        bpp = self.sensor_info['bitdep']

        # duplicating r_lut here - when correcting add LUTs for each channel in config.yml and load here
        rd_lut = np.uint16(np.array(self.parm_oecf['r_lut']))
        gr_lut = np.uint16(np.array(self.parm_oecf['r_lut']))
        gb_lut = np.uint16(np.array(self.parm_oecf['r_lut']))
        bl_lut = np.uint16(np.array(self.parm_oecf['r_lut']))

        raw_oecf = np.zeros(raw.shape)

        if bayer == 'rggb':
            
            raw_oecf[0::2, 0::2] = rd_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = gr_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = gb_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = bl_lut[raw[1::2, 1::2]]

        elif bayer == 'bggr':
            raw_oecf[0::2, 0::2] = bl_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = gb_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = gr_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = rd_lut[raw[1::2, 1::2]]
        
        elif bayer == 'grbg':
            raw_oecf[0::2, 0::2] = gr_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = rd_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = bl_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = gb_lut[raw[1::2, 1::2]]
        
        elif bayer == 'gbrg':
            raw_oecf[0::2, 0::2] = gb_lut[raw[0::2, 0::2]]
            raw_oecf[0::2, 1::2] = bl_lut[raw[0::2, 1::2]]
            raw_oecf[1::2, 0::2] = rd_lut[raw[1::2, 0::2]]
            raw_oecf[1::2, 1::2] = gr_lut[raw[1::2, 1::2]]
        
        raw_oecf = np.uint16(np.clip(raw_oecf, 0, (2**bpp)-1))
        return raw_oecf
    
