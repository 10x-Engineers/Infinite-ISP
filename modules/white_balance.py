import numpy as np


class WhiteBalance:
    'White balancing'

    def __init__(self, img, sensor_info, parm_wbc):
        self.img = img
        self.enable = parm_wbc['isEnable']
        self.sensor_info = sensor_info
        self.parm_wbc = parm_wbc
        

    def apply_grayworld(self):

        # raw: raw input image
        # bayer: the bayer pattern of the raw image (rggb/bggr/grbg/gbrg)
        # bpp: the required bit-width of each pixel in the RAW image (8 to 16 bit)
        # returns greyworld white balanced raw image    
        
        self.bayer = self.sensor_info['bayer_pattern']
        self.bpp = self.sensor_info['bitdep']
        self.raw = self.img
        
        self.raw = np.float32(self.raw)
        if self.bayer == 'rggb':
            red_mean = self.raw[::2, ::2].mean()
            green_mean = (self.raw[::2, 1::2].mean() + self.raw[1::2, ::2].mean()) / 2
            blue_mean = self.raw[1::2, 1::2].mean()
        elif self.bayer == 'bggr':
            blue_mean = self.raw[::2, ::2].mean()
            green_mean = (self.raw[::2, 1::2].mean() + self.raw[1::2, ::2].mean()) / 2
            red_mean = self.raw[1::2, 1::2].mean()
        elif self.bayer == 'grbg':
            red_mean = self.raw[::2, 1::2].mean()
            green_mean = (self.raw[::2, ::2].mean() + self.raw[1::2, 1::2].mean()) / 2
            blue_mean = self.raw[1::2, ::2].mean()
        elif self.bayer == 'gbrg':
            red_mean = self.raw[1::2, ::2].mean()
            green_mean = (self.raw[::2, ::2].mean() + self.raw[1::2, 1::2].mean()) / 2
            blue_mean = self.raw[::2, 1::2].mean()

        redgain, bluegain = green_mean/red_mean, green_mean/blue_mean

        if self.bayer == 'rggb':
            self.raw[::2, ::2] = self.raw[::2, ::2] * redgain
            self.raw[1::2, 1::2] = self.raw[1::2, 1::2] * bluegain
        elif self.bayer == 'bggr':
            self.raw[::2, ::2] = self.raw[::2, ::2] * bluegain
            self.raw[1::2, 1::2] = self.raw[1::2, 1::2] * redgain
        elif self.bayer == 'grbg':
            self.raw[1::2, ::2] = self.raw[1::2, ::2] * bluegain
            self.raw[::2, 1::2] = self.raw[::2, 1::2] * redgain
        elif self.bayer == 'gbrg':
            self.raw[1::2, ::2] = self.raw[1::2, ::2] * redgain
            self.raw[::2, 1::2] = self.raw[::2, 1::2] * bluegain

        raw_whitebal = np.uint16(np.clip(self.raw, 0, (2**self.bpp)-1))

        return raw_whitebal

    def execute(self):
        print('White balancing = ' + str(self.enable))

        if self.enable == False:
            return self.raw

        return self.apply_grayworld()

    
        