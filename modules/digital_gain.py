# File: digital_gain.py
# Description: Applies the digital gain based on config file also interacts with AE when adjusting exposure
# Code / Paper  Reference:                          
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np


class DigitalGain:
    'Digital Gain'

    def __init__(self, img, sensor_info, parm_dga):
        self.img = img
        #self.enable = parm_dga['isEnable']
        self.debug = parm_dga['isDebug']
        self.gains_array = parm_dga['gain_array']
        self.current_gain = parm_dga['current_gain']
        self.sensor_info = sensor_info
        self.param_dga = parm_dga



    def apply_digitalGain(self, ae_correction):

        # get desired param from config
        bpp = self.sensor_info['bitdep']
        # dg = self.param_dga['dg_gain']

        #converting to float image
        self.img = np.float32(self.img)

        # Gains are applied on the basis of AE-Feedback.
        # 'ae_correction == 0' - Default Gain is applied before AE feedback
        # 'ae_correction > 0' - Image is overexposed  (No action required)
        # 'ae_correction < 0' - Image is underexposed

        if ae_correction < 0:
            # max/min functions is applied to not allow digital gains exceed the defined limits
            self.current_gain = min(len(self.gains_array)-1,  self.current_gain+1)
            
        # Gain_Array is an array of pre-defined digital gains for ISP
        self.img =  self.gains_array[self.current_gain] * self.img
        if self.debug:
            print('   - DG - Applied Gain = ', self.gains_array[self.current_gain])

        #np.uint16 bit to contain the bpp bit raw
        self.img = np.uint16(np.clip(self.img, 0, ((2**bpp)-1)))
        return self.img

    def execute(self, ae_correction = 0):
        print('Digital Gain (default) = True ' )

        # # If digital gain is disabled return image as it is.
        # if self.enable == False:
        #     return self.img

        # ae_coorection indicated if the gain is default digital gain or AE-correction gain.
        return self.apply_digitalGain(ae_correction)
