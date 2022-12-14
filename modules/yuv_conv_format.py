# File: yuv_conv_format.py
# Description: 
# Code / Paper  Reference:  https://web.archive.org/web/20190220164028/http://www.sunrayimage.com/examples.html
#                           https://en.wikipedia.org/wiki/YUV
#                           https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-8-bit-yuv-formats-for-video-rendering
#                           https://www.flir.com/support-center/iis/machine-vision/knowledge-base/understanding-yuv-data-formats/
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt


class YUVConvFormat:
    'YUV Conversion Formats - 444, 442'


    def __init__(self, img, sensor_info, parm_yuv, inputfile_name, parm_csc):
        self.img = img
        self.enable = parm_yuv['isEnable']
        #self.is_csc_enable = parm_csc['isEnable']
        self.sensor_info = sensor_info
        self.param_yuv = parm_yuv
        self.infile = inputfile_name

    def convert2yuv_format(self):
        conv_type = self.param_yuv['conv_type']

        if conv_type == '422':
            y0 = self.img[:, 0::2, 0].reshape(-1, 1)
            u = self.img[:, 0::2, 1].reshape(-1, 1)
            v = self.img[:, 0::2, 2].reshape(-1, 1)
            y1 = self.img[:, 1::2, 0].reshape(-1, 1)
            yuv = np.concatenate([y0, u, y1, v], axis=1)
            
        elif conv_type == '444':
            y0 = self.img[:,:,0].reshape(-1,1)
            u0 = self.img[:,:,1].reshape(-1,1)
            v0 = self.img[:,:,2].reshape(-1,1)
            yuv = np.concatenate([y0, u0, v0], axis=1)

        out_path = './out_frames/out_' + self.infile + '.yuv'

        raw_wb = open(out_path, 'wb')
        yuv.flatten().tofile(raw_wb)
        raw_wb.close()
        
    def execute(self):
        print('YUV conversion format '+ self.param_yuv['conv_type'] + ' = ' + str(self.enable))

        if (self.enable) == False:
            return self.img
        else:
            return self.convert2yuv_format()
