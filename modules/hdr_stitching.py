import numpy as np

class HdrStitching:
    'HDR Stitching'
    def __init__(self, img, sensor_info, parm_hdr):
         self.img = img
         self.enable = parm_hdr['isEnable']
         self.sensor_info = sensor_info
         self.parm_hdr = parm_hdr

    def execute(self):
        print('HDR Stitching = ' + str(self.enable))

        if self.enable == False:
            
            return self.img
        else:

            # Write here 
            return self.img