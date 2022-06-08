import numpy as np

class BlackLevelCorrection:
    'Black Level Correction'
    def __init__(self, img, sensor_info, parm_blc):
         self.img = img
         self.enable = parm_blc['isEnable']
         self.sensor_info = sensor_info
         self.param_blc = parm_blc

    def execute(self):
        print('Black Level Correction = ' + str(self.enable))
        
        if self.enable == False:
            return self.img
        else:
            return self.img        

        

