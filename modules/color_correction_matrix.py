import numpy as np


class ColorCorrectionMatrix:
    'Apply the color correction 3x3 matrix'
    def __init__(self, img, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm['isEnable']
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm

    def apply_ccm(self):
        #get ccm parm
        r1 = np.array(self.parm_ccm['corrected_red'])
        r2 = np.array(self.parm_ccm['corrected_green'])
        r3 = np.array(self.parm_ccm['corrected_blue'])
        
        ccm_mat = np.float32([r1, r2, r3])
        self.ccm_mat = ccm_mat

        #normalize 8bit to 0-1 img
        self.img = np.float32(self.img) / 255.0
        
        #convert to nx3
        img1 = self.img.reshape(
            ((self.img.shape[0] * self.img.shape[1], 3)))
        
        #keeping imatest convention of colum sum to 1 mat. O*A => A = ccm
        out = np.matmul(img1, self.ccm_mat.transpose())

        #clipping after ccm is must to eliminate neg values
        out = np.float32(np.clip(out, 0, 1))

        #convert back
        out = out.reshape(self.img.shape).astype(self.img.dtype)
        out = np.uint8(out * 255)

        return out

    def execute(self):
        print('Color Correction Matrix = ' + str(self.enable))
        
        if self.enable == False:
            return self.img
        else:
            return self.apply_ccm()
