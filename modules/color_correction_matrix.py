import numpy as np


class ColorCorrectionMatrix:
    'Apply the color correction 3x3 matrix'
    def __init__(self, img, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm['isEnable']
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm

    def apply_ccm(self):
        r1 = np.array(self.parm_ccm['corrected_red'])
        r2 = np.array(self.parm_ccm['corrected_green'])
        r3 = np.array(self.parm_ccm['corrected_blue'])
        ccm_mat = np.float32([r1, r2, r3])
        self.ccm_mat = ccm_mat

        self.img = np.float32(self.img) / 255.0
        img1 = self.img.reshape(
            ((self.img.shape[0] * self.img.shape[1], 3)))
        out = np.matmul(img1, self.ccm_mat.transpose())
        out = np.float32(np.clip(out, 0, 1))
        out = out.reshape(self.img.shape).astype(self.img.dtype)
        out = np.uint8(out * 255)

        return out

    def execute(self):
        print('Color Correction Matrix = ' + str(self.enable))
        
        if self.enable == False:
            return self.img
        else:
            # g1 = np.array(self.parm_ccm['corrected_red'])
            # r1 = np.array(self.parm_ccm['corrected_red'])
            # r2 = np.array(self.parm_ccm['corrected_green'])
            # r3 = np.array(self.parm_ccm['corrected_blue'])
            # ccm_mat = np.float32([r1, r2, r3])
            # self.ccm_mat = ccm_mat

            # self.img = np.float32(self.img) / 255.0
            # img1 = self.img.reshape(
            #     ((self.img.shape[0] * self.img.shape[1], 3)))

            # customVar = self.ccm_mat.transpose
            # print(customVar)
            # out = np.matmul(customVar,img1)
            # out = np.float32(np.clip(out, 0, 1))
            # out = out.reshape(self.img.shape).astype(self.img.dtype)
            # out = np.uint8(out * 255)

            # return out
            return self.apply_ccm()
