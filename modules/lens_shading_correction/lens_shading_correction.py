"""
File: lens_shading_correction.py
Description:
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""


class LensShadingCorrection:
    """
    Lens Shading Correction
    """

    def __init__(self, img, platform, sensor_info, parm_lsc):
        self.img = img
        self.enable = parm_lsc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_lsc = parm_lsc
        self.platform = platform

    def execute(self):
        """
        Execute Lens Shading Correction
        """
        print("Lens Shading Correction = " + str(self.enable))

        if self.enable:
            return self.img
        else:
            return self.img
