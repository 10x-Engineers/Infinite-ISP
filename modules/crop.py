# File: crop.py
# Description: Crops the bayer image keeping cfa pattern intact
# Code / Paper  Reference: 
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np

class Crop:
    """
    Parameters:
    ----------
    img: 2D array
    new_size: 2-tuple with required height and width.
    
    Return:
    ------
    cropped_img: Generated coloured image of required size with same dtype 
    as the input image.
    """
    def __init__(self, img, sensor_info, parm_cro):
        self.img = img
        self.old_size = (sensor_info["height"], sensor_info["width"])
        self.new_size = (parm_cro["new_height"], parm_cro["new_width"])
        self.enable   = parm_cro["isEnable"]
        self.is_debug = parm_cro["isDebug"]
        self.update_sensor_info(sensor_info)

    def update_sensor_info(self, dict):
        if self.enable:
            if dict["height"]!= self.new_size[0] or dict["width"]!=self.new_size[1]:
                dict["height"] = self.new_size[0]
                dict["width"]  = self.new_size[1]
                dict["orig_size"]= str(self.img.T.shape)

    def crop(self, img, rows_to_crop=0, cols_to_crop=0):
        
        """
        Crop 2D array.
        Parameter:
        ---------
        img: image (2D array) to be cropped.
        rows_to_crop: Number of rows to crop. If it is an even integer, 
                      equal number of rows are cropped from either side of the image. 
                      Otherwise the image is cropped from the extreme right/bottom.
        cols_to_crop: Number of columns to crop. Works exactly as rows_to_crop.
        
        Output: cropped image
        """
        
        if rows_to_crop or cols_to_crop:
            if rows_to_crop%4==0 and cols_to_crop%4==0:
                img = img[rows_to_crop//2:-rows_to_crop//2, cols_to_crop//2:-cols_to_crop//2]
            else:
                print("   - Input/Output heights are not compatible." \
                      " Bayer pattern will be disturbed if cropped!")
        return img 
    
    def apply_cropping(self):
        if self.old_size==self.new_size:
            print('   - Output size is the same as input size.')
            return self.img
                
        if self.old_size[0]<self.new_size[0] or self.old_size[1]<self.new_size[1]: 
            print("   - Invalid output size {}x{}.\n   - Make sure output size is smaller than input size!".format( \
            self.new_size[0], self.new_size[1])) 
            return self.img

        crop_rows = self.old_size[0] - self.new_size[0]
        crop_cols = self.old_size[1] - self.new_size[1]
        cropped_img  = np.empty((self.new_size[0], self.new_size[1]), dtype=self.img.dtype)
        
    
        cropped_img = self.crop(self.img,crop_rows, crop_cols)
        
        if self.is_debug:
                print('   - Number of rows cropped = ', crop_rows)
                print('   - Number of columns cropped = ', crop_cols)
                print('   - Shape of cropped image = ', cropped_img.shape)
        return cropped_img
    
    def execute(self):
        
        print('Crop = ' + str(self.enable))
        if self.enable:
            cropped_img = self.apply_cropping()
            return cropped_img
        else:
            return self.img    
##########################################################################