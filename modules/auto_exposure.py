# File: auto_exposure.py
# Description: 3A-AE Runs the Auto exposure algorithm in a loop
# Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
#                          http://tomlr.free.fr/Math%E9matiques/Math%20Complete/Probability%20and%20statistics/CRC%20-%20standard%20probability%20and%20Statistics%20tables%20and%20formulae%20-%20DANIEL%20ZWILLINGER.pdf
# Author: xx-isp (ispinfinite@gmail.com)
#------------------------------------------------------------

import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt


class AutoExposure:
    'Auto Exposure'

    def __init__(self, img, sensor_info, parm_ae, oecf_raw, dga, lsc, bnr, wb, cfa_inter, awb,  ccm, gc):
        self.img = img
        self.enable = parm_ae['isEnable']
        self.debug = parm_ae['isDebug']
        self.center_illuminance = parm_ae['center_illuminance']
        self.histogram_skewness_range = parm_ae['histogram_skewness']
        self.sensor_info = sensor_info
        self.param_ae = parm_ae

        # Pipeline modules included in AE Feedback Loop
        self.dga = dga
        self.lsc = lsc
        self.bnr = bnr
        self.wb = wb
        self.cfa_inter = cfa_inter
        self.awb = awb
        self.ccm = ccm
        self.gc = gc
        self.oecf_raw = oecf_raw
        self.ae_iterations = 0


    def get_correct_exposure(self):
        
        while (True):
            
             #calculate the exposure metric
            correction = self.determine_exposure()

            # check if correction is needed
            if correction == 0:

                #Check whether the input image was already correct 
                if self.ae_iterations > 0:

                    #Go through the final pipeline
                    self.generate_final_image(correction)
                    
                # No correction is needed image is already correct
                return self.img
            else:
                self.apply_digital_gain(correction)



    def determine_exposure(self):

        # plt.imshow(self.img)
        # plt.show()

        # For Luminance Histograms, Image is first converted into greyscale image
        # Function also returns average luminance of image which is used as AE-Stat
        grey_img, avg_luminance = self.get_greyscale_image(self.img)

        # Histogram skewness Calculation for AE Stats
        skewness = self.get_luminance_histogram_skewness(grey_img)
        
        if self.debug:
            # To Visulaize luminance Histogram of an Image.
            # hist, bin_edges = np.histogram(grey_img,  range=(0, 255), bins=255, density=True)
            # plt.bar(bin_edges[:-1], hist, width = 1)
            # plt.xlim(min(bin_edges), max(bin_edges))
            # plt.show()
            print('   - AE - Histogram Skewness = ', skewness)

        #get the ranges
        lower_limit = -1 * self.histogram_skewness_range
        upper_limit = 1 * self.histogram_skewness_range
        
        # Cannot apply anymore gains so return 0 so that final image is generated 
        if(self.dga.current_gain >= len(self.dga.gains_array)-1):
            print('   - DG gain limit reached')
            return 0

        #see if skewness is within range
        if(skewness >= lower_limit):
            return 0
        else:
            return skewness


    def apply_digital_gain(self, correction):

        # Digital Gain is applied at the start of ISP Pipline
        # In order to calculate AE-Stats all processing between Digital Gain and Auto Exposure is done again.
        
        self.ae_iterations += 1
        self.dga.img = self.oecf_raw
        dga_raw = self.dga.execute(correction)

        #  White balancing
        self.wb.img  = dga_raw
        wb_raw = self.wb.execute()

        #  CFA demosaicing
        self.cfa_inter.img  = wb_raw
        demos_img = self.cfa_inter.execute()

        # Auto WHite Balance
        self.awb.img = demos_img
        awb_img = self.awb.execute()

        #  Color correction matrix
        self.ccm.img  = awb_img
        ccm_img = self.ccm.execute()

        #  Gamma
        self.gc.img  = ccm_img
        self.img = self.gc.execute()

    def generate_final_image(self, correction):

        # Digital Gain is applied at the start of ISP Pipline
        # In order to calculate AE-Stats all processing between Digital Gain and Auto Exposure is done again.
        self.dga.img = self.oecf_raw
        dga_raw = self.dga.execute(correction)

        #  Lens shading correction
        self.lsc.img = dga_raw
        lsc_raw = self.lsc.execute()

        #  Bayer noise reduction
        self.bnr.img = lsc_raw
        bnr_raw = self.bnr.execute()

        #  White balancing
        self.wb.img  = bnr_raw
        wb_raw = self.wb.execute()

        #  CFA demosaicing
        self.cfa_inter.img  = wb_raw
        demos_img = self.cfa_inter.execute()

        self.awb.img = demos_img
        awb_img = self.awb.execute()

        #  Color correction matrix
        self.ccm.img  = awb_img
        ccm_img = self.ccm.execute()

        #  Gamma
        self.gc.img  = ccm_img
        self.img = self.gc.execute()




    def get_greyscale_image(self, img):

        # Formula for Conversion of an Image into Greyscale Image.
        grey_img = np.clip(np.dot(img[...,:3], [0.299, 0.587, 0.144]), 0, 255).astype(np.uint8)
        return grey_img, np.average(grey_img, axis=(0,1))

    def get_luminance_histogram_skewness(self, img): 
       
        # Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics Tables and Formulae. Chapman & Hall: New York. 2000. Section 2.2.24.1
        
        # First subtract central luminance to calculate skewness around it
        img = img-self.center_illuminance 

        # The sample skewness is computed as the Fisher-Pearson coefficient of skewness, i.e. g1 = (m3 / m2**(3/2)) * G1
        # where m2 is 2nd moment (variance) and m3 is third moment skewness
        img = (img.astype(np.float64)-np.average(img))
        N =  img.size
        m2 = np.sum(np.power(img,2))/N
        m3 = np.sum(np.power(img,3))/N

        G1 = np.sqrt(N*(N-1))/(N-2)
        return np.nan_to_num((m3/abs(m2)**(3/2))*G1)
        

    def execute(self):
        print('Auto Exposure= ' + str(self.enable))

        if self.enable == False:
            return self.img
        return self.get_correct_exposure()
