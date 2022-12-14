import numpy as np
from scipy.signal import correlate2d


class CFAInterpolation:
    'CFA Interpolation'

    def __init__(self, img, sensor_info, parm_dem):
        self.img = img
        #self.enable = parm_dem['isEnable']
        self.bayer = sensor_info['bayer_pattern']
        self.bitdepth = sensor_info['bitdep']

    def masks_CFA_Bayer(self):
        # ----generating masks for the given bayer pattern-----
        
        pattern = self.bayer
        # dict will be creating 3 channel boolean type array of given shape with the name
        # tag like 'r': [False False ....] , 'g': [False False ....] , 'b': [False False ....]
        channels = dict((channel, np.zeros(self.img.shape, dtype=bool))
                        for channel in 'rgb')
        
        # Following comment will create boolean masks for each channel r, g and b
        for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            channels[channel][y::2, x::2] = True

        # tuple will return 3 channel boolean pattern for r,g and b with True at corresponding value
        # For example in rggb pattern, the r mask would then be
        # [ [ True, False, True, False], [ False, False, False, False]]
        return tuple(channels[c] for c in 'rgb')

    def apply_cfa(self):
        # ----demosaicing the given raw image-----

        # 3D masks accoridng to the given bayer
        mask_r, mask_g, mask_b = self.masks_CFA_Bayer()  
        raw_in = np.float32(self.img)
        
        # Declaring 3D Demosaiced image
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))

        # 5x5 2D Filter coefficients for linear interpolation of r,g and b channels
        # These filters helps to retain corresponding pixels information using laplacian while interpolation

        # g at r & b location,
        g_at_r_and_b = np.float32(
            [[0, 0, -1, 0, 0],
             [0, 0, 2, 0, 0],
             [-1, 2, 4, 2, -1],
             [0, 0, 2, 0, 0],
             [0, 0, -1, 0, 0]]) * 0.125

        # r at green in r row & b column -- b at green in b row & r column
        r_at_Gr_and_b_at_Gb = np.float32(
            [[0, 0, 0.5, 0, 0],
             [0, -1, 0, -1, 0],
             [-1, 4, 5, 4, - 1],
             [0, -1, 0, -1, 0],
             [0, 0, 0.5, 0, 0]]) * 0.125

        # r at green in b row & r column -- b at green in r row & b column
        r_at_Gb_and_b_at_Gr = np.transpose(r_at_Gr_and_b_at_Gb)

        # r at blue in b row & b column -- b at red in r row & r column
        r_at_B_and_b_at_R = np.float32(
            [[0, 0, -1.5, 0, 0],
             [0, 2, 0, 2, 0],
             [-1.5, 0, 6, 0, -1.5],
             [0, 2, 0, 2, 0],
             [0, 0, -1.5, 0, 0]]) * 0.125  
        
        # Creating r, g & b channels from raw_in
        r = raw_in * mask_r
        g = raw_in * mask_g
        b = raw_in * mask_b

        # Creating g channel first after applying g_at_r_and_b filter
        g = np.where(np.logical_or(mask_r == 1, mask_b == 1), correlate2d(raw_in, g_at_r_and_b, mode='same', boundary='symm'),
                     g)

        # Applying other linear filters
        rb_at_g_rbbr = correlate2d(raw_in, r_at_Gr_and_b_at_Gb, mode='same', boundary='symm')
        rb_at_g_brrb = correlate2d(raw_in, r_at_Gb_and_b_at_Gr, mode='same', boundary='symm')
        rb_at_gr_bbrr = correlate2d(raw_in, r_at_B_and_b_at_R, mode='same', boundary='symm')

        # After convolving the input raw image with rest of the filters, now we have the respective
        # interpolated data, now we just have to extract the updated pixels values according to the
        # position they are meant to be updated

        # Extracting Red rows.
        r_rows = np.transpose(np.any(mask_r == 1, axis=1)[np.newaxis]) * np.ones(r.shape, dtype=np.float32)

        # Extracting Red columns.
        r_col = np.any(mask_r == 1, axis=0)[np.newaxis] * np.ones(r.shape, dtype=np.float32)

        # Extracting Blue rows.
        b_rows = np.transpose(np.any(mask_b == 1, axis=1)[np.newaxis]) * np.ones(b.shape, dtype=np.float32)

        # Extracting Blue columns
        b_col = np.any(mask_b == 1, axis=0)[np.newaxis] * np.ones(b.shape, dtype=np.float32)

        # For R channel we have to update pixels at [r rows and b cols] & at [b rows and r cols]
        # 3 pixels need to be updated near one given r
        r = np.where(np.logical_and(r_rows == 1, b_col == 1), rb_at_g_rbbr, r)
        r = np.where(np.logical_and(b_rows == 1, r_col == 1), rb_at_g_brrb, r)

        # Similarly for B channel we have to update pixels at [r rows and b cols] & at [b rows and r cols]
        # 3 pixels need to be updated near one given b
        b = np.where(np.logical_and(b_rows == 1, r_col == 1), rb_at_g_rbbr, b)
        b = np.where(np.logical_and(r_rows == 1, b_col == 1), rb_at_g_brrb, b)

        # Final r & b channels
        r = np.where(np.logical_and(b_rows == 1, b_col == 1), rb_at_gr_bbrr, r)
        b = np.where(np.logical_and(r_rows == 1, r_col == 1), rb_at_gr_bbrr, b)

        demos_out[:, :, 0] = r
        demos_out[:, :, 1] = g
        demos_out[:, :, 2] = b

        # Clipping the pixels values within the bit range
        demos_out = np.clip(demos_out, 0, 2 ** self.bitdepth - 1)
        demos_out = demos_out/(2**self.bitdepth)

        demos_out = np.uint8(demos_out*255)
        return demos_out

    def execute(self):
        print('CFA interpolation (default) = True')

        # if self.enable == False:
        #     return self.img
        # elif self.enable == True:
        return self.apply_cfa()