import numpy as np
from tqdm import tqdm

class DeadPixelCorrection:
    'Dead Pixel Correction'

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc['isEnable']
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform['disable_progress_bar']
        self.is_leave = platform['leave_pbar_string']

    def padding(self):
        """Return a mirror padded copy of image."""

        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad

    def apply_dead_pixel_correction(self):
        """This function detects and corrects Dead pixels."""

        height, width = self.sensor_info["height"], self.sensor_info["width"]
        self.bpp = self.sensor_info["bitdep"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["isDebug"]

        # Mirror padding is applied to self.img.
        img_padded = np.float32(self.padding())
        dpc_img = np.empty((height, width), np.float32)
        corrected_pv_count = 0

        # Loop over the padded image to ensure that each pixel is tested.
        for y in tqdm(range(img_padded.shape[0] - 4), disable=self.is_progress, leave=self.is_leave):
            for x in range(img_padded.shape[1] - 4):
                top_left = img_padded[y, x]
                top_mid = img_padded[y, x + 2]
                top_right = img_padded[y, x + 4]

                left_of_center_pixel = img_padded[y + 2, x]
                center_pixel = img_padded[y + 2, x + 2]    # pixel under test
                right_of_center_pixel = img_padded[y + 2, x + 4]

                bottom_right = img_padded[y + 4, x]
                bottom_mid = img_padded[y + 4, x + 2]
                bottom_left = img_padded[y + 4, x + 4]

                neighbors = np.array([top_left, top_mid, top_right, left_of_center_pixel, right_of_center_pixel,
                                      bottom_right, bottom_mid, bottom_left])

                # center_pixel is good if pixel value is between min and max of a 3x3 neighborhhood.
                if not(min(neighbors) < center_pixel < max(neighbors)):

                    # ""center_pixel is corrected only if the difference of center_pixel and every
                    # neighboring pixel is greater than the speciified threshold.
                    # The two if conditions are used in combination to reduce False positives.""

                    diff_with_center_pixel = abs(neighbors-center_pixel)
                    thresh = np.full_like(
                        diff_with_center_pixel, self.threshold)

                    # element-wise comparison of numpy arrays
                    if np.all(diff_with_center_pixel > thresh):
                        corrected_pv_count+=1 

                        # Compute gradients
                        vertical_grad = abs(
                            2 * center_pixel - top_mid - bottom_mid)
                        horizontal_grad = abs(
                            2 * center_pixel - left_of_center_pixel - right_of_center_pixel)
                        left_diagonal_grad = abs(
                            2 * center_pixel - top_left - bottom_left)
                        right_diagonal_grad = abs(
                            2 * center_pixel - top_right - bottom_right)

                        min_grad = min(vertical_grad, horizontal_grad,
                                       left_diagonal_grad, right_diagonal_grad)

                        # Correct value is computed using neighbors in the direction of minimum gradient.
                        if (min_grad == vertical_grad):
                            center_pixel = (top_mid + bottom_mid) / 2
                        elif (min_grad == horizontal_grad):
                            center_pixel = (
                                left_of_center_pixel + right_of_center_pixel) / 2
                        elif (min_grad == left_diagonal_grad):
                            center_pixel = (top_left + bottom_left) / 2
                        else:
                            center_pixel = (top_right + bottom_right) / 2

                # Corrected pixels are placed in non-padded image.
                dpc_img[y, x] = center_pixel
        self.img = np.uint16(np.clip(dpc_img, 0, (2**self.bpp)-1))
        if self.is_debug:
            print('   - Number of corrected pixels = ', corrected_pv_count)
            print('   - Threshold = ', self.threshold)
        return self.img

    def execute(self):
        print('Dead Pixel Correction = ' + str(self.enable))

        if self.enable == False:
            return self.img
        else:
            return self.apply_dead_pixel_correction()
