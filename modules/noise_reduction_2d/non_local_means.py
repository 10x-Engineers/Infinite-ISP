"""
File: non_local_,means.py
Description: Denoise the image usi9ng non local means filter
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf
Implementation inspired from:
Fast Open ISP Author: Qiu Jueqin (qiujueqin@gmail.com)
& scikit-image (nl_means_denoising)
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
from tqdm import tqdm


class NLM:
    """
    Non local means filter for noise reduction
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform):
        self.img = img
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.platform = platform
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]

    def get_weights(self):
        """
        Applying weights
        """
        # wts is the strength parameter to assign weights to the similar
        # pixels in the image
        wts = self.parm_2dnr["wts"]

        # Avoiding division by zero
        if wts <= 0:
            wts = 1

        # The similarity between pixels is compared on the bases of the
        # Euclidean distance between them
        distance = np.arange(255**2)
        lut = np.exp(-distance / wts**2) * 1024

        return lut.astype(np.int32)

    def apply_nlm(self):
        """
        Applying Non-local Means Filter
        """
        # Input YUV image
        in_image = self.img

        # Search window and patch sizes
        window_size = self.parm_2dnr["window_size"]
        patch_size = self.parm_2dnr["patch_size"]

        # Patch size should be odd
        if patch_size % 2 == 0:
            patch_size = patch_size + 1
            print("    -Making patch size odd: ", patch_size)

        # Extracting Y channel to apply the 2DNR module
        input_image = in_image[:, :, 0]

        if in_image.dtype == "float32":
            input_image = np.round(255 * input_image).astype(np.uint8)

        # Declaring empty array for output image after denoising
        denoised_out = np.empty(in_image.shape, dtype=np.uint8)

        # Padding the input_image
        pads = window_size // 2
        wtspadded_y_in = np.pad(input_image, pads, mode="reflect")

        # Decalaration of denoised Y channel and weights
        denoised_y_channel = np.zeros(input_image.shape)
        final_weights = np.zeros(input_image.shape)

        # Generating LUT weighs based on euclidean distance between intensities
        # Assigning weights to similar pixels in descending order (most similar
        # will have the largest weight_for_each_shifted_array)
        weights_lut = self.get_weights()

        for i in tqdm(
            range(window_size), disable=self.is_progress, leave=self.is_leave
        ):
            for j in range(window_size):
                # Creating arrays starting from pixels according to the search window --
                # There will N = search_window*search_window stacked arrays of input
                # image size
                array_for_each_pixel_in_sw = np.int32(
                    wtspadded_y_in[
                        i : i + input_image.shape[0], j : j + input_image.shape[1], ...
                    ]
                )

                # Finding euclidean distance between pixels based on their intensities
                # & applying mean filter
                euc_distance = (input_image - array_for_each_pixel_in_sw) ** 2
                distance = self.apply_mean_filter(euc_distance, patch_size=patch_size)

                # Assigning weights to the pixels based on their distance (most similar
                # will have largest weight)
                weight_for_each_shifted_array = weights_lut[distance]

                # Adding up all the weighted similar pixels
                denoised_y_channel += (
                    array_for_each_pixel_in_sw * weight_for_each_shifted_array
                )

                # Adding up all the weights for final mean values at each pixel location
                final_weights += weight_for_each_shifted_array

        # Averaging out all the pixel
        denoised_y_channel = denoised_y_channel / final_weights

        if in_image.dtype == "float32":
            denoised_y_channel = np.float32((denoised_y_channel) / 255.0)
            denoised_out = denoised_out.astype("float32")

        # Reconstructing the final output
        denoised_out[:, :, 0] = denoised_y_channel
        denoised_out[:, :, 1] = in_image[:, :, 1]
        denoised_out[:, :, 2] = in_image[:, :, 2]

        return denoised_out

    def apply_mean_filter(self, array, patch_size):
        """
        Applying Mean Filter
        """
        padded_array = np.pad(array, patch_size // 2, mode="reflect")
        summed_up_arrays = np.zeros([array.shape[0], array.shape[1]])

        for i in range(patch_size):
            for j in range(patch_size):
                summed_up_arrays += np.int32(
                    padded_array[i : i + array.shape[0], j : j + array.shape[1], ...]
                )

        output_mean_filtered = ((summed_up_arrays) / patch_size**2).astype(
            array.dtype
        )
        return output_mean_filtered
