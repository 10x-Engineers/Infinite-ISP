"""
File: scale.py
Description: Implements both hardware friendly and non hardware freindly scaling
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import re
import numpy as np
from util.utils import crop

from util.utils import save_output_array_yuv, save_output_array
from modules.scale.nearest_neighbor import NearestNeighbor as NN
from modules.scale.bilinear_interpolation import BilinearInterpolation as BLI


class Scale:
    """Scale color image to given size."""

    def __init__(self, img, platform, sensor_info, parm_sca, conv_std):
        self.img = img
        self.enable = parm_sca["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
        self.is_save = parm_sca["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        self.get_scaling_params()

    def apply_scaling(self):
        """Execute scaling."""

        # check if no change in size
        if self.old_size == self.new_size:
            if self.is_debug:
                print("   - Output size is the same as input size.")
            return self.img

        if self.img.dtype == "float32":
            scaled_img = np.empty(
                (self.new_size[0], self.new_size[1], 3), dtype="float32"
            )
        else:
            scaled_img = np.empty(
                (self.new_size[0], self.new_size[1], 3), dtype="uint8"
            )

        # Loop over each channel to resize the image
        for i in range(3):

            ch_arr = self.img[:, :, i]
            scale_2d = Scale2D(ch_arr, self.sensor_info, self.parm_sca)
            scaled_ch = scale_2d.execute()

            # If input size is invalid, the Scale2D class returns the original image.
            if scaled_ch.shape == self.old_size:
                return self.img
            else:
                scaled_img[:, :, i] = scaled_ch

            # Because each channel is scaled in the same way, the isDeug flag is turned
            # off after the first channel has been scaled.
            self.parm_sca["is_debug"] = False

        return scaled_img

    def get_scaling_params(self):
        """Save parameters as instance attributes."""
        self.is_debug = self.parm_sca["is_debug"]
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])

    def save(self):
        """
        Function to save module output
        """
        # update size of array in filename
        self.platform["in_file"] = re.sub(
            r"\d+x\d+",
            f"{self.img.shape[1]}x{self.img.shape[0]}",
            self.platform["in_file"],
        )
        if self.is_save:
            if self.platform["rgb_output"]:
                save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_",
                    self.platform,
                    self.sensor_info["bit_depth"],
                    self.sensor_info["bayer_pattern"],
                )
            else:
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_",
                    self.platform,
                    self.conv_std,
                )

    def execute(self):
        """Execute scaling if enabled."""
        print("Scale = " + str(self.enable))

        if self.enable:
            start = time.time()
            scaled_img = self.apply_scaling()
            print(f"  Execution time: {time.time() - start:.3f}s")
            return scaled_img
        return self.img


################################################################################
class Scale2D:
    """Scale 2D image to given size."""

    def __init__(self, single_channel, sensor_info, parm_sca):
        self.single_channel = np.float32(single_channel)
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
        self.get_scaling_params()

    def resize_by_non_int_fact(self, red_fact, method):

        """ "
        Resize 2D array by non-integer factor n/d.
        Firstly, the array is upsacled n times then downscaled d times.
        Parameter:
        ---------
        red_fact: list of tuples with scale factors for height (at index 0) and width (at index 1).
        method: list with algorithms used for upscaling(at index 0) and downscaling(at index 1).
        Output: scaled img
        """

        for i in range(2):
            if bool(red_fact[i]):

                # reduction factor = n/d means:
                # Upscale the cropped image n times then downscale d times
                upscale_fact = red_fact[i][0]
                downscale_fact = red_fact[i][1]

                upscale_to_size = (
                    (
                        upscale_fact * self.single_channel.shape[0],
                        self.single_channel.shape[1],
                    )
                    if i == 0
                    else (
                        self.single_channel.shape[0],
                        upscale_fact * self.single_channel.shape[1],
                    )
                )
                if method[0] == "Nearest_Neighbor":
                    nn_obj = NN(self.single_channel, upscale_to_size)
                    self.single_channel = nn_obj.scale_nearest_neighbor()
                elif method[0] == "Bilinear":
                    bilinear_obj = BLI(self.single_channel, upscale_to_size)
                    self.single_channel = bilinear_obj.bilinear_interpolation()
                else:
                    if self.is_debug and i == 0:
                        print(
                            "   - Invalid scale method.\n"
                            + "   - UpScaling with default Nearest Neighbour method..."
                        )
                    nn_obj = NN(self.single_channel, upscale_to_size)
                    self.single_channel = nn_obj.scale_nearest_neighbor()
                downscale_to_size = (
                    (
                        int(np.round(self.single_channel.shape[0] / downscale_fact)),
                        self.single_channel.shape[1],
                    )
                    if i == 0
                    else (
                        self.single_channel.shape[0],
                        int(np.round(self.single_channel.shape[1] // downscale_fact)),
                    )
                )

                if method[1] == "Nearest_Neighbor":
                    nn_obj = NN(self.single_channel, downscale_to_size)
                    self.single_channel = nn_obj.downscale_nearest_neighbor()
                elif method[1] == "Bilinear":
                    bilinear_obj = BLI(self.single_channel, downscale_to_size)
                    self.single_channel = bilinear_obj.downscale_by_int_factor()
                else:
                    if self.is_debug and i == 0:
                        print(
                            "   - Invalid scale method.\n"
                            + "   - DownScaling with default Nearest Neighbour method..."
                        )
                    nn_obj = NN(self.single_channel, downscale_to_size)
                    self.single_channel = nn_obj.downscale_nearest_neighbor()

        return self.single_channel

    def hardware_dep_scaling(self):
        """Set algorithm workflow."""
        # check and set the flow of the algorithm
        scale_info = self.validate_input_output()

        # apply scaling according to the flow
        return self.apply_algo(
            scale_info, ([self.upscale_method, self.downscale_method])
        )

    def apply_algo(self, scale_info, method):

        """
        Scale 2D array using hardware friendly approach comprising of 3 steps:
           1. Downscale with int factor
           2. Crop
           3. Scale with non-integer-factor"""

        # check if input size is valid
        if scale_info == [[None, None, None], [None, None, None]]:
            print(
                "   - Invalid input size. It must be one of the following:\n"
                "   - 1920x1080\n"
                "   - 2592x1536\n"
                "   - 2592x1944"
            )
            return self.single_channel

        # check if output size is valid
        if scale_info == [[1, 0, None], [1, 0, None]]:
            print("   - Invalid output size.")
            return self.single_channel
        else:
            # step 1: Downscale by int factor using bilinear interpolation
            if scale_info[0][0] > 1 or scale_info[1][0] > 1:
                bilinear_obj = BLI(self.single_channel, self.new_size)
                self.single_channel = bilinear_obj.downscale_by_int_factor(
                    (
                        self.old_size[0] // scale_info[0][0],
                        self.old_size[1] // scale_info[1][0],
                    )
                )

                if self.is_debug:
                    print(
                        "   - Shape after downscaling by integer factor "
                        + f"({scale_info[0][0]}, {scale_info[1][0]})",
                        self.single_channel.shape,
                    )

            # step 2: crop
            if scale_info[0][1] > 0 or scale_info[1][1] > 0:
                self.single_channel = crop(
                    self.single_channel, scale_info[0][1], scale_info[1][1]
                )

                if self.is_debug:
                    print(
                        "   - Shape after cropping "
                        + f"({scale_info[0][1]}, {scale_info[1][1]}): ",
                        self.single_channel.shape,
                    )
            # step 3: Scale with non-int factor
            if bool(scale_info[0][2]) or bool(scale_info[1][2]):
                self.single_channel = self.resize_by_non_int_fact(
                    (scale_info[0][2], scale_info[1][2]), method
                )

                if self.is_debug:
                    print(
                        "   - Shape after scaling by non-integer factor "
                        + f"({scale_info[0][2]}, {scale_info[1][2]}): ",
                        self.single_channel.shape,
                    )
            return self.single_channel

    def validate_input_output(self):
        """Chcek if the size of the input image is valid according to the set workflow."""
        valid_size = [(1080, 1920), (1536, 2592), (1944, 2592)]
        sizes = [OUT1080x1920, OUT1536x2592, OUT1944x2592]

        # Check if input size is valid
        if self.old_size not in valid_size:
            scale_info = [[None, None, None], [None, None, None]]
            return scale_info

        idx = valid_size.index(self.old_size)
        size_obj = sizes[idx](self.new_size)
        scale_info = size_obj.execute()
        return scale_info

    def execute(self):
        """Execute scaling."""

        if self.is_hardware:
            self.single_channel = self.hardware_dep_scaling()
        else:

            self.single_channel = self.hardware_indp_scaling()

        if self.is_debug:
            print(
                "   - Shape of scaled image for a single channel = ",
                self.single_channel.shape,
            )
        return self.single_channel

    def hardware_indp_scaling(self):
        """Execute hardware independent scaling."""
        if self.algo == "Nearest_Neighbor":
            if self.is_debug:
                print("   - Scaling with Nearest Neighbor method...")

            nn_obj = NN(self.single_channel, self.new_size)
            return nn_obj.scale_nearest_neighbor()

        elif self.algo == "Bilinear":
            if self.is_debug:
                print("   - Scaling with Bilinear method...")

            bilinear_obj = BLI(self.single_channel, self.new_size)
            return bilinear_obj.bilinear_interpolation()

        else:
            if self.is_debug:
                print(
                    "   - Invalid scale method."
                    + "   - Scaling with default Nearest Neighbor method..."
                )

            nn_obj = NN(self.single_channel, self.new_size)
            return nn_obj.scale_nearest_neighbor()

    def get_scaling_params(self):
        """Save parameters as instance attributes."""
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])
        self.is_debug = self.parm_sca["is_debug"]
        self.is_hardware = self.parm_sca["is_hardware"]
        self.algo = self.parm_sca["algorithm"]
        self.upscale_method = self.parm_sca["upscale_method"]
        self.downscale_method = self.parm_sca["downscale_method"]


#############################################################################
# The structure and working of the following three classes is exactlly the same
class OUT1080x1920:
    """
    The hardware friendly approach can only be used for specific input and output sizes.
    This class checks if the given output size can be achieved by this approach and
    creates a nested list (SCALE_LIST below) with corresponding constants used to execute
    this scaling approach comprising of the following three steps:
    1. Downscale with int factor
    2. Crop
    3. Scale with non-integer-factor
    The elements in each list correspond to the following constants:
    1. Scale factor [int]: (default 1) scale factor for downscaling.
    2. Crop value [int] : (defaut 0) number of rows or columns to be croped.
    3. Non-int scale factor [tuple with 2 enteries]: (default None) a rational scale factor
       of form n/d where n is the first entry (index 0) and d is the second entry(index 1)
       of this tuple.
    Instance Attributes:
    -------------------
    SCALE_LIST:  a nested list with two sublists containing constants used in order to
    scale height (index 0) and width (index 1) to the given NEW_SIZE using the three
    steps above.
    """

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (720, 1280):
            self.scale_list = [[1, 0, (2, 3)], [1, 0, (2, 3)]]

        elif self.new_size == (480, 640):
            self.scale_list = [[2, 60, None], [3, 0, None]]

        elif self.new_size == (360, 640):
            self.scale_list = [[3, 0, None], [3, 0, None]]

        else:
            self.scale_list = [[1, 0, None], [1, 0, None]]

    def execute(self):
        """Get crop/scale factors to the corresponding input size"""
        return self.scale_list


#############################################################################
class OUT1536x2592:
    """Same as class OUT_1080x1920"""

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (1080, 1920):
            self.scale_list = [[1, 24, (5, 7)], [1, 32, (3, 4)]]

        elif self.new_size == (720, 1280):
            self.scale_list = [[2, 48, None], [2, 16, None]]

        elif self.new_size == (480, 640):
            self.scale_list = [[3, 32, None], [4, 8, None]]

        elif self.new_size == (360, 640):
            self.scale_list = [[4, 24, None], [4, 8, None]]

        else:
            self.scale_list = [[1, 0, None], [1, 0, None]]

    def execute(self):
        """Get crop/scale factors to the corresponding input size"""
        return self.scale_list


#############################################################################
class OUT1944x2592:
    """Same as OUT_1080x1920"""

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (1440, 2560):
            self.scale_list = [[1, 24, (3, 4)], [1, 32, None]]

        elif self.new_size == (1080, 1920):
            self.scale_list = [[1, 54, (4, 7)], [1, 32, (3, 4)]]

        elif self.new_size == (960, 1280):
            self.scale_list = [[2, 16, None], [2, 12, None]]

        elif self.new_size == (720, 1280):
            self.scale_list = [[2, 16, (3, 4)], [2, 12, None]]

        elif self.new_size == (480, 640):
            self.scale_list = [[4, 6, None], [4, 8, None]]

        elif self.new_size == (360, 640):
            self.scale_list = [[4, 6, (3, 4)], [4, 8, None]]

        else:
            self.scale_list = [[1, 0, None], [1, 0, None]]

    def execute(self):
        """Get crop/scale factors to the corresponding input size"""
        return self.scale_list
