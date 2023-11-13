"""
Helper functions for automatic config file parameter update
"""

import re
import rawpy
import numpy as np


def parse_file_name(filename):
    """
    Parse the file name
    """
    # Expected pattern for file name
    pattern = r"(.+)_(\d+)x(\d+)_(\d+)(?:bit|bits)_(RGGB|GRBG|GBRG|BGGR)"
    # Check pattern in the string
    match_parttern = re.match(pattern, filename)
    if match_parttern:
        _, width, height, bit_depth, bayer = match_parttern.groups()
        # Convert width, height, and bits to integers and bayer tp lower case
        return [int(width), int(height), int(bit_depth), bayer.lower()]
    return False


def extract_raw_metadata(filename):
    """
    Extract Exif/Metadata Information from Raw File
    """
    with rawpy.imread(filename) as raw:

        # Get the Bayer pattern
        # The pattern is returned as a 2D numpy array, where 0=red, 1=green, 2=blue
        bayer_array = raw.raw_pattern
        # Map the numerical values to color letters
        color_map = {0: "r", 1: "g", 2: "b", 3: "g"}
        bayer_pattern = "".join((np.vectorize(color_map.get)(bayer_array)).flatten())

        # Get the bit depth
        # The white_level attribute gives the maximum possible value
        # for a pixel, which can be used to infer the bit depth
        white_level = raw.white_level
        bit_depth = white_level.bit_length()

        # Get the dimensions
        # These are the dimensions of the raw image data,
        # which includes any extra pixels around the edges
        # used by some cameras
        height, width = raw.raw_image.shape

        black_level = raw.black_level_per_channel
        wb_gains = raw.camera_whitebalance
        ccm = raw.color_matrix

        return [
            int(width),
            int(height),
            int(bit_depth),
            bayer_pattern,
            black_level,
            white_level,
            wb_gains,
            ccm,
        ]
