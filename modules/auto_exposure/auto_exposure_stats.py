"""
File: auto_exposure_stats.py
Description: 3A - AE statistics. Measures the raw frame (after digital gain) over a
             3x3 grid of cells. For each cell: how many green pixels are below a
             shadow threshold, how many are above a highlight threshold, the mean
             green value, and how many pixels of any channel are clipped to 0 or to
             full scale. Green is used as the luma proxy (it's the Bayer phase most
             correlated with brightness, and there are twice as many green pixels as
             red or blue). This module only measures - the auto_exposure module
             decides what to do with these numbers.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

GRID_ROWS = GRID_COLS = 3


def green_parity(bayer):
    """(row + col) % 2 of the green pixels: 1 for rggb / bggr, 0 for grbg / gbrg."""
    return 1 if bayer in ("rggb", "bggr") else 0


class AutoExposureStats:
    """
    AE statistics on a 3x3 grid of the raw frame (green-only luma, all-channel clips)
    """

    def __init__(self, img, sensor_info, parm_aes):
        self.img = img
        self.sensor_info = sensor_info
        self.enable = parm_aes["is_enable"]
        self.is_debug = parm_aes["is_debug"]
        self.bit_depth = sensor_info["bit_depth"]
        self.bayer = sensor_info["bayer_pattern"]

        # 4 column / 4 row boundaries (pixels), the last closing the grid
        self.col_starts = parm_aes["col_starts"]
        self.row_starts = parm_aes["row_starts"]
        self.shadow_threshold = parm_aes["shadow_threshold"]
        self.highlight_threshold = parm_aes["highlight_threshold"]

    def green_mask(self):
        """True where (row, col) is the green Bayer phase."""
        rows, cols = np.indices(self.img.shape[:2])
        return (rows + cols) % 2 == green_parity(self.bayer)

    def compute_grid_stats(self):
        """One dict of statistics per cell (cell index = row * 3 + col)."""
        img = self.img.astype(np.float64)
        green = self.green_mask()
        full_scale = (1 << self.bit_depth) - 1

        stats = []
        for r_idx in range(GRID_ROWS):
            r_0, r_1 = self.row_starts[r_idx], self.row_starts[r_idx + 1]
            for c_idx in range(GRID_COLS):
                c_0, c_1 = self.col_starts[c_idx], self.col_starts[c_idx + 1]
                roi = img[r_0:r_1, c_0:c_1]
                greens = roi[green[r_0:r_1, c_0:c_1]]
                # clip-corrected: a fully clipped green no longer carries real
                # brightness information, so it's excluded from the mean
                valid_greens = greens[(greens > 0) & (greens < full_scale)]
                stats.append(
                    {
                        "green_count": int(greens.size),
                        "pixel_count": int(roi.size),
                        "valid_green_mean": (
                            float(np.mean(valid_greens)) if valid_greens.size else 0.0
                        ),
                        "valid_green_count": int(valid_greens.size),
                        "shadow_count": int(np.sum(greens < self.shadow_threshold)),
                        "highlight_count": int(
                            np.sum(greens > self.highlight_threshold)
                        ),
                        "black_clip_count": int(np.sum(roi == 0)),
                        "white_clip_count": int(np.sum(roi == full_scale)),
                    }
                )
        return stats

    def execute(self):
        """
        Execute AE statistics: returns a list of 9 per-cell dicts, or None if disabled
        """
        print("Auto Exposure Statistics = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        stats = self.compute_grid_stats()
        if self.is_debug:
            for i, cell in enumerate(stats):
                print(f"   - AES - cell {i}: {cell}")
        print(f"  Execution time: {time.time() - start:.3f}s")
        return stats
