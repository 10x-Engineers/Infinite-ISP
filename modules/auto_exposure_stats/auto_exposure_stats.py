"""
File: auto_exposure_stats.py
Description: AE statistics block - five statistics per cell
             of a configurable 3x3 grid, computed on the linear Bayer raw.

             Per grid cell:
               - shadow_count      : # green pixels with value <  shadow_threshold
               - highlight_count   : # green pixels with value >  highlight_threshold
               - grid_mean_sum     : sum of green pixel values (the AE control divides)
               - black_clip_count  : # pixels (any CFA channel) with value == 0
               - white_clip_count  : # pixels (any CFA channel) with value == 2**bit_depth - 1
             plus the green / total pixel counts of the cell (the divisors).

             Luma statistics use GREEN pixels only: raw R, G and B have different
             pre-white-balance levels, and green is the standard luma proxy.  Clip
             counts use every channel.  Thresholds use strict inequalities, so a
             pixel between them is in neither count.

             Where it runs: on the digital-gain output - linear, black-level
             corrected, before LSC and noise reduction.  In the reference firmware
             the block taps the BLC output because the AE moves the sensor
             exposure; in Infinite-ISP digital gain stands in for the sensor
             exposure, so the statistics must see it.

Code / Paper  Reference: Infinite-ISP_ReferenceModel modules/auto_exposure_stats.py
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_ae_statistics

GRID_ROWS = GRID_COLS = 3


class AutoExposureStats:
    """
    AE statistics on a 3x3 grid of the Bayer raw (green-only luma, all-channel clips)
    """

    def __init__(self, img, platform, sensor_info, parm_aes):
        self.img = img
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_aes["is_enable"]
        self.is_save = parm_aes["is_save"]
        self.is_debug = parm_aes["is_debug"]
        self.bayer = sensor_info["bayer_pattern"]
        self.max_val = (1 << sensor_info["bit_depth"]) - 1

        # grid boundaries: 4 column / 4 row starts, the last being the frame end
        self.col_starts = [int(c) for c in parm_aes["col_starts"]]
        self.row_starts = [int(r) for r in parm_aes["row_starts"]]
        self.shadow_threshold = int(parm_aes["shadow_threshold"])
        self.highlight_threshold = int(parm_aes["highlight_threshold"])
        self.check_grid()

    def check_grid(self):
        """Refuse a grid that does not fit the frame, naming the fix."""
        height, width = self.img.shape[:2]
        for name, starts, size in (("col_starts", self.col_starts, width),
                                   ("row_starts", self.row_starts, height)):
            if len(starts) != GRID_COLS + 1 or starts != sorted(starts) or \
                    starts[0] < 0 or starts[-1] > size:
                raise ValueError(
                    f"auto_exposure_stats.{name} = {starts} must be 4 ascending "
                    f"boundaries within 0..{size} (the frame after crop), e.g. "
                    f"[0, {size // 3}, {2 * size // 3}, {size}].")

    def green_mask(self, height, width):
        """True at green pixels: (row + col) odd for RGGB/BGGR, even for GRBG/GBRG."""
        rows, cols = np.indices((height, width))
        parity = (rows + cols) % 2
        return parity == (1 if self.bayer in ("rggb", "bggr") else 0)

    def compute_grid_stats(self):
        """The five statistics for each of the 9 cells, cell index = row * 3 + col."""
        img = self.img
        green = self.green_mask(*img.shape[:2])
        keys = ("shadow_count", "highlight_count", "grid_mean_sum", "black_clip_count",
                "white_clip_count", "green_pixel_count", "pixel_count")
        stats = {k: [0] * (GRID_ROWS * GRID_COLS) for k in keys}
        for r_idx in range(GRID_ROWS):
            r_0, r_1 = self.row_starts[r_idx], self.row_starts[r_idx + 1]
            for c_idx in range(GRID_COLS):
                c_0, c_1 = self.col_starts[c_idx], self.col_starts[c_idx + 1]
                cell = r_idx * GRID_COLS + c_idx
                roi = img[r_0:r_1, c_0:c_1]
                greens = roi[green[r_0:r_1, c_0:c_1]]
                stats["shadow_count"][cell] = int(np.sum(greens < self.shadow_threshold))
                stats["highlight_count"][cell] = int(np.sum(greens > self.highlight_threshold))
                stats["grid_mean_sum"][cell] = int(np.sum(greens, dtype=np.int64))
                stats["green_pixel_count"][cell] = int(greens.size)
                stats["black_clip_count"][cell] = int(np.sum(roi == 0))
                stats["white_clip_count"][cell] = int(np.sum(roi == self.max_val))
                stats["pixel_count"][cell] = int(roi.size)
        return stats

    def execute(self):
        """Execute AE statistics; returns the statistics dict, or None when disabled."""
        print("Auto Exposure Statistics = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        stats = self.compute_grid_stats()
        if self.is_debug:
            greens = np.maximum(stats["green_pixel_count"], 1)
            print(f"   - AES - shadow / highlight threshold = "
                  f"{self.shadow_threshold} / {self.highlight_threshold}")
            print("   - AES - green mean per cell        =",
                  [int(s // g) for s, g in zip(stats["grid_mean_sum"], greens)])
            print("   - AES - shadow % per cell          =",
                  [round(100 * s / g, 2) for s, g in zip(stats["shadow_count"], greens)])
            print("   - AES - highlight % per cell       =",
                  [round(100 * h / g, 2) for h, g in zip(stats["highlight_count"], greens)])
            print("   - AES - black / white clip counts  =",
                  stats["black_clip_count"], "/", stats["white_clip_count"])
        if self.is_save:
            save_ae_statistics(self.platform["in_file"], stats, "Out_auto_exposure_stats_")
        print(f"  Execution time: {time.time() - start:.3f}s")
        return stats
