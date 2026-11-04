"""
File: test_lens_shading_correction.py
Description: Unit tests for the lens shading correction module.
Run from the repository root with: python -m pytest tests
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.lens_shading_correction.lens_shading_correction import (
    LensShadingCorrection as LSC,
)
from util.utils import get_bayer_channel_offsets, lsc_gain_grids_from_flat_field

BIT_DEPTH = 12
HEIGHT, WIDTH = 480, 640
GRID_ROWS, GRID_COLS = 13, 17
BAYER_PATTERNS = ["rggb", "bggr", "grbg", "gbrg"]

# Falloff strength of each channel (corners at about 40 to 55 percent of the
# center) and an off-center optical center: red shades the most, as is typical
# of the IR-cut filter at oblique angles
FALLOFF = {"r": 0.45, "gr": 0.35, "gb": 0.36, "b": 0.30}
OPTICAL_CENTER = (0.46, 0.53)


def make_shading(bayer):
    """Per-pixel shading (1.0 at the optical center) for each bayer channel"""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    y_norm = rows / (HEIGHT - 1) - OPTICAL_CENTER[0]
    x_norm = cols / (WIDTH - 1) - OPTICAL_CENTER[1]
    # Slightly elliptical, so the shading is not radially symmetric
    radius_sq = (x_norm**2 + 1.3 * y_norm**2) / 0.5
    shading = np.ones((HEIGHT, WIDTH))
    for channel, (row, col) in get_bayer_channel_offsets(bayer).items():
        # cos^4-like falloff
        falloff = 1 / (1 + FALLOFF[channel] * radius_sq) ** 2
        shading[row::2, col::2] = falloff[row::2, col::2]
    return shading


def run_lsc(img, bayer, parm_lsc):
    """Run the module on a raw image"""
    sensor_info = {"bayer_pattern": bayer, "bit_depth": BIT_DEPTH}
    platform = {"in_file": "test", "save_format": "npy"}
    return LSC(img, platform, sensor_info, parm_lsc).execute()


def channel_non_uniformity(img, bayer):
    """(max - min) / mean of each bayer channel"""
    result = {}
    for channel, (row, col) in get_bayer_channel_offsets(bayer).items():
        plane = np.float64(img[row::2, col::2])
        result[channel] = (plane.max() - plane.min()) / plane.mean()
    return result


def test_calibrated_grids_flatten_vignetted_flat_field():
    """Gains calibrated from one noisy flat field flatten another one"""
    rng = np.random.default_rng(0)
    for bayer in BAYER_PATTERNS:
        shading = make_shading(bayer)
        calib_flat = 2500 * shading + rng.normal(0, 8, shading.shape)
        parm_lsc = lsc_gain_grids_from_flat_field(
            np.uint16(np.clip(calib_flat, 0, 4095)), bayer, GRID_ROWS, GRID_COLS
        )
        parm_lsc.update({"is_enable": True, "is_save": False})

        # A noise-free flat field at a different level
        test_flat = np.uint16(2000 * shading)
        before = channel_non_uniformity(test_flat, bayer)
        after = channel_non_uniformity(run_lsc(test_flat, bayer, parm_lsc), bayer)

        for channel in FALLOFF:
            assert before[channel] > 0.4, (bayer, channel, before[channel])
            assert after[channel] < 0.015, (bayer, channel, after[channel])


def test_gains_follow_bayer_pattern():
    """Each CFA position gets its own channel's gain"""
    gains = {"r": 1.5, "gr": 1.25, "gb": 1.125, "b": 2.0}
    img = np.full((8, 8), 1000, dtype=np.uint16)
    for bayer in BAYER_PATTERNS:
        parm_lsc = {"is_enable": True, "is_save": False}
        for channel, gain in gains.items():
            parm_lsc[channel + "_gain_grid"] = np.full((3, 3), gain).tolist()

        out = run_lsc(img, bayer, parm_lsc)
        for channel, (row, col) in get_bayer_channel_offsets(bayer).items():
            assert np.all(out[row::2, col::2] == 1000 * gains[channel]), (
                bayer,
                channel,
            )


def test_unity_grid_and_disabled_are_bit_exact():
    """Unity gains or a disabled module leave the image unchanged"""
    rng = np.random.default_rng(1)
    img = np.uint16(rng.integers(0, 2**BIT_DEPTH, (HEIGHT, WIDTH)))
    unity = np.ones((GRID_ROWS, GRID_COLS)).tolist()
    parm_lsc = {"is_enable": True, "is_save": False}
    for channel in FALLOFF:
        parm_lsc[channel + "_gain_grid"] = unity

    assert np.array_equal(run_lsc(img, "rggb", parm_lsc), img)

    parm_lsc["is_enable"] = False
    parm_lsc["r_gain_grid"] = (2 * np.ones((GRID_ROWS, GRID_COLS))).tolist()
    assert np.array_equal(run_lsc(img, "rggb", parm_lsc), img)


def test_output_clipped_to_bit_depth():
    """Gained pixels saturate at the sensor white level"""
    img = np.full((4, 4), 4000, dtype=np.uint16)
    parm_lsc = {"is_enable": True, "is_save": False}
    for channel in FALLOFF:
        parm_lsc[channel + "_gain_grid"] = [[2.0, 2.0], [2.0, 2.0]]

    out = run_lsc(img, "rggb", parm_lsc)
    assert out.dtype == np.uint16
    assert np.all(out == 2**BIT_DEPTH - 1)


if __name__ == "__main__":
    test_calibrated_grids_flatten_vignetted_flat_field()
    test_gains_follow_bayer_pattern()
    test_unity_grid_and_disabled_are_bit_exact()
    test_output_clipped_to_bit_depth()
    print("All lens shading correction tests passed")
