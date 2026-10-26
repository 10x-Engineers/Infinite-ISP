"""
File: saturation_enhancement.py
Description: Saturation enhancement in YUV.  Runs after 2D noise
             reduction, before RGB conversion.

Two algorithms, selected by the config key `algorithm`:

  - "global"      : Cb and Cr scaled by one gain, quantised to the RTL's SAT_GAIN
                    register (12-bit unsigned Q4.8, truncated) - the same result as
                    the fixed-point reference model and RTL.  Simple
                    and cheap, but it amplifies CCM errors uniformly and ignores
                    luminance and hue.

  - "hue_rolloff" : per-pixel gain =
                       saturation_gain                    (global base)
                     x hue_sat_lut[hue_bin(Cb, Cr)]       (hue-selective)
                     x luminance factor(Y)                (shadow + highlight roll-off)
                     x chroma-magnitude factor(|C|)       (neutral-pixel protection)
                    Skin and greens stay restrained, sky gets punchier, highlights
                    roll off to white, shadow noise is not coloured.

Chroma is centred at 128, the offset Infinite-ISP's colour space conversion adds and
its RGB conversion removes (the reference firmware's pipeline centres it at 127).
Y is never changed.

Code / Paper  Reference: Infinite-ISP_ReferenceModel modules/saturation_enhancement.py
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array_yuv

CHROMA_CENTRE = 128          # 8-bit YUV chroma offset of color_space_conversion


def q4_8(value):
    """Unsigned Q4.8 (the 12-bit SAT_GAIN register, 8 fractional bits): truncate,
    saturate."""
    return min(np.floor(value * 256.0), 4095.0) / 256.0


def default_hue_sat_lut(n_bins=64):
    """Camera-like hue-saturation LUT (multipliers around 1.0), YCbCr hue angles
    for the BT.601 conversion: magenta ~49 deg, red ~109, yellow ~170, green ~232,
    cyan ~289, blue ~341.
        skin / red ... 1.00   (never push skin)
        yellow ....... 0.95   (over-saturated yellow looks jaundiced)
        green ........ 0.90   (over-saturated green looks radioactive)
        cyan ......... 1.10
        blue / sky ... 1.15
        magenta ...... 1.05
        elsewhere .... 1.00
    """
    lut = np.ones(n_bins, dtype=np.float32)
    deg_centres = (np.arange(n_bins) + 0.5) * (360.0 / n_bins)
    for i, deg in enumerate(deg_centres):
        if 30.0 <= deg <= 70.0:
            lut[i] = 1.05
        elif 90.0 <= deg <= 130.0:
            lut[i] = 1.00
        elif 150.0 <= deg <= 190.0:
            lut[i] = 0.95
        elif 210.0 <= deg <= 260.0:
            lut[i] = 0.90
        elif 270.0 <= deg <= 300.0:
            lut[i] = 1.10
        elif 300.0 <= deg <= 345.0:
            lut[i] = 1.15
    return lut


class SaturationEnhancement:
    """
    Saturation Enhancement
    """

    def __init__(self, img, platform, sensor_info, parm_se, conv_std):
        self.img = img.copy()
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_se["is_enable"]
        self.is_save = parm_se["is_save"]
        self.conv_std = conv_std
        self.algorithm = parm_se["algorithm"]
        self.gain = float(parm_se["saturation_gain"])
        if self.algorithm not in ("global", "hue_rolloff"):
            raise ValueError(f"saturation_enhancement.algorithm = {self.algorithm!r}: "
                             f"use 'global' or 'hue_rolloff'.")
        if self.algorithm == "hue_rolloff":
            self.highlight_rolloff = float(parm_se["highlight_rolloff"])
            self.shadow_fade = float(parm_se["shadow_fade"])
            self.mag_fade = float(parm_se["mag_fade"])
            user_lut = parm_se.get("hue_sat_lut")
            self.hue_sat_lut = (np.asarray(user_lut, dtype=np.float32)
                                if user_lut is not None else default_hue_sat_lut(64))

    def apply_global(self):
        """Uniform Cb, Cr scaling by one Q4.8 gain."""
        gain = q4_8(self.gain)
        u_ch = np.float64(self.img[:, :, 1]) - CHROMA_CENTRE
        v_ch = np.float64(self.img[:, :, 2]) - CHROMA_CENTRE
        u_ch = np.clip(u_ch * gain + CHROMA_CENTRE, 0, 255).astype(np.uint8)
        v_ch = np.clip(v_ch * gain + CHROMA_CENTRE, 0, 255).astype(np.uint8)
        return np.stack((self.img[:, :, 0], u_ch, v_ch), axis=-1)

    def apply_hue_rolloff(self):
        """Per-pixel Cb/Cr gain:
            base    = saturation_gain * (1 + (hue_gain - 1) * mag_factor)
            gain_px = 1 + (base - 1) * lum_factor
        lum_factor fades the gain to 1.0 in deep shadows and near white;
        mag_factor fades the hue LUT to neutral where chroma is tiny (noise)."""
        y_ch = self.img[:, :, 0].astype(np.float32)
        c_b = self.img[:, :, 1].astype(np.float32) - CHROMA_CENTRE
        c_r = self.img[:, :, 2].astype(np.float32) - CHROMA_CENTRE

        y_norm = y_ch / 255.0
        high = np.clip((1.0 - y_norm) / max(1e-6, 1.0 - self.highlight_rolloff), 0.0, 1.0)
        t_lo = np.clip(y_norm / max(1e-6, self.shadow_fade), 0.0, 1.0)
        low = t_lo * t_lo * (3.0 - 2.0 * t_lo)
        lum_fac = high * low

        mag_fac = np.clip(np.sqrt(c_b * c_b + c_r * c_r) / max(1e-6, self.mag_fade), 0.0, 1.0)

        n_bins = len(self.hue_sat_lut)
        hue_deg = (np.degrees(np.arctan2(c_r, c_b)) + 360.0) % 360.0
        hue_bin = np.clip((hue_deg * n_bins / 360.0).astype(np.int32), 0, n_bins - 1)
        base = self.gain * (1.0 + (self.hue_sat_lut[hue_bin] - 1.0) * mag_fac)
        gain_px = 1.0 + (base - 1.0) * lum_fac

        cb_out = np.clip(c_b * gain_px + CHROMA_CENTRE, 0, 255).astype(np.uint8)
        cr_out = np.clip(c_r * gain_px + CHROMA_CENTRE, 0, 255).astype(np.uint8)
        return np.stack((y_ch.astype(np.uint8), cb_out, cr_out), axis=-1)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(self.platform["in_file"], self.img,
                                  "Out_saturation_enhancement_", self.platform, self.conv_std)

    def execute(self):
        """
        Execute Saturation Enhancement
        """
        print("Saturation Enhancement = " + str(self.enable))
        if self.enable:
            start = time.time()
            print(f"   - SE - algorithm / gain = {self.algorithm} / {self.gain}")
            self.img = (self.apply_hue_rolloff() if self.algorithm == "hue_rolloff"
                        else self.apply_global())
            print(f"  Execution time: {time.time() - start:.3f}s")
        self.save()
        return self.img
