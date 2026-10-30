"""
File: auto_exposure.py
Description: 3A-AE - an EV-law auto-exposure control law, driving Infinite-ISP's
             digital gain.

             Input: the AE statistics block (auto_exposure_stats).  Per frame:

             1. Meter (ae_meter): per grid cell, a clip-corrected green mean -
                saturated greens are removed from the sum (the all-pixel clip counts
                halved to green and bounded by the green-only counters), cells with
                too few valid greens are dropped - then a centre-weighted mean `vm`
                over the cells.  Plus three FRAME-level fractions, in per mille:
                greens above the highlight threshold, pixels at full scale, greens
                below the shadow threshold.
             2. Decide (ae_expose_ev): the exposure error in EV,
                    ev = log2(target_mean) - log2(vm)                (Q8: 256 = 1 EV)
                capped DOWN when the highlight or clip fraction is over its budget,
                floored at 0 when most of the frame is dark (a dark frame is never
                driven darker).  Inside the deadband (+ hysteresis once converged)
                nothing moves; otherwise a damped, slew-limited EV move is made.
             3. Actuate: the firmware turns the EV move into sensor shutter / gain
                steps; here it moves along the digital-gain ladder (`gain_array` of
                the digital_gain module) to the gain nearest the requested exposure.
                A target between two gains is settled on one of them (the smaller
                error, never a brighter gain that breaks the highlight / clip
                budget) instead of alternating; when the ladder cannot get closer
                the loop has converged as far as the gains allow.

             Integer arithmetic and the Q8 log2 follow a reference firmware model,
             generalised from the firmware's fixed 10-bit, fixed-grid constants to
             this frame's bit depth and grid.

Code / Paper  Reference: EV-law auto-exposure control firmware
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time

EV_ONE = 256                                  # Q8: 256 = 1 EV
_K_LOG2 = [0, 44, 82, 118, 150, 179, 207, 232, 256]


def cdiv(num, den):
    """C integer division (truncates toward zero; Python // floors negatives)."""
    quo = abs(num) // abs(den)
    return quo if (num < 0) == (den < 0) else -quo


def log2_q8(value):
    """Firmware log2 in Q8: exponent + 3-bit-indexed, linearly interpolated mantissa."""
    if value < 2:
        return 0
    exp = 0
    while exp < 31 and (value >> (exp + 1)):
        exp += 1
    frac = ((value >> (exp - 6)) & 63) if exp >= 6 else ((value << (6 - exp)) & 63)
    idx, rem = frac >> 3, frac & 7
    mant = _K_LOG2[idx] + cdiv((_K_LOG2[idx + 1] - _K_LOG2[idx]) * rem, 8)
    return exp * EV_ONE + mant


def new_ae_state():
    """Controller memory carried from frame to frame: the firmware's (filtered mean,
    convergence) plus the gain ladder's (the gain it came from and how many frames
    ago, the held bracket)."""
    return {"vm_filt_q4": 0, "converged": 0, "came_from": None, "age": 0, "hold": None}


class AutoExposure:
    """
    Auto Exposure control (EV law) on the AE statistics
    """

    def __init__(self, ae_stats, sensor_info, parm_ae, parm_dga, state, temporal):
        self.stats = ae_stats
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.full_scale = (1 << sensor_info["bit_depth"]) - 1
        self.prm = {k: int(parm_ae[k]) for k in (
            "target_mean", "valid_min_pct", "ev_deadband", "ev_hyst", "ev_damp",
            "ev_damp_fast", "ev_slew_max", "ev_slew_fast", "scene_change_ev", "vm_iir_a",
            "hi_frac_pm", "hi_k", "clip_frac_pm", "clip_k", "dark_frac_pm", "ev_max_pull")}
        self.weights = [int(w) for w in parm_ae["grid_weights"]]
        self.gains = list(parm_dga["gain_array"])
        self.gain_index = int(parm_dga["current_gain"])
        self.state = state
        # temporal=False: the same frame is processed again (render_3a), so there is
        # no frame-to-frame noise for the mean's IIR to filter - it is bypassed
        self.temporal = temporal
        if self.enable:
            if self.stats is None:
                raise ValueError("auto_exposure needs the AE statistics: set "
                                 "auto_exposure_stats.is_enable to true.")
            if len(self.weights) != len(self.stats["grid_mean_sum"]):
                raise ValueError(f"auto_exposure.grid_weights needs one weight per grid "
                                 f"cell ({len(self.stats['grid_mean_sum'])}).")

    # ------------------------------------------------------------------ 1. meter
    def meter(self):
        """ae_meter: clip-corrected, centre-weighted green mean + frame fractions."""
        st, fsc = self.stats, self.full_scale
        acc = wsum = hi_cnt = wc_cnt = dk_cnt = 0
        for i, weight in enumerate(self.weights):
            gpc = st["green_pixel_count"][i]
            hi_cnt += st["highlight_count"][i]
            wc_cnt += st["white_clip_count"][i]
            dk_cnt += st["shadow_count"][i]
            # all-pixel clip counts halved to green, bounded by the green-only
            # counters (a green at full scale is necessarily a highlight, at 0 a shadow)
            wgc = min(st["white_clip_count"][i] >> 1, st["highlight_count"][i])
            bgc = min(st["black_clip_count"][i] >> 1, st["shadow_count"][i])
            vcnt = gpc - (wgc + bgc)
            if vcnt <= 0 or vcnt < gpc * self.prm["valid_min_pct"] // 100:
                continue
            cell_mean = max(0, st["grid_mean_sum"][i] - wgc * fsc) // vcnt
            acc += cell_mean * weight
            wsum += weight
        g_tot = max(1, sum(st["green_pixel_count"]))
        p_tot = max(1, sum(st["pixel_count"]))
        return {"valid": 1 if wsum else 0, "vm": acc // wsum if wsum else 0,
                "hi_frac_pm": hi_cnt * 1000 // g_tot,
                "dark_frac_pm": dk_cnt * 1000 // g_tot,
                "clip_frac_pm": wc_cnt * 1000 // p_tot}

    # ------------------------------------------------------------------ 2. decide
    def decide(self, mtr):
        """ae_expose_ev decision: (ev_want, ev_move) in Q8 EV; ev_move 0 = hold."""
        prm, state = self.prm, self.state
        vm_q4 = (mtr["vm"] if mtr["vm"] else 1) << 4
        iir_a = prm["vm_iir_a"] if self.temporal else 100
        if not state["vm_filt_q4"]:
            state["vm_filt_q4"] = vm_q4
        else:
            state["vm_filt_q4"] += cdiv((vm_q4 - state["vm_filt_q4"]) * iir_a, 100)
        vmf = max(1, state["vm_filt_q4"] >> 4)

        if not mtr["valid"]:
            ev_want = (-prm["ev_slew_max"] if mtr["clip_frac_pm"] > mtr["dark_frac_pm"]
                       else prm["ev_slew_max"])
        else:
            ev_mean = log2_q8(prm["target_mean"]) - log2_q8(vmf)
            ev_want = ev_mean
            if mtr["hi_frac_pm"]:                 # passes through 0 at the budget
                over = log2_q8(mtr["hi_frac_pm"]) - log2_q8(prm["hi_frac_pm"])
                ev_want = min(ev_want, cdiv(-(over * prm["hi_k"]), 100))
            if mtr["clip_frac_pm"]:
                over = log2_q8(mtr["clip_frac_pm"]) - log2_q8(prm["clip_frac_pm"])
                ev_want = min(ev_want, cdiv(-(over * prm["clip_k"]), 100))
            ev_want = max(ev_want, ev_mean - prm["ev_max_pull"])
            if mtr["dark_frac_pm"] > prm["dark_frac_pm"] and ev_want < 0:
                ev_want = 0

        band = prm["ev_deadband"] + (prm["ev_hyst"] if state["converged"] else 0)
        if abs(ev_want) <= band:
            state["converged"] = 1
            return ev_want, 0
        state["converged"] = 0
        damp, slew = prm["ev_damp"], prm["ev_slew_max"]
        if abs(ev_want) > prm["scene_change_ev"]:
            damp, slew = prm["ev_damp_fast"], prm["ev_slew_fast"]
        ev_move = max(-slew, min(slew, cdiv(ev_want * damp, 100)))
        if ev_move == 0:
            ev_move = 1 if ev_want > 0 else -1
        return ev_want, ev_move

    # ------------------------------------------------------------------ 3. actuate
    def actuate(self, mtr, ev_want, ev_move):
        """Move along the digital-gain ladder.

        The gain nearest the requested exposure (gains compared in Q8 log2 of their
        Q8 value); if that is the current gain, still one step when it lands closer
        to the full target (the firmware's minimum move).  The ladder is coarse
        where the firmware's shutter is fine, so a target can fall BETWEEN two
        gains: the loop then asks to go back to the gain it came from (A -> B ->
        A), the error having changed sign - possibly after dwelling at B for a few
        frames while the mean's IIR catches up.  That bracket is settled on the end
        with the smaller error - never on the brighter end if it breaks the
        highlight or clip budget (the default policy: protect highlights) - and held
        while the scene stays put: the RAW metered mean at the held gain within the
        deadband (the filtered error still drifts for frames after a move, so it
        cannot tell a scene change from the filter catching up).  The reference of
        the end held is the raw mean metered AT that gain: when it is the end being
        moved to, its reference is checked on arrival; staying at the current end is
        decided only when the other end was metered on the previous frame, so an old
        reading can never pin the gain."""
        cur, state, prm = self.gain_index, self.state, self.prm
        over = (mtr["hi_frac_pm"] > prm["hi_frac_pm"] or
                mtr["clip_frac_pm"] > prm["clip_frac_pm"])
        ev_raw = log2_q8(max(1, mtr["vm"]))
        best = cur
        if ev_move != 0:
            ev_of = [log2_q8(int(g * EV_ONE)) - 8 * EV_ONE for g in self.gains]
            aim = ev_of[cur] + ev_move
            best = min(range(len(self.gains)), key=lambda k: abs(ev_of[k] - aim))
            if best == cur:
                step = cur + (1 if ev_move > 0 else -1)
                target = ev_of[cur] + ev_want
                if 0 <= step < len(self.gains) and \
                        abs(ev_of[step] - target) < abs(ev_of[cur] - target):
                    best = step

        # an end: (gain index, ev_want, over budget, log2 raw mean) metered at a gain
        entry = (cur, ev_want, over, ev_raw)
        hold, came = state["hold"], state["came_from"]
        if hold and hold[0] == cur and abs(ev_raw - hold[2]) > prm["ev_deadband"]:
            hold = came = None                            # the scene moved: release
        if hold and hold[0] == cur and best == hold[1]:
            best = cur                                    # still bracketed: stay
        elif came and best != cur and best == came[0] and (came[1] > 0) != (ev_want > 0):
            darker, brighter = sorted([entry, came], key=lambda e: self.gains[e[0]])
            pick = darker if brighter[2] else min((darker, brighter), key=lambda e: abs(e[1]))
            if pick is came or state["age"] == 1:
                other = brighter if pick is darker else darker
                hold, best = (pick[0], other[0], pick[3]), pick[0]
        if best != cur:
            came, state["age"] = entry, 0
        state["age"] = min(state["age"] + 1, 3)        # only age == 1 is used
        state["hold"], state["came_from"] = hold, came
        return best

    def execute(self):
        """Execute AE: returns the decision for the next frame, or None when disabled."""
        print("Auto Exposure (EV law) = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        mtr = self.meter()
        ev_want, ev_move = self.decide(mtr)
        index = self.actuate(mtr, ev_want, ev_move)
        result = dict(mtr, target_mean=self.prm["target_mean"], ev_err=ev_want,
                      ev_move=ev_move, converged=bool(self.state["converged"]),
                      gain_index=index, gain=self.gains[index],
                      moved=index != self.gain_index)
        if self.is_debug:
            print(f"   - AE - metered mean / target   = {mtr['vm']} / {self.prm['target_mean']}")
            print(f"   - AE - highlight / clip / dark = {mtr['hi_frac_pm']} / "
                  f"{mtr['clip_frac_pm']} / {mtr['dark_frac_pm']} per mille")
            print(f"   - AE - EV error / move         = {ev_want / EV_ONE:+.2f} / "
                  f"{ev_move / EV_ONE:+.2f} EV")
            print(f"   - AE - digital gain            = {self.gains[self.gain_index]} -> "
                  f"{self.gains[index]}")
        print(f"  Execution time: {time.time() - start:.3f}s")
        return result
