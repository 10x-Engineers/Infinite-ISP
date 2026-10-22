"""
File: auto_exposure.py
Description: 3A-AE control - EV-law algorithm. Once per frame: meters a
             clip-corrected, centre-weighted green mean from the AE statistics,
             compares it against a target in EV (log2) terms, protects against
             blown highlights / clipping / an already-dark frame, and moves the
             digital gain towards that target with a damped, slew-limited step - a
             deadband (widened by hysteresis once converged) holds the gain once
             it's close enough. Replaces the histogram-skewness AE.

             This is the algorithm-design counterpart of the fixed-point EV-law
             control in Infinite-ISP_ReferenceModel: same algorithm (meter / decide
             / actuate, same caps and bracket-hold), but worked directly in floating
             point and plain EV units - no Q8 fixed-point scaling, register-width
             checks or hardware-shaped bit tricks, since none of that is needed here.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import math


def new_ae_state():
    """The block's memory, carried frame to frame: the filtered metered brightness,
    the convergence flag, and the gain-ladder bracket-hold bookkeeping."""
    return {
        "vm_filt": None,
        "converged": False,
        "came_from": None,
        "age": 0,
        "hold": None,
    }


class AutoExposure:
    """
    Auto Exposure control (EV law)
    """

    def __init__(self, ae_stats, parm_ae, parm_dga, state, temporal):
        self.stats = ae_stats
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.prm = parm_ae
        self.weights = parm_ae["grid_weights"]
        self.gains = list(parm_dga["gain_array"])
        self.gain_index = int(parm_dga["current_gain"])
        self.state = state
        # temporal=False is used for render_3a: the same frame is re-processed to
        # converge, so there is no real frame-to-frame history to smooth against
        self.temporal = temporal
        if self.enable and self.stats is None:
            raise ValueError(
                "auto_exposure needs the AE statistics: set "
                "auto_exposure_stats.is_enable to true."
            )

    # ------------------------------------------------------------------ 1. meter
    def meter(self):
        """Clip-corrected, centre-weighted green mean + frame-wide fractions."""
        prm = self.prm
        acc = wsum = 0.0
        hi_cnt = wc_cnt = dk_cnt = g_tot = p_tot = 0
        for cell, weight in zip(self.stats, self.weights):
            g_tot += cell["green_count"]
            p_tot += cell["pixel_count"]
            hi_cnt += cell["highlight_count"]
            wc_cnt += cell["white_clip_count"]
            dk_cnt += cell["shadow_count"]

            if cell["valid_green_count"] < cell["green_count"] * prm["valid_min_pct"] / 100:
                continue
            acc += cell["valid_green_mean"] * weight
            wsum += weight

        g_tot, p_tot = max(1, g_tot), max(1, p_tot)
        return {
            "valid": wsum > 0,
            "vm": acc / wsum if wsum else 0.0,
            "hi_frac_pm": 1000 * hi_cnt / g_tot,
            "dark_frac_pm": 1000 * dk_cnt / g_tot,
            "clip_frac_pm": 1000 * wc_cnt / p_tot,
        }

    # ------------------------------------------------------------------ 2. decide
    def decide(self, mtr):
        """Decision: (ev_want, ev_move) in EV; ev_move == 0 means hold."""
        prm, state = self.prm, self.state

        # smooth the metered mean across frames (skipped for render_3a: see __init__)
        iir_a = prm["vm_iir_a"] if self.temporal else 100
        vm = mtr["vm"] if mtr["vm"] else 1.0
        if state["vm_filt"] is None:
            state["vm_filt"] = vm
        else:
            state["vm_filt"] += (vm - state["vm_filt"]) * iir_a / 100
        vmf = max(1.0, state["vm_filt"])

        if not mtr["valid"]:
            # no cell had enough unclipped greens to meter - move cautiously
            ev_want = (
                -prm["ev_slew_max"]
                if mtr["clip_frac_pm"] > mtr["dark_frac_pm"]
                else prm["ev_slew_max"]
            )
        else:
            ev_mean = math.log2(prm["target_mean"]) - math.log2(vmf)
            ev_want = ev_mean
            # highlight / clip protection: only ever pulls the request darker
            if mtr["hi_frac_pm"]:
                over = math.log2(mtr["hi_frac_pm"]) - math.log2(prm["hi_frac_pm"])
                ev_want = min(ev_want, -(over * prm["hi_k"] / 100))
            if mtr["clip_frac_pm"]:
                over = math.log2(mtr["clip_frac_pm"]) - math.log2(prm["clip_frac_pm"])
                ev_want = min(ev_want, -(over * prm["clip_k"] / 100))
            # the caps above can't pull more than ev_max_pull below the plain target
            ev_want = max(ev_want, ev_mean - prm["ev_max_pull"])
            # never darken a frame that's already mostly dark
            if mtr["dark_frac_pm"] > prm["dark_frac_pm"] and ev_want < 0:
                ev_want = 0.0

        band = prm["ev_deadband"] + (prm["ev_hyst"] if state["converged"] else 0)
        if abs(ev_want) <= band:
            state["converged"] = True
            return ev_want, 0.0
        state["converged"] = False

        damp, slew = prm["ev_damp"], prm["ev_slew_max"]
        if abs(ev_want) > prm["scene_change_ev"]:
            damp, slew = prm["ev_damp_fast"], prm["ev_slew_fast"]
        ev_move = max(-slew, min(slew, ev_want * damp / 100))
        return ev_want, ev_move

    # ------------------------------------------------------------------ 3. actuate
    def gain_ev(self, index):
        """EV of a gain-table entry (gain = 1 is the 0 EV reference point)."""
        return math.log2(self.gains[index])

    def actuate(self, mtr, ev_want, ev_move):
        """Move along the gain table towards ev_move, without oscillating when the
        ideal gain falls between two adjacent table entries (see auto_exposure.py's
        module docstring / the AE walkthrough for why the bracket-hold is needed)."""
        cur, state, prm = self.gain_index, self.state, self.prm
        over = mtr["hi_frac_pm"] > prm["hi_frac_pm"] or mtr["clip_frac_pm"] > prm["clip_frac_pm"]
        ev_raw = math.log2(max(1.0, mtr["vm"]))

        best = cur
        if ev_move != 0:
            aim = self.gain_ev(cur) + ev_move
            best = min(range(len(self.gains)), key=lambda k: abs(self.gain_ev(k) - aim))
            if best == cur:
                # the table can't reach the aim point - still take one step if it
                # lands closer to the full (undamped) target
                step = cur + (1 if ev_move > 0 else -1)
                target = self.gain_ev(cur) + ev_want
                if 0 <= step < len(self.gains) and abs(self.gain_ev(step) - target) < abs(
                    self.gain_ev(cur) - target
                ):
                    best = step

        # an "end" of a bracket: (gain index, ev_want, over budget, raw metered EV)
        entry = (cur, ev_want, over, ev_raw)
        hold, came = state["hold"], state["came_from"]
        if hold and hold[0] == cur and abs(ev_raw - hold[2]) > prm["ev_deadband"]:
            hold = came = None  # the scene moved on: release the hold
        if hold and hold[0] == cur and best == hold[1]:
            best = cur  # still bracketed between the same two gains: stay put
        elif came and best != cur and best == came[0] and (came[1] > 0) != (ev_want > 0):
            # about to bounce back to where we came from - settle the bracket
            entry_darker = self.gains[entry[0]] <= self.gains[came[0]]
            darker, brighter = (entry, came) if entry_darker else (came, entry)
            pick = darker if brighter[2] or abs(darker[1]) <= abs(brighter[1]) else brighter
            if pick is came or state["age"] == 1:
                other = brighter if pick is darker else darker
                hold, best = (pick[0], other[0], pick[3]), pick[0]

        if best != cur:
            came, state["age"] = entry, 0
        state["age"] = min(state["age"] + 1, 3)
        state["hold"], state["came_from"] = hold, came
        return best

    def execute(self):
        """
        Execute AE: returns the decision for the next frame, or None when disabled
        """
        print("Auto Exposure (EV law) = " + str(self.enable))
        if not self.enable:
            return None
        start = time.time()
        mtr = self.meter()
        ev_want, ev_move = self.decide(mtr)
        index = self.actuate(mtr, ev_want, ev_move)
        result = dict(
            mtr,
            target_mean=self.prm["target_mean"],
            ev_err=ev_want,
            ev_move=ev_move,
            converged=self.state["converged"],
            gain_index=index,
            gain=self.gains[index],
            moved=index != self.gain_index,
        )
        if self.is_debug:
            print(f"   - AE - metered mean / target   = {mtr['vm']:.1f} / {self.prm['target_mean']}")
            print(
                f"   - AE - highlight/clip/dark     = "
                f"{mtr['hi_frac_pm']:.1f}/{mtr['clip_frac_pm']:.1f}/{mtr['dark_frac_pm']:.1f} per mille"
            )
            print(f"   - AE - EV error / move         = {ev_want:+.3f} / {ev_move:+.3f} EV")
            print(
                f"   - AE - digital gain             = "
                f"{self.gains[self.gain_index]} -> {self.gains[index]}"
            )
        print(f"  Execution time: {time.time() - start:.3f}s")
        return result
