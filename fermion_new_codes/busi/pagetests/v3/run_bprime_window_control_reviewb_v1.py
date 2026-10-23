#!/usr/bin/env python
"""run_bprime_window_control_reviewb_v1.py (busi; B' v2 CPU control: reviewer B's artificial-window control, ReviewB_pc13 sec. 1.7, the launch
driver ReviewB_bpchk_v1 found missing). ARTIFICIAL (labelled): the block's onsite levels are not realizable by h[N, v].
Per call: one n_w (--nw), the arms --arms (mixing, bright) x --slides (0 = stationary; a monotone slide of --slide-windows window widths
over the observation time), the native kappa = 1 chain port on --family's profile at size --L, coupling norm G^2 = --g2 T0^2 fixed as n_w
changes. Saved per arm: the summary (emitted number, the depletion spectrum, largest/median, the return diagonal, the hole's exit, the
paired mode's nested entropies) in the json, and the complete escape Gram at the end clock (and at every clock for 2 n_w <= 200) with the
block's levels and couplings in <tag>.npz. Box CPU."""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_wca_v2 as WC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nw", type=int, required=True)
    ap.add_argument("--family", choices=["F1", "F2"], default="F1")
    ap.add_argument("--L", type=float, default=64.0)
    ap.add_argument("--arms", default="mixing,bright")
    ap.add_argument("--slides", default="0,1")
    ap.add_argument("--g2", type=float, default=0.1)
    ap.add_argument("--T-obs-over-L", type=float, default=200.0)
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    prof = CU.dense_record_profile(*CU.REC_F1) if a.family == "F1" else CU.f2_profile()
    L = a.L; T0 = 1.0 / (4 * math.pi * L)
    res = {"script": "run_bprime_window_control_reviewb_v1.py", "args": vars(a), "profile": prof["label"], "label": "ARTIFICIAL block on the native chain port",
           "own_md5": K3.own_md5s(), "runs": []}
    npz = {}
    for arm in a.arms.split(","):
        for sw in [float(x) for x in a.slides.split(",")]:
            rate = sw * T0 / (a.T_obs_over_L * L)
            full, r = WC.window_control_reviewb_v2(a.nw, L, prof, arm=arm, slide_rate=rate, G2_over_T0=a.g2, T_obs_over_L=a.T_obs_over_L, n_bins=a.n_bins)
            r["slide_windows"] = sw
            res["runs"].append(r)
            key = "%s_slide%g" % (arm, sw)
            npz[key + "_G_end"] = full["G_end"]
            if full["G_bins"] is not None:
                npz[key + "_G_bins"] = full["G_bins"]
            npz[key + "_e_s"] = full["e_s"]; npz[key + "_e_m"] = full["e_m"]; npz[key + "_g"] = full["g"]; npz[key + "_Gc"] = np.array(full["Gc"])
            K3.log("n_w %d %s slide %g: emitted %.4f, depletion top %s, largest/median %.3g, n(p>1e-2) %d, hole exit %.4f (%.0f s)" % (
                a.nw, arm, sw, r["emitted_number_supply"], np.round(r["depletion_spectrum_end"][:3], 4), r["largest_over_median"], r["n_p_above_1e-2"],
                r["hole_exit"], time.time() - t0))
            K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    npz["t"] = full["t"]
    np.savez_compressed(os.path.join(a.out, a.tag + ".npz"), **npz)
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s" % a.tag)


if __name__ == "__main__":
    main()
