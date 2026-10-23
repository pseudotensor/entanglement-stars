#!/usr/bin/env python
"""ISO64_bridge_ladder_v1.py (composition 4, the solve lane): the ladder row of the isotropic bridge member ISO_64_s4 in the form of
MASS_ladder_v1.3 / SCALING_ladder_v1.3, by the v1.3 producer itself (ladders_readout_v1_3.py, imported unchanged from the box code dir
<internal>/qi2_ladders_code). Two members are added to the producer's member list:
  ISO_64_s4        the composed isoB history (<internal>/collapse_isoB.json, 227cd66d): the plain v36n member to
                   13.5, the Newton episode J2 13.625 to 14.5, the plain continuation J3 to 40 (verify_isoB_v1 PASS); role comparator:P64_s4,
                   so the producer's own spread loop prints the pair (P64_s4, ISO_64_s4) at matched clocks
  ISO_64_s4_plain  the original plain member (fde06ed7), the run the composition replaces past 13.5; role control_solver:ISO_64_s4
One change to the reading, and only for records that carry a solve_v32 block while the run's args say mix plain (the 8 Newton records of the
composition): the producer reads a member's confirming residual only when args.mix is newton, so these records' confirming residuals
(solve_v32.confirming_residual) are put back and the production rule (kept and confirming both <= 1e-3; banded above 1e-4) is re-applied to
them. Every other number is the producer's. Fits are the producer's (the added members enter none of them: the fits name their members).
Usage (box CPU): python ISO64_bridge_ladder_v1.py OUT_DIR"""
import json
import math
import os
import sys

import numpy as np

CODE = "<internal>/qi2_ladders_code"
sys.path.insert(0, CODE)
import ladders_readout_v1_3 as P  # noqa: E402
import engine_ladder_compare_v5 as LC  # noqa: E402

ISO_B = "<internal>/collapse_isoB.json"
ISO_PLAIN = ("<internal>/lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n"
             "collapse_lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n.json")
O = "<internal>/ladders_readout"

P.MEMBERS.append(("ISO_64_s4", lambda: ISO_B, "(0.2, 64) sigma/a 4 tau 8, isotropic dust set (the composed isoB history: plain v36n to 13.5, "
                  "Newton episode 13.625 to 14.5, plain to 40)", "comparator:P64_s4"))
P.MEMBERS.append(("ISO_64_s4_plain", lambda: ISO_PLAIN, "(0.2, 64) sigma/a 4 tau 8, isotropic dust set, the original plain run",
                  "control_solver:ISO_64_s4"))

_orig_load = P.load_member
NEWTON_PATCH = {}


def load_member(name, path, label, role, snap_dir):
    meta, cols, ints, zeros_all = _orig_load(name, path, label, role, snap_dir)
    if meta["newton"]:
        return meta, cols, ints, zeros_all
    J = json.load(open(path))
    recs = [r for r in (J.get("records") or []) if isinstance(r, dict) and "T_over_rs" in r and r.get("E") is not None]
    patched = []
    for i, r in enumerate(recs):
        sv = LC.v32_solve(r)
        if sv is None:
            continue
        c = sv.get("confirming_residual")
        res = cols["res"][i]
        cols["conf"][i] = c if c is not None else np.nan
        cf = cols["conf"][i]
        ints["conf_fail"][i] = 1 if (not np.isfinite(cf)) or cf > 1e-3 else 0
        ints["above"][i] = 1 if (np.isfinite(res) and res > 1e-3) else 0
        ok = np.isfinite(res) and res <= 1e-3 and not ints["conf_fail"][i]
        ints["counts"][i] = 1 if ok else 0
        ints["banded"][i] = 1 if (ok and (res > 1e-4 or (np.isfinite(cf) and cf > 1e-4))) else 0
        patched.append({"T": float(cols["T"][i]), "kept": float(res), "confirming": (float(cf) if np.isfinite(cf) else None),
                        "counts": bool(ints["counts"][i]), "banded": bool(ints["banded"][i])})
    meta["newton_records_in_plain_args"] = patched
    NEWTON_PATCH[name] = patched
    return meta, cols, ints, zeros_all


P.load_member = load_member


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sys.argv = [sys.argv[0], "--catalog", O + "/catalog_v1.json", "--catalog", O + "/catalog_v1_A.json", "--catalog", O + "/catalog_v1_B.json",
                "--out-dir", out, "--certext", O + "/v1_3/certext_production_v1.json", "--followup", O + "/v1_3/certext_followup_v1.json"]
    P.main()
    json.dump(NEWTON_PATCH, open(os.path.join(out, "newton_patch_v1.json"), "w"), indent=1)
    print("newton records patched:", {k: len(v) for k, v in NEWTON_PATCH.items()})


if __name__ == "__main__":
    main()
