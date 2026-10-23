#!/usr/bin/env python3
"""obs_bw_modular_v3.py (builder analyst; B9 of SPEC_final4b_observables.md sec. 11.6, the one L = 8192 row of sec. 11.5: version 3 of
obs_bw_modular_v2.py, whose loop always ran both cuts (the centre and +W) per (profile, W)). Version 3 adds --shifts (the cut positions in
units of W, default 0: the centre cut only) and otherwise imports version 2's run_case unchanged: per (L, m_c) the v = 0 reference row (R_0,
the floor the rule compares against) and the requested (profile, W, cut) rows, with the same columns and the same separation
(R_in(uniform) - R_in(prediction))/R_0. Args: --L --W --mc --profiles --shifts --beta-max --gpu --out-json --md."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_bw_modular_v2 as J2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[8192])
    ap.add_argument("--W", type=float, nargs="+", default=[256.0])
    ap.add_argument("--mc", type=float, nargs="+", default=[0.0])
    ap.add_argument("--profiles", default="0:0.6")
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.0], help="cut positions in units of W (0: the centre)")
    ap.add_argument("--beta-max", type=float, default=3.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J2.GPU["dev"] = a.gpu
    profs = [tuple(float(y) for y in p.split(":")) for p in a.profiles.split(",")]
    res = []
    for L in a.L:
        for mc in a.mc:
            ref = J2.run_case(L, mc, ("uniform", 0.0), "v = 0 (reference)", a.beta_max)
            res.append(ref)
            print(json.dumps({k: ref[k] for k in ("label", "L", "mc", "n_test")}), {n: "%.3e" % ref[n]["R_in"] for n in ("pred", "uniform", "naive")}, "%.0f s" % ref["seconds"], flush=True)
            R0 = ref["uniform"]["R_in"]
            json.dump({"script": "obs_bw_modular_v3.py", "args": vars(a), "cases": res}, open(a.out_json, "w"), indent=1, default=float)
            for (v0, v1) in profs:
                for W in a.W:
                    for sh in a.shifts:
                        shift = sh * W
                        xp = L / 2 - 0.5 - shift
                        r = J2.run_case(L, mc, ("tanh", v0, v1, W, xp), "tanh v0 %g v1 %g W %g, cut at centre + %g" % (v0, v1, W, shift), a.beta_max)
                        r["separation_over_R0"] = (r["uniform"]["R_in"] - r["pred"]["R_in"]) / max(R0, 1e-300)
                        r["R0"] = R0
                        r["R_in_pred_over_R0"] = r["pred"]["R_in"] / max(R0, 1e-300)
                        res.append(r)
                        print(r["label"], L, mc, {n: "%.3e" % r[n]["R_in"] for n in ("pred", "uniform", "naive")}, "sep/R0 %.3f" % r["separation_over_R0"],
                              "R_in(pred)/R_0 %.4f" % r["R_in_pred_over_R0"], "%.0f s" % r["seconds"], flush=True)
                        json.dump({"script": "obs_bw_modular_v3.py", "args": vars(a), "cases": res}, open(a.out_json, "w"), indent=1, default=float)
    L_ = ["## J2 at L = %s: the one row of observables 11.5 (obs_bw_modular_v3.py; version 2's run_case)" % ", ".join(str(x) for x in a.L), "",
          "| case | L | m_c | n_test | R_in pred | R_in uniform | R_in naive | R_out pred | R_out uniform | commutator pred | commutator uniform | FD max pred | (R_unif - R_pred)/R_0 | R_in pred/R_0 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L_.append("| %s | %d | %g | %d | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %s | %s | %s |" % (
            r["label"], r["L"], r["mc"], r["n_test"], r["pred"]["R_in"], r["uniform"]["R_in"], r["naive"]["R_in"], r["pred"]["R_out"], r["uniform"]["R_out"],
            r["pred"]["commutator"], r["uniform"]["commutator"], ("%.2e" % r["pred"]["FD_max"]) if r["pred"]["FD_max"] is not None else "-",
            ("%.3f" % r["separation_over_R0"]) if "separation_over_R0" in r else "-", ("%.4f" % r["R_in_pred_over_R0"]) if "R_in_pred_over_R0" in r else "-"))
    L_ += ["", "- every number above is printed by the script; the rule (observables 11.5): R_in(pred)/R_0 within 20 percent of its L = 4096 value or falling."]
    open(a.md, "w").write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
