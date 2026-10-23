#!/usr/bin/env python
"""engine_massless_nsub_test_v1.py (builder analyst; engine spec 30.2, the decisive resolution test in place of the queued relaunch:
"the first re-solve from the prepared r_s 32 state (the queued member's parameters) at tau 2 with the convention's sub-step count nsub 1,
4, 16, 64 (c_LZ 2 is nsub 2), on v28 with Anderson, printing per nsub the continued intervals, the pairs' shell energy and its interior
integral to R_0 - 3w, the S9 (a) ratio per channel, the certificate failures and E/E0 ... The reading, chosen now: if the continued
interval's extent and the pairs' interior energy fall by more than a factor 2 from nsub 1 to 64, the member is a resolution problem with a
known cost ...; if they do not fall, the member's statement is FILE 4").

The member (the withdrawn relaunch's parameters by default): r_s 32, 1024 sites, m 0, g 1/4, ell 4, 10 channels, tau 2, R_0 = 6 r_s,
(k_0, w) = (0.4, 20) (--shell-width-over-rs 0.625), c_LZ 1, site fields averaged. The preparation (engine_collapse_v28's, i.e. v22's fill
to the charge on fields v17) runs once and is reused for every nsub (copies of its fields and dust columns); per nsub engine_collapse_v28
runs the T = 0 record and the first re-solve (T = tau) with --force-nsub nsub (the continuation of the re-solve along nsub sub-steps of
the linear path of the fields, the eigendecompositions at the nsub - 1 intermediate configurations, ops v4's assignment at each) and
--mix anderson and --fields (v18 by default: bitwise v17 wherever v17 solves; the withdrawn relaunch's preparation failed
on pod 1 inside its empty interior), its records in --out (collapse_<tag>_nsub<n>.json and npz). Printed per nsub from the re-solve's record: the continued
intervals and their total extent, the pairs' shell energy sum_n eps_pairs,n and its interior part (the sites below R_0 - 3w), the S9 (a)
ratios per channel (||Delta h||_2/(nsub g_mix) and ||Delta h||_2/(tau g_mix^2), g_mix, the Lanczos norm), the certificate's failures
(2 eta_n >= g_n), E/E0, the iterations and the stop (v28's rule), the seconds; then the smallest listed nsub at which every channel's
sub-step ratio is below 1/8 (or none) beside the estimate 8 times the largest sub-step ratio at the first listed nsub, and the verdict line,
30.2's pre-stated reading applied to the first and last listed nsub: both the extent and |the interior pairs' energy| falling by more than
a factor 2, or not. The record line of the test carries 30.2's sentence: the massless S9 (a) finding is the gapless case of 29.3's
level-spacing population (A3). Every number is printed by the script. H100 (pod 1 lane 4) at full size; one step timed first with
--nsubs 1. Args: --rs --nsites --tau --ell --kappa-sea --g --shell-R0-over-rs --shell-width-over-rs --shell-k0 --c-LZ --nsubs --mix --maxit
--stall-fd-dim --fields --gpu --out --tag --md --json."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v28 as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=32.0); ap.add_argument("--nsites", type=int, default=1024); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--ell", type=int, default=4); ap.add_argument("--kappa-sea", type=int, default=10); ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--shell-R0-over-rs", type=float, default=6.0); ap.add_argument("--shell-width-over-rs", type=float, default=0.625)
    ap.add_argument("--shell-k0", type=float, default=0.4); ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--nsubs", default="1,4,16,64"); ap.add_argument("--mix", default="anderson", choices=["anderson", "plain"])
    ap.add_argument("--maxit", type=int, default=20); ap.add_argument("--stall-fd-dim", type=int, default=4); ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fields", default="v18", choices=["v17", "v18"], help="engine_fields_v18 (v17 with the flat fallback inside the empty interior; bitwise v17 where v17 solves) or v17")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nsubs = [int(x) for x in a.nsubs.split(",") if x.strip()]
    E.C5.backend(a.gpu)
    orig = E.prepare_massless_shell_v26
    keep = {}

    def prepared(chs, g, R, ell, rs, args, passes=200, tol=1e-10):
        """the preparation once (engine_collapse_v28's), then copies of its fields, dust columns and head."""
        if "prep" not in keep:
            t0 = time.time()
            N, v, dust, info, head = orig(chs, g, R, ell, rs, args, passes, tol)
            keep["prep"] = (N.copy(), v.copy(), {k: (P.clone(), np.asarray(f).copy()) for k, (P, f) in dust.items()}, info, json.loads(json.dumps(head, default=float)))
            keep["seconds"] = time.time() - t0
        N, v, dust, info, head = keep["prep"]
        return N.copy(), v.copy(), {k: (P.clone(), f.copy()) for k, (P, f) in dust.items()}, info, copy.deepcopy(head)
    E.prepare_massless_shell_v26 = prepared
    R0 = a.shell_R0_over_rs * a.rs; w = a.shell_width_over_rs * a.rs; r_cut = R0 - 3.0 * w
    rows = []
    t_all = time.time()
    for nsub in nsubs:
        tag = "%s_nsub%d" % (a.tag, nsub)
        argv = ["--massless-shell", "--m", "0", "--g", repr(a.g), "--rs", repr(a.rs), "--nsites", str(a.nsites), "--ell", str(a.ell), "--kappa-sea", str(a.kappa_sea),
                "--tau", repr(a.tau), "--T-over-rs", repr(a.tau / a.rs), "--reference", "R2", "--site-fields", "averaged", "--shell-R0-over-rs", repr(a.shell_R0_over_rs),
                "--shell-width-over-rs", repr(a.shell_width_over_rs), "--shell-k0", repr(a.shell_k0), "--c-LZ", repr(a.c_LZ), "--mix", a.mix, "--maxit", str(a.maxit),
                "--stall-fd-dim", str(a.stall_fd_dim), "--fields", a.fields, "--force-nsub", str(nsub), "--out", a.out, "--tag", tag]
        if a.gpu is not None:
            argv += ["--gpu", str(a.gpu)]
        args = E.configure(E.build_parser().parse_args(argv))
        t0 = time.time()
        E.run(args)
        sec = time.time() - t0
        J = json.load(open(os.path.join(a.out, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(a.out, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        R = np.asarray(Z["R"], float)
        recs = J["records"]
        rec = recs[1] if len(recs) > 1 else recs[-1]
        row = {"nsub": nsub, "tag": tag, "seconds_total": sec, "records": len(recs)}
        if rec.get("E") is None:
            row["failed"] = rec.get("failed")
        else:
            T = np.asarray(Z["T"], float); k = int(np.argmin(np.abs(T - float(rec["T"]))))
            ep = np.asarray(Z["eps_pairs"][k], float)
            ivs = (rec.get("gauge") or {}).get("intervals") or []
            v26 = rec.get("v26") or {}; dg = (v26.get("step_diag") or {}).get("per_channel") or {}; c = v26.get("certificate") or {}; stp = v26.get("stop") or {}
            row.update({"T_over_rs": rec["T_over_rs"], "E_over_E0": rec["E_over_E0"], "continued": ivs, "extent": float(sum(b_ - a_ for a_, b_ in ivs)),
                        "pairs_shell_energy": float(rec["E_pairs_shells"]), "pairs_interior": float(ep[R < r_cut].sum()), "r_cut": r_cut,
                        "s9a": {kk: {q: dg[kk].get(q) for q in ("ratio_step", "ratio_rate", "g_mix", "norm2_lanczos", "nsub")} for kk in sorted(dg, key=int)},
                        "certificate_failures": c.get("failures"), "channels": c.get("channels"), "iterations": rec["iterations"], "converged": rec.get("converged"),
                        "accepted": stp.get("accepted"), "unconverged": {q: (rec.get("unconverged") or {}).get(q) for q in ("residual", "iteration", "dE")} if rec.get("unconverged") else None,
                        "forced_nsub": stp.get("forced_nsub"), "seconds_record": rec.get("seconds"), "record_label": rec.get("record_label")})
            t0r = recs[0]
            row["T0"] = {"E_over_E0": t0r.get("E_over_E0"), "iterations": t0r.get("iterations"), "converged": t0r.get("converged"), "seconds": t0r.get("seconds"),
                         "accepted": ((t0r.get("v26") or {}).get("stop") or {}).get("accepted")}
        rows.append(row)
        print("nsub %d done (%.0f s): %s" % (nsub, sec, json.dumps({q: row.get(q) for q in ("extent", "pairs_shell_energy", "pairs_interior", "E_over_E0", "certificate_failures", "iterations", "failed")}, default=float)), flush=True)
    ok = [r for r in rows if not r.get("failed")]
    first, last = (ok[0], ok[-1]) if ok else (None, None)

    def ratio(x, y):
        if x is None or y is None:
            return None
        return math.inf if y == 0.0 and x != 0.0 else (abs(x) / abs(y) if y != 0.0 else None)
    meets = [r["nsub"] for r in ok if r["s9a"] and all((q["ratio_step"] is not None and q["ratio_step"] < 0.125) for q in r["s9a"].values())]
    est = (8.0 * max(q["ratio_step"] for q in first["s9a"].values()) * first["nsub"]) if (first and first["s9a"]) else None
    verdict = None
    if first is not None and last is not None and first is not last:
        re_ = ratio(first["extent"], last["extent"]); rp_ = ratio(first["pairs_interior"], last["pairs_interior"])
        both = bool(re_ is not None and rp_ is not None and re_ > 2.0 and rp_ > 2.0)
        verdict = {"from_nsub": first["nsub"], "to_nsub": last["nsub"], "extent": [first["extent"], last["extent"]], "extent_ratio": re_,
                   "pairs_interior": [first["pairs_interior"], last["pairs_interior"]], "pairs_interior_ratio": rp_, "falls_by_more_than_2_in_both": both}
    out = {"script": "engine_massless_nsub_test_v1.py", "args": vars(a), "preparation_seconds": keep.get("seconds"), "rows": rows, "smallest_nsub_meeting_1_8": (min(meets) if meets else None),
           "estimate_8_times_ratio": est, "verdict": verdict, "seconds": time.time() - t_all}
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    f = lambda x, fm="%.6e": (fm % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and math.isfinite(x) else ("inf" if isinstance(x, float) and math.isinf(x) else ("-" if x is None else str(x)))
    L = ["## The massless member's decisive resolution test (engine_massless_nsub_test_v1.py; engine spec 30.2)", "",
         "- the member: r_s %g, %d sites, m 0, g %g, ell %d, %d channels, tau %g, R_0 = %g r_s, w = %g r_s (k_0 %g), c_LZ %g, site fields averaged; engine_collapse_v28 with --mix %s, --maxit %d, --stall-fd-dim %d, --fields %s; the preparation once (%s s), reused for every nsub; per nsub the T = 0 record and the first re-solve with the continuation forced along nsub sub-steps." % (
             a.rs, a.nsites, a.g, a.ell, a.kappa_sea, a.tau, a.shell_R0_over_rs, a.shell_width_over_rs, a.shell_k0, a.c_LZ, a.mix, a.maxit, a.stall_fd_dim, a.fields, f(keep.get("seconds"), "%.0f")),
         "- the record line: the massless S9 (a) finding is the gapless case of 29.3's level-spacing population (A3; engine spec 30.2).", "",
         "| nsub | T/rs | continued intervals | total extent | pairs' shell energy | pairs' interior energy (R < %g) | E/E0 | certificate failures | iterations | stop | T = 0 record (iterations; stop) | seconds |" % r_cut,
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("failed"):
            L.append("| %d | failed: %s |  |  |  |  |  |  |  |  |  | %s |" % (r["nsub"], r["failed"], f(r["seconds_total"], "%.0f")))
            continue
        stop = ("converged at iteration %d" % r["accepted"]["iteration"]) if r.get("accepted") else ("unconverged (residual %s)" % f((r.get("unconverged") or {}).get("residual"), "%.2e"))
        t0s = r["T0"]
        L.append("| %d | %.4f | %s | %s | %s | %s | %.8f | %s of %s | %s | %s | %s; %s | %s |" % (
            r["nsub"], r["T_over_rs"], ", ".join("[%.2f, %.2f]" % (x, y) for x, y in r["continued"]) or "none", f(r["extent"], "%.3f"), f(r["pairs_shell_energy"]),
            f(r["pairs_interior"]), r["E_over_E0"], r["certificate_failures"], r["channels"], r["iterations"], stop, t0s.get("iterations"),
            (("converged at iteration %d" % t0s["accepted"]["iteration"]) if t0s.get("accepted") else ("converged" if t0s.get("converged") else "unconverged")), f(r["seconds_total"], "%.0f")))
    L += ["", "### The S9 (a) ratios per channel at the first re-solve: ||Delta h||_2/(nsub g_mix); ||Delta h||_2/(tau g_mix^2) (1/8 the criterion)", "",
          "| nsub | " + " | ".join("kappa %d" % (i + 1) for i in range(a.kappa_sea)) + " |", "|---" * (a.kappa_sea + 1) + "|"]
    for r in ok:
        L.append("| %d | " % r["nsub"] + " | ".join(("%s; %s" % (f(r["s9a"][kk]["ratio_step"], "%.3e"), f(r["s9a"][kk]["ratio_rate"], "%.3e"))) for kk in sorted(r["s9a"], key=int)) + " |")
    L += ["", "| nsub | " + " | ".join("g_mix kappa %d" % (i + 1) for i in range(a.kappa_sea)) + " |", "|---" * (a.kappa_sea + 1) + "|"]
    for r in ok:
        L.append("| %d | " % r["nsub"] + " | ".join(f(r["s9a"][kk]["g_mix"], "%.3e") for kk in sorted(r["s9a"], key=int)) + " |")
    L += ["", "- the smallest listed nsub with every channel's sub-step ratio below 1/8: %s; the estimate 8 nsub max(||Delta h||_2/(nsub g_mix)) at nsub %s: %s." % (
        (str(min(meets)) if meets else "none of %s" % ",".join(str(x) for x in nsubs)), first["nsub"] if first else "-", f(est, "%.1f"))]
    if verdict:
        L += ["", "- VERDICT (30.2's pre-stated reading, nsub %d to %d): the continued interval's total extent %s -> %s (ratio %s), the pairs' interior energy %s -> %s (ratio %s): %s." % (
            verdict["from_nsub"], verdict["to_nsub"], f(verdict["extent"][0], "%.3f"), f(verdict["extent"][1], "%.3f"), f(verdict["extent_ratio"], "%.3f"),
            f(verdict["pairs_interior"][0]), f(verdict["pairs_interior"][1]), f(verdict["pairs_interior_ratio"], "%.3f"),
            ("both fall by more than a factor 2: a resolution problem with a known cost (30.2: the relaunch at the smallest nsub meeting the 1/8 criterion on the first step, or the largest affordable one, labelled)"
             if verdict["falls_by_more_than_2_in_both"] else
             ("they do not both fall by more than a factor 2: 30.2's other branch (the member's statement is FILE 4; no further massless run)"
              + ("; no continued interval at either end" if (verdict["extent"][0] == 0.0 and verdict["extent"][1] == 0.0) else ""))))]
    else:
        L += ["", "- VERDICT: not formed (fewer than two completed nsub values)."]
    L += ["", "- every number above is printed by the script (%.0f s)." % (time.time() - t_all)]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
