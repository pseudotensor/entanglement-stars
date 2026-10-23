#!/usr/bin/env python
"""compare_capture_v10_gate_v1.py (builder analyst; engine spec 38.5 (3): "the gate a 40 r_s run of the minimum agreeing with version 9's
readings within 1e-3 of sigma at every reading, the absorbed weight printed (expected below the 0.01 threshold before 40)"): per background
and probe, at every reading time of the version 9 run, sigma of version 10 against version 9's, |sigma_10 - sigma_9|/sigma_9, the largest
over all readings against --tol (1e-3): PASS or FAIL; beside it, at every reading, the outer port's absorbed weight summed as sigma is over
pi b_c(omega, m)^2 (the run's own norm), its largest value against --threshold (0.01, the chain-end reading aid's threshold), and, as
information, the largest differences of the other summed columns (in transit, inside, beyond R_max, the inner port), the per-channel A,
the channel counts and the settle decisions. Every number printed from the two JSONs. Args: --v9 --v10 [--tol 1e-3] [--threshold 0.01]."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", required=True); ap.add_argument("--v10", required=True)
    ap.add_argument("--tol", type=float, default=1e-3); ap.add_argument("--threshold", type=float, default=0.01)
    a = ap.parse_args()
    A9 = json.load(open(a.v9)); A10 = json.load(open(a.v10))
    print("version 9: %s (%s, slice %s, start %s UTC); version 10: %s (%s, slice %s, start %s UTC); the outer port: %s"
          % (a.v9, A9["script"], A9["slice"], A9["start_utc"], a.v10, A10["script"], A10["slice"], A10["start_utc"], A10["outer_port"]["text"]))
    assert A9["slice"] == A10["slice"] and A9["E"] == A10["E"], "different slices"
    worst = 0.0; worst_at = None; n = 0; absmax = 0.0; absmax_at = None
    info = {"inward_transit_summed": 0.0, "inward_inside_summed": 0.0, "beyond_Rmax_summed": 0.0, "sigma_absorber": 0.0}
    amax = 0.0; chan_diff = []; settle_diff = []
    for r9, r10 in zip(A9["results"], A10["results"]):
        assert (r9["background"], r9["probe"]) == (r10["background"], r10["probe"])
        nrm = r10["norm_ratio"]
        print("\n%s | %s (pi b_c(omega, m)^2 %.6e)" % (r10["background"], r10["probe"], nrm))
        print("  T/r_s | sigma v9 | sigma v10 | relative difference | outer port absorbed, summed as sigma, over pi b_c(omega, m)^2")
        for T in r9["T_list"]:
            k = "%.4f" % T
            if k not in r10["sigma"]:
                print("  %5.1f | %.6e | (no reading) |" % (T / A9["r_s"], r9["sigma"][k])); worst = float("inf"); continue
            s9, s10 = r9["sigma"][k], r10["sigma"][k]
            d = abs(s10 - s9) / s9 if s9 != 0 else (0.0 if s10 == 0 else float("inf"))
            ab = r10["outer_absorbed_summed"].get(k, 0.0) / nrm if r10.get("outer_absorbed_summed") else 0.0
            n += 1
            if d > worst:
                worst, worst_at = d, (r10["background"][:12], r10["probe"], T / A9["r_s"])
            if ab > absmax:
                absmax, absmax_at = ab, (r10["background"][:12], r10["probe"], T / A9["r_s"])
            for f in info:
                x9 = (r9.get(f) or {}).get(k); x10 = (r10.get(f) or {}).get(k)
                if x9 is not None and x10 is not None:
                    info[f] = max(info[f], abs(x10 - x9) / nrm)
            print("  %5.1f | %.6e | %.6e | %.3e | %.3e" % (T / A9["r_s"], s9, s10, d, ab))
        c10 = {c["kappa"]: c for c in r10["channels"]}
        if len(r9["channels"]) != len(r10["channels"]):
            chan_diff.append((r10["background"][:12], r10["probe"], len(r9["channels"]), len(r10["channels"])))
        for c in r9["channels"]:
            cc = c10.get(c["kappa"])
            if cc is None:
                continue
            for s_ in ("plus", "minus"):
                for k, x in (c[s_].get("A") or {}).items():
                    y = (cc[s_].get("A") or {}).get(k)
                    if y is not None:
                        amax = max(amax, abs(y - x))
        if bool(r9["settle"]["settled"]) != bool(r10["settle"]["settled"]) or r9["settle"].get("T_over_rs") != r10["settle"].get("T_over_rs"):
            settle_diff.append((r10["background"][:12], r10["probe"], r9["settle"].get("T_over_rs"), r10["settle"].get("T_over_rs")))
    print("\nreadings compared: %d; the largest |sigma_10 - sigma_9|/sigma_9: %.3e (%s) against %.0e" % (n, worst, worst_at, a.tol))
    print("the outer port's absorbed weight, summed as sigma, over pi b_c(omega, m)^2: largest %.3e (%s) against %g: %s"
          % (absmax, absmax_at, a.threshold, "below" if absmax < a.threshold else "NOT below"))
    print("information: the largest differences over pi b_c(omega, m)^2: " + ", ".join("%s %.3e" % (f, x) for f, x in info.items())
          + "; the per-channel A: %.3e; channel counts differing: %s; settle decisions differing: %s" % (amax, chan_diff or "none", settle_diff or "none"))
    ok = worst < a.tol and not chan_diff and not settle_diff
    print("gate (engine 38.5 (3)): every reading within %.0e of sigma: %s; the absorbed weight %s %g before 40 r_s: %s"
          % (a.tol, "PASS" if ok else "FAIL", "below" if absmax < a.threshold else "not below", a.threshold, "as expected" if absmax < a.threshold else "printed, not expected"))


if __name__ == "__main__":
    main()
