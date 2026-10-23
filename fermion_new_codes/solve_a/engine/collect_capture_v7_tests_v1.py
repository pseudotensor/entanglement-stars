"""collect_capture_v7_tests_v1.py (builder analyst): every number the C-capture v7 builder report quotes from the pod 1 tests in
CAP7TEST/ (the re-solve's E at 1 and 8 BLAS threads against version 6's; the timing runs' per-step lines; the serial/workers comparison)."""
import json

d6 = json.load(open("CAP/capture_v6_prod19_last.json"))
d1 = json.load(open("CAP7TEST/test1_regress_last.json")); d1b = json.load(open("CAP7TEST/test1b_regress_last_t8.json"))
print("E: version 6 (runner, 8 BLAS threads) %.17g; v7 at 8 threads %.17g (equal %s); v7 at 1 thread %.17g; relative shift %.3e"
      % (d6["E"], d1b["E"], d6["E"] == d1b["E"], d1["E"], abs(d1["E"] - d6["E"]) / d6["E"]))
for tag in ("test3_steptime_w1", "test3_steptime_w8"):
    d = json.load(open("CAP7TEST/%s.json" % tag))
    print("== %s: --workers %d, the run's length %.0f s (%s to %s UTC); slice %d, E %.6f, r_s %.4f, R_t %.4f"
          % (tag, d["args"]["workers"], d["run_length_s"], d["start_utc"], d["end_utc"], d["slice"], d["E"], d["r_s"], d["R_t"]))
    for r in d["results"]:
        nrm = r["norm_ratio"]
        for s, T in zip(r["steps"], r["T_list"]):
            k = "%.4f" % T
            print("  %s | %s: T %g r_s: step %.1f s, channels %d, sigma/(pi b_c(omega, m)^2) %.4f, inward-branch weight inside R_t + 2 %.4f, plain inside %.4f, in transit %.4f, beyond R_max %.4f (each over pi b_c(omega, m)^2)"
                  % (r["background"][:12], r["probe"], s["T_over_rs"], s["step_s"], s["channels"], r["sigma"][k] / nrm, r["inward_inside_summed"][k] / nrm,
                     r["plain_inside_summed"][k] / nrm, r["inward_transit_summed"][k] / nrm, r["beyond_Rmax_summed"][k] / nrm))
        bounds = [c[sg]["bound"] for c in r["channels"] for sg in ("plus", "minus") if "bound" in c[sg]]
        errs = sum(1 for c in r["channels"] for sg in ("plus", "minus") if "error" in c[sg])
        print("  %s | %s: Chebyshev bound of h_m over the channels %.3f to %.3f; evanescent (error) channels %d" % (r["background"][:12], r["probe"], min(bounds), max(bounds), errs))
w1 = json.load(open("CAP7TEST/test3_steptime_w1.json")); w8 = json.load(open("CAP7TEST/test3_steptime_w8.json"))
for r1 in w1["results"]:
    r8 = [r for r in w8["results"] if r["probe"] == r1["probe"]][0]
    for s1, s8 in zip(r1["steps"], r8["steps"]):
        print("step time ratio serial/workers 8 | %s: T %g r_s: %.2f" % (r1["probe"], s1["T_over_rs"], s1["step_s"] / s8["step_s"]))
