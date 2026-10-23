#!/usr/bin/env python
"""m325_repro_v1.py (busi reviewer A, 20 Oct 2026; marks325 third-completion ruling): three reproductions on the stored W3 files, box CPU.
(a) the pair energy of each history at 12.125 to 12.75 localised inside the point-sampled core (the printed value is eps_pairs.sum());
(b) point-core: the kernel-averaged fields the paired form would apply at the record's own dense solution (engine_fields_v12.site_fields_kernel_average
    on parity splines of the stored 4 a nodes) against the point-sampled ones at the core sites;
(c) the core gate G = ell max(|v'/N|, |v/(r N)|) from the stored nodes for all four histories (validated on the point-core's printed values)."""
import sys, json, numpy as np
sys.path.insert(0, "<internal>/code")
import engine_fields_v12 as F12
from scipy.interpolate import CubicSpline
B = "<internal>/runs"
P = {"own": "<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40",
     "paired-avg": "<internal>/collapse_w3_paired-avg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g",
     "frozen-sources": B + "w3_frozen-sources/collapse_w3_frozen-sources_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g",
     "point-core": B + "w3_point-core/collapse_w3_point-core_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g"}
ell = 4.0


def spl(rn, yn, parity):
    rn = np.asarray(rn, float); yn = np.asarray(yn, float)
    rm = np.concatenate([-rn[:0:-1], rn]); ym = np.concatenate([parity * yn[:0:-1], yn])
    return CubicSpline(rm, ym)


print("=== (a) pair energy (the printed value is eps_pairs.sum()): total, R<=16 a, 16<R<=24, R>24, the share inside 16 a")
for name, p in P.items():
    F = np.load(p + "_fields.npz", allow_pickle=True); J = json.load(open(p + ".json")); R = F["R"]; T = F["T"] / 64.0
    for t in (12.125, 12.25, 12.375, 12.5, 12.625, 12.75):
        i = np.where(np.isclose(T, t))[0]
        if not len(i):
            continue
        i = i[0]; ep = F["eps_pairs"][i]; rec = J["records"][i]
        assert abs(rec["T"] / 64.0 - t) < 1e-9
        tot = ep.sum(); a_ = ep[R <= 16].sum(); b_ = ep[(R > 16) & (R <= 24)].sum(); c_ = ep[R > 24].sum()
        print("  %-14s T/rs %.3f printed %+.4e, sum %+.4e, <=16: %+.4e, 16-24: %+.4e, >24: %+.4e, share<=16 %.2f" % (name, t, rec["E_pairs_shells"], tot, a_, b_, c_, a_ / tot))
print("=== (b) point-core: the kernel-averaged fields of the record's own dense solution against the point-sampled ones at the core sites")
F = np.load(P["point-core"] + "_fields.npz", allow_pickle=True); R = F["R"]; T = F["T"] / 64.0
for t in (11.25, 12.125, 12.5):
    i = np.where(np.isclose(T, t))[0][0]
    rn = np.array(F["r_nodes"][i], float); Nn = np.array(F["N_nodes"][i], float); vn = np.array(F["v_nodes"][i], float)
    sN = spl(rn, Nn, +1); sv = spl(rn, vn, -1)
    Na, va, norms = F12.site_fields_kernel_average({"N_dense": sN, "v_dense": sv}, R, ell)
    Np = F["N_applied"][i]; vp = F["v_applied"][i]
    print("  T/rs %.3f  K0 norms %.6f..%.6f ; spline at the nodes 4, 8, 12 a minus N_applied there: %s" % (t, norms.min(), norms.max(), np.round([sN(4.0) - Np[3], sN(8.0) - Np[7], sN(12.0) - Np[11]], 6)))
    print("   R    N_point   N_avg   rel     v_point   v_avg   rel")
    for j in (0, 1, 2, 3, 5, 7, 11, 15, 19, 23, 31):
        print("   %4.0f  %.5f  %.5f  %+.3f   %.5f  %.5f  %+.3f" % (R[j], Np[j], Na[j], (Np[j] - Na[j]) / Na[j], vp[j], va[j], (vp[j] - va[j]) / abs(va[j]) if va[j] != 0 else float("nan")))
    m16 = R <= 16
    print("   max rel |N_point-N_avg| over R<=16: %.3f ; max |v_point-v_avg| over R<=16: %.4f (max |v| %.3f)" % (np.max(np.abs((Np - Na) / Na)[m16]), np.max(np.abs(vp - va)[m16]), np.max(np.abs(va))))
print("=== (c) the core gate G = ell max(|v'/N|, |v/(r N)|) from the stored 4 a nodes (parity splines), r in [0.25, 48] step 0.25: max and its r, N(0)")
rr = np.arange(0.25, 48.01, 0.25)
for name in ("point-core", "own", "paired-avg", "frozen-sources"):
    F = np.load(P[name] + "_fields.npz", allow_pickle=True); T = F["T"] / 64.0
    out = []
    for t in (12.0, 12.125, 12.25, 12.375, 12.5, 12.625, 12.75, 12.875, 13.0, 13.25, 13.5, 14.0, 15.0, 16.0):
        i = np.where(np.isclose(T, t))[0]
        if not len(i):
            continue
        i = i[0]; rn = np.array(F["r_nodes"][i], float); Nn = np.array(F["N_nodes"][i], float); vn = np.array(F["v_nodes"][i], float)
        sN = spl(rn, Nn, +1); sv = spl(rn, vn, -1)
        G = ell * np.maximum(np.abs(sv(rr, 1) / sN(rr)), np.abs(sv(rr) / (rr * sN(rr))))
        k = G.argmax(); out.append("%.3f: %.3f at %.2f a, N(0) %.3f" % (t, G[k], rr[k], sN(0.0)))
    print("  %-14s" % name)
    for o in out:
        print("     " + o)
