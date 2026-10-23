#!/usr/bin/env python
"""s7_halfzone_check_v1.py (builder analyst; engine spec 36.3 (2); the brief lead/BRIEF_s7_v3_builder.md): the check of
engine_s7_twocopy_v3's half-zone split before its timing run, on version 2's stored end states (S7/s7_twocopy_v2_tau8_states.npz, the
tau 8 member at its last record, T/rs 40, both configurations). Read only; nothing it prints enters a deliverable as a reading.
(1) G rebuilt from the stored copy-1 blocks A, B and the occupations f as version 2 builds it (G = B f A^dagger - A f B^dagger); its
anti-Hermiticity; the exact bound beyond R_t(T0) (version 2's fixed site) against version 2's stored bound_Rt0 at that record (the stored
arrays are contiguous copies of the run's views, so equality to rounding, not bitwise, is the expectation).
(2) halfzone() on that region (with the cross bound) and on the whole chain: the bounds, the shares, the moments, Parseval, the compression check.
(3) The same half-zone bounds by explicit dense projectors (numpy: F built element by element in the fftfreq order, Pi_z = F_z^dagger F_z
(x) 1_2, the bound 2 max |eig(i Pi_z G_D Pi_z)|) against the FFT route; Pi_lo + Pi_hi = 1 and Pi_lo Pi_hi = 0.
(4) Plane waves on the DFT grid in spinor component 0 (a p = 2 pi q0/M for q0 below and above M/4, and q0 = M/4 when 4 divides M):
G = i x x^dagger on the region; the bound 2 in the expected half-zone and 0 in the other, the weight share 1, E[cos a p] against
(M - 1)/M cos(a p).
(5) The split's wall time per configuration: beyond R_t(T0) with the cross bound, and the whole chain with its unsplit bound."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "flow"))
import engine_s7_twocopy_v3 as V3  # noqa: E402

C5 = V3.C5


def dense_bound(Gd, F, z):
    """2 max |eig(i Pi G_D Pi)| with Pi = F_z^dagger F_z (x) 1_2 built densely."""
    Fz = F[z]
    Pi = np.kron(Fz.conj().T @ Fz, np.eye(2))
    S = Pi @ Gd @ Pi
    return 2.0 * float(np.abs(np.linalg.eigvalsh(1j * S)).max()), Pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="version 2's states npz (end_cfg*_A, _B, _f, cfg*_params)")
    ap.add_argument("--v2-json", required=True, help="version 2's member json (the stored bound_Rt0 at the last record)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    C5.backend(None)
    torch = C5.BK["torch"]
    torch.set_num_threads(a.threads)
    J = json.load(open(a.v2_json))
    h = J["head"]; last = J["s7"][-1]
    n = int(h["nsites"])
    R = np.arange(n, dtype=float) + 1.0
    i0 = V3.site_beyond(R, h["trapping_zeros_at_T0"]["R_t"])
    print("version 2's member %s, last record T/rs %.3f; R_t(T0) %.4f, the first site beyond it R %.0f (version 2's R_t0_site %s); n %d, M %d" % (
        h["tag"], last["T_over_rs"], h["trapping_zeros_at_T0"]["R_t"], R[i0], h["R_t0_site"], n, n - i0), flush=True)
    Z = np.load(a.npz)
    out = {"v2_json": a.v2_json, "npz": a.npz, "T_over_rs": last["T_over_rs"], "i0": i0, "R_first": float(R[i0]), "configs": []}
    ok_all = True
    M = n - i0
    qs, lo, hi, quarters, nb = V3.zone_masks(M)
    F = np.exp(-2j * np.pi * np.outer(qs, np.arange(M)) / M) / math.sqrt(M)
    for i_ in range(2):
        A = Z["end_cfg%d_A" % i_]; B = Z["end_cfg%d_B" % i_]; f = Z["end_cfg%d_f" % i_]; prm = Z["cfg%d_params" % i_]
        At = C5.T_(A); Bt = C5.T_(B); ft = torch.as_tensor(np.asarray(f, float), dtype=torch.complex128)
        G = (Bt * ft[None, :]) @ At.conj().T - (At * ft[None, :]) @ Bt.conj().T
        anti = float((G + G.conj().T).abs().max())
        b0 = V3.opnorm_region(G, i0)
        stored = last["configs"][i_]["bound_Rt0"]
        t = time.time()
        hz = V3.halfzone(G, i0, cross=True, unsplit=b0)
        t_hz = time.time() - t
        t = time.time()
        b_all = V3.opnorm_region(G, 0)
        hza = V3.halfzone(G, 0, cross=False, unsplit=b_all)
        t_all = time.time() - t
        Gd = G[2 * i0:, 2 * i0:].cpu().numpy()
        d_lo, P_lo = dense_bound(Gd, F, lo)
        d_hi, P_hi = dense_bound(Gd, F, hi)
        comp = float(np.abs(P_lo + P_hi - np.eye(2 * M)).max()); orth = float(np.abs(P_lo @ P_hi).max())
        rel_lo = abs(d_lo - hz["bound_lo"]) / max(d_lo, 1e-300); rel_hi = abs(d_hi - hz["bound_hi"]) / max(d_hi, 1e-300)
        shares_sum = sum(hz["quarter_shares"])
        ok = (rel_lo < 1e-10 and rel_hi < 1e-10 and comp < 1e-12 and orth < 1e-12 and hz["parseval"] < 1e-12 and abs(shares_sum - 1) < 1e-12
              and hz["check_lower"] <= 1e-15 and hz["check_upper"] <= 1e-15 and abs(b0 - stored) / stored < 1e-10)
        ok_all = ok_all and ok
        row = {"kappa": float(prm[0]), "X": float(prm[1]), "theta": float(prm[2]), "anti_hermiticity": anti, "bound_Rt0": b0, "stored_bound_Rt0": stored,
               "bound_Rt0_rel_diff": abs(b0 - stored) / stored, "bound_Rt0_bitwise": b0 == stored, "halfzone_Rt0": hz, "dense_lo": d_lo, "dense_hi": d_hi,
               "dense_rel_lo": rel_lo, "dense_rel_hi": rel_hi, "Pi_lo_plus_Pi_hi_minus_1": comp, "Pi_lo_Pi_hi": orth, "quarter_shares_sum": shares_sum,
               "bound_chain": b_all, "halfzone_chain": hza, "seconds_Rt0_with_cross": t_hz, "seconds_chain_with_unsplit": t_all, "ok": ok}
        out["configs"].append(row)
        print("kappa %d (theta %.6f): |G + G^dagger| %.1e; the bound beyond R_t(T0) %.6e against version 2's stored %.6e (rel. %.1e, bitwise %s)" % (
            int(prm[0]), float(prm[2]), anti, b0, stored, row["bound_Rt0_rel_diff"], row["bound_Rt0_bitwise"]), flush=True)
        print("  split (FFT): lo %.6e hi %.6e cross %.6e; dense projectors: lo %.6e hi %.6e (rel. %.1e, %.1e); |Pi_lo + Pi_hi - 1| %.1e, |Pi_lo Pi_hi| %.1e" % (
            hz["bound_lo"], hz["bound_hi"], hz["bound_cross"], d_lo, d_hi, rel_lo, rel_hi, comp, orth), flush=True)
        print("  weight lo %.4f hi %.4f; quarters %s (sum %.15f); middle %.4f; E[cos ap] %+.4f E[cos 2ap] %+.4f; Parseval %.1e; checks %+.1e %+.1e; M %d, at pi/2 %d; %.2f s" % (
            hz["share_lo"], hz["share_hi"], ", ".join("%.4f" % x for x in hz["quarter_shares"]), shares_sum, hz["share_middle"], hz["cos_ap"], hz["cos_2ap"],
            hz["parseval"], hz["check_lower"], hz["check_upper"], hz["M"], hz["boundary_momenta"], t_hz), flush=True)
        print("  the whole chain: 2||G|| %.6e; lo %.6e hi %.6e; weight lo %.4f hi %.4f; E[cos ap] %+.4f; check %+.1e; M %d, at pi/2 %d; %.2f s" % (
            b_all, hza["bound_lo"], hza["bound_hi"], hza["share_lo"], hza["share_hi"], hza["cos_ap"], hza["check_lower"], hza["M"], hza["boundary_momenta"], t_all), flush=True)
        print("  verdict: %s" % ("ok" if ok else "FAILED"), flush=True)
        del G, At, Bt
    # plane waves on the grid
    waves = []
    for M_ in sorted({M, M + 2}):                                         # M + 2 = 1000 here: 4 divides it, so the boundary momentum is tested
        i0_ = n - M_
        qs_, lo_, hi_, qu_, nb_ = V3.zone_masks(M_)
        for q0 in [int(round(0.15 * M_)), int(round(0.40 * M_))] + ([M_ // 4] if M_ % 4 == 0 else []):
            x = np.zeros((M_, 2), complex)
            x[:, 0] = np.exp(2j * np.pi * q0 * np.arange(M_) / M_) / math.sqrt(M_)
            xs = x.reshape(-1)
            Gs = np.zeros((2 * n, 2 * n), complex)
            Gs[2 * i0_:, 2 * i0_:] = 1j * np.outer(xs, xs.conj())
            Gt = C5.T_(Gs)
            bs = V3.opnorm_region(Gt, i0_)
            z = V3.halfzone(Gt, i0_, cross=True, unsplit=bs)
            pz = 2 * math.pi * q0 / M_
            exp_hi = 4 * q0 >= M_
            good = (abs(bs - 2.0) < 1e-12 and abs((z["bound_hi"] if exp_hi else z["bound_lo"]) - 2.0) < 1e-12 and (z["bound_lo"] if exp_hi else z["bound_hi"]) < 1e-12
                    and abs((z["share_hi"] if exp_hi else z["share_lo"]) - 1.0) < 1e-12 and abs(z["cos_ap"] - (M_ - 1) / M_ * math.cos(pz)) < 1e-12)
            ok_all = ok_all and good
            waves.append({"M": M_, "q0": q0, "a_p_over_pi": pz / math.pi, "expected": "upper" if exp_hi else "lower", "bound": bs, "lo": z["bound_lo"], "hi": z["bound_hi"],
                          "cross": z["bound_cross"], "share_lo": z["share_lo"], "share_hi": z["share_hi"], "cos_ap": z["cos_ap"], "cos_ap_expected": (M_ - 1) / M_ * math.cos(pz),
                          "boundary_momenta": z["boundary_momenta"], "ok": good})
            print("plane wave M %d q0 %d (a p = %.4f pi, expected the %s half-zone): unsplit %.15f; lo %.3e hi %.3e cross %.1e; weight lo %.15f hi %.15f; E[cos ap] %+.12f (expected %+.12f); at pi/2 %d: %s" % (
                M_, q0, pz / math.pi, "upper" if exp_hi else "lower", bs, z["bound_lo"], z["bound_hi"], z["bound_cross"], z["share_lo"], z["share_hi"], z["cos_ap"],
                (M_ - 1) / M_ * math.cos(pz), z["boundary_momenta"], "ok" if good else "FAILED"), flush=True)
    out["plane_waves"] = waves
    out["ok"] = ok_all
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("the check: %s" % ("PASS" if ok_all else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
