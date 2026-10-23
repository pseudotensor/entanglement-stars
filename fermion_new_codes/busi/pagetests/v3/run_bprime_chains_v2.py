#!/usr/bin/env python
"""run_bprime_chains_v2.py (busi; the B' chain comparator as merged in round 13: ReviewA_pc13_v1 sec. 1.2 item 5, ReviewB_pc13_v1 sec. 1.1).
The paper's angular chains kappa = 1 .. --kmax (pt3_core.ChainOps = page_ops_v6.chain_sparse, m = 0; the chain operator itself, which shares the
cube's continuum spherical Dirac limit but not its zone), weights 16 kappa, on the SAME pc13 fields as run_bprime_v2 (F1 / F2 / A32, splice,
clock scan e^(-d f) on N and v, f = pt3b_cube.clock_f), the same port (reviewer A's sin^2 absorber beyond --onset L to --eta-max over 0.5 L, box
--box-L L), the same retained space (sites inside R_I; default the profile's outer dense root, --R-I-over-L to match the cube's averaged root),
the same inputs (interior-block eigenvectors in (0, T0) and (-T0, 0), T0 = 1/(4 pi L)), the same clocks (T/L = 1e2, 1e3 and ln 2 L/theta) and
the same readouts (pt3b_kernels.pc13_readouts, matrix-pencil widths at two sample lengths, collected_entropies of the declared --cin).
Fields on the chain are the profile's radial values at the sites (no back-average on the chain; labelled). Propagation exact: one dense
non-Hermitian matrix exponential per dt. Supersedes the --operator chains arm of run_bprime_v1.py (pc12 matched profile). Box CPU only."""
import argparse
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_kernels as KR  # noqa: E402


def one_channel(kappa, prof, L, a, Ts, T0):
    n = int(round(a.box_L * L))
    ops = K3.ChainOps(n, kappa)
    x = ops.R / L
    N = np.asarray(prof["N"](x), float); v = np.asarray(prof["v"](x), float)
    if a.d:
        s = np.exp(-a.d * CU.clock_f(x)); N = N * s; v = v * s
    h = ops.h(N, v)
    eta = np.repeat(CU.cap_reviewa(ops.R, L, a.onset, 0.5, a.eta_max), 2)
    R_I = a.R_I_over_L * L if a.R_I_over_L else max(prof["roots_over_L"]) * L
    imask = ops.site_mask(R_I)
    q = ops.h(np.zeros(ops.n), -np.ones(ops.n))
    es, Us, dI = KR.interior_window(h, imask, 0.0, T0)
    ee, Ue, _ = KR.interior_window(h, imask, -T0, 0.0)
    out = {"kappa": kappa, "n_sites": n, "R_I": R_I, "interior_dim": dI, "n_supply": len(es), "n_exit": len(ee),
           "e_over_T0": [float(z) for z in np.concatenate([es, ee]) / T0]}
    K = len(es) + len(ee)
    if K == 0:
        return out
    F = np.concatenate([Us, Ue], axis=1); e_all = np.concatenate([es, ee]); ns = len(es)
    qv = np.real(np.einsum("ij,ij->j", F.conj(), q @ F))
    Hn = h.toarray() - 1j * np.diag(eta)
    dt = a.dt_over_L * L
    U1 = expm(-1j * Hn * dt)
    Psi = F.copy(); t = 0.0; S = []; G_by_T = {}
    for T in Ts:
        while t < T - 1e-9:
            hh = min(dt, T - t)
            Psi = (U1 @ Psi) if abs(hh - dt) < 1e-12 else expm(-1j * Hn * hh) @ Psi
            t += hh
            if abs(hh - dt) < 1e-12:
                S.append(np.diag(F.conj().T @ Psi))
        G = np.eye(K) - Psi.conj().T @ Psi
        G_by_T[T] = 0.5 * (G + G.conj().T)
    S = np.array(S)
    widths = []
    for j in range(K):
        sig = S[:, j]
        if len(sig) < 16:
            widths.append(float("nan")); continue
        w1, c1, _ = KR.matrix_pencil(sig, dt, n_poles=min(6, len(sig) // 4))
        widths.append(float(-2 * w1[int(np.argmax(np.abs(c1)))].imag))
    out["supply"] = KR.pc13_readouts(es, {T: G[:ns, :ns] for T, G in G_by_T.items()}, T0, L, qv[:ns], widths=widths[:ns]) if ns else None
    out["exit"] = KR.pc13_readouts(ee, {T: G[ns:, ns:] for T, G in G_by_T.items()}, T0, L, qv[ns:], widths=widths[ns:]) if K > ns else None
    C_in = np.zeros((K, K), complex)
    if a.cin == "supply_filled":
        C_in[np.arange(ns), np.arange(ns)] = 1.0
    elif a.cin == "exit_hole":
        C_in[np.arange(ns, K - 1), np.arange(ns, K - 1)] = 1.0
    out["collected_entropies"] = KR.collected_entropies(sorted(G_by_T.items()), C_in) if np.any(C_in) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["F1", "F2", "A32"], default="F2")
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--splice", choices=["none", "reviewb", "reviewb_out"], default="none")
    ap.add_argument("--d", type=float, default=0.0)
    ap.add_argument("--box-L", type=float, default=2.5)
    ap.add_argument("--onset", type=float, default=2.0)
    ap.add_argument("--eta-max", type=float, default=1.0)
    ap.add_argument("--R-I-over-L", type=float, default=0.0)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--T-max-over-L", type=float, default=1000.0)
    ap.add_argument("--dt-over-L", type=float, default=0.25)
    ap.add_argument("--cin", default="supply_filled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    L = a.L; T0 = 1.0 / (4 * math.pi * L)
    prof = {"F1": lambda: CU.dense_record_profile(*CU.REC_F1), "F2": CU.f2_profile, "A32": CU.a32_profile}[a.family]()
    prof = CU.splice(prof, a.splice)
    thetas = [1e-1, 1e-2, 1e-3, 1e-4]
    Ts = sorted(set([x * L for x in (1e2, 1e3)] + [math.log(2) * L / th for th in thetas]))
    Ts = [T for T in Ts if T <= a.T_max_over_L * L + 1e-9]
    res = {"script": "run_bprime_chains_v2.py", "args": vars(a), "profile": prof["label"], "L": L, "T0": T0, "clocks_over_L": [T / L for T in Ts],
           "own_md5": K3.own_md5s(), "channels": {}}
    for k in range(1, a.kmax + 1):
        res["channels"][str(k)] = one_channel(k, prof, L, a, Ts, T0)
        c = res["channels"][str(k)]
        K3.log("kappa %d: supply %d, exit %d (%.0f s)" % (k, c["n_supply"], c["n_exit"], time.time() - t0))
        K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    tot = {}
    for lab in ("supply", "exit"):
        chans = [(int(k), c[lab]) for k, c in res["channels"].items() if c.get(lab)]
        tot[lab] = {"N_w_weighted": float(sum(16 * k * r_["N_w"] for k, r_ in chans)),
                    "N_acc_time_weighted": {th: float(sum(16 * k * r_["N_acc_time"].get(th, 0) for k, r_ in chans)) for th in ("0.1", "0.01", "0.001", "0.0001")},
                    "N_acc_poles_weighted": {th: float(sum(16 * k * r_["N_acc_poles"].get(th, 0) for k, r_ in chans)) for th in ("0.1", "0.01", "0.001", "0.0001")}}
    res["weighted_totals"] = tot
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s (%.0f s): weighted %s" % (a.tag, res["seconds"], {k: v["N_w_weighted"] for k, v in tot.items()}))


if __name__ == "__main__":
    main()
