#!/usr/bin/env python
"""run_bprime_v3.py (busi; = run_bprime_v2.py plus the raw arrays saved to <tag>.npz (the escape Gram G(T) per clock, the return
samples S(t) (diagonal at every sample, the full matrix at every 16th: the file stays small), the inputs' energies and q_r), so the early-late blocks, the retained-output modes and any re-read need no re-propagation;
run_bprime_v2.py is kept unchanged. Original header: B' as merged in round 13: ReviewA_pc13_v1 sec. 1 and ReviewB_pc13_v1 sec. 1; one size, family, clock and
interface per call; where the two specifications differ, both readouts). Supersedes run_bprime_v1.py (built on the pc12 interim specs).

Fields: --family F1 (reviewer A: the sigma/a 4, r_s/a 64 reference collapse P64_ref_c19r11 at T/r_s 30, dense re-solve) | F2 (reviewer A: the frozen dust
ball) | A32 (reviewer B: the set17 r_s/a 32 record at T/r_s 18); sizes s = L/a with L = 2 g E the declared exterior Schwarzschild length; --splice
none (reviewer A: the record's exterior) | reviewb (preserve to 1.5 L, splice to vacuum over (1.5, 2) L) | reviewb_out ((2, 2.5) L, the moved-splice
control); back-average W_ell^dag on the cube at --ell-over-L (reviewer A 1/16 with sigma/a >= 2; reviewer B/8, controls 1/16, 1/4); clock scan
(N, v) x e^(-d f), f = 1 below 0.85 L and 0 above L, --d in {0, 1/2, 1, 2}, the averaged root recomputed.
Port: reviewer A's absorber beyond --onset L rising (sin^2) to --eta-max over 0.5 L (controls: eta_max x 2, onset 2.25); the collected output is the
absorber's exact Markovian dilation (bath modes orthogonal across time bins). reviewer B's unitary-exterior requirement is met only through that
dilation here (sec. 4 of the design note names the gap).
Retained space: --retained root (reviewer A: inside the averaged root) | interface:<r/L> (reviewer B: an interface outside the near-root region, moved by
the scan). Inputs: the retained block's eigenvectors in the supply window (0, T0) (and (0, 3 T0) with --wide) and the exit window (-T0, 0),
T0 = 1/(4 pi L), structural zero modes excluded and counted; supply and exit inputs propagated together (their cross-correlations kept).
Clocks: T/L = 1e2, 1e3 (reviewer B) and T_theta = ln 2 L/theta for theta = 1e-1 .. 1e-4 and theta(chi) = <aq> chi/60.8 (reviewer A), capped by --T-max-over-L.
Per clock: the escape Gram G(T) = 1 - Psi^dag Psi over all inputs (reviewer B's K^dag K, its whole spectrum per window and jointly), the retained
weights, the return matrix S = F^dag Psi; S sampled every --dt-sample for the matrix-pencil poles (reviewer A's widths Gamma_j with errors from
two sample lengths); collected-output entropies (pt3b_kernels.collected_entropies) for the declared input covariance --cin (supply_filled:
the supply inputs occupied, the exit inputs empty, the trapped-supply hypothesis, labelled; exit_hole: the exit inputs occupied but the highest).
Device: cpu, cuda:1 or cuda:2 (the B' launch itself is held for the models' check)."""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K3  # noqa: E402
import pt3b_cube as CU  # noqa: E402
import pt3b_kernels as KR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["F1", "F2", "A32"], default="F1")
    ap.add_argument("--L", type=float, required=True, help="s = L/a")
    ap.add_argument("--splice", choices=["none", "reviewb", "reviewb_out"], default="none")
    ap.add_argument("--ell-over-L", type=float, default=1.0 / 8)
    ap.add_argument("--d", type=float, default=0.0)
    ap.add_argument("--box-L", type=float, default=2.5)
    ap.add_argument("--onset", type=float, default=2.0)
    ap.add_argument("--eta-max", type=float, default=1.0)
    ap.add_argument("--retained", default="root")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--T-max-over-L", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--dt-sample", type=float, default=8.0)
    ap.add_argument("--cin", default="supply_filled")
    ap.add_argument("--max-inputs", type=int, default=0)
    ap.add_argument("--dense-max", type=int, default=6000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    torch.set_num_threads(a.threads)
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    L = a.L; H = int(round(a.box_L * L)); T0 = 1.0 / (4 * math.pi * L)
    prof = {"F1": lambda: CU.dense_record_profile(*CU.REC_F1), "F2": CU.f2_profile, "A32": CU.a32_profile}[a.family]()
    prof = CU.splice(prof, a.splice)
    N, v, r, _ = CU.cube_fields(prof, L, H, a.d, a.ell_over_L)
    roots = CU.averaged_root(N, v, r, H)
    R_root = max(roots["axis"]) if roots["axis"] else max(prof["roots_over_L"]) * L
    R_I = R_root if a.retained == "root" else float(a.retained.split(":")[1]) * L
    eta = CU.cap_reviewa(r, L, a.onset, 0.5, a.eta_max)
    res = {"script": "run_bprime_v3.py", "args": vars(a), "profile": prof["label"], "L": L, "T0": T0, "averaged_roots": roots, "R_root": R_root, "R_I": R_I,
           "box": {"H": H, "amplitudes": 4 * (2 * H + 1) ** 3}, "kernel_identities": CU.kernel_identities(L, min(H, 24), a.ell_over_L), "own_md5": K3.own_md5s()}
    K3.log("B' v2 %s L %g splice %s ell/L %.4g d %g: averaged roots %s, R_I %.3f (%s), T0 %.4e" % (a.family, L, a.splice, a.ell_over_L, a.d, roots, R_I, a.retained, T0))
    Hs = int(math.ceil(R_I)) + 2
    off = H - Hs; sl = slice(off, off + 2 * Hs + 1)
    hs = CU.cubic_csr(N[sl, sl, sl], [c[sl, sl, sl] for c in v])
    ims = CU.masks(r[sl, sl, sl], R_I)
    q = CU.q_r_csr(N[sl, sl, sl], None, r[sl, sl, sl])
    zi = {}
    hi_s = 3 * T0 if a.wide else T0
    es, Us, dI = KR.interior_window(hs, ims, 0.0, hi_s, dense_max=a.dense_max, info=zi)
    ee, Ue, _ = KR.interior_window(hs, ims, -T0, 0.0, dense_max=a.dense_max, info=zi)
    if a.max_inputs:
        es, Us, ee, Ue = es[: a.max_inputs], Us[:, : a.max_inputs], ee[-a.max_inputs:], Ue[:, -a.max_inputs:]
    e_all = np.concatenate([es, ee]); U_all = np.concatenate([Us, Ue], axis=1)
    ns, ne = len(es), len(ee); K = ns + ne
    qv = np.real(np.einsum("ij,ij->j", U_all.conj(), q @ U_all))
    res["inputs"] = {"interior_dim": dI, "n_supply": ns, "n_exit": ne, "structural_zero_modes": zi.get("n_zero", 0), "e_over_T0": [float(x) for x in e_all / T0],
                     "q_r": [float(x) for x in qv]}
    K3.log("inputs: supply %d, exit %d (zero modes excluded %d), retained dim %d" % (ns, ne, zi.get("n_zero", 0), dI))
    aq = float(np.mean(np.abs(qv[:ns]))) if ns else (float(np.mean(np.abs(qv))) if K else 0.0)
    thetas = [1e-1, 1e-2, 1e-3, 1e-4] + ([aq / L / 60.8] if aq > 0 else [])
    Ts = sorted(set([x * L for x in (1e2, 1e3)] + [math.log(2) * L / th for th in thetas]))
    Ts = [T for T in Ts if T <= a.T_max_over_L * L + 1e-9]
    res["clocks_over_L"] = [T / L for T in Ts]; res["theta_chi"] = thetas[-1] if aq > 0 else None
    if K == 0:
        res["note"] = "no window inputs"
        K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
        return
    Ls = 2 * Hs + 1; Lf = 2 * H + 1
    F = np.zeros((K, 4, Lf, Lf, Lf), complex)
    F[:, :, sl, sl, sl] = U_all.T.reshape(K, 4, Ls, Ls, Ls)
    cube = CU.Cube(N, v, eta, a.device)
    X = torch.as_tensor(F, dtype=CU.CD, device=cube.dev); X0 = X.clone()
    imf = torch.as_tensor(np.broadcast_to((r < R_I)[None], (4, Lf, Lf, Lf)).copy(), device=cube.dev)
    t = 0.0; S_samples = []; G_by_T = {}; ret_by_T = {}
    t_next_s = 0.0
    for T in Ts:
        while t < T - 1e-9:
            hh = min(a.dt, T - t)
            X = cube.step_cap(X, hh); t += hh
            if t >= t_next_s - 1e-9:
                S_samples.append((t, (X0.reshape(K, -1).conj() @ X.reshape(K, -1).T).cpu().numpy()))
                t_next_s += a.dt_sample
        Xf = X.reshape(K, -1)
        G = np.eye(K) - (Xf.conj() @ Xf.T).cpu().numpy()
        G_by_T[T] = 0.5 * (G + G.conj().T)
        ret_by_T[T] = (X.abs() ** 2 * imf).reshape(K, -1).sum(dim=1).cpu().numpy()
        K3.log("  T/L %.4g: Tr K^dag K %.4f (supply %.4f, exit %.4f), retained mean %.4f (%.0f s)" % (
            T / L, float(np.real(np.trace(G))), float(np.real(np.trace(G[:ns, :ns]))), float(np.real(np.trace(G[ns:, ns:]))), float(np.mean(ret_by_T[T])), time.time() - t0))
        K3.dump_json({"partial_T_over_L": T / L}, os.path.join(a.out, a.tag + ".progress.json"))
    # reviewer A's widths: poles of each input's return signal by the matrix pencil, at two sample lengths (the error)
    tt = np.array([s_[0] for s_ in S_samples])
    widths = []; poles = []
    for j in range(K):
        sig = np.array([s_[1][j, j] for s_ in S_samples])
        if len(sig) < 16:
            continue
        w1, c1, _ = KR.matrix_pencil(sig, a.dt_sample, n_poles=min(6, len(sig) // 4))
        w2, c2, _ = KR.matrix_pencil(sig[: len(sig) // 2], a.dt_sample, n_poles=min(6, len(sig) // 8))
        j1 = int(np.argmax(np.abs(c1))); j2 = int(np.argmin(np.abs(w2 - w1[j1])))
        g1 = float(-2 * w1[j1].imag); g2 = float(-2 * w2[j2].imag)
        poles.append({"input": j, "e_over_T0": float(e_all[j] / T0), "Re_over_T0": float(w1[j1].real / T0), "Gamma_L": g1 * L, "Gamma_L_half_sample": g2 * L,
                      "rel_err": abs(g1 - g2) / max(abs(g1), 1e-300)})
        widths.append(g1)
    res["poles"] = poles
    Gs = {T: G[:ns, :ns] for T, G in G_by_T.items()}; Ge = {T: G[ns:, ns:] for T, G in G_by_T.items()}
    res["supply"] = KR.pc13_readouts(es, Gs, T0, L, qv[:ns], widths=widths[:ns] if len(widths) >= ns else None) if ns else None
    res["exit"] = KR.pc13_readouts(ee, Ge, T0, L, qv[ns:], widths=widths[ns:] if len(widths) >= K else None) if ne else None
    res["joint_spectra"] = {"%.6g" % (T / L): [float(x) for x in np.sort(np.linalg.eigvalsh(G))[::-1]] for T, G in G_by_T.items()}
    res["particle_hole_cross_block_norm"] = {"%.6g" % (T / L): float(np.linalg.norm(G[:ns, ns:])) for T, G in G_by_T.items()}
    res["retained"] = {"%.6g" % (T / L): [float(x) for x in rv] for T, rv in ret_by_T.items()}
    # collected-output entropies of the declared input covariance (the trapped-supply hypothesis; labelled)
    C_in = np.zeros((K, K), complex)
    if a.cin == "supply_filled":
        C_in[np.arange(ns), np.arange(ns)] = 1.0
    elif a.cin == "exit_hole":
        C_in[np.arange(ns, K - 1), np.arange(ns, K - 1)] = 1.0
    res["cin"] = a.cin
    res["collected_entropies"] = KR.collected_entropies(sorted(G_by_T.items()), C_in) if np.any(C_in) else None
    np.savez_compressed(os.path.join(a.out, a.tag + ".npz"), T=np.array(sorted(G_by_T)), G=np.array([G_by_T[T] for T in sorted(G_by_T)]),
                        S_t=tt, S_diag=np.array([np.diag(s_[1]) for s_ in S_samples]),
                        S_full_every16=np.array([s_[1] for s_ in S_samples[::16]]), e=e_all, q_r=qv, n_supply=ns, C_in=C_in)
    res["npz"] = a.tag + ".npz"
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s (%.0f s)" % (a.tag, res["seconds"]))


if __name__ == "__main__":
    main()
