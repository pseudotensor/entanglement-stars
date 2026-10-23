#!/usr/bin/env python
"""run_bprime_v1.py (busi; B' driver, one size and one background per call). ReviewA_pc12_v1 sec. 3.1 and ReviewB_pc12_v1 sec. 3 Computation A.

--operator cube: the four-component cubic h[N, v] (pt3b_cube) on a box of half-side --box-rs r_s with the absorber beyond --cap-rs r_s.
--operator chains: kappa = 1 .. --kmax (weights 16 kappa) on the same radial profile (pt3b_chain), the comparator.
--profile matched (reviewer B item 2: the rooted record's dimensionless profile resampled at r_s/a) or stored:<path.npz>,<T/r_s> (reviewer A item 1).
--clock d,mode: the backreaction / isolation scan (mode reviewa: Delta ln N = -d inside the root; reviewb: N, v x e^(-d f)).
Retained inputs: the interior block's eigenvectors (sites inside the outer root) in the supply window (0, T_H) and the exit window (-T_H, 0),
from a sub-box CSR (dense eigh below --dense-max, else shift-invert). The kernel: propagation under h - i eta to the clocks
T_theta = ln 2 r_s/theta (--thetas) plus --extra-T; per clock the escape spectrum p (reviewer B's whole singular spectrum of K^dag K), the
retained weights and the return matrix S = F^dag U(T) F; the readouts of pt3b_kernels.readouts (N_w, N_acc(theta), densities, D); q_r.
--max-inputs caps the inputs per window (smoke tests); --batch the columns propagated together. Device: cpu, or cuda:1 / cuda:2 only."""
import argparse
import json
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
import pt3b_chain as CH  # noqa: E402


def profile(a):
    if a.profile == "matched":
        return CU.matched_profile(a.rs_a)
    path, Trs = a.profile.split(":", 1)[1].split(",")
    Z = np.load(path, allow_pickle=True)
    J = json.load(open(path.replace("_fields.npz", ".json")))
    rs = float(J["args"]["rs"])
    T = np.asarray(Z["T"], float)
    k = int(np.argmin(np.abs(T / rs - float(Trs))))
    recs = {round(float(r["T"]), 6): r for r in J["records"] if isinstance(r, dict) and r.get("T") is not None}
    rec = recs[round(float(T[k]), 6)]
    R = np.asarray(Z["R"], float)
    E = float(rec["E"]); g = float(Z["g"])
    meta = {"label": "%s at T/r_s %.4f (stored record, own size)" % (os.path.basename(path), T[k] / rs), "T_H": 1.0 / (8 * math.pi * g * E),
            "roots": list(rec.get("trapping_zeros") or []), "rs_a": rs}
    return R, np.asarray(Z["N_applied"][k], float), np.abs(np.asarray(Z["v_applied"][k], float)), 2 * g * E, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", choices=["cube", "chains"], default="cube")
    ap.add_argument("--rs-a", type=float, required=True)
    ap.add_argument("--profile", default="matched")
    ap.add_argument("--clock", default="0,reviewb")
    ap.add_argument("--box-rs", type=float, default=2.5)
    ap.add_argument("--cap-rs", type=float, default=2.0)
    ap.add_argument("--eta0", type=float, default=0.05)
    ap.add_argument("--thetas", default="0.1,0.01,0.001")
    ap.add_argument("--extra-T", default="")
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--dense-max", type=int, default=6000)
    ap.add_argument("--max-inputs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--interface-shift", type=float, default=0.0, help="reviewer B: move the retained/exterior interface to R_t + shift (sites)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    rp, Np, up, rs_vac, meta = profile(a)
    rs_a = meta.get("rs_a", a.rs_a)
    R_t = max(meta["roots"]) if meta["roots"] else None
    if R_t is None:
        raise SystemExit("no trapping root in this background: B' needs a rooted record (%s)" % meta["label"])
    T_H = meta["T_H"]
    R_root = R_t
    R_t = R_t + a.interface_shift                                  # the retained space's radius (the root by default)
    d, mode = a.clock.split(","); d = float(d)
    thetas = [float(x) for x in a.thetas.split(",")]
    T_list = KR.T_list_for(rs_a, thetas, [float(x) * rs_a for x in a.extra_T.split(",") if x])
    res = {"script": "run_bprime_v1.py", "args": vars(a), "profile": meta, "R_t": R_t, "R_root": R_root, "T_H": T_H, "T_list": T_list, "own_md5": K3.own_md5s()}
    K3.log("B' %s r_s/a %g (%s), R_t %.3f, T_H %.4e, clock %s, clocks %s" % (a.operator, rs_a, meta["label"], R_t, T_H, a.clock, [round(x / rs_a, 2) for x in T_list]))
    if a.operator == "chains":
        per, tot = CH.chains_comparator(rp, Np, up, rs_vac, rs_a, R_t, T_H, kmax=a.kmax, box_rs=a.box_rs, cap_rs=a.cap_rs, eta0=a.eta0, T_list=T_list,
                                        dt=a.dt, clock=(d, mode) if d else None)
        res["per_channel"] = per; res["weighted"] = tot
        K3.log("chains: weighted N_w supply %s exit %s; N_acc supply %s" % (tot["supply"]["N_w_weighted"], tot["exit"]["N_w_weighted"], tot["supply"]["N_acc_weighted"]))
    else:
        H = int(round(a.box_rs * rs_a)); L = 2 * H + 1
        N, v, r = CU.fields_from_profile(H, rp, Np, up, rs_vac)
        N, v = CU.clock_scan(N, v, r, R_root, d, mode)
        eta = CU.cap(r, a.cap_rs * rs_a, H * math.sqrt(3), a.eta0)
        Hs = int(math.ceil(R_t)) + 2
        Ns, vs, rsb = CU.fields_from_profile(Hs, rp, Np, up, rs_vac)
        Ns, vs = CU.clock_scan(Ns, vs, rsb, R_root, d, mode)
        hs = CU.cubic_csr(Ns, vs)
        ims = CU.masks(rsb, R_t)
        q = CU.q_r_csr(Ns, vs, rsb)
        cube = CU.Cube(N, v, eta, a.device)
        res["box"] = {"H": H, "L": L, "amplitudes": 4 * L ** 3, "interior_dim": int(ims.sum())}
        Ls = 2 * Hs + 1
        off = H - Hs

        def embed(Us):
            """sub-box columns (4 Ls^3, m) -> full-box tensor (m, 4, L, L, L)."""
            m = Us.shape[1]
            X = np.zeros((m, 4, L, L, L), complex)
            X[:, :, off:off + Ls, off:off + Ls, off:off + Ls] = Us.T.reshape(m, 4, Ls, Ls, Ls)
            return X
        imf = torch.as_tensor(np.broadcast_to((r < R_t)[None], (4, L, L, L)).copy(), device=cube.dev)
        for lab, (lo, hi) in {"supply": (0.0, T_H), "exit": (-T_H, 0.0)}.items():
            zinfo = {}
            e, Us, dI = KR.interior_window(hs, ims, lo, hi, dense_max=a.dense_max, info=zinfo)
            if a.max_inputs:
                e, Us = e[: a.max_inputs], Us[:, : a.max_inputs]
            K3.log("%s window: %d interior inputs (interior dim %d): e/T_H %s" % (lab, len(e), dI, np.round(e / T_H, 3).tolist()[:20]))
            rr = {"N_w": int(len(e)), "n_structural_zero_modes_excluded": zinfo.get("n_zero", 0), "e_over_TH": [float(x) for x in e / T_H],
                  "q_r": [float(x) for x in np.real(np.einsum("ij,ij->j", Us.conj(), q @ Us))]}
            if len(e):
                kern = [{"T": T, "G": np.zeros((len(e), len(e)), complex), "retained": np.zeros(len(e)), "S": np.zeros((len(e), len(e)), complex)} for T in T_list]
                F = embed(Us)
                for b0 in range(0, len(e), a.batch):
                    X = torch.as_tensor(F[b0:b0 + a.batch], dtype=CU.CD, device=cube.dev)
                    X0 = X.clone()
                    t = 0.0
                    for k_, T in enumerate(T_list):
                        while t < T - 1e-9:
                            hh = min(a.dt, T - t)
                            X = cube.step_cap(X, hh); t += hh
                        # the block's Gram against every other block needs all columns at T: keep the block diagonal Grams exactly and the
                        # cross-block Grams from the stored columns (small K): here the full set is propagated in one batch when K <= batch
                        Xf = X.reshape(X.shape[0], -1)
                        Gb = (Xf.conj() @ Xf.T).cpu().numpy()
                        sl = slice(b0, b0 + X.shape[0])
                        kern[k_]["G"][sl, sl] = Gb
                        kern[k_]["retained"][sl] = (X.abs() ** 2 * imf).reshape(X.shape[0], -1).sum(dim=1).cpu().numpy()
                        kern[k_]["S"][sl, sl] = (X0.reshape(X.shape[0], -1).conj() @ Xf.T).cpu().numpy()
                        K3.log("  %s batch %d: T = %.2f r_s, norms %s (%.0f s)" % (lab, b0, T / rs_a, np.round(np.real(np.diag(Gb)), 5).tolist()[:8], time.time() - t0))
                    del X, X0
                ker = []
                for kk in kern:
                    Gm = np.eye(len(e)) - kk["G"]; Gm = 0.5 * (Gm + Gm.conj().T)
                    p = np.linalg.eigvalsh(Gm) if len(e) <= a.batch else np.real(np.diag(Gm))
                    ker.append({"T": kk["T"], "p": p, "trace_G": float(np.real(np.trace(Gm))), "retained": kk["retained"], "S": kk["S"],
                                "unphysical_p": int(np.sum((p < -1e-10) | (p > 1 + 1e-10))), "singular_spectrum": len(e) <= a.batch})
                rr.update(KR.readouts(e, ker, T_H, rs_a, thetas))
                rr["return_S_diag"] = {"%.4g" % (kk["T"] / rs_a): [complex(x) for x in np.diag(kk["S"])] for kk in ker}
            res[lab] = rr
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s (%.0f s)" % (os.path.join(a.out, a.tag + ".json"), res["seconds"]))


if __name__ == "__main__":
    main()
