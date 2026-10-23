#!/usr/bin/env python
"""run_bprime_v4.py (busi; the B' cube driver of design v2: run_bprime_v3.py with the changes both checks of design v1 named,
ReviewA_bpchk_v1 and ReviewB_bpchk_v1, inbox #454 (2); run_bprime_v2/v3 are kept unchanged).

Two phases, so the card needs no record files and no eigensolve:
  --phase prep (box CPU): fields and inputs -> <out>/<tag>_prep.npz (+ json manifest);
  --phase run (card or CPU): loads --prep <npz>, propagates, writes <tag>.json and <tag>.npz;  --phase both does the two in one call.

Fields (prep): --family F1 (re-solved dense, with the re-solve diagnostics and the engine-kernel lift in the manifest) | F2 | A32 (site rows;
--sigma-over-L 0 is its applied-row baseline, a positive value a constructed re-average, labelled); --splice none | reviewb | reviewb_out;
the production back-average: the 3D Gaussian of std --sigma-over-L L (= the engine's site_fields_kernel_average in 3D; --kernel compact |
cosine are labelled alternatives); the clock scan e^(-d f) on N and v with the ramp relative to the baseline (d = 0) averaged root
(--ramp root, default; --ramp L the pc13 literal); the port: the sin^2 CAP beyond --onset L to --eta-max over 0.5 L, half-side at least
onset + 0.5 L (moved splice: onset 2.5 L) unless --no-auto-geometry; the hydrodynamic gate sigma |H|, sigma ||K||_op along r.
Inputs (prep): the retained block's eigenvectors in (0, T0) [or (0, 3 T0) with --wide] and (-T0, 0), deterministic, the zero-mode cluster
printed, R_I = the averaged root (--retained root) or interface:<r/L>; saved: the basis on the sub-box (complex128), F^dag q_r F (full), the
energies.
Run: Chebyshev-Strang steps of --dt; columns in --col-batch batches (states checkpointed at the clocks, cross-batch blocks assembled by
pt3b_v2lib.gram_assemble); per clock G(T) and the retained Gram R(T); the return S(t) = F^dag Psi every --dt-sample (diagonal) and in full
every 16th sample; every 16th sample per column the radial profiles (unit bins) of |psi|^2, the energy density Re psi^* h psi and the q_r
density Re psi^* q_r psi, and the absorbed flux 2 eta |psi|^2 accumulated per sample interval in three direction sectors ([100], [110],
[111]-like; a labelled proxy for the channel content; the O_h irreps are deferred). Readouts: pc13_readouts per window, nearest-energy poles
with errors and deduplication, N_acc censored (never zero), the G4 reader (bins, early-late, retained-output) on --cin (supply_filled,
exit_hole, or file:<npz> with key C on the inputs), refused for a supply-exit coherent C unless --allow-cross. --backend free (the only one
built; coupling 0; reviewer B's injectable Page-arm hook, recorded in the manifest). Timing per step and peak device memory recorded.
Device: cpu, cuda:1 / cuda:2 on the box, or any cuda:N with --card (a rented card running this alone)."""
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
import pt3b_v2lib as V2  # noqa: E402


def sectors(r, X, Y, Z):
    ax = np.stack([np.abs(X), np.abs(Y), np.abs(Z)]); mx = ax.max(axis=0); mn = ax.min(axis=0)
    rr = np.where(r > 0, r, 1.0)
    s100 = mx > 0.85 * rr
    s111 = (mn > 0.45 * rr) & ~s100
    s110 = ~(s100 | s111)
    return [s100, s110, s111]


def prep(a):
    t0 = time.time()
    L = a.L; T0 = 1.0 / (4 * math.pi * L)
    onset, boxL = V2.exterior_geometry(a.splice, a.onset, a.box_L, auto=not a.no_auto_geometry)
    H = int(round(boxL * L))
    if a.family == "F1":
        prof = V2.dense_record_profile_diag(*CU.REC_F1)
    else:
        prof = {"F2": CU.f2_profile, "A32": CU.a32_profile}[a.family]()
    prof = CU.splice(prof, a.splice)
    sig = a.sigma_over_L * L
    N0, v0, r, inf0 = V2.gauss_cube_fields(prof, L, H, 0.0, ell=sig, kernel=a.kernel)
    R_b, roots0 = V2.outer_root_axis(N0, v0, r, H)
    if R_b is None:
        R_b = max(prof["roots_over_L"]) * L
    if a.d:
        N, v, r, inf = V2.gauss_cube_fields(prof, L, H, a.d, ell=sig, kernel=a.kernel, ramp=a.ramp, R_ramp=R_b)
    else:
        N, v, inf = N0, v0, inf0
    R_root, roots = V2.outer_root_axis(N, v, r, H)
    R_root = R_root or R_b
    R_I = R_root if a.retained == "root" else float(a.retained.split(":")[1]) * L
    eta = CU.cap_reviewa(r, L, onset, 0.5, a.eta_max)
    man = {"family": a.family, "profile": prof["label"], "L": L, "T0": T0, "box_L": boxL, "onset_L": onset, "H": H, "sigma_sites": sig,
           "kernel_info": inf, "baseline_root": R_b, "baseline_roots": roots0, "roots_d": roots, "R_root": R_root, "R_I": R_I,
           "hydro_gate": V2.hydro_gate(prof, L, max(sig, 1.0), 2.5 * L), "resolve_info": prof.get("resolve_info"), "kernel_lift": prof.get("kernel_lift"),
           "npz_md5": prof.get("npz_md5"), "args": vars(a)}
    if a.family == "A32":
        man["A32_note"] = "site rows (already back-averaged by the engine): sigma 0 is the applied-row baseline; sigma > 0 a constructed re-average"
    K3.log("prep %s L %g sigma %.3g (%s) d %g ramp %s: box %.2f L onset %.2f L, roots %s, R_I %.3f; hydro max sigma|H| %.3g" % (
        a.family, L, sig, a.kernel, a.d, a.ramp, boxL, onset, roots, R_I, man["hydro_gate"]["max_sigma_H"]))
    Hs = int(math.ceil(R_I)) + 2
    off = H - Hs; sl = slice(off, off + 2 * Hs + 1)
    hs = CU.cubic_csr(N[sl, sl, sl], [c[sl, sl, sl] for c in v])
    ims = CU.masks(r[sl, sl, sl], R_I)
    q = CU.q_r_csr(N[sl, sl, sl], None, r[sl, sl, sl])
    zi = {}
    hi_s = 3 * T0 if a.wide else T0
    es, Us, dI = V2.interior_window_det(hs, ims, 0.0, hi_s, dense_max=a.dense_max, info=zi)
    ee, Ue, _ = V2.interior_window_det(hs, ims, -T0, 0.0, dense_max=a.dense_max, info=zi)
    if a.max_inputs:
        es, Us, ee, Ue = es[: a.max_inputs], Us[:, : a.max_inputs], ee[-a.max_inputs:], Ue[:, -a.max_inputs:]
    e_all = np.concatenate([es, ee]); U_all = np.concatenate([Us, Ue], axis=1)
    qF = U_all.conj().T @ (q @ U_all)
    man["inputs"] = {"retained_dim": dI, "n_supply": len(es), "n_exit": len(ee), "zero": zi, "e_over_T0": [float(x) for x in e_all / T0],
                     "q_r_diag": [float(x) for x in np.real(np.diag(qF))]}
    K3.log("inputs: supply %d, exit %d, zero modes %d, retained dim %d (%.0f s)" % (len(es), len(ee), zi.get("n_zero", 0), dI, time.time() - t0))
    path = os.path.join(a.out, a.tag + "_prep.npz")
    np.savez_compressed(path, N=N, vx=v[0], vy=v[1], vz=v[2], eta=eta, U_sub=U_all, e=e_all, n_supply=len(es), Hs=Hs, H=H, qF=qF, L=L, T0=T0, R_I=R_I,
                        manifest=json.dumps(man, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    K3.dump_json(man, os.path.join(a.out, a.tag + "_prep.json"))
    return path


def run(a, prep_path):
    t0 = time.time()
    dev = torch.device(a.device)
    Z = np.load(prep_path)
    man = json.loads(str(Z["manifest"]))
    L = float(Z["L"]); T0 = float(Z["T0"]); H = int(Z["H"]); Hs = int(Z["Hs"]); R_I = float(Z["R_I"]); ns = int(Z["n_supply"])
    N = Z["N"]; v = [Z["vx"], Z["vy"], Z["vz"]]; eta = Z["eta"]
    U_sub = Z["U_sub"]; e_all = Z["e"]; qF = Z["qF"]; K = len(e_all)
    if a.max_columns:
        keep = list(range(min(ns, a.max_columns // 2))) + list(range(ns, ns + min(K - ns, a.max_columns - min(ns, a.max_columns // 2))))
        U_sub = U_sub[:, keep]; e_all = e_all[keep]; qF = qF[np.ix_(keep, keep)]; ns = min(ns, a.max_columns // 2); K = len(keep)
    X_, Y_, Z_, r = CU.grid(H)
    Lf = 2 * H + 1; Ls = 2 * Hs + 1; off = H - Hs; sl = slice(off, off + Ls)
    res = {"script": "run_bprime_v4.py", "args": vars(a), "prep": os.path.basename(prep_path), "prep_manifest": man, "own_md5": K3.own_md5s(),
           "backend": {"name": a.backend, "coupling": 0.0, "note": "reviewer B bpchk sec. 4: injectable Page-arm backend; 'free' = h[N, v] at coupling 0"},
           "device": str(dev), "K": K, "n_supply": ns}
    if a.backend != "free":
        raise SystemExit("only the free backend is built")
    thetas = [1e-1, 1e-2, 1e-3, 1e-4]
    Ts = sorted(set([x * L for x in (1e2, 1e3)] + [math.log(2) * L / th for th in thetas] + [x * L for x in a.extra_T_over_L]))
    Ts = [T for T in Ts if T <= a.T_max_over_L * L + 1e-9]
    res["clocks_over_L"] = [T / L for T in Ts]
    cube = CU.Cube(N, v, eta, a.device)
    rh = [np.where(r > 0, c / np.where(r > 0, r, 1), 0.0) for c in (X_, Y_, Z_)]
    qcube = CU.Cube(np.zeros_like(N), [-x for x in rh], None, a.device)
    imf = torch.as_tensor(np.broadcast_to((r < R_I)[None], (4, Lf, Lf, Lf)).copy(), device=dev)
    rbin = torch.as_tensor(np.floor(r).astype(np.int64).ravel(), device=dev); nrb = int(np.floor(r).max()) + 1
    secs = [torch.as_tensor(s_.ravel(), device=dev) for s_ in sectors(r, X_, Y_, Z_)]
    eta_t = torch.as_tensor(eta.ravel(), dtype=CU.RD, device=dev)
    Fsub = torch.as_tensor(U_sub.T.reshape(K, 4, Ls, Ls, Ls), dtype=CU.CD, device=dev)
    nb_ = a.col_batch or K
    batches = [list(range(i, min(K, i + nb_))) for i in range(0, K, nb_)]
    ckpt = {T: [] for T in Ts}
    S_diag, S_full16, prof16, flux16, t_samp = [], [], [], [], []
    step_times = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for bi, cols in enumerate(batches):
        kb = len(cols)
        X = torch.zeros((kb, 4, Lf, Lf, Lf), dtype=CU.CD, device=dev)
        X[:, :, sl, sl, sl] = Fsub[cols]
        t = 0.0; t_next = a.dt_sample; isamp = 0
        Sd_b, Sf_b, pr_b, fl_b, ts_b = [], [], [], [], []
        flux_acc = torch.zeros((kb, 3), dtype=CU.RD, device=dev); t_last = 0.0
        for T in Ts:
            while t < T - 1e-9:
                hh = min(a.dt, T - t)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                ts0 = time.time()
                dens = (X.abs() ** 2).sum(dim=1).reshape(kb, -1)
                flux_acc += hh * torch.stack([(2 * eta_t * dens * s_).sum(dim=1) for s_ in secs], dim=1)
                X = cube.step_cap(X, hh); t += hh
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                step_times.append(time.time() - ts0)
                if t >= t_next - 1e-9:
                    Ssub = X[:, :, sl, sl, sl].reshape(kb, -1)
                    S = (Fsub.reshape(K, -1).conj() @ Ssub.T)                     # (K, kb): <F_i | psi_j>
                    Sd_b.append(S[cols, torch.arange(kb)].cpu().numpy()); ts_b.append(t)
                    if isamp % 16 == 0:
                        Sf_b.append(S.cpu().numpy())
                        hX = cube.apply(X)
                        qX = qcube.apply(X)
                        flat = lambda Y: Y.reshape(kb, 4, -1)
                        d_ = (flat(X).abs() ** 2).sum(dim=1)
                        en = torch.real(flat(X).conj() * flat(hX)).sum(dim=1)
                        qd = torch.real(flat(X).conj() * flat(qX)).sum(dim=1)
                        rows = []
                        for arr in (d_, en, qd):
                            o = torch.zeros((kb, nrb), dtype=CU.RD, device=dev)
                            o.index_add_(1, rbin, arr)
                            rows.append(o)
                        pr_b.append(torch.stack(rows, dim=1).float().cpu().numpy())
                        fl_b.append(flux_acc.cpu().numpy().copy())
                        del hX, qX
                    isamp += 1; t_next += a.dt_sample
            if len(batches) == 1:
                Xf = X.reshape(kb, -1)
                Gd = torch.eye(kb, dtype=CU.CD, device=dev) - Xf.conj() @ Xf.T
                Xi = (X * imf).reshape(kb, -1)
                Rd = Xi.conj() @ Xi.T
                ckpt[T] = ("direct", 0.5 * (Gd + Gd.conj().T).cpu().numpy(), 0.5 * (Rd + Rd.conj().T).cpu().numpy())
            else:
                ckpt[T].append(X.cpu().numpy() if not a.ckpt_disk else _save(a, bi, T, X))
            K3.log("  batch %d/%d T/L %.4g (%.0f s; step %.3f s median)" % (bi + 1, len(batches), T / L, time.time() - t0, float(np.median(step_times))))
            K3.dump_json({"partial": [bi, T / L]}, os.path.join(a.out, a.tag + ".progress.json"))
        S_diag.append(np.array(Sd_b)); S_full16.append(np.array(Sf_b)); prof16.append(np.array(pr_b)); flux16.append(np.array(fl_b)); t_samp = ts_b
        del X
    S_diag = np.concatenate(S_diag, axis=1); S_full16 = np.concatenate(S_full16, axis=2); prof16 = np.concatenate(prof16, axis=1)
    flux16 = np.concatenate(flux16, axis=1)
    imask_flat = np.broadcast_to((r < R_I)[None], (4, Lf, Lf, Lf)).ravel()
    G_by_T, R_by_T = {}, {}
    for T in Ts:
        if isinstance(ckpt[T], tuple):
            G_by_T[T] = ckpt[T][1]; R_by_T[T] = ckpt[T][2]
            continue
        parts = [(_load(p_) if isinstance(p_, str) else p_).reshape(-1, Lf ** 3 * 4).T for p_ in ckpt[T]]
        G, R = V2.gram_assemble(parts, imask_flat)
        G_by_T[T] = G; R_by_T[T] = R
    res["timing"] = {"steps": len(step_times), "step_median_s": float(np.median(step_times)), "step_mean_s": float(np.mean(step_times)),
                     "cheb_bound": cube.bound, "dt": a.dt, "seconds_total": time.time() - t0,
                     "peak_mem_GB": (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None,
                     "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "col_batch": nb_, "columns": K}
    Gs = {T: G[:ns, :ns] for T, G in G_by_T.items()}; Ge = {T: G[ns:, ns:] for T, G in G_by_T.items()}
    prow = V2.poles_for_inputs(S_diag, a.dt_sample, e_all, L, Ts[-1])
    res["poles"] = prow
    res["supply"] = KR.pc13_readouts(e_all[:ns], Gs, T0, L, np.real(np.diag(qF))[:ns]) if ns else None
    res["exit"] = KR.pc13_readouts(e_all[ns:], Ge, T0, L, np.real(np.diag(qF))[ns:]) if K > ns else None
    if ns:
        res["supply"]["N_acc_censored"] = V2.censored_counts(Gs, L, thetas, Ts[-1], prow[:ns])
    if K > ns:
        res["exit"]["N_acc_censored"] = V2.censored_counts(Ge, L, thetas, Ts[-1], prow[ns:])
    res["joint_spectra"] = {"%.6g" % (T / L): [float(x) for x in np.sort(np.linalg.eigvalsh(G))[::-1]] for T, G in G_by_T.items()}
    C_in = np.zeros((K, K), complex)
    if a.cin == "supply_filled":
        C_in[np.arange(ns), np.arange(ns)] = 1.0
    elif a.cin == "exit_hole":
        C_in[np.arange(ns, K - 1), np.arange(ns, K - 1)] = 1.0
    elif a.cin.startswith("file:"):
        C_in = np.load(a.cin[5:])["C"]
    res["cin"] = a.cin
    try:
        rd = V2.bins_reader(sorted(G_by_T.items()), C_in, allow_cross=a.allow_cross, ns=ns, R_list=[R_by_T[T] for T in sorted(G_by_T)]) if np.any(C_in) else None
        res["reader"] = V2.bins_reader_json(rd) if rd else None
    except ValueError as ex:
        res["reader"] = {"refused": str(ex)}
    np.savez_compressed(os.path.join(a.out, a.tag + ".npz"), T=np.array(sorted(G_by_T)), G=np.array([G_by_T[T] for T in sorted(G_by_T)]),
                        R=np.array([R_by_T[T] for T in sorted(G_by_T)]), S_t=np.array(t_samp), S_diag=S_diag, S_full_every16=S_full16,
                        radial_profiles_every16=prof16, absorbed_flux_sectors_every16=flux16, e=e_all, qF=qF, C_in=C_in, n_supply=ns)
    res["npz"] = a.tag + ".npz"
    res["seconds"] = time.time() - t0
    K3.dump_json(res, os.path.join(a.out, a.tag + ".json"))
    K3.log("wrote %s (%.0f s)" % (a.tag, res["seconds"]))


def _save(a, bi, T, X):
    p = os.path.join(a.out, "%s_ckpt_b%d_T%.6g.npy" % (a.tag, bi, T))
    np.save(p, X.cpu().numpy())
    return p


def _load(p):
    Y = np.load(p)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "run", "both"], default="both")
    ap.add_argument("--prep", default="")
    ap.add_argument("--family", choices=["F1", "F2", "A32"], default="F1")
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--splice", choices=["none", "reviewb", "reviewb_out"], default="none")
    ap.add_argument("--sigma-over-L", type=float, default=1.0 / 8)
    ap.add_argument("--kernel", choices=["gauss", "compact", "cosine"], default="gauss")
    ap.add_argument("--d", type=float, default=0.0)
    ap.add_argument("--ramp", choices=["root", "L"], default="root")
    ap.add_argument("--box-L", type=float, default=2.5)
    ap.add_argument("--onset", type=float, default=2.0)
    ap.add_argument("--no-auto-geometry", action="store_true")
    ap.add_argument("--eta-max", type=float, default=1.0)
    ap.add_argument("--retained", default="root")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--max-inputs", type=int, default=0)
    ap.add_argument("--dense-max", type=int, default=6000)
    ap.add_argument("--T-max-over-L", type=float, default=100.0)
    ap.add_argument("--extra-T-over-L", type=float, nargs="*", default=[])
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--dt-sample", type=float, default=8.0)
    ap.add_argument("--col-batch", type=int, default=0)
    ap.add_argument("--max-columns", type=int, default=0)
    ap.add_argument("--ckpt-disk", action="store_true")
    ap.add_argument("--cin", default="supply_filled")
    ap.add_argument("--allow-cross", action="store_true")
    ap.add_argument("--backend", default="free")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--card", action="store_true", help="a rented card running this alone: any cuda:N allowed")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.device.startswith("cuda") and not a.card and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("on the box only cuda:1 or cuda:2")
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    p = a.prep
    if a.phase in ("prep", "both"):
        p = prep(a)
    if a.phase in ("run", "both"):
        run(a, p)


if __name__ == "__main__":
    main()
