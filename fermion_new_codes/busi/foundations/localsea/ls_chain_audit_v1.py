#!/usr/bin/env python
"""ls_chain_audit_v1.py (busi, inbox #447; ReviewB_found4_v1.md sec. 5.2, ReviewA_found4_v1.md sec. 3 item 4): the fixed-record
local-sea audit on the stored chain records, radius by radius. Box CPU only (torch complex128, at most 8 threads). Read-only on
every input.

The production engine's chains (engine_vacflow_v1.chain_sparse, engine_collapse_v5.Chan; re-implemented here and checked against the
slice's stored spectrum): channel kappa = 1..K, weight w = 16 kappa, two components per site, R_n = n + 1, on-site m sz + (kappa/R) sy,
sine bond -(i/2) sx, two-step Q -(i/4) 1; h_m = {N, h}/2 + {v, Q}/2 at the applied (back-averaged) fields. Densities per site,
eps_n = sum_kappa w Re(phi^+ h phi)_n and J_n with Q (the engine's dens()).

Modes:
  slice : one slice record. (a) rebuild h_m per channel, eigh, check against the stored eigenvalues; (b) the production reference R2
          (stored columns) against the instantaneous negative projector P_neg of h_m: occupation mismatch n_sea - ||R2^+ U_neg||_F^2
          and the density difference; (c) the band character of every eigenvector against the clock-only sea P_N (the negative
          projector of {N, h}/2, globally gapped): the count of mixed vectors (b in (0.05, 0.95)); (d) radius by radius: the
          production reference's polarization eps_R2 - eps_0, J_R2 - J_0 (eps_0 the flat chain sea, N = 1, v = 0); the propagated
          sea's (the state's sea columns) against R2, i.e. the pairs; (e) the 3D local sea at second order on the same radial
          fields (ls_zone_v1 coefficients, reviewer A's basis; the cubic term sphere-averaged) as shell totals per site.
  scale : the configuration scaled by lambda (fields u(r / lambda) on lambda times as many sites; kappa / r scales with it): the
          shell totals of P_neg - P_0 at r = lambda R_n, whose Richardson combination in 1/lambda^2 separates the chain sea's
          second-order (gradient) polarization from its fourth and higher orders: the all-orders production reference against
          the second-order local sea (reviewer B) and against the fourth-order one (reviewer A), radius by radius.
  rates : every record in a clock window, from the stored applied fields alone (R2 = P_neg on records whose json says
          mask_equals_negative_set): eps_R2, J_R2 per site, checked against the handoff's eps_R2; then the reference's rate at the
          record's outer root, -d/dt int_0^R 4 pi r^2 (rho_R2 + w J_R2) (smoothed with the field module's kernel), against the
          evolution row's largest term max(4 pi N R^2 |rho_l|, 4 pi N R^2 |p_r|, 8 pi N R^2 |J_l|) at the root.
Outputs: <out>/ls_chain_audit_v1_<tag>.npz and .json."""
import os
import sys


def _threads():
    t = "8"
    if "--threads" in sys.argv:
        t = sys.argv[sys.argv.index("--threads") + 1]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = t
    return int(t)


NT = _threads()
import argparse
import hashlib
import importlib.util
import json
import math
import time
import numpy as np
import torch

torch.set_num_threads(NT)
CD = torch.complex128
FM_PATH = "<internal>/engine_fields_v16.py"   # md5 7a69faad (the reader's field module)


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def field_module():
    sys.dont_write_bytecode = True
    d = os.path.dirname(FM_PATH)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("engine_fields_v16_ro", FM_PATH)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M


# ------------------------------------------------------------------ chains
def chain_dense(R, m, kappa):
    """dense h (unit clock, no flow) and Q, as engine_vacflow_v1.chain_sparse."""
    n = len(R)
    h = torch.zeros(2 * n, 2 * n, dtype=CD)
    idx = torch.arange(n)
    b = 2 * idx
    h[b, b] = m; h[b + 1, b + 1] = -m
    h[b, b + 1] = torch.as_tensor(-1j * kappa / R, dtype=CD); h[b + 1, b] = torch.as_tensor(1j * kappa / R, dtype=CD)
    i = torch.arange(n - 1); bi, ci = 2 * i, 2 * i + 2
    h[bi, ci + 1] = -0.5j; h[ci + 1, bi] = 0.5j
    h[bi + 1, ci] = -0.5j; h[ci, bi + 1] = 0.5j
    Q = torch.zeros(2 * n, 2 * n, dtype=CD)
    j = torch.arange(n - 2)
    for a in range(2):
        r_, c_ = 2 * j + a, 2 * j + 4 + a
        Q[r_, c_] = -0.25j; Q[c_, r_] = 0.25j
    return h, Q


def clocked(h, Q, N, v):
    Nd = torch.as_tensor(np.repeat(np.asarray(N, float), 2), dtype=CD)
    Vd = torch.as_tensor(np.repeat(np.asarray(v, float), 2), dtype=CD)
    return 0.5 * (Nd[:, None] * h + h * Nd[None, :]) + 0.5 * (Vd[:, None] * Q + Q * Vd[None, :])


def dens(A, Phi):
    """per site sum over the columns of Re(Phi^+ P_n A Phi)."""
    if Phi.shape[1] == 0:
        return np.zeros(A.shape[0] // 2)
    d = torch.real((Phi.conj() * (A @ Phi)).sum(1))
    return d.reshape(-1, 2).sum(1).numpy()


def neg_cols(H):
    e, U = torch.linalg.eigh(H)
    return e, U, U[:, e < 0]


# ------------------------------------------------------------------ the 3D local sea at second order on radial fields
def local_sea_3d(R, N, v, co):
    """shell totals per site (4 pi r^2 x density, unit radial spacing) of the 3D local sea's second-order energy and radial
    momentum densities on the radial fields, from the reduced functional L = 4 pi r^2 F(v', v/r)/N, F = s1 (x+2y)^2
    + s2 (x^2 + 2 y^2) + s3 [y^2 + 2 x y + (3/5)(x - y)^2] (x = v', y = v/r; the cubic term sphere-averaged; sb, sc drop out).
    rho_shell = dL/dN = -L/N; J_shell = dL/dv - d/dr dL/dv'."""
    s1, s2, s3 = co["s1"], co["s2"], co["s3"]
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    x = np.gradient(v, R); y = v / R
    F = s1 * (x + 2 * y) ** 2 + s2 * (x ** 2 + 2 * y ** 2) + s3 * (y ** 2 + 2 * x * y + 0.6 * (x - y) ** 2)
    Fx = 2 * s1 * (x + 2 * y) + 2 * s2 * x + s3 * (2 * y + 1.2 * (x - y))
    Fy = 4 * s1 * (x + 2 * y) + 4 * s2 * y + s3 * (2 * y + 2 * x - 1.2 * (x - y))
    w4 = 4 * math.pi * R ** 2
    L = w4 * F / N
    rho = -L / N
    J = w4 * Fy / (R * N) - np.gradient(w4 * Fx / N, R)
    return rho, J


def load_slice(path):
    Z = np.load(path, allow_pickle=False)
    out = {k: Z[k] for k in Z.files if not k.startswith(("sea_", "dust_", "R2_", "e_", "mask_", "f_"))}
    out["Z"] = Z
    return out


def mode_slice(a, co):
    S = load_slice(a.slice)
    Z = S["Z"]
    R = np.asarray(S["R"], float); N = np.asarray(S["N_applied"], float); v = np.asarray(S["v_applied"], float)
    kappas = [int(k) for k in S["kappas"]]; weights = [float(w) for w in S["weights"]]
    nsites = len(R)
    acc = {k: np.zeros(nsites) for k in ("eps_R2", "eps_neg", "eps_0", "eps_N", "eps_sea", "J_R2", "J_neg", "J_0", "J_N", "J_sea")}
    per = {}
    t0 = time.time()
    for kap, w in zip(kappas, weights):
        h, Q = chain_dense(R, a.m, kap)
        Hm = clocked(h, Q, N, v)
        e, U, Uneg = neg_cols(Hm)
        e_st = torch.as_tensor(np.asarray(Z["e_%d" % kap], float))
        R2 = torch.as_tensor(Z["R2_%d" % kap]).to(CD)
        sea = torch.as_tensor(Z["sea_%d" % kap]).to(CD)
        _, _, U0 = neg_cols(h)
        HN = clocked(h, Q, N, np.zeros(nsites))
        _, _, UN = neg_cols(HN)
        ov = R2.conj().T @ Uneg
        mis = float(R2.shape[1] - (ov.abs() ** 2).sum())
        # band character of every eigenvector of h_m against the clock-only sea
        bN = ((UN.conj().T @ U).abs() ** 2).sum(0).numpy()
        neg = (e < 0).numpy()
        mixed = int(np.sum((bN > 0.05) & (bN < 0.95)))
        wrong = int(np.sum(neg & (bN < 0.5)) + np.sum(~neg & (bN > 0.5)))
        # the lower band by character: mixed eigenvectors (avoided crossings) paired with their energy neighbour and rotated to
        # diagonalize the clock-only sea in the pair; P_lb = the vectors of character > 1/2
        Ulb = U.clone(); bl = bN.copy()
        midx = list(np.flatnonzero((bN > 0.05) & (bN < 0.95)))
        used = set(); pairs_ = []
        for j in midx:
            if j in used:
                continue
            cand = [i for i in midx if i != j and i not in used]
            if not cand:
                continue
            i = min(cand, key=lambda i: abs(float(e[i] - e[j])))
            used.update([i, j]); pairs_.append((j, i, float(abs(e[i] - e[j]))))
            Vp = U[:, [j, i]]
            Mp = (UN.conj().T @ Vp); Mp = Mp.conj().T @ Mp
            wv, Wv = torch.linalg.eigh(Mp)
            Ulb[:, [j, i]] = Vp @ Wv
            bl[j], bl[i] = float(wv[0]), float(wv[1])
        lb = bl > 0.5
        Plb = Ulb[:, torch.as_tensor(lb)]
        occ_j = ((R2.conj().T @ U).abs() ** 2).sum(0).numpy()      # the production occupation of each eigenvector (0 or 1)
        n_occ_upper = int(np.sum((occ_j > 0.5) & (bN < 0.5))); n_emp_lower = int(np.sum((occ_j < 0.5) & (bN > 0.5)))
        mis_lb = float(R2.shape[1] - ((R2.conj().T @ Plb).abs() ** 2).sum())
        acc.setdefault("eps_lb", np.zeros(nsites)); acc.setdefault("J_lb", np.zeros(nsites))
        acc["eps_lb"] += w * dens(h, Plb); acc["J_lb"] += w * dens(Q, Plb)
        # commutator of R2 with h_m (R2 an invariant subspace?)
        HR = Hm @ R2
        comm = float(torch.linalg.norm(HR - R2 @ (R2.conj().T @ HR)))
        for lab, P in (("R2", R2), ("neg", Uneg), ("0", U0), ("N", UN), ("sea", sea)):
            acc["eps_" + lab] += w * dens(h, P); acc["J_" + lab] += w * dens(Q, P)
        per[kap] = {"n_sea": int(R2.shape[1]), "n_neg": int(Uneg.shape[1]), "R2_vs_neg_mismatch": mis,
                    "eig_vs_stored_max": float((e - e_st).abs().max()) if e_st.numel() == e.numel() else None,
                    "bN_mixed_count": mixed, "neg_set_vs_band_character_disagree": wrong, "min_abs_e": float(e.abs().min()),
                    "comm_h_R2_F": comm, "e_min": float(e.min()), "e_max": float(e.max()),
                    "n_lb": int(lb.sum()), "R2_vs_lb_mismatch": mis_lb, "occupied_upper_character": n_occ_upper,
                    "empty_lower_character": n_emp_lower, "mixed_pairs": len(pairs_), "mixed_pair_gaps": [p_[2] for p_ in pairs_][:20],
                    "mixed_after_pairing": int(np.sum((bl > 0.05) & (bl < 0.95))),
                    "positive_occupied": int(np.sum((occ_j > 0.5) & ~neg)), "negative_empty": int(np.sum((occ_j < 0.5) & neg))}
        print("kappa %d" % kap, json.dumps(per[kap]), "%.1fs" % (time.time() - t0), flush=True)
    rho3, J3 = local_sea_3d(R, N, v, co)
    return R, N, v, acc, per, rho3, J3


def interp_fields(R, N, v, lam):
    """fields u(r / lambda) on lambda * nsites sites R' = 1..lambda nsites (cubic spline of the applied site fields)."""
    from scipy.interpolate import CubicSpline
    Rp = np.arange(1, lam * len(R) + 1, dtype=float)
    sN = CubicSpline(np.concatenate([[0.0], R]), np.concatenate([[N[0]], N]))   # N even at the centre: extend by N(0) ~ N(R_1)
    sv = CubicSpline(np.concatenate([[0.0], R]), np.concatenate([[0.0], v]))    # v odd: v(0) = 0
    x = Rp / lam
    return Rp, sN(x), sv(x)


def mode_scale(a):
    S = load_slice(a.slice)
    R = np.asarray(S["R"], float); N = np.asarray(S["N_applied"], float); v = np.asarray(S["v_applied"], float)
    kappas = [int(k) for k in S["kappas"]]; weights = [float(w) for w in S["weights"]]
    lam = a.lam
    Rp, Np, vp = (R, N, v) if lam == 1 else interp_fields(R, N, v, lam)
    acc = {k: np.zeros(len(Rp)) for k in ("eps_pol", "J_pol")}
    t0 = time.time()
    info = {}
    for kap, w in zip(kappas, weights):
        h, Q = chain_dense(Rp, a.m, kap)
        e, U, Uneg = neg_cols(clocked(h, Q, Np, vp))
        _, _, U0 = neg_cols(h)
        acc["eps_pol"] += w * (dens(h, Uneg) - dens(h, U0)); acc["J_pol"] += w * (dens(Q, Uneg) - dens(Q, U0))
        info[kap] = {"min_abs_e": float(e.abs().min()), "n_neg": int(Uneg.shape[1]), "n_half": len(Rp)}
        print("lambda %d kappa %d" % (lam, kap), json.dumps(info[kap]), "%.1fs" % (time.time() - t0), flush=True)
        del h, Q, e, U, Uneg, U0
    return Rp, Np, vp, acc, info


def mode_rates(a, co):
    """records in [T0, T1] of one history: eps_R2, J_R2 from the applied fields (R2 = P_neg where the json says so)."""
    J = json.load(open(a.json))
    recs = J["records"]
    F = np.load(a.fields, allow_pickle=True)
    Ts = np.asarray(F["T"], float)
    rs = float(J["args"]["rs"])
    H = np.load(a.handoff, allow_pickle=True) if a.handoff else None
    R = np.asarray(F["R"], float)
    out = []
    t0 = time.time()
    hcache = {}
    for k, T in enumerate(Ts):
        Tr = T / rs
        if not (a.T0 - 1e-9 <= Tr <= a.T1 + 1e-9):
            continue
        rec = [x for x in recs if abs(x["T"] - T) < 1e-9]
        rec = rec[0] if rec else None
        N = np.asarray(F["N_applied"][k], float); v = np.asarray(F["v_applied"][k], float)
        eps = np.zeros(len(R)); Jr = np.zeros(len(R)); mism = {}
        for kap in range(1, a.K + 1):
            if kap not in hcache:
                hcache[kap] = chain_dense(R, a.m, float(kap))
            h, Q = hcache[kap]
            e, U, Uneg = neg_cols(clocked(h, Q, N, v))
            eps += 16.0 * kap * dens(h, Uneg); Jr += 16.0 * kap * dens(Q, Uneg)
            mism[kap] = float(e.abs().min())
        row = {"k": k, "T_over_rs": Tr, "mask_equals_negative_set": rec.get("mask_equals_negative_set") if rec else None,
               "R_t": rec.get("R_t") if rec else None, "min_abs_e": min(mism.values())}
        if H is not None:
            Th = np.asarray(H["T"], float)
            kk = np.flatnonzero(np.abs(Th - T) < 1e-9)
            if len(kk):
                eh = np.asarray(H["eps_R2"][kk[0]], float)
                row["handoff_eps_R2_maxdiff"] = float(np.max(np.abs(eh - eps)))
                row["handoff_eps_R2_maxabs"] = float(np.max(np.abs(eh)))
        out.append({"row": row, "eps": eps, "J": Jr})
        print(json.dumps(row), "%.1fs" % (time.time() - t0), flush=True)
    return R, out, F, recs, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["slice", "scale", "rates"])
    ap.add_argument("--slice")
    ap.add_argument("--lam", type=int, default=1)
    ap.add_argument("--json"); ap.add_argument("--fields"); ap.add_argument("--handoff")
    ap.add_argument("--T0", type=float, default=0.0); ap.add_argument("--T1", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--coef", default="", help="s1,s2,s3 of the 3D local sea (ls_zone_v1)")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    co = dict(zip(("s1", "s2", "s3"), [float(x) for x in a.coef.split(",")])) if a.coef else None
    os.makedirs(a.out, exist_ok=True)
    meta = {"script": "ls_chain_audit_v1.py", "mode": a.mode, "m": a.m, "coef": co, "threads": NT}
    t0 = time.time()
    if a.mode == "slice":
        meta["slice"] = a.slice; meta["slice_md5"] = md5(a.slice) if os.path.getsize(a.slice) < (4 << 30) else None
        R, N, v, acc, per, rho3, J3 = mode_slice(a, co)
        np.savez_compressed(os.path.join(a.out, "ls_chain_audit_v1_%s.npz" % a.tag), R=R, N=N, v=v, rho3d=rho3, J3d=J3, **acc)
        meta["per_channel"] = per
    elif a.mode == "scale":
        meta["slice"] = a.slice; meta["lam"] = a.lam
        Rp, Np, vp, acc, info = mode_scale(a)
        np.savez_compressed(os.path.join(a.out, "ls_chain_audit_v1_%s.npz" % a.tag), R=Rp, N=Np, v=vp, **acc)
        meta["per_channel"] = info
    else:
        meta.update({"json": a.json, "fields": a.fields, "handoff": a.handoff, "T0": a.T0, "T1": a.T1,
                     "json_md5": md5(a.json), "fields_md5": md5(a.fields)})
        R, out, F, recs, rs = mode_rates(a, co)
        np.savez_compressed(os.path.join(a.out, "ls_chain_audit_v1_%s.npz" % a.tag), R=R,
                            k=np.array([o["row"]["k"] for o in out]), T_over_rs=np.array([o["row"]["T_over_rs"] for o in out]),
                            eps_R2=np.array([o["eps"] for o in out]), J_R2=np.array([o["J"] for o in out]))
        meta["rows"] = [o["row"] for o in out]
    meta["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "ls_chain_audit_v1_%s.json" % a.tag), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print("DONE %.1fs" % meta["seconds"])


if __name__ == "__main__":
    main()
