#!/usr/bin/env python
"""Frozen native action, lines 7-10 of integ65 with the integ69_nativedefs2 promotions (data script).

Job kinds (one JSON per job in <out>/jobs/):
  main      ring4 / plaq; promotion x subtraction x charge sector q = 0 (vacuum), +1, +2 x m x U.  k = 1: E(P) on the twist
            grid (regular-branch continuation at P = 0, warm Newton elsewhere), exact b-response dH/db at every P
            (mixed vertex), M'_b, R = H + dH/db (the sigma response equals the b response at fixed S; finite
            differences in b with steps 1e-3, 1e-4 as the step check), mass lock P_E R P_E on every energy block,
            adjoint multiplier and Ward residuals (two generators, with and without feedback), sigma_min(J), and
            for END the literal right-placement operator against the endpoint operator.  k = 10, 100, 1e4 at five
            P for F_0, F_2.  E_free(P) = lowest level of sum_x e_x(I) (the k -> infinity limit).
  cube      Ward and R checks on the open 2x2x2 cube (no beta: mass m times the identity, empty reference).
  register  ring4 with a two-mode register (one reading particle, reading energy eps_r n_f1 in e_0), solved on
            the whole fixed-number sector; leakage of S, e(S), H out of Pi without any reprojection, after a
            random perturbation of the converged rulers is removed by Newton.
  pinned    no hopping, U = 0: the one-site pinned formulas.
Momentum convention: P = theta / L_x is the twist momentum per particle and P_tot = q P; I from
E = E_0 + P_tot^2/(2 I) + ... so that free particles give I = q m.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import socket
import time
import traceback
from multiprocessing import Pool

import numpy as np

import native_config as cfg
import native_core as nc
import native_sources as ns
from native_block import BlockBridge

K_CACHE = {}
P_K = (-0.05, -0.025, 0.0, 0.025, 0.05)
P_LIST = sorted({round(float(p), 12) + 0.0 for p in cfg.P_GRID})
EPS_R = 1e-3


def K_of(latname):
    if latname not in K_CACHE:
        K_CACHE[latname] = nc.K_matrix(nc.lattice(latname)["positions"])
    return K_CACHE[latname]


def twist(latname, P):
    return cfg.twist_for(latname, P) if latname != "cube" else (0.0, 0.0, 0.0)


def build(latname, promotion, subtraction, q, m, U, P, k, register=False, nohop=False):
    lat = nc.lattice(latname, twist(latname, P))
    if nohop:
        lat["D"] = np.zeros_like(lat["D"])
    nm = lat["n"] * lat["ncomp"] + (cfg.REGISTER_READINGS if register else 0)
    fock = ns.Fock(nm)
    G, nsea = ns.sea_contractions(lat, m, nm)
    Ntop = nsea + q + (1 if register else 0)
    K, Gm = K_of(latname)
    extra = {0: (lambda sec: EPS_R * sec.number(nm - 1))} if register else None
    if promotion == "FI":
        cos = {N: ns.Coefficients(lat, fock, N, m, U, G, subtraction, extra) for N in range(Ntop + 1)}
        exprs, blocks, dims = ns.fi_chain_exprs(lat, fock, Ntop, cos)
        co = cos[Ntop]
    else:
        co = ns.Coefficients(lat, fock, Ntop, m, U, G, subtraction, extra)
        exprs = ns.single_sector_exprs(promotion, co, lat["n"])
        blocks, dims = [(x, Ntop) for x in range(lat["n"])], {Ntop: co.d}
    br = BlockBridge(exprs, blocks, dims, K, k, Ntop)
    return {"lat": lat, "fock": fock, "G": G, "nsea": nsea, "Ntop": Ntop, "co": co, "br": br, "Gm": Gm}


def lowest_block(H, tol=1e-9):
    w, V, blocks = nc.energy_blocks(H, tol)
    bl = blocks[0]
    return w, V, blocks, float(w[bl[0]]), len(bl)


def block_response(V, bl, dH):
    ev = np.linalg.eigvalsh(nc.herm(V[:, bl].conj().T @ dH @ V[:, bl]))
    return float(ev.mean()), float(ev.max() - ev.min())


def free_level(prob):
    co = prob["co"]
    Hm = sum(co.scalar_source(x, np.ones(prob["lat"]["n"])) for x in range(prob["lat"]["n"]))
    w = np.linalg.eigvalsh(nc.herm(Hm))
    return float(w[0]), nc.herm(Hm)


def generators(prob, seed=11):
    """Two number-preserving Hermitian generators per sector: a random one-body and the site-position one-body."""
    lat = prob["lat"]
    nm = prob["fock"].nmodes
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(nm, nm)) + 1j * rng.normal(size=(nm, nm))
    h = nc.herm(h)
    pos = np.zeros((nm, nm))
    for i in range(lat["n"] * lat["ncomp"]):
        x = lat["site_of_mode"][i]
        p = lat["positions"][x]
        pos[i, i] = p[0] + 0.31 * p[1] + 0.17 * p[2]
    gens = {}
    for name, hh in (("random_one_body", h), ("position", pos)):
        gens[name] = {s: prob["fock"].sector(s).bilinear(hh) for s in prob["br"].sectors}
    return gens


def summarize_info(info):
    out = {"ok": bool(info.get("ok", False))}
    if "path" in info:
        out["lam_steps"] = len(info["path"]) - 1
        out["min_eig_path"] = float(min(p["min_eig"] for p in info["path"]))
        if not info.get("ok"):
            out["lost_at_lam"] = info.get("lost_at_lam")
            out["why"] = info.get("why")
    if "residuals" in info and info["residuals"]:
        out["final_residual"] = float(info["residuals"][-1])
    elif "path" in info:
        out["final_residual"] = float(info["path"][-1]["residual"])
    return out


def solve_at(prob, S_warm=None, b=1.0):
    br = prob["br"]
    if S_warm is not None:
        S, info = br.newton(S_warm, b=b)
        if info["ok"]:
            return S, info
    return br.continuation(b=b)


def mass_lock(H, R, tol=1e-9):
    worst, per = nc.mass_lock_test(H, R, tol)
    return worst, per[0]["max_abs"], per[0]["deg"]


# ------------------------------------------------------------------ main job
def job_main(spec):
    latname, prom, sub, q, m, U = spec["lat"], spec["prom"], spec["sub"], spec["q"], spec["m"], spec["U"]
    if prom == "PIN":
        return job_pin(spec)
    t0 = time.time()
    out = {"spec": spec, "kind": "main"}
    Ps = sorted(P_LIST, key=lambda p: (abs(p), p))
    disp = {}
    S0 = None
    warm = {}
    for P in Ps:
        prob = build(latname, prom, sub, q, m, U, P, 1.0)
        br = prob["br"]
        start = None if P == 0 else warm.get(np.sign(P), S0)
        S, info = solve_at(prob, start)
        if not info.get("ok", False):
            disp[P] = {"info": summarize_info(info)}
            if P == 0:
                out["base_lost"] = summarize_info(info)
                out["seconds"] = time.time() - t0
                return out
            continue
        warm[np.sign(P)] = S
        if P == 0:
            S0 = S
        H = br.Q(S)
        w, V, blocks, E0, deg = lowest_block(H)
        dH, res_lr, code_lr = br.dH_db(S)
        dEb, dEb_spread = block_response(V, blocks[0], dH)
        Ef, _ = free_level(prob)
        disp[P] = {"E": E0, "deg": deg, "dE_db": dEb, "dE_db_spread": dEb_spread, "E_free": Ef,
                   "levels": [float(v) for v in w[:6]], "lr_residual": res_lr, "lr_code": code_lr,
                   "info": summarize_info(info)}
        if P == 0.025:
            # Ward identity at a twisted point as well (the plaquette has no hopping at P = 0)
            bl = blocks[0]
            rho = V[:, bl] @ V[:, bl].conj().T / len(bl)
            Lam, res_adj, _ = br.adjoint_multiplier(S, rho)
            wd = {"adjoint_residual": res_adj}
            for gname, Dg in generators(prob).items():
                full, nofb = br.ward(S, Lam, Dg)
                wd[gname] = {"full": abs(full), "no_feedback": abs(nofb)}
            disp[P]["ward"] = wd
            R = H + dH
            disp[P]["mass_lock_worst"], disp[P]["mass_lock_lowest"], _ = mass_lock(H, R)
        if P == 0:
            base = {"prob": prob, "S": S, "H": H, "dH": dH, "V": V, "blocks": blocks}
    out["dispersion"] = {repr(float(P)): v for P, v in disp.items()}
    # ---- base point extras (line 9 and controls)
    prob, S, H, dH = base["prob"], base["S"], base["H"], base["dH"]
    br = prob["br"]
    R = H + dH
    worst, low, ldeg = mass_lock(H, R)
    fd = {}
    for h in cfg.B_STEPS:
        Hb = {}
        for sgn in (1, -1):
            bb = 1 + sgn * h
            Sb, ib = br.newton(S, b=bb)
            Hb[sgn] = (bb * br.Q(Sb, b=bb), br.Q(Sb, b=bb), ib["ok"])
        R_fd = (Hb[1][0] - Hb[-1][0]) / (2 * h)
        dH_fd = (Hb[1][1] - Hb[-1][1]) / (2 * h)
        V0, bl0 = base["V"], base["blocks"][0]
        fd[repr(h)] = {"R_fd_minus_R_exact": float(np.abs(R_fd - R).max()),
                       "dH_fd_minus_exact": float(np.abs(dH_fd - dH).max()),
                       "Mb_fd": block_response(V0, bl0, dH_fd)[0], "ok": bool(Hb[1][2] and Hb[-1][2])}
    V0, bl0 = base["V"], base["blocks"][0]
    rho = V0[:, bl0] @ V0[:, bl0].conj().T / len(bl0)
    Lam, res_adj, code_adj = br.adjoint_multiplier(S, rho)
    ward = {}
    for gname, Dg in generators(prob).items():
        full, nofb = br.ward(S, Lam, Dg)
        ward[gname] = {"full": abs(full), "no_feedback": abs(nofb)}
    try:
        smin = br.sigma_min(S)
    except Exception as exc:  # noqa: BLE001
        smin = float("nan")
        out["sigma_min_error"] = repr(exc)
    out["base"] = {"mass_lock_worst": worst, "mass_lock_lowest": low, "lowest_deg": ldeg,
                   "R_lowest_mean": block_response(V0, bl0, R)[0], "fd": fd, "adjoint_residual": res_adj,
                   "adjoint_code": code_adj, "ward": ward, "sigma_min_J": smin,
                   "min_eig_S": float(min(np.linalg.eigvalsh(s).min() for s in S)),
                   "sea_size": prob["nsea"], "Ntop": prob["Ntop"], "dim": int(prob["co"].d)}
    if prom == "END":
        top = [S[br.index[(x, prob["Ntop"])]] for x in range(prob["lat"]["n"])]
        er = ns.right_placement_e(prob["lat"], prob["fock"], prob["Ntop"], prob["co"], prob["G"], U, sub, top)
        ev = br.source_values(S)
        out["base"]["right_placement_vs_endpoint"] = float(max(np.abs(a - b).max() for a, b in zip(er, ev)))
    # ---- k scan for F_0, F_2
    kscan = {}
    for k in cfg.K_FREE_LIMIT[1:]:
        Sk0 = None
        warmk = {}
        rows = {}
        for P in sorted(P_K, key=lambda p: (abs(p), p)):
            pk = build(latname, prom, sub, q, m, U, P, k)
            start = None if P == 0 else warmk.get(np.sign(P), Sk0)
            Sk, ik = solve_at(pk, start)
            if not ik.get("ok", False):
                rows[repr(float(P))] = {"info": summarize_info(ik)}
                continue
            warmk[np.sign(P)] = Sk
            if P == 0:
                Sk0 = Sk
            _, Vk, blk, Ek, dk = lowest_block(pk["br"].Q(Sk))
            rows[repr(float(P))] = {"E": Ek, "deg": dk, "E_free": free_level(pk)[0], "info": summarize_info(ik)}
            if P == 0 and k == max(cfg.K_FREE_LIMIT):
                dHk, _, _ = pk["br"].dH_db(Sk)
                rows[repr(float(P))]["dE_db"] = block_response(Vk, blk[0], dHk)[0]
        kscan[repr(k)] = rows
    out["kscan"] = kscan
    out["derived"] = derive(out, q)
    out["seconds"] = time.time() - t0
    return out


def derive(out, q):
    disp = out["dispersion"]
    Ps = sorted(float(p) for p, v in disp.items() if "E" in v)
    E = np.array([disp[repr(p)]["E"] for p in Ps])
    Ef = np.array([disp[repr(p)]["E_free"] for p in Ps])
    dEb = np.array([disp[repr(p)]["dE_db"] for p in Ps])
    qe = q if q != 0 else 1  # q = 0 (vacuum sector): curvature in the twist momentum itself
    Pt = qe * np.array(Ps)
    d = {}
    if len(Ps) >= 5:
        E0, I, c4 = nc.dispersion_fit(Pt, E, order=min(8, 2 * (len(Ps) // 2 - 1) if len(Ps) < 9 else 8))
        _, If, _ = nc.dispersion_fit(Pt, Ef, order=8 if len(Ps) >= 9 else 4)
        A = np.stack([np.ones_like(Pt), Pt ** 2, Pt ** 4], axis=1)
        cm, *_ = np.linalg.lstsq(A, dEb, rcond=None)
        M = disp[repr(0.0)]["E"]
        Mb = disp[repr(0.0)]["dE_db"]
        sdiv = lambda a, b: a / b if b != 0 and np.isfinite(b) else float("nan")
        d.update(M=M, I=I, I_free=If, E_free0=disp[repr(0.0)]["E_free"], Mb=Mb, q_charge=(M - Mb) / 2,
                 EP_ratio=sdiv(M - Mb, 2 * I), I_over_M_minus_1=sdiv(I, M) - 1, mixed_vertex_P2=float(cm[1]),
                 mixed_vertex_P0=float(cm[0]), c4=c4)
    F = {}
    for kk, rows in out.get("kscan", {}).items():
        k = float(kk)
        pts = [(float(p), v) for p, v in rows.items() if "E" in v]
        if len(pts) >= 3:
            Pk = qe * np.array([p for p, _ in pts])
            y = k * np.array([v["E"] - v["E_free"] for _, v in pts])
            A = np.stack([np.ones_like(Pk), Pk ** 2, Pk ** 4], axis=1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            F[kk] = {"F0": float(c[0]), "F2": float(c[1])}
    d["F"] = F
    if F and "M" in d:
        kbig = max(F, key=float)
        F0, F2 = F[kbig]["F0"], F[kbig]["F2"]
        d.update(F0=F0, F2=F2, M_pred=d["E_free0"] + F0, I_pred=d["I_free"] - 2 * d["I_free"] ** 2 * F2,
                 M_minus_pred=d["M"] - (d["E_free0"] + F0), I_minus_pred=d["I"] - (d["I_free"] - 2 * d["I_free"] ** 2 * F2))
    return d


# ------------------------------------------------------------------ pinching
def scalar_bridge(src, K, k, b=1.0, mu_steps=None):
    """mu e_x(s) = k s_x [K(s - b)]_x for a scalar source e_x(s) = src(s) (vector), continuation mu: 0 -> 1."""
    n = K.shape[0]
    s = b * np.ones(n)
    for mu in (mu_steps or np.linspace(0.1, 1.0, 10)):
        for _ in range(60):
            F = mu * src(s) - k * s * (K @ (s - b))
            if np.max(np.abs(F)) < 1e-14:
                break
            h = 1e-7
            Jm = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = h
                Jm[:, j] = mu * (src(s + e) - src(s - e)) / (2 * h)
            Jm -= k * (np.diag(K @ (s - b)) + np.diag(s) @ K)
            ds = np.linalg.solve(Jm, -F)
            t = 1.0
            while np.any(s + t * ds <= 0):
                t *= 0.5
            s = s + t * ds
    F = src(s) - k * s * (K @ (s - b))
    return s, float(np.max(np.abs(F)))


def pin_states(prob, ores=None):
    lat, co = prob["lat"], prob["co"]
    _, Hm = free_level(prob)
    Ores = ns.resolving_operator(lat, prob["fock"].sector(prob["Ntop"])) if ores is None else ores
    Vb = ns.pinning_basis(Hm, Ores)
    n = lat["n"]
    exp = lambda A: np.real(np.einsum('ia,ij,ja->a', Vb.conj(), A, Vb))
    mx = np.array([exp(co.M[x]) for x in range(n)])
    vx = np.array([exp(co.V[x]) for x in range(n)])
    tb = {bd: exp(co.B[bd] + co.B[bd].conj().T) for bd in co.B}
    return mx, vx, tb


def pin_energies(prob, k, b=1.0, nstates=None, ores=None):
    K, _ = K_of(prob["lat"]["name"])
    mx, vx, tb = pin_states(prob, ores)
    n = mx.shape[0]
    ns_ = mx.shape[1] if nstates is None else nstates
    out = []
    for a in range(ns_):
        def src(s, a=a):
            e = mx[:, a].copy()
            for (i, j), t in tb.items():
                e[i] += 0.5 * t[a] / (s[i] * s[j])
                e[j] += 0.5 * t[a] / (s[i] * s[j])
            return e + s ** -6 * vx[:, a]
        s, res = scalar_bridge(src, K, k, b)
        E = k * np.sum(K @ (s - b))
        # b response: J ds = -k b_x s_x
        h = 1e-7
        Jm = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Jm[:, j] = (src(s + e) - src(s - e)) / (2 * h)
        Jm -= k * (np.diag(K @ (s - b)) + np.diag(s) @ K)
        ds = np.linalg.solve(Jm, -k * (K @ np.ones(n)) * s)
        dE = -k * np.sum(K @ np.ones(n)) + k * np.sum(K @ ds)
        out.append({"E": float(E), "dE_db": float(dE), "res": res, "min_s": float(s.min()),
                    "sigma_min": float(np.linalg.svd(Jm, compute_uv=False)[-1])})
    return out


def job_pin(spec):
    latname, sub, q, m, U = spec["lat"], spec["sub"], spec["q"], spec["m"], spec["U"]
    t0 = time.time()
    out = {"spec": spec, "kind": "main"}
    disp = {}
    for P in P_LIST:
        prob = build(latname, "J_W6", sub, q, m, U, P, 1.0)
        st = pin_energies(prob, 1.0)
        Es = np.array([r["E"] for r in st])
        a0 = int(np.argmin(Es))
        deg = int(np.sum(Es < Es[a0] + 1e-9))
        low = [r for r in st if r["E"] < Es[a0] + 1e-9]
        disp[float(P)] = {"E": float(Es[a0]), "deg": deg, "dE_db": float(np.mean([r["dE_db"] for r in low])),
                          "dE_db_spread": float(np.ptp([r["dE_db"] for r in low])), "E_free": free_level(prob)[0],
                          "levels": [float(v) for v in np.sort(Es)[:6]],
                          "info": {"ok": bool(max(r["res"] for r in st) < 1e-12), "final_residual": max(r["res"] for r in st),
                                   "min_eig_path": min(r["min_s"] for r in st)}}
        if P == 0:
            Rdiag = np.array([r["E"] + r["dE_db"] for r in st])
            out["base"] = {"mass_lock_worst": float(np.abs(Rdiag).max()),
                           "mass_lock_lowest": float(np.abs(Rdiag[Es < Es[a0] + 1e-9]).max()),
                           "lowest_deg": deg, "R_lowest_mean": float(np.mean(Rdiag[Es < Es[a0] + 1e-9])),
                           "sigma_min_J": float(min(r["sigma_min"] for r in st)), "ward": "not applicable (diagonal)",
                           "min_eig_S": float(min(r["min_s"] for r in st)), "dim": int(len(st))}
    out["dispersion"] = {repr(float(P)): v for P, v in disp.items()}
    kscan = {}
    for k in cfg.K_FREE_LIMIT[1:]:
        rows = {}
        for P in P_K:
            prob = build(latname, "J_W6", sub, q, m, U, P, k)
            st = pin_energies(prob, k)
            a0 = int(np.argmin([r["E"] for r in st]))
            rows[repr(float(P))] = {"E": float(st[a0]["E"]), "E_free": free_level(prob)[0], "info": {"ok": True}}
            if P == 0:
                Emin = st[a0]["E"]
                rows[repr(float(P))]["dE_db"] = float(np.mean([r["dE_db"] for r in st if r["E"] < Emin + 1e-9]))
        kscan[repr(k)] = rows
    out["kscan"] = kscan
    out["derived"] = derive(out, q)
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ cube, register, pinned
def job_cube(spec):
    t0 = time.time()
    prob = build("cube", spec["prom"], "wick", spec["N"], spec["m"], spec["U"], 0.0, spec.get("k", 1.0))
    br = prob["br"]
    S, info = br.continuation()
    out = {"spec": spec, "kind": "cube", "info": summarize_info(info), "dim": int(prob["co"].d)}
    if not info.get("ok"):
        # where the branch is lost: smallest singular value of the Jacobian at the last accepted lam
        lam_last = info["path"][-1]["lam"]
        try:
            out["sigma_min_at_last_lam"] = br.sigma_min(S, lam=lam_last)
        except Exception as exc:  # noqa: BLE001
            out["sigma_min_error"] = repr(exc)
        out["last_lam"] = lam_last
        out["sigma_min_at_lam0"] = br.sigma_min(br.identity_start(), lam=0.0) if prob["co"].d <= 16 else None
        out["seconds"] = time.time() - t0
        return out
    H = br.Q(S)
    dH, res_lr, _ = br.dH_db(S)
    R = H + dH
    worst, low, ldeg = mass_lock(H, R)
    w, V, blocks = nc.energy_blocks(H)
    bl0 = blocks[0]
    rho = V[:, bl0] @ V[:, bl0].conj().T / len(bl0)
    Lam, res_adj, code = br.adjoint_multiplier(S, rho)
    # also a pure state inside the (highly degenerate) lowest block: the block average can hide the control
    v0 = V[:, bl0[0]]
    Lam1, res_adj1, _ = br.adjoint_multiplier(S, np.outer(v0, v0.conj()))
    ward = {}
    gens = generators(prob)
    for gname, Dg in gens.items():
        full, nofb = br.ward(S, Lam, Dg)
        full1, nofb1 = br.ward(S, Lam1, Dg)
        ward[gname] = {"full": abs(full), "no_feedback": abs(nofb), "pure_full": abs(full1), "pure_no_feedback": abs(nofb1)}
    h = 1e-4
    Hs = {}
    for sgn in (1, -1):
        Sb, _ = br.newton(S, b=1 + sgn * h)
        Hs[sgn] = (1 + sgn * h) * br.Q(Sb, b=1 + sgn * h)
    R_fd = (Hs[1] - Hs[-1]) / (2 * h)
    out.update(E0=float(w[0]), lowest_deg=len(bl0), mass_lock_worst=worst, mass_lock_lowest=low,
               R_lowest_mean=block_response(V, bl0, R)[0], R_fd_minus_exact=float(np.abs(R_fd - R).max()),
               adjoint_residual=res_adj, lr_residual=res_lr, ward=ward,
               min_eig_S=float(min(np.linalg.eigvalsh(s).min() for s in S)), seconds=time.time() - t0)
    return out


def job_register(spec):
    t0 = time.time()
    prob = build("ring4", spec["prom"], spec["sub"], 1, spec["m"], spec["U"], 0.0, 1.0, register=True)
    br = prob["br"]
    S, info = br.continuation()
    out = {"spec": spec, "kind": "register", "info": summarize_info(info), "dim": int(prob["co"].d)}
    if not info.get("ok"):
        out["seconds"] = time.time() - t0
        return out
    nm = prob["fock"].nmodes
    occ = prob["fock"].sector(prob["Ntop"]).occupations()
    Pi = np.diag((occ[:, nm - 2] + occ[:, nm - 1] == 1).astype(float))
    Qm = np.eye(len(Pi)) - Pi
    rng = np.random.default_rng(5)
    Sp = [nc.herm(s + 1e-3 * (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape))) for s in S]
    leak0 = max(np.linalg.norm(Qm @ s @ Pi, 2) for s in Sp)
    S2, info2 = br.newton(Sp)
    ev = br.source_values(S2)
    H = br.Q(S2)
    out.update(perturbed_leak_S=float(leak0), newton_after_perturbation=summarize_info(info2),
               leak_S=float(max(np.linalg.norm(Qm @ s @ Pi, 2) for s in S2)),
               leak_e=float(max(np.linalg.norm(Qm @ e @ Pi, 2) for e in ev)),
               leak_H=float(np.linalg.norm(Qm @ H @ Pi, 2)),
               rulers_moved=float(max(np.abs(a - b).max() for a, b in zip(S, S2))),
               E_reading=[float(np.linalg.eigvalsh(nc.herm(H[np.ix_(idx, idx)]))[0])
                          for idx in [np.where((occ[:, nm - 2] == 1) & (occ[:, nm - 1] == 0))[0],
                                      np.where((occ[:, nm - 2] == 0) & (occ[:, nm - 1] == 1))[0]]],
               seconds=time.time() - t0)
    return out


def job_pinned(spec):
    """No hopping, U = 0, q = +1: M = (k/2g)(sqrt(1 + 4gm/k) - 1), q = m/sqrt(1 + 4gm/k)."""
    t0 = time.time()
    k, m = spec["k"], spec["m"]
    prob = build(spec["lat"], spec["prom"], "wick", 1, m, 0.0, 0.0, k, nohop=True)
    br = prob["br"]
    S, info = br.continuation()
    H = br.Q(S)
    w, V, blocks, E0, deg = lowest_block(H)
    dH, _, _ = br.dH_db(S)
    Mb = block_response(V, blocks[0], dH)[0]
    g = prob["Gm"][0, 0]
    M_or = k / (2 * g) * (np.sqrt(1 + 4 * g * m / k) - 1)
    q_or = m / np.sqrt(1 + 4 * g * m / k)
    return {"spec": spec, "kind": "pinned", "M": E0, "M_formula": M_or, "q": (E0 - Mb) / 2, "q_formula": q_or,
            "info": summarize_info(info), "seconds": time.time() - t0}


def run_job(spec):
    try:
        fn = {"main": job_main, "cube": job_cube, "register": job_register, "pinned": job_pinned}[spec["kind"]]
        res = fn(spec)
    except Exception:  # noqa: BLE001
        res = {"spec": spec, "kind": spec["kind"], "error": traceback.format_exc()}
    return res


def job_name(spec):
    return "_".join("%s%s" % (k, v) for k, v in sorted(spec.items())).replace("/", "")


def all_jobs(which):
    jobs = []
    if "main" in which:
        for lat in ("ring4", "plaq"):
            for prom in ns.PROMOTIONS:
                for sub in ns.SUBTRACTIONS:
                    for q in (0, 1, 2):
                        for m in cfg.MASSES.values():
                            for U in cfg.U_VALUES:
                                jobs.append({"kind": "main", "lat": lat, "prom": prom, "sub": sub, "q": q, "m": m, "U": U})
    if "cube" in which:
        for prom in ("J_W6", "J_CVC", "END", "SJ", "FI"):
            for N in (1, 2):
                for k in (1.0, 10.0):
                    jobs.append({"kind": "cube", "prom": prom, "N": N, "m": 0.1, "U": 0.1, "k": k})
    if "register" in which:
        for prom in ("J_W6", "J_CVC", "END", "SJ"):
            for sub in ns.SUBTRACTIONS:
                jobs.append({"kind": "register", "prom": prom, "sub": sub, "m": 0.1, "U": 0.1})
    if "pinned" in which:
        for prom in ("J_W6", "END", "FI"):
            for k in (1.0, 10.0):
                jobs.append({"kind": "pinned", "lat": "ring4", "prom": prom, "k": k, "m": 0.1})
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--which", default="pinned,main,cube,register")
    ap.add_argument("--filter", default="", help="comma list key=value restricting jobs")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "jobs"), exist_ok=True)
    jobs = all_jobs(args.which.split(","))
    for kv in filter(None, args.filter.split(",")):
        k, v = kv.split("=")
        jobs = [j for j in jobs if str(j.get(k)) == v]
    if args.skip_done:
        jobs = [j for j in jobs if not os.path.exists(os.path.join(args.out, "jobs", job_name(j) + ".json"))]
    # expensive first: FI, q = 1, registers
    cost = lambda j: (j["kind"] == "register") * 10 + (j.get("prom") == "FI") * 5 + (j.get("q") == 1) * 2 + (j["kind"] == "cube") * 3
    jobs.sort(key=lambda j: -cost(j))
    t0 = time.time()
    print("native run: %d jobs, %d workers, host %s" % (len(jobs), args.workers, socket.gethostname()), flush=True)
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            with open(os.path.join(args.out, "jobs", job_name(res["spec"]) + ".json"), "w") as fh:
                json.dump(res, fh, indent=1, default=float)
            print("  %d/%d %s %s %.0f s" % (i, len(jobs), job_name(res["spec"]), "ERROR" if "error" in res else "",
                                            time.time() - t0), flush=True)
    print("native run: done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
