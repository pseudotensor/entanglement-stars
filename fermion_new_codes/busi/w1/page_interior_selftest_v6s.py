#!/usr/bin/env python
"""page_interior_selftest_v6s.py (busi lane, inbox #309): known-answer controls for page_history_v6s.py's three options and page_interior_v6s.py,
on small chains on the host (CPU, float64/complex128). Usage: python page_interior_selftest_v6s.py [--b2-record J --b2-fields F] [--out json]
Stages (each PASS/FAIL with its numbers):
 S1 defaults: ChannelOps(kappa) builds chain_sparse(R, 0, float(kappa)) bit for bit; IR.mass_view returns the channel itself.
 S2 kappa = 0 operator: no sigma_y entry; the sector symmetry [S_stag, h] = 0 and the pairing h + conj h = 0 hold (operator_audit); the flat chain's
    h_+ has no zero level and n/2 occupied levels; the extra parity P = diag((-1)^j sigma_z) commutes with h at kappa = 0 (the + and - sectors carry
    the same spectrum, spec h_+ = -spec h_+), and anticommutes with S_stag.
 S3 the contour block against the dense P_{-,+}: max entry difference and the interior entropies, kappa 1 and 0, flat and trapped fields, two N_I.
 S4 the purity identity on a propagated state: S_I(C) = S_E(C) within the floor of the measured impurity, and S_I(P_-) = S_E(P_-) for the dense
    vacuum (kappa 1 and 0, the production's sub-steps on time-dependent trapped fields).
 S5 the held step: one Chebyshev series over 64 units against 64 unit steps on a fixed h_+ (max column difference).
 S6 the hold: History extended by copies after the hold record (fields equal, spacing tau, E held).
 S7 kappa = 0 centre: the flat vacuum's interior entropy of the + sector against the boundary CFT law (1/6) ln(2 l) + const per Dirac fermion:
    the slope per doubling of l against ln(2)/6 = 0.11552 (Calabrese-Cardy, c = 1, one boundary).
 S8 (with --b2-record/--b2-fields) B2's fields at T = 0, 50 and 100 r_s on 8192 sites, kappa 1 and 0: the operator audit, the zero levels of h_+
    (|e| < 1e-12), min |e|, the occupied count n_+ against n/2.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_history_v6s as V6S          # noqa: E402
import page_interior_v6s as IR          # noqa: E402
import page_ops_v6 as OPS               # noqa: E402
import page_w1_sector_v1d as W1S        # noqa: E402

CD = OPS.CD
DEV = torch.device("cpu")
RES = {}


def rec(name, ok, **kw):
    RES[name] = dict(kw, ok=bool(ok))
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, ", ".join("%s %s" % (k, ("%.3e" % v) if isinstance(v, float) else v) for k, v in kw.items())), flush=True)


def fields(R, t, trapped=True, g=0.25, E=60.0):
    R = np.asarray(R, float)
    N = 1.0 - (0.5 if trapped else 0.0) * np.exp(-((R - 40.0) / 25.0) ** 2) * (1.0 + 0.1 * math.sin(0.05 * t))
    v = -np.sqrt(2.0 * g * E * (1.0 + 0.05 * math.sin(0.03 * t)) / R) * (1.0 if trapped else 0.0)
    return N, v


def ent(ev):
    return float(OPS.binary_entropy_t(ev).sum()) if ev.numel() else 0.0


def dense_plus(hp):
    e, V = torch.linalg.eigh(torch.as_tensor(hp.toarray(), dtype=CD))
    return e, V


def s1():
    R = np.arange(1, 301, dtype=float)
    ok = True
    for k in (1, 2, 5):
        ops = V6S.ChannelOps(k, R, DEV)
        h0, Q = OPS.chain_sparse(R, 0.0, float(k))
        ok &= (ops.h0 != h0).nnz == 0 and (ops.Q != Q).nnz == 0 and IR.mass_view(ops) is ops and ops.kappa_mass == float(k)
    rec("S1 defaults bitwise", ok)


def s2():
    n = 400; R = np.arange(1, n + 1, dtype=float)
    ops = V6S.ChannelOps(1, R, DEV, kappa_zero=True)
    h0d = ops.h0.toarray()
    no_sy = all(h0d[2 * j, 2 * j + 1] == 0 for j in range(n))
    N, v = fields(R, 0.0)
    h = ops.h_csr(N, v)
    aud = W1S.operator_audit(h)
    hf = ops.h_csr(np.ones(n), np.zeros(n)); hpf = W1S.sector_plus_csr(hf)
    ef = np.linalg.eigvalsh(hpf.toarray())
    P = sp.diags(np.repeat(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), 2) * np.tile([1.0, -1.0], n))
    S = W1S.stag_sparse(n)
    comm = abs(P @ h - h @ P).max(); acomm = abs(P @ S + S @ P).max()
    hp = W1S.sector_plus_csr(h); e = np.linalg.eigvalsh(hp.toarray())
    sym = float(np.max(np.abs(np.sort(e) - np.sort(-e))))
    ok = no_sy and aud["ok"] and int((np.abs(ef) < 1e-12).sum()) == 0 and int((ef < 0).sum()) == n // 2 and comm < 1e-14 and acomm < 1e-14 and sym < 1e-10
    rec("S2 kappa=0 operator", ok, no_sigma_y=no_sy, audit_ok=aud["ok"], max_h_plus_conj_h=aud["max_h_plus_conj_h"], max_comm_S=aud["max_commutator_S_stag"],
        flat_zero_levels=int((np.abs(ef) < 1e-12).sum()), flat_min_abs_e=float(np.min(np.abs(ef))), flat_n_neg=int((ef < 0).sum()), parity_comm=float(comm),
        parity_S_anticomm=float(acomm), spec_hplus_symmetric=sym, trapped_zero_levels=int((np.abs(e) < 1e-12).sum()), trapped_n_neg=int((e < 0).sum()))


def s3():
    n = 600; R = np.arange(1, n + 1, dtype=float)
    worst = 0.0; worstS = 0.0; ok = True; info = {}
    for kz in (False, True):
        for trapped in (False, True):
            ops = V6S.ChannelOps(1, R, DEV, kappa_zero=kz)
            N, v = fields(R, 3.0, trapped)
            hp = W1S.sector_plus_csr(ops.h_csr(N, v))
            e, V = dense_plus(hp)
            Vn = V[:, e < 0]
            for nI in (64, 200):
                P, ci = IR.sign_block_contour(hp, nI, threads=4)
                Pd = (Vn[:nI] @ Vn[:nI].mH).numpy()
                d = float(np.max(np.abs(P - Pd)))
                S1_ = ent(torch.linalg.eigvalsh(torch.as_tensor(P))); S2_ = ent(torch.linalg.eigvalsh(torch.as_tensor(Pd)))
                worst = max(worst, d); worstS = max(worstS, abs(S1_ - S2_))
                info["k%s_%s_nI%d" % ("0" if kz else "1", "trap" if trapped else "flat", nI)] = "%.1e/%.1e (%d nodes, quad %.1e)" % (d, abs(S1_ - S2_), ci["nodes"], ci["quad_check_2hs_max"])
    ok = worst < 1e-11 and worstS < 1e-10
    rec("S3 contour vs dense", ok, max_entry_diff=worst, max_entropy_diff=worstS, **info)


def propagate_sub(ops, Phi, t0, nrec, tau=1.0, nsub=8, trapped=True):
    """the production's linear sub-steps between records spaced tau on the synthetic time-dependent fields."""
    R = ops.R; ttot = 0.0; K = 0; tail = 0.0; ns = 0
    for r in range(nrec):
        ta = t0 + r * tau
        N0, v0 = fields(R, ta, trapped); N1, v1 = fields(R, ta + tau, trapped)
        for i in range(nsub):
            u = (i + 0.5) / nsub
            hp = W1S.sector_plus_csr(ops.h_csr(N0 + u * (N1 - N0), v0 + u * (v1 - v0)))
            Phi, K_, tl = V6S.propagate(OPS.Banded(hp, DEV), Phi, tau / nsub, 1e-16, 4096)
            K = max(K, K_); tail += tl; ns += 1
    return Phi, K, tail, ns


def s4():
    n = 500; R = np.arange(1, n + 1, dtype=float)
    ok = True; out = {}
    for kz in (False, True):
        ops = V6S.ChannelOps(1, R, DEV, kappa_zero=kz)
        N, v = fields(R, 0.0)
        hp0 = W1S.sector_plus_csr(ops.h_csr(N, v))
        e, V = dense_plus(hp0)
        Phi = V[:, e < 0].clone()
        Phi, K, tail, ns = propagate_sub(ops, Phi, 0.0, 40)
        G = Phi.mH @ Phi; G.diagonal().sub_(1.0); Tm = float(torch.linalg.eigvalsh(0.5 * (G + G.mH)).abs().sum())
        N1, v1 = fields(R, 40.0); hp1 = W1S.sector_plus_csr(ops.h_csr(N1, v1)); e1, V1 = dense_plus(hp1); Vn = V1[:, e1 < 0]
        for m in (20, 80, 250):
            SI = ent(torch.linalg.eigvalsh(Phi[:m] @ Phi[:m].mH)); SE = ent(W1S.gram_eigs(Phi[m:])[0])
            PI = ent(W1S.gram_eigs(Vn[:m])[0]); PE = ent(W1S.gram_eigs(Vn[m:])[0])
            mm = min(m, Phi.shape[1])
            tol = W1S.entropy_modulus(mm, Tm) + W1S.entropy_modulus(mm, mm * 8 * 2.22e-16) + 1e-12
            ok &= abs(SI - SE) <= tol and abs(PI - PE) <= 1e-9
            out["k%d_m%d" % (0 if kz else 1, m)] = "S_I %.6f S_E %.6f |d| %.1e (tol %.1e); P_I-P_E %.1e; excess %.6f" % (SI, SE, abs(SI - SE), tol, abs(PI - PE), SI - PI)
        out["k%d_impurity" % (0 if kz else 1)] = "%.2e (sub-steps %d, K_max %d)" % (Tm, ns, K)
    rec("S4 purity identity", ok, **out)


def s5():
    n = 400; R = np.arange(1, n + 1, dtype=float)
    ops = V6S.ChannelOps(1, R, DEV)
    N, v = fields(R, 5.0)
    hp = W1S.sector_plus_csr(ops.h_csr(N, v)); B = OPS.Banded(hp, DEV)
    e, V = dense_plus(hp); Phi0 = V[:, e < 0].clone()
    X = torch.randn(n, 50, dtype=torch.float64).to(CD) + 1j * torch.randn(n, 50, dtype=torch.float64).to(CD)
    X = torch.linalg.qr(X)[0]
    A, K1, t1 = V6S.propagate(B, X.clone(), 64.0, 1e-16, 4096)
    Bm = X.clone()
    for _ in range(64):
        Bm, K2, t2 = V6S.propagate(B, Bm, 1.0, 1e-16, 4096)
    Ue = (V * torch.exp(-1j * 64.0 * e.to(CD))[None, :]) @ V.mH
    Ex = Ue @ X
    d1 = float((A - Ex).abs().max()); d2 = float((Bm - Ex).abs().max())
    ok = d1 < 1e-11 and d2 < 1e-11
    rec("S5 held step", ok, big_step_vs_exact=d1, unit_steps_vs_exact=d2, K_big=K1, K_unit=K2)


def s6():
    n = 64; R = np.arange(1, n + 1, dtype=float); tau = 8.0; T = tau * np.arange(11)
    Nt = np.array([1.0 - 0.01 * i * np.exp(-R / 10.0) for i in range(11)]); vt = np.array([-0.1 * (1 + 0.01 * i) / np.sqrt(R) for i in range(11)])
    syn = {"json": {"args": {"g": 0.25, "rs": 4.0}, "records": [{"T": float(t), "E": 10.0 + 0.1 * i} for i, t in enumerate(T)]},
           "npz": {"T": T, "R": R, "N_applied": Nt, "v_applied": vt, "tau": tau, "ell": 4.0}}
    H = V6S.History(None, None, n, 0.25, "applied", synthetic=syn)
    info = IR.hold_extend(H, 5 * tau, 30 * tau, tau, log=lambda m: None)
    ok = len(H) == 31 and info["k_hold"] == 5 and abs(H.T[-1] - 30 * tau) < 1e-12 and np.all(np.diff(H.T) == tau)
    N5, v5 = H.fields(5)
    ok &= all(np.array_equal(H.fields(k)[0], N5) and np.array_equal(H.fields(k)[1], v5) for k in range(5, 31))
    ok &= np.all(H.E[5:] == H.E[5]) and not np.array_equal(H.fields(4)[0], N5)
    rec("S6 hold", ok, records=len(H), k_hold=info["k_hold"], T_last=float(H.T[-1]))


def s7():
    n = 2048; R = np.arange(1, n + 1, dtype=float)
    out = {}; ok = True
    for kz in (True, False):
        ops = V6S.ChannelOps(1, R, DEV, kappa_zero=kz)
        hp = W1S.sector_plus_csr(ops.h_csr(np.ones(n), np.zeros(n)))
        e, V = dense_plus(hp); Vn = V[:, e < 0]
        S = {}
        for l in (31, 32, 63, 64, 127, 128, 255, 256):
            S[l] = ent(torch.linalg.eigvalsh(Vn[:l] @ Vn[:l].mH))
        Sa = {l: 0.5 * (S[l - 1] + S[l]) for l in (32, 64, 128, 256)}           # the even/odd (Friedel) average
        sl = [Sa[2 * l] - Sa[l] for l in (32, 64, 128)]
        out["k%d_slopes" % (0 if kz else 1)] = ", ".join("%.5f" % x for x in sl)
        out["k%d_S" % (0 if kz else 1)] = ", ".join("%d:%.5f" % (l, Sa[l]) for l in (32, 64, 128, 256))
        if kz:
            ok = all(abs(x - math.log(2) / 6) < 0.1 * math.log(2) / 6 for x in sl[1:])
    rec("S7 kappa=0 centre (BCFT slope ln2/6 = %.5f per + sector)" % (math.log(2) / 6), ok, **out)


def s9():
    """the lean mode's host initial vacuum (zheevr negative subset on A^T) against the dense torch eigh: the projectors equal."""
    n = 700; R = np.arange(1, n + 1, dtype=float)
    out = {}; ok = True
    for kz in (False, True):
        ops = V6S.ChannelOps(1, R, DEV, kappa_zero=kz)
        N, v = fields(R, 2.0)
        hp = W1S.sector_plus_csr(ops.h_csr(N, v))
        Phi, info = IR._dense_init_plus(hp, DEV, torch.device("cpu"), 4, lambda m: None)
        e, V = dense_plus(hp); Vn = V[:, e < 0]
        d = float((Phi @ Phi.mH - Vn @ Vn.mH).abs().max())
        G = Phi.mH @ Phi; G.diagonal().sub_(1.0); imp = float(torch.linalg.eigvalsh(0.5 * (G + G.mH)).abs().sum())
        ok &= d < 1e-11 and info["n_neg_plus"] == int((e < 0).sum())
        out["k%d" % (0 if kz else 1)] = "projector diff %.1e, impurity %.1e, n_neg %d, levels vs banded %.1e" % (d, imp, info["n_neg_plus"], info["neg_levels_max_dev_from_banded"])
    rec("S9 host initial vacuum (zheevr subset) vs dense eigh", ok, **out)


def s10():
    """the lean mode's in-place Chebyshev (BandedLean, propagate_lean) against page_history's propagate on OPS.Banded: sub-steps and a held step."""
    n = 500; R = np.arange(1, n + 1, dtype=float)
    out = {}; ok = True
    for kz in (False, True):
        ops = V6S.ChannelOps(1, R, DEV, kappa_zero=kz)
        N, v = fields(R, 1.0)
        hp = W1S.sector_plus_csr(ops.h_csr(N, v))
        e, V = dense_plus(hp); X0 = V[:, e < 0].clone()
        for dt, ch in ((0.125, 4096), (1.0, 97), (64.0, 64)):
            A, K1, t1 = V6S.propagate(OPS.Banded(hp, DEV), X0.clone(), dt, 1e-16, ch)
            B, K2, t2 = IR.propagate_lean(IR.BandedLean(hp, DEV), X0.clone(), dt, 1e-16, ch)
            d = float((A - B).abs().max())
            ok &= d < 1e-13 and K1 == K2
            out["k%d_dt%g" % (0 if kz else 1, dt)] = "%.1e (K %d)" % (d, K2)
    rec("S10 lean propagator vs production propagate", ok, **out)


def s8(rec_json, fields_npz):
    out = {}; ok = True
    H = V6S.History(rec_json, fields_npz, 8192, None, "applied")
    rs = float(H.rs_nom)
    for Trs in (0.0, 50.0, 100.0):
        k = int(np.argmin(np.abs(H.T - Trs * rs)))
        N, v = H.fields(k)
        for kz in (False, True):
            ops = V6S.ChannelOps(1, H.R, DEV, kappa_zero=kz)
            h = ops.h_csr(N, v); aud = W1S.operator_audit(h); hp = W1S.sector_plus_csr(h)
            ab, u = IR._herm_band_upper(hp)
            e = np.asarray(sla.eigvals_banded(ab, lower=False, check_finite=False), float)
            nz = int((np.abs(e) < 1e-12).sum()); nneg = int((e < 0).sum())
            ok &= aud["ok"] and nz == 0 and nneg == len(e) // 2
            out["T%g_k%d" % (Trs, 0 if kz else 1)] = "audit %s, zero levels %d, min|e| %.2e, n_neg %d of %d" % (aud["ok"], nz, float(np.min(np.abs(e))), nneg, len(e))
    rec("S8 B2 fields: the halving's premises at kappa 1 and 0", ok, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2-record", default=None); ap.add_argument("--b2-fields", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    s1(); s2(); s3(); s4(); s5(); s6(); s7(); s9(); s10()
    if a.b2_record and a.b2_fields:
        s8(a.b2_record, a.b2_fields)
    n_ok = sum(1 for v in RES.values() if v["ok"])
    print("SELFTEST %d/%d PASS (%.1f s)" % (n_ok, len(RES), time.time() - t0), flush=True)
    if a.out:
        json.dump(RES, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
