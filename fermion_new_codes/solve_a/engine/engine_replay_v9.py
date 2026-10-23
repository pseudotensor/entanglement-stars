#!/usr/bin/env python
"""engine_replay_v9.py (builder analyst; engine spec 37.5, S5, the lead's turn 45; the brief lead/BRIEF_s7_copyslot_replay9_builder.md;
inbox #31 S5; the icloud lane's strand_E_web40f_rows.tex ER1-ER23 and its scope file, inbox #33, compared where they meet).

37.5 (1), binding: "engine_replay_v9 (new; v8 with per-record hooks, every engine mapping of v8) prints at every native record and channel
b_in = Tr[(C - P) B_in] and b_out, the two finite-zone moments d and f per shell (dust and pair sectors apart) with R_t^zone = 4 pi G
(d + w f) and R_r^zone = 4 pi G [d + r d' - (1 + r v'/v) w f] (S5 (a)), and writes them to the handoff; S5 (b)'s R_hat = R^ref + R^zone is
composed from replay v4's reference columns and these, against engine_certificate_v3's direct residuals at the raw scales per node, the
share within 1/10 printed as the criterion's quantity, nothing ruled; (d) the ledger with -Delta W, W = (1/2) integral of v J over the
stored sources per record, beside a + b + c; (e) the clipped intervals (W below 0, engine_fields_v17 line 409) listed per record with
R_E = -2 f r^2 N^2 rho and R_J = f r N^2 J. (2) Device: pod 1 lane 5 for the production and the r_s 32 set, then the ladder members; one
table per run under results/engine/CERT3/; a certificate version 1.3 only on the lead's reading (turn 46)."

Version 9 = engine_replay_v8 (its loop, checks, table and handoff kept; every engine mapping of v8) plus:
(0) The mapping engine_collapse_v17 -> (engine_collapse_v17, engine_fields_v12, engine_stress_v1) with the run's own rule: the C-init state
re-prepared, R2 continued by engine_collapse_v17.continue_v17 per record (its Weyl gap and level state carried as the run carries them), the
state propagated by the record's eigendecomposition: engine_replay_v2's replay of a v17 run (the r_s 32 set, the production's parent
segment). A v17 run stores no masks or stresses of its own: with --compare-prefix (replay v2's <prefix>_handoff.npz and _spectra.npz of
the same run) they are compared there, else the columns read "-". BUILDER, labelled: 37.5 names the r_s 32 set, whose members are v17 runs.
(1) Per native record (the hooks), per channel and sector (the dust D_d = Phi_d f Phi_d^dagger; the pairs D_p = sea sea^dagger - P, P the
continued reference R2), channel weights 16 kappa applied once:
(c) b_in = Tr[D B_in], b_out = Tr[D B_out], B_in = (i sigma_x/8)(|1><0| - |0><1|), B_out = (i sigma_x/8)(|L-2><L-1| - |L-1><L-2|) (a = 1;
ER18-ER19, P37 (39)): per column -(1/4) Im(c_1^dagger sigma_x c_0) and -(1/4) Im(c_{L-2}^dagger sigma_x c_{L-1}); their sum against -F_last
(the stress module's prefix face flux at the chain's last face, dust plus pairs: ER18's identity i[h_b, Q] = B_in + B_out, its check);
the local-allocation shift Delta S_r = b_in/(4 pi R^2) at R_t printed (ER19; the stored stress is the prefix one).
(a) The finite-zone moments per shell: d_x = Tr[a_x({1 - C2, h}/2) D], f_x = Tr[a_x({1 - C2, Q}/2) D] (a_x(B) = {e_x, B}/2, the engine's
site density), C2 = (T^2 + T^dagger^2)/2 (x) 1_2 the radial chain's cos 2ak (T the site shift, Q = (T^2 - T^dagger^2)/(4i) the two-step
current, the sine bond sigma_x (T - T^dagger)/(2i): engine_vacflow_v1.chain_sparse's stencils): ER6's D = rho - <cos 2ak h_0> and
F = J - <cos 2ak q> with the symbol products as symmetrized operator products (equal at leading gradient order), on the radial chain's own
operators (P35 3.2's conditional application: the on-site kappa/r term and the open ends retained, not removed); dust and pairs apart, their
sum the total; smoothed by the run's own source kernels (engine_fields_v12.kernels: K0 for d as for rho, K1 for f as for J); at the
certificate's nodes r = ell, 2 ell, ..., r_collar - 2 ell (engine_certificate_v3's), with w = v/N, v' from the dense fields of the record's
re-solve (the stored shell totals, the run's field module, as version 8 re-solves them) and d' by engine_rows_v2.deriv at h = ell/32:
R_t^zone = 4 pi G (d + w f), R_r^zone = 4 pi G [d + r d' - (v + r v') f/N] (ER11; (1 + r v'/v) w f = (v + r v') f/N, the regular form),
nodes with |v| < 1e-12 excluded and listed; per sector and total; printed per record at the centre node, at R_t (where ER11 gives
4 pi G (d - f)) and their largest |value| over the support (rho_ell above 1e-6 of its largest), with R_t^zone/H^2 at the centre node beside
ER14's full-band control 1.633520710 (ma 0.2; inherited, not a bound), H^2 = (A^2 + 2 B^2)/3.
(d) The ledger (ER20-ER23; P19 (11)): per record E_G (the record's stored E), K = Tr(h_m D) by the density route sum_x (N_x eps_x + v_x J_x)
at the record's applied fields (all channels), K_src = int dmu (N rho_ell + v J_ell), W = (1/2) int dmu v J_ell (the dense fields, the stored
sources smoothed; dmu = 4 pi r^2 dr, the trapezoid on the solve's fine grid to the collar) with its site form (1/2) sum_x v_x J_x beside it,
K_chi = X1 - X2 (the record's M1_parts), K_out = K - K_chi, eps_adj = K_src - K, eps_G = E_G - (K_src - W); per interval Delta E_G,
a + b + c (the record's M1_balance: evolution_exact + field_work + reference_reset), -Delta W, Delta K_out, Delta eps_adj, Delta eps_G and the
remainder Delta E_G - (a + b + c - Delta W + Delta K_out + Delta eps_adj + Delta eps_G) (ER23: zero up to M1's own closure, printed beside).
(e) The clipped intervals: the coarse nodes where the record's re-solved W_nodes <= 0 (v = 0 there: engine_fields_v17 line 409; the same
clip in engine_fields_v12, v14 and v16), grouped into intervals, with the undivided defects R_E = -8 pi G r^2 N^2 rho_ell and
R_J = 4 pi G r N^2 J_ell at each (P16 (21): f = 4 pi G), the largest per interval; the dense fine grid's clipped points counted.
(2) The handoff: version 8's keys plus d_dust, f_dust, d_pairs, f_pairs (per shell per record), the node arrays of R^zone per sector,
b_in and b_out per channel and sector, the ledger's per-record terms; the json: per record the hooks' summaries.
(3) --selftest: the moment operators against ER14's one-dimensional analogue (the uniform chain without the kappa term: the positive band
Delta = P_+ gives d = rho + p_Q and f = 0 in the bulk, rho = int eps dk/2pi, p_Q = int q^2/eps dk/2pi, q = sin 2k/2) and the chain identities
(C2 against Q and the bond, i[h_b, Q] = B_in + B_out on the stencil).
(4) --compose-rows (CPU, after the runs): S5 (b): per record R^ref from the handoff's reference columns, centred, q_rho = (deps_R2(k) +
deps_R2(k+1))/(2 tau) and the same for J (one-sided at the ends, labelled; replay v4's columns, carried by v8 and here), smoothed by the same
kernels, R_t^ref = -(4 pi G/(N r^2 v)) int_0^r s^2 (N q_rho + v q_J) ds by engine_certificate_v3's Gauss-Legendre quadrature from the centre,
R_r^ref = (-4 pi G q_rho/N - (A + B) R_t^ref)/B; R_hat = R^ref + R^zone (P35 5); against engine_certificate_v3's direct residuals recomputed
on the certificate's own data npz by its certificate() (its nodes, support, h and rule): the agreement |N R_hat - res_direct|/|res_direct| at
the support nodes and the share within 1/10 per row (the criterion's quantity; nothing ruled), with R^zone alone, R^ref alone and the
certificate's own quadrature beside; the moments table.
Args: --run dir/tag --resume-from --gpu --records --md --out-json --out-npz --compare-prefix --engine --fields --stress; --selftest --md;
--compose-rows handoff.npz[,handoff2.npz] --replay-json a.json[,b.json] --data --fm-segments --fm-default --t-ranges --md --out-json.
Fix (20:15 UTC, before any trapped record had been written): the at-R_t readout's one-element kernel products are indexed (at_root()); the
first deployment failed there at each run's first trapped record; the outputs of the c17 parent (T/r_s 0 to 11, no root) are unaffected.
Version 8's docstring follows.
engine_replay_v8.py (builder analyst; engine spec 31.5: "engine_replay_v8 (a new file: v7 with engine_collapse_v27, v29, v30 and
v31 mapped to the v25 modules and ops v3, the preparation driven by the stored C-init fields as v7 does; the gate bitwise v7 on three
records of a v25 run)"): version 7 with the runs of engine_collapse_v27, v29, v30 and v31 (the fixed-preparation ladder's engines:
version 25 with --prep-ell, the field tolerances, the damped and the Newton-Krylov solves as options) replayed by engine_collapse_v25's
functions, engine_fields_v16, engine_stress_v2 and ops v3, and the dust prepared at the run's --prep-ell where it has one (the run's
own preparation scale: version 27 prepared the initial data at prep_ell and evolved at ell; version 7 prepared at ell), the sea still
built from the stored C-init fields. The replay re-solves no self-consistent iteration, so the run's solver (plain, damped or Newton) does
not enter. (Versions 3 to 7: the masks bitwise against the stored ones, the stresses of the dust and the pairs against the stored face flux
and centrifugal force, the Gauss identity and the interval closure, engine 28.2's moving-reference columns deps_R2, dJ_R2 and the collar
budget; their docstrings are in engine_replay_v8.py.)"""
import argparse
import importlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# the run's engine -> (engine, field module, stress module), as each engine version imports them (version 8's map; version 9 adds v17)
MODULES = {"engine_collapse_v19": ("engine_collapse_v19", "engine_fields_v14", "engine_stress_v1"),
           "engine_collapse_v24": ("engine_collapse_v24", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v25": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v27": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v29": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v30": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           "engine_collapse_v31": ("engine_collapse_v25", "engine_fields_v16", "engine_stress_v2"),
           # version 9: a v17 run (the r_s 32 set, the production's parent segment), engine_replay_v2's modules
           "engine_collapse_v17": ("engine_collapse_v17", "engine_fields_v12", "engine_stress_v1")}
CONT = {"engine_collapse_v19": "ops v2", "engine_collapse_v24": "ops v2", "engine_collapse_v25": "ops v3", "engine_collapse_v17": "v17 rule"}
FULL_BAND_CONTROL = 1.633520710   # ER14 / NS13: (3/2)(1 + p_Q/rho) at ma 0.2 (inherited zone integral; a control, not a bound)
E19 = F14 = STR = OPS2 = OPS3 = None


def rel(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.max(np.abs(x - y))) / max(float(np.max(np.abs(y))), 1e-300)


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


# ------------------------------------------------------------------------------------------------------------------ the hooks' operators
def chain_shift(n):
    """T = sum_i |i><i+1| (x) 1_2 on the site-major two-component chain (engine_vacflow_v1.chain_sparse's ordering)."""
    T = sp.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")
    return sp.kron(T, sp.identity(2, format="csr"), format="csr")


def moment_operators(ch):
    """the radial chain's cos 2ak, C2 = (T^2 + T^dagger^2)/2 (x) 1, and the moment operators D_op = {1 - C2, h}/2, F_op = {1 - C2, Q}/2 (host
    CSR); their device CSR beside (the channel's own construction)."""
    n2 = ch.hs.shape[0]
    T = chain_shift(n2 // 2)
    C2 = 0.5 * (T @ T + (T @ T).T)
    I_ = sp.identity(n2, format="csr")
    Om = (I_ - C2).astype(complex)
    D = (0.5 * (Om @ ch.hs + ch.hs @ Om)).tocsr()
    Fo = (0.5 * (Om @ ch.Qs + ch.Qs @ Om)).tocsr()
    return C2.tocsr(), D, Fo


def to_dev(A, torch, dev):
    A = A.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                   torch.as_tensor(A.data.astype(complex), dtype=torch.complex128), size=A.shape, device=dev)


def b_terms(cols, wts):
    """b_in, b_out of the columns (2n rows, site-major) with signed weights wts: -(1/4) sum_j w_j Im(c_1^dag sigma_x c_0) and
    -(1/4) sum_j w_j Im(c_{L-2}^dag sigma_x c_{L-1}) (ER19 with a = 1: Tr[D B] per column)."""
    if cols.shape[1] == 0:
        return 0.0, 0.0
    c = cols
    u_in = c[2].conj() * c[1] + c[3].conj() * c[0]           # c_1^dag sigma_x c_0
    u_out = c[-4].conj() * c[-1] + c[-3].conj() * c[-2]      # c_{L-2}^dag sigma_x c_{L-1}
    w_ = wts
    b_in = -0.25 * float((w_ * u_in.imag).sum()); b_out = -0.25 * float((w_ * u_out.imag).sum())
    return b_in, b_out


def node_rows(Fk, R, ell, g, inf_, d_sites, f_sites, nodes, h):
    """ER11 at the nodes: d_ell = K0 d, f_ell = K1 f (the run's kernels), w = v/N and v' from the dense fields, d' by engine_rows_v2.deriv;
    R_t^zone = 4 pi G (d + w f), R_r^zone = 4 pi G [d + r d' - (v + r v') f/N]; nodes with |v| < 1e-12 give NaN."""
    import engine_rows_v2 as RW
    sel = (np.abs(d_sites) > 0) | (np.abs(f_sites) > 0)
    Rs = np.asarray(R, float)[sel]; ds = np.asarray(d_sites, float)[sel]; fs = np.asarray(f_sites, float)[sel]
    if not sel.any():
        N, N1, _ = RW.deriv(inf_["N_dense"], nodes, h)
        v, v1, _ = RW.deriv(inf_["v_dense"], nodes, h)
        z = np.zeros(len(nodes)); ok = np.abs(v) >= 1e-12
        return {"d": z, "f": z, "dprime": z, "Rt": np.where(ok, 0.0, np.nan), "Rr": np.where(ok, 0.0, np.nan), "N": N, "v": v, "v1": v1, "ok": ok}

    def dl(r):
        r = np.atleast_1d(np.asarray(r, float)); K0, K1 = Fk.kernels(r, Rs, ell); return K0 @ ds

    def fl(r):
        r = np.atleast_1d(np.asarray(r, float)); K0, K1 = Fk.kernels(r, Rs, ell); return K1 @ fs
    d0, d1, _ = RW.deriv(dl, nodes, h)
    f0 = fl(nodes)
    N, N1, _ = RW.deriv(inf_["N_dense"], nodes, h)
    v, v1, _ = RW.deriv(inf_["v_dense"], nodes, h)
    ok = np.abs(v) >= 1e-12
    c = 4.0 * math.pi * g
    Rt = np.where(ok, c * (d0 + (v / N) * f0), np.nan)
    Rr = np.where(ok, c * (d0 + nodes * d1 - (v + nodes * v1) * f0 / N), np.nan)
    return {"d": d0, "f": f0, "dprime": d1, "Rt": Rt, "Rr": Rr, "N": N, "v": v, "v1": v1, "ok": ok}


def at_root(KF, R, ell, g, inf_, d_sites, f_sites, Rt):
    """ER11 at the record's stored R_t (the dense root): d, f smoothed there by the run's kernels, w = v/N there; 4 pi G (d + w f) beside the
    root form 4 pi G (d - f) (equal when w = -1 exactly). (The fix of 20:15 UTC: the one-element kernel products are indexed, numpy's float()
    refuses a one-element array; the first version failed at the first trapped record.)"""
    rr_ = np.array([float(Rt)])
    sel = (np.abs(d_sites) > 0) | (np.abs(f_sites) > 0)
    K0, K1 = KF.kernels(rr_, np.asarray(R, float)[sel], ell)
    dR = float(np.asarray(K0 @ np.asarray(d_sites, float)[sel]).ravel()[0]); fR = float(np.asarray(K1 @ np.asarray(f_sites, float)[sel]).ravel()[0])
    wR = float(np.asarray(inf_["v_dense"](rr_)).ravel()[0] / np.asarray(inf_["N_dense"](rr_)).ravel()[0])
    return {"R_t": float(Rt), "d": dR, "f": fR, "w": wR, "Rt_zone": 4.0 * math.pi * g * (dR + wR * fR), "Rt_zone_root_form": 4.0 * math.pi * g * (dR - fR)}


def ledger_terms(inf_, R, g, es, Js, N_app, v_app, rec):
    """ER21's per-record values: K by the density route (the applied fields), K_src and W by the trapezoid on the solve's fine grid to the
    collar (rho_ell, J_ell, the dense N, v), W's site form, K_chi from the record's M1_parts, E_G the record's stored E."""
    out = {"E_G": rec.get("E"), "K": float(np.sum(N_app * es + v_app * Js)), "W_site": 0.5 * float(np.sum(v_app * Js))}
    try:
        rc = float(inf_["r_collar"]); rf = np.asarray(inf_["rfine"], float); rf = rf[rf <= rc + 1e-12]
        Nn = np.asarray(inf_["N_dense"](rf), float); vv = np.asarray(inf_["v_dense"](rf), float)
        rho = np.asarray(inf_["rho_ell"](rf), float); J = np.asarray(inf_["J_ell"](rf), float)
        mu = 4.0 * math.pi * rf ** 2
        tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        out["K_src"] = float(tz(mu * (Nn * rho + vv * J), rf)); out["W"] = 0.5 * float(tz(mu * vv * J, rf))
        out["M_trapezoid"] = float(tz(mu * (rho + (vv / Nn) * J), rf))
        out["r_collar"] = rc
    except Exception as ex:  # noqa: BLE001
        out["K_src"] = out["W"] = None; out["error"] = repr(ex)
    mp = rec.get("M1_parts") or {}
    out["K_chi"] = (float(mp["X1"]) - float(mp["X2"])) if ("X1" in mp and "X2" in mp) else None
    if out.get("K_src") is not None:
        out["eps_adj"] = out["K_src"] - out["K"]
        out["eps_G"] = (float(out["E_G"]) - (out["K_src"] - out["W"])) if out["E_G"] is not None else None
    out["K_out"] = (out["K"] - out["K_chi"]) if out["K_chi"] is not None else None
    bal = rec.get("M1_balance") or {}
    out["a"] = bal.get("evolution_exact"); out["b"] = bal.get("field_work"); out["c"] = bal.get("reference_reset")
    out["dHin"] = bal.get("dHin"); out["M1_closure"] = bal.get("closure"); out["cut_changed"] = bal.get("cut_changed")
    return out


def clipped(inf_, g, r_nodes):
    """(e): the coarse nodes r_nodes (the field module's coarse_grid) with W_nodes <= 0 (node 0 excluded: v = 0 there by the centre's
    parity), grouped into intervals, with R_E = -8 pi G r^2 N^2 rho_ell and R_J = 4 pi G r N^2 J_ell (P16 (21), f = 4 pi G); the dense fine
    grid's points with W <= 0 counted."""
    out = {"intervals": [], "n_nodes": 0, "dense_points": None}
    Wn = inf_.get("W_nodes")
    if Wn is None:
        return out
    Wn = np.asarray(Wn, float)
    rn = np.asarray(r_nodes, float)
    if len(rn) != len(Wn):
        out["why"] = "W_nodes (%d) and the coarse grid (%d) differ in length" % (len(Wn), len(rn))
        return out
    bad = np.nonzero(Wn[1:] <= 0.0)[0] + 1
    out["n_nodes"] = int(len(bad))
    sc_E = 8.0 * math.pi * g * float(np.max(np.abs(rn ** 2 * np.asarray(inf_["N_dense"](rn), float) ** 2 * np.asarray(inf_["rho_ell"](rn), float)))) if len(rn) else 1.0
    if len(bad):
        groups = np.split(bad, np.nonzero(np.diff(bad) > 1)[0] + 1)
        for gq in groups:
            r_ = rn[gq]; N_ = np.asarray(inf_["N_dense"](r_), float)
            rho = np.asarray(inf_["rho_ell"](r_), float); J = np.asarray(inf_["J_ell"](r_), float)
            RE = -8.0 * math.pi * g * r_ ** 2 * N_ ** 2 * rho; RJ = 4.0 * math.pi * g * r_ * N_ ** 2 * J
            out["intervals"].append({"r_first": float(r_[0]), "r_last": float(r_[-1]), "nodes": int(len(gq)), "R_E_max_abs": float(np.max(np.abs(RE))),
                                     "R_J_max_abs": float(np.max(np.abs(RJ))), "R_E_rel_scale": float(np.max(np.abs(RE))) / max(sc_E, 1e-300),
                                     "W_min": float(np.min(Wn[gq]))})
    try:
        rf = np.asarray(inf_["rfine"], float); rc = float(inf_["r_collar"]); rf = rf[(rf > 0) & (rf <= rc)]
        Wd = np.asarray(inf_["W_field"](rf), float)
        out["dense_points"] = int(np.sum(Wd <= 0.0)); out["dense_total"] = int(len(rf))
    except Exception:  # noqa: BLE001
        pass
    out["scale_R_E"] = sc_E
    return out


# ------------------------------------------------------------------------------------------------------------------ the replay (version 8's loop)
def replay(x):
    global E19, F14, STR, OPS2, OPS3
    d, tag = os.path.split(x.run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    a = SimpleNamespace(**J["args"])
    run_eng = os.path.basename(str(J.get("script", "engine_collapse_v19.py"))).replace(".py", "")
    if run_eng not in MODULES and not (x.engine and x.fields and x.stress):
        raise SystemExit("engine_replay_v9: the run's script %s has no module set; pass --engine, --fields and --stress" % run_eng)
    eng_m, fld_m, str_m = MODULES.get(run_eng, (None, None, None))
    eng_m = x.engine or eng_m; fld_m = x.fields or fld_m; str_m = x.stress or str_m
    E19 = importlib.import_module(eng_m); F14 = importlib.import_module(fld_m); STR = importlib.import_module(str_m)
    KF = importlib.import_module("engine_fields_v12")                              # the source kernels every field module uses
    cont = CONT.get(os.path.basename(eng_m), "ops v2")
    v17 = cont == "v17 rule"
    if not v17:
        OPS2 = importlib.import_module("engine_ops_v2"); OPS3 = importlib.import_module("engine_ops_v3")
        H = np.load(os.path.join(d, "collapse_%s_handoff.npz" % tag), allow_pickle=True)
        S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    else:
        H = np.load(x.compare_prefix + "_handoff.npz", allow_pickle=True) if x.compare_prefix else None
        S = np.load(x.compare_prefix + "_spectra.npz", allow_pickle=True) if x.compare_prefix else None
    print("engine_replay_v9: the run's script %s; replaying with %s, %s, %s; the continuation %s%s" % (
        run_eng, eng_m, fld_m, str_m, cont, ("; masks and stresses compared with %s" % x.compare_prefix) if (v17 and x.compare_prefix) else ""), flush=True)
    if x.resume_from:
        a.resume_from = x.resume_from
    E19.OPT["collar_policy"] = a.collar_policy; E19.OPT["support_rel"] = a.support_rel; E19.OPT["site_fields"] = a.site_fields
    E19.C5.backend(x.gpu)
    torch = E19.BK["torch"]; dev = E19.BK["dev"]
    rs = float(a.rs); tau = float(a.tau)
    chs = [E19.Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    shell = [float(q) * rs for q in a.shell.split(",")] if a.shell else None
    resume = None
    if getattr(a, "resume_from", None):
        if v17:
            raise SystemExit("engine_replay_v9: a resumed v17 run is not replayed by the v17 path")
        st, f_ref, stored, head, resume = (E19.resume_state_v25 if cont == "ops v3" else E19.resume_state)(a, chs, R, shell)
        if "records_replayed" in resume:
            start_txt = "resumed: %d records of %s replayed by v17's rule to T/rs %.3f, the largest source differences eps %.2e, J %.2e (acceptance 1e-8); the C-init state re-prepared from the stored C-init fields (%.2e, %.2e)" % (
                resume["records_replayed"], resume["from"], resume["T_over_rs"], resume["max_eps_rel"], resume["max_J_rel"], resume["cinit"]["dN"], resume["cinit"]["dv"])
        else:
            pr = resume.get("parent_resume") or {}
            start_txt = ("resumed (v25 from a v19 parent): the grandparent's %s records replayed by v17's rule to T/rs %s (sources eps %s, J %s; the C-init state %s, %s); "
                         "then %d native records of %s replayed by its own rule to T/rs %.3f, the largest source differences eps %.2e, J %.2e, the masks equal in %d of %d channel-records (acceptance 1e-8)") % (
                pr.get("records_replayed", "-"), ("%.3f" % pr["T_over_rs"]) if pr.get("T_over_rs") is not None else "-", ("%.2e" % pr["max_eps_rel"]) if pr.get("max_eps_rel") is not None else "-",
                ("%.2e" % pr["max_J_rel"]) if pr.get("max_J_rel") is not None else "-", ("%.2e" % pr["cinit"]["dN"]) if pr.get("cinit") else "-", ("%.2e" % pr["cinit"]["dv"]) if pr.get("cinit") else "-",
                resume["native_records_replayed"], resume["from"], resume["T_over_rs"], resume["native_max_eps_rel"], resume["native_max_J_rel"], resume["native_masks_equal"][0], resume["native_masks_equal"][1])
    else:
        prep_ell = getattr(a, "prep_ell", None) or a.ell  # version 8: the run's own preparation scale
        Nc, vc, dust, info0, head0 = E19.prepare_dust(chs, a.g, R, prep_ell, rs / (2.0 * a.g * a.m), a.R0_over_rs * rs, a.m, a.reading, shell)
        Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
        st = {}
        for ch in chs:
            e, U = E19.eigT(E19.clocked_gpu(ch, Ncs, vcs))
            neg = e < 0
            Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=dev), np.zeros(0)))
            en = e.real.cpu().numpy(); kk = int(neg.sum())
            gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
            ref0 = U[:, neg].clone()
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": kk, "window": [{"ref": ref0, "f": (Ncs.copy(), vcs.copy()), "gap": gap}],
                            "gap_between_prev": gap, "levels_prev": (np.asarray(en[:kk], float), np.asarray(en[kk:], float), 0)}
            del e, U
        f_ref = (Ncs, vcs)
        start_txt = "from T = 0: the C-init state re-prepared, the stored C-init fields driving it (max |N - N_cinit| %.2e, |v - v_cinit| %.2e)" % (
            float(np.max(np.abs(Nc - Ncs))), float(np.max(np.abs(vc - vcs))))
    print(start_txt, flush=True)
    Tz = np.asarray(Z["T"], float)
    TS = np.asarray(S["T"], float) if (S is not None and not v17) else Tz
    TH = np.asarray(H["T"], float) if H is not None else None
    TSc = np.asarray(S["T"], float) if S is not None else None
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    cstr = {ch.kappa: STR.ChanStress(ch, torch, dev) for ch in chs}
    mops = {}
    for ch in chs:
        C2h, Dh, Fh = moment_operators(ch)
        mops[ch.kappa] = (to_dev(Dh, torch, dev), to_dev(Fh, torch, dev))
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    r_nodes = F14.coarse_grid(R, a.ell)
    K = len(TS) if x.records is None else min(len(TS), x.records)
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "J_R2": [], "deps_R2": [], "dJ_R2": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc"):
            hand["%s_%s" % (q, part)] = []
    for q in ("d_dust", "f_dust", "d_pairs", "f_pairs", "b_in_dust", "b_in_pairs", "b_out_dust", "b_out_pairs", "nodes", "Rt_zone_dust", "Rr_zone_dust",
              "Rt_zone_pairs", "Rr_zone_pairs", "eps_rep", "J_rep", "J_stored", "eps_stored"):
        hand[q] = []
    rows = []
    hooks = []
    L = ["## %s replayed on its applied-field history with the per-record hooks of engine spec 37.5 (engine_replay_v9.py, version 8's table; %s with %s and %s, the continuation %s; r_s %g, tau %g, ell %g, site fields %s, c_LZ %s, window %s)" % (
        tag, J.get("script"), fld_m, str_m, cont, rs, tau, a.ell, a.site_fields, getattr(a, "c_LZ", "-"), getattr(a, "window_max", "-")), "",
         "Start: %s." % start_txt, "",
         "| T/rs | eps, J (total) | eps, J (dust) | eps, J (pairs) | mask = stored (channels) | mask = negative set | F, f dust | F, f pairs (pairs' max abs F) | Gauss identity | interval closure | seconds |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    prev = None
    t_all = time.time()
    s1rows = []
    prevS1 = None
    T_ = lambda X: torch.as_tensor(X, device=dev)

    def cut_current_D(Hhost, DW, cut, lo, hi):
        """the engine's energy_cut_current form with a density-matrix window DW (rows and columns lo:hi) in place of the occupied columns."""
        h2W = T_((Hhost[lo:hi, :] @ Hhost[:, lo:hi]).toarray()).to(torch.complex128)
        P = torch.zeros(hi - lo, dtype=torch.complex128, device=dev); P[: 2 * cut - lo] = 1.0
        Cm = h2W * P[None, :] - P[:, None] * h2W
        return float((0.5j * (DW.T * Cm).sum()).real)

    def dump():
        np.savez_compressed(x.out_npz + "_handoff.npz", R=R, r_nodes=r_nodes, g=a.g, ell=a.ell, m=a.m, tau=tau, rs=rs, kappas=np.array([ch.kappa for ch in chs]),
                            weights=np.array([ch.w for ch in chs]),
                            **{k: (np.array(v_, dtype=object) if k.startswith("R") and k.endswith(("_dust", "_pairs")) or k == "nodes" else np.array(v_)) for k, v_ in hand.items()})
        json.dump({"script": "engine_replay_v9.py", "modules": [eng_m, fld_m, str_m], "continuation": cont, "run": x.run, "tag": tag, "rs": rs, "tau": tau, "ell": a.ell, "g": a.g,
                   "start": start_txt, "resume": json.loads(json.dumps(resume, default=lambda o: float(o) if np.isscalar(o) else str(o))), "rows": rows, "s1": s1rows, "hooks": hooks,
                   "full_band_control": FULL_BAND_CONTROL}, open(x.out_json, "w"), indent=1, default=float)
    for i in range(K):
        tr = time.time()
        E19.tm_reset()
        T = float(TS[i])
        kz = int(np.argmin(np.abs(Tz - T)))
        kh = int(np.argmin(np.abs(TH - T))) if TH is not None else None
        ks_ = int(np.argmin(np.abs(TSc - T))) if TSc is not None else None
        if abs(Tz[kz] - T) > 1e-9 or (not v17 and abs(TH[kh] - T) > 1e-9):
            L.append("| %.3f | the stored fields or handoff hold no record at this time: stopped |" % (T / rs)); break
        has_H = TH is not None and abs(TH[kh] - T) <= 1e-9
        has_S = TSc is not None and abs(TSc[ks_] - T) <= 1e-9
        rec = recs.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][kz], float); v_app = np.asarray(Z["v_applied"][kz], float)
        eig = E19.eigh_all(chs, N_app, v_app)
        cache = {}
        nmask_equal = 0; mask_neg = True; mask_diff_ch = []
        why = None
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            if v17:
                ref, ci = E19.continue_v17(ch, s_["ref"], f_ref, (N_app, v_app), e, U, s_["n_sea"], s_.get("gap_between_prev"), s_.get("levels_prev"),
                                           force_subdivide=bool(rec.get("verification_differs")))
                if ref is None:
                    why = "kappa %d: the continuation failed (%s)" % (ch.kappa, ci.get("why")); break
                mask = (((ref.conj().T @ U).abs() ** 2).sum(dim=0) > 0.5).cpu().numpy()   # engine_replay_v2's occupied set
            elif cont == "ops v3":
                ref, ci, mask = OPS3.continue_sea_v3(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], tau, a.c_LZ, E19.eig_at, E19.clocked_host_sparse)
            else:
                ref, ci, mask = OPS2.continue_sea_v2(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], tau, a.c_LZ, E19.eig_at, E19.clocked_host_sparse, a.window_max)
            mask = np.asarray(mask, bool)
            ci["_mask"] = mask
            if has_S:
                smask = np.unpackbits(np.asarray(S["mask_%d" % ch.kappa][ks_], np.uint8))[:mask.size].astype(bool)
                if np.array_equal(mask, smask):
                    nmask_equal += 1
                else:
                    mask_diff_ch.append("%d (%d bits)" % (ch.kappa, int(np.sum(mask != smask))))
            mask_neg = mask_neg and bool(np.array_equal(mask, (e < 0).cpu().numpy()))
            cache[ch.kappa] = (e, U, ref, ci)
        del eig
        if why:
            L.append("| %.3f | REPLAY STOPPED: %s |" % (T / rs, why)); print(why, flush=True)
            break
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites); epsP = np.zeros(a.nsites); JP = np.zeros(a.nsites)
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_R2 = np.zeros(a.nsites)
        dD = np.zeros(a.nsites); fD = np.zeros(a.nsites); dPp = np.zeros(a.nsites); fPp = np.zeros(a.nsites)
        bch = {"in_dust": [], "in_pairs": [], "out_dust": [], "out_pairs": []}
        Pd = Pp = None
        t_hk = 0.0
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            hs_ = E19.dens(ch.h, s_["sea"]); hr_ = E19.dens(ch.h, ref); hd_ = E19.dens(ch.h, s_["dust"], s_["f"])
            qs_ = E19.dens(ch.Q, s_["sea"]); qr_ = E19.dens(ch.Q, ref); qd_ = E19.dens(ch.Q, s_["dust"], s_["f"])
            eps += ch.w * (hs_ - hr_ + hd_); Jt += ch.w * (qs_ - qr_ + qd_); epsP += ch.w * (hs_ - hr_); JP += ch.w * (qs_ - qr_)
            eps_sea += ch.w * hs_; eps_R2 += ch.w * hr_; eps_dust += ch.w * hd_; J_R2 += ch.w * qr_
            cs = cstr[ch.kappa]
            pd_ = dict(STR.stress_parts(cs, s_["dust"], s_["f"]))
            pp_ = dict(STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])])))
            Pd = {q: ch.w * np.asarray(v_, float) for q, v_ in pd_.items()} if Pd is None else {q: Pd[q] + ch.w * np.asarray(pd_[q], float) for q in Pd}
            Pp = {q: ch.w * np.asarray(v_, float) for q, v_ in pp_.items()} if Pp is None else {q: Pp[q] + ch.w * np.asarray(pp_[q], float) for q in Pp}
            # version 9's hooks: (a) the moments, (c) the origin pair, per channel and sector
            t_h = time.time()
            Dd, Fdv = mops[ch.kappa]
            dD += ch.w * E19.dens(Dd, s_["dust"], s_["f"]); fD += ch.w * E19.dens(Fdv, s_["dust"], s_["f"])
            dPp += ch.w * (E19.dens(Dd, s_["sea"]) - E19.dens(Dd, ref)); fPp += ch.w * (E19.dens(Fdv, s_["sea"]) - E19.dens(Fdv, ref))
            wdust = torch.as_tensor(np.asarray(s_["f"], float), dtype=torch.float64, device=dev)
            one_s = torch.ones(s_["sea"].shape[1], dtype=torch.float64, device=dev); one_r = torch.ones(ref.shape[1], dtype=torch.float64, device=dev)
            bi_d, bo_d = b_terms(s_["dust"], wdust); bi_s, bo_s = b_terms(s_["sea"], one_s); bi_r, bo_r = b_terms(ref, one_r)
            bch["in_dust"].append(ch.w * bi_d); bch["out_dust"].append(ch.w * bo_d)
            bch["in_pairs"].append(ch.w * (bi_s - bi_r)); bch["out_pairs"].append(ch.w * (bo_s - bo_r))
            t_hk += time.time() - t_h
        es = np.asarray(Z["eps_tot"][kz], float); Js = np.asarray(Z["J_tot"][kz], float)
        esP = np.asarray(Z["eps_pairs"][kz], float); JsP = np.asarray(Z["J_pairs"][kz], float)
        d_tot = (rel(eps, es), rel(Jt, Js)); d_dust = (rel(eps - epsP, es - esP), rel(Jt - JP, Js - JsP)); d_pairs = (rel(epsP, esP), rel(JP, JsP))
        SVd = STR.shell_values(Pd, R, a.m); SVp = STR.shell_values(Pp, R, a.m)
        if has_H:
            s_dust = (rel(SVd["F_face"], H["F_face_dust"][kh]), rel(Pd["fc"], H["fc_dust"][kh]))
            s_pairs = (rel(SVp["F_face"], H["F_face_pairs"][kh]), rel(Pp["fc"], H["fc_pairs"][kh]))
            pmax = float(np.max(np.abs(np.asarray(H["F_face_pairs"][kh], float))))
        else:
            s_dust = s_pairs = (float("nan"), float("nan")); pmax = float("nan")
        _, _, _, inf_ = F14.solve_constraints_ivp(es, Js, a.g, R, a.ell, collar_policy=E19.OPT["collar_policy"], support_rel=E19.OPT["support_rel"])
        if inf_.get("failed"):
            N2 = np.full(a.nsites, np.nan)
        else:
            N2 = F14.site_fields_kernel_average({"N_dense": (lambda r_: inf_["N_dense"](r_) ** 2), "v_dense": zero}, R, a.ell)[0]
        E = float(rec.get("E", np.nan))
        gid = (E - float(np.sum(es * N2))) / abs(E) if E == E else float("nan")
        eC = eps_sea + eps_dust
        clos = None; clos_parts = None
        if prev is not None:
            a_ = 0.5 * (prev["N2"] + N2)
            evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eps_R2 - prev["eR"])))
            lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + es)))
            clos = (E - prev["E"]) - (evo + rst + lap)
            clos_parts = {"evolution": evo, "reset": rst, "lapse": lap}
        prev = {"N2": N2, "eC": eC, "eR": eps_R2, "eps": es, "E": E}
        # engine 28.2 (S1): the moving reference since the previous record, the collar budget and the first-adiabatic current (version 8's)
        s1 = None
        cutk = int(rec["M1_parts"]["cut"]) if rec.get("M1_parts") else None
        if prevS1 is not None and cutk is not None:
            t_s1 = time.time()
            deps = eps_R2 - prevS1["eps_R2"]; dJ = J_R2 - prevS1["J_R2"]
            trh_dens = float(np.sum(N_app * deps + v_app * dJ))
            trhR = float(np.sum((N_app * deps + v_app * dJ)[:cutk]))
            trh_spec = 0.0; FC1 = {1e-6: 0.0, 1e-4: 0.0}; drop = {1e-6: 0.0, 1e-4: 0.0}; pd_tot = 0.0
            for ch in chs:
                e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
                en = e.real.to(torch.float64)
                Ak = U.conj().T @ ref; Bk = U.conj().T @ s_["ref"]
                dk = (Ak.abs() ** 2).sum(dim=1) - (Bk.abs() ** 2).sum(dim=1)
                trh_spec += ch.w * float((en * dk).sum())
                occ = torch.as_tensor(np.asarray(ci["_mask"], bool), device=dev)
                oi = torch.nonzero(occ).ravel(); ei = torch.nonzero(~occ).ravel()
                if len(oi) == 0 or len(ei) == 0:
                    continue
                Pdot = ((Ak[oi] @ Ak[ei].conj().T) - (Bk[oi] @ Bk[ei].conj().T)) / tau
                pd_tot += ch.w * float((Pdot.abs() ** 2).sum())
                de = en[oi][:, None] - en[ei][None, :]
                Hk = E19.clocked_host_sparse(ch, N_app, v_app)
                lo, hi = 2 * max(cutk - 8, 0), 2 * min(cutk + 8, Hk.shape[0] // 2)
                UWo = U[lo:hi][:, oi]; UWe = U[lo:hi][:, ei]
                for dl in (1e-6, 1e-4):
                    keep = de.abs() >= dl
                    X = torch.where(keep, 1j * Pdot / torch.where(keep, de, torch.ones_like(de)).to(Pdot.dtype), torch.zeros_like(Pdot))
                    drop[dl] += ch.w * float((Pdot.abs() ** 2 * (~keep)).sum())
                    DW = UWo @ X @ UWe.conj().T; DW = DW + DW.conj().T
                    FC1[dl] += ch.w * cut_current_D(Hk, DW, cutk, lo, hi)
                del Ak, Bk, Pdot, de
            NRk = float(Z["N"][kz][cutk]); vRk = float(Z["v"][kz][cutk]); Rk = float(R[cutk])
            MR = Rk * (vRk / NRk) ** 2 / (2.0 * a.g)
            NR = 0.5 * (NRk + prevS1["N_at"](cutk)); MR0 = prevS1["M_at"](cutk)
            bal = rec.get("M1_balance") or {}
            FRdT = float(bal.get("evolution_trapezoid", float("nan")))
            DRdT = -NR * (MR - MR0) - FRdT
            rst_g = clos_parts.get("reset") if isinstance(clos_parts, dict) else None
            s1 = {"T_over_rs": T / rs, "cut": cutk, "R": Rk, "tr_h_dP_spectral": trh_spec, "tr_h_dP_density": trh_dens, "tr_hR_dP": trhR,
                  "F_C1_1e-6": FC1[1e-6], "F_C1_1e-4": FC1[1e-4], "dropped_1e-6": drop[1e-6], "dropped_1e-4": drop[1e-4], "P_dot_opposite_weight": pd_tot,
                  "N_R": NR, "dM_R": MR - MR0, "F_R_dT": FRdT, "D_R_dT": DRdT, "D_R_rate": DRdT / tau,
                  "residual_C1_plus_DR_1e-6": FC1[1e-6] + DRdT / tau, "residual_C1_plus_DR_1e-4": FC1[1e-4] + DRdT / tau,
                  "residual_C1_plus_trhR_1e-6": FC1[1e-6] + trhR / tau, "gauss_reset": rst_g, "seconds": time.time() - t_s1}
            s1rows.append(s1)
        prevS1 = {"eps_R2": eps_R2.copy(), "J_R2": J_R2.copy(), "N_at": (lambda c, kz=kz: float(Z["N"][kz][c])),
                  "M_at": (lambda c, kz=kz: float(R[c]) * (float(Z["v"][kz][c]) / float(Z["N"][kz][c])) ** 2 / (2.0 * a.g))}
        hand["J_R2"].append(J_R2)
        hand["deps_R2"].append(eps_R2 - (hand["eps_R2"][-1] if hand["eps_R2"] else eps_R2)); hand["dJ_R2"].append(J_R2 - (hand["J_R2"][-2] if len(hand["J_R2"]) > 1 else J_R2))
        hand["T"].append(T); hand["N2avg"].append(N2); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(eps_dust)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
        # ---------------------------------------------------------------- version 9's hooks at this record
        t_h = time.time()
        hk = {"T_over_rs": T / rs, "k": i}
        bin_d = float(np.sum(bch["in_dust"])); bin_p = float(np.sum(bch["in_pairs"])); bout_d = float(np.sum(bch["out_dust"])); bout_p = float(np.sum(bch["out_pairs"]))
        F_last = float(SVd["F_face"][-1] + SVp["F_face"][-1])
        Rt_rec = rec.get("R_t")
        hk["b"] = {"in_dust": bin_d, "in_pairs": bin_p, "out_dust": bout_d, "out_pairs": bout_p, "in": bin_d + bin_p, "out": bout_d + bout_p,
                   "F_last": F_last, "identity": (bin_d + bin_p + bout_d + bout_p) + F_last,
                   "identity_rel": abs((bin_d + bin_p + bout_d + bout_p) + F_last) / max(abs(F_last), 1e-300),
                   "dSr_at_Rt": ((bin_d + bin_p) / (4.0 * math.pi * float(Rt_rec) ** 2)) if Rt_rec else None,
                   "per_channel_in": [bch["in_dust"][j_] + bch["in_pairs"][j_] for j_ in range(len(chs))],
                   "per_channel_out": [bch["out_dust"][j_] + bch["out_pairs"][j_] for j_ in range(len(chs))]}
        if not inf_.get("failed"):
            rc = float(inf_["r_collar"]); nodes = np.arange(a.ell, rc - 2.0 * a.ell + 1e-9, a.ell)
            rho_n = np.asarray(inf_["rho_ell"](nodes), float); support = np.abs(rho_n) > 1e-6 * max(float(np.max(np.abs(rho_n))), 1e-300)
            hh = a.ell / 32.0
            zr = {"dust": node_rows(KF, R, a.ell, a.g, inf_, dD, fD, nodes, hh), "pairs": node_rows(KF, R, a.ell, a.g, inf_, dPp, fPp, nodes, hh)}
            Rt_tot = zr["dust"]["Rt"] + zr["pairs"]["Rt"]; Rr_tot = zr["dust"]["Rr"] + zr["pairs"]["Rr"]
            N0, v0, v10 = zr["dust"]["N"], zr["dust"]["v"], zr["dust"]["v1"]
            Hc2 = ((v10 / N0) ** 2 + 2.0 * (v0 / (nodes * N0)) ** 2) / 3.0
            sm = lambda arr: (float(np.nanmax(np.abs(arr[support]))) if support.any() and np.isfinite(arr[support]).any() else None)
            at_R = at_root(KF, R, a.ell, a.g, inf_, dD + dPp, fD + fPp, Rt_rec) if Rt_rec else None
            hk["zone"] = {"nodes": len(nodes), "support_nodes": int(support.sum()), "excluded_v0": nodes[~zr["dust"]["ok"]].tolist(),
                          "centre": {"r": float(nodes[0]), "Rt": float(Rt_tot[0]), "Rr": float(Rr_tot[0]), "H2": float(Hc2[0]),
                                     "Rt_over_H2": float(Rt_tot[0] / Hc2[0]) if Hc2[0] > 0 else None, "Rr_over_H2": float(Rr_tot[0] / Hc2[0]) if Hc2[0] > 0 else None,
                                     "dust_Rt": float(zr["dust"]["Rt"][0]), "pairs_Rt": float(zr["pairs"]["Rt"][0])},
                          "max_abs": {"Rt": sm(Rt_tot), "Rr": sm(Rr_tot), "Rt_dust": sm(zr["dust"]["Rt"]), "Rt_pairs": sm(zr["pairs"]["Rt"]),
                                      "Rr_dust": sm(zr["dust"]["Rr"]), "Rr_pairs": sm(zr["pairs"]["Rr"])},
                          "at_R_t": at_R,
                          "moments_max": {"d_dust": float(np.max(np.abs(dD))), "f_dust": float(np.max(np.abs(fD))), "d_pairs": float(np.max(np.abs(dPp))),
                                          "f_pairs": float(np.max(np.abs(fPp))), "eps_dust": float(np.max(np.abs(eps - epsP))), "eps_pairs": float(np.max(np.abs(epsP)))},
                          "d_over_rho_dust_sum": float(np.sum(dD)) / max(abs(float(np.sum(eps - epsP))), 1e-300)}
            hand["nodes"].append(nodes); hand["Rt_zone_dust"].append(zr["dust"]["Rt"]); hand["Rr_zone_dust"].append(zr["dust"]["Rr"])
            hand["Rt_zone_pairs"].append(zr["pairs"]["Rt"]); hand["Rr_zone_pairs"].append(zr["pairs"]["Rr"])
            hk["ledger"] = ledger_terms(inf_, R, a.g, eps, Jt, N_app, v_app, rec)
            hk["clipped"] = clipped(inf_, a.g, r_nodes)
        else:
            hk["zone"] = None; hk["ledger"] = None; hk["clipped"] = None
            for q in ("nodes", "Rt_zone_dust", "Rr_zone_dust", "Rt_zone_pairs", "Rr_zone_pairs"):
                hand[q].append(np.zeros(0))
        hand["d_dust"].append(dD); hand["f_dust"].append(fD); hand["d_pairs"].append(dPp); hand["f_pairs"].append(fPp)
        hand["b_in_dust"].append(np.asarray(bch["in_dust"])); hand["b_in_pairs"].append(np.asarray(bch["in_pairs"]))
        hand["b_out_dust"].append(np.asarray(bch["out_dust"])); hand["b_out_pairs"].append(np.asarray(bch["out_pairs"]))
        hand["eps_rep"].append(eps); hand["J_rep"].append(Jt); hand["eps_stored"].append(es); hand["J_stored"].append(Js)
        hk["seconds_hooks"] = t_hk + (time.time() - t_h)
        hooks.append(hk)
        # the reference's state after the record and the propagation (version 8's; the v17 path carries its gap and levels)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if v17:
                if "occ_e" in ci:
                    s_["levels_prev"] = (ci.pop("occ_e"), ci.pop("emp_e"), ci.get("partial_pairs", 0))
                if ci.get("gap_between_blocks") is not None or "weyl" in ci:
                    s_["gap_between_prev"] = ci.get("gap_between_blocks")
            elif cont == "ops v3":
                s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks"),
                                     "weyl_gap": ci.get("weyl_gap", ci.get("gap_between_blocks")), "mask": np.asarray(ci["_mask"], bool)})
                s_["window"] = s_["window"][-1:]
            else:
                s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks")})
                s_["window"] = s_["window"][-a.window_max:]
            ph = torch.exp(-1j * e.to(torch.complex128) * tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        if v17:
            f_ref = (N_app, v_app)
        del cache
        row = {"T_over_rs": T / rs, "eps_rel": d_tot[0], "J_rel": d_tot[1], "dust_rel": list(d_dust), "pairs_rel": list(d_pairs), "masks_equal_channels": nmask_equal,
               "masks_compared": bool(has_S), "masks_differ": mask_diff_ch, "mask_equals_negative_set": mask_neg, "stress_dust_rel": list(s_dust), "stress_pairs_rel": list(s_pairs),
               "pairs_F_max_abs": pmax, "gauss_identity": gid, "interval_closure": clos, "solve_failed": inf_.get("failed"), "seconds": time.time() - tr}
        rows.append(row)
        L.append("| %.3f | %.1e, %.1e | %.1e, %.1e | %.1e, %.1e | %s | %s | %s, %s | %s, %s (%s) | %+.2e | %s | %.1f |" % (
            T / rs, d_tot[0], d_tot[1], d_dust[0], d_dust[1], d_pairs[0], d_pairs[1],
            ("%d of %d%s" % (nmask_equal, len(chs), (" (differ: %s)" % "; ".join(mask_diff_ch)) if mask_diff_ch else "")) if has_S else "- (none stored)",
            mask_neg, fmt(s_dust[0], "%.1e"), fmt(s_dust[1], "%.1e"), fmt(s_pairs[0], "%.1e"), fmt(s_pairs[1], "%.1e"), fmt(pmax, "%.1e"), gid,
            ("%+.2e" % clos) if clos is not None else "-", row["seconds"]))
        zz = hk.get("zone") or {}
        print(L[-1] + " hooks: b_in %+.4e b_out %+.4e (b_in + b_out + F_last %.1e); R_t^zone centre %s (R_t/H^2 %s), max %s; R_r^zone max %s; clipped nodes %s; %.1f s" % (
            hk["b"]["in"], hk["b"]["out"], hk["b"]["identity"], fmt((zz.get("centre") or {}).get("Rt")), fmt((zz.get("centre") or {}).get("Rt_over_H2"), "%.4f"),
            fmt((zz.get("max_abs") or {}).get("Rt")), fmt((zz.get("max_abs") or {}).get("Rr")), (hk.get("clipped") or {}).get("n_nodes"), hk["seconds_hooks"]), flush=True)
        if i % 10 == 0 or i == K - 1:
            dump()
    write_md(L, rows, s1rows, hooks, chs, t_all, fld_m, tau, x)
    dump()


def write_md(L, rows, s1rows, hooks, chs, t_all, fld_m, tau, x):
    n = len(rows)
    eq_all = sum(1 for r in rows if r["masks_compared"] and r["masks_equal_channels"] == len(chs))
    ncmp = sum(1 for r in rows if r["masks_compared"])
    w = lambda key, j=None: max(((r[key][j] if j is not None else r[key]) for r in rows if ((r[key][j] if j is not None else r[key]) == (r[key][j] if j is not None else r[key]))), default=float("nan"))
    L += ["", "- %d native records replayed in %.0f s (from T/rs %.3f to %.3f); the masks equal the stored masks bitwise in every channel on %d of the %d records with a stored mask; the masks equal the negative-energy set on %d records." % (
        n, time.time() - t_all, rows[0]["T_over_rs"] if rows else float("nan"), rows[-1]["T_over_rs"] if rows else float("nan"), eq_all, ncmp, sum(1 for r in rows if r["mask_equals_negative_set"])),
          "- the largest source differences from the stored shell totals (each relative to its own largest value): total eps %.2e, J %.2e; dust eps %.2e, J %.2e; pairs eps %.2e, J %.2e (acceptance 1e-8)." % (
              w("eps_rel"), w("J_rel"), w("dust_rel", 0), w("dust_rel", 1), w("pairs_rel", 0), w("pairs_rel", 1)),
          "- the stresses against the stored ones (relative to the largest stored value; '-' where none is stored): dust F %s, f %s; pairs F %s, f %s." % (
              fmt(w("stress_dust_rel", 0), "%.2e"), fmt(w("stress_dust_rel", 1), "%.2e"), fmt(w("stress_pairs_rel", 0), "%.2e"), fmt(w("stress_pairs_rel", 1), "%.2e")),
          "- the Gauss identity E = sum_n eps_n <N^2>_n with the stored E and the replay's <N^2>_n (the dense lapse re-solved from the stored shell totals by %s): the largest |error| %.2e; the interval closure dE - (evolution + reset + lapse): the largest |value| %.2e." % (
              fld_m, max((abs(r["gauss_identity"]) for r in rows if r["gauss_identity"] == r["gauss_identity"]), default=float("nan")),
              max((abs(r["interval_closure"]) for r in rows if r["interval_closure"] is not None), default=float("nan")))]
    L += ["", "### Engine 28.2 (S1): the moving reference, the collar budget and the first-adiabatic current per interval (version 8's)", "",
          "| T/rs | Tr(h dP) spectral | Tr(h dP) density | Tr(h_R dP) | F_R[C1] (1e-6) | F_R[C1] (1e-4) | dropped weight (1e-6, 1e-4) | N_R dM_R | F_R dT | D_R dT | F_R[C1] + D_R (1e-6) | F_R[C1] + Tr(h_R dP)/tau | Gauss reset |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in s1rows:
        L.append("| %.3f | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %.2e, %.2e | %+.4e | %+.4e | %+.4e | %+.4e | %+.4e | %s |" % (
            q["T_over_rs"], q["tr_h_dP_spectral"], q["tr_h_dP_density"], q["tr_hR_dP"], q["F_C1_1e-6"], q["F_C1_1e-4"], q["dropped_1e-6"], q["dropped_1e-4"],
            q["N_R"] * q["dM_R"], q["F_R_dT"], q["D_R_dT"], q["residual_C1_plus_DR_1e-6"], q["residual_C1_plus_trhR_1e-6"], fmt(q["gauss_reset"], "%+.4e")))
    # version 9's hooks
    L += ["", "### Version 9's hooks per record (engine spec 37.5 (1)): (c) the origin pair, (a) the finite-zone rows, (e) the clipped intervals", "",
          "| T/rs | b_in: dust; pairs | b_out: dust; pairs | b_in + b_out + F_last (relative to F_last) | b_in/(4 pi R_t^2) | R_t^zone at the centre node (dust; pairs) | its ratio to H^2 there | max abs R_t^zone over the support (dust; pairs) | max abs R_r^zone (dust; pairs) | at R_t: 4 pi G (d + w f); 4 pi G (d - f); w | clipped: intervals, nodes; max abs R_E, R_J (R_E over its scale); dense points |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for hk in hooks:
        b = hk["b"]; z = hk.get("zone") or {}; c = z.get("centre") or {}; mx = z.get("max_abs") or {}; ar = z.get("at_R_t") or {}; cl = hk.get("clipped") or {}
        ivs = cl.get("intervals") or []
        L.append("| %.3f | %+.4e; %+.4e | %+.4e; %+.4e | %+.2e (%.1e) | %s | %s (%s; %s) | %s | %s (%s; %s) | %s (%s; %s) | %s; %s; %s | %d, %d; %s, %s (%s); %s |" % (
            hk["T_over_rs"], b["in_dust"], b["in_pairs"], b["out_dust"], b["out_pairs"], b["identity"], b["identity_rel"], fmt(b["dSr_at_Rt"], "%+.3e"),
            fmt(c.get("Rt"), "%+.4e"), fmt(c.get("dust_Rt"), "%+.3e"), fmt(c.get("pairs_Rt"), "%+.3e"), fmt(c.get("Rt_over_H2"), "%.4f"),
            fmt(mx.get("Rt"), "%.3e"), fmt(mx.get("Rt_dust"), "%.3e"), fmt(mx.get("Rt_pairs"), "%.3e"), fmt(mx.get("Rr"), "%.3e"), fmt(mx.get("Rr_dust"), "%.3e"), fmt(mx.get("Rr_pairs"), "%.3e"),
            fmt(ar.get("Rt_zone"), "%+.4e"), fmt(ar.get("Rt_zone_root_form"), "%+.4e"), fmt(ar.get("w"), "%+.4f"),
            len(ivs), cl.get("n_nodes") or 0, fmt(max((q["R_E_max_abs"] for q in ivs), default=None), "%.3e"), fmt(max((q["R_J_max_abs"] for q in ivs), default=None), "%.3e"),
            fmt(max((q["R_E_rel_scale"] for q in ivs), default=None), "%.2e"), cl.get("dense_points")))
    # the ledger per interval
    L += ["", "### The charge ledger per interval (37.5 (1) (d); ER21-ER23, P19 (11)): Delta E_G against a + b + c - Delta W + Delta K_out + Delta eps_adj + Delta eps_G", "",
          "| T/rs | E_G | Delta E_G | a + b + c (M1) | -Delta W | Delta K_out | Delta eps_adj | Delta eps_G | the remainder | M1's closure | cut changed | W (continuum; site form) | K; K_src; K_chi |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    prevL = None
    for hk in hooks:
        lg = hk.get("ledger")
        if not lg:
            prevL = None; continue
        row = "| %.3f | %s |" % (hk["T_over_rs"], fmt(lg["E_G"], "%.10f"))
        if prevL is not None and None not in (lg.get("a"), lg.get("b"), lg.get("c"), lg.get("W"), prevL.get("W"), lg.get("K_out"), prevL.get("K_out"),
                                               lg.get("eps_adj"), prevL.get("eps_adj"), lg.get("eps_G"), prevL.get("eps_G"), lg.get("E_G"), prevL.get("E_G")):
            dE = float(lg["E_G"]) - float(prevL["E_G"]); abc = float(lg["a"]) + float(lg["b"]) + float(lg["c"]); dW = lg["W"] - prevL["W"]
            dKo = lg["K_out"] - prevL["K_out"]; dEa = lg["eps_adj"] - prevL["eps_adj"]; dEg = lg["eps_G"] - prevL["eps_G"]
            rem = dE - (abc - dW + dKo + dEa + dEg)
            hk["ledger"]["interval"] = {"dE_G": dE, "abc": abc, "minus_dW": -dW, "dK_out": dKo, "deps_adj": dEa, "deps_G": dEg, "remainder": rem}
            row += " %+.6e | %+.6e | %+.6e | %+.6e | %+.6e | %+.6e | %+.2e | %s | %s |" % (dE, abc, -dW, dKo, dEa, dEg, rem, fmt(lg.get("M1_closure"), "%+.2e"), lg.get("cut_changed"))
        else:
            row += " - | - | - | - | - | - | - | %s | %s |" % (fmt(lg.get("M1_closure"), "%+.2e"), lg.get("cut_changed"))
        row += " %s; %s | %s; %s; %s |" % (fmt(lg.get("W"), "%+.6e"), fmt(lg.get("W_site"), "%+.6e"), fmt(lg.get("K"), "%.6f"), fmt(lg.get("K_src"), "%.6f"), fmt(lg.get("K_chi"), "%.6f"))
        L.append(row)
        prevL = lg
    ids = [abs(hk["b"]["identity_rel"]) for hk in hooks]
    ctr = [(hk["T_over_rs"], hk["zone"]["centre"]["Rt_over_H2"]) for hk in hooks if hk.get("zone") and hk["zone"]["centre"].get("Rt_over_H2") is not None]
    rems = [abs(hk["ledger"]["interval"]["remainder"]) for hk in hooks if hk.get("ledger") and hk["ledger"].get("interval")]
    L += ["", "- (c) the identity b_in + b_out = -F_last (ER18 through the stress module's prefix flux): the largest relative difference over %d records %s." % (len(ids), fmt(max(ids) if ids else None, "%.2e")),
          "- (a) R_t^zone/H^2 at the centre node over %d records: from %s to %s (ER14's full-band control %.9f at ma 0.2 is a homogeneous-core control, not a bound)." % (
              len(ctr), fmt(min(q[1] for q in ctr) if ctr else None, "%.4f"), fmt(max(q[1] for q in ctr) if ctr else None, "%.4f"), FULL_BAND_CONTROL),
          "- (d) the ledger's remainder (zero up to M1's own closure by ER23's construction): the largest |value| over %d intervals %s." % (len(rems), fmt(max(rems) if rems else None, "%.2e")),
          "- (e) records with clipped nodes: %d of %d." % (sum(1 for hk in hooks if (hk.get("clipped") or {}).get("n_nodes")), len(hooks)),
          "- every number above is printed by the script; S5 (b)'s R_hat against the certificate's direct residuals is --compose-rows' table; nothing is ruled."]
    open(x.md, "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------------ the self-test
def selftest(x):
    """the moment operators on a uniform open chain (m 0.2, the kappa term off): Delta = P_+ (the positive band) gives, in the bulk, d/rho =
    1 + p_Q/rho and f = 0 (ER14's one-dimensional analogue: d(q eps)/dk = cos 2k eps + q^2/eps over the zone, q = sin 2k/2, eps = (m^2 +
    sin^2 k)^(1/2)); the stencil identities [C2, Q] and [C2, T] on the bulk, i[h_b, Q] = B_in + B_out (ER18)."""
    import engine_vacflow_v1 as VF
    m = 0.2; n = 600
    R = np.arange(n, dtype=float) + 1.0
    hs, Qs = VF.chain_sparse(R, m, 0.0)
    T = chain_shift(n)
    C2 = 0.5 * (T @ T + (T @ T).T)
    I_ = sp.identity(2 * n, format="csr")
    Om = (I_ - C2).astype(complex)
    D = (0.5 * (Om @ hs + hs @ Om)).toarray(); Fo = (0.5 * (Om @ Qs + Qs @ Om)).toarray()
    e, U = np.linalg.eigh(hs.toarray())
    Up = U[:, e > 0]
    dens = lambda A, X: np.real(np.sum(X.conj() * (A @ X), axis=1)).reshape(n, 2).sum(axis=1)
    rho = dens(hs.toarray(), Up); J = dens(Qs.toarray(), Up); dd = dens(D, Up); ff = dens(Fo, Up)
    mid = slice(n // 2 - 50, n // 2 + 50)
    k = np.linspace(-np.pi, np.pi, 200001)[:-1]
    eps = np.sqrt(m * m + np.sin(k) ** 2); q = 0.5 * np.sin(2 * k)
    rho_i = float(np.mean(eps)); pQ_i = float(np.mean(q * q / eps)); ec_i = float(np.mean(np.cos(2 * k) * eps))
    Tm = T.toarray()
    bulk = slice(10, 2 * n - 10)
    c_Q = np.abs((C2 @ Qs - Qs @ C2).toarray()[bulk, bulk]).max(); c_T = np.abs((C2 @ T - T @ C2).toarray()[bulk, bulk]).max()
    hb = hs.toarray().copy()
    for i in range(n):
        hb[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 0.0
    Bc = 1j * (hb @ Qs.toarray() - Qs.toarray() @ hb)
    sx = np.array([[0, 1], [1, 0]], complex)
    Bin = np.zeros((2 * n, 2 * n), complex); Bout = np.zeros((2 * n, 2 * n), complex)
    Bin[2:4, 0:2] = 1j * sx / 8.0; Bin[0:2, 2:4] = -1j * sx / 8.0
    Bout[2 * n - 4:2 * n - 2, 2 * n - 2:] = 1j * sx / 8.0; Bout[2 * n - 2:, 2 * n - 4:2 * n - 2] = -1j * sx / 8.0
    er18 = float(np.abs(Bc - Bin - Bout).max())
    # b_terms against the trace with B_in, B_out on random columns
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2 * n, 5)) + 1j * rng.normal(size=(2 * n, 5)); wts = rng.normal(size=5)
    tr_in = float(np.real(np.sum(wts * np.sum(X.conj() * (Bin @ X), axis=0)))); tr_out = float(np.real(np.sum(wts * np.sum(X.conj() * (Bout @ X), axis=0))))
    import torch
    bi, bo = b_terms(torch.as_tensor(X), torch.as_tensor(wts))
    out = {"rho_bulk": float(np.mean(rho[mid])), "rho_integral": rho_i, "d_over_rho_bulk": float(np.mean(dd[mid] / rho[mid])), "one_plus_pQ_over_rho": 1.0 + pQ_i / rho_i,
           "ec_integral_plus_pQ": ec_i + pQ_i, "d_spread_bulk": float(np.std(dd[mid] / rho[mid])), "J_bulk_max": float(np.abs(J[mid]).max()), "f_bulk_max": float(np.abs(ff[mid]).max()),
           "C2_Q_commutator_bulk": float(c_Q), "C2_T_commutator_bulk": float(c_T), "ER18_identity": er18, "b_in_vs_trace": abs(bi - tr_in), "b_out_vs_trace": abs(bo - tr_out)}
    ok = (abs(out["d_over_rho_bulk"] - out["one_plus_pQ_over_rho"]) < 1e-8 and out["f_bulk_max"] < 1e-10 and out["ER18_identity"] < 1e-14 and c_Q < 1e-15
          and out["b_in_vs_trace"] < 1e-12 and out["b_out_vs_trace"] < 1e-12 and abs(out["ec_integral_plus_pQ"]) < 1e-10)
    out["pass"] = bool(ok)
    L = ["## engine_replay_v9 --selftest: the moment operators and the origin pair on a uniform open chain (m %g, %d sites, the kappa term off)" % (m, n), ""]
    L += ["- the positive band Delta = P_+: in the bulk (100 middle sites) rho %.12f against the zone integral %.12f; d/rho %.12f (spread %.1e) against 1 + p_Q/rho %.12f; the zone identity <cos 2k eps> + p_Q = %.1e; max |J| %.1e, max |f| %.1e (both 0 expected)." % (
        out["rho_bulk"], out["rho_integral"], out["d_over_rho_bulk"], out["d_spread_bulk"], out["one_plus_pQ_over_rho"], out["ec_integral_plus_pQ"], out["J_bulk_max"], out["f_bulk_max"]),
          "- the stencil: max |[C2, Q]| %.1e and |[C2, T]| %.1e on the bulk; i[h_b, Q] - B_in - B_out max %.1e (ER18); b_terms against Tr[D B_in], Tr[D B_out] on random columns %.1e, %.1e." % (
              out["C2_Q_commutator_bulk"], out["C2_T_commutator_bulk"], out["ER18_identity"], out["b_in_vs_trace"], out["b_out_vs_trace"]),
          "- the self-test: %s." % ("PASS" if ok else "FAIL")]
    open(x.md, "w").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)
    if x.out_json:
        json.dump(out, open(x.out_json, "w"), indent=1, default=float)


# ------------------------------------------------------------------------------------------------------------------ S5 (b): R_hat against the certificate
def compose_rows(x):
    import engine_certificate_v3 as CERT
    import engine_stress_v2 as STR2
    KF = importlib.import_module("engine_fields_v12")
    paths = x.compose_rows.split(",")
    HS = [np.load(p, allow_pickle=True) for p in paths]
    RJ = [json.load(open(p)) for p in x.replay_json.split(",")] if x.replay_json else []
    hooks_by_T = {}
    for rj in RJ:
        for hk in rj.get("hooks", []):
            hooks_by_T[round(float(hk["T_over_rs"]), 6)] = hk
    Z = np.load(x.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    # the handoffs' records by time (the later handoff wins a repeated time)
    mom = {}
    for H in HS:
        Th = np.asarray(H["T"], float)
        for j, t in enumerate(Th):
            mom[round(float(t) / rs, 6)] = {"d_dust": np.asarray(H["d_dust"][j], float), "f_dust": np.asarray(H["f_dust"][j], float),
                                            "d_pairs": np.asarray(H["d_pairs"][j], float), "f_pairs": np.asarray(H["f_pairs"][j], float),
                                            "eps_R2": np.asarray(H["eps_R2"][j], float), "J_R2": np.asarray(H["J_R2"][j], float), "T": float(t)}
    times = sorted(mom.keys())
    seg_names = [str(q) for q in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(q.split("=", 1) for q in x.fm_segments.split(";")) if x.fm_segments else {}
    fms = {}

    def fm_for(k):
        name = fmap.get(seg_names[seg[k]], x.fm_default) if seg_names else x.fm_default
        if name not in fms:
            fms[name] = importlib.import_module(name)
        return fms[name], name
    cache = {}

    def fields(k):
        """engine_certificate_v3.main's fields(), copied: the dense fields re-solved from the data's shell totals by the segment's module."""
        if k in cache:
            return cache[k]
        FM, name = fm_for(k)
        e = np.asarray(Z["eps_tot"][k], float); Jt = np.asarray(Z["J_tot"][k], float)
        _, _, _, info = FM.solve_constraints_ivp(e, Jt, g, R, ell, collar_policy=x.collar_policy, support_rel=x.support_rel)
        Sr = np.asarray(Z["S_r_dust"][k], float) + np.asarray(Z["S_r_pairs"][k], float)
        St = np.asarray(Z["S_t_dust"][k], float) + np.asarray(Z["S_t_pairs"][k], float)
        F = {"N": info["N_dense"], "v": info["v_dense"], "rho": info["rho_ell"], "J": info["J_ell"],
             "sr": (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[0]),
             "st": (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[1]),
             "info": info, "fm": name, "failed": bool(info.get("failed"))}
        cache[k] = F
        for old in [q for q in cache if q < k - 2]:
            del cache[old]
        return F

    def smooth(vals, scalar):
        sel = np.abs(vals) > 0
        Rs = R[sel]; vs = vals[sel]

        def fn(r):
            r = np.atleast_1d(np.asarray(r, float))
            if not sel.any():
                return np.zeros_like(r)
            K0, K1 = KF.kernels(r, Rs, ell)
            return (K0 if scalar else K1) @ vs
        return fn

    def shares(pred, res, Nn, support):
        m = support & np.isfinite(pred) & np.isfinite(res)
        if not m.any():
            return None
        ag = np.abs(Nn[m] * pred[m] - res[m]) / np.maximum(np.abs(res[m]), 1e-300)
        return {"share_within_0.1": float(np.mean(ag <= 0.1)), "median": float(np.median(ag)), "max": float(ag.max()), "n": int(m.sum())}
    ks = list(range(len(T))) if x.records == "all" else [int(q) for q in x.records.split(",")]
    out_rows = []
    t0 = time.time()
    for k in ks:
        key = round(T[k] / rs, 6)
        if key not in mom:
            out_rows.append({"k": k, "T_over_rs": T[k] / rs, "missing": "no replay record at this time"}); continue
        F0 = fields(k)
        if F0["failed"]:
            out_rows.append({"k": k, "T_over_rs": T[k] / rs, "failed": True}); continue
        Fm = fields(k - 1) if k - 1 >= 0 else None
        Fp = fields(k + 1) if k + 1 < len(T) else None
        rc = float(F0["info"]["r_collar"]); nodes = np.arange(ell, rc - 2.0 * ell + 1e-9, ell)
        c = CERT.certificate(Fm, F0, Fp, (T[k] - T[k - 1]) if Fm else None, (T[k + 1] - T[k]) if Fp else None, g, nodes, x.gl, ell / x.hfrac)
        rho_n = c["rho_nodes"]; support = np.abs(rho_n) > 1e-6 * max(float(np.max(np.abs(rho_n))), 1e-300)
        mk = mom[key]
        h = ell / x.hfrac
        zt = node_rows(KF, R, ell, g, F0["info"], mk["d_dust"] + mk["d_pairs"], mk["f_dust"] + mk["f_pairs"], nodes, h)
        zd = node_rows(KF, R, ell, g, F0["info"], mk["d_dust"], mk["f_dust"], nodes, h)
        # the reference rate, centred over the neighbouring replay records (one-sided at the ends)
        i_ = times.index(key)
        jm = times[i_ - 1] if i_ > 0 else None; jp = times[i_ + 1] if i_ + 1 < len(times) else None
        if jm is not None and jp is not None:
            dt_ = mom[jp]["T"] - mom[jm]["T"]; qr = (mom[jp]["eps_R2"] - mom[jm]["eps_R2"]) / dt_; qj = (mom[jp]["J_R2"] - mom[jm]["J_R2"]) / dt_; how = "centred"
        elif jp is not None:
            dt_ = mom[jp]["T"] - mk["T"]; qr = (mom[jp]["eps_R2"] - mk["eps_R2"]) / dt_; qj = (mom[jp]["J_R2"] - mk["J_R2"]) / dt_; how = "one-sided (forward)"
        else:
            dt_ = mk["T"] - mom[jm]["T"]; qr = (mk["eps_R2"] - mom[jm]["eps_R2"]) / dt_; qj = (mk["J_R2"] - mom[jm]["J_R2"]) / dt_; how = "one-sided (backward)"
        qrl = smooth(qr, True); qjl = smooth(qj, False)
        xg, wg = np.polynomial.legendre.leggauss(int(x.gl))
        edges = np.concatenate([[0.0], nodes]); a_ = edges[:-1][:, None]; b_ = edges[1:][:, None]
        pts = (0.5 * (b_ - a_) * xg[None, :] + 0.5 * (b_ + a_)).ravel()
        Np = np.asarray(F0["N"](pts), float); vp = np.asarray(F0["v"](pts), float)
        integ = (pts ** 2 * (Np * qrl(pts) + vp * qjl(pts))).reshape(len(nodes), len(xg))
        cumn = np.cumsum((integ * wg[None, :]).sum(axis=1) * 0.5 * (edges[1:] - edges[:-1]))
        Nn = zt["N"]; vn = zt["v"]; v1n = zt["v1"]; ok = np.abs(vn) >= 1e-12
        A = v1n / Nn; B = vn / (nodes * Nn)
        Rt_ref = np.where(ok, -4.0 * math.pi * g * cumn / (Nn * nodes ** 2 * np.where(ok, vn, 1.0)), np.nan)
        Rr_ref = np.where(ok, (-4.0 * math.pi * g * qrl(nodes) / Nn - (A + B) * Rt_ref) / np.where(ok, B, 1.0), np.nan)
        Rt_hat = Rt_ref + zt["Rt"]; Rr_hat = Rr_ref + zt["Rr"]
        row = {"k": k, "T_over_rs": T[k] / rs, "fm": F0["fm"], "time_derivative": c["how"], "reference_rate": how, "nodes": len(nodes), "support_nodes": int(support.sum()),
               "tt": {"hat": shares(Rt_hat, c["res_tt"], c["N_nodes"], support), "zone": shares(zt["Rt"], c["res_tt"], c["N_nodes"], support),
                      "ref": shares(Rt_ref, c["res_tt"], c["N_nodes"], support), "quadrature": shares(c["Rt_q"], c["res_tt"], c["N_nodes"], support)},
               "rr": {"hat": shares(Rr_hat, c["res_rr"], c["N_nodes"], support), "zone": shares(zt["Rr"], c["res_rr"], c["N_nodes"], support),
                      "ref": shares(Rr_ref, c["res_rr"], c["N_nodes"], support), "quadrature": shares(c["Rr_q"], c["res_rr"], c["N_nodes"], support)}}
        m_ = support & np.isfinite(c["Rt_q"]) & np.isfinite(Rt_hat)
        if m_.any():
            sc_t = np.maximum(np.abs(c["Rt_q"][m_]), 1e-300)
            row["hat_vs_quadrature_tt_median"] = float(np.median(np.abs(Rt_hat[m_] - c["Rt_q"][m_]) / sc_t))
            row["zone_dust_share_of_zone_median"] = float(np.median(np.abs(zd["Rt"][m_]) / np.maximum(np.abs(zt["Rt"][m_]), 1e-300)))
        rmax = lambda arr: float(np.nanmax(np.abs(arr[support]))) if support.any() and np.isfinite(arr[support]).any() else None
        row["max_abs"] = {"Rt_zone": rmax(zt["Rt"]), "Rt_ref": rmax(Rt_ref), "Rt_quadrature": rmax(c["Rt_q"]), "res_tt_over_N": rmax(c["res_tt"] / c["N_nodes"]),
                          "Rr_zone": rmax(zt["Rr"]), "Rr_ref": rmax(Rr_ref), "Rr_quadrature": rmax(c["Rr_q"]), "res_rr_over_N": rmax(c["res_rr"] / c["N_nodes"])}
        row["hooks"] = hooks_by_T.get(key)
        out_rows.append(row)
        if len(out_rows) % 20 == 1:
            print("record %d (T/rs %.3f) done, %.0f s" % (k, T[k] / rs, time.time() - t0), flush=True)
    # the table
    f_ = lambda q, k_: fmt((q or {}).get(k_), "%.3f")
    L = ["## S5 (b): R_hat = R^ref + R^zone against engine_certificate_v3's direct residuals per node, and the moments table (engine_replay_v9.py --compose-rows; engine spec 37.5 (1); P35 5; the icloud lane's ER11, ER15)", "",
         "- inputs: the replay handoffs %s; the certificate's data %s (g %g, ell %g, r_s %g; %d records; field modules %s); derivatives h = ell/%g, %d-point Gauss-Legendre rules, the support rho_ell above 1e-6 of its largest value (the certificate's)." % (
             ", ".join(os.path.basename(p) for p in paths), os.path.basename(x.data), g, ell, rs, len(T), x.fm_segments or x.fm_default, x.hfrac, x.gl),
         "- R^zone = ER11 from the replay's moments (dust plus pairs); R^ref from the replay's reference densities, the rate centred over the neighbouring records, R_t^ref = -(4 pi G/(N r^2 v)) int_0^r s^2 (N q_rho + v q_J) ds, R_r^ref = (-4 pi G q_rho/N - (A + B) R_t^ref)/B; the agreement |N R - res_direct|/|res_direct| at the support nodes; its share within 1/10 is the criterion's quantity (37.5 (1)); nothing is ruled. 'quadrature' is the certificate's own reconstruction from the continuity defects (its share reproduces the certificate's table).", "",
         "| T/rs | d_t; reference rate | tt: R_hat share (median) | tt: R^zone share | tt: R^ref share | tt: the quadrature's share | rr: R_hat share (median) | rr: R^zone share | rr: R^ref share | rr: the quadrature's share | max abs R_t: zone; ref; quadrature; res/N | median abs(R_hat - R_q)/abs(R_q) (tt) | the hooks: b_in; R_t^zone/H^2 at the centre; clipped nodes |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        if r.get("missing") or r.get("failed"):
            L.append("| %.3f | %s | - | - | - | - | - | - | - | - | - | - | - |" % (r["T_over_rs"], r.get("missing") or "the solve failed")); continue
        tt = r["tt"]; rr = r["rr"]; mx = r["max_abs"]; hk = r.get("hooks") or {}
        L.append("| %.3f | %s; %s | %s (%s) | %s | %s | %s | %s (%s) | %s | %s | %s | %s; %s; %s; %s | %s | %s; %s; %s |" % (
            r["T_over_rs"], "c" if r["time_derivative"] == "centred" else r["time_derivative"], "c" if r["reference_rate"] == "centred" else r["reference_rate"],
            f_(tt["hat"], "share_within_0.1"), fmt((tt["hat"] or {}).get("median"), "%.2e"), f_(tt["zone"], "share_within_0.1"), f_(tt["ref"], "share_within_0.1"), f_(tt["quadrature"], "share_within_0.1"),
            f_(rr["hat"], "share_within_0.1"), fmt((rr["hat"] or {}).get("median"), "%.2e"), f_(rr["zone"], "share_within_0.1"), f_(rr["ref"], "share_within_0.1"), f_(rr["quadrature"], "share_within_0.1"),
            fmt(mx["Rt_zone"], "%.2e"), fmt(mx["Rt_ref"], "%.2e"), fmt(mx["Rt_quadrature"], "%.2e"), fmt(mx["res_tt_over_N"], "%.2e"), fmt(r.get("hat_vs_quadrature_tt_median"), "%.2e"),
            fmt((hk.get("b") or {}).get("in"), "%+.3e"), fmt(((hk.get("zone") or {}).get("centre") or {}).get("Rt_over_H2"), "%.4f"), (hk.get("clipped") or {}).get("n_nodes")))
    good = [r for r in out_rows if not (r.get("missing") or r.get("failed"))]

    def dist(row_, kind):
        v_ = [r[row_][kind]["share_within_0.1"] for r in good if r[row_][kind] is not None]
        return (min(v_), float(np.mean(v_)), max(v_), sum(1 for q in v_ if q >= 0.8), len(v_)) if v_ else None
    L.append("")
    for row_ in ("tt", "rr"):
        for kind in ("hat", "zone", "ref", "quadrature"):
            dd = dist(row_, kind)
            L.append("- %s, %s: the share within 1/10 over %s records: min %s, mean %s, max %s; records with share at least 0.800: %s." % (
                row_, {"hat": "R_hat = R^ref + R^zone", "zone": "R^zone alone", "ref": "R^ref alone", "quadrature": "the certificate's quadrature (its own table's check)"}[kind],
                dd[4] if dd else 0, fmt(dd[0] if dd else None, "%.3f"), fmt(dd[1] if dd else None, "%.3f"), fmt(dd[2] if dd else None, "%.3f"), dd[3] if dd else "-"))
    L += ["- every number above is printed by the script (%d records in %.0f s); nothing is ruled (37.5 (1))." % (len(out_rows), time.time() - t0)]
    open(x.md, "w").write("\n".join(L) + "\n")
    if x.out_json:
        json.dump({"script": "engine_replay_v9.py --compose-rows", "handoffs": paths, "data": x.data, "rows": out_rows}, open(x.out_json, "w"), indent=1, default=float)
    print("\n".join(L[-10:]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--resume-from", default=None, help="the stored v17 run's dir/tag on this host (overrides the run's own path)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--tag", default=None, help="accepted for the GPU runner, unused")
    ap.add_argument("--engine", default=None); ap.add_argument("--fields", default=None); ap.add_argument("--stress", default=None)
    ap.add_argument("--compare-prefix", default=None, help="a v17 run: replay v2's <prefix>_handoff.npz and _spectra.npz of the same run")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compose-rows", default=None, help="the replay handoffs (time order, comma list)")
    ap.add_argument("--replay-json", default=None, help="the matching replay jsons (comma list)")
    ap.add_argument("--data", default=None, help="the certificate's data npz (COLLAPSE_records)")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--collar-policy", default="flag"); ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--rows-records", dest="records_rows", default="all")
    ap.add_argument("--hfrac", type=float, default=32.0); ap.add_argument("--gl", type=int, default=8)
    x = ap.parse_args()
    if x.selftest:
        return selftest(x)
    if x.compose_rows:
        x.records = x.records_rows
        return compose_rows(x)
    if not (x.run and x.md and x.out_json and x.out_npz):
        raise SystemExit("engine_replay_v9: --run, --md, --out-json and --out-npz are required for a replay")
    replay(x)


if __name__ == "__main__":
    main()
