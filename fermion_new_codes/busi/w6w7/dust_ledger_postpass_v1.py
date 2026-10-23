#!/usr/bin/env python
"""dust_ledger_postpass_v1.py -- W7 (busi lane): the ledger of requirements edit 24 as a post-pass over DUST_root_history v2b's outputs.

Reads dust_root_history_v2.json and .npz, which dust_root_history_v2b.py rewrites after every completed launch key, and the history's
records; dust_root_history_v2b.py is neither edited nor stopped. Per launch key it prints (W7_dust_v2b_audit.md, section 5):

1. The same-set fixed-cut ledger (no engine). With p_j = f . (norm_j - beyond_Rt_j) the occupation-weighted weight of the key's columns
   inside the root at record j, T_j the net outward transport through the OLD cut R_j over the step j -> j+1 and S_j the sweep of the new mask,
       p_(j+1) - p_j = -T_j + S_j,
   S_j = -moving_cut_step[j+1] (= -f . (Wall_(j+1)[R > R_(j+1)] - Wall_(j+1)[R > R_j])) and
   T_j = (f . beyond_Rt[j+1] - moving_cut_step[j+1]) - f . beyond_Rt[j] (= f . (Wall_(j+1)[R > R_j] - Wall_j[R > R_j]));
   the residual p_(j+1) - p_j + T_j - S_j is the norm drift. v2b's instantaneous current through the record's own root times the step
   prints beside as the one-point quadrature of T_j. Both cuts (dense: the record's trapping zeros; applied: the applied fields' zeros).
2. The never-inside capacity at the record's own charge: C(t) = (E(t) - R_t(t)/(2G))/eps_p, R_t/(2G) the charge a root of N + v encloses
   (M = r u^2/(2G) at u = -1), eps_p = m and 1.2 m; against the block beyond 16 and 24 r_s; at E(t) and at E_0.
3. The same-packet inside-origin certificate per column c: [sqrt(p_late,c) - sqrt(eps_early,c)]_+^2 <= the weight of the column beyond the cut
   at the late record that was inside the launch root (the triangle inequality under unitary propagation), eps_early,c = beyond_Rt[0, c]/norm[0, c],
   p_late,c = b16_all[j, c]/norm[j, c] (and b24); weighted by f; the upper bound min(1 - eps_early, (sqrt p + sqrt eps)^2) beside; the check
   cert <= b16_in (v2b's launch-tagged split, which the certificate replaces) per column.
4. (engine) At the held records: the key's state regenerated with v2b's own propagator from the stored columns (checked against v2b's stored
   per-column weights at that record), the exterior block X = 1_(R > cut) S, its spectral weights ||1_bin(h) X_c||^2 on the held operator
   h(N(t), v(t)) in the 5 bins of horizon3_bins_v1, the extended/bound split of the (m, b] eigenvectors by their weight inside R_cut, and the
   escape certificate f . w_ac(m,b] / f . ||X||^2, certified above --escape-certify (0.92); the whole-column bins checked against v2b's bins_end.

CPU; the thread cap by --threads (8 beside the running v2b, 16 after it). No new production run; no plotting; no dates.
"""
import os
import sys


def _pre_threads(default="8"):  # W7
    t = default
    for i, s in enumerate(sys.argv):
        if s == "--threads" and i + 1 < len(sys.argv):
            t = sys.argv[i + 1]
        elif s.startswith("--threads="):
            t = s.split("=", 1)[1]
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, t)


_pre_threads()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402

T0 = time.time()
CUTS = ("dense", "applied")
BIN_LABELS = ["(-inf,-b]", "(-b,-m]", "(-m,m]", "(m,b]", "(b,inf)"]


def log(msg):  # W7
    print("[+%7.0f s] %s" % (time.time() - T0, msg), flush=True)


def md5(path, block=1 << 22):  # W7
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fmt(x, nd=6):  # W7
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "True" if x else "False"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if math.isnan(xf) else ("%%.%dg" % nd) % xf


def _jdefault(o):  # W7
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(repr(o))


# ---------------------------------------------------------------- the records without the engine (W7)
def load_record_list(path):
    """The collapse json's record list (the dicts carrying T_over_rs) and the run args."""
    with open(path) as fh:
        head = json.load(fh)
    run_args = head.get("args", {}) if isinstance(head, dict) and isinstance(head.get("args"), dict) else {}
    records = None
    if isinstance(head, dict):
        for key in ("passes", "records", "history", "members", "steps"):
            val = head.get(key)
            if isinstance(val, list) and val and all(isinstance(x, dict) for x in val) and any("T_over_rs" in x for x in val):
                records = val
                break
        if records is None:
            for key, val in head.items():
                if isinstance(val, list) and val and all(isinstance(x, dict) for x in val) and any("T_over_rs" in x for x in val):
                    records = val
                    break
    elif isinstance(head, list):
        records = head
    if records is None:
        raise SystemExit("no list of record dicts carrying T_over_rs in %s" % path)
    return run_args, records


def _tors(rec, rs):
    t = rec.get("T_over_rs")
    if t is None:
        t = rec.get("T")
        return float("nan") if t is None else float(t) / rs
    return float(t)


def history_table(rec1, rec2, switch_T, rs):
    """T/r_s -> {E, Rt (max trapping zero), Ri, src}; the first file's records up to the switch clock, the second's beyond; E_0 the first
    record's E of the first file. Where both files carry a clock, both E values are kept (E_other)."""
    tab = {}
    args1, recs1 = load_record_list(rec1)
    E0 = None
    for rec in recs1:
        t = _tors(rec, rs)
        if not np.isfinite(t):
            continue
        if E0 is None and rec.get("E") is not None:
            E0 = float(rec["E"])
        z = [float(x) for x in (rec.get("trapping_zeros") or []) if x is not None]
        tab[round(t, 6)] = {"T": t, "E": rec.get("E"), "Rt": max(z) if z else rec.get("R_t"), "Ri": (min(z) if len(z) > 1 else rec.get("r_in")), "src": 1}
    args2 = {}
    if rec2 and os.path.exists(rec2):
        args2, recs2 = load_record_list(rec2)
        for rec in recs2:
            t = _tors(rec, rs)
            if not np.isfinite(t):
                continue
            z = [float(x) for x in (rec.get("trapping_zeros") or []) if x is not None]
            entry = {"T": t, "E": rec.get("E"), "Rt": max(z) if z else rec.get("R_t"), "Ri": (min(z) if len(z) > 1 else rec.get("r_in")), "src": 2}
            key = round(t, 6)
            if key in tab and t <= switch_T + 1e-9:
                tab[key]["E_other"] = rec.get("E")
            else:
                if key in tab:
                    entry["E_other"] = tab[key].get("E")
                tab[key] = entry
    return tab, E0, args1, args2


def lookup(tab, t):
    return tab.get(round(float(t), 6))


# ---------------------------------------------------------------- the ledger from v2b's arrays (W7)
def ledger_arrays(Z, key, cut, rs):
    """The same-set fixed-cut ledger per record step j-1 -> j from the stored arrays (the row j describes the step whose old cut is R_(j-1))."""
    T = np.asarray(Z["%s_T" % key], float)
    f = np.asarray(Z["%s_f" % key], float)
    beyond = np.asarray(Z["%s_beyond_Rt_%s" % (key, cut)], float)
    mcs = np.asarray(Z["%s_moving_cut_step_%s" % (key, cut)], float)
    norm = np.asarray(Z["%s_norm" % key], float)
    Rt = np.asarray(Z["%s_Rt_%s" % (key, cut)], float)
    cur = np.asarray(Z["%s_current_outer_%s" % (key, cut)], float)
    nr = T.size
    p_out = beyond @ f
    p_norm = norm @ f
    p_in = p_norm - p_out
    out = {k: np.full(nr, np.nan) for k in ("T_j", "S_j", "dp_in", "resid", "q_end", "q_trap", "dt")}
    for j in range(1, nr):
        if np.isfinite(Rt[j - 1]) and np.isfinite(Rt[j]) and np.isfinite(mcs[j]):
            S = -mcs[j]
            Tj = (p_out[j] - mcs[j]) - p_out[j - 1]
            d = p_in[j] - p_in[j - 1]
            dtj = (T[j] - T[j - 1]) * rs
            out["T_j"][j] = Tj
            out["S_j"][j] = S
            out["dp_in"][j] = d
            out["resid"][j] = d + Tj - S
            out["dt"][j] = dtj
            out["q_end"][j] = cur[j - 1] * dtj
            out["q_trap"][j] = 0.5 * (cur[j - 1] + cur[j]) * dtj
    out.update(T=T, p_in=p_in, p_out=p_out, p_norm=p_norm, Rt=Rt, f=f)
    return out


def episode_sums(L):
    """Sums of T_j and S_j over the maximal runs of consecutive rooted steps."""
    fin = np.isfinite(L["T_j"])
    runs = []
    j = 0
    nr = fin.size
    while j < nr:
        if fin[j]:
            k = j
            while k + 1 < nr and fin[k + 1]:
                k += 1
            sl = slice(j, k + 1)
            runs.append({"T_from": float(L["T"][j - 1]), "T_to": float(L["T"][k]), "n_steps": int(k - j + 1), "sum_T": float(np.sum(L["T_j"][sl])),
                         "sum_S": float(np.sum(L["S_j"][sl])), "sum_dp_in": float(np.sum(L["dp_in"][sl])), "max_abs_resid": float(np.max(np.abs(L["resid"][sl]))),
                         "sum_q_end": float(np.sum(L["q_end"][sl])), "p_in_start": float(L["p_in"][j - 1]), "p_in_end": float(L["p_in"][k])})
            j = k + 1
        else:
            j += 1
    return runs


# ---------------------------------------------------------------- the capacity (W7)
def capacity_rows(tab, T, Rt, G, m, eps_units, E0, blocks16, blocks24):
    rows = []
    for j in range(T.size):
        ent = lookup(tab, T[j])
        if ent is None or ent.get("E") is None or not np.isfinite(Rt[j]):
            continue
        E = float(ent["E"])
        Mroot = float(Rt[j]) / (2.0 * G)
        row = {"T": float(T[j]), "E": E, "E_over_E0": (E / E0) if E0 else None, "Rt": float(Rt[j]), "M_root": Mroot, "E_ext": E - Mroot,
               "E_ext_E0": (E0 - Mroot) if E0 is not None else None, "block16": float(blocks16[j]), "block24": float(blocks24[j])}
        for eu in eps_units:
            row["C_at_%gm" % eu] = (E - Mroot) / (eu * m)
            row["C_E0_at_%gm" % eu] = ((E0 - Mroot) / (eu * m)) if E0 is not None else None
            row["fits_at_%gm" % eu] = bool(float(blocks16[j]) <= (E - Mroot) / (eu * m))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- the inside-origin certificate (W7)
def certificates(Z, key, cut):
    f = np.asarray(Z["%s_f" % key], float)
    norm = np.asarray(Z["%s_norm" % key], float)
    beyond0 = np.asarray(Z["%s_beyond_Rt_%s" % (key, cut)], float)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        eps_early = beyond0 / norm[0]
    out = {"eps_early": eps_early, "weighted_eps_early": float(np.dot(f, np.where(np.isfinite(eps_early), eps_early, 0.0)))}
    for c in ("b16", "b24"):
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.asarray(Z["%s_%s_all" % (key, c)], float) / norm
            lower = np.clip(np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(eps_early, 0, None))[None, :], 0.0, None) ** 2
            upper = np.minimum((np.sqrt(np.clip(p, 0, None)) + np.sqrt(np.clip(eps_early, 0, None))[None, :]) ** 2, (1.0 - eps_early)[None, :])
            upper = np.minimum(upper, p)
            b_in = np.asarray(Z["%s_%s_in_%s" % (key, c, cut)], float) / norm
        out[c] = {"lower": lower, "upper": upper, "lower_weighted": lower @ f, "upper_weighted": upper @ f, "p_weighted": p @ f,
                  "b_in_weighted": b_in @ f, "n_violations_lower_gt_b_in": int(np.sum(np.isfinite(b_in) & (lower > b_in + 1e-9)))}
    return out


# ---------------------------------------------------------------- the engine step (W7)
def import_v2b(cands):
    for d in cands:
        if d and os.path.exists(os.path.join(d, "dust_root_history_v2b.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import dust_root_history_v2b as D  # noqa: E402
            return D, d
    raise SystemExit("dust_root_history_v2b.py not found in %s; pass --v2b-dir" % [c for c in cands if c])


def regenerate_state(D, BE, OPS, H, h0, Q, W, k0, k_held):
    """v2b's propagation of the launch columns from the launch record k0 to the record k_held: at every record the previous record's frozen
    clocked operator acts over the record step (the same statements as v2b's loop)."""
    S = np.array(W, np.complex128)
    Kmax = 0
    h_prev = None
    for kk in range(k0 + 1, k_held + 1):
        if h_prev is None:
            Np, vp = D.fields(H, kk - 1)
            h_prev = OPS.clocked_csr(h0, Q, Np, vp)
        tau = float(H.T[kk] - H.T[kk - 1])
        S, K = BE.propagate(h_prev, S, tau)
        Kmax = max(Kmax, int(K))
        Nk, vk = D.fields(H, kk)
        h_prev = OPS.clocked_csr(h0, Q, Nk, vk)
    if h_prev is None:
        Nk, vk = D.fields(H, k_held)
        h_prev = OPS.clocked_csr(h0, Q, Nk, vk)
    return S, Kmax, h_prev


def bin_index(e, m, b):
    """The 5 half-open bins (lo, hi] of horizon3_bins_v1: (-inf,-b], (-b,-m], (-m,m], (m,b], (b,inf)."""
    return np.searchsorted(np.array([-b, -m, m, b], float), np.asarray(e, float), side="left")


def spectral_weights(U, X):
    """|<u_k|X_c>|^2 for every eigenvector k and column c, without a copy of U: (X^H U)^H."""
    V = (X.conj().T @ U)
    return (np.abs(V) ** 2).T


def exterior_certificate(e, U, S, R, rs, m, b, f, cuts_rs, bound_cuts_rs, thresholds):
    """The exterior block's spectral weights on the held operator at the cuts, the extended/bound split within (m, b] and the escape fraction."""
    n = R.size
    idx = bin_index(e, m, b)
    inband = idx == 3
    win = {}
    for Rc in bound_cuts_rs:
        n_in = int(np.sum(R <= Rc * rs))
        win[Rc] = (np.abs(U[:2 * n_in, :]) ** 2).sum(axis=0)
    res = {"n_in_band": int(inband.sum()), "e_min": float(e.min()), "e_max": float(e.max()), "bin_counts": [int(np.sum(idx == k)) for k in range(5)],
           "win": {Rc: {"n_bound_in_band": {thr: int(np.sum(inband & (win[Rc] >= thr))) for thr in thresholds},
                        "in_band_inside_weight_quantiles": [float(x) for x in np.quantile(win[Rc][inband], [0.0, 0.1, 0.5, 0.9, 1.0])] if inband.any() else None}
                   for Rc in bound_cuts_rs},
           "cuts": {}}
    for cut in cuts_rs:
        mask = np.repeat(R > cut * rs, 2)
        X = S * mask[:, None]
        w = spectral_weights(U, X)
        tot = w.sum(axis=0)
        bins = np.stack([w[idx == k].sum(axis=0) for k in range(5)])
        entry = {"total_per_column": tot, "total_weighted": float(np.dot(f, tot)), "bins_per_column": bins, "bins_weighted": bins @ f, "split": {}}
        for Rc in bound_cuts_rs:
            for thr in thresholds:
                ext = inband & (win[Rc] < thr)
                bnd = inband & (win[Rc] >= thr)
                ac = w[ext].sum(axis=0)
                bd = w[bnd].sum(axis=0)
                tw = float(np.dot(f, tot))
                entry["split"][(Rc, thr)] = {"ac_mb_per_column": ac, "bound_mb_per_column": bd, "ac_mb_weighted": float(np.dot(f, ac)),
                                            "bound_mb_weighted": float(np.dot(f, bd)), "escape_fraction": (float(np.dot(f, ac)) / tw) if tw > 0 else float("nan")}
        res["cuts"][cut] = entry
    return res, win


def run_engine_step(a, out_v2b, Z, keys, tab, results):
    cands = [a.v2b_dir, os.path.dirname(os.path.abspath(a.v2b_json)), "<internal>/code", "<internal>/DUST"]
    D, v2b_dir = import_v2b(cands)
    log("imported dust_root_history_v2b from %s (md5 %s)" % (v2b_dir, md5(os.path.join(v2b_dir, "dust_root_history_v2b.py"))))
    args = out_v2b.get("args") or {}
    ns = types.SimpleNamespace(propagator=args.get("propagator", "ops"), cheb_tol=float(args.get("cheb_tol", 1e-14)),
                               eig_backend=args.get("eig_backend", "scipy"), eig_driver=args.get("eig_driver", "evr"))
    torch, BV2 = D.load_engine(a.engine_dir or args.get("engine_dir", "<internal>/code"),
                               a.page_v6_path or args.get("page_v6_path", "<internal>/v6"))
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    BE = D.Backend(ns, torch, BV2, dev)
    OPS = BE.OPS
    rec1 = a.record or args.get("record")
    fld1 = a.fields or args.get("fields")
    rec2 = a.record2 or args.get("record2")
    fld2 = a.fields2 or args.get("fields2")
    switch = float(args.get("switch_T_rs", 40.0))
    H = BV2.CombinedHistory(rec1, fld1, int(args.get("nsites", 8192)), rec2 or None, fld2 or None, switch if rec2 else None)
    rs = float(H.rs_nom)
    R = np.asarray(H.R, float)
    n = R.size
    TT = np.asarray(H.T, float) / rs
    m = float(out_v2b["m"])
    b = float(out_v2b["b"])
    O = np.load(a.columns_npz or args.get("columns_npz"))
    launches = {L["key"]: L for L in out_v2b.get("launches", [])}
    cuts_rs = [float(x) for x in a.cuts_rs.split(",") if x]
    bcuts = [float(x) for x in a.bound_R_cut_rs.split(",") if x]
    thrs = [float(x) for x in a.bound_thresholds.split(",") if x]
    # the groups (kappa, held record)
    groups = {}
    for key in keys:
        L = launches[key]
        helds = []
        for s in str(a.held_T_rs).split(","):
            s = s.strip()
            if not s:
                continue
            if s == "end":
                helds.append(int(L["k_end"]))
            else:
                helds.append(int(np.argmin(np.abs(TT - float(s)))))
        for kh in helds:
            groups.setdefault((int(L["kappa"]), kh), []).append(key)
    held_out = {}
    for (kap, kh), gkeys in sorted(groups.items()):
        t_g = time.time()
        h0, Q = OPS.chain_sparse(R, m, float(kap))
        Nh, vh = D.fields(H, kh)
        e, U = BE.eigh(OPS.clocked_csr(h0, Q, Nh, vh))
        e = np.asarray(e, float)
        U = np.asarray(U, np.complex128)
        log("kappa %d, held record %d (T/r_s %.3f): eigendecomposition done in %.0f s; e in [%.6g, %.6g]; bins counts %s" % (
            kap, kh, TT[kh], time.time() - t_g, e.min(), e.max(), [int(x) for x in np.bincount(bin_index(e, m, b), minlength=5)]))
        held_out[(kap, kh)] = {"kappa": kap, "record": kh, "T_over_rs": float(TT[kh]), "e_min": float(e.min()), "e_max": float(e.max()),
                               "bin_counts": [int(x) for x in np.bincount(bin_index(e, m, b), minlength=5)], "seconds_eigh": time.time() - t_g, "keys": {}}
        for key in gkeys:
            L = launches[key]
            tg = L["tag"]
            k0 = int(L["k0"])
            kd = "cols_dust_%s_k%d" % (tg, kap)
            A_ = np.asarray(O[kd], np.complex128)
            f = np.asarray(O["f_dust_%s_k%d" % (tg, kap)], float)
            W = np.zeros((2 * n, A_.shape[1]), np.complex128)
            W[:A_.shape[0]] = A_
            S, Kmax, _ = regenerate_state(D, BE, OPS, H, h0, Q, W, k0, kh)
            Wall = D.site_w(S, n)
            jh = kh - k0
            chk = {}
            nr = int(L["n_records"])
            if 0 <= jh < nr:
                for name, arr in (("b16_all", Wall[R > 16 * rs].sum(axis=0)), ("b24_all", Wall[R > 24 * rs].sum(axis=0)), ("norm", Wall.sum(axis=0))):
                    stored = np.asarray(Z["%s_%s" % (key, name)], float)[jh]
                    chk[name] = float(np.max(np.abs(arr - stored)))
                rt = float(np.asarray(Z["%s_Rt_dense" % key], float)[jh])
                if np.isfinite(rt):
                    chk["beyond_Rt_dense"] = float(np.max(np.abs(Wall[R > rt].sum(axis=0) - np.asarray(Z["%s_beyond_Rt_dense" % key], float)[jh])))
            cert, win = exterior_certificate(e, U, S, R, rs, m, b, f, cuts_rs, bcuts, thrs)
            # the whole column's bins against v2b's bins_end (when the held record is the end record)
            w_full = spectral_weights(U, S)
            idx = bin_index(e, m, b)
            bins_full = np.stack([w_full[idx == k].sum(axis=0) for k in range(5)])
            bins_chk = None
            if kh == int(L["k_end"]):
                bins_chk = float(np.max(np.abs(bins_full - np.asarray(Z["%s_bins_end" % key], float))))
            # the whole-column occupation-weighted bins of the block's complement for the record
            held_out[(kap, kh)]["keys"][key] = {"cheb_K_max": Kmax, "state_check_max_abs_diff": chk, "state_check_pass": (max(chk.values()) <= a.state_check_tol) if chk else None,
                                                "bins_full_weighted": (bins_full @ f).tolist(), "bins_full_vs_v2b_bins_end_max_abs_diff": bins_chk, "cert": cert, "f": f.tolist()}
            log("%s at the held record %d: regenerated state checked against v2b (%s); block beyond 16 r_s %.5g, escape fraction (R_cut 16 r_s, 0.99) %s" % (
                key, kh, chk, cert["cuts"][cuts_rs[0]]["total_weighted"] if cuts_rs else float("nan"),
                fmt(cert["cuts"][cuts_rs[0]]["split"].get((bcuts[0], thrs[0]), {}).get("escape_fraction")) if cuts_rs else "-"))
        del U, e
    results["held"] = held_out
    results["engine"] = {"v2b_dir": v2b_dir, "rs": rs, "n_sites": n, "cuts_rs": cuts_rs, "bound_R_cut_rs": bcuts, "thresholds": thrs,
                         "history_E_check": {}}
    # E(t) of the engine's records against the table read without the engine
    diffs = []
    for k in range(len(TT)):
        ent = lookup(tab, TT[k])
        Ek = H.rec[k].get("E") if isinstance(H.rec[k], dict) else None
        if ent is not None and ent.get("E") is not None and Ek is not None:
            diffs.append(abs(float(Ek) - float(ent["E"])))
    results["engine"]["history_E_check"] = {"n": len(diffs), "max_abs_diff": (max(diffs) if diffs else None)}
    return results


# ---------------------------------------------------------------- the md (W7)
def compose_md(a, out_v2b, results, keys):
    L = ["# dust_ledger_postpass_v1 (W7): the ledger of requirements edit 24 over DUST_root_history v2b's outputs", ""]
    L.append("Inputs: v2b json %s (md5 %s), npz %s (md5 %s); the history %s / %s (switch at T/r_s %s); G = %s, m = %s, b = %s, r_s = %s; E_0 = %s (the first record's E). "
             "v2b is not edited; the ledger items 1 to 3 are arithmetic on v2b's stored arrays and the records' E; item 4 regenerates each key's state with v2b's propagator "
             "and checks it against v2b's stored weights. Keys: %s." % (
                 a.v2b_json, results["v2b_json_md5"], a.v2b_npz, results["v2b_npz_md5"], results["record"], results["record2"], fmt(results["switch_T_rs"]), fmt(results["G"]),
                 fmt(results["m"]), fmt(results["b"], 10), fmt(results["rs"]), fmt(results["E0"], 10), ", ".join(keys)))
    L.append("")
    L.append("Symbols (W7_dust_v2b_audit.md section 0): p_j = f . (norm_j - beyond_Rt_j) the weight inside the root at record j; T_j the net outward transport through the fixed old cut R_j over the step; "
             "S_j = -moving_cut_step[j+1] the sweep of the new mask; p_(j+1) - p_j = -T_j + S_j, the residual the norm drift. C(t) = (E(t) - R_t/(2G))/eps_p the never-inside capacity "
             "(R_t/(2G) the charge a root encloses). The certificate [sqrt(p_late) - sqrt(eps_early)]_+^2 per column, weighted. The escape fraction: the exterior block's weight on the extended "
             "eigenvectors with energy in (m, b] of the held operator over the block's weight; extended = weight inside R_cut below the threshold (a finite-chain proxy for P_ac).")
    for key in keys:
        K = results["keys"][key]
        L.append("")
        L.append("## %s (launch T/r_s %s, %d records to %s, %d columns, occupation sum %s)" % (key, fmt(K["T_launch"]), K["n_records"], fmt(K["T_end"]), K["n_columns"], fmt(K["occupation_sum"])))
        for cut in CUTS:
            Lg = K["ledger"][cut]
            L.append("")
            L.append("### 1. Same-set fixed-cut ledger, %s cut: %d rooted steps; max |residual| (norm drift) %s" % (cut, Lg["n_steps"], fmt(Lg["max_abs_resid"], 3)))
            L.append("| episode T_from | T_to | steps | p_in start | p_in end | sum T_j (outward transport through the old cuts) | sum S_j (sweeps) | sum of v2b's current x step | max abs residual |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for ep in Lg["episodes"]:
                L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (fmt(ep["T_from"]), fmt(ep["T_to"]), ep["n_steps"], fmt(ep["p_in_start"]), fmt(ep["p_in_end"]), fmt(ep["sum_T"]), fmt(ep["sum_S"]), fmt(ep["sum_q_end"]), fmt(ep["max_abs_resid"], 2)))
            if Lg["rows_window"]:
                L.append("")
                L.append("Steps in the window T/r_s %s to %s (the transfer basis; T_j and S_j separate; the current quadratures beside):" % (fmt(a.window_from), fmt(a.window_to)))
                L.append("| T_a | T_b | R_a (old cut) | R_b | p_in(a) | p_in(b) | T_j | S_j | residual | current(a) x dt | trapezoid x dt |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|")
                for r_ in Lg["rows_window"]:
                    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % tuple(fmt(r_[k], 6 if k not in ("resid",) else 2) for k in ("T_a", "T_b", "R_a", "R_b", "p_in_a", "p_in_b", "T_j", "S_j", "resid", "q_end", "q_trap")))
        L.append("")
        L.append("### 2. Never-inside capacity at the record's own charge (the block beyond 16 r_s against C(t); the dense root)")
        L.append("| T/r_s | E | E/E_0 | R_t | R_t/(2G) | E - R_t/(2G) | C at m | C at 1.2 m | C at m with E_0 | block beyond 16 | beyond 24 | fits at m | fits at 1.2 m |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in K["capacity_rows_selected"]:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                fmt(row["T"]), fmt(row["E"], 8), fmt(row.get("E_over_E0"), 4), fmt(row["Rt"], 6), fmt(row["M_root"], 6), fmt(row["E_ext"], 6), fmt(row.get("C_at_1m")), fmt(row.get("C_at_1.2m")),
                fmt(row.get("C_E0_at_1m")), fmt(row["block16"]), fmt(row["block24"]), fmt(row.get("fits_at_1m")), fmt(row.get("fits_at_1.2m"))))
        L.append("")
        L.append("### 3. Same-packet inside-origin certificate (weighted lower bound on the far content that was inside the launch root; upper bound beside)")
        L.append("| cut | launch root | weighted eps_early | at the end: beyond 16 r_s | lower bound 16 | upper bound 16 | v2b's inside-tagged 16 | violations | beyond 24 r_s | lower bound 24 | upper bound 24 | v2b's inside-tagged 24 | violations |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for cut in CUTS:
            C = K["certificate"][cut]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %d |" % (
                cut, fmt(K["root_launch"][cut], 8), fmt(C["weighted_eps_early"]), fmt(C["end"]["b16"]["p"]), fmt(C["end"]["b16"]["lower"]), fmt(C["end"]["b16"]["upper"]), fmt(C["end"]["b16"]["b_in"]), C["end"]["b16"]["violations"],
                fmt(C["end"]["b24"]["p"]), fmt(C["end"]["b24"]["lower"]), fmt(C["end"]["b24"]["upper"]), fmt(C["end"]["b24"]["b_in"]), C["end"]["b24"]["violations"]))
        L.append("")
        L.append("Per column at the end (dense cut, 16 r_s): eps_early %s; p_late %s; lower %s; upper %s." % (
            [fmt(x, 4) for x in K["certificate"]["dense"]["eps_early_per_column"]], [fmt(x, 4) for x in K["certificate"]["dense"]["end"]["b16"]["p_per_column"]],
            [fmt(x, 4) for x in K["certificate"]["dense"]["end"]["b16"]["lower_per_column"]], [fmt(x, 4) for x in K["certificate"]["dense"]["end"]["b16"]["upper_per_column"]]))
    if results.get("held"):
        L.append("")
        L.append("## 4. The exterior block's Killing split and the escape certificate on the held records (engine step)")
        L.append("Each key's state at the held record is regenerated with v2b's propagator from the stored columns and checked against v2b's stored per-column weights there; "
                 "the whole column's 5 bins are checked against v2b's bins_end where the held record is the end record. Extended eigenvectors: weight inside R_cut below the threshold.")
        tot_block = {}
        tot_ac = {}
        for (kap, kh), Hd in sorted(results["held"].items()):
            L.append("")
            L.append("### kappa %d, held record %d (T/r_s %s): eigenvalues in [%s, %s], counts per bin %s, %.0f s" % (kap, kh, fmt(Hd["T_over_rs"]), fmt(Hd["e_min"]), fmt(Hd["e_max"]), Hd["bin_counts"], Hd["seconds_eigh"]))
            for key, Kd in Hd["keys"].items():
                C = Kd["cert"]
                L.append("")
                L.append("%s: regenerated-state check %s (pass %s); whole-column bins weighted %s (max |diff| from v2b's bins_end %s); Chebyshev K max %d." % (
                    key, {k: fmt(v, 2) for k, v in Kd["state_check_max_abs_diff"].items()}, fmt(Kd["state_check_pass"]), [fmt(x) for x in Kd["bins_full_weighted"]], fmt(Kd["bins_full_vs_v2b_bins_end_max_abs_diff"], 2), Kd["cheb_K_max"]))
                for Rc, wd in C["win"].items():
                    L.append("eigenvectors in (m, b]: %d; bound (weight inside %g r_s at least thr): %s; the in-band inside-weight quantiles (0, 0.1, 0.5, 0.9, 1): %s." % (
                        C["n_in_band"], Rc, {fmt(t): v for t, v in wd["n_bound_in_band"].items()}, [fmt(x, 3) for x in (wd["in_band_inside_weight_quantiles"] or [])]))
                L.append("| cut r_s | block weight (weighted) | bins weighted (-inf,-b] (-b,-m] (-m,m] (m,b] (b,inf) | R_cut r_s | threshold | ac in (m,b] | bound in (m,b] | escape fraction | certified above %s |" % fmt(a.escape_certify))
                L.append("|---|---|---|---|---|---|---|---|---|")
                for cut, ent in C["cuts"].items():
                    for (Rc, thr), sp_ in ent["split"].items():
                        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                            fmt(cut), fmt(ent["total_weighted"]), [fmt(x) for x in ent["bins_weighted"]], fmt(Rc), fmt(thr), fmt(sp_["ac_mb_weighted"]), fmt(sp_["bound_mb_weighted"]),
                            fmt(sp_["escape_fraction"], 4), fmt(bool(sp_["escape_fraction"] > a.escape_certify) if np.isfinite(sp_["escape_fraction"]) else None)))
                    key16 = (cut, Hd["T_over_rs"])
                    tot_block[key16] = tot_block.get(key16, 0.0) + ent["total_weighted"]
                    sp0 = ent["split"].get((results["engine"]["bound_R_cut_rs"][0], results["engine"]["thresholds"][0]))
                    if sp0:
                        tot_ac[key16] = tot_ac.get(key16, 0.0) + sp0["ac_mb_weighted"]
        if tot_block:
            L.append("")
            L.append("Summed over the keys processed (R_cut %s r_s, threshold %s): " % (fmt(results["engine"]["bound_R_cut_rs"][0]), fmt(results["engine"]["thresholds"][0])) + "; ".join(
                "cut %s r_s at T/r_s %s: block %s, ac in (m,b] %s, fraction %s (%s)" % (fmt(c), fmt(t), fmt(tot_block[(c, t)]), fmt(tot_ac.get((c, t))), fmt(tot_ac.get((c, t), 0.0) / tot_block[(c, t)] if tot_block[(c, t)] > 0 else float("nan"), 4),
                                                                                "certified" if tot_block[(c, t)] > 0 and tot_ac.get((c, t), 0.0) / tot_block[(c, t)] > a.escape_certify else "not certified") for (c, t) in sorted(tot_block)))
            L.append("The global statement about the 50.72 beyond 16 r_s needs every key of the history in this sum; the keys here are %s." % ", ".join(keys))
    L.append("")
    L.append("Label: the dust = the stored dust-tagged occupied columns with their occupations f on the production's solved history (the records, then B's continuation); the massive channel ma 0.2 only; "
             "counts in fermions; the ledger's sign: T_j outward positive, S_j positive when the new mask takes content in; the certificate is a lower bound, not a count; the escape fraction is read on the "
             "finite chain through the extended/bound proxy named with it; weight above b has no propagating partner at infinity and is never read as unbound; no mutual information.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main (W7)
def main():
    ap = argparse.ArgumentParser(description="W7: the ledger of requirements edit 24 over DUST_root_history v2b's outputs")
    ap.add_argument("--v2b-json", default="<internal>/dust_root_history_v2.json")
    ap.add_argument("--v2b-npz", default="<internal>/dust_root_history_v2.npz")
    ap.add_argument("--v2b-dir", default=None, help="the directory of dust_root_history_v2b.py and horizon3_bins_v1.py (the engine step); default: the v2b json's directory, then the engine code dir, then the DUST dir")
    ap.add_argument("--keys", default=None, help="comma list of launch keys; default every launch in the v2b json")
    ap.add_argument("--record", default=None)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--record2", default=None)
    ap.add_argument("--fields2", default=None)
    ap.add_argument("--columns-npz", default=None)
    ap.add_argument("--engine-dir", default=None)
    ap.add_argument("--page-v6-path", default=None)
    ap.add_argument("--G", type=float, default=None, help="Newton's constant g (default: the record json's args 'g', else 0.25 with a note)")
    ap.add_argument("--eps-per-particle", default="1,1.2", help="the per-particle Killing energy in units of m for the capacity")
    ap.add_argument("--window-from", type=float, default=15.5)
    ap.add_argument("--window-to", type=float, default=16.125)
    ap.add_argument("--held-T-rs", default="end", help="comma list of held records (T/r_s) for the escape certificate; 'end' = each launch's end record")
    ap.add_argument("--cuts-rs", default="16,24")
    ap.add_argument("--bound-R-cut-rs", default="16,2")
    ap.add_argument("--bound-thresholds", default="0.99,0.9,0.999")
    ap.add_argument("--escape-certify", type=float, default=0.92)
    ap.add_argument("--state-check-tol", type=float, default=1e-8)
    ap.add_argument("--no-engine", action="store_true", help="items 1 to 3 only (no import of the engine; seconds)")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-prefix", default="<internal>/dust_ledger_postpass_v1")
    a = ap.parse_args()
    with open(a.v2b_json) as fh:
        out_v2b = json.load(fh)
    Z = np.load(a.v2b_npz)
    launches = {L["key"]: L for L in out_v2b.get("launches", []) if isinstance(L, dict) and "key" in L}
    keys = [k for k in a.keys.split(",") if k] if a.keys else list(launches.keys())
    missing = [k for k in keys if k not in launches]
    if missing:
        raise SystemExit("keys not in the v2b json (not yet completed?): %s; completed: %s" % (missing, list(launches)))
    args = out_v2b.get("args") or {}
    rec1 = a.record or args.get("record")
    rec2 = a.record2 or args.get("record2")
    switch = float(args.get("switch_T_rs", 40.0))
    rs = float(out_v2b["rs_nominal"])
    m = float(out_v2b["m"])
    b = float(out_v2b["b"])
    tab, E0, args1, args2 = history_table(rec1, rec2, switch, rs)
    G = a.G if a.G is not None else args1.get("g")
    note_G = ""
    if G is None:
        G = 0.25
        note_G = "G not in the record json's args: 0.25 assumed"
    G = float(G)
    eps_units = [float(x) for x in a.eps_per_particle.split(",") if x]
    results = {"script": os.path.basename(__file__), "args": vars(a), "v2b_json_md5": md5(a.v2b_json), "v2b_npz_md5": md5(a.v2b_npz), "record": rec1, "record2": rec2,
               "switch_T_rs": switch, "G": G, "note_G": note_G, "m": m, "b": b, "rs": rs, "E0": E0, "n_history_records": len(tab), "keys": {}}
    log("v2b json %s: %d launches completed (%s); keys processed %s; E_0 %s; G %s %s" % (a.v2b_json, len(launches), ", ".join(launches), keys, fmt(E0, 10), fmt(G), note_G))
    arrays_out = {}
    for key in keys:
        L = launches[key]
        f = np.asarray(Z["%s_f" % key], float)
        K = {"T_launch": L["T_launch_over_rs"], "T_end": L["T_end_over_rs"], "n_records": L["n_records"], "n_columns": L["n_columns"], "occupation_sum": L["occupation_sum"],
             "root_launch": {"dense": L["roots_launch"]["dense_outer"], "applied": L["roots_launch"]["applied_outer"], "record_zeros": L["roots_launch"].get("record_zeros")},
             "ledger": {}, "certificate": {}, "capacity_rows": [], "capacity_rows_selected": []}
        for cut in CUTS:
            Lg = ledger_arrays(Z, key, cut, rs)
            eps_ = episode_sums(Lg)
            rows_w = []
            for j in range(1, Lg["T"].size):
                if np.isfinite(Lg["T_j"][j]) and a.window_from - 1e-9 <= Lg["T"][j - 1] and Lg["T"][j] <= a.window_to + 1e-9:
                    rows_w.append({"T_a": float(Lg["T"][j - 1]), "T_b": float(Lg["T"][j]), "R_a": float(Lg["Rt"][j - 1]), "R_b": float(Lg["Rt"][j]), "p_in_a": float(Lg["p_in"][j - 1]),
                                   "p_in_b": float(Lg["p_in"][j]), "T_j": float(Lg["T_j"][j]), "S_j": float(Lg["S_j"][j]), "resid": float(Lg["resid"][j]), "q_end": float(Lg["q_end"][j]), "q_trap": float(Lg["q_trap"][j])})
            fin = np.isfinite(Lg["T_j"])
            K["ledger"][cut] = {"n_steps": int(fin.sum()), "max_abs_resid": float(np.max(np.abs(Lg["resid"][fin]))) if fin.any() else None, "episodes": eps_, "rows_window": rows_w,
                                "sum_T_all": float(np.sum(Lg["T_j"][fin])) if fin.any() else None, "sum_S_all": float(np.sum(Lg["S_j"][fin])) if fin.any() else None}
            for nm in ("T_j", "S_j", "dp_in", "resid", "q_end", "q_trap", "p_in", "p_out", "dt"):
                arrays_out["%s_%s_%s" % (key, cut, nm)] = Lg[nm]
            # the certificate
            C = certificates(Z, key, cut)
            end = {}
            for c in ("b16", "b24"):
                end[c] = {"p": float(C[c]["p_weighted"][-1]), "lower": float(C[c]["lower_weighted"][-1]), "upper": float(C[c]["upper_weighted"][-1]), "b_in": float(C[c]["b_in_weighted"][-1]),
                          "violations": C[c]["n_violations_lower_gt_b_in"], "p_per_column": C[c]["lower"][-1] * 0 + (np.asarray(Z["%s_%s_all" % (key, c)], float)[-1] / np.asarray(Z["%s_norm" % key], float)[-1]),
                          "lower_per_column": C[c]["lower"][-1], "upper_per_column": C[c]["upper"][-1]}
                arrays_out["%s_%s_%s_cert_lower_weighted" % (key, cut, c)] = C[c]["lower_weighted"]
                arrays_out["%s_%s_%s_cert_upper_weighted" % (key, cut, c)] = C[c]["upper_weighted"]
                arrays_out["%s_%s_%s_cert_lower" % (key, cut, c)] = C[c]["lower"]
            K["certificate"][cut] = {"weighted_eps_early": C["weighted_eps_early"], "eps_early_per_column": C["eps_early"], "end": end}
            arrays_out["%s_%s_eps_early" % (key, cut)] = C["eps_early"]
        # the capacity (dense root)
        T = np.asarray(Z["%s_T" % key], float)
        Rt = np.asarray(Z["%s_Rt_dense" % key], float)
        b16 = np.asarray(Z["%s_b16_all" % key], float) @ f
        b24 = np.asarray(Z["%s_b24_all" % key], float) @ f
        rows = capacity_rows(tab, T, Rt, G, m, eps_units, E0, b16, b24)
        K["capacity_rows"] = rows
        sel_T = {float(T[0]), 40.0, float(T[-1])}
        K["capacity_rows_selected"] = [r_ for r_ in rows if any(abs(r_["T"] - s) < 1e-6 for s in sel_T)]
        # the end weights against the json
        K["end_check"] = {"b16_all_json": L["end"]["all"]["b16"], "b16_all_arrays": float(b16[-1]), "b24_all_json": L["end"]["all"]["b24"], "b24_all_arrays": float(b24[-1])}
        results["keys"][key] = K
        log("%s: ledger steps dense %d (max residual %s), applied %d; certificate at the end (dense, 16 r_s): beyond %s, lower %s, upper %s, v2b inside-tagged %s; "
            "capacity at the end: E %s, R_t %s, C at m %s, block %s" % (
                key, K["ledger"]["dense"]["n_steps"], fmt(K["ledger"]["dense"]["max_abs_resid"], 2), K["ledger"]["applied"]["n_steps"], fmt(end["b16"]["p"]) if False else fmt(K["certificate"]["dense"]["end"]["b16"]["p"]),
                fmt(K["certificate"]["dense"]["end"]["b16"]["lower"]), fmt(K["certificate"]["dense"]["end"]["b16"]["upper"]), fmt(K["certificate"]["dense"]["end"]["b16"]["b_in"]),
                fmt(rows[-1]["E"], 8) if rows else "-", fmt(rows[-1]["Rt"]) if rows else "-", fmt(rows[-1].get("C_at_1m")) if rows else "-", fmt(rows[-1]["block16"]) if rows else "-"))
    if not a.no_engine:
        run_engine_step(a, out_v2b, Z, keys, tab, results)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    # json: tuples as keys are not serializable: flatten the held dict
    res_json = dict(results)
    if "held" in res_json:
        res_json["held"] = [{"kappa": kap, "record": kh, **{k: v for k, v in Hd.items() if k != "keys"},
                             "keys": {key: {kk: (vv if kk != "cert" else {"n_in_band": vv["n_in_band"], "e_min": vv["e_min"], "e_max": vv["e_max"], "bin_counts": vv["bin_counts"],
                                                                         "win": {str(Rc): {"n_bound_in_band": {str(t): c for t, c in wd["n_bound_in_band"].items()}, "quantiles": wd["in_band_inside_weight_quantiles"]} for Rc, wd in vv["win"].items()},
                                                                         "cuts": {str(cut): {"total_per_column": ent["total_per_column"], "total_weighted": ent["total_weighted"], "bins_per_column": ent["bins_per_column"], "bins_weighted": ent["bins_weighted"],
                                                                                              "split": {"%g,%g" % rt: sp_ for rt, sp_ in ent["split"].items()}} for cut, ent in vv["cuts"].items()}})
                                            for kk, vv in Kd.items()} for key, Kd in Hd["keys"].items()}} for (kap, kh), Hd in results["held"].items()]
    with open(a.out_prefix + ".json", "w") as fh:
        json.dump(res_json, fh, default=_jdefault)
    np.savez(a.out_prefix + ".npz", **arrays_out)
    with open(a.out_prefix + ".md", "w") as fh:
        fh.write(compose_md(a, out_v2b, results, keys))
    log("wrote %s.{json,npz,md}" % a.out_prefix)


if __name__ == "__main__":
    main()
