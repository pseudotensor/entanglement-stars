#!/usr/bin/env python
"""gate_common_v1.py (busi lane, GATECOMP kits, inbox #400; ReviewA_gatecomp_v1.md secs. 2 and 3, ReviewB_gatecomp_v1.md where reviewer A is
silent). Shared readers for chainval_cmp_v1.py (kit (i)) and gate_triple_read_v1.py (kit (ii)). Pure functions of a run's json and fields
npz; nothing is ruled here, every number is printed with its definition by the callers.

Definitions (printed by the callers):
  m(T)       the record's min_N_minus_absv = min_r (N - |v|) (engine's own number; < 0 where a trapping zero exists).
  birth      reviewer A sec. 3.2: T_birth is the zero of m(T), linear through the two bracketing records (the last record with m >= 0 and the
             first with m < 0); R_birth is the record's r_min_N_minus_absv interpolated linearly to T_birth. The error printed is the larger
             of |T_lin - T_quad| (the 3-point quadratic through the record before the bracket) and nothing else; the bracket is printed beside.
  K_op       the slice's extrinsic curvature for a radial flow, K_ij = (d_i v_j + d_j v_i)/(2N): eigenvalues v'/N (radial) and v/(rN)
             (tangential, twice); ||K||_op = max(|v'|, |v/r|)/N at each site. v' by numpy.gradient on the site grid R (spacing a).
  H          H = -dv/dr (the paper's one-term flow gradient), |H| at each site.
  gate row   sigma * max over 0 < r <= r_max of ||K||_op (and of |H|), r_max = 2 R_t at a record with a root, 2 R_birth before it
             (reviewer A sec. 3.1: "max over 0 <= r <= 2R_t").
  six-term   the code of record's definition (solve/code/engine/engine_core_slices_v3.py 84cf2053, S7 of WEB40F; recovered in
             deliveries/busi/GATECOMP_iii_read_v1.md sec. 2): Lambda_geom = max(|v'/N|, |v/(N r)|, |N'/N|, sqrt|D k_r|, sqrt|D k_t|,
             sqrt|N''/N|), D = (d_T + v d_r)/N, on the dense fields re-solved from the record's stored eps_tot, J_tot by the field module's
             solve_constraints_ivp (collar flag, support 1e-14), derivatives Richardson at h = sigma/32, d_T central between the adjacent
             records (one-sided at the ends), all by gate_profiles_v1.py's own functions (sym, kfields; imported, unchanged). GATED on the
             code of record's nodes r = sigma, 2 sigma, ... up to r_max (T_gate uses it); the values at r < sigma (the regular centre,
             fine grid sigma/8 from r = 0) are REPORTED separately and never gate: whether they count is reviewer A's open question (GATECOMP_iii
             sec. 3.3). Needs --fm-dir (the field module's dir); without it the six-term column is OPEN.
"""
import importlib.util
import json
import math
import os

import numpy as np


def load_run(d, tag):
    """(json dict, fields npz or None) of collapse_TAG.json / collapse_TAG_fields.npz in directory d."""
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    p = os.path.join(d, "collapse_%s_fields.npz" % tag)
    F = np.load(p, allow_pickle=True) if os.path.exists(p) else None
    return J, F


def split_spec(s):
    """'DIR/TAG' -> (DIR, TAG) where the json is DIR/collapse_TAG.json."""
    d, t = os.path.split(s.rstrip("/"))
    return d, t


def rs_of(J, rs=None):
    return float(rs) if rs else float(J["args"]["rs"])


def recs_by_T(J, rs):
    return {round(float(r["T"]) / rs, 6): r for r in J["records"] if r.get("E") is not None}


def rows_by_T(F, rs):
    if F is None:
        return {}
    return {round(float(t) / rs, 6): i for i, t in enumerate(np.asarray(F["T"], float))}


def kept(r):
    """(kept_residual, meets_stop_rule, converged) of a record (None where the engine wrote none)."""
    s = r.get("solve_v32") or {}
    return s.get("kept_residual"), s.get("meets_stop_rule"), r.get("converged")


def certified(r):
    """a record the engine certified: converged True (meets_stop_rule where it exists)."""
    kr, ms, cv = kept(r)
    if cv is False:
        return False
    if ms is False:
        return False
    return True


def two_gE(J):
    return 2.0 * float(J["args"]["g"]) * float(J["head"]["E0"])


def birth(J, rs):
    """reviewer A sec. 3.2. -> dict or None (no crossing)."""
    R = [r for r in sorted(J["records"], key=lambda x: x["T"]) if r.get("min_N_minus_absv") is not None]
    T = np.array([r["T"] / rs for r in R]); m = np.array([float(r["min_N_minus_absv"]) for r in R])
    idx = [i for i in range(1, len(R)) if m[i] < 0.0 <= m[i - 1]]
    if not idx:
        return None
    i = idx[0]
    t0, t1, m0, m1 = T[i - 1], T[i], m[i - 1], m[i]
    tl = t0 + m0 * (t1 - t0) / (m0 - m1)
    tq = None
    if i >= 2:
        c = np.polyfit(T[i - 2:i + 1], m[i - 2:i + 1], 2)
        roots = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and t0 - 1e-12 <= x.real <= t1 + 1e-12]
        tq = float(roots[0]) if roots else None
    r0 = R[i - 1].get("r_min_N_minus_absv"); r1 = R[i].get("r_min_N_minus_absv")
    Rl = (float(r0) + (float(r1) - float(r0)) * (tl - t0) / (t1 - t0)) if (r0 is not None and r1 is not None) else None
    g2 = two_gE(J)
    return {"T_lo": float(t0), "T_hi": float(t1), "m_lo": float(m0), "m_hi": float(m1), "T_birth": float(tl), "T_quad": tq,
            "T_err": (abs(tl - tq) if tq is not None else None), "R_lo": r0, "R_hi": r1, "R_birth": Rl,
            "R_birth_over_2gE": (Rl / g2 if Rl is not None else None), "R_t_first": R[i].get("R_t"), "two_gE0": g2,
            "later_crossings": len(idx) - 1}


def slice_clock(J):
    """a control's restart clock T/r_s (head resume_slice_v34), or None for a member."""
    return (J.get("head", {}).get("resume_slice_v34") or {}).get("T_over_rs")


def own(J):
    """ReviewA_resview_v1 (3): a control's OWN records only (T > its restart slice's clock); a member unchanged. Matched by clock."""
    tc = slice_clock(J)
    if tc is None:
        return J
    rs = float(J["args"]["rs"])
    K = dict(J); K["records"] = [r for r in J["records"] if r["T"] / rs > tc + 1e-9]
    return K


def point_free(J, F, rs, sixterm, t_from, t_to):
    """ReviewA_resview_v1, the point-sampled control's free version: per stored record in [t_from, t_to], the dense fields re-solved from
    the record's shell totals (sixterm.solve), point-sampled at the sites, against the record's applied (sigma-averaged) site fields:
    -> rows (T, min(N - |v|) averaged, at r, point-sampled, at r, the applied outer sign change of N - |v| averaged, point-sampled,
    max |N_p - N_a|, max |v_p - v_a| on r <= 2 R)."""
    Rg = np.asarray(F["R"], float); RB = recs_by_T(J, rs); IB = rows_by_T(F, rs); out = []
    def outer_zero(R, f):
        s = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
        if not len(s):
            return None
        i = int(s[-1]); return float(R[i] + f[i] * (R[i + 1] - R[i]) / (f[i] - f[i + 1]))
    for t in sorted(set(RB) & set(IB)):
        if t < t_from - 1e-9 or t > t_to + 1e-9:
            continue
        i = IB[t]; S = sixterm.solve(F, i)
        if S is None:
            continue
        Np = np.asarray(S[0](Rg), float); vp = np.asarray(S[1](Rg), float)
        Na = np.asarray(F["N_applied"][i], float); va = np.asarray(F["v_applied"][i], float)
        fa = Na - np.abs(va); fp = Np - np.abs(vp)
        Rt = RB[t].get("R_t"); rr = 2.0 * float(Rt) if Rt else float(Rg[-1]); sel = Rg <= rr
        ja = int(np.argmin(np.where(sel, fa, np.inf))); jp = int(np.argmin(np.where(sel, fp, np.inf)))
        out.append((t, float(fa[ja]), float(Rg[ja]), float(fp[jp]), float(Rg[jp]), outer_zero(Rg[sel], fa[sel]), outer_zero(Rg[sel], fp[sel]),
                    float(np.max(np.abs(Np - Na)[sel])), float(np.max(np.abs(vp - va)[sel]))))
    return out


def kop_rows(R, N, v):
    """-> (||K||_op, |H|) per site; sites with N <= 0 or R <= 0 get nan."""
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    dv = np.gradient(v, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kr = np.abs(dv) / N
        Kt = np.abs(v) / (R * N)
    K = np.maximum(Kr, Kt)
    K[(N <= 0) | (R <= 0)] = np.nan
    return K, np.abs(dv)


def load_sixterm(fm_dir, fm="engine_fields_v16"):
    """-> a SixTerm reader (the code of record's definition through gate_profiles_v1's functions), or None without fm_dir."""
    if not fm_dir:
        return None
    import sys as _s
    _s.path.insert(0, fm_dir)
    FM = __import__(fm)
    gp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gate_profiles_v1.py")
    sp = importlib.util.spec_from_file_location("gate_profiles_v1", gp)
    GP = importlib.util.module_from_spec(sp); sp.loader.exec_module(GP)
    return SixTerm(FM, GP)


class SixTerm(object):
    """per run: solve(F, k) caches the re-solved dense callables of fields row k; at(F, k, sigma, rmax) -> (nodes max, nodes r, dominant,
    centre max on r < sigma, its r, dT mode)."""
    def __init__(self, FM, GP):
        self.FM, self.GP, self.cache = FM, GP, {}

    def solve(self, F, k):
        key = (id(F), k)
        if key not in self.cache:
            Rd = np.asarray(F["R"], float)
            _, _, _, info = self.FM.solve_constraints_ivp(np.asarray(F["eps_tot"][k], float), np.asarray(F["J_tot"][k], float), float(F["g"]), Rd,
                                                          float(F["ell"]), collar_policy="flag", support_rel=1e-14)
            self.cache[key] = None if info.get("failed") else (self.GP.sym(info["N_dense"], +1), self.GP.sym(info["v_dense"], -1))
        return self.cache[key]

    def at(self, F, k, sigma, rmax):
        GP = self.GP; Td = np.asarray(F["T"], float); n = len(Td)
        S0 = self.solve(F, k)
        if S0 is None:
            return None
        Sp = self.solve(F, k - 1) if k - 1 >= 0 else None; Sn = self.solve(F, k + 1) if k + 1 < n else None
        h = sigma / 32.0
        rn = np.arange(sigma, rmax + 1e-9, sigma); rc = np.arange(0.0, sigma - 1e-9, sigma / 8.0)
        out = []
        for r in (rn, rc):
            if not len(r):
                out.append(None); continue
            k0 = GP.kfields(S0[0], S0[1], r, h)
            kp = GP.kfields(Sp[0], Sp[1], r, h) if Sp else None; kn = GP.kfields(Sn[0], Sn[1], r, h) if Sn else None
            if kp is not None and kn is not None:
                dTr = (kn["kr"] - kp["kr"]) / (Td[k + 1] - Td[k - 1]); dTt = (kn["kt"] - kp["kt"]) / (Td[k + 1] - Td[k - 1]); mode = "central"
            elif kp is not None:
                dTr = (k0["kr"] - kp["kr"]) / (Td[k] - Td[k - 1]); dTt = (k0["kt"] - kp["kt"]) / (Td[k] - Td[k - 1]); mode = "backward"
            else:
                dTr = (kn["kr"] - k0["kr"]) / (Td[k + 1] - Td[k]); dTt = (kn["kt"] - k0["kt"]) / (Td[k + 1] - Td[k]); mode = "forward"
            Dkr = (dTr + k0["v"] * k0["kr1"]) / k0["N"]; Dkt = (dTt + k0["v"] * k0["kt1"]) / k0["N"]
            T6 = {"abs(v'/N)": np.abs(k0["kr"]), "abs(v/(N r))": np.abs(k0["kt"]), "abs(N'/N)": np.abs(k0["N1"] / k0["N"]),
                  "sqrt(abs(D k_r))": np.sqrt(np.abs(Dkr)), "sqrt(abs(D k_t))": np.sqrt(np.abs(Dkt)), "sqrt(abs(N''/N))": np.sqrt(np.abs(k0["N2"] / k0["N"]))}
            L = np.max(np.array(list(T6.values())), axis=0); i = int(np.argmax(L))
            dom = max(T6, key=lambda t: T6[t][i])
            out.append((float(sigma * L[i]), float(r[i]), dom, mode))
        return out


def gate_profile(J, F, rs, sigma, b=None, t_from=None, sixterm=None, line=0.1):
    """reviewer A sec. 3.1 on one run. -> {"rows": [(T, r_max, sKop, r_at, sH, six_nodes or None, R_t, six_nodes_r, six_dominant,
    six_centre (r < sigma, reported only), six_centre_r, dT mode)], "T_gate": first T with sigma||K||_op or the six-term (nodes) > line,
    "window": T_gate - T_birth}. Rows from t_from (default T_birth - 1) on; r_max = 2 R_t, or 2 R_birth before the root."""
    if F is None:
        return {"rows": [], "T_gate": None, "window": None, "note": "no fields npz"}
    b = b if b is not None else birth(J, rs)
    Rg = np.asarray(F["R"], float)
    rb = (b or {}).get("R_birth")
    t_from = t_from if t_from is not None else ((b["T_birth"] - 1.0) if b else 0.0)
    RB = recs_by_T(J, rs); IB = rows_by_T(F, rs)
    rows = []; T_gate = None
    for t in sorted(set(RB) & set(IB)):
        if t < t_from - 1e-9:
            continue
        rec = RB[t]; i = IB[t]
        Rt = rec.get("R_t")
        rmax = 2.0 * float(Rt) if Rt else (2.0 * float(rb) if rb else None)
        if rmax is None:
            continue
        K, H = kop_rows(Rg, F["N"][i], F["v"][i])
        sel = (Rg > 0) & (Rg <= rmax) & np.isfinite(K)
        if not sel.any():
            continue
        j = int(np.nanargmax(np.where(sel, K, -np.inf)))
        sK = sigma * float(K[j]); sH = sigma * float(np.max(H[sel]))
        q = sixterm.at(F, i, sigma, rmax) if sixterm else None
        six = q[0][0] if (q and q[0]) else None
        rows.append((t, rmax, sK, float(Rg[j]), sH, six, Rt, (q[0][1] if (q and q[0]) else None), (q[0][2] if (q and q[0]) else None),
                     (q[1][0] if (q and q[1]) else None), (q[1][1] if (q and q[1]) else None), (q[0][3] if (q and q[0]) else None)))
        if T_gate is None and (sK > line or (six is not None and six > line)):
            T_gate = t
    win = (T_gate - b["T_birth"]) if (T_gate is not None and b) else None
    return {"rows": rows, "T_gate": T_gate, "window": win, "sixterm": bool(sixterm), "r_birth": rb}


def fmt(x, f="%.4g"):
    return "None" if x is None else (f % x)
