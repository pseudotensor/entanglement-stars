#!/usr/bin/env python
"""engine_critical_width_scan_v1.py (builder analyst; engine spec 37.3 Ruled (2): "on the ell 16 record's stored dense fields and stored
applied fields separately, at T/rs 15.375 and at its first record, the back-average at width sigma with the engine's own kernels (l = 0 for
N, l = 1 for v), the maximum over r of the averaged flow over the averaged clock (P21's W_1[N u]/W_0[N], u = v/N) against sigma, the sigma
at which it crosses 1 by bisection where it does, reported in units of 2gE against 0.2609 and 0.2282; the same scan on the production's last
untrapped record as the case with a zero"). Record-only, CPU, no eigendecomposition.

Per record and field set: the dense fields are re-solved from the stored shell totals by the run's own field module (the record's solve:
N_dense, v_dense as functions of r); the applied fields are the stored site arrays N_applied, v_applied, made functions of r by a cubic
spline through the sites with the parity extension through the centre (N even, v odd) and the declared vacuum exterior beyond the last
site (N = 1, v = -sqrt(2 g E/r), E the record's charge). The back-average at width sigma is engine_fields_v12.site_fields_kernel_average with
ell = sigma (the engine's K_0 for N and K_1 for v, evaluated at the sites R_n, band 8 sigma, trapezoid at sigma/16; W_1[N u] = W_1[v]);
Mcal(sigma) = max_n |W_1[v](R_n)|/W_0[N](R_n) (WEB40F P21 eq. 12.1; non-increasing in sigma by its semigroup argument, eq. 12.2);
Mcal(0) the unaveraged maximum of |v|/N (the dense fields on a grid of spacing 1/64 from 1/64 to the last site; the applied fields at the
sites). Where Mcal - 1 changes sign between two scan widths the crossing sigma_c is bisected to 1e-4 a and printed with sigma_c/(2 g E)
beside the returns' constants 0.260875 (WEB40F P21/P20: unit lapse, 0 <= M <= E, a single return) and 0.228195 (P34; P21 sec. 12.4's
quadrature estimate 0.2282052018: the same width on the source and the return, a positive current-free source), quoted as the returns' and not as this lane's criterion; where it
does not change sign the scan's range is printed. A check: the scan of the dense fields at sigma = the run's ell against the stored applied
fields' own Mcal(0) (the record's applied fields are that average, to the self-consistency tolerance).
Records: --record label=json=data_npz=T_over_rs (repeated; data_npz a records data npz with eps_tot, J_tot, N_applied, v_applied, or a run's
fields npz with the same keys) --fm-segments --fm-default --md."""
import argparse
import importlib
import json
import math
import os
import sys

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v12 as F12  # noqa: E402

C_RETURN = 0.260875404864
C_DOUBLE = 0.228195          # P34 (the reviewer verified it); P21 sec. 12.4 estimates 0.2282052018 by quadrature
SIGMAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0]


def site_callables(R, N, v, g, E):
    """the applied site arrays as functions of r: cubic splines with the parity extension, the vacuum exterior beyond the last site."""
    R = np.asarray(R, float)
    rr = np.concatenate([-R[::-1], R])
    sN = make_interp_spline(rr, np.concatenate([N[::-1], N]), k=3)
    sv = make_interp_spline(rr, np.concatenate([-v[::-1], v]), k=3)
    Rl = float(R[-1])

    def Nf(r):
        r = np.atleast_1d(np.asarray(r, float))
        return np.where(r <= Rl, sN(np.minimum(r, Rl)), 1.0)

    def vf(r):
        r = np.atleast_1d(np.asarray(r, float))
        return np.where(r <= Rl, sv(np.minimum(r, Rl)), -np.sqrt(2.0 * g * E / np.maximum(r, 1e-300)))
    return Nf, vf


def mcal(Nf, vf, R, sigma):
    Na, va, _ = F12.site_fields_kernel_average({"N_dense": Nf, "v_dense": vf}, R, sigma)
    q = np.abs(va) / Na
    i = int(np.argmax(q))
    return float(q[i]), float(R[i])


def mcal0(Nf, vf, R, dense):
    rr = np.arange(1.0 / 64.0, float(R[-1]) + 1e-9, 1.0 / 64.0) if dense else np.asarray(R, float)
    q = np.abs(np.asarray(vf(rr), float)) / np.asarray(Nf(rr), float)
    i = int(np.argmax(q))
    return float(q[i]), float(rr[i])


def scan(Nf, vf, R, dense):
    m0, r0 = mcal0(Nf, vf, R, dense)
    rows = [(0.0, m0, r0)]
    for s in SIGMAS:
        m, r = mcal(Nf, vf, R, s)
        rows.append((s, m, r))
    cross = None
    for (s1, m1, _), (s2, m2, _) in zip(rows[:-1], rows[1:]):
        if (m1 - 1.0) * (m2 - 1.0) < 0.0:
            lo, hi, mlo = s1, s2, m1
            if lo == 0.0:
                lo = 1e-3
                mlo, _ = mcal(Nf, vf, R, lo)
                if (mlo - 1.0) * (m2 - 1.0) >= 0.0:
                    break
            while hi - lo > 1e-4:
                mid = 0.5 * (lo + hi)
                mm, _ = mcal(Nf, vf, R, mid)
                if (mm - 1.0) * (mlo - 1.0) > 0.0:
                    lo, mlo = mid, mm
                else:
                    hi = mid
            cross = 0.5 * (lo + hi)
            break
    return rows, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="append", required=True, help="label=json=npz=T_over_rs")
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14")
    ap.add_argument("--fm-default", default=None, help="the field module when the npz holds no segments (default: by the run's script)")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    script_fm = {"engine_collapse_v17": "engine_fields_v12", "engine_collapse_v19": "engine_fields_v14", "engine_collapse_v24": "engine_fields_v16",
                 "engine_collapse_v25": "engine_fields_v16", "engine_collapse_v26": "engine_fields_v16", "engine_collapse_v27": "engine_fields_v16"}
    L = ["## The fixed-field critical-width scan (engine_critical_width_scan_v1.py; engine spec 37.3 (2), WEB40F P21 sec. 12, P34; record-only, no eigendecomposition)", "",
         "- Mcal(sigma) = max over the sites of |W_1[v]|/W_0[N] (the engine's K_1 and K_0 at width sigma: engine_fields_v12.site_fields_kernel_average); Mcal(0) the unaveraged max |v|/N; the dense fields re-solved from the stored shell totals by the run's field module, the applied fields the stored site arrays (cubic spline with the parity extension, the vacuum exterior beyond the last site); sigma_c the bisected crossing of Mcal = 1 (to 1e-4 a) where the scan changes sign; the returns' constants quoted: %.6f (single return, unit lapse, 0 <= M <= E) and %.6f (source and return at one width, positive current-free source)." % (C_RETURN, C_DOUBLE),
         "- scan widths (a): 0, %s." % ", ".join("%g" % s for s in SIGMAS), ""]
    for spec in a.record:
        lab, jp, zp, tq = spec.split("=", 3)
        J = json.load(open(jp))
        R_ = [r for r in J["records"] if r.get("E") is not None and "T_over_rs" in r]
        T_ = float(tq)
        rec = min(R_, key=lambda r: abs(r["T_over_rs"] - T_))
        if abs(rec["T_over_rs"] - T_) > 1e-9:
            L.append("### %s: no record at T/rs %g" % (lab, T_)); continue
        Z = np.load(zp, allow_pickle=True)
        Tz = np.asarray(Z["T"], float); k = int(np.argmin(np.abs(Tz - rec["T"])))
        if abs(Tz[k] - rec["T"]) > 1e-9:
            L.append("### %s: the npz holds no record at T/rs %g" % (lab, T_)); continue
        R = np.asarray(Z["R"], float); g = float(J["args"]["g"]); ell = float(J["args"]["ell"]); E = float(rec["E"])
        if "segment_names" in Z.files:
            fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";"))
            fmn = fmap[str(Z["segment_names"][int(Z["segment"][k])])]
        else:
            fmn = a.fm_default or script_fm[str(J.get("script", "")).replace(".py", "")]
        FM = importlib.import_module(fmn)
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, R, ell, collar_policy="flag", support_rel=1e-14)
        Nd, vd = info["N_dense"], info["v_dense"]
        Na_, va_ = np.asarray(Z["N_applied"][k], float), np.asarray(Z["v_applied"][k], float)
        Naf, vaf = site_callables(R, Na_, va_, g, E)
        L.append("### %s (%s, T/rs %.3f; ell %g; E %.6f, 2 g E %.4f; dense fields by %s; the record's min(N - |v|) %+.6f at r %.3f, zeros %s)" % (
            lab, os.path.basename(jp), rec["T_over_rs"], ell, E, 2.0 * g * E, fmn, rec["min_N_minus_absv"], rec["r_min_N_minus_absv"],
            ", ".join("%.3f" % z for z in (rec.get("trapping_zeros") or [])) or "none"))
        L.append("")
        res = {}
        for fs, (Nf, vf, dense) in (("dense", (Nd, vd, True)), ("applied", (Naf, vaf, False))):
            rows, cross = scan(Nf, vf, R, dense)
            res[fs] = (rows, cross)
        L.append("| sigma (a) | sigma/(2gE) | Mcal dense | at r | Mcal applied | at r |")
        L.append("|---|---|---|---|---|---|")
        for (s, md, rd), (_, ma, ra) in zip(res["dense"][0], res["applied"][0]):
            L.append("| %g | %.6f | %.6f | %.3f | %.6f | %.3f |" % (s, s / (2.0 * g * E), md, rd, ma, ra))
        L.append("")
        for fs in ("dense", "applied"):
            rows, cross = res[fs]
            if cross is not None:
                L.append("- %s fields: Mcal crosses 1 at sigma_c %.4f a, sigma_c/(2gE) %.6f (against %.6f and %.6f: ratio %.4f and %.4f)." % (
                    fs, cross, cross / (2.0 * g * E), C_RETURN, C_DOUBLE, cross / (2.0 * g * E) / C_RETURN, cross / (2.0 * g * E) / C_DOUBLE))
            else:
                L.append("- %s fields: no crossing of Mcal = 1 over the scan (Mcal from %.6f at sigma 0 to %.6f at sigma %g): %s." % (
                    fs, rows[0][1], rows[-1][1], rows[-1][0], "no zero to remove (the unaveraged maximum is below 1)" if rows[0][1] < 1.0 else "above 1 throughout the scan"))
        md_ell = mcal(Nd, vd, R, ell)
        ma0 = res["applied"][0][0]
        L.append("- check: the dense fields averaged at the run's ell %g give Mcal %.8f at r %.3f; the stored applied fields' own unaveraged maximum %.8f at r %.3f (difference %.2e)." % (
            ell, md_ell[0], md_ell[1], ma0[1], ma0[2], md_ell[0] - ma0[1]))
        L.append("")
        F12._KAVG.clear()
    L.append("- every number above is printed by the script from the runs' jsons and npz files; the constants 0.260875 and 0.228195 (P21's estimate of the latter 0.2282052018) are the returns' (quoted); nothing is attributed or ruled.")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
