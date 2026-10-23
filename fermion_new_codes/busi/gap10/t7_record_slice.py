#!/usr/bin/env python
"""t7_record_slice.py -- the T7 record table for gap10_run_v2.py from a solve-lane collapse history (GAP10 build v1, section 4; RUN_ORDER_v2 step 3).

Reads a collapse history of engine_collapse_v17 in place (its *_fields.npz and, when present, the collapse_<tag>.json beside it), picks one
stored slice, and writes its applied back-averaged fields as the radial table the run reads with
    --profile file --profile_file record_slice.npz --sampling point
keys r, N, v in lattice units (a = 1), v negative inward, one row per shell of the solve's chain. Beside the table a JSON with the provenance:
the source paths and md5s, the slice clock, 2GE/a (the tail's r v^2 and 2 g E of the record), ma, ell/a, RBFRAC (the matter edge over 2GE),
the roots of N - |v| on the written table (linear interpolation, the solve lane's applied-root rule, and the natural-spline crossings on the
[100] and [111] axes as gap10_run_v2 allocates them with point sampling), the source and receiver sites of the T7 depths with their field values,
and the exterior check against N = 1, v = -sqrt(2GE/r) (the deviation table and the radius beyond which the stated tolerances hold).
--check prints everything and writes nothing. Exit code 2 when the chosen slice has no trapped region (then T7 is skipped).

Which fields: N_applied, v_applied (the fields of the last self-consistency iteration's h_m, the Hamiltonian that continued the state; in a
run with --site-fields averaged they are the kernel averages of the dense constraint solution, engine_fields_v12.site_fields_kernel_average,
an isotropic Gaussian of standard deviation ell per axis). They are what the fermion saw; no second averaging is applied by the run.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

try:
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

SQRT3 = math.sqrt(3.0)
DEFAULT_SOURCE = "<internal>/collapse_set17_rs32_A_R2_tau8_ell4_avg_fields.npz"
DEFAULT_OUT = "<internal>/record_slice.npz"
POD_RECORD = "<internal>/record_slice.npz"
POD_CODE = "<internal>/gap10"
POD_RUNS = "<internal>/gap10"


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_rows(a):
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def sibling_json(npz_path):
    """collapse_<tag>_fields.npz -> collapse_<tag>.json beside it (None when absent)."""
    d, b = os.path.split(npz_path)
    if b.endswith("_fields.npz"):
        p = os.path.join(d, b[:-len("_fields.npz")] + ".json")
        if os.path.exists(p):
            return p
    return None


def roots_linear(r, c):
    """The zeros of c(r) between the table's points by linear interpolation (the solve lane's applied-root rule), with the slope per a."""
    out = []
    s = np.sign(c)
    for i in np.flatnonzero(s[:-1] * s[1:] < 0):
        sl = (c[i + 1] - c[i]) / (r[i + 1] - r[i])
        out.append((float(r[i] - c[i] / sl), float(sl)))
    return out


def axis_crossings(r_tab, N_tab, u_tab, direction, rs_int, H):
    """The crossings of N - |v| on the [100] or [111] axis as gap10_run_v2 computes them with --sampling point: the nodes n e (n = 0 .. H + 2)
    carry the table's values at r = |n e| (linear interpolation; u = 0 at the origin, N(0) = the table's first N), a natural cubic spline
    through them, the zeros on [0.5, min(3 rs, H)] bracketed on 20001 points and refined by brentq, the slope by the central difference at
    h = 0.05 (gap10_core_v2.crossings_between). Returns (list of (r, slope, inverse |slope|), node radii)."""
    nax = np.arange(0, H + 3).astype(float)
    rn = nax if direction == "100" else nax * SQRT3
    u_n = np.where(rn > 0.0, np.interp(np.maximum(rn, r_tab[0]), r_tab, u_tab), 0.0)
    N_n = np.interp(rn, r_tab, N_tab, left=N_tab[0])
    if not HAVE_SCIPY:
        return None, rn
    cs = CubicSpline(rn, N_n - u_n, bc_type="natural")
    lo, hi = 0.5, min(3.0 * rs_int, float(H))
    rg = np.linspace(lo, hi, 20001)
    g = cs(rg)
    out = []
    for i in np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:])):
        try:
            rr = brentq(cs, rg[i], rg[i + 1])
        except ValueError:
            continue
        sl = float((cs(rr + 0.05) - cs(rr - 0.05)) / 0.1)
        if sl != 0.0:
            out.append((float(rr), sl, 1.0 / abs(sl)))
    return out, rn


def site_on_direction(r, direction):
    if direction == "100":
        n = int(round(r))
        return [n, 0, 0], float(n)
    n = int(round(r / SQRT3))
    return [n, n, n], float(n * SQRT3)


def slice_table(R, Nk, vk):
    """Per stored slice: the linear-interpolation roots of N - |v|, its minimum and the tail's r v^2."""
    rows = []
    for k in range(Nk.shape[0]):
        c = Nk[k] - np.abs(vk[k])
        rt = roots_linear(R, c)
        rows.append(dict(k=k, roots=[q[0] for q in rt], slopes=[q[1] for q in rt], min_c=float(c.min()), r_min_c=float(R[int(np.argmin(c))]),
                         two_gE_tail=float(R[-1] * vk[k][-1] ** 2), trapped=bool(c.min() < 0.0 and len(rt) >= 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="write the T7 record table (r, N, v) of one trapped slice of a solve-lane collapse history")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="the history's *_fields.npz (read in place)")
    ap.add_argument("--json", default=None, help="the collapse_<tag>.json beside it (default: found beside --source when present)")
    ap.add_argument("--fields", choices=["applied", "dense"], default="applied", help="applied = N_applied, v_applied (what the fermion saw); dense = N, v")
    ap.add_argument("--select", choices=["clock", "deepest", "widest", "latest"], default="clock",
                    help="clock: the slice at --T_over_rs; deepest: the most negative min(N - |v|); widest: the widest annulus; latest: the last trapped slice")
    ap.add_argument("--T_over_rs", type=float, default=18.0, help="the slice clock in units of the record's nominal r_s (the engine's --rs)")
    ap.add_argument("--rs_int", type=int, default=None, help="override the recommended integer --rs of the run (default round(2GE/a from the tail))")
    ap.add_argument("--depths", type=float, nargs="*", default=[2.0, 8.0])
    ap.add_argument("--receivers", type=float, nargs="*", default=[1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--tol_N", type=float, default=1e-4, help="exterior tolerance on |N - 1|")
    ap.add_argument("--tol_v", type=float, default=1e-3, help="exterior tolerance on |r v^2 / 2GE - 1|")
    ap.add_argument("--out", default=DEFAULT_OUT, help="the npz to write (the JSON goes beside it with the same stem)")
    ap.add_argument("--check", action="store_true", help="print the numbers and write nothing")
    a = ap.parse_args()

    src = os.path.abspath(a.source)
    jpath = a.json if a.json else sibling_json(src)
    z = np.load(src, allow_pickle=False)
    keys = list(z.files)
    R = np.asarray(z["R"], dtype=np.float64)
    T = np.asarray(z["T"], dtype=np.float64)
    Nkey, vkey = ("N_applied", "v_applied") if a.fields == "applied" else ("N", "v")
    Nk = np.asarray(z[Nkey], dtype=np.float64)
    vk = np.asarray(z[vkey], dtype=np.float64)
    g = float(z["g"]) if "g" in keys else None
    ell = float(z["ell"]) if "ell" in keys else None
    tau = float(z["tau"]) if "tau" in keys else None
    J = json.load(open(jpath)) if jpath else None
    jargs = (J or {}).get("args", {}) or {}
    m = float(jargs["m"]) if "m" in jargs else (float(z["m"]) if "m" in keys else None)
    rs_nom = float(jargs["rs"]) if "rs" in jargs else (float(z["rs"]) if "rs" in keys else None)
    if g is None and "g" in jargs:
        g = float(jargs["g"])
    if ell is None and "ell" in jargs:
        ell = float(jargs["ell"])
    if m is None or rs_nom is None:
        sys.exit("the fermion mass m or the nominal r_s is not in the npz nor in a json beside it: give --json")
    site_fields = (J or {}).get("head", {}).get("site_fields") if J else None

    # ---- the stored slices and the choice
    rows = slice_table(R, Nk, vk)
    trapped = [q for q in rows if q["trapped"]]
    if a.select == "clock":
        hit = [k for k in range(len(T)) if abs(T[k] / rs_nom - a.T_over_rs) < 1e-6]
        if not hit:
            sys.exit("no stored slice at T/rs %g (stored clocks %s ...)" % (a.T_over_rs, np.round(T[:8] / rs_nom, 3).tolist()))
        k0 = hit[0]
    elif a.select == "deepest":
        k0 = min(rows, key=lambda q: q["min_c"])["k"]
    elif a.select == "widest":
        k0 = max(trapped, key=lambda q: q["roots"][-1] - q["roots"][0])["k"] if trapped else None
    else:
        k0 = trapped[-1]["k"] if trapped else None
    print("source %s (%d slices, chain of %d shells r = %g .. %g; fields %s/%s; g %s, ell %s, tau %s, m %s, nominal r_s %s; site fields: %s)" % (
        src, len(T), len(R), R[0], R[-1], Nkey, vkey, g, ell, tau, m, rs_nom, site_fields))
    print("json %s" % (jpath or "none"))
    print("stored slices with a trapped region of %s (linear-interpolation roots of N - |v|):" % Nkey)
    for q in trapped:
        print("  k %3d  T/rs %7.3f  roots %s  min(N-|v|) %+.5f at r %g  tail r v^2 %.5f" % (
            q["k"], T[q["k"]] / rs_nom, ["%.4f" % x for x in q["roots"]], q["min_c"], q["r_min_c"], q["two_gE_tail"]))
    if not trapped:
        print("no stored slice of this history holds a trapped region of the %s fields: T7 is skipped" % Nkey)
    if k0 is None or not rows[k0]["trapped"]:
        print("the chosen slice (%s) holds no trapped region: T7 is skipped" % (("k %d, T/rs %.3f" % (k0, T[k0] / rs_nom)) if k0 is not None else "none"))
        sys.exit(2)

    # ---- the slice
    N = Nk[k0].copy()
    v = vk[k0].copy()
    u = np.abs(v)
    c = N - u
    if not np.all(v <= 0.0):
        print("WARNING: v is not negative everywhere (max v %+.3e): the inward sign convention is violated; the table is written as stored" % v.max())
    rec = J["records"][k0] if (J and "records" in J and len(J["records"]) > k0) else None
    if rec is not None and abs(float(rec.get("T", T[k0])) - T[k0]) > 1e-9:
        rec = next((q for q in J["records"] if abs(float(q.get("T", -1)) - T[k0]) < 1e-9), None)
    E = float(rec["E"]) if rec and rec.get("E") is not None else None
    two_gE_json = (2.0 * g * E) if (E is not None and g is not None) else None
    two_gE_tail = float(R[-1] * v[-1] ** 2)
    two_gE_tail64 = float(np.mean(R[-64:] * v[-64:] ** 2))
    rs_int = a.rs_int if a.rs_int is not None else int(round(two_gE_tail))
    rel = abs(two_gE_tail - rs_int) / rs_int
    H = 3 * rs_int + 16
    rt_lin = roots_linear(R, c)
    print("\nchosen slice: k %d, T = %g = %.3f nominal r_s; E %s, 2 g E %s; tail r v^2 at r = %g: %.5f (mean over the last 64 shells %.5f)" % (
        k0, T[k0], T[k0] / rs_nom, ("%.6f" % E) if E is not None else "n/a", ("%.5f" % two_gE_json) if two_gE_json is not None else "n/a", R[-1], two_gE_tail, two_gE_tail64))
    print("the run's --rs %d: rs_file %.5f, relative difference %.3e (%s the 1 percent rule); H = %d, L = %d, corner at %.1f a" % (
        rs_int, two_gE_tail, rel, "inside" if rel < 0.01 else "OUTSIDE", H, 2 * H + 1, SQRT3 * H))
    print("N at r = %g: %.5f; N min %.5f at r %g; N max %.8f; v at r = %g: %+.5f; max |N - N_applied| on this slice %.2e, |v - v_applied| %.2e" % (
        R[0], N[0], N.min(), R[int(np.argmin(N))], N.max(), R[-1], v[-1], np.abs(z["N"][k0] - z["N_applied"][k0]).max(), np.abs(z["v"][k0] - z["v_applied"][k0]).max()))
    print("roots of N - |v| on the written table (linear interpolation; r, slope per a): %s; min(N - |v|) %+.5f at r %g" % (
        ["(%.4f, %+.6f)" % q for q in rt_lin], c.min(), R[int(np.argmin(c))]))
    dense = None
    if rec is not None:
        dense = dict(R_t=rec.get("R_t"), r_in=rec.get("r_in"), trapping_zeros=rec.get("trapping_zeros"), kappa_T=rec.get("kappa_T"),
                     min_N_minus_absv=rec.get("min_N_minus_absv"), N_min=rec.get("N_min"), E_over_E0=rec.get("E_over_E0"), R_sh=rec.get("R_sh"),
                     R_c=rec.get("R_c"), record_label=rec.get("record_label"), late_surface=rec.get("late_surface"),
                     indicator_pass=(rec.get("indicator") or {}).get("indicator_pass"))
        print("the record's dense roots (the constraint solution before the back-averaging): R_t %s, r_in %s, kappa_T %s, min(N - |v|) %s; E/E0 %s; R_sh %s, R_c %s; label: %s; indicator pass %s" % (
            dense["R_t"], dense["r_in"], dense["kappa_T"], dense["min_N_minus_absv"], dense["E_over_E0"], dense["R_sh"], dense["R_c"], dense["record_label"], dense["indicator_pass"]))

    # ---- the matter edge
    matter = None
    if "eps_tot" in keys and "eps_pairs" in keys:
        ed = np.asarray(z["eps_tot"][k0], dtype=np.float64) - np.asarray(z["eps_pairs"][k0], dtype=np.float64)
        tot = float(ed.sum())
        if tot > 0:
            cum = np.cumsum(ed)
            def r_at(q):
                return float(R[min(int(np.searchsorted(cum, q * tot)), len(R) - 1)])
            matter = dict(dust_energy=tot, R_50=r_at(0.5), R_90=r_at(0.9), R_sh_99=r_at(0.99), R_999=r_at(0.999), R_c=float(ed @ R / tot),
                          R_last_above_1em3=float(R[np.flatnonzero(ed > 1e-3 * ed.max())[-1]]), R_last_above_1em6=float(R[np.flatnonzero(ed > 1e-6 * ed.max())[-1]]))
            print("the dust (eps_tot - eps_pairs): total %.4f; 50, 90, 99, 99.9 percent inside r = %g, %g, %g, %g; centroid %.2f; last shell above 1e-3 (1e-6) of the maximum %g (%g)" % (
                tot, matter["R_50"], matter["R_90"], matter["R_sh_99"], matter["R_999"], matter["R_c"], matter["R_last_above_1em3"], matter["R_last_above_1em6"]))
    rbfrac = round(matter["R_sh_99"] / two_gE_tail, 2) if matter else None
    print("RBFRAC = R_sh / 2GE = %s (R_sh the radius enclosing 99 percent of the dust energy, the engine's own R_sh; the 90 percent radius over 2GE would be %s)" % (
        rbfrac, (round(matter["R_90"] / two_gE_tail, 2) if matter else None)))

    # ---- the exterior check
    devN = np.abs(N - 1.0)
    devv = np.abs(R * v * v / two_gE_tail - 1.0)
    bad = np.flatnonzero((devN > a.tol_N) | (devv > a.tol_v))
    r_vac = float(R[bad[-1] + 1]) if (bad.size and bad[-1] + 1 < len(R)) else (float(R[0]) if not bad.size else None)
    ext_rows = []
    for rr in sorted(set([1.25 * rs_int, 1.5 * rs_int, 2.0 * rs_int, 3.0 * rs_int, float(H), SQRT3 * H, 300.0, 500.0, float(R[-1])])):
        if rr > R[-1]:
            continue
        i = int(np.argmin(np.abs(R - rr)))
        ext_rows.append(dict(r=float(R[i]), N=float(N[i]), abs_v=float(u[i]), vac_abs_v=float(math.sqrt(two_gE_tail / R[i])), r_v2_over_2GE=float(R[i] * v[i] ** 2 / two_gE_tail)))
    print("exterior against N = 1, v = -sqrt(2GE/r) with 2GE = %.5f: tolerances |N - 1| < %g and |r v^2/2GE - 1| < %g hold for every shell beyond r_vac = %s" % (
        two_gE_tail, a.tol_N, a.tol_v, r_vac))
    for q in ext_rows:
        print("  r %7.1f: N %.6f  |v| %.5f  vacuum |v| %.5f  r v^2 / 2GE %.5f" % (q["r"], q["N"], q["abs_v"], q["vac_abs_v"], q["r_v2_over_2GE"]))
    if two_gE_json is not None:
        print("  the tail against 2 g E of the record: relative difference %.2e" % (abs(two_gE_tail - two_gE_json) / two_gE_json))

    # ---- the run's axes: crossings, sources, receivers
    axes = {}
    for d in ("100", "111"):
        cr, _ = axis_crossings(R, N, u, d, rs_int, H)
        ax = dict(crossings=cr, r_h=(cr[-1][0] if cr else None), kappa_h=(cr[-1][1] if cr else None), sources={}, receivers={})
        if cr:
            print("[%s] crossings as the run allocates them (r, slope, inverse |slope|): %s; r_h %.4f, kappa_h %.6f (1/(2 rs) = %.6f)" % (
                d, ["(%.4f, %+.6f, %.1f)" % q for q in cr], ax["r_h"], ax["kappa_h"], 1.0 / (2.0 * rs_int)))
            for dp in a.depths:
                pos, r = site_on_direction(ax["r_h"] - dp, d)
                Ns, us = float(np.interp(r, R, N, left=N[0])), float(np.interp(r, R, u)) if r > 0 else 0.0
                inner = cr[0][0] if len(cr) >= 2 else None
                ax["sources"]["%g" % dp] = dict(site=pos, r=r, N=Ns, abs_v=us, N_minus_abs_v=Ns - us, below_inner_root=(bool(r < inner) if inner else None))
                print("   depth %g: site %s r = %.4f  N %.5f  |v| %.5f  N - |v| %+.5f%s" % (dp, pos, r, Ns, us, Ns - us, "  (below the inner root %.4f: the untrapped core)" % inner if (inner and r < inner) else ""))
            for fr in a.receivers:
                pos, r = site_on_direction(fr * rs_int, d)
                if max(abs(x) for x in pos) > H:
                    continue
                Ns, us = float(np.interp(r, R, N)), float(np.interp(r, R, u))
                ax["receivers"]["%g" % fr] = dict(site=pos, r=r, N=Ns, abs_v=us, vac_abs_v=math.sqrt(two_gE_tail / r))
                print("   receiver %g rs: site %s r = %.3f  N %.5f  |v| %.5f  (vacuum %.5f)" % (fr, pos, r, Ns, us, math.sqrt(two_gE_tail / r)))
        else:
            print("[%s] crossings: %s" % (d, "scipy unavailable: not computed" if cr is None else "none"))
        axes[d] = ax

    # ---- the T7 fill values and commands
    fill = dict(RS=rs_int, MA=m, ELL=(int(ell) if ell is not None and float(ell).is_integer() else ell), RBFRAC=rbfrac)
    cmds = []
    for d in ("100", "111"):
        for dp in a.depths:
            cmds.append("python %s/gap10_run_v2.py --rs %d --ma %g --ell %s --Rb_frac %s --profile file --profile_file %s --sampling point --direction %s --depth %g --source both --out %s/runs/T7_record --inside_twins_max_depth -1" % (
                POD_CODE, rs_int, m, fill["ELL"], rbfrac, POD_RECORD, d, dp, POD_RUNS))
    print("\nT7 fill values: RS %d, MA %g, ELL %s (the record's ell: the engine's Gaussian kernel of standard deviation ell per axis; a label under --sampling point), RBFRAC %s" % (rs_int, m, fill["ELL"], rbfrac))
    for s_ in cmds:
        print("  " + s_)

    prov = dict(
        script="t7_record_slice.py", source_npz=src, md5_source_npz=md5_file(src), source_json=jpath, md5_source_json=(md5_file(jpath) if jpath else None),
        engine_script=(J or {}).get("script"), engine_args=dict(m=m, g=g, rs_nominal=rs_nom, ell=ell, tau=tau, site_fields=site_fields,
                                                                 kappa_sea=jargs.get("kappa_sea"), reference=jargs.get("reference"), R0_over_rs=jargs.get("R0_over_rs"), nsites=jargs.get("nsites"), tag=jargs.get("tag")),
        fields_written=dict(N=Nkey, v=vkey, units="lattice units, a = 1; v negative inward", n=int(len(R)), r_first=float(R[0]), r_last=float(R[-1]),
                            row_md5_N=md5_rows(N), row_md5_v=md5_rows(v), max_abs_N_minus_N_applied=float(np.abs(z["N"][k0] - z["N_applied"][k0]).max()),
                            max_abs_v_minus_v_applied=float(np.abs(z["v"][k0] - z["v_applied"][k0]).max()), v_nonpositive=bool(np.all(v <= 0.0)),
                            N_at_r_first=float(N[0]), N_min=float(N.min()), N_max=float(N.max())),
        slice=dict(index=k0, T=float(T[k0]), T_over_rs_nominal=float(T[k0] / rs_nom), E=E, E_over_E0=(dense or {}).get("E_over_E0"), two_gE_json=two_gE_json,
                   two_gE_tail_last=two_gE_tail, two_gE_tail_mean64=two_gE_tail64, dense_record=dense),
        roots_on_table=dict(rule="zeros of N - |v| between the table's shells by linear interpolation (the solve lane's applied-root rule)",
                            roots=[dict(r=q[0], slope=q[1]) for q in rt_lin], min_N_minus_abs_v=float(c.min()), r_min=float(R[int(np.argmin(c))])),
        axes=axes, matter=matter,
        exterior=dict(tol_N=a.tol_N, tol_v=a.tol_v, r_vac=r_vac, two_gE_used=two_gE_tail, table=ext_rows,
                      tail_vs_json_rel=(abs(two_gE_tail - two_gE_json) / two_gE_json if two_gE_json else None),
                      note="the vacuum form holds to the tolerances only beyond r_vac; inside the run's box the record's exterior is its own infalling matter"),
        T7=dict(fill=fill, rs_file_check=dict(rs_file=two_gE_tail, RS=rs_int, rel_diff=rel, inside_1_percent=bool(rel < 0.01)),
                ELL_note="the engine's ell is the standard deviation per axis of its Gaussian kernel; gap10's K_ell has per-axis variance ell^2/15, so the variance-matched K_ell width would be %.2f a; with --sampling point no averaging is applied and --ell is a label" % (SQRT3 * math.sqrt(5.0) * ell if ell else float("nan")),
                RBFRAC_note="R_sh (99 percent of the dust energy) over 2GE from the tail", commands=cmds),
        trapped_slices=[dict(k=q["k"], T_over_rs=float(T[q["k"]] / rs_nom), roots=q["roots"], min_c=q["min_c"], r_min_c=q["r_min_c"], two_gE_tail=q["two_gE_tail"]) for q in trapped],
    )
    if a.check:
        print("\n--check: nothing written (the table would go to %s)" % a.out)
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, r=R.astype(np.float64), N=N.astype(np.float64), v=v.astype(np.float64))
    prov["written_npz"] = os.path.abspath(a.out)
    prov["md5_written_npz"] = md5_file(a.out)
    jout = os.path.splitext(os.path.abspath(a.out))[0] + ".json"
    with open(jout, "w") as f:
        json.dump(prov, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    zz = np.load(a.out)
    assert np.array_equal(zz["r"], R) and np.array_equal(zz["N"], N) and np.array_equal(zz["v"], v)
    print("\nwritten %s (keys r, N, v; %d rows; md5 %s) and %s" % (a.out, len(R), prov["md5_written_npz"], jout))


if __name__ == "__main__":
    main()
