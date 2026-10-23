#!/usr/bin/env python
"""massless_char_compare_v1.py: the characteristic comparison of MASSLESS_ball_pathfinder v1.2 section 9.5 (ii), on stored files only.

The ingoing characteristic of the flat-slice chart is carried from r = R0 at T = 0 through a run's stored fields with
dr/dT = -(N + abs(v)), the group velocity dH/dp at p < 0 of the dictionary H = N E + v.p the reports print (inward v
negative); at N = 1 and v = -(r_s/r)^(1/2) this is -(1 + (r_s/r)^(1/2)) with the closed-form clock
T(r) = 2 r_s [ln 3 - x^2/2 + x - ln(1 + x)], x = (r/r_s)^(1/2), from 4 r_s, 1 + 2 ln(3/2) r_s at r_s (the control). The
alternative --speed lapse-shift uses dr/dT = -N (1 + abs(v)), the N v shift convention, which coincides with the
dictionary's at N = 1. Along the ray the label r v^2/(2 g N^2) is read with the fields at the ray, beside r v^2/(2 g),
the mass function M and W/(2g) of the s11 npz at the same (T, r), and the dense roots of the nearest record. Two rays
are carried: through the dense fields N, v and through the applied fields N_applied, v_applied of the fields npz.
Fields are linear in T between records and linear in r on their grid; the step is RK2 at 1/8 of the record spacing; a
ray stops at r_stop (default: the cell's edge ell) or at the last record. Nothing is re-run; no stored file is written.
usage: python massless_char_compare_v1.py --run DIR/TAG --md OUT.md --out-json OUT.json [--r-stop R_A] [--out-every-rs X] [--speed dictionary|lapse-shift]
       python massless_char_compare_v1.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np


class _NPZ(dict):
    """a dict standing in for a loaded npz in the self-test (files = keys)"""
    files = property(lambda self: list(self.keys()))


class Field:
    """a field stored per record on a grid: linear in T (in a) between records, linear in r on the grid"""

    def __init__(self, Ta, r, A):
        o = np.argsort(np.asarray(r, float))
        self.Ta, self.r, self.A = np.asarray(Ta, float), np.asarray(r, float)[o], np.asarray(A, float)[:, o]

    def __call__(self, Ta, r):
        j = int(np.clip(np.searchsorted(self.Ta, Ta, side="right") - 1, 0, len(self.Ta) - 2))
        w = min(max((Ta - self.Ta[j]) / (self.Ta[j + 1] - self.Ta[j]), 0.0), 1.0)
        return float(np.interp(r, self.r, (1.0 - w) * self.A[j] + w * self.A[j + 1]))


def orient(A, nrec, grids):
    """a per-record field array as (nrec, ngrid) with the name of the grid its other axis matches, else None"""
    A = np.asarray(A, float)
    if A.ndim != 2:
        return None
    for name, r in grids.items():
        if A.shape == (nrec, len(r)):
            return A, name
        if A.shape == (len(r), nrec) and len(r) != nrec:
            return A.T, name
    return None


def npz_T_in_a(T_rec_rs, T_npz, rs):
    """the npz T in a: the npz stores T in r_s (it matches T_over_rs) or in a (it matches T_over_rs times r_s)"""
    Tz = np.asarray(T_npz, float).ravel()
    for unit in (1.0, rs):
        t = Tz / unit
        k = np.array([int(np.argmin(np.abs(t - x))) for x in T_rec_rs])
        if np.all(np.abs(t[k] - np.asarray(T_rec_rs)) < 1e-6):
            return Tz * (rs / unit)
    sys.exit("the npz T (%d values) matches the records' T_over_rs (%d) neither in r_s nor in a" % (len(Tz), len(T_rec_rs)))


def grids_of(Z, J):
    """the candidate grids of the npz: R and r_nodes when 1-D, else the provisional 1..nsites"""
    G = {n: np.asarray(Z[n], float).ravel() for n in ("R", "r_nodes") if n in Z.files and np.asarray(Z[n]).ndim == 1}
    if "R" not in G and "nsites" in J["args"]:
        G["sites 1..nsites (provisional: the npz R unusable)"] = np.arange(1, int(J["args"]["nsites"]) + 1, dtype=float)
    return G


def _radii(z):
    """the radii in a stored trapping_zeros value: a number, a list of numbers, a list of dicts carrying r, radius or root, a list of lists with the radius first, or a dict carrying roots, r, radii or zeros; anything else is skipped"""
    if z is None or isinstance(z, bool):
        return []
    if isinstance(z, (int, float)):
        return [float(z)]
    if isinstance(z, dict):
        for key in ("roots", "r", "radii", "zeros"):
            if key in z:
                return _radii(z[key])
        return []
    out = []
    for x in (z if isinstance(z, (list, tuple)) else []):
        if isinstance(x, dict):
            for key in ("r", "radius", "root"):
                if key in x:
                    out += _radii(x[key])
                    break
        elif isinstance(x, (list, tuple)):
            out += _radii(x[0]) if x else []
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.append(float(x))
    return out


def stored_roots(rec):
    """the record's dense roots as the json stores them; an unreadable trapping_zeros gives no roots"""
    return sorted(_radii(rec.get("trapping_zeros")))


def control_clock(r, rs):
    """the control's clock in a at r from 4 r_s: N = 1, v = -(r_s/r)^(1/2), dr/dT = -(1 + (r_s/r)^(1/2))"""
    x = np.sqrt(r / rs)
    return 2.0 * rs * (np.log(3.0) - 0.5 * x * x + x - np.log(1.0 + x))


def carry(N, v, Ta, r0, r_stop, dt, marks, speed):
    """RK2 ingoing ray from (Ta[0], r0): the path [(T, r)] in a and the clocks at which the radii in marks are crossed"""
    def f(T, r):
        n, a = N(T, r), abs(v(T, r))
        return -(n + a) if speed == "dictionary" else -n * (1.0 + a)
    T, r, path, cross = float(Ta[0]), float(r0), [(float(Ta[0]), float(r0))], {}
    while r > r_stop and T < Ta[-1] - 1e-12:
        h = min(dt, float(Ta[-1]) - T)
        k1 = f(T, r)
        rn = r + h * f(T + 0.5 * h, r + 0.5 * h * k1)
        for m in marks:
            if m not in cross and r > m >= rn:
                cross[m] = T + h * (r - m) / (r - rn)
        T, r = T + h, rn
        path.append((T, r))
    return path, cross


def compare(J, Z, S, run, r_stop, out_every_rs, speed):
    rs, g, ell = float(J["args"]["rs"]), float(J["args"]["g"]), float(J["args"]["ell"])
    ball = J.get("head", {}).get("ball", {})
    R0 = float(ball.get("R0", float(J["args"]["ball_R0_over_rs"]) * rs))
    x0 = float(ball.get("x0", 0.5 * R0))
    recs = J["records"]
    T_rec = np.array([float(x["T_over_rs"]) for x in recs])
    Ta = npz_T_in_a(T_rec, Z["T"], rs)
    G = grids_of(Z, J)
    r_stop = ell if r_stop is None else float(r_stop)
    marks = sorted({x0, rs, r_stop}, reverse=True)
    dT = np.diff(np.unique(Ta))
    if not (len(dT) and dT.min() > 0):
        sys.exit("the npz T has no positive record spacing")
    dt = float(np.median(dT)) / 8.0
    first = next((i for i, x in enumerate(recs) if stored_roots(x)), None)
    M = W = None
    if S is not None:
        Ts = npz_T_in_a(T_rec, S["T"], rs)
        sg = {n: np.asarray(S[n], float).ravel() for n in ("r_nodes",) if n in S.files}
        oM = orient(S["M"], len(Ts), sg) if "M" in S.files else None
        oW = orient(S["W"], len(Ts), sg) if "W" in S.files else None
        M = Field(Ts, sg[oM[1]], oM[0]) if oM else None
        W = Field(Ts, sg[oW[1]], oW[0]) if oW else None
    out = {"script": "massless_char_compare_v1.py", "run": run, "rs": rs, "g": g, "ell": ell, "R0": R0, "x0": x0, "r_stop": r_stop, "dt_a": dt, "speed": speed,
           "control_clock_rs": {"%g" % m: float(control_clock(m, rs) / rs) for m in marks},
           "first_root_record_T_over_rs": (float(T_rec[first]) if first is not None else None), "rays": {}}
    for label, nk, vk in (("dense", "N", "v"), ("applied", "N_applied", "v_applied")):
        a = orient(Z[nk], len(Ta), G) if nk in Z.files else None
        b = orient(Z[vk], len(Ta), G) if vk in Z.files else None
        if a is None or b is None or a[1] != b[1]:
            out["rays"][label] = {"skipped": "%s and %s match no grid among %s" % (nk, vk, {n: len(r) for n, r in G.items()})}
            continue
        N, v = Field(Ta, G[a[1]], a[0]), Field(Ta, G[a[1]], b[0])
        path, cross = carry(N, v, Ta, R0, r_stop, dt, marks, speed)
        P = np.array(path)
        Ts_out = set(np.round(np.arange(0.0, P[-1, 0] + 1e-9, out_every_rs * rs), 9)) | set(np.round(list(cross.values()), 9))
        if first is not None:
            Ts_out.add(round(float(T_rec[first]) * rs, 9))
        rows = []
        for t in sorted(float(x) for x in Ts_out if x <= P[-1, 0] + 1e-9):
            r = float(np.interp(t, P[:, 0], P[:, 1]))
            n, vv = N(t, r), v(t, r)
            i = int(np.argmin(np.abs(T_rec * rs - t)))
            zs = stored_roots(recs[i])
            where = "inside the cell" if r <= ell else ("between the dense roots" if zs and zs[0] < r < zs[-1] else "outside the dense roots" if zs else "no dense root")
            rows.append({"T_over_rs": t / rs, "r_a": r, "r_over_rs": r / rs, "N": n, "v": vv, "label": r * vv * vv / (2.0 * g * n * n), "N2_label": r * vv * vv / (2.0 * g),
                         "M_s11": (M(t, r) if M else None), "W_over_2g_s11": (W(t, r) / (2.0 * g) if W else None),
                         "dense_roots_nearest_record": zs, "nearest_record_T_over_rs": float(T_rec[i]), "position": where})
        out["rays"][label] = {"grid": a[1], "points": len(G[a[1]]), "end_T_over_rs": float(P[-1, 0] / rs), "end_r_a": float(P[-1, 1]), "reached_r_stop": bool(P[-1, 1] <= r_stop),
                              "crossings_T_over_rs": {"%g" % m: float(c / rs) for m, c in cross.items()}, "label_at_R0": rows[0]["label"], "rows": rows}
    return out


def write_md(out, path):
    L = ["# The characteristic comparison on %s" % out["run"], "",
         "r_s/a %g, g %g, ell %g a, R0 %g a, x0 %g a. The ingoing characteristic dr/dT = %s is carried from r = R0 at T = 0 (RK2, step %.4g a, the fields linear in T between records and in r on their grid) and stopped at r %g a or at the last record; the control (N = 1, v = -(r_s/r)^(1/2)) crosses %s; the first record carrying a dense root is at T/r_s %s. Read by massless_char_compare_v1.py from the stored json and npz files (nothing re-run)."
         % (out["rs"], out["g"], out["ell"], out["R0"], out["x0"], "-(N + abs(v)), the dictionary's" if out["speed"] == "dictionary" else "-N (1 + abs(v)), the N v shift convention's", out["dt_a"], out["r_stop"],
            "; ".join("r %s a at T/r_s %.4f" % (m, c) for m, c in out["control_clock_rs"].items()), out["first_root_record_T_over_rs"])]
    for label, ray in out["rays"].items():
        L += ["", "## The ray through the %s fields" % label]
        if "skipped" in ray:
            L.append("skipped: %s" % ray["skipped"])
            continue
        L += ["on the grid %s (%d points); crossings: %s; the ray ends at T/r_s %.4f at r %.3f a (%s); the label at R0: %.6f."
              % (ray["grid"], ray["points"], "; ".join("r %s a at T/r_s %.4f" % (m, c) for m, c in ray["crossings_T_over_rs"].items()) or "none",
                 ray["end_T_over_rs"], ray["end_r_a"], "r_stop reached" if ray["reached_r_stop"] else "the last record reached first", ray["label_at_R0"]),
              "", "| T/r_s | r (a) | r/r_s | N | v | r v^2/(2 g N^2) | r v^2/(2g) | M (s11 npz) | W/(2g) (s11 npz) | dense roots of the nearest record (its T/r_s) | position |",
              "|---" * 11 + "|"]
        for w in ray["rows"]:
            f = lambda x: "-" if x is None else "%.6f" % x
            L.append("| %.4f | %.3f | %.4f | %.6f | %.6f | %.6f | %.6f | %s | %s | %s (%.4f) | %s |"
                     % (w["T_over_rs"], w["r_a"], w["r_over_rs"], w["N"], w["v"], w["label"], w["N2_label"], f(w["M_s11"]), f(w["W_over_2g_s11"]),
                        ", ".join("%.3f" % z for z in w["dense_roots_nearest_record"]) or "none", w["nearest_record_T_over_rs"], w["position"]))
    L += ["", "- the control's clocks are 2 r_s [ln 3 - x^2/2 + x - ln(1 + x)], x = (r/r_s)^(1/2), from 4 r_s; the label is the fields' r v^2/(2 g N^2) at the ray; every number above is computed by massless_char_compare_v1.py from the files named."]
    open(path, "w").write("\n".join(L) + "\n")


def selftest():
    rs, g, ell = 16.0, 0.25, 2.0
    r = np.arange(0.5, 80.0, 0.25)
    T = np.arange(0.0, 3.0001, 0.125)
    v = -np.sqrt(rs / r)[None, :] * np.ones((len(T), 1))
    N = np.ones_like(v)
    recs = [{"T_over_rs": float(t), "trapping_zeros": []} for t in T]
    J = {"args": {"rs": rs, "g": g, "ell": ell, "ball_R0_over_rs": 4.0, "nsites": len(r)}, "head": {"ball": {"R0": 4.0 * rs, "x0": 2.0 * rs}}, "records": recs}
    Z = _NPZ({"T": T, "R": r, "r_nodes": np.linspace(0.5, 80.0, 9), "N": N, "v": v, "N_applied": N.T, "v_applied": v.T})
    S = _NPZ({"T": T * rs, "r_nodes": r, "M": (rs / (2.0 * g)) * np.ones_like(v), "W": rs * np.ones_like(v)})
    E0 = rs / (2.0 * g)
    for speed in ("dictionary", "lapse-shift"):
        out = compare(J, Z, S, "selftest", None, 0.25, speed)
        assert abs(out["control_clock_rs"]["16"] - (1.0 + 2.0 * np.log(1.5))) < 1e-12, out["control_clock_rs"]
        for label in ("dense", "applied"):
            ray = out["rays"][label]
            assert ray["reached_r_stop"] and set(ray["crossings_T_over_rs"]) == {"32", "16", "2"}, ray
            for m, c in ray["crossings_T_over_rs"].items():
                assert abs(c - out["control_clock_rs"][m]) < 2e-3, (speed, label, m, c, out["control_clock_rs"][m])
            labs = [w["label"] for w in ray["rows"] if w["r_a"] > 2.0 * ell]
            assert max(abs(x - E0) for x in labs) < 1e-3 * E0, (speed, label, labs)
            assert all(abs(w["M_s11"] - E0) < 1e-9 and abs(w["W_over_2g_s11"] - E0) < 1e-9 for w in ray["rows"]), label
        write_md(out, os.devnull)
    print("SELFTEST PASS: the stationary control (N = 1, v = -(r_s/r)^(1/2), r_s/a 16) reaches 2 r_s, r_s and the cell edge at the closed-form clocks within 2e-3 r_s on both speed conventions, the label r v^2/(2 g N^2) stays r_s/(2g) = %g within 1e-3, the s11 fields read in a and the fields in r_s" % E0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--md")
    ap.add_argument("--out-json")
    ap.add_argument("--r-stop", type=float, default=None, help="the ray's stopping radius in a (default: the cell's edge ell)")
    ap.add_argument("--out-every-rs", type=float, default=0.125)
    ap.add_argument("--speed", choices=("dictionary", "lapse-shift"), default="dictionary")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.run and a.md and a.out_json):
        ap.error("--run, --md and --out-json are needed (or --selftest)")
    d, tag = os.path.split(a.run.rstrip("/"))
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ps = os.path.join(d, "collapse_%s_s11.npz" % tag)
    S = np.load(ps, allow_pickle=True) if os.path.exists(ps) else None
    out = compare(J, Z, S, a.run, a.r_stop, a.out_every_rs, a.speed)
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    write_md(out, a.md)
    for label, ray in out["rays"].items():
        print("%s: %s" % (label, json.dumps({k: ray[k] for k in ray if k != "rows"}, default=float)))
    print("composed %s" % a.md)


if __name__ == "__main__":
    main()
