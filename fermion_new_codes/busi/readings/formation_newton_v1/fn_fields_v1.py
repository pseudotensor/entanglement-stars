#!/usr/bin/env python3
"""busi reading formation_newton_v1 (reviewer A): the fields-level reading. From the stored fields files (site fields N, v, N_applied,
v_applied on 2048 sites; node fields r_nodes, N_nodes, v_nodes read through a restricted unpickler that admits numpy's array
reconstruction only): 1 the Newton run's site and node fields against the production's at the 101 matched clocks; 2 the kept residual
against max |applied - record| on the sites; 3 min(N - |v|), its location and the roots of N - |v| on sites and nodes (linear
interpolation and a local cubic through four nodes solved with torch.linalg.solve) against the json's min, r_min, R_t, r_in; 4 the
production's to40 fields file against the from-0 member's fields file. CPU, torch for the small solves. Output: fields.json in OUT."""
import io, json, os, pickle, sys, zipfile
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fn_common import *
from numpy.lib import format as npfmt

torch.set_num_threads(8)
OUTJ = {}


def say(*a):
    print(*a, flush=True)


class RestrictedUnpickler(pickle.Unpickler):
    """admits only numpy's array/dtype reconstruction; anything else raises."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED or (module == "numpy.dtypes" and name.endswith("DType")):
            import importlib
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError("refused %s.%s" % (module, name))


def load_object_member(npz_path, member):
    """read one object-array member of an npz without np.load's pickle.load: header by numpy.lib.format, payload by the restricted unpickler."""
    with zipfile.ZipFile(npz_path) as z:
        with z.open(member + ".npy") as f:
            data = f.read()
    b = io.BytesIO(data)
    version = npfmt.read_magic(b)
    shape, fortran, dtype = npfmt._read_array_header(b, version)
    if dtype.hasobject:
        arr = RestrictedUnpickler(b).load()
        return np.asarray(arr, dtype=object), shape, str(dtype)
    arr = np.frombuffer(b.read(), dtype=dtype).reshape(shape)
    return arr, shape, str(dtype)


def load_fields(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("R", "T", "N", "v", "N_applied", "v_applied", "N_cinit", "v_cinit")}
    out["keys"] = list(z.files)
    nodes = {}
    for k in ("r_nodes", "N_nodes", "v_nodes"):
        try:
            a, shape, dt = load_object_member(path, k)
            nodes[k] = a
            nodes[k + "_shape"] = (list(shape), dt)
        except Exception as e:
            nodes[k] = None
            nodes[k + "_error"] = repr(e)[:200]
    out["nodes"] = nodes
    return out


def node_arrays(F, i):
    """the node radii and fields of record i as float arrays, whatever the object layout (per-record arrays, or one shared radius array)."""
    nd = F["nodes"]
    if nd.get("r_nodes") is None or nd.get("N_nodes") is None:
        return None, None, None

    def pick(a):
        a = np.asarray(a, dtype=object) if isinstance(a, np.ndarray) and a.dtype == object else a
        if isinstance(a, np.ndarray) and a.dtype == object:
            if a.ndim == 0:
                return pick(a.item())
            if a.shape[0] == F["T"].shape[0]:
                return np.asarray(a[i], dtype=float)
            return np.asarray(list(a), dtype=float)
        a = np.asarray(a, dtype=float)
        if a.ndim == 2 and a.shape[0] == F["T"].shape[0]:
            return a[i]
        return a
    r = pick(nd["r_nodes"])
    N = pick(nd["N_nodes"])
    v = pick(nd["v_nodes"])
    return r, N, v


def roots_linear(x, f):
    """all sign-change roots of f on the grid x by linear interpolation."""
    out = []
    for j in range(len(x) - 1):
        a, b = f[j], f[j + 1]
        if a == 0.0:
            out.append(float(x[j]))
        elif (a < 0 < b) or (b < 0 < a):
            out.append(float(x[j] + (x[j + 1] - x[j]) * a / (a - b)))
    return out


def root_cubic(x, f, j):
    """a cubic through the four grid points j-1..j+2 around the bracket [j, j+1] (Vandermonde solved with torch.linalg.solve), then
    Newton's iteration from the linear root; returns the refined root inside the bracket or None."""
    lo, hi = max(j - 1, 0), min(j + 3, len(x))
    xs = torch.tensor(x[lo:hi], dtype=torch.float64)
    fs = torch.tensor(f[lo:hi], dtype=torch.float64)
    if xs.numel() < 4:
        return None
    x0 = xs.mean()
    V = torch.stack([(xs - x0) ** k for k in range(xs.numel())], dim=1)
    c = torch.linalg.solve(V, fs)
    a, b = x[j], x[j + 1]
    t = torch.tensor(a + (b - a) * f[j] / (f[j] - f[j + 1]), dtype=torch.float64)
    for _ in range(50):
        u = t - x0
        p = sum(c[k] * u ** k for k in range(c.numel()))
        dp = sum(k * c[k] * u ** (k - 1) for k in range(1, c.numel()))
        if dp == 0:
            break
        step = p / dp
        t = t - step
        if abs(step) < 1e-14:
            break
    t = float(t)
    return t if (a - 1e-9 <= t <= b + 1e-9) else None


def analyse_grid(x, N, v):
    f = N - np.abs(v)
    k = int(np.argmin(f))
    roots = roots_linear(x, f)
    refined = []
    for j in range(len(x) - 1):
        a, b = f[j], f[j + 1]
        if (a < 0 < b) or (b < 0 < a):
            refined.append(root_cubic(x, f, j))
    # the minimum refined by a local cubic as well (the vertex of the cubic through 4 points around k)
    return {"min": float(f[k]), "r_min": float(x[k]), "roots_linear": roots, "roots_cubic": refined, "n_roots": len(roots), "spacing": float(np.median(np.diff(x))), "n": int(len(x))}


say("== load")
FN, FP, F0 = load_fields(NF), load_fields(PF), load_fields(P0F)
for tag, F in (("newton", FN), ("to40", FP), ("from0", F0)):
    say("  %s: T %s sites %d (R %g..%g) N %s; nodes: %s" % (tag, F["T"].shape, F["R"].shape[0], F["R"][0], F["R"][-1], F["N"].shape,
                                                          {k: v for k, v in F["nodes"].items() if k.endswith("_shape") or k.endswith("_error")}))
JN, JP = load(NJ), load(PJ)
RN, RP = records(JN), records(JP)
TN = np.array([float(r["T_over_rs"]) for r in RN])
TP = {ckey(r["T_over_rs"]): r for r in RP}
OUTJ["layout"] = {tag: {"T_n": int(F["T"].shape[0]), "sites": int(F["R"].shape[0]), "R_first": float(F["R"][0]), "R_last": float(F["R"][-1]),
                        "site_spacing": float(np.median(np.diff(F["R"]))), "keys": F["keys"], "nodes": {k: v for k, v in F["nodes"].items() if k.endswith("_shape") or k.endswith("_error")}}
                  for tag, F in (("newton", FN), ("to40", FP), ("from0", F0))}
# the fields files' T against the jsons' clocks
OUTJ["layout"]["newton_T_equals_json"] = bool(np.allclose(FN["T"], TN * RS, atol=1e-9))
OUTJ["layout"]["to40_T_equals_json"] = bool(np.allclose(FP["T"], np.array([float(r["T_over_rs"]) for r in RP]) * RS, atol=1e-9))
say("  newton fields T equals json clocks:", OUTJ["layout"]["newton_T_equals_json"], "; to40:", OUTJ["layout"]["to40_T_equals_json"])
# index maps by absolute T
ip = {float(t): i for i, t in enumerate(FP["T"])}
i0 = {float(t): i for i, t in enumerate(F0["T"])}
rn, Nn, vn, Nan, van = FN["R"], FN["N"], FN["v"], FN["N_applied"], FN["v_applied"]
OUTJ["layout"]["R_grids_equal"] = bool(np.array_equal(FN["R"], FP["R"]) and np.array_equal(FN["R"], F0["R"]))
OUTJ["layout"]["cinit_equal"] = {"N_cinit_max_abs_diff": float(np.max(np.abs(FN["N_cinit"] - FP["N_cinit"]))), "v_cinit_max_abs_diff": float(np.max(np.abs(FN["v_cinit"] - FP["v_cinit"])))}
say("  R grids equal:", OUTJ["layout"]["R_grids_equal"], "cinit:", OUTJ["layout"]["cinit_equal"])

# ---------------------------------------------------------------- 4 (first, cheap): to40 against from0
say("== 4. to40 fields against from0 fields")
common_T = sorted(set(ip) & set(i0))
eqN = [t for t in common_T if np.array_equal(FP["N"][ip[t]], F0["N"][i0[t]]) and np.array_equal(FP["v"][ip[t]], F0["v"][i0[t]]) and np.array_equal(FP["N_applied"][ip[t]], F0["N_applied"][i0[t]]) and np.array_equal(FP["v_applied"][ip[t]], F0["v_applied"][i0[t]])]
OUTJ["to40_vs_from0"] = {"common_records": len(common_T), "bitwise_equal_records": len(eqN), "differing_T_over_rs": [t / RS for t in common_T if t not in set(eqN)],
                         "matched_0_to_12p5_bitwise": all(t in set(eqN) for t in TN * RS)}
say("  ", json.dumps(OUTJ["to40_vs_from0"]))

# ---------------------------------------------------------------- 1. site fields, Newton against production
say("== 1. site fields at the 101 matched clocks")
rows = []
for i, t in enumerate(TN):
    j = ip[float(t * RS)]
    j0 = i0[float(t * RS)]
    dN = np.abs(Nn[i] - FP["N"][j]); dv = np.abs(vn[i] - FP["v"][j])
    dNa = np.abs(Nan[i] - FP["N_applied"][j]); dva = np.abs(van[i] - FP["v_applied"][j])
    fn_ = Nn[i] - np.abs(vn[i]); fp_ = FP["N"][j] - np.abs(FP["v"][j])
    rows.append({"T": float(t), "max_dN": float(dN.max()), "r_max_dN": float(rn[int(np.argmax(dN))]), "max_dv": float(dv.max()), "r_max_dv": float(rn[int(np.argmax(dv))]),
                 "max_dN_applied": float(dNa.max()), "max_dv_applied": float(dva.max()), "max_dNmv": float(np.abs(fn_ - fp_).max()),
                 "rms_dN": float(np.sqrt(np.mean(dN ** 2))), "rms_dv": float(np.sqrt(np.mean(dv ** 2))),
                 "from0_equals_to40_here": bool(np.array_equal(FP["N"][j], F0["N"][j0]) and np.array_equal(FP["v"][j], F0["v"][j0]))})
OUTJ["site_rows"] = rows
mx = max(rows, key=lambda x: x["max_dN"]); mv = max(rows, key=lambda x: x["max_dv"])
OUTJ["site_summary"] = {"max_dN": mx["max_dN"], "max_dN_T": mx["T"], "max_dN_r": mx["r_max_dN"], "max_dv": mv["max_dv"], "max_dv_T": mv["T"], "max_dv_r": mv["r_max_dv"],
                        "max_dN_applied": max(x["max_dN_applied"] for x in rows), "max_dv_applied": max(x["max_dv_applied"] for x in rows),
                        "max_dNmv": max(x["max_dNmv"] for x in rows), "max_dNmv_T": max(rows, key=lambda x: x["max_dNmv"])["T"],
                        "at_T0": {k: rows[0][k] for k in ("max_dN", "max_dv")}, "at_0p125": {k: rows[1][k] for k in ("max_dN", "max_dv")},
                        "samples": {("%.3f" % x["T"]): (x["max_dN"], x["max_dv"]) for x in rows if abs(x["T"] - round(x["T"])) < 1e-9 or x["T"] >= 11.5},
                        "falls_dN": sum(1 for a, b in zip(rows, rows[1:]) if b["max_dN"] < a["max_dN"]), "max_at_last": (rows[-1]["max_dN"] == mx["max_dN"])}
say("  ", json.dumps(OUTJ["site_summary"])[:1500])

# ---------------------------------------------------------------- 2. the kept residual against max |applied - record|
say("== 2. kept residual against max |applied - record| (sites)")
kr = []
for i, r in enumerate(RN):
    q = newton_residuals(r)
    dN = float(np.abs(Nan[i] - Nn[i]).max()); dv = float(np.abs(van[i] - vn[i]).max())
    d = max(dN, dv)
    kr.append({"T": float(TN[i]), "max_applied_minus_record": d, "dN": dN, "dv": dv, "kept": q["kept"], "conf": q["conf"],
               "ratio_to_kept": (d / q["kept"] if q["kept"] else None), "ratio_to_conf": (d / q["conf"] if q["conf"] else None),
               "equals_kept": (q["kept"] is not None and abs(d - q["kept"]) <= 1e-12 * max(1.0, q["kept"]) + 1e-18),
               "equals_conf": (q["conf"] is not None and abs(d - q["conf"]) <= 1e-12 * max(1.0, q["conf"]) + 1e-18)})
pr = []
for i, t in enumerate(TN):
    j = ip[float(t * RS)]
    r = TP[ckey(t)]
    q = plain_residuals(r)
    dN = float(np.abs(FP["N_applied"][j] - FP["N"][j]).max()); dv = float(np.abs(FP["v_applied"][j] - FP["v"][j]).max())
    d = max(dN, dv)
    pr.append({"T": float(t), "max_applied_minus_record": d, "kept": q["kept"], "ratio": (d / q["kept"] if q["kept"] else None),
               "equals_kept": (q["kept"] is not None and abs(d - q["kept"]) <= 1e-12 * max(1.0, q["kept"]) + 1e-18)})
OUTJ["kept_identity"] = {"newton_rows": kr, "newton_equals_kept_count": sum(1 for x in kr if x["equals_kept"]), "newton_equals_conf_count": sum(1 for x in kr if x["equals_conf"]),
                         "newton_ratio_to_kept_range": [min(x["ratio_to_kept"] for x in kr if x["ratio_to_kept"]), max(x["ratio_to_kept"] for x in kr if x["ratio_to_kept"])],
                         "newton_ratio_to_conf_range": [min(x["ratio_to_conf"] for x in kr if x["ratio_to_conf"]), max(x["ratio_to_conf"] for x in kr if x["ratio_to_conf"])],
                         "prod_rows": pr, "prod_equals_kept_count": sum(1 for x in pr if x["equals_kept"]),
                         "prod_ratio_range": [min(x["ratio"] for x in pr if x["ratio"]), max(x["ratio"] for x in pr if x["ratio"])]}
say("  newton: equals kept on %d, equals conf on %d of 101; ratio to kept %s; ratio to conf %s" % (OUTJ["kept_identity"]["newton_equals_kept_count"], OUTJ["kept_identity"]["newton_equals_conf_count"], OUTJ["kept_identity"]["newton_ratio_to_kept_range"], OUTJ["kept_identity"]["newton_ratio_to_conf_range"]))
say("  prod: equals kept on %d of 101; ratio range %s" % (OUTJ["kept_identity"]["prod_equals_kept_count"], OUTJ["kept_identity"]["prod_ratio_range"]))
for x in kr:
    if x["T"] in (0.125, 11.5, 12.125, 12.5):
        say("   T %.3f max|applied-record| %.3e (dN %.3e dv %.3e) kept %.3e conf %.3e" % (x["T"], x["max_applied_minus_record"], x["dN"], x["dv"], x["kept"], x["conf"]))

# ---------------------------------------------------------------- 3. min(N - |v|) and the roots on sites and nodes
say("== 3. min(N - |v|) and roots, sites and nodes, against the json")
mins = []
for i, r in enumerate(RN):
    q = quantities(r)
    t = float(TN[i])
    j = ip[float(t * RS)]
    sn = analyse_grid(rn, Nn[i], vn[i])
    sp = analyse_grid(FP["R"], FP["N"][j], FP["v"][j])
    rN, NN, vN = node_arrays(FN, i)
    rP, NP, vP = node_arrays(FP, j)
    nn = analyse_grid(rN, NN, vN) if rN is not None else None
    npr = analyse_grid(rP, NP, vP) if rP is not None else None
    qp = quantities(TP[ckey(t)])
    mins.append({"T": t, "json_min_n": q["minNv"], "json_rmin_n": q["r_minNv"], "json_Rt_n": q["Rt"], "json_rin_n": q["rin"], "json_zeros_n": q["zeros"],
                 "json_min_p": qp["minNv"], "json_rmin_p": qp["r_minNv"], "json_Rt_p": qp["Rt"], "json_rin_p": qp["rin"],
                 "site_n": sn, "site_p": sp, "node_n": nn, "node_p": npr})
OUTJ["mins"] = mins
# summaries
def summarize(which):
    out = {"n": 0, "min_minus_json_max": 0.0, "min_minus_json_T": None, "roots_count_equal_json": 0, "sign_equal_json": 0,
           "Rt_linear_minus_json_max": None, "Rt_cubic_minus_json_max": None, "rin_cubic_minus_json_max": None, "rows_with_roots": []}
    for m in mins:
        g = m[which]
        if g is None:
            continue
        out["n"] += 1
        d = g["min"] - m["json_min_n"]
        if abs(d) > abs(out["min_minus_json_max"]):
            out["min_minus_json_max"], out["min_minus_json_T"] = d, m["T"]
        if g["n_roots"] == len(m["json_zeros_n"]):
            out["roots_count_equal_json"] += 1
        if (g["min"] < 0) == (m["json_min_n"] < 0):
            out["sign_equal_json"] += 1
        if m["json_Rt_n"] is not None and g["n_roots"] >= 2:
            Rl, Rc = max(g["roots_linear"]), max(x for x in g["roots_cubic"] if x is not None) if any(x is not None for x in g["roots_cubic"]) else None
            rl, rc = min(g["roots_linear"]), min(x for x in g["roots_cubic"] if x is not None) if any(x is not None for x in g["roots_cubic"]) else None
            row = {"T": m["T"], "json_Rt": m["json_Rt_n"], "Rt_linear": Rl, "Rt_cubic": Rc, "json_rin": m["json_rin_n"], "rin_linear": rl, "rin_cubic": rc,
                   "json_min": m["json_min_n"], "json_rmin": m["json_rmin_n"], "grid_min": g["min"], "grid_rmin": g["r_min"]}
            out["rows_with_roots"].append(row)
            for key, val, ref in (("Rt_linear_minus_json_max", Rl, m["json_Rt_n"]), ("Rt_cubic_minus_json_max", Rc, m["json_Rt_n"]), ("rin_cubic_minus_json_max", rc, m["json_rin_n"])):
                if val is not None and ref is not None and (out[key] is None or abs(val - ref) > abs(out[key])):
                    out[key] = val - ref
    return out


OUTJ["site_vs_json_newton"] = summarize("site_n")
OUTJ["node_vs_json_newton"] = summarize("node_n")
# the production's own node roots against its json (same routine on the production's node fields and json values)
prod_rows = []
for m in mins:
    g = m["node_p"]
    if g is None or m["json_Rt_p"] is None or g["n_roots"] < 2:
        continue
    Rc = max(x for x in g["roots_cubic"] if x is not None); rc = min(x for x in g["roots_cubic"] if x is not None)
    prod_rows.append({"T": m["T"], "json_Rt": m["json_Rt_p"], "Rt_cubic": Rc, "json_rin": m["json_rin_p"], "rin_cubic": rc, "json_min": m["json_min_p"], "node_min": g["min"], "node_rmin": g["r_min"]})
OUTJ["node_vs_json_prod"] = prod_rows
# node against node, Newton vs production
nn_rows = []
for i, m in enumerate(mins):
    if m["node_n"] is None or m["node_p"] is None:
        continue
    rN, NN, vN = node_arrays(FN, i)
    j = ip[float(m["T"] * RS)]
    rP, NP, vP = node_arrays(FP, j)
    if rN.shape == rP.shape and np.allclose(rN, rP):
        nn_rows.append({"T": m["T"], "max_dN_nodes": float(np.abs(NN - NP).max()), "max_dv_nodes": float(np.abs(vN - vP).max()), "n_nodes": int(len(rN)), "node_spacing": float(np.median(np.diff(rN)))})
    else:
        nn_rows.append({"T": m["T"], "grids_differ": True, "n_nodes_n": int(len(rN)), "n_nodes_p": int(len(rP))})
OUTJ["node_vs_node"] = {"rows": nn_rows, "max_dN_nodes": max((x.get("max_dN_nodes", 0.0) for x in nn_rows), default=None), "max_dv_nodes": max((x.get("max_dv_nodes", 0.0) for x in nn_rows), default=None),
                        "grids_differ_any": any(x.get("grids_differ") for x in nn_rows)}
say("  site vs json (newton):", json.dumps({k: v for k, v in OUTJ["site_vs_json_newton"].items() if k != "rows_with_roots"}))
say("  node vs json (newton):", json.dumps({k: v for k, v in OUTJ["node_vs_json_newton"].items() if k != "rows_with_roots"}))
for row in OUTJ["node_vs_json_newton"]["rows_with_roots"]:
    say("   nodes T %.3f json R_t %.6f cubic %s linear %s | json r_in %.6f cubic %s | json min %.6f at %.4f; node min %.6f at %.3f" % (row["T"], row["json_Rt"], fmt(row["Rt_cubic"], "%.6f"), fmt(row["Rt_linear"], "%.6f"), row["json_rin"], fmt(row["rin_cubic"], "%.6f"), row["json_min"], row["json_rmin"], row["grid_min"], row["grid_rmin"]))
for row in OUTJ["site_vs_json_newton"]["rows_with_roots"]:
    say("   sites T %.3f json R_t %.6f cubic %s linear %s | json r_in %.6f cubic %s | json min %.6f; site min %.6f at %.1f" % (row["T"], row["json_Rt"], fmt(row["Rt_cubic"], "%.6f"), fmt(row["Rt_linear"], "%.6f"), row["json_rin"], fmt(row["rin_cubic"], "%.6f"), row["json_min"], row["grid_min"], row["grid_rmin"]))
for row in prod_rows:
    say("   prod nodes T %.3f json R_t %.6f cubic %.6f | json r_in %.6f cubic %.6f | json min %.6f node min %.6f at %.3f" % (row["T"], row["json_Rt"], row["Rt_cubic"], row["json_rin"], row["rin_cubic"], row["json_min"], row["node_min"], row["node_rmin"]))
say("  node vs node:", json.dumps({k: v for k, v in OUTJ["node_vs_node"].items() if k != "rows"}), [(x["T"], x.get("max_dN_nodes"), x.get("max_dv_nodes")) for x in nn_rows if x["T"] in (0.125, 11.5, 12.125, 12.5)])
# the site minimum relative to the json's (the ISO128 reading's "site against node" offset), over all records
off = [(m["T"], m["site_n"]["min"] - m["json_min_n"]) for m in mins]
OUTJ["site_min_offset"] = {"max_abs": max(abs(d) for _, d in off), "at_T": max(off, key=lambda x: abs(x[1]))[0], "at_12p125": next(d for t, d in off if t == 12.125),
                           "first_negative_site_min_T": next((m["T"] for m in mins if m["site_n"]["min"] < 0), None),
                           "first_negative_node_min_T": next((m["T"] for m in mins if m["node_n"] is not None and m["node_n"]["min"] < 0), None),
                           "first_negative_json_min_T": next((m["T"] for m in mins if m["json_min_n"] < 0), None)}
say("  site-min offset:", json.dumps(OUTJ["site_min_offset"]))
h = dump(OUTJ, OUT + "/fields.json")
say("wrote", OUT + "/fields.json", h)
