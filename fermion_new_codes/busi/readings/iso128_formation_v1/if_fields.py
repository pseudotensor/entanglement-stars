#!/usr/bin/env python
"""if_fields.py (busi reading of ISO128_formation_readout_v1): which stored fields carry the records' min(N - |v|), and what the
T0011.50 slice holds. Own code on the H2 run's _fields.npz (site fields N, v, N_applied, v_applied on 4096 sites; node fields on
1025 nodes), the T0011.50 and T0005.00 slices (site fields N_applied, v_applied only; the orbitals are not read) and the json.
Writes OUT/fields.json. CPU only; numpy on 4096-length arrays."""
import os
import sys
import zipfile

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import if_common as C  # noqa: E402


def minimum(x, f):
    i = int(np.argmin(f))
    return {"min": float(f[i]), "r": float(x[i]), "i": i}


def sign_changes(f):
    s = np.sign(f)
    s = s[s != 0]
    return int(np.sum(s[:-1] != s[1:]))


def roots_linear(x, f):
    out = []
    for i in range(len(f) - 1):
        if f[i] == 0:
            out.append(float(x[i]))
        elif f[i] * f[i + 1] < 0:
            out.append(float(x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])))
    return out


def spline_min(x, f, lo, hi, k=5, n=20001):
    """The minimum of the degree-k interpolating spline through (x, f) on a fine grid in [lo, hi] (the sites are at spacing 1 a;
    far from the centre, so no parity extension is needed)."""
    s = make_interp_spline(x, f, k=k)
    xx = np.linspace(lo, hi, n)
    yy = s(xx)
    j = int(np.argmin(yy))
    return {"min": float(yy[j]), "r": float(xx[j]), "sign_changes": sign_changes(yy)}


def main():
    os.makedirs(C.OUT, exist_ok=True)
    out = {}
    F = np.load(C.H2_FIELDS, allow_pickle=True)
    out["fields_keys"] = {k: (str(F[k].dtype), list(F[k].shape)) for k in F.files}
    R = F["R"]
    T = F["T"]
    J = C.load(C.H2_JSON)
    recs = C.records(J)
    rs = float(J["args"]["rs"])
    assert len(T) == len(recs) and np.allclose(T / rs, [r["T_over_rs"] for r in recs])
    out["sites"] = {"n": int(len(R)), "first": float(R[0]), "last": float(R[-1]), "spacing_uniform_1a": bool(np.allclose(np.diff(R), 1.0))}
    rn0 = np.asarray(F["r_nodes"][0], dtype=float)
    out["nodes_record0"] = {"n": int(len(rn0)), "first": float(rn0[0]), "last": float(rn0[-1]), "spacing_uniform": bool(np.allclose(np.diff(rn0), np.diff(rn0)[0])),
                            "spacing": float(np.diff(rn0)[0])}
    per = []
    for i, r in enumerate(recs):
        fs = F["N"][i] - np.abs(F["v"][i])
        fa = F["N_applied"][i] - np.abs(F["v_applied"][i])
        rn = np.asarray(F["r_nodes"][i], dtype=float)
        fn = np.asarray(F["N_nodes"][i], dtype=float) - np.abs(np.asarray(F["v_nodes"][i], dtype=float))
        ms, ma, mn = minimum(R, fs), minimum(R, fa), minimum(rn, fn)
        per.append({"T": float(r["T_over_rs"]), "json_min": float(r["min_N_minus_absv"]), "json_r": float(r["r_min_N_minus_absv"]),
                    "site_min": ms["min"], "site_r": ms["r"], "applied_min": ma["min"], "applied_r": ma["r"], "node_min": mn["min"], "node_r": mn["r"],
                    "n_nodes": int(len(rn)), "site_minus_json": ms["min"] - float(r["min_N_minus_absv"]), "node_minus_json": mn["min"] - float(r["min_N_minus_absv"]),
                    "site_minus_node": ms["min"] - mn["min"], "applied_minus_site_maxabs": float(np.max(np.abs(F["N_applied"][i] - F["N"][i]))),
                    "vapplied_minus_v_maxabs": float(np.max(np.abs(F["v_applied"][i] - F["v"][i]))), "site_sign_changes": sign_changes(fs), "node_sign_changes": sign_changes(fn),
                    "json_zeros": C.zeros_of(r)})
    out["per_record"] = per
    sel = {p["T"]: p for p in per}
    out["at_clocks"] = {t: sel[t] for t in (5.0, 11.0, 11.0625, 11.125, 11.1875, 11.25) if t in sel}
    out["offset_summary"] = {"site_minus_node_max": max(p["site_minus_node"] for p in per), "site_minus_node_min": min(p["site_minus_node"] for p in per),
                             "node_minus_json_maxabs": max(abs(p["node_minus_json"]) for p in per), "site_minus_json_maxabs": max(abs(p["site_minus_json"]) for p in per),
                             "node_minus_json_at_11p25": sel[11.25]["node_minus_json"], "site_minus_node_last_8": [(p["T"], p["site_minus_node"]) for p in per[-8:]],
                             "records_with_site_sign_change": [p["T"] for p in per if p["site_sign_changes"]], "records_with_node_sign_change": [p["T"] for p in per if p["node_sign_changes"]]}
    # a spline through the site fields at 11.25: does the sub-site minimum reach the node reading?
    i = C.at_clock(recs, 11.25)
    fs = F["N"][i] - np.abs(F["v"][i])
    out["spline_11p25"] = {"site_quintic": spline_min(R, fs, 60.0, 160.0), "node_reading": sel[11.25]["node_min"], "json": sel[11.25]["json_min"]}
    # the T0011.50 slice
    z = zipfile.ZipFile(C.SLICE_11p5)
    names = [n.filename for n in z.infolist()]
    out["slice_11p5_members"] = names
    out["slice_11p5_has_node_fields"] = any(("node" in n) for n in names)
    S = np.load(C.SLICE_11p5)
    Rs, Na, va = S["R"], S["N_applied"], S["v_applied"]
    f = Na - np.abs(va)
    out["slice_11p5"] = {"T": float(S["T"]), "rs": float(S["rs"]), "T_over_rs": float(S["T"]) / float(S["rs"]), "n_sites": int(len(Rs)),
                         "R_equals_fields_R": bool(np.array_equal(Rs, R)), "min": minimum(Rs, f), "sign_changes": sign_changes(f), "roots_linear": roots_linear(Rs, f),
                         "values_around_min": [(float(Rs[j]), float(f[j])) for j in range(max(0, int(np.argmin(f)) - 4), min(len(f), int(np.argmin(f)) + 5))],
                         "site_quintic": spline_min(Rs, f, 50.0, 110.0), "site_cubic": spline_min(Rs, f, 50.0, 110.0, k=3),
                         "N_applied_at_min": float(Na[int(np.argmin(f))]), "v_applied_at_min": float(va[int(np.argmin(f))]),
                         "dtype": str(S["dtype"]) if "dtype" in S.files else None, "kappas": S["kappas"].tolist() if "kappas" in S.files else None}
    # the T0005.00 slice against the fields file at 5.0 (the site fields the slice stores against the record's own)
    S5 = np.load(C.SLICE_5)
    i5 = C.at_clock(recs, 5.0)
    out["slice_5_vs_fields"] = {"T_over_rs": float(S5["T"]) / float(S5["rs"]), "N_applied_vs_fields_N_applied": float(np.max(np.abs(S5["N_applied"] - F["N_applied"][i5]))),
                                "v_applied_vs_fields_v_applied": float(np.max(np.abs(S5["v_applied"] - F["v_applied"][i5]))),
                                "N_applied_vs_fields_N": float(np.max(np.abs(S5["N_applied"] - F["N"][i5]))), "v_applied_vs_fields_v": float(np.max(np.abs(S5["v_applied"] - F["v"][i5]))),
                                "slice_min": minimum(S5["R"], S5["N_applied"] - np.abs(S5["v_applied"])), "fields_applied_min": sel[5.0]["applied_min"], "json_min": sel[5.0]["json_min"]}
    # the same fields-file check at every record: N_applied against N (the record's own fields)
    out["applied_vs_site_all_records"] = {"N_maxabs": max(p["applied_minus_site_maxabs"] for p in per), "v_maxabs": max(p["vapplied_minus_v_maxabs"] for p in per)}
    C.jdump(out, C.OUT + "/fields.json")
    print("fields keys:", out["fields_keys"])
    print("sites:", out["sites"], "nodes:", out["nodes_record0"])
    for t, p in out["at_clocks"].items():
        print("T %.4f json %.6f at %.3f | site %.6f at %.1f | applied %.6f at %.1f | node %.6f at %.1f | site-node %.6f | node-json %.2e" % (
            t, p["json_min"], p["json_r"], p["site_min"], p["site_r"], p["applied_min"], p["applied_r"], p["node_min"], p["node_r"], p["site_minus_node"], p["node_minus_json"]))
    print("offsets:", out["offset_summary"])
    print("spline 11.25:", out["spline_11p25"])
    print("slice 11.5:", {k: v for k, v in out["slice_11p5"].items()})
    print("slice has node fields:", out["slice_11p5_has_node_fields"])
    print("slice 5 vs fields:", out["slice_5_vs_fields"])
    print("applied vs site, all records:", out["applied_vs_site_all_records"])


if __name__ == "__main__":
    main()
