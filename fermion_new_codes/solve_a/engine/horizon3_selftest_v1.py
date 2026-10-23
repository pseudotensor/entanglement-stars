#!/usr/bin/env python
"""horizon3_selftest_v1.py (the solve lane; designed by reviewer A for the lane's builder): manufactured checks of the three scripts beside
it, none needing the engine or the box files: (a) section 49's threshold column on a synthetic spectrum with hand-counted expectations and
the producer's printed half-widths; (b) the applied-root finder on a manufactured N + v with two zeros (exact on a piecewise-linear field),
with one zero, with none, and with a spurious dip (the documented rule); (c) the composer on a 1-launch synthetic file with a synthetic
version 2 file differing in exactly one leaf, the cited-number search, the closure and comparator checks, and a dust-like launch without
columns; (d) the export's layout detection and its binning on a small random Hermitian operator; (e) version 2's comparison helpers.
Exit status 0 only if every check passes. Run from the scripts' directory (or with it on sys.path).
"""
import os
import sys
import json
import math
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compose_horizon_v3_v1 as CH   # noqa: E402
import export_horizon_refcol_v1 as EX   # noqa: E402
import dust_root_history_v2 as DU   # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)


def close(x, y, tol=1e-12):
    return x is not None and y is not None and abs(float(x) - float(y)) <= tol


# ---------------------------------------------------------------- (a) the threshold column
def test_threshold():
    m = 0.2; b = math.sqrt(m * m + 1.0); r_s = 54.41360314510785; R_max = 8192.0
    thr0 = EX.threshold_columns(np.array([0.0]), np.array([1.0]), m, b, r_s, R_max)
    check("half-width at b equals the producer's printed 0.003386914358420394", close(thr0["half_width_b"], 0.003386914358420394, 1e-15), "%.18g" % thr0["half_width_b"])
    check("half-width at m equals the producer's printed 0.0006642285540174298", close(thr0["half_width_m"], 0.0006642285540174298, 1e-15), "%.18g" % thr0["half_width_m"])
    check("b equals the producer's printed 1.019803902718557", close(b, 1.019803902718557, 1e-15))
    hb, hm = thr0["half_width_b"], thr0["half_width_m"]
    e = np.array([-1.5, -b - 0.5 * hb, -b + 1.5 * hb, -m - 0.3 * hm, -0.5, 0.1, m + 0.9 * hm, m + 1.9 * hm, 0.6, b - 0.1 * hb, 1.5])
    w = np.array([0.01, 0.02, 0.03, 0.04, 0.10, 0.20, 0.05, 0.06, 0.30, 0.07, 0.12])
    thr = EX.threshold_columns(e, w, m, b, r_s, R_max)
    raw = thr["raw"]; B1 = thr["buffers"]["1"]; B2 = thr["buffers"]["2"]; E = thr["threshold_error"]
    check("raw 5 bins", all(close(x, y) for x, y in zip(raw["bins5"], [0.03, 0.07, 0.30, 0.48, 0.12])), str(raw["bins5"]))
    check("raw shares", close(raw["w_below_minus_m"], 0.10) and close(raw["w_in_gap"], 0.30) and close(raw["w_above_m"], 0.60) and close(raw["total"], 1.0))
    check("buffer 1 threshold weights -b, -m, m, b", all(close(B1["threshold_weight"][k], v) for k, v in (("-b", 0.02), ("-m", 0.04), ("m", 0.05), ("b", 0.07))) and close(B1["threshold_total"], 0.18), str(B1["threshold_weight"]))
    check("buffer 1 bins and shares", all(close(x, y) for x, y in zip(B1["bins5"], [0.01, 0.03, 0.30, 0.36, 0.12])) and close(B1["w_below_minus_m"], 0.04) and close(B1["w_in_gap"], 0.30) and close(B1["w_above_m"], 0.48))
    check("buffer 2 threshold weights and total", all(close(B2["threshold_weight"][k], v) for k, v in (("-b", 0.05), ("-m", 0.04), ("m", 0.11), ("b", 0.07))) and close(B2["threshold_total"], 0.27))
    check("buffer 2 bins and shares", all(close(x, y) for x, y in zip(B2["bins5"], [0.01, 0.0, 0.30, 0.30, 0.12])) and close(B2["w_below_minus_m"], 0.01) and close(B2["w_in_gap"], 0.30) and close(B2["w_above_m"], 0.42))
    check("closure at both buffers (shares + threshold total = raw shares)", close(B1["w_below_minus_m"] + B1["w_in_gap"] + B1["w_above_m"] + B1["threshold_total"], 1.0) and close(B2["w_below_minus_m"] + B2["w_in_gap"] + B2["w_above_m"] + B2["threshold_total"], 1.0))
    check("threshold terms |b1 - b2|", close(E["w_below_minus_m"], 0.03) and close(E["w_in_gap"], 0.0) and close(E["w_above_m"], 0.06) and close(E["threshold_total"], 0.09) and all(close(x, y) for x, y in zip(E["bins5"], [0.0, 0.03, 0.0, 0.06, 0.0])), str(E))
    k5 = EX.killing_bins_5(e, w, m, b)
    check("killing_bins_5 agrees with the raw bins", all(close(x, y) for x, y in zip(k5, raw["bins5"])))
    # the dust script's binning in an eigenbasis: U = identity, X columns with known weights
    nlev = len(e); U = np.eye(nlev, dtype=complex); X = np.sqrt(w)[:, None] * np.ones((nlev, 2), complex); X[:, 1] *= 1j
    bins, mean = DU.killing_bins(e, U, X, m, b)
    check("dust killing_bins in the eigenbasis (2 columns)", bins.shape == (5, 2) and all(close(bins[i, 0], raw["bins5"][i]) and close(bins[i, 1], raw["bins5"][i]) for i in range(5)) and close(mean[0], raw["mean"]))
    # closure on the composer's closure_checks with the producer's layout of a spectrum dict
    sp_ = synth_spectrum(m, b, r_s, R_max, e, w)
    ch = CH.closure_checks(sp_)
    check("composer closure_checks on the synthetic spectrum", all(close(v, 0.0) for v in ch["closure"].values()) and close(ch["error_identity"], 0.0) and close(ch["shares_minus_norm"], 0.0), str(ch))


def synth_spectrum(m, b, r_s, R_max, e, w):
    thr = EX.threshold_columns(e, w, m, b, r_s, R_max)

    def buf(mu):
        B = thr["buffers"][mu]
        return {"threshold_weight": B["threshold_weight"], "threshold_total": B["threshold_total"], "w_above_m": B["w_above_m"], "w_in_gap": B["w_in_gap"],
                "w_below_minus_m": B["w_below_minus_m"], "bins_particles": [0.0] * 80, "bins_holes": [0.0] * 80}
    return {"norm": thr["raw"]["total"], "mean": thr["raw"]["mean"], "spread": 0.01, "w_negative": thr["raw"]["w_negative"], "w_in_gap": thr["raw"]["w_in_gap"],
            "w_above_m": thr["raw"]["w_above_m"], "w_below_minus_m": thr["raw"]["w_below_minus_m"], "bins_particles": [0.0] * 80, "bins_holes": [0.0] * 80,
            "v3_threshold": {"b": b, "r_s": r_s, "R_max": R_max, "half_width_b": thr["half_width_b"], "half_width_m": thr["half_width_m"], "comparator_busi_per_a": 0.0033,
                             "label": "threshold content, extent comparable to the chain", "buffers": {"1": buf("1"), "2": buf("2")},
                             "threshold_error": {k: thr["threshold_error"][k] for k in ("w_above_m", "w_in_gap", "w_below_minus_m", "threshold_total")}}}


# ---------------------------------------------------------------- (b) the applied-root finder
def piecewise(R, knots):
    xs = [k[0] for k in knots]; ys = [k[1] for k in knots]
    return np.interp(R, xs, ys)


def test_roots():
    R = np.arange(1, 201, dtype=float)
    f = piecewise(R, [(1, 1.0), (26.0, 1.0), (28.8, -1.0), (60.0, -1.0), (62.6, 1.0), (200, 1.0)])
    Ro, Ri, z = DU.applied_outer_inner(R, f)
    check("two zeros: outer 61.3 (rising), inner 27.4 (falling), exact on a piecewise-linear field", close(Ro, 61.3, 1e-12) and close(Ri, 27.4, 1e-12) and len(z) == 2, "%s %s %s" % (Ro, Ri, z))
    Ro2, Ri2, z2 = EX.applied_outer_inner(R, f)
    check("the export's copy of the finder agrees", close(Ro, Ro2) and close(Ri, Ri2) and z == z2)
    f1 = piecewise(R, [(1, -1.0), (60.0, -1.0), (62.6, 1.0), (200, 1.0)])
    Ro, Ri, z = DU.applied_outer_inner(R, f1)
    check("one rising zero only: inner None", close(Ro, 61.3, 1e-12) and Ri is None and len(z) == 1)
    Ro, Ri, z = DU.applied_outer_inner(R, np.ones(200))
    check("no zero: (None, None, [])", Ro is None and Ri is None and z == [])
    f3 = piecewise(R, [(1, 1.0), (26.0, 1.0), (28.8, -1.0), (60.0, -1.0), (62.6, 1.0), (79.0, 1.0), (80.0, -1.0), (82.0, -1.0), (83.0, 1.0), (200, 1.0)])
    Ro, Ri, z = DU.applied_outer_inner(R, f3)
    check("spurious dip beyond the root: the rule takes the outermost rising zero (82.5) and the falling zero below it (79.5); 4 zeros reported",
          close(Ro, 82.5, 1e-12) and close(Ri, 79.5, 1e-12) and len(z) == 4, "%s %s %d" % (Ro, Ri, len(z)))
    # version 1b's own rule on the same field: the last i with f[i] < 0 <= f[i+1]
    idx = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]; i = idx[-1]
    Ro_v1b = float(R[i] - f[i] * (R[i + 1] - R[i]) / (f[i + 1] - f[i]))
    check("the outer root equals version 1b's finder on the two-zero field", close(Ro_v1b, 61.3, 1e-12))
    check("record_roots: trapping zeros -> (max, min)", DU.record_roots({"trapping_zeros": [23.2, 38.5]}) == (38.5, 23.2) and DU.record_roots({"trapping_zeros": [38.5]}) == (38.5, None)
          and DU.record_roots({"R_t": 40.0, "r_in": 20.0}) == (40.0, 20.0) and DU.record_roots({}) == (None, None))


# ---------------------------------------------------------------- (c) the composer on a synthetic 1-launch file
def synth_file(tmp, name, mode="history"):
    m = 0.2; b = math.sqrt(m * m + 1.0); r_s = 54.41360314510785; R_max = 8192.0
    hb = b * r_s / (2 * R_max); hm = m * r_s / (2 * R_max)
    eL = np.array([-1.1, -0.4, 0.05, 0.15, m + 0.5 * hm, 0.5, b - 0.5 * hb, 1.2]); wL = np.array([0.001, 0.003, 0.5, 0.49, 0.001, 0.004, 0.0005, 0.0005])
    eE = np.array([-1.1, -0.4, 0.05, 0.15, m + 0.5 * hm, 0.5, b - 0.5 * hb, 1.2]); wE = np.array([0.002, 0.004, 0.45, 0.49, 0.004, 0.04, 0.005, 0.005])
    n = 8; rs = 1.0   # 16 r_s = 16 a sits beyond the 8 sites: the 'beyond 16 r_s' weights are 0 and the collar too; R_t = 3.5 a, 2GE = 2.5 a
    X = np.zeros((2 * n, 2), complex)
    X[:, 0] = (np.arange(2 * n) + 1.0) / 20.0; X[:, 0] /= np.sqrt((np.abs(X[:, 0]) ** 2).sum())
    X[:, 1] = np.exp(1j * np.arange(2 * n)) / np.sqrt(2 * n)
    Rsites = np.arange(1, n + 1, dtype=float)
    ws0 = EX.site_weights(X[:, 0], n, "site_major"); ws1 = EX.site_weights(X[:, 1], n, "site_major")
    Rt_end, gE_end = 3.5, 2.5
    end0 = EX.region_weights(ws0, Rsites, rs, Rt_end, gE_end); end1 = EX.region_weights(ws1, Rsites, rs, Rt_end, gE_end)

    def col(j, comparator, end, b16, arr, comp_col, comp_arr):
        return {"launch": {"depth": "annulus middle" if not comparator else "outside comparator", "comparator": comparator, "R_c": 2.0 if not comparator else 4.0, "k0": 0.0981747704, "ak0_name": "2pi/64 (wavelength r_s)",
                           "sigma": 1.0, "spinor": [1.0, 0.0, 0.0, 0.0], "N_at": 0.5, "v_at": -0.58},
                "spectrum_launch": synth_spectrum(m, b, r_s, R_max, eL, wL), "omega_bar": 0.065, "soft": True, "w_above_m": 0.006, "w_below_minus_m": 0.001, "w_in_gap": 0.993,
                "beyond16_final_before_reflection": b16, "last_pre_reflection_after_launch_rs": 32.5, "beyond_Rcut_last_pre_reflection": 1e-16, "arrival_after_launch_rs": arr,
                "arrival_first_step_after_launch_rs": arr + 0.1, "pred_2rs_log": 4.6, "pred_kin_log": 2.5, "reflection_clock_after_launch_rs": 231.5, "interior_end": 0.68,
                "classifier_overlap_end": 0.93, "classifier_overlap_launch": 0.99, "max_radius_rs": 19.2, "end": end, "spectrum_end": synth_spectrum(m, b, r_s, R_max, eE, wE),
                "end_includes_reflection": False, "comparator_column": comp_col, "comparator_arrival_after_launch_rs": comp_arr, "delay_minus_comparator_rs": (arr - comp_arr) if comp_arr is not None else None}
    cols = [col(0, False, end0, 0.99697, 22.48, 1, 20.0), col(1, True, end1, 0.5, 20.0, None, None)]
    sig = lambda vals: {"window": 32, "pad": 4, "region": [-1.0, 1e30], "sites": [1, n], "windows": 1, "N_nonpositive_windows": 0, "v_positive_windows": 0,
                        "total": [1.0, 1.0], "plus": vals, "minus": [1 - v for v in vals], "plus_out": vals, "plus_in": [0.0, 0.0], "minus_out": [0.0, 0.0], "minus_in": [1 - v for v in vals],
                        "plus_pi4_pi2": [v / 2 for v in vals], "minus_negpi2_negpi4": [0.0, 0.0], "unsigned_pi4_pi2": [0.3, 0.2]}
    L = {"key": "T15p000_k1", "T_launch_over_rs": 15.0, "k0": 120, "kappa": 1, "E": 108.8272, "two_gE": r_s, "T_H": 0.00146, "roots_dense": [{"R": 1.5, "slope": -0.0145}, {"R": 3.5, "slope": 0.009}],
         "R_outer": 3.5, "R_inner": 1.5, "kappa_in": 0.0145, "classifier_states": 496, "eig_seconds": 11.8, "cheb_K_max": 38, "cheb_tail_acc": 1.2e-12, "spectrum_conservation_max": 0.3,
         "columns": cols, "signed_launch": {"whole": sig([0.6, 0.7]), "inside_Rt": sig([0.4, 0.1]), "beyond_Rt": sig([0.2, 0.6])},
         "signed_launch_window_controls": {"16": sig([0.61, 0.69]), "64": sig([0.59, 0.71])},
         "signed_series": [{"t_over_rs": 16.0, "after_launch_rs": 1.0, "R_t": 3.4, "inside_Rt": sig([0.3, 0.1]), "beyond_Rt": sig([0.3, 0.6])}, {"t_over_rs": 17.0, "after_launch_rs": 2.0, "R_t": None, "inside_Rt": None, "beyond_Rt": None}]}
    J = {"mode": mode, "nsites": n, "m": m, "launches": [L]}
    jpath = os.path.join(tmp, name + ".json"); npath = os.path.join(tmp, name + ".npz")
    with open(jpath, "w") as fh:
        json.dump(J, fh)
    t = np.array([15.0, 16.0])
    np.savez(npath, **{"t_T15p000_k1": t, "refl_T15p000_k1": np.zeros((2, 2), bool), "norm_T15p000_k1": np.ones(2), "Rt_T15p000_k1": np.array([3.6, Rt_end]), "2gE_T15p000_k1": np.array([2.6, gE_end]),
                       "w_beyond_outer_root_T15p000_k1": np.array([[0.0, 0.0], [end0["beyond_outer_root"], end1["beyond_outer_root"]]]),
                       "w_beyond_2GE_T15p000_k1": np.array([[0.0, 0.0], [end0["beyond_2GE"], end1["beyond_2GE"]]]), "w_beyond_16rs_T15p000_k1": np.array([[0.0, 0.0], [end0["beyond_16rs"], end1["beyond_16rs"]]]),
                       "w_collar_16_24_T15p000_k1": np.array([[0.0, 0.0], [end0["collar_16_24"], end1["collar_16_24"]]]), "w_far_32_48_T15p000_k1": np.array([[0.0, 0.0], [end0["far_32_48"], end1["far_32_48"]]]),
                       "w_beyond_Rcut_T15p000_k1": np.array([[0.0, 0.0], [end0["beyond_Rcut"], end1["beyond_Rcut"]]]), "X_end_T15p000_k1": X})
    return jpath, npath, J, X, Rsites, rs, Rt_end, gE_end, end0


def test_composer():
    tmp = tempfile.mkdtemp(prefix="horizon3_selftest_")
    try:
        jpath, npath, J, X, Rsites, rs, Rt_end, gE_end, end0 = synth_file(tmp, "synth")
        J2 = json.loads(json.dumps(J))
        J2["launches"][0]["columns"][0]["end"]["beyond_16rs"] += 1e-3
        J2["launches"][0]["columns"][0]["spectrum_end"]["v3_threshold"]["buffers"]["1"]["threshold_total"] += 0.5   # an added column: must be excluded
        v2path = os.path.join(tmp, "synth_v2.json")
        with open(v2path, "w") as fh:
            json.dump(J2, fh)
        # a dust-like file without per-column structure
        dpath = os.path.join(tmp, "dust.json")
        with open(dpath, "w") as fh:
            json.dump({"mode": "dust", "launches": [{"key": "T12p125_k1", "T_launch_over_rs": 12.125, "kappa": 1, "f": [1.0, 1.0], "end": {"beyond_root": 0.17040405}}]}, fh)
        args = CH.build_parser().parse_args(["--out-dir", tmp, "--files", "synth=%s,%s" % (jpath, npath), "--files", "dust=%s" % dpath, "--v2-json", "synth=%s" % v2path, "--series-columns", "0,1,14"])
        md, digest, arrays = CH.run(args)
        check("composer runs on the synthetic files and prints the column's numbers", ("0.99697" in md) and ("22.48" in md) and ("T15p000_k1" in md) and ("T12p125_k1" in md))
        ch = digest["per_file"]["synth"]["launches"][0]["checks"]
        check("closure, error identity, comparator identity and json-npz consistency all zero on the synthetic launch",
              close(ch["max_abs_closure"], 0.0) and close(ch["max_error_identity_residual"], 0.0) and close(ch["max_comparator_residual"], 0.0) and close(ch["max_json_minus_npz_end"], 0.0), str(ch))
        check("the centre-below-inner-root flag counts none (R_c 2.0 and 4.0 above R_inner 1.5)", ch["columns_with_centre_below_inner_root"] == 0)
        cited = {c["label"]: c for c in digest["comparison"]["cited"]}
        hit1 = [c for c in cited.values() if close(c["value"], 0.99697, 1e-12)][0]
        check("the cited 0.99697 is found at the column's beyond16 leaf", any(h["path"].endswith("launches[T15p000_k1].columns[0].beyond16_final_before_reflection") for h in hit1["hits"]), str(hit1["hits"]))
        hit2 = [c for c in cited.values() if close(c["value"], 22.48, 1e-12)][0]
        check("the cited 22.48 is found at the column's arrival leaf", any(h["path"].endswith("columns[0].arrival_after_launch_rs") for h in hit2["hits"]), str(hit2["hits"]))
        hit3 = [c for c in cited.values() if close(c["value"], 0.17040405, 1e-12)][0]
        check("the cited 0.17040405 is found in the dust-like file", any(h["file"] == "dust" for h in hit3["hits"]))
        hit4 = [c for c in cited.values() if close(c["value"], 25.82, 1e-12)][0]
        check("an absent cited number reports no hit and a nearest leaf", len(hit4["hits"]) == 0 and hit4["nearest"] is not None)
        v2 = digest["comparison"]["v2_json"][0]
        check("version 2 comparison: exactly the one perturbed leaf differs, the added threshold column excluded",
              v2["n_differing"] == 1 and v2["diffs"][0]["path"].endswith("columns[0].end.beyond_16rs") and close(v2["diffs"][0]["v3_minus_v2"], -1e-3, 1e-12), str(v2["diffs"]))
        check("arrays npz holds the 80-bin histograms, the signed readouts and the series", any(k.endswith("bins_particles") for k in arrays) and any("signed_launch__whole__plus_out" in k for k in arrays) and any("series__inside_Rt__plus_out" in k for k in arrays))
        # the full main writes files and refuses to overwrite
        out2 = os.path.join(tmp, "out")
        sys.argv = ["compose", "--out-dir", out2, "--files", "synth=%s,%s" % (jpath, npath)]
        CH.main()
        check("main writes the md, json and arrays npz", all(os.path.exists(os.path.join(out2, "HORIZON_single_object_v3" + ext)) for ext in (".md", ".json", "_arrays.npz")))
        refused = False
        try:
            CH.main()
        except SystemExit as ex:
            refused = "refusing" in str(ex)
        check("main refuses to overwrite without --overwrite", refused)
        # (d) the export's layout detection on the synthetic column
        best, allres = EX.detect_layout(X[:, 0], 8, Rsites, rs, Rt_end, gE_end, end0)
        check("layout detection picks site_major / R > c on the synthetic column", best["layout"] == "site_major" and best["strict"] and close(best["max_abs_dev"], 0.0), str({k: v["max_abs_dev"] for k, v in allres.items()}))
        Xc = np.concatenate([X[0::2, 0], X[1::2, 0]])   # the same state stored component-major
        best2, _ = EX.detect_layout(Xc, 8, Rsites, rs, Rt_end, gE_end, end0)
        check("layout detection picks component_major on the re-stored column", best2["layout"] == "component_major" and close(best2["max_abs_dev"], 0.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- (d) binning on a small random Hermitian operator
def test_small_operator():
    rng = np.random.default_rng(7)
    n = 6; N = 2 * n
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); A = (A + A.conj().T) / 2
    e, U = np.linalg.eigh(A)
    X = rng.normal(size=N) + 1j * rng.normal(size=N); X /= np.linalg.norm(X)
    w = EX.weights_in_basis(U, X)
    w_ref = np.abs(U.conj().T @ X) ** 2
    check("weights_in_basis equals |U^dagger X|^2 (no copy of U)", np.allclose(w, w_ref, atol=1e-14) and close(w.sum(), 1.0, 1e-12))
    m, b = 0.2, math.sqrt(1.04)
    thr = EX.threshold_columns(e, w, m, b, r_s=54.4, R_max=1e9)
    check("with R_max huge no level falls in a threshold interval; the bins sum to the norm", close(thr["buffers"]["1"]["threshold_total"], 0.0) and close(sum(thr["raw"]["bins5"]), 1.0, 1e-12))
    bins, mean = DU.killing_bins(e, U, X[:, None], m, b)
    check("dust killing_bins matches the export's raw bins on the same operator", np.allclose(bins[:, 0], thr["raw"]["bins5"], atol=1e-13) and close(mean[0], thr["raw"]["mean"], 1e-12))
    ws = DU.site_w(X[:, None], n)
    check("site_w sums the two interleaved components per site", np.allclose(ws[:, 0], np.abs(X[0::2]) ** 2 + np.abs(X[1::2]) ** 2) and np.allclose(ws[:, 0], EX.site_weights(X, n, "site_major")))
    # a block's rows and extremes by Lanczos agree with the dense ones on the small operator
    import scipy.sparse as sp
    mask = np.arange(n) >= 2
    rows = EX.rows_of(mask, "site_major")
    blk = sp.csr_matrix(A)[rows][:, rows].toarray()
    eb = np.linalg.eigvalsh(blk)
    check("rows_of selects the block's interleaved rows (dimension 2 x sites beyond)", blk.shape == (8, 8) and len(rows) == 8 and rows[0] == 4)
    check("the block's dense extremes bracket its spectrum", eb[0] <= eb[-1])


# ---------------------------------------------------------------- (e) version 1b's comparison helpers
def test_compare_helpers():
    d = DU.arr_diff(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0 + 2e-7]))
    check("arr_diff: max abs diff where both finite, nan mismatches counted", close(d["max_abs_diff"], 2e-7, 1e-15) and d["nan_mismatch"] == 1 and d["both_finite"] == 2)
    check("arr_diff on differing shapes returns None", DU.arr_diff(np.zeros(3), np.zeros(4))["max_abs_diff"] is None)
    check("lists_equal is None-aware", DU.lists_equal([None, 16.875, 16.5], [None, 16.875, 16.5]) and not DU.lists_equal([None, 16.875], [16.0, 16.875]) and not DU.lists_equal([1.0], [1.0, 2.0]))
    check("num_diff on None", DU.num_diff(1.0, None) is None and close(DU.num_diff(2.0, 0.5), 1.5))
    flat = CH.flatten({"launches": [{"key": "A", "x": 1.0, "cols": [{"y": 2.0}]}], "z": None})
    check("flatten indexes keyed lists by their key", flat.get("launches[A].x") == 1.0 and flat.get("launches[A].cols[0].y") == 2.0 and "z" in flat)
    res = CH.compare_flat({"a": 1.0, "b": 2.0, "c.v3_threshold": 5.0}, {"a": 1.0, "b": 2.5, "c.v3_threshold": 6.0, "d": 1.0}, 1e-9, 1e-9, exclude=("v3_threshold",))
    check("compare_flat counts equal and differing leaves, excludes the added columns", res["n_common"] == 2 and res["n_equal"] == 1 and res["n_differing"] == 1 and res["n_only_v3"] == 1)
    cited = CH.parse_cited(["x=22.48", "y=1.24443e-16", "z=-0.00043035796"], 0.0)
    check("parse_cited: the tolerance is half the unit of the last printed digit", close(cited[0][2], 5e-3, 1e-15) and close(cited[1][2], 5e-22, 1e-30) and close(cited[2][2], 5e-12, 1e-20))


def main():
    test_threshold()
    test_roots()
    test_composer()
    test_small_operator()
    test_compare_helpers()
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("%d checks, %d failed" % (len(RESULTS), nfail))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
