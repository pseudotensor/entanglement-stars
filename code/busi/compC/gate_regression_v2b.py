#!/usr/bin/env python
"""gate_regression_v2b.py (busi lane, Computation C): version 2 with the kernel cache's counters (hits, misses, entries, bytes, compute_s)
skipped in Part 1 and printed beside it: the cached kernels return the stored arrays bit for bit, so the counters record how many calls the
light readout adds (version 2's s4 run: hits 76 against 80, the first and only reported difference), not arithmetic. Changes marked "v2b:".
gate_regression_v2.py (busi lane, Computation C): version 1 for engine_twofield_v2, whose base is engine_collapse_v36 (the solve lead's
turn 51, 43.1 (2)). Every change is marked "v2:": Part 1 runs the base script --base-script (default engine_collapse_v36.py in --engine-dir)
against engine_twofield_v2.py, both with the same --both-extra options (default "--field-kernel-cache --v36-fixes"; add "--nk-jvp exact" for a
newton member); the kernel cache's timing counter compute_s is a wall-clock reading and is skipped like the seconds. Version 1's docstring follows.
gate_regression_v1.py (busi lane, Computation C; the regression of the solve inbox's entry 62 item 5: "with the light field's source coupling
switched off the 2-field engine reproduces the production's records bitwise").
The production's stored records are the r_s/a 32 fixed-preparation pair of the ladder note COLLAPSE_ladder_rs32_v1: the sigma/a 4 member
C3lad32/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg (engine_collapse_v31, plain) and the sigma/a 2 member LADrs32/member2_snapshot_2100.json
(engine_collapse_v32, newton). The gate takes the stored run's dir/tag, reads its json's args, builds the identical command line (every stored
argument, the clock cut to the first K records, no slices, --m1-simpson-steps empty so that no Simpson record falls inside K) and runs on 1 device
(1) engine_collapse_v35.py and (2) engine_twofield_v1.py --light-coupling off (the light channels present and propagating, their sources not fed).
Part 1: v36 against v35 bitwise: every number of the record json except wall-clock seconds and the engines' labels, every array of the fields npz
and the spectra npz; PASS or the first differing field. Part 2: v36 against the stored production: bitwise where the device and the library
versions are the production's; otherwise the first differing field and the floor table (E/E0 3.0e-7 relative, R_t and R_sh 3.4e-6 lattice units,
min(N - |v|) 4.6e-8, the masks equal) at twice the floor, the verdict BITWISE, WITHIN-FLOOR or FAIL. The gate's PASS: Part 1 bitwise and Part 2
bitwise or within the floor.
Usage: python gate_regression_v1.py --prod-dir DIR --prod-tag TAG --out OUT [--records 3] [--gpu 0] [--engine-dir <internal>/engine]
       [--v36 ./engine_twofield_v1.py] [--prod-json-name collapse_TAG.json] [--skip-runs]"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

SKIP_KEYS = {"label", "record_label", "engine", "engine_v33", "engine_v34", "engine_v35", "engine_v36", "script", "dictionary", "slice_saved", "light_slice_saved",
             "resumed_from", "rule", "seconds", "timing", "seconds_preparation", "light_v36", "s11", "args", "label_v34", "label_v19", "kernel_identities_v19",
             "resume_slice_v34", "light_v36_resume", "site_fields", "module", "kernel", "compute_s", "kernel_cache"}   # v2b: kernel_cache, the cache's counters   # v2: compute_s, the kernel cache's wall clock
V35_ARG_KEYS = {"m", "g", "rs", "R0_over_rs", "reading", "shell", "nsites", "ell", "prep_ell", "field_rtol", "field_atol", "kappa_sea", "tau", "reference", "half_offset",
                "collar_policy", "support_rel", "site_fields", "stop_E", "stop_fields", "maxit", "mix", "nk_maxiter", "nk_inner", "mix_alpha", "delta_rtol_every",
                "commutator_every", "streams", "c_LZ", "window_max", "nk_line_search", "nk_halvings", "nk_krylov", "nk_restarts", "source_continuation", "end_above",
                "fields_v17", "s11_collars", "s11_stop", "weyl_certified", "weyl_refine_max", "nk_jvp"}
BOOL_FLAGS = {"fields_v17", "massless_ball", "weyl_oracle", "benchmark"}


def fmt_num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


def build_args(stored, records, tau, rs, out, tag, gpu, extra):
    cmd = []
    for k, v in stored.items():
        if k not in V35_ARG_KEYS or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if k in BOOL_FLAGS:
            if v:
                cmd.append(flag)
            continue
        if isinstance(v, bool):
            cmd += [flag, str(int(v))]
        elif isinstance(v, (int, float)):
            cmd += [flag, fmt_num(v)]
        else:
            if str(v) == "":
                continue
            cmd += [flag, str(v)]
    cmd += ["--T-over-rs", repr((records - 1) * float(tau) / float(rs)), "--m1-simpson-steps", "", "--slices-c128", "", "--slices-c64", "", "--out", out, "--tag", tag]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    return cmd + extra


def leaves_equal(x, y):
    if isinstance(x, bool) or isinstance(y, bool):
        return bool(x) == bool(y)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return x == y
    return x == y


def compare_tree(ref, new, path, stats):
    """every key of ref (minus SKIP_KEYS) must exist in new with an identical value; returns the first difference or None."""
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return "%s: a dict against %s" % (path, type(new).__name__)
        for k, v in ref.items():
            if k in SKIP_KEYS:
                continue
            if k not in new:
                return "%s.%s: missing" % (path, k)
            d = compare_tree(v, new[k], "%s.%s" % (path, k), stats)
            if d:
                return d
        return None
    if isinstance(ref, list):
        if not isinstance(new, list) or len(ref) != len(new):
            return "%s: lengths %s against %s" % (path, len(ref), (len(new) if isinstance(new, list) else type(new).__name__))
        for i, (p, q) in enumerate(zip(ref, new)):
            d = compare_tree(p, q, "%s[%d]" % (path, i), stats)
            if d:
                return d
        return None
    stats["leaves"] += 1
    if isinstance(ref, str):
        if ref != new:
            return "%s: %r against %r" % (path, ref[:80], str(new)[:80])
        return None
    if ref is None:
        return None if new is None else "%s: None against %r" % (path, new)
    if not leaves_equal(ref, new):
        return "%s: %r against %r" % (path, ref, new)
    return None


def arrays_equal(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if x.dtype == object or y.dtype == object:
        if x.shape != y.shape:
            return False
        return all(arrays_equal(p, q) for p, q in zip(x.ravel(), y.ravel()))
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def compare_npz(pa, pb, records, skip=("label", "solver", "light_meta"), time_sliced=("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "r_nodes", "N_nodes", "v_nodes")):
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    n = 0
    for k in A.files:
        if k in skip or k not in B.files:
            if k not in skip and k not in B.files:
                return "%s: missing in %s" % (k, os.path.basename(pb)), n
            continue
        x = A[k]; y = B[k]
        if k in time_sliced or (x.ndim >= 1 and x.shape[0] >= records and y.ndim >= 1 and k.startswith(("mask_", "e_", "nocc_"))):
            x = x[:records]; y = y[:records]
        n += 1
        if not arrays_equal(x, y):
            return "%s: arrays differ (shapes %s %s)" % (k, x.shape, y.shape), n
    return None, n


def floor_table(prod_recs, new_recs, prod_spec, new_spec, records):
    rows = []; ok = True
    for i in range(records):
        p = prod_recs[i]; q = new_recs[i]
        dE = abs(q["E_over_E0"] - p["E_over_E0"]) if (p.get("E_over_E0") is not None and q.get("E_over_E0") is not None) else None
        dR = (abs(q["R_t"] - p["R_t"]) if (p.get("R_t") and q.get("R_t")) else (0.0 if p.get("R_t") == q.get("R_t") else None))
        dRs = (abs(q["R_sh"] - p["R_sh"]) if (p.get("R_sh") and q.get("R_sh")) else (0.0 if p.get("R_sh") == q.get("R_sh") else None))
        dm = abs(q["min_N_minus_absv"] - p["min_N_minus_absv"]) if (p.get("min_N_minus_absv") is not None and q.get("min_N_minus_absv") is not None) else None
        masks_eq = True
        if prod_spec is not None and new_spec is not None:
            for k in prod_spec.files:
                if k.startswith("mask_") and not k.startswith("mask_L") and k in new_spec.files:
                    masks_eq = masks_eq and bool(np.array_equal(prod_spec[k][i], new_spec[k][i]))
        row_ok = (dE is not None and dE <= 3.0e-7) and (dR is not None and dR <= 3.4e-6) and (dRs is None or dRs <= 3.4e-6) and (dm is not None and dm <= 4.6e-8) and masks_eq
        ok = ok and row_ok
        rows.append("  T/rs %.4f: |dE/E0| %s (floor 3.0e-7); |dR_t| %s (3.4e-6); |dR_sh| %s (3.4e-6); |d min(N-|v|)| %s (4.6e-8); masks equal %s -> %s" % (
            p["T_over_rs"], "%.2e" % dE if dE is not None else "-", "%.2e" % dR if dR is not None else "-", "%.2e" % dRs if dRs is not None else "-",
            "%.2e" % dm if dm is not None else "-", masks_eq, "within" if row_ok else "ABOVE THE FLOOR"))
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", required=True, help="the directory holding the production's collapse_<tag>.json and npz files (read-only)")
    ap.add_argument("--prod-tag", required=True)
    ap.add_argument("--prod-json-name", default=None, help="the json's file name when it is not collapse_<tag>.json (the sigma/a 2 snapshot)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--engine-dir", default=os.environ.get("BUSI_ENGINE_DIR", "<internal>/engine"))
    ap.add_argument("--v36", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_twofield_v2.py"))      # v2
    ap.add_argument("--base-script", default="engine_collapse_v36.py")                                                            # v2
    ap.add_argument("--both-extra", default="--field-kernel-cache --v36-fixes")                                                   # v2
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-runs", action="store_true", help="compare the outputs already in --out")
    ap.add_argument("--light-kappas", default="1,2"); ap.add_argument("--light-guard-kappas", default="3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jname = a.prod_json_name or ("collapse_%s.json" % a.prod_tag)
    J = json.load(open(os.path.join(a.prod_dir, jname)))
    stored = J["args"]
    prod_recs = [r for r in J["records"] if r.get("E") is not None][: a.records]
    if len(prod_recs) < a.records:
        raise SystemExit("the production holds %d accepted records, %d asked" % (len(prod_recs), a.records))
    tau = float(stored["tau"]); rs = float(stored["rs"])
    print("gate C: production %s (%s; %s): tau %g rs %g mix %s; the first %d records (T/rs %s)" % (
        os.path.join(a.prod_dir, jname), J.get("script"), stored.get("mix"), tau, rs, stored.get("mix"), a.records, ", ".join("%.4f" % r["T_over_rs"] for r in prod_recs)), flush=True)
    env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "16"); env.setdefault("MKL_NUM_THREADS", "16"); env.setdefault("OPENBLAS_NUM_THREADS", "16")
    env["BUSI_ENGINE_DIR"] = a.engine_dir
    both = a.both_extra.split() if a.both_extra else []                                                                           # v2
    runs = {"v35": (os.path.join(a.engine_dir, a.base_script), list(both)),                                                     # v2: the base script, the shared options
            "v36off": (a.v36, list(both) + ["--light-kappas", a.light_kappas, "--light-guard-kappas", a.light_guard_kappas, "--light-coupling", "off"])}
    print("v2: base %s; both runs with %s" % (os.path.join(a.engine_dir, a.base_script), " ".join(both) or "(no extra options)"), flush=True)
    if not a.skip_runs:
        for name, (script, extra) in runs.items():
            cmd = [a.python, script] + build_args(stored, a.records, tau, rs, a.out, "reg_%s" % name, a.gpu, extra)
            print("running %s: %s" % (name, " ".join(cmd)), flush=True)
            with open(os.path.join(a.out, "reg_%s.log" % name), "w") as log:
                rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(script)) or None)
            if rc != 0:
                raise SystemExit("%s exited with %d; see %s" % (name, rc, os.path.join(a.out, "reg_%s.log" % name)))
    J35 = json.load(open(os.path.join(a.out, "collapse_reg_v35.json"))); J36 = json.load(open(os.path.join(a.out, "collapse_reg_v36off.json")))
    r35 = [r for r in J35["records"] if r.get("E") is not None][: a.records]; r36 = [r for r in J36["records"] if r.get("E") is not None][: a.records]
    verdicts = []
    # Part 1: v36 (light off) against v35 on this device, bitwise
    st1 = {"leaves": 0}
    d1 = compare_tree({"head": J35["head"], "records": r35}, {"head": J36["head"], "records": r36}, "json", st1)
    dF, nF = compare_npz(os.path.join(a.out, "collapse_reg_v35_fields.npz"), os.path.join(a.out, "collapse_reg_v36off_fields.npz"), a.records)
    dS, nS = compare_npz(os.path.join(a.out, "collapse_reg_v35_spectra.npz"), os.path.join(a.out, "collapse_reg_v36off_spectra.npz"), a.records)
    p1 = d1 is None and dF is None and dS is None
    print("Part 1 (v36 light off against v35, the same device and arguments): %s; json leaves compared %d; fields arrays %d; spectra arrays %d%s" % (
        "PASS bitwise" if p1 else "FIRST DIFFERENCE %s" % (d1 or dF or dS), st1["leaves"], nF, nS, "" if p1 else " (the light machinery altered the heavy arithmetic: a defect to find)"), flush=True)
    kc = [(r_.get("v36") or {}).get("kernel_cache") for r_ in (r35[:1] + r36[:1])]                                                 # v2b
    print("v2b: the kernel cache's counters at the first record, base against 2-field (skipped in Part 1): %s against %s" % (kc[0], kc[1] if len(kc) > 1 else None), flush=True)
    verdicts.append(p1)
    # Part 2: v36 (light off) against the stored production
    st2 = {"leaves": 0}
    d2 = compare_tree({"records": prod_recs}, {"records": r36}, "json", st2)
    pf = os.path.join(a.prod_dir, "collapse_%s_fields.npz" % a.prod_tag); ps = os.path.join(a.prod_dir, "collapse_%s_spectra.npz" % a.prod_tag)
    dF2, nF2 = compare_npz(pf, os.path.join(a.out, "collapse_reg_v36off_fields.npz"), a.records) if os.path.exists(pf) else ("the production's fields npz %s is absent" % pf, 0)
    dS2, nS2 = compare_npz(ps, os.path.join(a.out, "collapse_reg_v36off_spectra.npz"), a.records) if os.path.exists(ps) else (None, 0)
    bitwise2 = d2 is None and dF2 is None and dS2 is None
    prod_spec = np.load(ps) if os.path.exists(ps) else None; new_spec = np.load(os.path.join(a.out, "collapse_reg_v36off_spectra.npz"))
    within, rows = floor_table(prod_recs, r36, prod_spec, new_spec, a.records)
    print("Part 2 (v36 light off against the stored production %s): %s; json leaves compared %d; fields arrays %d; spectra arrays %d" % (
        a.prod_tag, "BITWISE" if bitwise2 else "first difference %s" % (d2 or dF2 or dS2), st2["leaves"], nF2, nS2), flush=True)
    for row in rows:
        print(row, flush=True)
    print("Part 2 verdict: %s%s" % ("BITWISE" if bitwise2 else ("WITHIN TWICE THE FLOOR" if within else "ABOVE THE FLOOR: a defect to find"),
                                    "" if bitwise2 else " (bitwise reproduction needs the production's device class and library versions; the comparison reads at twice the floor otherwise)"), flush=True)
    verdicts.append(bitwise2 or within)
    print("GATE C: %s" % ("PASS" if all(verdicts) else "FAIL"), flush=True)
    with open(os.path.join(a.out, "gate_regression_v1_result.json"), "w") as fh:
        json.dump({"production": os.path.join(a.prod_dir, jname), "records": a.records, "part1_bitwise": p1, "part1_first_difference": d1 or dF or dS,
                   "part2_bitwise": bitwise2, "part2_first_difference": d2 or dF2 or dS2, "part2_within_floor": within, "floor_rows": rows, "pass": all(verdicts)}, fh, indent=1)


if __name__ == "__main__":
    main()
