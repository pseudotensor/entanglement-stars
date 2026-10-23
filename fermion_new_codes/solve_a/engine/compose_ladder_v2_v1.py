#!/usr/bin/env python
"""compose_ladder_v2_v1.py (the solve lane's author; engine spec 55.1, the paragraph "COLLAPSE_ladder version 2 when the rungs land";
the plan solve/specs/COLLAPSE_ladder_v2_plan.md). Computes every table of the plan on the box CPU from the jsons of the busi builder's
manifest (their paths are the defaults) and writes them, beside the raw outputs of the two tools it reuses, to one directory
(results/engine/LADV2 by default):
  (1) engine_ladder_compare_v5.py on the three single-version rungs (sigma/a 2; sigma/a 4 and 8 as their _to40 jsons, the histories to
      40): its coverage, formation, matched-time, reading-rule and criterion tables as it prints them;
  (2) compare_member_floor_v2.py at twice the floor on five pairs, a sixth when the v32 member's json is present: the rung 4 and rung 8
      continuations against their first runs (the resumes at 18), the two bounce replays against the first runs (the half-clock states),
      rung 2 against the composite history (the comparator), the composite against the member (version 1.7's seam table recomputed);
  (3) new code: the per-member table with the engine and solver as the records state them, the stop-rule counts and residual bands, the
      resume entries and the slices the records mark saved; the episodes under one rule; the resume, bounce and comparator readings of
      the tool's rows; the record-step table carried from version 1.7's printed tables with its derived columns marked derived;
      hreduce's conditional bound as a conditions table (no numerical bound is computed; none can be from the inputs); the inventory of
      the complex128 slices on the box (names, clocks, sizes; no slice is read); the three rungs' records at the slice clocks.
Every number in the outputs is printed by this script or by the two tools, or carried from version 1.7 and marked so, or marked derived.
No date is written; elapsed seconds are. --self-test runs the whole pipeline on manufactured records in a temporary directory and exits
0 only when every check passes. --check-inputs lists the default paths with their presence and size."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import engine_ladder_compare_v5 as LC  # noqa: E402  (the lane's tool: rec_residual, confirm_residual, pockets)

CMF = os.path.join(HERE, "compare_member_floor_v2.py")
CV5 = os.path.join(HERE, "engine_ladder_compare_v5.py")
try:
    import compare_member_floor_v2 as _CF  # noqa: E402  (the tool's constants; it needs numpy)
    FLOOR, TWICE = dict(_CF.FLOOR), dict(_CF.TWICE)
except Exception:
    FLOOR = {"E_over_E0": 1.5e-07, "roots": 1.7e-06, "min_N_minus_absv": 2.3e-08}
    TWICE = {k: 2.0 * v for k, v in FLOOR.items()}

BULK = "<internal>/solve_a_bulk"
SOLVE = os.path.normpath(os.path.join(HERE, "..", ".."))
ROLES = ("rung2", "rung4_first", "rung4", "rung8_first", "rung8", "composite", "bounce4", "bounce8")
HISTORY_ROLES = ("rung2", "rung4", "rung8")
SIGMA = {"rung2": 2, "rung4": 4, "rung8": 8, "rung4_first": 4, "rung8_first": 8, "bounce4": 4, "bounce8": 8}
LABEL = {
    "rung2": "sigma/a 2 (rung 2; one run)",
    "rung4_first": "sigma/a 4, the first run to 18.125",
    "rung4": "sigma/a 4 (rung 4; the first run, then its continuation from the T0018.00 slice: the history to 40)",
    "rung8_first": "sigma/a 8, the first run to 18.125",
    "rung8": "sigma/a 8 (rung 8; the first run, then its continuation from the T0018.00 slice: the history to 40)",
    "composite": "the composite history (comparator, not a member)",
    "bounce4": "rung 4 bounce replay (R2) from the T0011.00 slice",
    "bounce8": "rung 8 bounce replay (R2) from the T0011.00 slice",
}
ARGKEYS = ("ell", "tau", "rs", "nsites", "mix", "T_over_rs", "kappa_sea", "prep_ell", "reading", "reference", "R0_over_rs", "m", "g",
           "stop_E", "stop_fields", "maxit", "nk_set", "resume_slice", "resume_run")
HEADWORDS = ("e0", "engine", "version", "script", "resume", "md5", "commit", "git", "tag", "device", "precision", "dtype")
SLICE_RE = re.compile(r"^orbitals_(?P<tag>.+)_T(?P<clock>\d+(?:\.\d+)?)\.npz$")
SLIM_KEYS = ("T_over_rs", "E_over_E0", "E_over_E0_ref", "dE", "dE_growth", "dE_increment", "min", "min_ref", "dmin", "dmin_growth",
             "roots", "roots_ref", "droots", "masks_equal", "masks_all_equal", "within", "evaluations", "evaluations_ref", "seconds", "seconds_ref")
CMP_HDR = ["T/r_s", "E/E0 run", "difference", "growth", "min(N - abs(v)) run", "difference", "roots run", "root differences",
           "masks equal", "within", "kept residual run; ref", "converged run; ref", "wall s run; ref"]


def default_cfg():
    return {
        "inputs": {
            "rung2": "<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_plain_v36d_from0.json",
            "rung4_first": BULK + "/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json",
            "rung4": BULK + "/v36d_from0/runs_pod7_ell4/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json",
            "rung8_first": BULK + "/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0.json",
            "rung8": BULK + "/pod_sync/pod2_engine/v36d_from0/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json",
            "composite": BULK + "/v36d_relaunch/records/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json",
            "bounce4": BULK + "/v36d_from0/bounce_R2_runs/ell4/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_bounce.json",
            "bounce8": BULK + "/v36d_from0/bounce_R2_runs/ell8/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_bounce.json",
        },
        "slice_dirs": [BULK + "/v36d_from0/" + d for d in ("slices", "slices_pod7", "slices_pod3", "slices_spill", "slices_bounce_R2")],
        "member_json": BULK + "/ladder_v15/member2_final_175.json",
        "v17_md": "<internal>/COLLAPSE_ladder_v1.7.md",
        "cmp_v36d_json": os.path.join(SOLVE, "results", "engine", "V36D", "cmp_v36d_vs_member.json"),
        "out_dir": os.path.join(SOLVE, "results", "engine", "LADV2"),
        "gate18": {},
        "extra_members": [],
        "every": 0.5,
        "restart_window": (17.125, 18.125),
        "seam_window": (16.0, 18.125),
        "flip_clocks": (17.625, 18.125),
        "integer_clocks": (1, 40),
        "md5": True,
        "md5_duplicates": False,
    }


# ----------------------------------------------------------------------------------------------------------------------- small helpers
def short(x, n=90):
    if x is None:
        return None
    s = json.dumps(x, sort_keys=True, default=str) if isinstance(x, (dict, list, tuple)) else str(x)
    return s if len(s) <= n else s[:n - 3] + "..."


def fmt(x, f="%.3f"):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        try:
            return f % x
        except (TypeError, ValueError):
            return str(x)
    return str(x)


def cell(x):
    return str(x).replace("|", "/").replace("\n", " ")


def table(headers, rows):
    L = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    L += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return L


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def clock_of(path):
    m = re.search(r"_T(\d+(?:\.\d+)?)\.npz$", str(path or ""))
    return float(m.group(1)) if m else None


def ranges_text(clocks):
    cl = sorted(set(round(float(c), 6) for c in clocks))
    if not cl:
        return "none"
    parts, i = [], 0
    while i < len(cl):
        j = i
        while j + 1 < len(cl) and abs(cl[j + 1] - cl[j] - 1.0) < 1e-9:
            j += 1
        parts.append("%g" % cl[i] if j == i else "%g to %g" % (cl[i], cl[j]))
        i = j + 1
    return ", ".join(parts)


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def sub(a, b):
    return (a - b) if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else None


def sublist(a, b):
    return [x - y for x, y in zip(a, b)] if (a and b and len(a) == len(b)) else None


# ----------------------------------------------------------------------------------------------------------------------- the jsons
def find_key(J, key):
    for sect in ("args", "head"):
        d = J.get(sect)
        if isinstance(d, dict):
            if d.get(key) is not None:
                return d[key]
            for v in d.values():
                if isinstance(v, dict) and v.get(key) is not None:
                    return v[key]
    return None


def head_subset(head, n=14):
    if not isinstance(head, dict):
        return {"head": short(head)}
    out = {}
    for k in sorted(head):
        if any(w in k.lower() for w in HEADWORDS):
            out[k] = short(head[k], 100)
            if len(out) >= n:
                break
    return out


def v36_text(r):
    v = r.get("v36")
    if isinstance(v, dict):
        for k in ("version", "engine", "script", "tag", "name"):
            if v.get(k) is not None:
                return short(v[k], 40)
        return "v36 entry: " + ",".join(sorted(v)[:5])
    return short(v, 40)


def load_slim(path, md5=False):
    """one json read; only the record fields the tables use are kept (the records' arrays are dropped)."""
    t0 = time.time()
    with open(path) as f:
        J = json.load(f)
    args = J.get("args") if isinstance(J.get("args"), dict) else {}
    newton = args.get("mix") == "newton"
    recs = []
    for r in J.get("records") or []:
        if not isinstance(r, dict) or "E" not in r or "T_over_rs" not in r:
            continue
        tz = r.get("trapping_zeros") or []
        sv = r.get("solve_v32") if isinstance(r.get("solve_v32"), dict) else {}
        recs.append({
            "T_over_rs": float(r["T_over_rs"]), "T": r.get("T"), "E": r.get("E"), "E_over_E0": r.get("E_over_E0"),
            "min_N_minus_absv": r.get("min_N_minus_absv"), "R_t": r.get("R_t"), "R_t_over_2gE": r.get("R_t_over_2gE"), "R_c": r.get("R_c"),
            "trapping_zeros": [float(z) for z in tz] if isinstance(tz, (list, tuple)) else [], "has_tz_key": "trapping_zeros" in r,
            "n_pocket": safe(LC.pockets, r) or 0, "converged": r.get("converged"), "failed": bool(r.get("failed")),
            "unconverged": bool(r.get("unconverged")), "residual": safe(LC.rec_residual, r, newton),
            "confirming": safe(LC.confirm_residual, r) if newton else None, "meets_stop_rule": sv.get("meets_stop_rule"),
            "engine": short(r.get("engine"), 60), "v36": v36_text(r), "solver": short(r.get("solver"), 60),
            "continuation_label": short(r.get("continuation_label"), 90), "record_label": short(r.get("record_label"), 90),
            "slice_saved": bool(r.get("slice_saved")), "seconds": r.get("seconds"), "iterations": r.get("iterations"),
        })
    meta = {"path": path, "bytes": os.path.getsize(path), "md5": md5sum(path) if md5 else None, "script": J.get("script"),
            "args": {k: args.get(k) for k in ARGKEYS if k in args},
            "args_solver": {k: v for k, v in args.items() if k.startswith("nk_") or k in ("mix", "mix_alpha", "source_continuation", "end_above")},
            "head": head_subset(J.get("head")), "resume_slice": find_key(J, "resume_slice"), "resume_run": find_key(J, "resume_run"),
            "newton": newton, "records_raw": len(J.get("records") or []), "load_seconds": round(time.time() - t0, 1)}
    del J
    return {"meta": meta, "recs": recs}


def rle(recs, key):
    runs = []
    for r in recs:
        v = r.get(key)
        if runs and runs[-1]["value"] == v:
            runs[-1]["last"] = r["T_over_rs"]
            runs[-1]["records"] += 1
        else:
            runs.append({"value": v, "first": r["T_over_rs"], "last": r["T_over_rs"], "records": 1})
    return runs


def rle_text(runs, n=6):
    parts = ["%s (%.3f to %.3f, %d)" % ("-" if u["value"] is None else u["value"], u["first"], u["last"], u["records"]) for u in runs[:n]]
    if len(runs) > n:
        parts.append("... %d more runs" % (len(runs) - n))
    return "; ".join(parts) if parts else "-"


def grid_check(recs, step=0.125):
    if not recs:
        return {"step": step, "expected": 0, "missing": [], "missing_count": 0, "duplicate_clocks": 0}
    f, l = recs[0]["T_over_rs"], recs[-1]["T_over_rs"]
    have = set(int(round(r["T_over_rs"] / step)) for r in recs)
    n = int(round((l - f) / step)) + 1
    k0 = int(round(f / step))
    missing = [round((k0 + k) * step, 6) for k in range(n) if (k0 + k) not in have]
    return {"step": step, "expected": n, "missing": missing[:20], "missing_count": len(missing), "duplicate_clocks": len(recs) - len(have)}


def episodes(recs):
    """one rule on every member (the STEP_controls ruling): a maximal run of records holding a trapping zero; the closing clock is the record after its last."""
    has_key = any(r["has_tz_key"] for r in recs)

    def hz(r):
        if has_key:
            return bool(r["trapping_zeros"])
        x = r.get("R_t")
        return isinstance(x, (int, float)) and x == x and x > 0

    rule = ("a record holds a trapping zero when its trapping_zeros list is non-empty" if has_key else
            "trapping_zeros is absent from every record: a record holds a zero when R_t is finite and positive")
    eps, cur = [], None
    for r in recs:
        if hz(r):
            if cur is None:
                cur = {"first": r["T_over_rs"], "last": r["T_over_rs"], "records": 0, "R_t_first": r["R_t"], "R2_first": r["R_t_over_2gE"],
                       "R_t_last": None, "R2_last": None, "min_value": None, "min_at": None, "closes_at": None}
            cur["last"], cur["R_t_last"], cur["R2_last"] = r["T_over_rs"], r["R_t"], r["R_t_over_2gE"]
            cur["records"] += 1
            m = r["min_N_minus_absv"]
            if isinstance(m, (int, float)) and (cur["min_value"] is None or m < cur["min_value"]):
                cur["min_value"], cur["min_at"] = m, r["T_over_rs"]
        elif cur is not None:
            cur["closes_at"] = r["T_over_rs"]
            eps.append(cur)
            cur = None
    if cur is not None:
        eps.append(cur)
    return rule, eps


def residual_bands(recs):
    rr = [(r["T_over_rs"], r["residual"]) for r in recs if isinstance(r["residual"], (int, float))]
    band = [t for t, x in rr if 1e-4 <= x <= 1e-3]
    above = [t for t, x in rr if x > 1e-3]
    mx = max(rr, key=lambda p: p[1]) if rr else None
    return {"with_residual": len(rr), "below_1e-4": sum(1 for _, x in rr if x < 1e-4), "band_1e-4_1e-3": len(band), "band_clocks": band[:20],
            "above_1e-3_count": len(above), "above_1e-3_first": above[0] if above else None, "max": mx[1] if mx else None,
            "max_at": mx[0] if mx else None, "converged_true": sum(1 for r in recs if r["converged"] is True),
            "converged_false": sum(1 for r in recs if r["converged"] is False), "converged_absent": sum(1 for r in recs if r["converged"] is None),
            "meets_stop_rule_true": sum(1 for r in recs if r["meets_stop_rule"] is True),
            "unconverged_entries": sum(1 for r in recs if r["unconverged"]), "failed": sum(1 for r in recs if r["failed"])}


def member_summary(label, role, run):
    m, R = run["meta"], run["recs"]
    rule, eps = episodes(R)
    return {"label": label, "role": role, "path": m["path"], "basename": os.path.basename(m["path"]), "bytes": m["bytes"], "md5": m["md5"],
            "script": m["script"], "args": m["args"], "args_solver": m["args_solver"], "head": m["head"], "resume_slice": m["resume_slice"],
            "resume_run": m["resume_run"], "resume_clock": clock_of(m["resume_slice"]), "newton": m["newton"], "records": len(R),
            "records_raw": m["records_raw"], "first": R[0]["T_over_rs"] if R else None, "last": R[-1]["T_over_rs"] if R else None,
            "grid": grid_check(R), "engine_runs": rle(R, "engine"), "v36_runs": rle(R, "v36"), "solver_runs": rle(R, "solver"),
            "continuation_runs": rle(R, "continuation_label"), "record_label_runs": rle(R, "record_label"), "residual": residual_bands(R),
            "episode_rule": rule, "episodes": eps, "slices_saved": [r["T_over_rs"] for r in R if r["slice_saved"]],
            "E_over_E0_first": R[0]["E_over_E0"] if R else None, "E_over_E0_last": R[-1]["E_over_E0"] if R else None,
            "load_seconds": m["load_seconds"]}


# ----------------------------------------------------------------------------------------------------------------------- the tools
def tool_target(json_path, scratch):
    """compare_member_floor_v2 takes DIR/TAG and opens DIR/collapse_TAG.json; a json not named so gets a symlink in the scratch directory."""
    d, b = os.path.split(os.path.abspath(json_path))
    if b.startswith("collapse_") and b.endswith(".json"):
        return os.path.join(d, b[len("collapse_"):-len(".json")])
    tag = b[:-5] if b.endswith(".json") else b
    os.makedirs(scratch, exist_ok=True)
    link = os.path.join(scratch, "collapse_%s.json" % tag)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(d, b), link)
    return os.path.join(scratch, tag)


def run_tool(cmd):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        rc, so, se = p.returncode, p.stdout, p.stderr
    except Exception as e:
        rc, so, se = -1, "", repr(e)
    return {"command": " ".join(cmd), "returncode": rc, "stdout_tail": so.strip()[-300:], "stderr_tail": se.strip()[-800:],
            "seconds": round(time.time() - t0, 1)}


def run_cmf(run_json, ref_json, out_base, label_run, label_ref, scratch):
    md, js = out_base + ".md", out_base + ".json"
    cmd = [sys.executable, CMF, "--run", tool_target(run_json, scratch), "--ref", tool_target(ref_json, scratch), "--label-run", label_run,
           "--label-ref", label_ref, "--md", md, "--out-json", js, "--criterion", "twice"]
    res = run_tool(cmd)
    res.update({"tool": "compare_member_floor_v2.py", "md": md, "json": js, "label_run": label_run, "label_ref": label_ref,
                "run_json": run_json, "ref_json": ref_json})
    if res["returncode"] == 0 and os.path.exists(js):
        with open(js) as f:
            res["data"] = json.load(f)
    return res


def run_cv5(members, out_md, label, every):
    cmd = [sys.executable, CV5]
    for k, p in members:
        cmd += ["--member", "%s=%s" % (k, p)]
    cmd += ["--every", str(every), "--md", out_md, "--label", label]
    res = run_tool(cmd)
    res.update({"tool": "engine_ladder_compare_v5.py", "md": out_md})
    if res["returncode"] == 0 and os.path.exists(out_md):
        with open(out_md) as f:
            res["text"] = f.read()
    return res


# ----------------------------------------------------------------------------------------------------------------------- the tool's rows
def slim_row(x):
    y = {k: x.get(k) for k in SLIM_KEYS}
    sr, sq = x.get("solve_run") or {}, x.get("solve_ref") or {}
    y.update({"kept_run": sr.get("kept_residual"), "kept_ref": sq.get("kept_residual"), "converged_run": sr.get("converged"),
              "converged_ref": sq.get("converged"), "continuation_run": sr.get("continuation"), "continuation_ref": sq.get("continuation")})
    return y


def row_nonzero(x):
    if x.get("dE") or x.get("dmin"):
        return True
    if x.get("droots") is None:
        return bool(x.get("roots") or x.get("roots_ref"))
    return any(d != 0 for d in x["droots"])


def maxabs(rows, key):
    best = None
    for x in rows:
        v = x.get(key)
        if isinstance(v, (int, float)) and (best is None or abs(v) > abs(best[0])):
            best = (v, x["T_over_rs"])
    return best


def maxabs_root(rows):
    best = None
    for x in rows:
        for i, d in enumerate(x.get("droots") or []):
            if best is None or abs(d) > abs(best[0]):
                best = (d, x["T_over_rs"], i)
    return best


def roots_text(z):
    return ", ".join("%.6f" % v for v in z) if z else "none"


def droots_text(x):
    if x.get("droots") is None:
        a, b = len(x.get("roots") or []), len(x.get("roots_ref") or [])
        return "count %d against %d" % (a, b) if (a or b) else ""
    return ", ".join("%+.2e" % d for d in x["droots"])


def masks_text(x):
    me = x.get("masks_equal") or []
    if not me or all(m is None for m in me):
        return "not stored"
    return "%d of %d" % (sum(1 for m in me if m), len(me))


def cmp_row(x, extra=()):
    return [fmt(x["T_over_rs"], "%.4f"), fmt(x.get("E_over_E0"), "%.8f"), fmt(x.get("dE"), "%+.2e"), fmt(x.get("dE_growth"), "%.2f"),
            fmt(x.get("min"), "%+.6f"), fmt(x.get("dmin"), "%+.2e"), roots_text(x.get("roots")), droots_text(x), masks_text(x), fmt(x.get("within")),
            "%s; %s" % (fmt(x.get("kept_run"), "%.2e"), fmt(x.get("kept_ref"), "%.2e")),
            "%s; %s" % (fmt(x.get("converged_run")), fmt(x.get("converged_ref"))),
            "%s; %s" % (fmt(x.get("seconds"), "%.0f"), fmt(x.get("seconds_ref"), "%.0f"))] + list(extra)


def pair_reading(cmf, resume_clock, kind):
    """a resume (kind 'resume') or a bounce replay (kind 'bounce') read from the tool's rows: carried rows up to the slice's clock,
    computed rows after it."""
    d = cmf.get("data")
    if not d:
        return {"error": cmf.get("stderr_tail") or "no data", "command": cmf.get("command"), "resume_clock": resume_clock}
    rows = [slim_row(x) for x in d["rows"]]
    rc = resume_clock
    carried = [x for x in rows if rc is not None and x["T_over_rs"] <= rc + 1e-9]
    computed = [x for x in rows if rc is None or x["T_over_rs"] > rc + 1e-9]
    nz = [x["T_over_rs"] for x in carried if row_nonzero(x)]
    last = rows[-1]["T_over_rs"] if rows else None
    if kind == "resume":
        def show(x):
            return (rc is not None and x["T_over_rs"] >= rc - 0.5 - 1e-9) or (not x["within"]) or (x["T_over_rs"] in nz)
    else:
        def show(x):
            t = x["T_over_rs"]
            return ((rc is not None and t > rc and abs(t - math.floor(t) - 0.5) < 1e-9) or t == last or (not x["within"]) or (t in nz))
    return {"kind": kind, "resume_clock": rc, "common": len(rows), "first": rows[0]["T_over_rs"] if rows else None, "last": last,
            "carried": len(carried), "carried_nonzero_count": len(nz), "carried_nonzero_clocks": nz[:20], "computed": len(computed),
            "all_within": d.get("all_within"), "first_not_within": d.get("first_not_within"),
            "resumed_record": computed[0] if computed else None, "max_abs_dE_computed": maxabs(computed, "dE"),
            "max_abs_dmin_computed": maxabs(computed, "dmin"), "max_abs_droot_computed": maxabs_root(computed),
            "rows_shown": [x for x in rows if show(x)], "rows_all": rows, "md": cmf.get("md"), "json": cmf.get("json"),
            "command": cmf.get("command"), "seconds": cmf.get("seconds"), "label_run": cmf.get("label_run"), "label_ref": cmf.get("label_ref")}


def select_cmp_rows(rows, every, windows, extra_clocks):
    sel, prev = [], None
    for x in rows:
        t = x["T_over_rs"]
        on_grid = abs(t / every - round(t / every)) < 1e-9
        in_win = any(a - 1e-9 <= t <= b + 1e-9 for a, b in windows)
        extra = any(abs(t - c) < 1e-9 for c in extra_clocks)
        flip = prev is not None and x["within"] != prev
        if on_grid or in_win or extra or flip:
            sel.append(x)
        prev = x["within"]
    return sel


def comparator_reading(cmf, composite_recs, windows, every, extra_clocks):
    d = cmf.get("data")
    if not d:
        return {"error": cmf.get("stderr_tail") or "no data", "command": cmf.get("command")}
    seg = {round(r["T_over_rs"], 6): r for r in composite_recs}
    rows = []
    for x in d["rows"]:
        y = slim_row(x)
        c = seg.get(round(y["T_over_rs"], 6)) or {}
        y["composite_engine"] = c.get("engine")
        y["composite_v36"] = c.get("v36")
        y["composite_label"] = c.get("continuation_label") or c.get("record_label")
        y["reading"] = "within" if y["within"] else ("beyond twice the floor: the composite history's propagated term (its segment here: %s)"
                                                      % (y["composite_engine"] or "-"))
        rows.append(y)
    win = {}
    for a, b in windows:
        w = [x for x in rows if a - 1e-9 <= x["T_over_rs"] <= b + 1e-9]
        win["%.3f to %.3f" % (a, b)] = {"n": len(w), "n_within": sum(1 for x in w if x["within"]),
                                        "all_within": (all(x["within"] for x in w) if w else None),
                                        "first_not_within": next((x["T_over_rs"] for x in w if not x["within"]), None),
                                        "max_abs_dE": maxabs(w, "dE"), "max_abs_dmin": maxabs(w, "dmin"), "max_abs_droot": maxabs_root(w)}
    fnw = d.get("first_not_within")
    return {"common": len(rows), "first": rows[0]["T_over_rs"] if rows else None, "last": rows[-1]["T_over_rs"] if rows else None,
            "all_within": d.get("all_within"), "first_not_within": fnw,
            "segment_at_first_not_within": (seg.get(round(fnw, 6)) or {}).get("engine") if fnw is not None else None,
            "n_within": sum(1 for x in rows if x["within"]), "n_not_within": sum(1 for x in rows if not x["within"]),
            "count_mismatch_clocks": [x["T_over_rs"] for x in rows if x["droots"] is None and (x["roots"] or x["roots_ref"])][:40],
            "max_abs_dE": maxabs(rows, "dE"), "max_abs_dmin": maxabs(rows, "dmin"), "max_abs_droot": maxabs_root(rows), "windows": win,
            "rows_shown": select_cmp_rows(rows, every, windows, extra_clocks + ([fnw] if fnw is not None else [])), "rows_all": rows,
            "md": cmf.get("md"), "json": cmf.get("json"), "command": cmf.get("command"), "seconds": cmf.get("seconds"),
            "label_run": cmf.get("label_run"), "label_ref": cmf.get("label_ref")}


# ----------------------------------------------------------------------------------------------------------------------- version 1.7, the flip
def parse_v17(text):
    """light parsing of version 1.7's head: the seam table (point minus member) and the flip table (stopped run minus member)."""
    out = {"seam": {}, "flip": {}}
    sec = None
    for line in text.splitlines():
        if line.startswith("## The point's seam"):
            sec = "seam"
            continue
        if line.startswith("## The flip's record-step term"):
            sec = "flip"
            continue
        if line.startswith("## Version 1.6"):
            sec = None
            continue
        if sec and line.startswith("| ") and len(line) > 2 and line[2].isdigit():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            try:
                t, dE, dm = round(float(c[0]), 6), float(c[1]), float(c[2])
            except (ValueError, IndexError):
                continue
            try:
                dr = [float(v) for v in c[4].split(",")] if (len(c) > 4 and c[4] not in ("", "-")) else None
            except ValueError:
                dr = None
            e = {"dE": dE, "dmin": dm, "roots": c[3] if len(c) > 3 else "", "droots": dr, "masks": c[5] if len(c) > 5 else ""}
            if sec == "seam" and len(c) >= 9:
                e.update({"within": c[6], "kept": c[7], "reading": c[8]})
            elif sec == "flip" and len(c) >= 7:
                e["kept"] = c[6]
            out[sec][t] = e
    return out


def record_step(v17, member_rows, rung_rows, flip_clocks):
    seam, flip = (v17 or {}).get("seam", {}), (v17 or {}).get("flip", {})
    bm = {round(x["T_over_rs"], 6): x for x in (member_rows or [])}
    br = {round(x["T_over_rs"], 6): x for x in (rung_rows or [])}
    rows = []
    for t in sorted(flip):
        if not (flip_clocks[0] - 1e-9 <= t <= flip_clocks[1] + 1e-9):
            continue
        s, p, pm, rc = flip[t], seam.get(t), bm.get(t), br.get(t)
        if pm:
            p_dE, p_dmin, p_dr, src = pm["dE"], pm["dmin"], pm["droots"], "the box compare of the composite against the member"
        elif p:
            p_dE, p_dmin, p_dr, src = p["dE"], p["dmin"], p["droots"], "version 1.7's seam table (printed digits)"
        else:
            p_dE = p_dmin = p_dr = None
            src = "not available"
        sp = {"dE": sub(s["dE"], p_dE), "dmin": sub(s["dmin"], p_dmin), "droots": sublist(s["droots"], p_dr)}
        r2 = {"dE": rc["dE"] if rc else None, "dmin": rc["dmin"] if rc else None, "droots": rc["droots"] if rc else None}
        sr = {"dE": sub(sp["dE"], r2["dE"]), "dmin": sub(sp["dmin"], r2["dmin"]), "droots": sublist(sp["droots"], r2["droots"])}
        rows.append({"T_over_rs": t,
                     "stopped_minus_member": {"dE": s["dE"], "dmin": s["dmin"], "droots": s["droots"], "masks": s.get("masks"), "kept": s.get("kept")},
                     "point_minus_member": {"dE": p_dE, "dmin": p_dmin, "droots": p_dr, "source": src},
                     "stopped_minus_composite_derived": sp, "rung2_minus_composite": r2, "stopped_minus_rung2_derived": sr})
    mx = None
    for t in flip:
        if mx is None or abs(flip[t]["dE"]) > abs(mx[0]):
            mx = (flip[t]["dE"], t)
    last = max(flip) if flip else None
    return {"rows": rows, "flip_rows_parsed": len(flip), "seam_rows_parsed": len(seam), "stopped_minus_member_max_abs_dE": mx,
            "stopped_minus_member_at_last_clock": ({"T_over_rs": last, "dE": flip[last]["dE"], "dmin": flip[last]["dmin"],
                                                    "droots": flip[last]["droots"]} if last is not None else None)}


HREDUCE_INEQ = [
    "hreduce's (1): x(t) <= X(t) = 2 (e^{L Delta} - 1) e^{L (t - t_+)} for k = 2 and t >= t_+, L = 2 n H R, R = AB/(1 - ABp), ABp < 1; "
    "the red team's equal form Z = 2 b e^{Lambda (t - t_b)} (e^{Lambda d} - 1)/Lambda, d = t_b - t_a <= Delta, b = Lambda = 2 r beta s, "
    "s = alpha/(1 - alpha p), with alpha = AB, beta = H, s = R.",
    "hreduce's (2): |E_B - E_A|/E_0 <= (B_infinity/E_0) X(t); |R_B - R_A|/R_norm <= 2 G B_* X(t)/(R_norm m_*), m_* = inf N_R |d_R| > 0, "
    "d_R = 1 - 8 pi G R^2 (rho_R - J_R); the red team's |Delta E|/E_0 <= H Z/E_0 and |Delta R|/R_* <= 2 G L_R Z/(nu c R_*).",
    "An initial difference x_- at t_- adds e^{L (t - t_-)} x_-; an independent forcing eta adds integral e^{L (t - s)} eta(s) ds; applied roots "
    "retain the row-defect integrals; at a fold (d_R = 0) the root bound does not apply; a permanent branch difference is not a one-step forcing.",
]


def hreduce_block(rs_over_a, rstep):
    delta_rs = 0.125
    delta_a = delta_rs * rs_over_a if rs_over_a else None
    conditions = [
        ("the two histories share C(t_-), or the initial difference x_- is carried as e^{L (t - t_-)} x_-",
         "not zero: version 1.7's pre-flip seam terms (point against the member; 54.1 reads their size as 9.05e-08 in E/E0 and 3.72e-06 in the roots "
         "at 17.125 to 17.5) and, for rung 2 against the composite, the whole difference before the composite's first seam at 16.0; carried as the "
         "initial-difference term, not evaluated"),
        ("the same k labels placed within one step [t_a, t_b], d = t_b - t_a <= Delta",
         "k = 2 by the stored masks (two changed bits; 53.1 as hreduce reads it, the placement 17.625 to 17.75); two mask bits do not establish the "
         "physical change's trace weight (the review); Delta = %.3f r_s between adjacent records%s (derived; conditional on the clock units, the red team's conversion)"
         % (delta_rs, (", %g a at r_s/a %g" % (delta_a, rs_over_a)) if delta_a else "")),
        ("a common continued branch (the same compatible chart) after t_b", "not established: equal sorted masks after the step do not prove it (the review, the red team)"),
        ("response constants A, B, p, H (alpha, p, beta) with ABp < 1, giving R and L", "none supplied by the inputs"),
        ("n = Tr C (r = rank C), finite on the parent", "about 2048 occupied columns per channel on this lattice (derived from the STEP_controls ruling's slice arithmetic), "
         "10 channels; unused without the other constants"),
        ("source-variation bounds B_infinity and B_* (or H = sup ||h|| and L_R)", "none supplied"),
        ("a connected simple outer-root family with m_* = inf N_R |d_R| > 0 (no fold, birth, loss or relabelling of the outer root)",
         "not established: both histories hold an outer root at each of the five clocks (version 1.7's tables), a record-by-record presence; the records' "
         "slope at R_t is a per-record diagnostic, not the uniform collar factor"),
        ("R_norm (R_*) for the root test", "a, the lattice unit: the tables print the root differences in lattice units (the review, the red team)"),
        ("independent forcing eta (solver residuals, propagation defects)", "retained, not evaluated; the kept residuals print beside every row"),
        ("the recorded gap 9.52e-06: a distance to zero g_0 or an occupied/empty separation g_P, with a path bound V (V Delta < g_0, or 2 V Delta < g_P; "
         "the red team's g_0/3)", "not identified; no path bound is supplied; endpoint gaps alone locate no crossing"),
    ]
    tests = []
    for row in (rstep or {}).get("rows", []):
        sp = row["stopped_minus_composite_derived"]
        if row["T_over_rs"] >= 17.75 - 1e-9 and sp["dE"] is not None:
            tests.append({"T_over_rs": row["T_over_rs"], "stopped_minus_point_dE_derived": sp["dE"],
                          "stopped_minus_point_outer_root_a_derived": (sp["droots"][-1] if sp["droots"] else None)})
    return {"inequalities": HREDUCE_INEQ, "conditions": conditions, "k": 2, "gap_recorded": 9.52e-06, "delta_rs_derived": delta_rs,
            "delta_a_derived": delta_a, "filed_tests": {"E_over_E0": 9.6e-05, "outer_root_a": 1.6e-02}, "tests_derived_after_t_b": tests,
            "numerical_bound": "none computed: the constants are not supplied by the inputs; the filed values and the derived values after t_b are its "
                               "tests, untested; a growth factor fitted to the rows is not L and does not print as it"}


# ----------------------------------------------------------------------------------------------------------------------- the slices
def rung_of(tag):
    m = re.search(r"_ell(\d+)_", tag)
    return int(m.group(1)) if m else None


def kind_of(tag):
    return "bounce" if "_bounce" in tag else ("continuation" if "_to40" in tag else "first run")


def slice_inventory(dirs, members, integer_clocks, md5_dups):
    inv, missing_dirs = [], []
    for d in dirs:
        if not os.path.isdir(d):
            missing_dirs.append(d)
            continue
        for fn in sorted(os.listdir(d)):
            m = SLICE_RE.match(fn)
            if m:
                p = os.path.join(d, fn)
                inv.append({"dir": d, "tag": m.group("tag"), "clock": float(m.group("clock")), "bytes": os.path.getsize(p), "path": p})
    groups = {}
    for e in inv:
        g = groups.setdefault((e["dir"], e["tag"]), {"dir": e["dir"], "tag": e["tag"], "rung": rung_of(e["tag"]), "kind": kind_of(e["tag"]), "clocks": [], "bytes": 0})
        g["clocks"].append(e["clock"])
        g["bytes"] += e["bytes"]
    glist = sorted(groups.values(), key=lambda g: (g["rung"] or 0, g["kind"], g["dir"]))
    for g in glist:
        g["clocks"].sort()
    lo, hi = integer_clocks
    per = {}
    for s in (2, 4, 8):
        hist = sorted(set(e["clock"] for e in inv if rung_of(e["tag"]) == s and kind_of(e["tag"]) != "bounce"))
        half = sorted(set(e["clock"] for e in inv if rung_of(e["tag"]) == s and kind_of(e["tag"]) == "bounce"))
        ints = set(int(round(c)) for c in hist if abs(c - round(c)) < 1e-9)
        per[s] = {"clocks": hist, "integer_missing": [k for k in range(lo, hi + 1) if k not in ints],
                  "non_integer": [c for c in hist if abs(c - round(c)) > 1e-9], "half_clock_states": half}
    seen = {}
    for e in inv:
        seen.setdefault((e["tag"], e["clock"]), []).append(e)
    dups = []
    for (tag, clock), es in sorted(seen.items()):
        if len(es) > 1:
            dd = {"tag": tag, "clock": clock, "paths": [e["path"] for e in es], "bytes": [e["bytes"] for e in es], "md5": None, "md5_equal": None}
            if md5_dups:
                dd["md5"] = [md5sum(e["path"]) for e in es]
                dd["md5_equal"] = len(set(dd["md5"])) == 1
            dups.append(dd)
    flags = {}
    for s in (2, 4, 8):
        marked = set()
        for m in members:
            if SIGMA.get(m["role"]) == s and m["role"] in ("rung2", "rung4", "rung8", "rung4_first", "rung8_first"):
                marked |= set(round(c, 6) for c in m["slices_saved"])
        box = set(round(c, 6) for c in per[s]["clocks"])
        flags[s] = {"marked_saved": sorted(marked), "marked_not_on_box": sorted(marked - box), "on_box_not_marked": sorted(box - marked)}
    return {"entries": len(inv), "missing_dirs": missing_dirs, "groups": glist, "per_rung": per, "duplicates": dups, "records_flags": flags}


# ----------------------------------------------------------------------------------------------------------------------- the md
def compose_md(S, cfg):
    tw, fl = S["twice_floor"], S["floor"]
    L = ["# COLLAPSE_ladder version 2: the tables (compose_ladder_v2_v1.py)", "",
         "Composed on the box CPU by code/engine/compose_ladder_v2_v1.py from the jsons listed under Inputs. engine_ladder_compare_v5.py and "
         "compare_member_floor_v2.py ran as subprocesses; their files sit beside this one in %s. Every number is printed by one of the three "
         "scripts, or carried from version 1.7's printed tables and marked so, or marked derived. No engine ran; no slice's contents were read. "
         "Elapsed %.1f s." % (S["out_dir"], S["elapsed_seconds"]), "",
         "The floor (two runs of the same arguments compared; compare_member_floor_v2's constants): E/E0 %.1e, roots %.1e lattice units, "
         "min(N - |v|) %.1e. Twice the floor, the criterion of every compare below: %.1e, %.1e, %.1e. A record is within when its E/E0 and "
         "min(N - |v|) differences and every root difference are within those allowances, the root counts are equal and no stored mask is unequal; "
         "the masks read \"not stored\" where no spectra npz sits beside a json." % (fl["E_over_E0"], fl["roots"], fl["min_N_minus_absv"],
                                                                                   tw["E_over_E0"], tw["roots"], tw["min_N_minus_absv"]), ""]
    if S["errors"]:
        L += ["Stages that failed (their tables are absent below): " + "; ".join("%s: %s" % (e["stage"], e["error"]) for e in S["errors"]), ""]
    members, extras = S.get("members") or [], S.get("extra_members") or []
    # 1
    L += ["## 1 Inputs and the per-member table", "", "### Inputs", ""]
    L += table(["role", "label", "path", "bytes", "md5", "records", "first T/r_s", "last T/r_s", "script field", "mix", "resume slice", "resume run"],
               [[m["role"], m["label"], m["path"], m["bytes"], m["md5"] or "-", m["records"], fmt(m["first"]), fmt(m["last"]), m["script"] or "-",
                 m["args"].get("mix", "-"), short(m["resume_slice"], 100) or "-", short(m["resume_run"], 80) or "-"] for m in members + extras])
    L += ["", "### The per-member table (the engine and solver as the records state them; the residual compare_v5's rec_residual; the stop rule the record's converged entry)", ""]
    hdr = ["member", "json", "script field", "engine (records: value, first to last clock, count)", "solver (args mix; records' solver entries)", "records",
           "first", "last", "clocks missing on the 0.125 grid", "converged true / false / absent", "failed", "residual max (at)",
           "below 1e-4; 1e-4 to 1e-3; above 1e-3 first at", "arguments' nk entries", "head entries", "slices saved per the records"]
    rows = []
    for m in [x for x in members if x["role"] in HISTORY_ROLES + ("composite",)] + extras:
        rb = m["residual"]
        rows.append([m["label"], m["basename"], m["script"] or "-", rle_text(m["engine_runs"]), "%s; %s" % (m["args"].get("mix", "-"), rle_text(m["solver_runs"], 3)),
                     m["records"], fmt(m["first"]), fmt(m["last"]),
                     "%d%s" % (m["grid"]["missing_count"], (" (" + ", ".join("%.3f" % t for t in m["grid"]["missing"]) + ")") if m["grid"]["missing"] else ""),
                     "%d / %d / %d" % (rb["converged_true"], rb["converged_false"], rb["converged_absent"]), rb["failed"],
                     "%s (%s)" % (fmt(rb["max"], "%.2e"), fmt(rb["max_at"])),
                     "%d; %d; %s" % (rb["below_1e-4"], rb["band_1e-4_1e-3"], fmt(rb["above_1e-3_first"]) if rb["above_1e-3_first"] is not None else "no record"),
                     short(m["args_solver"], 120) or "-", "; ".join("%s=%s" % (k, v) for k, v in m["head"].items()) or "-",
                     "%d (%s)" % (len(m["slices_saved"]), ranges_text(m["slices_saved"]))])
    L += table(hdr, rows)
    L += ["", "The v36 entries of the records, the continuation labels and the record labels (runs of equal values):", ""]
    for m in [x for x in members if x["role"] in HISTORY_ROLES + ("composite",)]:
        L.append("- %s: v36 %s; continuation labels %s; record labels %s" % (m["label"], rle_text(m["v36_runs"], 4), rle_text(m["continuation_runs"], 6), rle_text(m["record_label_runs"], 4)))
    L += ["", "### The first runs and the replays (coverage)", ""]
    L += table(["run", "json", "records", "first", "last", "resume slice (clock)", "converged true / false", "residual max (at)", "slices saved per the records"],
               [[m["label"], m["basename"], m["records"], fmt(m["first"]), fmt(m["last"]), "%s (%s)" % (short(m["resume_slice"], 70) or "-", fmt(m["resume_clock"])),
                 "%d / %d" % (m["residual"]["converged_true"], m["residual"]["converged_false"]), "%s (%s)" % (fmt(m["residual"]["max"], "%.2e"), fmt(m["residual"]["max_at"])),
                 ranges_text(m["slices_saved"])] for m in members if m["role"] not in HISTORY_ROLES + ("composite",)])
    # 2
    L += ["", "## 2 Episodes (one rule on every member: a maximal run of records holding a trapping zero; the closing clock is the record after its last)", ""]
    rows = []
    for m in [x for x in members if x["role"] in HISTORY_ROLES + ("composite",)] + extras:
        if not m["episodes"]:
            rows.append([m["label"], "-", "none to %s" % fmt(m["last"]), "-", "-", "-", "-", "-", "-", m["episode_rule"]])
        for k, e in enumerate(m["episodes"], 1):
            rows.append([m["label"], k, fmt(e["first"]), fmt(e["last"]), e["records"],
                         fmt(e["closes_at"]) if e["closes_at"] is not None else "none to %s" % fmt(m["last"]),
                         "%s; %s" % (fmt(e["R_t_first"]), fmt(e["R_t_last"])), "%s; %s" % (fmt(e["R2_first"], "%.4f"), fmt(e["R2_last"], "%.4f")),
                         "%s at %s" % (fmt(e["min_value"], "%.4f"), fmt(e["min_at"])), m["episode_rule"] if k == 1 else ""])
    L += table(["member", "episode", "first T/r_s", "last T/r_s", "records", "closes at", "R_t first; last", "R_t/(2 g E) first; last (as stored)",
                "smallest min(N - abs v) in the episode", "rule"], rows)
    # 3
    L += ["", "## 3 engine_ladder_compare_v5 on the three rungs (the tool's file %s, whole)" % os.path.basename((S.get("cv5") or {}).get("md") or "-"), ""]
    cv5 = S.get("cv5") or {}
    if cv5.get("text"):
        L += [cv5["text"].rstrip("\n")]
    else:
        L += ["compare_v5 did not run or failed: returncode %s; %s" % (cv5.get("returncode"), cv5.get("stderr_tail") or cv5.get("error") or "-")]
    # 4
    L += ["", "## 4 The resumes at 18.125 to 18 (rungs 4 and 8): the continuation (_to40) against the first run at twice the floor; "
              "differences = continuation minus first run; resume (the resumability rule), not a seam", ""]
    for role in ("rung4", "rung8"):
        R = (S.get("resumes") or {}).get(role)
        L += ["### %s" % LABEL[role], ""]
        if not R or R.get("error"):
            L += ["- compare failed or absent: %s" % ((R or {}).get("error") or "no compare")]
            continue
        rr = R["resumed_record"] or {}
        L += ["- run = %s; ref = %s; the tool's files %s and %s" % (R["label_run"], R["label_ref"], os.path.basename(R["md"]), os.path.basename(R["json"])),
              "- resume slice clock %s: records at clocks up to it are carried from the first run, later records are computed by the continuation" % fmt(R["resume_clock"]),
              "- common records %d (%s to %s); carried rows %d, of which with any nonzero difference %d%s; computed rows %d" % (
                  R["common"], fmt(R["first"], "%.4f"), fmt(R["last"], "%.4f"), R["carried"], R["carried_nonzero_count"],
                  (" (at T/r_s " + ", ".join("%.4f" % t for t in R["carried_nonzero_clocks"]) + "; a carried record that differs is not a copy)") if R["carried_nonzero_clocks"] else "",
                  R["computed"]),
              "- every common record within twice the floor: %s; first not within: %s" % (fmt(R["all_within"]), fmt(R["first_not_within"], "%.4f")),
              "- the resumed record at %s: E/E0 difference %s, min(N - |v|) difference %s, root differences %s, masks %s, within %s" % (
                  fmt(rr.get("T_over_rs"), "%.4f"), fmt(rr.get("dE"), "%+.2e"), fmt(rr.get("dmin"), "%+.2e"), droots_text(rr) if rr else "-",
                  masks_text(rr) if rr else "-", fmt(rr.get("within"))),
              "- over the computed rows: largest |E/E0 difference| %s, |min difference| %s, |root difference| %s" % (
                  "%+.2e at %.4f" % R["max_abs_dE_computed"] if R["max_abs_dE_computed"] else "-",
                  "%+.2e at %.4f" % R["max_abs_dmin_computed"] if R["max_abs_dmin_computed"] else "-",
                  "%+.2e at %.4f (root %d)" % R["max_abs_droot_computed"] if R["max_abs_droot_computed"] else "-"), ""]
        L += table(CMP_HDR + ["row"], [cmp_row(x, ["carried" if (R["resume_clock"] is not None and x["T_over_rs"] <= R["resume_clock"] + 1e-9)
                                                     else ("resumed record" if rr and x["T_over_rs"] == rr.get("T_over_rs") else "computed")]) for x in R["rows_shown"]])
        L.append("")
    G = S.get("gates") or {}
    if G:
        L += ["### Resume-gate runs named on the box (run = the gate run, ref = the first run)", ""]
        for k, R in G.items():
            if R.get("error"):
                L.append("- %s: %s" % (k, R["error"]))
            else:
                L.append("- %s: common %d, every one within %s, first not within %s, resume clock %s; file %s" % (
                    k, R["common"], fmt(R["all_within"]), fmt(R["first_not_within"], "%.4f"), fmt(R["resume_clock"]), os.path.basename(R["md"])))
        L.append("")
    else:
        L += ["Rung 4's gate at 18 on pod 7 is input 12 of the plan and is cited, not re-run; rung 8's gate file is not among the inputs; "
              "the compares above are the box's own check of both resumes.", ""]
    # 5
    L += ["## 5 Stored states (no new run; no slice is read)", ""]
    SL = S.get("slices")
    if SL:
        L += ["### The complex128 slices on the box (orbitals_<tag>_T<clock>.npz; names, clocks and sizes only)", ""]
        if SL["missing_dirs"]:
            L.append("- directories absent: " + ", ".join(SL["missing_dirs"]))
        L += table(["directory", "tag", "rung", "kind", "clocks (count)", "bytes (sum)"],
                   [[g["dir"], g["tag"], g["rung"] or "-", g["kind"], "%s (%d)" % (ranges_text(g["clocks"]), len(g["clocks"])), g["bytes"]] for g in SL["groups"]])
        L.append("")
        lo, hi = cfg["integer_clocks"]
        for s in (2, 4, 8):
            p = SL["per_rung"][s]
            f = SL["records_flags"][s]
            L.append("- rung %d: integer clocks %d to %d on the box: %s; non-integer clocks of the history's tags: %s; half-clock states (bounce): %s; "
                     "marked saved by the records but absent on the box: %s; on the box but not marked saved: %s" % (
                         s, lo, hi, "all" if not p["integer_missing"] else "missing " + ranges_text(p["integer_missing"]),
                         ranges_text(p["non_integer"]) if p["non_integer"] else "none", ranges_text(p["half_clock_states"]) if p["half_clock_states"] else "none",
                         ranges_text(f["marked_not_on_box"]) if f["marked_not_on_box"] else "none", ranges_text(f["on_box_not_marked"]) if f["on_box_not_marked"] else "none"))
        if SL["duplicates"]:
            L += ["", "Clocks held twice (one tag and clock in two directories):", ""]
            L += table(["tag", "clock", "paths", "bytes", "md5 equal"], [[d["tag"], fmt(d["clock"]), "; ".join(d["paths"]), ", ".join(str(b) for b in d["bytes"]),
                                                                       fmt(d["md5_equal"]) if d["md5_equal"] is not None else "not computed"] for d in SL["duplicates"]])
        else:
            L.append("- no clock is held twice")
        L.append("")
    else:
        L += ["- the slice inventory stage failed or was skipped", ""]
    A = S.get("at_slice_clocks")
    if A:
        L += ["### The three rungs' records at the slice clocks (R_t or none; min(N - abs v); E/E0; pockets; converged; slice on the box)", ""]
        L += table(A["headers"], A["rows"])
        L.append("")
    L += ["### The half-clock states: the bounce replays (R2) against the rungs' first runs at twice the floor (run = the replay, ref = the first run); "
          "stored states only, their contents not read", ""]
    for role in ("bounce4", "bounce8"):
        B = (S.get("bounces") or {}).get(role)
        L += ["#### %s" % LABEL[role], ""]
        if not B or B.get("error"):
            L += ["- compare failed or absent: %s" % ((B or {}).get("error") or "no compare"), ""]
            continue
        L += ["- resume slice clock %s; common records %d (%s to %s); carried rows %d (nonzero %d); computed rows %d; every common record within twice the floor: %s; "
              "first not within: %s; over the computed rows the largest |E/E0 difference| %s, |min difference| %s, |root difference| %s; files %s, %s" % (
                  fmt(B["resume_clock"]), B["common"], fmt(B["first"], "%.4f"), fmt(B["last"], "%.4f"), B["carried"], B["carried_nonzero_count"], B["computed"],
                  fmt(B["all_within"]), fmt(B["first_not_within"], "%.4f"),
                  "%+.2e at %.4f" % B["max_abs_dE_computed"] if B["max_abs_dE_computed"] else "-",
                  "%+.2e at %.4f" % B["max_abs_dmin_computed"] if B["max_abs_dmin_computed"] else "-",
                  "%+.2e at %.4f (root %d)" % B["max_abs_droot_computed"] if B["max_abs_droot_computed"] else "-",
                  os.path.basename(B["md"]), os.path.basename(B["json"])), ""]
        L += table(CMP_HDR, [cmp_row(x) for x in B["rows_shown"]])
        L.append("")
    # 6
    L += ["## 6 The comparator: the composite history against rung 2", ""]
    comp = next((m for m in members if m["role"] == "composite"), None)
    if comp:
        L += ["### 6a The composite history as its records state it", "",
              "- json %s; records %d (%s to %s); script field %s; mix %s; nk entries %s; resume slice %s (clock %s); resume run %s" % (
                  comp["path"], comp["records"], fmt(comp["first"]), fmt(comp["last"]), comp["script"], comp["args"].get("mix"), short(comp["args_solver"], 160),
                  short(comp["resume_slice"], 120), fmt(comp["resume_clock"]), short(comp["resume_run"], 120)),
              "- engine per clock: %s" % rle_text(comp["engine_runs"], 10), "- v36 entry per clock: %s" % rle_text(comp["v36_runs"], 10),
              "- continuation labels: %s" % rle_text(comp["continuation_runs"], 12), "- record labels: %s" % rle_text(comp["record_label_runs"], 8),
              "- the inputs' clocks of the segments: the v32 member to 16.0, v36cc to 17.0, the v36d relaunch from 17.125, the set flip at 17.625 (50.2, 54.1); "
              "where the records' fields do not resolve a boundary the runs above say so", ""]
    C = S.get("comparator")
    L += ["### 6b Rung 2 against the composite at twice the floor (run = rung 2, ref = the composite; differences = rung 2 minus composite; 55.1's direction)", ""]
    if not C or C.get("error"):
        L += ["- compare failed or absent: %s" % ((C or {}).get("error") or "no compare"), ""]
    else:
        L += ["- common records %d (%s to %s); within %d, not within %d; first not within %s (the composite's segment there: %s); clocks where the root counts "
              "differ (no root difference formed; not within by the tool's rule): %s" % (
                  C["common"], fmt(C["first"], "%.4f"), fmt(C["last"], "%.4f"), C["n_within"], C["n_not_within"], fmt(C["first_not_within"], "%.4f"),
                  C["segment_at_first_not_within"] or "-", ", ".join("%.4f" % t for t in C["count_mismatch_clocks"]) or "none"),
              "- over all common records: largest |E/E0 difference| %s, |min difference| %s, |root difference| %s" % (
                  "%+.2e at %.4f" % C["max_abs_dE"] if C["max_abs_dE"] else "-", "%+.2e at %.4f" % C["max_abs_dmin"] if C["max_abs_dmin"] else "-",
                  "%+.2e at %.4f (root %d)" % C["max_abs_droot"] if C["max_abs_droot"] else "-")]
        for w, v in C["windows"].items():
            L.append("- window %s: records %d, within %d, every one within %s, first not within %s, largest |E/E0 difference| %s, |min difference| %s, |root difference| %s" % (
                w, v["n"], v["n_within"], fmt(v["all_within"]), fmt(v["first_not_within"], "%.4f"),
                "%+.2e at %.4f" % v["max_abs_dE"] if v["max_abs_dE"] else "-", "%+.2e at %.4f" % v["max_abs_dmin"] if v["max_abs_dmin"] else "-",
                "%+.2e at %.4f (root %d)" % v["max_abs_droot"] if v["max_abs_droot"] else "-"))
        L += ["- the window 17.125 to 18.125 is 55.1's restart rule's; its consequence is the lead's. A difference beyond the floor reads as the composite "
              "history's propagated term (55.1); the single-version history has no seam to attribute.",
              "- rows: every %g r_s, every record in the windows, the formation clocks, every change of verdict; all rows in %s" % (cfg["every"], os.path.basename(C["md"])), ""]
        L += table(CMP_HDR + ["composite segment (engine)", "composite label", "reading"],
                   [cmp_row(x, [x.get("composite_engine") or "-", x.get("composite_label") or "-", x["reading"]]) for x in C["rows_shown"]])
        L.append("")
    M = S.get("composite_vs_member")
    L += ["### 6c The composite against the v32 member (run = the composite, ref = the member; version 1.7's direction)", ""]
    if not M:
        L += ["- the member's json is not on the box (or the compare was skipped); version 1.7's seam rows stand as printed", ""]
    elif M.get("error"):
        L += ["- compare failed: %s" % M["error"], ""]
    else:
        L += ["- common records %d (%s to %s); rows at clocks up to 16.0 are the member's records carried into the composite: with any nonzero difference %d%s; "
              "every common record within twice the floor: %s; first not within: %s; files %s, %s" % (
                  M["common"], fmt(M["first"], "%.4f"), fmt(M["last"], "%.4f"), M["carried_nonzero_count"],
                  (" (at " + ", ".join("%.4f" % t for t in M["carried_nonzero_clocks"]) + ")") if M["carried_nonzero_clocks"] else "",
                  fmt(M["all_within"]), fmt(M["first_not_within"], "%.4f"), os.path.basename(M["md"]), os.path.basename(M["json"])), ""]
        v17 = S.get("v17") or {}
        L += table(CMP_HDR + ["version 1.7's reading at this clock"],
                   [cmp_row(x, [(v17.get("seam", {}).get(round(x["T_over_rs"], 6)) or {}).get("reading", "-")]) for x in M["rows_shown"]])
        L.append("")
    RS = S.get("record_step")
    L += ["### 6d The record-step table: the flip at 17.625 (the label and the wording are the review of version 1.7's)", ""]
    if not RS or not RS.get("rows"):
        L += ["- version 1.7's tables were not parsed (file absent or no rows): %s" % ((RS or {}).get("error") or "-"), ""]
    else:
        L += ["- parsed from version 1.7: %d flip rows, %d seam rows; the point-minus-member column's source: %s" % (
                  RS["flip_rows_parsed"], RS["seam_rows_parsed"], RS["rows"][0]["point_minus_member"]["source"]),
              "- the stopped run against the member (carried): largest |E/E0 difference| %+.2e at T/r_s %.4f; at %.4f: E/E0 %+.2e, roots (inner, outer) %s lattice units" % (
                  RS["stopped_minus_member_max_abs_dE"][0], RS["stopped_minus_member_max_abs_dE"][1], RS["stopped_minus_member_at_last_clock"]["T_over_rs"],
                  RS["stopped_minus_member_at_last_clock"]["dE"], ", ".join("%+.2e" % d for d in (RS["stopped_minus_member_at_last_clock"]["droots"] or [])) or "-"),
              "- derived columns: stopped minus composite = (stopped minus member) minus (point minus member), the composite's record at these clocks being the point's; "
              "stopped minus rung 2 = (stopped minus composite) minus (rung 2 minus composite); their precision is that of the printed digits where those are the source", ""]

        def trip(g):
            return "%s; %s; %s" % (fmt(g.get("dE"), "%+.2e"), fmt(g.get("dmin"), "%+.2e"), ", ".join("%+.2e" % d for d in g["droots"]) if g.get("droots") else "-")
        L += table(["T/r_s", "stopped - member: E/E0; min; roots (inner, outer) [version 1.7, carried]", "masks (version 1.7)", "point - member [source above]",
                    "stopped - composite [derived]", "rung 2 - composite [6b, computed]", "stopped - rung 2 [derived]"],
                   [[fmt(r["T_over_rs"], "%.4f"), trip(r["stopped_minus_member"]), r["stopped_minus_member"].get("masks") or "-", trip(r["point_minus_member"]),
                     trip(r["stopped_minus_composite_derived"]), trip(r["rung2_minus_composite"]), trip(r["stopped_minus_rung2_derived"])] for r in RS["rows"]])
        L.append("")
    H = S.get("hreduce")
    if H:
        L += ["### 6e hreduce's bound beside (conditional; it bounds the composite history's seam and flip, not the single-version history)", ""]
        L += ["- " + s for s in H["inequalities"]]
        L += ["", "Premise by premise, what the inputs supply:", ""]
        L += table(["premise", "status in the inputs"], [list(c) for c in H["conditions"]])
        L += ["", "- k = %d; the recorded gap %.2e; Delta %.3f r_s%s (derived); the filed tests %.1e in E/E0 and %.1e in the outer root (lattice units), tests of (2) and not inputs" % (
                  H["k"], H["gap_recorded"], H["delta_rs_derived"], (", %g a" % H["delta_a_derived"]) if H["delta_a_derived"] else "", H["filed_tests"]["E_over_E0"], H["filed_tests"]["outer_root_a"]),
              "- the derived stopped-minus-point values after t_b (17.75 to 18.125), the bound's tests: " + ("; ".join(
                  "%.4f: E/E0 %s, outer root %s a" % (t["T_over_rs"], fmt(t["stopped_minus_point_dE_derived"], "%+.2e"), fmt(t["stopped_minus_point_outer_root_a_derived"], "%+.2e"))
                  for t in H["tests_derived_after_t_b"]) or "none (version 1.7's rows not parsed)"),
              "- numerical bound: " + H["numerical_bound"], ""]
    # 7
    L += ["## 7 Provenance", ""]
    L += ["- stages: " + "; ".join("%s %.1f s%s" % (s["stage"], s["seconds"], "" if s["ok"] else " FAILED") for s in S["stages"])]
    for k, v in (S.get("commands") or {}).items():
        L.append("- %s: %s (returncode %s, %.1f s)" % (k, v["command"], v["returncode"], v["seconds"]))
    L += ["- files: " + "; ".join("%s (%d bytes)" % (os.path.basename(p), os.path.getsize(p)) for p in S["files"] if os.path.exists(p)),
          "- elapsed %.1f s; no date is written" % S["elapsed_seconds"]]
    return "\n".join(L) + "\n"


def at_slice_clocks(runs, SL, lo, hi):
    byT = {role: {round(r["T_over_rs"], 6): r for r in runs[role]["recs"]} for role in HISTORY_ROLES if runs.get(role)}
    headers = ["T/r_s"]
    for role in HISTORY_ROLES:
        s = SIGMA[role]
        headers += ["R_t (s %d)" % s, "min(N-abs v)", "E/E0", "pockets", "converged", "slice on box"]
    rows = []
    for k in range(0, hi + 1):
        row = ["%d" % k]
        for role in HISTORY_ROLES:
            r = byT.get(role, {}).get(float(k))
            box = SL and (float(k) in set(SL["per_rung"][SIGMA[role]]["clocks"]))
            if r is None:
                row += ["-"] * 5 + [fmt(bool(box))]
                continue
            row += [fmt(r["R_t"]) if r["trapping_zeros"] else "none", fmt(r["min_N_minus_absv"], "%.4f"), fmt(r["E_over_E0"], "%.7f"), r["n_pocket"],
                    fmt(r["converged"]), fmt(bool(box))]
        rows.append(row)
    return {"headers": headers, "rows": rows}


# ----------------------------------------------------------------------------------------------------------------------- the pipeline
def pipeline(cfg, say=print):
    t_start = time.time()
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    scratch = os.path.join(out, "scratch_links")
    S = {"script": "compose_ladder_v2_v1.py", "out_dir": out, "floor": FLOOR, "twice_floor": TWICE, "config": cfg, "stages": [], "errors": [],
         "files": [], "commands": {}}

    def stage(name, fn):
        t0 = time.time()
        try:
            r = fn()
            S["stages"].append({"stage": name, "seconds": round(time.time() - t0, 1), "ok": True})
            say("stage %s: %.1f s" % (name, time.time() - t0))
            return r
        except Exception as e:
            S["stages"].append({"stage": name, "seconds": round(time.time() - t0, 1), "ok": False, "error": repr(e)})
            S["errors"].append({"stage": name, "error": repr(e)})
            say("stage %s FAILED: %r" % (name, e))
            return None

    runs = {}
    for role in ROLES:
        p = cfg["inputs"].get(role)
        if p:
            runs[role] = stage("load " + role, lambda p=p: load_slim(p, md5=cfg["md5"]))
    extra = []
    for lab, p in cfg.get("extra_members") or []:
        r = stage("load extra " + lab, lambda p=p: load_slim(p, md5=cfg["md5"]))
        if r:
            extra.append((lab, r))
    S["members"] = [member_summary(LABEL[role], role, runs[role]) for role in ROLES if runs.get(role)]
    S["extra_members"] = [member_summary(lab, "extra", r) for lab, r in extra]
    rs_over_a = None
    for role in HISTORY_ROLES:
        if runs.get(role) and runs[role]["meta"]["args"].get("rs"):
            rs_over_a = float(runs[role]["meta"]["args"]["rs"])
            break
    # compare_v5 on the three rungs
    if all(runs.get(r) for r in HISTORY_ROLES):
        S["cv5"] = stage("compare_v5", lambda: run_cv5([("2", cfg["inputs"]["rung2"]), ("4", cfg["inputs"]["rung4"]), ("8", cfg["inputs"]["rung8"])],
                                                       os.path.join(out, "ladder_v2_cmp_v5.md"),
                                                       "the ladder on one engine version at r_s/a 64 (the three v36d_from0 rungs; sigma/a 4 and 8 as their _to40 jsons)",
                                                       cfg["every"]))
        if S["cv5"]:
            S["commands"]["compare_v5"] = {k: S["cv5"][k] for k in ("command", "returncode", "seconds")}
            S["files"].append(S["cv5"]["md"])
    # the pairs at twice the floor
    pairs = [("resume_ell4", "rung4", "rung4_first", "the sigma/a 4 continuation (_to40)", "the sigma/a 4 first run"),
             ("resume_ell8", "rung8", "rung8_first", "the sigma/a 8 continuation (_to40)", "the sigma/a 8 first run"),
             ("bounce_ell4", "bounce4", "rung4_first", "the rung 4 bounce replay (R2)", "the sigma/a 4 first run"),
             ("bounce_ell8", "bounce8", "rung8_first", "the rung 8 bounce replay (R2)", "the sigma/a 8 first run"),
             ("rung2_vs_composite", "rung2", "composite", "rung 2 (v36d from T = 0)", "the composite history (v36cc then v36d)")]
    cmps = {}
    for name, a, b, la, lb in pairs:
        if runs.get(a) and runs.get(b):
            cmps[name] = stage("compare " + name, lambda a=a, b=b, la=la, lb=lb, name=name: run_cmf(
                cfg["inputs"][a], cfg["inputs"][b], os.path.join(out, "cmp_" + name), la, lb, scratch))
    if cfg.get("member_json") and os.path.exists(cfg["member_json"]) and runs.get("composite"):
        cmps["composite_vs_member"] = stage("compare composite_vs_member", lambda: run_cmf(
            cfg["inputs"]["composite"], cfg["member_json"], os.path.join(out, "cmp_composite_vs_member"), "the composite history", "the v32 member", scratch))
    for rung, gpath in (cfg.get("gate18") or {}).items():
        ref = cfg["inputs"].get(rung + "_first")
        if ref and os.path.exists(gpath):
            cmps["gate18_" + rung] = stage("compare gate18 " + rung, lambda gpath=gpath, ref=ref, rung=rung: run_cmf(
                gpath, ref, os.path.join(out, "cmp_gate18_" + rung), "the resume-gate run of " + rung, "the first run", scratch))
    for name, c in cmps.items():
        if c:
            S["commands"][name] = {k: c[k] for k in ("command", "returncode", "seconds")}
            S["files"] += [c["md"], c["json"]]
    # readings
    S["resumes"] = {}
    for name, role in (("resume_ell4", "rung4"), ("resume_ell8", "rung8")):
        if cmps.get(name):
            rc = clock_of(runs[role]["meta"]["resume_slice"])
            S["resumes"][role] = stage("read " + name, lambda c=cmps[name], rc=rc: pair_reading(c, rc, "resume"))
    S["bounces"] = {}
    for name, role in (("bounce_ell4", "bounce4"), ("bounce_ell8", "bounce8")):
        if cmps.get(name):
            rc = clock_of(runs[role]["meta"]["resume_slice"])
            S["bounces"][role] = stage("read " + name, lambda c=cmps[name], rc=rc: pair_reading(c, rc, "bounce"))
    S["gates"] = {}
    for name in [k for k in cmps if k.startswith("gate18_")]:
        rc = clock_of(find_key_path(cmps[name]["run_json"])) if cmps[name] else None
        S["gates"][name] = stage("read " + name, lambda c=cmps[name], rc=rc: pair_reading(c, rc, "resume"))
    if cmps.get("rung2_vs_composite") and runs.get("composite"):
        form = [m["episodes"][0]["first"] for m in S["members"] if m["role"] in ("rung2", "composite") and m["episodes"]]
        S["comparator"] = stage("read comparator", lambda: comparator_reading(
            cmps["rung2_vs_composite"], runs["composite"]["recs"], [cfg["restart_window"], cfg["seam_window"]], cfg["every"], form))
    if cmps.get("composite_vs_member"):
        def read_m():
            R = pair_reading(cmps["composite_vs_member"], 16.0, "resume")
            if not R.get("error"):
                a, b = cfg["seam_window"]
                R["rows_shown"] = [x for x in R["rows_all"] if a - 1e-9 <= x["T_over_rs"] <= b + 1e-9 or not x["within"] or x["T_over_rs"] in R["carried_nonzero_clocks"]]
            return R
        S["composite_vs_member"] = stage("read composite_vs_member", read_m)
    if cfg.get("v17_md") and os.path.exists(cfg["v17_md"]):
        S["v17"] = stage("parse version 1.7", lambda: parse_v17(open(cfg["v17_md"]).read()))
    else:
        S["v17"] = None
    if S.get("v17") and cfg.get("cmp_v36d_json") and os.path.exists(cfg["cmp_v36d_json"]) and not S.get("composite_vs_member"):
        def read_cmp_json():
            d = json.load(open(cfg["cmp_v36d_json"]))
            return {"rows_all": [slim_row(x) for x in d.get("rows", [])], "source": cfg["cmp_v36d_json"]}
        S["cmp_v36d_json_rows"] = stage("read cmp_v36d_vs_member.json", read_cmp_json)
    member_rows = (S.get("composite_vs_member") or {}).get("rows_all") or (S.get("cmp_v36d_json_rows") or {}).get("rows_all")
    rung_rows = (S.get("comparator") or {}).get("rows_all")
    S["record_step"] = stage("record-step table", lambda: record_step(S.get("v17"), member_rows, rung_rows, cfg["flip_clocks"]))
    S["hreduce"] = hreduce_block(rs_over_a, S.get("record_step") or {})
    S["slices"] = stage("slice inventory", lambda: slice_inventory(cfg["slice_dirs"], S["members"], cfg["integer_clocks"], cfg["md5_duplicates"]))
    S["at_slice_clocks"] = stage("records at the slice clocks", lambda: at_slice_clocks(runs, S.get("slices"), *cfg["integer_clocks"]))
    S["elapsed_seconds"] = round(time.time() - t_start, 1)
    md_path, js_path = os.path.join(out, "COLLAPSE_ladder_v2_tables.md"), os.path.join(out, "COLLAPSE_ladder_v2_tables.json")
    S["files"] += [md_path, js_path]
    md = compose_md(S, cfg)
    S["elapsed_seconds"] = round(time.time() - t_start, 1)
    with open(md_path, "w") as f:
        f.write(md)
    slim = {k: v for k, v in S.items()}
    with open(js_path, "w") as f:
        json.dump(slim, f, indent=1, default=str)
    return S, md_path, js_path


def find_key_path(p):
    try:
        return find_key(json.load(open(p)), "resume_slice")
    except Exception:
        return None


def summary_lines(S):
    L = []
    for m in S.get("members") or []:
        if m["role"] in HISTORY_ROLES + ("composite",):
            rb = m["residual"]
            L.append("%s: %d records %s to %s; converged %d/%d/%d; residual max %s at %s; above 1e-3 first at %s; episodes %d%s; engine %s" % (
                m["label"], m["records"], fmt(m["first"]), fmt(m["last"]), rb["converged_true"], rb["converged_false"], rb["converged_absent"],
                fmt(rb["max"], "%.2e"), fmt(rb["max_at"]), fmt(rb["above_1e-3_first"]) if rb["above_1e-3_first"] is not None else "no record", len(m["episodes"]),
                (" (first %s to %s, closes %s)" % (fmt(m["episodes"][0]["first"]), fmt(m["episodes"][0]["last"]), fmt(m["episodes"][0]["closes_at"]))) if m["episodes"] else "",
                rle_text(m["engine_runs"], 3)))
    for role, R in (S.get("resumes") or {}).items():
        if not R.get("error"):
            rr = R["resumed_record"] or {}
            L.append("resume %s: carried nonzero %d; resumed record %s dE %s dmin %s within %s; all within %s; first not within %s" % (
                role, R["carried_nonzero_count"], fmt(rr.get("T_over_rs"), "%.4f"), fmt(rr.get("dE"), "%+.2e"), fmt(rr.get("dmin"), "%+.2e"), fmt(rr.get("within")),
                fmt(R["all_within"]), fmt(R["first_not_within"], "%.4f")))
    for role, B in (S.get("bounces") or {}).items():
        if not B.get("error"):
            L.append("bounce %s: common %d, all within %s, first not within %s" % (role, B["common"], fmt(B["all_within"]), fmt(B["first_not_within"], "%.4f")))
    C = S.get("comparator")
    if C and not C.get("error"):
        L.append("comparator: common %d, within %d, first not within %s (segment %s)" % (C["common"], C["n_within"], fmt(C["first_not_within"], "%.4f"), C["segment_at_first_not_within"]))
        for w, v in C["windows"].items():
            L.append("  window %s: all within %s, first not within %s, max |dE| %s" % (w, fmt(v["all_within"]), fmt(v["first_not_within"], "%.4f"),
                                                                                       "%+.2e at %.4f" % v["max_abs_dE"] if v["max_abs_dE"] else "-"))
    RS = S.get("record_step")
    if RS and RS.get("rows"):
        L.append("record-step: stopped - member max |dE| %+.2e at %.4f; rows %d" % (RS["stopped_minus_member_max_abs_dE"][0], RS["stopped_minus_member_max_abs_dE"][1], len(RS["rows"])))
    if S.get("slices"):
        for s in (2, 4, 8):
            p = S["slices"]["per_rung"][s]
            L.append("slices rung %d: integer clocks missing %s; half clocks %s" % (s, ranges_text(p["integer_missing"]) if p["integer_missing"] else "none",
                                                                                     ranges_text(p["half_clock_states"]) if p["half_clock_states"] else "none"))
    L.append("errors: %d; elapsed %.1f s" % (len(S["errors"]), S["elapsed_seconds"]))
    return L


# ----------------------------------------------------------------------------------------------------------------------- the self-test
def _fake_record(i, ell, engine="fake_v36d", dE=0.0, zeros=None, newton=False, cont=None):
    t = i * 0.125
    if zeros is None:
        zeros = [5.0 + 0.01 * i, 40.0 - 0.02 * i] if (101 <= i <= 128 or i >= 144) else []
    r = {"T_over_rs": t, "T": t * 64.0, "E": 100.0 - 0.01 * i, "E_over_E0": 1.0 - 1e-4 * i + dE, "min_N_minus_absv": 0.5 - 0.005 * i,
         "R_t": zeros[-1] if zeros else None, "R_t_over_2gE": (zeros[-1] / 50.0) if zeros else None, "R_c": 100.0 - 0.5 * i, "trapping_zeros": zeros,
         "n_pocket": {"1": 2 if 105 <= i <= 110 else 0}, "converged": i != 5, "iteration_history": [[0, 1e-5, 2e-5], [1, 3e-7, 2e-7]], "iterations": 2,
         "seconds": 10, "engine": engine, "v36": {"version": "d"}, "solver": "plain", "slice_saved": bool(i and i % 8 == 0),
         "surface_jets": {"d_N_plus_v": [0.01, 0.0]} if zeros else {}, "label": "fake ell %d" % ell}
    if i == 5:
        r["unconverged"] = {"residual": 2e-3}
    if newton:
        r["solve_v32"] = {"kept_residual": 1e-8, "confirming_residual": 2e-8, "evaluations": 40, "meets_stop_rule": True, "kept_from": "direct",
                          "confirming_dE": 1e-10, "continuation": "not needed"}
    if cont:
        r["continuation_label"] = cont
    return r


def _fake_v17_text():
    L = ["# fake version 1.7", "", "## The point's seam (fake)", "",
         "| T/r_s | E/E0 difference | min(N - abs(v)) difference | roots (inner, outer) run | root differences | masks equal | within | kept residual run; member | reading |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in (17.625, 17.75, 17.875, 18.0, 18.125):
        L.append("| %.4f | -1.00e-08 | +2.00e-08 | 2.000000, 33.000000 | +1.00e-07, -2.00e-06 | 10 of 10 | yes | 1.0e-08; 2.0e-08 | within |" % t)
    L += ["", "## The flip's record-step term: the stopped run (the start's-set branch from 17.625) against the member", "",
          "| T/r_s | E/E0 difference | min(N - abs(v)) difference | roots (inner, outer) stopped run | root differences | masks equal | kept residual stopped run; member |",
          "|---|---|---|---|---|---|---|"]
    for t, d in zip((17.625, 17.75, 17.875, 18.0, 18.125), ("-1.00e-06", "-9.00e-05", "-8.00e-05", "-7.00e-05", "-7.50e-05")):
        L.append("| %.4f | %s | -5.00e-03 | 2.003333, 33.208156 | +5.00e-03, -1.50e-02 | 10 of 10 | 1.0e-07; 4.1e-07 |" % (t, d))
    L += ["", "## Version 1.6 (the record, unchanged)", "", "| 11.5000 | none | - |"]
    return "\n".join(L) + "\n"


def make_fakes(tmp):
    def write(name, args, head, records):
        p = os.path.join(tmp, "collapse_%s.json" % name)
        with open(p, "w") as f:
            json.dump({"args": args, "head": head, "records": records, "script": "engine_collapse_v34.py"}, f)
        return p
    a4 = {"ell": 4, "tau": 8.0, "rs": 64.0, "nsites": 2048, "mix": "plain", "T_over_rs": 18.125, "kappa_sea": 2}
    a8 = dict(a4, ell=8)
    r4f = write("fake_ell4_from0", dict(a4), {"E0": 118.0}, [_fake_record(i, 4) for i in range(146)])
    recs = [_fake_record(i, 4) for i in range(153)]
    recs[145]["E_over_E0"] += 1e-9
    r4 = write("fake_ell4_from0_to40", dict(a4, T_over_rs=19.0), {"E0": 118.0, "resume_slice": "/fake/orbitals_fake_ell4_from0_T0018.00.npz", "resume_run": "/fake/run4"}, recs)
    r8f = write("fake_ell8_from0", dict(a8), {"E0": 118.0}, [_fake_record(i, 8) for i in range(146)])
    recs = [_fake_record(i, 8) for i in range(153)]
    recs[80]["E_over_E0"] += 1e-10
    recs[145]["E_over_E0"] += 5e-7
    r8 = write("fake_ell8_from0_to40", dict(a8, T_over_rs=19.0), {"resume_slice": "/fake/orbitals_fake_ell8_from0_T0018.00.npz", "resume_run": "/fake/run8"}, recs)
    b4 = write("fake_ell4_from0_bounce", dict(a4, T_over_rs=16.0), {"resume_slice": "/fake/orbitals_fake_ell4_from0_T0011.00.npz"}, [_fake_record(i, 4) for i in range(129)])
    b8 = write("fake_ell8_from0_bounce", dict(a8, T_over_rs=16.0), {"resume_slice": "/fake/orbitals_fake_ell8_from0_T0011.00.npz"}, [_fake_record(i, 8) for i in range(129)])
    recs = [_fake_record(i, 2) for i in range(153)]
    recs[150]["trapping_zeros"] = [3.0, 20.0, 37.0]
    r2 = write("fake_ell2_from0", dict(a4, ell=2, T_over_rs=19.0), {"E0": 118.0}, recs)
    comp = []
    for i in range(153):
        eng = "fake_v32" if i <= 128 else ("fake_v36cc" if i <= 136 else "fake_v36d")
        dE = 0.0 if i <= 128 else (1e-8 if i <= 136 else min(1e-7 * 2 ** (i - 137), 1e-3))
        comp.append(_fake_record(i, 2, engine=eng, dE=-dE, newton=True, cont="the record on the previous record's set" if i == 141 else None))
    cp = write("fake_ell2_composite", dict(a4, ell=2, mix="newton", nk_krylov=40, nk_restarts=2, T_over_rs=19.0),
               {"resume_slice": "/fake/orbitals_fake_cc_T0017.00.npz", "resume_run": "/fake/cc"}, comp)
    dirs = [os.path.join(tmp, d) for d in ("sl_a", "sl_b", "sl_c")]
    names = (["orbitals_fake_ell4_from0_T0001.00.npz", "orbitals_fake_ell4_from0_T0018.00.npz"],
             ["orbitals_fake_ell4_from0_T0018.00.npz", "orbitals_fake_ell4_from0_to40_T0019.00.npz"], ["orbitals_fake_ell4_from0_bounce_T0011.50.npz"])
    for d, ns in zip(dirs, names):
        os.makedirs(d)
        for n in ns:
            open(os.path.join(d, n), "wb").close()
    v17p = os.path.join(tmp, "fake_v1.7.md")
    with open(v17p, "w") as f:
        f.write(_fake_v17_text())
    cfg = default_cfg()
    cfg.update({"inputs": {"rung2": r2, "rung4_first": r4f, "rung4": r4, "rung8_first": r8f, "rung8": r8, "composite": cp, "bounce4": b4, "bounce8": b8},
                "slice_dirs": dirs, "member_json": None, "v17_md": v17p, "cmp_v36d_json": None, "out_dir": os.path.join(tmp, "out"), "md5": True,
                "md5_duplicates": True, "integer_clocks": (1, 19)})
    return cfg


def self_test(keep=False):
    tmp = tempfile.mkdtemp(prefix="ladder_v2_selftest_")
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
    try:
        cfg = make_fakes(tmp)
        S, md_path, js_path = pipeline(cfg, say=lambda *a: None)
        check("eight jsons loaded", len(S["members"]) == 8)
        check("clock parse", clock_of("/x/orbitals_a_T0018.00.npz") == 18.0 and clock_of("/x/orbitals_a_T0011.50.npz") == 11.5)
        check("ranges text", ranges_text([1, 2, 3, 5]) == "1 to 3, 5")
        m4 = next(m for m in S["members"] if m["role"] == "rung4")
        eps = m4["episodes"]
        check("episodes planted", len(eps) == 2 and abs(eps[0]["first"] - 12.625) < 1e-9 and abs(eps[0]["last"] - 16.0) < 1e-9
              and abs(eps[0]["closes_at"] - 16.125) < 1e-9 and abs(eps[1]["first"] - 18.0) < 1e-9 and eps[1]["closes_at"] is None)
        check("episode rule stated", m4["episode_rule"].startswith("a record holds a trapping zero"))
        rb = m4["residual"]
        check("residual bands planted", rb["with_residual"] == 153 and rb["below_1e-4"] == 152 and rb["band_1e-4_1e-3"] == 0
              and abs(rb["above_1e-3_first"] - 0.625) < 1e-9 and rb["converged_false"] == 1)
        check("resume clock read from the head", abs(m4["resume_clock"] - 18.0) < 1e-9)
        R4 = S["resumes"].get("rung4") or {}
        check("rung 4 resume: carried rows copies", not R4.get("error") and R4["carried_nonzero_count"] == 0 and R4["carried"] == 145 and R4["computed"] == 1)
        rr = (R4.get("resumed_record") or {})
        check("rung 4 resume: resumed record within", abs((rr.get("T_over_rs") or 0) - 18.125) < 1e-9 and rr.get("within") is True and 0 < (rr.get("dE") or 0) < 3e-7)
        check("rung 4 resume: all within", R4.get("all_within") is True)
        R8 = S["resumes"].get("rung8") or {}
        check("rung 8 resume: carried anomaly flagged", not R8.get("error") and R8["carried_nonzero_clocks"] == [10.0])
        check("rung 8 resume: planted beyond-floor record not within", R8.get("all_within") is False and abs((R8.get("first_not_within") or 0) - 18.125) < 1e-9)
        B4 = S["bounces"].get("bounce4") or {}
        check("bounce: carried and computed rows", not B4.get("error") and abs(B4["resume_clock"] - 11.0) < 1e-9 and B4["computed"] == 40 and B4["all_within"] is True)
        check("bounce: half clocks shown", any(abs(x["T_over_rs"] - 13.5) < 1e-9 for x in B4.get("rows_shown", [])))
        C = S.get("comparator") or {}
        check("comparator: first not within planted", not C.get("error") and abs((C.get("first_not_within") or 0) - 17.375) < 1e-9)
        check("comparator: segment at the first not within", C.get("segment_at_first_not_within") == "fake_v36d")
        w = (C.get("windows") or {}).get("17.125 to 18.125") or {}
        check("comparator: restart window verdict", w.get("all_within") is False and abs((w.get("first_not_within") or 0) - 17.375) < 1e-9 and w.get("n") == 9)
        check("comparator: root-count mismatch clock", C.get("count_mismatch_clocks") == [18.75])
        comp = next(m for m in S["members"] if m["role"] == "composite")
        segs = [(u["value"], u["first"], u["last"]) for u in comp["engine_runs"]]
        check("composite segments from the records", segs == [("fake_v32", 0.0, 16.0), ("fake_v36cc", 16.125, 17.0), ("fake_v36d", 17.125, 19.0)])
        check("composite flip label run", any(u["value"] == "the record on the previous record's set" and abs(u["first"] - 17.625) < 1e-9 for u in comp["continuation_runs"]))
        RS = S.get("record_step") or {}
        mx = RS.get("stopped_minus_member_max_abs_dE")
        check("version 1.7 parsed: 5 flip rows, 5 seam rows", RS.get("flip_rows_parsed") == 5 and RS.get("seam_rows_parsed") == 5)
        check("record-step: maximum at 17.75", mx is not None and abs(mx[0] + 9.0e-05) < 1e-12 and abs(mx[1] - 17.75) < 1e-9)
        last = [r for r in RS.get("rows", []) if abs(r["T_over_rs"] - 18.125) < 1e-9]
        sp = last[0]["stopped_minus_composite_derived"] if last else {}
        sr = last[0]["stopped_minus_rung2_derived"] if last else {}
        check("record-step: derived stopped - composite", last and abs(sp["dE"] - (-7.5e-05 + 1.0e-08)) < 1e-12 and abs(sp["droots"][1] - (-1.5e-02 + 2.0e-06)) < 1e-12)
        check("record-step: derived stopped - rung 2", last and abs(sr["dE"] - (-7.5e-05 + 1.0e-08 - 2.56e-05)) < 1e-10)
        H = S.get("hreduce") or {}
        check("hreduce: ten premises, no numerical bound", len(H.get("conditions", [])) == 10 and H.get("numerical_bound", "").startswith("none computed")
              and abs((H.get("delta_a_derived") or 0) - 8.0) < 1e-9 and len(H.get("tests_derived_after_t_b", [])) == 4)
        SL = S.get("slices") or {}
        per4 = (SL.get("per_rung") or {}).get(4) or {}
        check("slices: clocks and half clocks", per4.get("clocks") == [1.0, 18.0, 19.0] and per4.get("half_clock_states") == [11.5] and 2 in per4.get("integer_missing", [])
              and 18 not in per4.get("integer_missing", []))
        d = (SL.get("duplicates") or [{}])[0]
        check("slices: duplicate found and md5 compared", d.get("tag") == "fake_ell4_from0" and d.get("clock") == 18.0 and d.get("md5_equal") is True)
        check("slices: records' flags against the box", 2.0 in (SL.get("records_flags") or {}).get(4, {}).get("marked_not_on_box", []))
        cv5 = S.get("cv5") or {}
        check("compare_v5 ran on the fakes", "formation spread" in cv5.get("text", "") and "29.1" in cv5.get("text", ""))
        txt = open(md_path).read() if os.path.exists(md_path) else ""
        check("tables md written with its sections", all(h in txt for h in ("## 1 ", "## 2 ", "## 3 ", "## 4 ", "## 5 ", "## 6 ", "## 7 ")) and os.path.exists(js_path))
        check("no stage failed", not S["errors"])
        for n, ok in checks:
            print("%s  %s" % ("PASS" if ok else "FAIL", n))
        if S["errors"]:
            print("stage errors: %s" % S["errors"])
        failed = [n for n, ok in checks if not ok]
        print("self-test: %d checks, %d failed; elapsed %.1f s%s" % (len(checks), len(failed), S["elapsed_seconds"], ("; kept " + tmp) if keep else ""))
        return 0 if not failed else 1
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="COLLAPSE_ladder version 2: the tables from the jsons on the box (the plan solve/specs/COLLAPSE_ladder_v2_plan.md).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--set", action="append", default=[], help="role=path overriding a manifest default; roles: " + ", ".join(ROLES))
    ap.add_argument("--slice-dir", action="append", default=None, help="a slice directory (repeat; replaces the defaults)")
    ap.add_argument("--member-json", default=None, help="the v32 member's json (default the ladder_v15 copy; skipped when absent)")
    ap.add_argument("--v17-md", default=None, help="COLLAPSE_ladder_v1.7.md (default the delivery; skipped when absent)")
    ap.add_argument("--cmp-v36d-json", default=None)
    ap.add_argument("--gate18", action="append", default=[], help="rung4=path or rung8=path: a resume-gate run's json, compared against the rung's first run")
    ap.add_argument("--extra-member", action="append", default=[], help="LABEL=PATH: a further run's lines in the per-member and episode tables")
    ap.add_argument("--no-md5", action="store_true", help="skip the md5 of the jsons")
    ap.add_argument("--md5-duplicates", action="store_true", help="md5 the slices held at one clock in two directories")
    ap.add_argument("--every", type=float, default=None)
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--keep-selftest", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test(a.keep_selftest))
    cfg = default_cfg()
    for s in a.set:
        role, p = s.split("=", 1)
        if role not in ROLES:
            sys.exit("unknown role %s" % role)
        cfg["inputs"][role] = p
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.slice_dir:
        cfg["slice_dirs"] = a.slice_dir
    if a.member_json is not None:
        cfg["member_json"] = a.member_json
    if a.v17_md is not None:
        cfg["v17_md"] = a.v17_md
    if a.cmp_v36d_json is not None:
        cfg["cmp_v36d_json"] = a.cmp_v36d_json
    for g in a.gate18:
        rung, p = g.split("=", 1)
        cfg["gate18"][rung] = p
    cfg["extra_members"] = [tuple(x.split("=", 1)) for x in a.extra_member]
    cfg["md5"] = not a.no_md5
    cfg["md5_duplicates"] = a.md5_duplicates
    if a.every:
        cfg["every"] = a.every
    if a.check_inputs:
        for role in ROLES:
            p = cfg["inputs"][role]
            print("%-12s %s %s %s" % (role, "present" if os.path.exists(p) else "ABSENT ", ("%d bytes" % os.path.getsize(p)) if os.path.exists(p) else "", p))
        for k in ("member_json", "v17_md", "cmp_v36d_json"):
            p = cfg[k]
            print("%-12s %s %s" % (k, "present" if (p and os.path.exists(p)) else "absent (optional)", p))
        for d in cfg["slice_dirs"]:
            print("%-12s %s %s" % ("slice dir", "present" if os.path.isdir(d) else "ABSENT ", d))
        for t in (CMF, CV5):
            print("%-12s %s %s" % ("tool", "present" if os.path.exists(t) else "ABSENT ", t))
        return
    S, md_path, js_path = pipeline(cfg)
    lines = summary_lines(S)
    sp = os.path.join(cfg["out_dir"], "compose_ladder_v2_v1.summary.txt")
    with open(sp, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written %s (%d bytes), %s (%d bytes), %s" % (md_path, os.path.getsize(md_path), js_path, os.path.getsize(js_path), sp))
    sys.exit(1 if S["errors"] else 0)


if __name__ == "__main__":
    main()
