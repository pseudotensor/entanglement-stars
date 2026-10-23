#!/usr/bin/env python
"""engine_collapse_v36x_selftest.py (busi, analyst; #299 PERF ISO-A2 and ISO-A3): the whole-engine Newton gate of engine_collapse_v36x.py
(build_engine_collapse_v36x.py over v36r) and of v36r's existing --pair-workers 2, on the CPU.
The fixture is the one the v36r build used (engine_collapse_v36n_selftest.py: m 0.2, r_s 16, 512 sites, kappa_sea 10, T/r_s 0 to 0.25, the
ladders' flags with --field-kernel-cache), run as m3c_nk_r runs: --mix newton, J2's NK flags (--nk-jvp exact, --nk-set resolve), M3c's
options (--dust-set isotropic --iter-cache drop) and --nk-plain-fallback on. Cases:
  R0  v36r                                   the reference
  RP  v36r --pair-workers 2                  ISO-A2: a launch argument on the production engine
  X0  v36x, defaults                         the build's defaults
  XC  v36x --nk-jvp-cache-gib 4              every channel cached
  XS  v36x --nk-jvp-cache-gib 0.1            the budget binds (38 MB a channel on the fixture): some channels cached, the rest fresh
  XD  v36x --nk-jvp-defer on                 the transfers deferred, nothing cached
  XA  v36x --nk-jvp-cache-gib 4 --nk-jvp-defer on --pair-workers 2   everything on
Each against R0 by the v36n self-test's compare_runs (records but the walls, head, fields, spectra, handoff npz, the T = 0 slice), the records
compared with the per-process entries removed as bench_protocol 4.4 (1) allows: "v36x" (the new counters), "v36b" (the pair workers' counters)
and v36.kernel_cache (the parent's cache counters omit the workers' solves). The args differ only by the new keys and the named options.
Checks beside: the J.v ran (jvp_v35 products > 0 on R0); XC hit (channel_hits > 0, uncached 0); XS left channels uncached (uncached > 0, hits
> 0); XD deferred every product (deferred_products = products); RP and XA paired every product (products_paired = products, no serial
fallback, no errors). Walls (the run's seconds, the records' jvp_v35 seconds) are printed as INDICATIVE: the box is loaded.
    python engine_collapse_v36x_selftest.py --code DIR --v36n-selftest .../engine_collapse_v36n_selftest.py --out DIR [--threads 4] [--only R0,X0]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types

NK = ["--nk-krylov", "40", "--nk-halvings", "6", "--nk-inner", "15", "--nk-jvp", "exact", "--nk-line-search", "halving", "--nk-set", "resolve",
      "--nk-maxiter", "30", "--nk-restarts", "2"]
PROD = ["--mix", "newton"] + NK + ["--dust-set", "isotropic", "--iter-cache", "drop", "--nk-plain-fallback", "on"]
CASES = [
    ("R0", "engine_collapse_v36r.py", []),
    ("RP", "engine_collapse_v36r.py", ["--pair-workers", "2"]),
    ("X0", "engine_collapse_v36x.py", []),
    ("XC", "engine_collapse_v36x.py", ["--nk-jvp-cache-gib", "4"]),
    ("XS", "engine_collapse_v36x.py", ["--nk-jvp-cache-gib", "0.1"]),
    ("XD", "engine_collapse_v36x.py", ["--nk-jvp-defer", "on"]),
    ("XA", "engine_collapse_v36x.py", ["--nk-jvp-cache-gib", "4", "--nk-jvp-defer", "on", "--pair-workers", "2"]),
]
NEWKEYS = {"nk_jvp_cache_gib", "nk_jvp_defer"}


def load(path, code_dir, name):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drop_proc(o):
    """the per-process entries (bench_protocol 4.4 (1)): v36x, v36b, v36.kernel_cache."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in ("v36x", "v36b"):
                continue
            if k == "v36" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "kernel_cache"}
            out[k] = drop_proc(v)
        return out
    if isinstance(o, list):
        return [drop_proc(v) for v in o]
    return o


def compare(S, A, B):
    """compare_runs on everything but the records, the records with drop_proc."""
    d = [x for x in S.compare_runs(A, B) if not x.startswith("records differ")]
    JA, JB = json.load(open(A["json"])), json.load(open(B["json"]))
    ra = [json.dumps(drop_proc(S.strip(q)), sort_keys=True, default=float) for q in JA["records"]]
    rb = [json.dumps(drop_proc(S.strip(q)), sort_keys=True, default=float) for q in JB["records"]]
    bad = [JA["records"][i].get("T_over_rs") for i in range(min(len(ra), len(rb))) if ra[i] != rb[i]]
    if bad:
        d.append("records differ at T/rs %s" % bad)
    return d, JA, JB


def sumrec(J, entry, key):
    return sum((q.get(entry) or {}).get(key, 0) or 0 for q in J["records"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--v36n-selftest", dest="v36n_selftest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--reuse", action="store_true", help="compare the runs already under --out (nothing recomputed)")
    a = ap.parse_args()
    S = load(a.v36n_selftest, a.code, "engine_collapse_v36n_selftest_lib")
    for f_ in ("engine_collapse_v36r.py", "engine_collapse_v36x.py", "engine_pairsolve_v1.py", "engine_collapse_v36c.py"):
        p_ = os.path.join(a.code, f_)
        print("%s md5 %s" % (f_, hashlib.md5(open(p_, "rb").read()).hexdigest()))
    ns = types.SimpleNamespace(code_dir=a.code, no_kcache=False, gpu=None, threads=a.threads)
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    fails, runs = [], {}
    # the command line: the new options refused outside the exact J.v
    eng = os.path.join(a.code, "engine_collapse_v36x.py")
    h = subprocess.run([sys.executable, eng, "--help"], cwd=a.code, capture_output=True, text=True).stdout
    r = subprocess.run([sys.executable, eng, "--m", "0.2", "--rs", "16", "--nsites", "64", "--mix", "plain", "--nk-jvp-defer", "on", "--out", a.out, "--tag", "cli"],
                       cwd=a.code, capture_output=True, text=True)
    ok = "--nk-jvp-cache-gib" in h and "--nk-jvp-defer" in h and r.returncode == 2 and "version 36x" in r.stderr
    print("cli: --help lists both options; --nk-jvp-defer on without --nk-jvp exact refused (exit 2): %s" % ok)
    if not ok:
        fails.append("cli: %r" % r.stderr[-300:])
    for name, e_, extra in CASES:
        if only and name not in only:
            continue
        jp = os.path.join(a.out, name, "collapse_fx_%s.json" % name)
        if a.reuse:   # compare only: the stored runs, nothing recomputed (walls from seconds.txt when present)
            o_ = os.path.join(a.out, name); sp = os.path.join(o_, "seconds.txt")
            runs[name] = {"name": name, "rc": 0 if os.path.exists(jp) else 1, "seconds": float(open(sp).read()) if os.path.exists(sp) else float("nan"),
                          "log": os.path.join(o_, "log.txt"), "json": jp, "fields": jp[:-5] + "_fields.npz", "spectra": jp[:-5] + "_spectra.npz",
                          "handoff": jp[:-5] + "_handoff.npz", "slice": os.path.join(o_, "states", "orbitals_fx_%s_T0000.00.npz" % name)}
            print("reused %s (%s)" % (name, jp))
            if runs[name]["rc"] != 0:
                fails.append("%s: no stored run to reuse" % name)
            continue
        if os.path.exists(jp):
            raise SystemExit("%s exists under %s: use a fresh --out (nothing overwritten)" % (name, a.out))
        print("running %s (%s %s) ..." % (name, e_, " ".join(extra)), flush=True)
        runs[name] = S.run_case(ns, name, e_, PROD + extra, a.out)
        open(os.path.join(a.out, name, "seconds.txt"), "w").write("%.3f\n" % runs[name]["seconds"])
        print("  exit %d in %.0f s (%s)" % (runs[name]["rc"], runs[name]["seconds"], runs[name]["log"]), flush=True)
        if runs[name]["rc"] != 0:
            fails.append("%s exit %d" % (name, runs[name]["rc"]))
    if "R0" not in runs or runs["R0"]["rc"] != 0:
        fails.append("no R0")
    else:
        J0 = json.load(open(runs["R0"]["json"]))
        prod0 = sumrec(J0, "jvp_v35", "products")
        print("R0: %d records, J.v products %d, extra evaluations %d, fallback lines %d" % (
            len(J0["records"]), prod0, sumrec(J0, "jvp_v35", "extra_evaluations"),
            sum(1 for ln in open(runs["R0"]["log"], errors="replace") if ln.startswith("  v36r fallback"))))
        if prod0 <= 0:
            fails.append("R0: no J.v product ran (the fixture does not exercise the exact J.v)")
        print("%-3s %-8s %-9s %s" % ("run", "wall_s", "jvp_s", "verdict against R0 (INDICATIVE walls: the box is loaded)"))
        print("%-3s %8.0f %9.1f reference" % ("R0", runs["R0"]["seconds"], sumrec(J0, "jvp_v35", "seconds")))
        for name, e_, extra in CASES:
            if name == "R0" or name not in runs or runs[name]["rc"] != 0:
                continue
            d, JB, _ = compare(S, runs[name], runs["R0"])   # JB: this arm's json (compare's first)
            ka = set(JB["args"]) ^ set(J0["args"])
            va = sorted(k for k in set(JB["args"]) & set(J0["args"]) if JB["args"][k] != J0["args"][k] and k not in ("out", "tag", "slice_dir"))
            allowed_k = NEWKEYS if e_.endswith("v36x.py") else set()
            allowed_v = {"pair_workers"} if "--pair-workers" in extra else set()
            if ka - allowed_k or set(va) - allowed_v:
                d.append("args differ beyond the options: keys %s values %s" % (sorted(ka - allowed_k), sorted(set(va) - allowed_v)))
            prod = sumrec(JB, "jvp_v35", "products")
            notes = []
            if prod != prod0:
                d.append("J.v products %d against %d" % (prod, prod0))
            if "--pair-workers" in extra:
                pp, fb = sumrec(JB, "v36b", "products_paired"), sumrec(JB, "v36b", "serial_fallbacks")
                errs = [q.get("v36b", {}).get("errors") for q in JB["records"] if (q.get("v36b") or {}).get("errors")]
                notes.append("paired %d of %d (a product with a zero source change, sc = 0, needs no solve), serial fallbacks %d, errors %s" % (pp, prod, fb, errs or "none"))
                if pp <= 0 or pp > prod or fb or errs:
                    d.append("pair workers: paired %d of %d, fallbacks %d, errors %s" % (pp, prod, fb, errs))
            if e_.endswith("v36x.py") and extra:
                hits, miss, unc = sumrec(JB, "v36x", "channel_hits"), sumrec(JB, "v36x", "channel_misses"), sumrec(JB, "v36x", "uncached")
                dfr = sumrec(JB, "v36x", "deferred_products"); bm = max((q.get("v36x") or {}).get("bytes_max", 0) for q in JB["records"])
                notes.append("channel hits %d, misses %d, uncached %d, workspaces %d, bytes_max %.1f MB, deferred %d" % (
                    hits, miss, unc, sumrec(JB, "v36x", "workspaces"), bm / 1e6, dfr))
                gib = float(extra[extra.index("--nk-jvp-cache-gib") + 1]) if "--nk-jvp-cache-gib" in extra else 0.0
                defer = "--nk-jvp-defer" in extra
                if gib >= 1 and (hits <= 0 or unc != 0):
                    d.append("cache: hits %d uncached %d (expected hits, none uncached)" % (hits, unc))
                if 0 < gib < 1 and (hits <= 0 or unc <= 0):
                    d.append("small budget: hits %d uncached %d (expected both > 0)" % (hits, unc))
                if gib == 0 and hits != 0:
                    d.append("no cache but %d hits" % hits)
                if defer and dfr != prod:
                    d.append("deferred %d of %d products" % (dfr, prod))
            if e_.endswith("v36x.py") and not extra and any("v36x" in q for q in JB["records"]):
                d.append("defaults wrote a v36x entry")
            print("%-3s %8.0f %9.1f %s%s" % (name, runs[name]["seconds"], sumrec(JB, "jvp_v35", "seconds"),
                                            "bitwise" if not d else "DIFFERS: " + "; ".join(d), (" | " + "; ".join(notes)) if notes else ""))
            if d:
                fails.append("%s: %s" % (name, "; ".join(d)))
    for f_ in fails:
        print("FAIL: " + f_)
    print("SELFTEST %s (%d checks failed)" % ("PASS" if not fails else "FAIL", len(fails)))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
