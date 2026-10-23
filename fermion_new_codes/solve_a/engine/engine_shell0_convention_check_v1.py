#!/usr/bin/env python
"""engine_shell0_convention_check_v1.py (builder analyst; engine spec 38.3 (3): "engine_shell0_convention_check_v1 (new; CPU; record-only):
the certificate's shares on the 281 certified records with shell 0's S_r by engine_stress_v2's regular limit, the flips at the 0.8 share
counted"; 38.3 Facts: "the records npz sets shell 0's S_r to its outer face, engine_stress_v2 to the regular limit"): the records data npz's
shell stresses (engine_collapse_records_v2.shell_stress: S_r at shell n the mean of its two faces' values, shell 0 its outer face's) with
shell 0's S_r replaced by engine_stress_v2.shell_values' form, 0.5 (S_t,0 + S_r,face 0) (the inner face at the regular limit S_r = S_t),
that is S_r[0] -> 0.5 (S_t[0] + S_r[0]) on the total (dust plus pairs; the form is linear); nothing else changed. On every record the
certificate certifies (engine_certificate_v3's json: both rows' shares within 1/10 at least 0.800 at three decimals), the certificate's own
evaluation repeated with the replaced stress (engine_certificate_v3.certificate and summarize with engine_origin_term_p37_v1's arguments: the
nodes ell to r_collar - 2 ell, the support rho_ell above 1e-6 of its largest value, h = ell/32, 8-point Gauss-Legendre, the field module
per segment, the neighbours' fields as stored): per row the share within 1/10 against the stored certificate's; the flips (a share below
0.800 at three decimals) counted per row and in all, listed; the shell-0 values (stored, replaced, their difference over the record's
largest abs S_r) and the largest relative change of s_r,ell on the support printed. Control: on every --control-every-th record the stored
stress is evaluated too and must reproduce the certificate json's shares at three decimals. --workers N splits the records into N
contiguous chunks (forked processes, each with its own field cache). Every number printed by the script; nothing is ruled. Args: --json
--data --cert --fm-segments --fm-default --workers --control-every --md --out-json."""
import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_certificate_v3 as CERT  # noqa: E402
import engine_stress_v2 as STR2  # noqa: E402

G = {}


def certified(r):
    s = (r or {}).get("summary") or {}
    tt, rr = s.get("tt"), s.get("rr")
    return bool(tt and rr and round(tt["share_within_0.1"], 3) >= 0.8 and round(rr["share_within_0.1"], 3) >= 0.8)


def work(chunk):
    a, Z, T, R, g, ell, crow = G["a"], G["Z"], G["T"], G["R"], G["g"], G["ell"], G["crow"]
    seg_names = [str(x) for x in Z["segment_names"]] if "segment_names" in Z.files else None
    seg = np.asarray(Z["segment"], int) if "segment" in Z.files else np.zeros(len(T), int)
    fmap = dict(x.split("=", 1) for x in a.fm_segments.split(";")) if a.fm_segments else {}
    mods = {}; cache = {}

    def fields(k, shell0=False):
        key = (k, shell0)
        if key in cache:
            return cache[key]
        name = fmap.get(seg_names[seg[k]], a.fm_default) if seg_names else a.fm_default
        if name not in mods:
            mods[name] = importlib.import_module(name)
        base = cache.get((k, False))
        if base is None:
            _, _, _, info = mods[name].solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), g, R, ell, collar_policy="flag", support_rel=1e-14)
            Sr = np.asarray(Z["S_r_dust"][k], float) + np.asarray(Z["S_r_pairs"][k], float)
            St = np.asarray(Z["S_t_dust"][k], float) + np.asarray(Z["S_t_pairs"][k], float)
            base = {"N": info["N_dense"], "v": info["v_dense"], "rho": info["rho_ell"], "J": info["J_ell"], "Sr": Sr, "St": St, "info": info, "fm": name,
                    "failed": bool(info.get("failed"))}
            base["sr"] = (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[0])
            base["st"] = (lambda r, Sr=Sr, St=St: STR2.smooth_stress(Sr, St, R, ell, np.atleast_1d(np.asarray(r, float)))[1])
            cache[(k, False)] = base
        if not shell0:
            return base
        S0 = dict(base)
        SrS = base["Sr"].copy(); SrS[0] = 0.5 * (base["St"][0] + base["Sr"][0]); St = base["St"]
        S0["Sr"] = SrS
        S0["sr"] = (lambda r, SrS=SrS, St=St: STR2.smooth_stress(SrS, St, R, ell, np.atleast_1d(np.asarray(r, float)))[0])
        S0["st"] = (lambda r, SrS=SrS, St=St: STR2.smooth_stress(SrS, St, R, ell, np.atleast_1d(np.asarray(r, float)))[1])
        cache[key] = S0
        return S0
    rows = []
    for n_, (k, control) in enumerate(chunk):
        for old in [x for x in cache if x[0] < k - 2]:
            del cache[old]
        F0 = fields(k)
        row = {"k": k, "T_over_rs": float(T[k] / G["rs"]), "stored": (crow.get(k) or {}).get("summary")}
        if F0["failed"]:
            row["note"] = "the re-solve failed"; rows.append(row); continue
        Fm = fields(k - 1) if k - 1 >= 0 else None
        Fp = fields(k + 1) if k + 1 < len(T) else None
        F0S = fields(k, shell0=True)
        rc = float(F0["info"]["r_collar"]); nodes = np.arange(ell, rc - 2.0 * ell + 1e-9, ell)
        dtm = (T[k] - T[k - 1]) if Fm is not None else None; dtp = (T[k + 1] - T[k]) if Fp is not None else None
        cS = CERT.certificate(Fm, F0S, Fp, dtm, dtp, g, nodes, a.gl, ell / a.hfrac)
        rho_n = cS["rho_nodes"]; support = np.abs(rho_n) > 1e-6 * max(float(np.max(np.abs(rho_n))), 1e-300)
        sS = CERT.summarize(cS, nodes, support)
        srW = np.asarray(F0["sr"](nodes), float); srS = np.asarray(F0S["sr"](nodes), float)
        m = support
        smax = float(np.max(np.abs(F0["Sr"])))
        row.update({"shell0_stored": float(F0["Sr"][0]), "shell0_regular": float(F0S["Sr"][0]), "shell0_diff_over_max": abs(float(F0S["Sr"][0] - F0["Sr"][0])) / max(smax, 1e-300),
                    "dsr_rel_max": float(np.max(np.abs(srS[m] - srW[m]) / np.maximum(np.abs(srW[m]), 1e-300))) if m.any() else None,
                    "shell0": sS, "time_derivative": cS["how"], "support_nodes": int(m.sum())})
        if control:
            cW = CERT.certificate(Fm, F0, Fp, dtm, dtp, g, nodes, a.gl, ell / a.hfrac)
            row["control_stored_eval"] = CERT.summarize(cW, nodes, support)
        rows.append(row)
        if n_ % 10 == 0:
            print("[%d] record %d (T/rs %.3f) done, %.0f s" % (os.getpid(), k, T[k] / G["rs"], time.time() - G["t0"]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--data", required=True); ap.add_argument("--cert", required=True)
    ap.add_argument("--fm-segments", default="v17 (replayed)=engine_fields_v12;v19 (native)=engine_fields_v14"); ap.add_argument("--fm-default", default="engine_fields_v12")
    ap.add_argument("--hfrac", type=float, default=32.0); ap.add_argument("--gl", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1); ap.add_argument("--control-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="a smoke test: the first this many certified records only (0: all)")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    for p in (a.md, a.out_json):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    t0 = time.time()
    J = json.load(open(a.json))
    RJ = [r for r in J["records"] if r.get("E") is not None and "T_over_rs" in r]
    Z = np.load(a.data, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); g = float(Z["g"]); ell = float(Z["ell"]); rs = float(Z["rs"])
    if len(RJ) != len(T) or float(np.max(np.abs(np.array([r["T"] for r in RJ]) - T))) > 1e-9:
        raise SystemExit("the json's records and the data npz's do not align")
    C = json.load(open(a.cert))
    crow = {int(r["k"]): r for r in C["rows"]}
    ks = [k for k in range(len(T)) if certified(crow.get(k))]
    ks_all = len(ks)
    if a.limit > 0:
        ks = ks[:a.limit]
    items = [(k, (i % a.control_every) == 0) for i, k in enumerate(ks)]
    G.update({"a": a, "Z": Z, "T": T, "R": R, "g": g, "ell": ell, "rs": rs, "crow": crow, "t0": t0})
    n = max(1, a.workers)
    chunks = [items[i * len(items) // n:(i + 1) * len(items) // n] for i in range(n)]
    print("%d certified records of %d; %d workers; controls every %d-th record (%d)" % (len(ks), len(T), n, a.control_every, sum(1 for _, c in items if c)), flush=True)
    if n > 1:
        with mp.get_context("fork").Pool(n) as pool:
            parts = pool.map(work, chunks, chunksize=1)
    else:
        parts = [work(chunks[0])]
    rows = sorted([r for p in parts for r in p], key=lambda r: r["k"])
    ok = [r for r in rows if not r.get("note")]
    sh = lambda s, rw: ((s or {}).get(rw) or {}).get("share_within_0.1")  # noqa: E731
    flips = {rw: [r for r in ok if sh(r["shell0"], rw) is not None and round(sh(r["shell0"], rw), 3) < 0.8] for rw in ("tt", "rr")}
    either = sorted(set(r["k"] for rw in flips for r in flips[rw]))
    ctl = [r for r in ok if r.get("control_stored_eval")]
    ctl_bad = [r for r in ctl if any(round(sh(r["control_stored_eval"], rw) or -1, 3) != round(sh(r["stored"], rw) or -1, 3) for rw in ("tt", "rr"))]
    f = lambda x, fm="%.3f": (fm % x) if x is not None else "-"  # noqa: E731
    L = ["## The certificate's shares with shell 0's S_r by engine_stress_v2's regular limit (engine_shell0_convention_check_v1.py; engine spec 38.3 (3); record-only)", "",
         "- the stored convention (the records data npz, engine_collapse_records_v2.shell_stress): S_r at shell 0 its outer face's value; the regular limit (engine_stress_v2.shell_values): 0.5 (S_t,0 + S_r,face 0), the inner face at r = 0 taken as S_r = S_t; applied to the total stress at shell 0 only, every other shell and S_t unchanged, smoothed by smooth_stress.",
         "- the certificate's evaluation (engine_certificate_v3.certificate and summarize; nodes ell to r_collar - 2 ell, the support rho_ell above 1e-6 of its largest value, h = ell/%g, %d-point Gauss-Legendre, the field module per segment %s, the neighbours' fields as stored) with the replaced stress; per row the share of support nodes within 1/10 (the criterion: at least 0.800 at three decimals) against the stored certificate's (%s)." % (a.hfrac, a.gl, a.fm_segments, a.cert),
         "- %d records certified of %d (%s); evaluated %d; the control (the stored stress evaluated on every %d-th record, %d records): %s." % (
             ks_all, len(T), ("a smoke test on the first %d" % a.limit) if a.limit > 0 else "all of them", len(ok), a.control_every, len(ctl), "every share reproduced at three decimals" if not ctl_bad else "NOT reproduced on %d: %s" % (len(ctl_bad), ", ".join("%.3f" % r["T_over_rs"] for r in ctl_bad))),
         "- the flips at the 0.8 share (certified records whose share with the regular limit is below 0.800): tt %d (%s), rr %d (%s); records with either row below: %d of %d." % (
             len(flips["tt"]), ", ".join("%.3f" % r["T_over_rs"] for r in flips["tt"]) or "none", len(flips["rr"]), ", ".join("%.3f" % r["T_over_rs"] for r in flips["rr"]) or "none", len(either), len(ok)),
         "- shell 0's two values differ by %s to %s of the record's largest abs S_r over these records; the largest relative change of s_r,ell on the support %s." % (
             f(min(r["shell0_diff_over_max"] for r in ok), "%.3e"), f(max(r["shell0_diff_over_max"] for r in ok), "%.3e"), f(max(r["dsr_rel_max"] for r in ok if r["dsr_rel_max"] is not None), "%.3e")), "",
         "| T/rs | shell 0 S_r: stored, regular limit (difference over max abs S_r) | max abs(d s_r)/abs(s_r) on the support | tt share: stored, regular limit | rr share: stored, regular limit | flip | control (stored stress evaluated: tt, rr) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("note"):
            L.append("| %.3f | %s | - | - | - | - | - |" % (r["T_over_rs"], r["note"])); continue
        fl = [rw for rw in ("tt", "rr") if sh(r["shell0"], rw) is not None and round(sh(r["shell0"], rw), 3) < 0.8]
        L.append("| %.3f | %+.4e, %+.4e (%.3e) | %s | %s, %s | %s, %s | %s | %s |" % (
            r["T_over_rs"], r["shell0_stored"], r["shell0_regular"], r["shell0_diff_over_max"], f(r["dsr_rel_max"], "%.3e"),
            f(sh(r["stored"], "tt")), f(sh(r["shell0"], "tt")), f(sh(r["stored"], "rr")), f(sh(r["shell0"], "rr")), ", ".join(fl) or "-",
            ("%s, %s" % (f(sh(r["control_stored_eval"], "tt")), f(sh(r["control_stored_eval"], "rr")))) if r.get("control_stored_eval") else "-"))
    L += ["", "- every number above is printed by the script from the production's json, its records data npz and the certificate json; nothing is attributed or ruled (read in turn 47 with the origin term, engine spec 38.3 (3), 38.6 (1))."]
    open(a.md, "x").write("\n".join(L) + "\n")
    json.dump({"script": "engine_shell0_convention_check_v1.py", "args": vars(a), "rows": rows, "flips_tt": [r["k"] for r in flips["tt"]], "flips_rr": [r["k"] for r in flips["rr"]]},
              open(a.out_json, "x"), default=float, indent=1)
    print("\n".join(L[:7])); print("done, %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
