#!/usr/bin/env python
"""engine_collapse_v36k_selftest.py (reviewer A for the busi lane; W2 dg ruling v2): the self-test of engine_collapse_v36k.py.
  --build-check V36F V36K: (b1) the builder's reverse replacements restore engine_collapse_v36f.py byte for byte, (b2) its forward replacements
    reproduce the installed v36k, (b3) every line not in v36f carries the marker.
  --fixture --code DIR --out DIR [--gpu N] [--threads 16] [--python PY] [--fixture-args "..."] [--only i,ii,...]: 8 engine runs on the fixture
    (FIXTURE below: r_s/a 16, nsites 512, 4 channels, tau 2 a, ell 4, reading A, R2, averaged, 3 records to T/r_s 0.375, a complex128 slice at
    0.125; the same fixture as engine_collapse_v36f_selftest.py, override with --fixture-args if that file's arguments differ) and the checks:
    (i)    v36k --step dg at the default flags reproduces v36f --step dg bitwise (json records and head, fields, handoff, spectra, the slice);
    (ii)   v36k --step frozen reproduces v36f --step frozen bitwise (the default path untouched);
    (iii)  --dg-chart frozen --dg-mix plain --dg-sub-max 0 against v36f --step dg: where the convention does not reassign (the records' pockets and
           inversions equal) E agrees to 1e-10 relative and the applied fields to 1e-8 (printed: bitwise or not);
    (iv)   conservation on the frozen chart: every step converged, E/E0 - 1 within 1e-9 at every record, gate 2 (the defect) below 1.01e-12 of E,
           gate 1 (r_chain) below 1e-9, and E_pred - E_end = defect - r_chain (|E_minus_E_end - E_minus_E_pred| below 2e-9 of E0);
    (v)    --dg-mix anderson (history 4, beta 1) reaches the same fixed points (E within 1e-10 relative, fields within 1e-5) in no more iterations
           than plain (+1 admitted);
    (vi)   --dg-sub-start 1 --dg-sub-max 1 (2 sub-steps of tau/2 per record): nsub 2 at every record, converged and conserving as (iv); against
           level 0 the fields and E are printed (second order: expected below 1e-4 and 1e-7);
    (vii)  the restart from the complex128 slice at T/r_s 0.125 is bitwise with the uninterrupted frozen-chart run (the slice's node fields,
           dg_sub_len_v36k and dg_level_v36k);
    (viii) the adjoint test's form on the chart (engine_adjoint_test_v1's fixture, r_s/a 8, nsites 256, 3 channels, no engine run): (viii-b) the
           chart's pairing sum_n (N_n eps_n + v_n J_n) is the collar charge to 1e-11; (viii-c) the tracked reference is spectral along the path:
           sum_n (N_n d eps_ref,n + v_n d J_ref,n) under a field perturbation delta is O(delta^2) (the ratio at delta and delta/2 within [3.5, 4.5],
           its size below 1e-2 of the unsigned pairing), with the tracked set's smallest occupied weight above 0.99."""
import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

FIXTURE = ("--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 4 --reference R2 --site-fields averaged --ell 4 --prep-ell 4 "
           "--tau 2 --mix plain --T-over-rs 0.375 --slices-c128 0.125")
SKIP = ("seconds", "time", "stamp", "date", "host", "pid", "args", "script")
FROZEN = ["--step", "dg", "--dg-chart", "frozen"]


def say(ok, name, text):
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, text), flush=True)
    return bool(ok)


def build_check(v36f, v36k):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("build_v36k", os.path.join(here, "build_engine_collapse_v36k.py"))
    B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    src = open(v36f, "rb").read().decode("utf-8"); built = open(v36k, "rb").read().decode("utf-8")
    ok1 = say(B.reverse(built) == src, "(b1) the reverse replacements restore engine_collapse_v36f.py", "%d bytes, md5 %s" % (len(src), hashlib.md5(src.encode("utf-8")).hexdigest()))
    ok2 = say(B.build(src) == built, "(b2) the forward replacements reproduce the installed engine_collapse_v36k.py", "md5 %s" % hashlib.md5(built.encode("utf-8")).hexdigest())
    srcset = set(src.splitlines())
    unmarked = [l_ for l_ in built.splitlines() if l_ not in srcset and "v36k chart" not in l_ and not l_.endswith("[v36f dg]")]
    marked = sum(1 for l_ in built.splitlines() if "# v36k chart" in l_ or "[v36k chart]" in l_)
    ok3 = say(not unmarked, "(b3) every line not in v36f carries the marker", "%d marked lines; unmarked: %s" % (marked, unmarked[:3]))
    return ok1 and ok2 and ok3


def run_engine(a, engine, out, tag, extra):
    os.makedirs(out, exist_ok=True)
    cmd = [a.python, os.path.join(a.code, engine)] + a.fixture_args.split() + ["--out", out, "--tag", tag, "--slice-dir", os.path.join(out, "states")] + list(extra)
    if a.gpu is not None:
        cmd += ["--gpu", str(a.gpu)]
    env = dict(os.environ); env.update({"OMP_NUM_THREADS": str(a.threads), "MKL_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads)})
    t0 = time.time()
    with open(os.path.join(out, tag + ".log"), "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=a.code)
    print("  %s %s -> rc %d, %.0f s (%s)" % (engine, " ".join(extra), rc, time.time() - t0, os.path.join(out, tag + ".log")), flush=True)
    return rc


def load(out, tag):
    J = json.load(open(os.path.join(out, "collapse_%s.json" % tag)))
    Z = {}
    for k in ("fields", "handoff", "spectra"):
        p = os.path.join(out, "collapse_%s_%s.npz" % (tag, k))
        if os.path.exists(p):
            Z[k] = np.load(p, allow_pickle=True)
    S = {os.path.basename(f).split("_T")[-1]: f for f in glob.glob(os.path.join(out, "states", "orbitals_%s_T*.npz" % tag))}
    return J, Z, S


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(k for k in a if not any(s in k for s in SKIP)); kb = set(k for k in b if not any(s in k for s in SKIP))
        if ka != kb:
            return ["keys at %s: %s" % (path, sorted(ka ^ kb))]
        return [d for k in sorted(ka) for d in diff(a[k], b[k], path + "/" + k)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return ["length at %s: %d vs %d" % (path, len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, "%s[%d]" % (path, i))]
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else ["%s: %r vs %r" % (path, a, b)]


def npz_diff(A, B):
    out = []
    ka = set(A.files); kb = set(B.files)
    if ka != kb:
        out.append("keys %s" % sorted(ka ^ kb))
    for k in sorted(ka & kb):
        x = A[k]; y = B[k]
        if x.dtype == object or y.dtype == object:
            if repr(x.tolist()) != repr(y.tolist()):
                out.append(k)
        elif x.shape != y.shape or not np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc" and y.dtype.kind in "fc")):
            out.append(k)
    return out


def bitwise(name, A, B):
    JA, ZA, SA = A; JB, ZB, SB = B
    d = diff(JA["records"], JB["records"], "records") + diff(JA.get("head"), JB.get("head"), "head")
    if set(ZA) != set(ZB):
        d.append("npz sets %s" % sorted(set(ZA) ^ set(ZB)))
    for k in sorted(set(ZA) & set(ZB)):
        d += ["%s.npz: %s" % (k, q) for q in npz_diff(ZA[k], ZB[k])]
    if set(SA) != set(SB):
        d.append("slice sets %s" % sorted(set(SA) ^ set(SB)))
    for k in sorted(set(SA) & set(SB)):
        d += ["slice %s: %s" % (k, q) for q in npz_diff(np.load(SA[k], allow_pickle=True), np.load(SB[k], allow_pickle=True))]
    return say(not d, name, ("bitwise: %d records, npz %s, %d slices" % (len(JA["records"]), sorted(ZA), len(SA))) if not d else ("%d differences, the first: %s" % (len(d), d[:5])))


def rs_of(J):
    return float(J["args"]["rs"])


def recs(J):
    return {round(r["T"] / rs_of(J), 6): r for r in J["records"] if r.get("E") is not None}


def events(r):
    pk = r.get("n_pocket"); pk = sum(pk.values()) if isinstance(pk, dict) else pk
    return (pk, r.get("inversions_total"))


def fields_rows(Z):
    return {round(float(T), 6): i for i, T in enumerate(np.asarray(Z["fields"]["T"], float))}


def compare_runs(name, A, B, tolE, tolF, soft=False):
    JA, ZA, _ = A; JB, ZB, _ = B
    RA = recs(JA); RB = recs(JB); rsA = rs_of(JA); worstE = 0.0; worstF = 0.0; ev_ok = True; same = True
    rowsA = fields_rows(ZA); rowsB = fields_rows(ZB)
    for t in sorted(set(RA) & set(RB)):
        worstE = max(worstE, abs(RA[t]["E"] / RB[t]["E"] - 1.0)); ev_ok = ev_ok and (events(RA[t]) == events(RB[t]))
        same = same and (RA[t]["E"] == RB[t]["E"])
        TA = RA[t]["T"]
        if round(TA, 6) in rowsA and round(TA, 6) in rowsB:
            for key in ("N_applied", "v_applied"):
                xa = np.asarray(ZA["fields"][key][rowsA[round(TA, 6)]], float); xb = np.asarray(ZB["fields"][key][rowsB[round(TA, 6)]], float)
                worstF = max(worstF, float(np.max(np.abs(xa - xb)))); same = same and bool(np.array_equal(xa, xb))
    ok = (worstE <= tolE and worstF <= tolF) or soft
    return say(ok and ev_ok, name, "E relative %.2e (admitted %.0e), fields %.2e (admitted %.0e), reference events %s, %s" % (
        worstE, tolE, worstF, tolF, "equal" if ev_ok else "DIFFER (a reassignment on the fixture: the comparison is not expected to hold)", "bitwise" if same else "not bitwise"))


def conservation(name, J, tolE=1e-9, tol_def=1.01e-12, tol_chain=1e-9, tol_eq=2e-9, nsub=None):
    E0 = float(J["head"]["E0"]); worst = 0.0; bad = []; its = []
    for t, r in sorted(recs(J).items()):
        d = r.get("dg_v36f")
        if d is None:
            continue
        worst = max(worst, abs(r["E"] / E0 - 1.0)); its.append(d.get("iterations", 0))
        if not d.get("converged"):
            bad.append("%.4f not converged" % t)
        if abs(d.get("energy_defect_final") or 0.0) > tol_def:
            bad.append("%.4f defect %.1e" % (t, d.get("energy_defect_final")))
        if d.get("r_chain_over_E_rec") is not None and abs(d["r_chain_over_E_rec"]) > tol_chain:
            bad.append("%.4f r_chain %.1e" % (t, d["r_chain_over_E_rec"]))
        if d.get("E_minus_E_end_over_E0") is not None and abs(d["E_minus_E_end_over_E0"] - (d.get("E_minus_E_pred_over_E0") or 0.0)) > tol_eq:
            bad.append("%.4f E_end vs E_pred %.1e" % (t, d["E_minus_E_end_over_E0"] - (d.get("E_minus_E_pred_over_E0") or 0.0)))
        if nsub is not None and d.get("nsub") != nsub:
            bad.append("%.4f nsub %s" % (t, d.get("nsub")))
    return say(worst <= tolE and not bad, name, "max |E/E0 - 1| %.2e (admitted %.0e), iterations %s%s" % (worst, tolE, its, ("; " + "; ".join(bad)) if bad else ""))


def iterations(J):
    return [r["dg_v36f"].get("iterations", 0) for t, r in sorted(recs(J).items()) if r.get("dg_v36f")]


def chart_identity(a):
    sys.path.insert(0, os.path.abspath(a.code))
    import engine_collapse_v5 as C5
    import engine_collapse_v36k as E
    import engine_fields_v16 as FLD
    C5.backend(a.gpu); torch = C5.BK["torch"]
    if a.gpu is None:
        torch.set_num_threads(max(1, int(a.threads)))
    E.OPT.update({"site_fields": "averaged", "collar_policy": "flag", "support_rel": 1e-14}); E.SOLVE_TOL.update({"rtol": 1e-12, "atol": 1e-16})
    rs, g, m, ell = 8.0, 0.25, 0.2, 4; NB = rs / (2.0 * g * m); R0 = 4.0 * rs
    chs = [C5.Chan(k, m, 256, False) for k in range(1, 4)]; R = np.asarray(chs[0].R, float); n = len(R)
    N, v, dust, info0, head = E.prepare_dust(chs, g, R, ell, NB, R0, m, "A", None)
    eps = np.zeros(n); J = np.zeros(n)
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]; eps += ch.w * C5.dens(ch.h, Phi, fk); J += ch.w * C5.dens(ch.Q, Phi, fk)
    Na, va, Wn, info = E.solve(eps, J, g, R, ell, None); Na = np.asarray(Na, float); va = np.asarray(va, float)
    Ec = float(FLD.charge(info, g)); pair = float(np.sum(Na * eps + va * J))
    ok_b = say(abs(Ec - pair) / abs(Ec) < 1e-11, "(viii-b) the chart's pairing is the collar charge", "E %.12f, sum_n (N_n eps_n + v_n J_n) %.12f, relative defect %.2e" % (Ec, pair, abs(Ec - pair) / abs(Ec)))
    eig = E.eigh_all(chs, Na, va, False); anchors = {}; d0 = {}
    for ch in chs:
        e, U = eig.pop(ch.kappa); ev = e.detach().cpu().numpy(); mask = ev < 0.0
        A = U[:, torch.from_numpy(mask).to(U.device)]
        anchors[ch.kappa] = (A, int(mask.sum())); d0[ch.kappa] = (C5.dens(ch.h, A), C5.dens(ch.Q, A))
        del e, U
    rng = np.random.default_rng(1); dN = rng.standard_normal(n); dv = rng.standard_normal(n)

    def pairing(delta):
        eg = E.eigh_all(chs, Na + delta * dN, va + delta * dv, False); out = 0.0; unsigned = 0.0; wmin = 1.0
        for ch in chs:
            e, U = eg.pop(ch.kappa); ref, mask, ti = E.dgk_track_v36k(e, U, anchors[ch.kappa])
            de = C5.dens(ch.h, ref) - d0[ch.kappa][0]; dq = C5.dens(ch.Q, ref) - d0[ch.kappa][1]
            out += ch.w * float(np.sum(Na * de + va * dq)); unsigned += ch.w * float(np.sum(np.abs(Na * de) + np.abs(va * dq)))
            wmin = min(wmin, ti["min_occupied_weight"] if ti["min_occupied_weight"] is not None else 1.0)
            del e, U, ref
        return out, unsigned, wmin
    p1, u1, w1 = pairing(1e-3); p2, u2, w2 = pairing(5e-4); pm, _um, _wm = pairing(-1e-3)
    ratio = (p1 / p2) if p2 else float("nan"); size = abs(p1) / max(u1, 1e-300)
    ok_c = say(3.5 <= ratio <= 4.5 and size < 1e-2 and min(w1, w2) > 0.99, "(viii-c) the tracked reference is spectral along the path (Tr(h P-dot) = 0)",
               "sum_n (N_n d eps_ref,n + v_n dJ_ref,n) = %.3e at delta 1e-3, %.3e at 5e-4 (ratio %.3f, 4 expected), %.3e at -1e-3; its size over the unsigned pairing %.1e; smallest occupied weight %.5f" % (
                   p1, p2, ratio, pm, size, min(w1, w2)))
    return ok_b and ok_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", dest="build_check", nargs=2, metavar=("V36F", "V36K"), default=None)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--code", default=".")
    ap.add_argument("--out", default="selftest_v36k")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fixture-args", dest="fixture_args", default=FIXTURE)
    ap.add_argument("--only", default=None, help="a comma list of the checks to run, e.g. i,iv,viii")
    a = ap.parse_args()
    results = {}
    if a.build_check:
        results["build"] = build_check(*a.build_check)
    if a.fixture:
        only = set(a.only.split(",")) if a.only else None
        want = lambda k: (only is None) or (k in only)
        out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
        print("the fixture: %s; code %s; device %s" % (a.fixture_args, os.path.abspath(a.code), "cpu" if a.gpu is None else "cuda:%d" % a.gpu), flush=True)
        if want("i") or want("iii") or want("iv") or want("v") or want("vi") or want("vii"):
            if want("i") or want("iii"):
                run_engine(a, "engine_collapse_v36f.py", os.path.join(out, "f_dg"), "fx", ["--step", "dg"])
            if want("i"):
                run_engine(a, "engine_collapse_v36k.py", os.path.join(out, "k_dg"), "fx", ["--step", "dg"])
                results["i"] = bitwise("(i) v36k --step dg at the default flags is v36f --step dg", load(os.path.join(out, "f_dg"), "fx"), load(os.path.join(out, "k_dg"), "fx"))
            if want("ii"):
                run_engine(a, "engine_collapse_v36f.py", os.path.join(out, "f_fr"), "fx", ["--step", "frozen"])
                run_engine(a, "engine_collapse_v36k.py", os.path.join(out, "k_fr"), "fx", ["--step", "frozen"])
                results["ii"] = bitwise("(ii) v36k --step frozen is v36f --step frozen", load(os.path.join(out, "f_fr"), "fx"), load(os.path.join(out, "k_fr"), "fx"))
            if want("iii") or want("iv") or want("v") or want("vi") or want("vii"):
                run_engine(a, "engine_collapse_v36k.py", os.path.join(out, "fz"), "fz", FROZEN + ["--dg-mix", "plain", "--dg-sub-max", "0"])
                FZ = load(os.path.join(out, "fz"), "fz")
            if want("iii"):
                results["iii"] = compare_runs("(iii) the frozen chart against v36f's continued chart where the convention does not reassign", FZ, load(os.path.join(out, "f_dg"), "fx"), 1e-10, 1e-8)
            if want("iv"):
                results["iv"] = conservation("(iv) conservation on the frozen chart (gates 1 and 2, E against E0, E_end against E_pred)", FZ[0])
            if want("v"):
                run_engine(a, "engine_collapse_v36k.py", os.path.join(out, "fa"), "fa", FROZEN + ["--dg-mix", "anderson", "--dg-anderson-m", "4", "--dg-beta", "1.0", "--dg-sub-max", "0"])
                FA = load(os.path.join(out, "fa"), "fa"); ip = iterations(FZ[0]); ia = iterations(FA[0])
                same_fp = compare_runs("(v-a) Anderson reaches plain's fixed points", FZ, FA, 1e-10, 1e-5)
                results["v"] = same_fp and say(all(x <= y + 1 for x, y in zip(ia, ip)) and len(ia) == len(ip), "(v-b) Anderson's iterations against plain's", "anderson %s, plain %s" % (ia, ip))
            if want("vi"):
                run_engine(a, "engine_collapse_v36k.py", os.path.join(out, "fs"), "fs", FROZEN + ["--dg-mix", "anderson", "--dg-sub-max", "1", "--dg-sub-start", "1"])
                FS = load(os.path.join(out, "fs"), "fs")
                results["vi"] = conservation("(vi-a) two sub-steps of tau/2 per record: converged and conserving", FS[0], nsub=2)
                compare_runs("(vi-b) the sub-stepped state against level 0 (informational: the step's second order)", FS, FZ, 1e-7, 1e-4, soft=True)
            if want("vii"):
                src = os.path.join(out, "fz"); dst = os.path.join(out, "vii")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                snap = os.path.join(dst, "snap"); os.makedirs(snap, exist_ok=True)
                for f in glob.glob(os.path.join(dst, "collapse_fz*")):
                    shutil.move(f, snap)
                slices = sorted(glob.glob(os.path.join(dst, "states", "orbitals_fz_T*.npz")))
                if not slices:
                    results["vii"] = say(False, "(vii) the restart", "no slice in %s" % os.path.join(dst, "states"))
                else:
                    run_engine(a, "engine_collapse_v36k.py", dst, "fz", FROZEN + ["--dg-mix", "plain", "--dg-sub-max", "0", "--resume-slice", slices[0], "--resume-run", os.path.join(snap, "fz")])
                    results["vii"] = bitwise("(vii) the restart from the complex128 slice is bitwise with the uninterrupted frozen-chart run", FZ, load(dst, "fz"))
        if want("viii"):
            results["viii"] = chart_identity(a)
    ok = all(results.values()) and bool(results)
    print("SUMMARY: %s -> %s" % (", ".join("(%s) %s" % (k, "PASS" if v else "FAIL") for k, v in results.items()), "PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
