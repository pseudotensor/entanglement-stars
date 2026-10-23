#!/usr/bin/env python
"""cmp_symw1_replay_v1.py (busi SYM-W1, inbox #370): the replay's records against the accuracy budget of busi/notes/PERF_SYMW1_build_v1.md sec. 2,
which was written before the test. (A) cheb against eigh (same box CPU, same threads, the same T0038.00 slice). (B) each arm against the production
(R3b H100: the final_40 run copy's json, the T0039.00 mirror's fields/spectra npz and its 39.0 slice).
Per record: every float leaf of the record dict compared (absolute and relative); the gated keys against their budget; every int/bool/str leaf (the
set decisions and counters) required equal except the skip set (walls, eigendecomposition counts, hbar_prop_v36w2, paths, labels); the fields npz
rows (N, v, N_applied, v_applied, eps_tot, J_tot) max |d|; the spectra masks equal and the eigenvalues' max |d|. The 39.0 slices: masks equal, e_k,
the site densities of sea_k and R2_k, the dust weights f_k. Box only; reads, never writes the inputs.
Usage: [SYMW1_REPLAY=<dir with eigh/ and cheb/>] cmp_symw1_replay_v1.py [out json path]"""
import json
import math
import os
import sys

import numpy as np

TAG = "lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc"
RP = os.environ.get("SYMW1_REPLAY", "<internal>/replay")   # the h100-f arms mirrored to the box
PROD_J = "<internal>/collapse_%s.json" % (TAG, TAG)
MIR = "<internal>/T0039.00" % TAG
SKIP_KEYS = {"seconds", "timing", "slice_saved", "eigendecompositions", "hbar_prop_v36w2", "engine_v34", "label", "record_label", "compute_s", "start"}
GATE_ABS = {"E_over_E0": 1e-9, "min_N_minus_absv": 4.6e-9}
ROOTS = {"R_t", "r_in", "R_c", "r_min_N_minus_absv"}            # |d| <= 3.4e-7 (a); trapping_zeros and threshold_radius handled as lists
REL_KEYS = {"N_min", "kappa_T", "H_in", "E_dust_shells", "E_pairs_shells", "pairs_inside_Rt", "pairs_outside_Rt", "misner_sharp_at_Rt", "v_at_Rc", "min_occupied_weight", "E"}
FLOOR = {"E_over_E0": 3.0e-7, "root": 3.4e-6, "min_N_minus_absv": 4.6e-8, "fields": 1e-5}
FIELD_KEYS = ("N", "v", "N_applied", "v_applied", "eps_tot", "J_tot")


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP_KEYS or k.endswith("_s"):
                continue
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(x, list):
        yield p + "#len", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x


def cmp_record(a, b):
    """returns (float diffs {path: (abs, rel)}, discrete differences [(path, a, b)])."""
    La, Lb = dict(leaves(a)), dict(leaves(b))
    fl = {}; disc = []
    for k in sorted(set(La) | set(Lb)):
        if "/dg_v36f" in k and ("/history" in k or "_hist" in k or "/iterations" in k or "/stall_events" in k):
            # the iterate-level traces (one entry per fixed-point iterate): their lengths are the iteration counts, reported separately
            if not (k.endswith("/iterations")):
                continue
        x, y = La.get(k, "<absent>"), Lb.get(k, "<absent>")
        if isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                disc.append((k, x, y))
        elif isinstance(x, int) and isinstance(y, int):
            if x != y:
                disc.append((k, x, y))
        else:
            x, y = float(x), float(y)
            if math.isnan(x) and math.isnan(y):
                continue
            d = abs(x - y)
            fl[k] = (d, d / max(abs(x), abs(y), 1e-300))
    return fl, disc


def gate_record(fl, disc, budget):
    """budget 'A' (the arms) or 'B' (against the production, the floor). Returns (ok, lines)."""
    out = []; ok = True
    e = fl.get("/E_over_E0", (0.0, 0.0))[0]
    lim = GATE_ABS["E_over_E0"] if budget == "A" else FLOOR["E_over_E0"]
    ok &= e <= lim; out.append("E/E0 |d| %.2e (<= %.1e)" % (e, lim))
    rl = 3.4e-7 if budget == "A" else FLOOR["root"]
    rd = max([fl[k][0] for k in fl if k.lstrip("/") in ROOTS or k.startswith("/trapping_zeros[") or k.startswith("/threshold_radius/")] or [0.0])
    ok &= rd <= rl; out.append("roots |d| max %.2e (<= %.1e)" % (rd, rl))
    m = fl.get("/min_N_minus_absv", (0.0, 0.0))[0]
    ml = GATE_ABS["min_N_minus_absv"] if budget == "A" else FLOOR["min_N_minus_absv"]
    ok &= m <= ml; out.append("min(N-|v|) |d| %.2e (<= %.1e)" % (m, ml))
    rsh = fl.get("/R_sh", (0.0, 0.0))[0]
    ok &= rsh == 0.0; out.append("R_sh |d| %.1e (== 0)" % rsh)
    if budget == "A":
        rr = max([fl["/" + k][1] for k in REL_KEYS if "/" + k in fl] or [0.0])
        ok &= rr <= 1e-6; out.append("named observables rel max %.2e (<= 1e-6)" % rr)
    iters = [(k, x, y) for k, x, y in disc if k.endswith("/iterations") or k == "/iterations"]
    hard = [(k, x, y) for k, x, y in disc if (k, x, y) not in iters]
    ok &= not hard
    out.append("set decisions and counters: %d differences%s; iteration counts differing at %d entries%s" % (
        len(hard), (" " + "; ".join("%s: %r vs %r" % (k, x, y) for k, x, y in hard[:6])) if hard else "", len(iters),
        (" " + "; ".join("%s: %r vs %r" % (k, x, y) for k, x, y in iters[:4])) if iters else ""))
    big = sorted(((v[1], k, v[0]) for k, v in fl.items()), reverse=True)[:3]
    out.append("largest relative float differences (reported, not gated): " + "; ".join("%s %.1e (abs %.1e)" % (k, r, d) for r, k, d in big))
    return ok, out


def rows_by_T(Z):
    T = np.asarray(Z["T"], float)
    return {round(float(t) / 64.0, 6): i for i, t in enumerate(T)}


def cmp_npz(ZA, ZB, ts, keys):
    ia, ib = rows_by_T(ZA), rows_by_T(ZB)
    res = {}
    for t in ts:
        if t in ia and t in ib:
            res[t] = {k: float(np.max(np.abs(ZA[k][ia[t]].astype(float) - ZB[k][ib[t]].astype(float)))) for k in keys if k in ZA.files and k in ZB.files}
    return res


def slice_cmp(pa, pb):
    A = np.load(pa, allow_pickle=True); B = np.load(pb, allow_pickle=True)
    ks = [int(k.split("_")[1]) for k in A.files if k.startswith("mask_")]
    out = {"masks_equal": True, "e": 0.0, "dens_sea": 0.0, "dens_R2": 0.0, "f": 0.0, "nodes": 0.0, "level_equal": None}
    for k in ks:
        out["masks_equal"] &= bool(np.array_equal(A["mask_%d" % k], B["mask_%d" % k]))
        out["e"] = max(out["e"], float(np.max(np.abs(A["e_%d" % k] - B["e_%d" % k]))))
        for nm, key in (("dens_sea", "sea_%d"), ("dens_R2", "R2_%d")):
            da = np.sum(np.abs(A[key % k]) ** 2, axis=1); db = np.sum(np.abs(B[key % k]) ** 2, axis=1)
            out[nm] = max(out[nm], float(np.max(np.abs(da - db))))
        fa, fb = A["f_%d" % k], B["f_%d" % k]
        if fa.shape == fb.shape and fa.size:
            out["f"] = max(out["f"], float(np.max(np.abs(fa - fb))))
        elif fa.shape != fb.shape:
            out["masks_equal"] = False
    out["nodes"] = max(float(np.max(np.abs(A["dg_nodes_N"] - B["dg_nodes_N"]))), float(np.max(np.abs(A["dg_nodes_v"] - B["dg_nodes_v"]))))
    out["level_equal"] = int(A["dg_level_v36k"]) == int(B["dg_level_v36k"])
    return out


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RP, "cmp_symw1_replay_v1.json")
    J = {arm: json.load(open(os.path.join(RP, arm, "collapse_%s.json" % TAG)))["records"] for arm in ("eigh", "cheb")}
    P = json.load(open(PROD_J))["records"]
    byT = lambda R: {round(r["T_over_rs"], 6): r for r in R}
    Je, Jc, Jp = byT(J["eigh"]), byT(J["cheb"]), byT(P)
    ts = sorted(t for t in set(Je) & set(Jc) if t > 38.0 + 1e-9)
    print("records replayed in both arms: %s" % ts)
    allok = True; rep = {"records": {}}
    Z = {arm: {k: np.load(os.path.join(RP, arm, "collapse_%s_%s.npz" % (TAG, k)), allow_pickle=True) for k in ("fields", "spectra")} for arm in ("eigh", "cheb")}
    Z["prod"] = {k: np.load(os.path.join(MIR, "collapse_%s_%s.npz" % (TAG, k)), allow_pickle=True) for k in ("fields", "spectra")}
    for t in ts:
        print("== T/rs %.4f" % t)
        ent = {}
        for name, a, b, bud in (("A cheb vs eigh", Jc[t], Je[t], "A"), ("B eigh vs prod", Je[t], Jp.get(t), "B"), ("B cheb vs prod", Jc[t], Jp.get(t), "B")):
            if b is None:
                print("  %s: no production record" % name); continue
            fl, disc = cmp_record(a, b)
            ok, lines = gate_record(fl, disc, bud)
            allok &= ok
            print("  %s: %s" % (name, "PASS" if ok else "FAIL"))
            for l_ in lines:
                print("     " + l_)
            ent[name] = {"ok": ok, "lines": lines}
        for name, A_, B_, lim in (("A cheb vs eigh", "cheb", "eigh", 1e-7), ("B eigh vs prod", "eigh", "prod", FLOOR["fields"]), ("B cheb vs prod", "cheb", "prod", FLOOR["fields"])):
            f = cmp_npz(Z[A_]["fields"], Z[B_]["fields"], [t], FIELD_KEYS).get(t)
            ks = [k for k in Z[A_]["spectra"].files if k.startswith("mask_")]
            sp = cmp_npz(Z[A_]["spectra"], Z[B_]["spectra"], [t], ks + [k.replace("mask", "e") for k in ks]).get(t)
            if f is None or sp is None:
                print("  %s fields/spectra: row missing" % name); continue
            fmax = max(f.get("N", 0), f.get("v", 0))
            mask_ok = all(sp[k] == 0.0 for k in ks)
            ok = fmax <= lim and mask_ok
            allok &= ok
            print("  %s fields: %s: max |dN|,|dv| %.2e (<= %.0e); N_applied %.1e v_applied %.1e eps_tot %.1e J_tot %.1e; spectra masks %s, e max |d| %.1e" % (
                name, "PASS" if ok else "FAIL", fmax, lim, f.get("N_applied", 0), f.get("v_applied", 0), f.get("eps_tot", 0), f.get("J_tot", 0),
                "equal" if mask_ok else "DIFFER", max(sp[k.replace("mask", "e")] for k in ks)))
            ent[name + " fields"] = {"ok": ok, "f": f, "e_max": max(sp[k.replace("mask", "e")] for k in ks), "masks_equal": mask_ok}
        rep["records"]["%.4f" % t] = ent
    s39 = {arm: os.path.join(RP, arm, "states", "orbitals_%s_T0039.00.npz" % TAG) for arm in ("eigh", "cheb")}
    s39["prod"] = os.path.join(MIR, "orbitals_%s_T0039.00.npz" % TAG)
    if os.path.exists(s39["eigh"]) and os.path.exists(s39["cheb"]):
        for name, A_, B_ in (("A cheb vs eigh", "cheb", "eigh"), ("B eigh vs prod", "eigh", "prod"), ("B cheb vs prod", "cheb", "prod")):
            s = slice_cmp(s39[A_], s39[B_])
            ok = s["masks_equal"] and s["level_equal"]
            allok &= ok
            print("slice 39.0 %s: %s: masks %s, level %s; e max |d| %.1e, sea densities %.1e, R2 densities %.1e, dust weights %.1e, dg nodes %.1e" % (
                name, "PASS" if ok else "FAIL", "equal" if s["masks_equal"] else "DIFFER", "equal" if s["level_equal"] else "DIFFER", s["e"], s["dens_sea"], s["dens_R2"], s["f"], s["nodes"]))
            rep["slice39 " + name] = s
    else:
        print("slice 39.0: not yet written by both arms")
    rep["verdict"] = "PASS" if (allok and ts) else "FAIL"
    print("VERDICT over %d records: %s" % (len(ts), rep["verdict"]))
    json.dump(rep, open(dst, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
