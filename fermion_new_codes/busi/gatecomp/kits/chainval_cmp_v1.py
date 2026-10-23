#!/usr/bin/env python
"""chainval_cmp_v1.py (busi lane, GATECOMP kit (i), inbox #400; ReviewA_gatecomp_v1.md sec. 2 (i)): the formation chain's validation.
X = ISO at r_s/a 128 on a 2048-site chain (16 r_s), fresh from T = 0; P = job (b)'s records (ISO_128_s4 on 4096 sites, 32 r_s: the nkqh2
history 0 to 5.0 carried in the RD-2 line's json, then the RD-2 line 5.0625 to its end at 12.0625). Matched clocks: the common record
clocks (both runs step tau = 8 a). reviewer A: "agreement to the record floor".

The record floor (printed and gated): tolcmp_v3's C lines, the solve lane's cross-device regression floor (STATUS #53; the #200 H1 floor):
E/E0 3.0e-7, roots 3.4e-6 a, depth min(N - |v|) 4.6e-8, site fields 2.5e-5; R_sh equal; converged flags and R_t presence equal.
Per record the json leaves are compared by tolcmp_v3.cmp_record (imported, unchanged: its classes E, E_over_E0, roots, argmin, R_sh,
depth, named, residual, indicator, trace, other, decisions, counters). Gated: E_over_E0, roots, depth, R_sh. Reported: everything else;
decisions are reported, not gated, because a 2048-site chain differs from a 4096-site chain in every chain-sized label by construction
(r_collar = nsites, the sea's level counts n_sea, the spectrum's list lengths, masks).
npz rows: P's rows are cropped to X's sites (the first nsites_X sites; the grids are equal there, checked) and, separately, to r <= --rcmp-rs
r_s (default 8: the matter and its near exterior, inside the collar of the shorter chain). Site fields N, v, N_applied, v_applied are gated
at 2.5e-5 on r <= rcmp; eps_tot, J_tot, eps_pairs, J_pairs and the dense nodes are reported. Spectra are not comparable (the chain's levels
depend on its length) and are not read.
Gate quantities (gate_common_v1): T_birth and R_birth for both runs (reviewer A sec. 3.2), sigma ||K||_op and sigma |H| over r <= 2 R_t per record
for both, their difference, and T_gate. Clocks past P's last certified record are printed and not gated.

  chainval_cmp_v1.py X_DIR/X_TAG P_DIR/P_TAG --rs 128 --sigma 4 --tolcmp PATH/tolcmp_v3.py [--rcmp-rs 8] [--fm-dir FIELD_MODULE_DIR] [--out J]
Exit 0 PASS, 1 FAIL, 2 no common certified clocks.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_common_v1 as GC  # noqa: E402

SITE_FIELDS = ("N", "v", "N_applied", "v_applied")
ROW_REPORT = ("eps_tot", "J_tot", "eps_pairs", "J_pairs")


def load_tolcmp(path):
    sp = importlib.util.spec_from_file_location("tolcmp_v3", path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("P")
    ap.add_argument("--rs", type=float, required=True); ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--tolcmp", required=True); ap.add_argument("--rcmp-rs", dest="rcmp_rs", type=float, default=8.0)
    ap.add_argument("--fm-dir", dest="fm_dir", default=None, help="the field module's dir (engine_fields_v16.py = v16d): the six-term on"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TC = load_tolcmp(a.tolcmp)
    LINE = dict(TC.C_LINES)
    xd, xt = GC.split_spec(a.X); pd, pt = GC.split_spec(a.P)
    JX, FX = GC.load_run(xd, xt); JP, FP = GC.load_run(pd, pt)
    rs = a.rs
    print("== chainval_cmp_v1: X %s (nsites %s) against P %s (nsites %s); rs %g, sigma %g; tolcmp %s" % (
        xt, JX["args"].get("nsites"), pt, JP["args"].get("nsites"), rs, a.sigma, os.path.basename(a.tolcmp)))
    adiff = sorted(k for k in set(JX["args"]) | set(JP["args"]) if JX["args"].get(k) != JP["args"].get(k))
    print("args differ in: %s" % ", ".join(adiff))
    print("record floor (tolcmp_v3 C_LINES): %s" % json.dumps(LINE))
    print("head E0: X %.12g P %.12g (rel %.2e); N_B X %s P %s; passes X %s P %s" % (
        JX["head"]["E0"], JP["head"]["E0"], abs(JX["head"]["E0"] - JP["head"]["E0"]) / JP["head"]["E0"], JX["head"].get("N_B"), JP["head"].get("N_B"),
        JX["head"].get("passes"), JP["head"].get("passes")))
    RX, RP = GC.recs_by_T(JX, rs), GC.recs_by_T(JP, rs)
    common = sorted(set(RX) & set(RP))
    gated = [t for t in common if GC.certified(RP[t]) and GC.certified(RX[t])]
    p_last = max([t for t in RP if GC.certified(RP[t])] or [None]) if RP else None
    print("records: X %d (to %.4f), P %d (to %.4f; last certified %s); common %d; gated (both certified) %d" % (
        len(RX), max(RX) if RX else -1, len(RP), max(RP) if RP else -1, p_last, len(common), len(gated)))
    if not gated:
        print("no common certified clocks"); sys.exit(2)
    # --- json leaves per record
    per = {}; dec_keys = {}; fails = []
    for t in common:
        acc = TC.Acc()
        TC.cmp_record(RX[t], RP[t], acc, t)
        p = {c: max([v[0] for v in L] or [0.0]) for c, L in acc.cls.items()}
        p["named_rel"] = max([v[1] for v in acc.cls.get("named", [])] or [0.0])
        p["n_decisions"] = len(acc.decisions)
        for q in acc.decisions:
            k = TC.keyname(q[0][len("%.5f" % t):])
            dec_keys[k] = dec_keys.get(k, 0) + 1
        p["conv"] = (RX[t].get("converged"), RP[t].get("converged"))
        p["Rt_presence"] = ((RX[t].get("R_t") is not None), (RP[t].get("R_t") is not None))
        per[t] = p
    # --- npz rows
    nX = int(JX["args"]["nsites"])
    rowsres = {}
    ok_grid = None
    if FX is not None and FP is not None:
        Rx = np.asarray(FX["R"], float); Rp = np.asarray(FP["R"], float)[:nX]
        ok_grid = bool(np.array_equal(Rx, Rp))
        sel = Rx <= a.rcmp_rs * rs
        IX, IP = GC.rows_by_T(FX, rs), GC.rows_by_T(FP, rs)
        for t in common:
            if t not in IX or t not in IP:
                continue
            d = {}
            for k in SITE_FIELDS + ROW_REPORT:
                if k in FX.files and k in FP.files:
                    x = np.asarray(FX[k][IX[t]], float); y = np.asarray(FP[k][IP[t]], float)[:nX]
                    d[k] = (float(np.max(np.abs(x[sel] - y[sel]))), float(np.max(np.abs(x - y))))
            if "r_nodes" in FX.files and "r_nodes" in FP.files:
                rx = np.asarray(FX["r_nodes"][IX[t]], float); ry = np.asarray(FP["r_nodes"][IP[t]], float)
                n = min(len(rx), len(ry)); same = bool(np.array_equal(rx[:n], ry[:n]))
                if same:
                    selN = rx[:n] <= a.rcmp_rs * rs
                    for k in ("N_nodes", "v_nodes"):
                        x = np.asarray(FX[k][IX[t]], float)[:n]; y = np.asarray(FP[k][IP[t]], float)[:n]
                        d[k] = (float(np.max(np.abs(x[selN] - y[selN]))), float(np.max(np.abs(x - y))))
            rowsres[t] = d
    print("grid: X's sites equal P's first %d sites: %s; comparison radius r <= %g r_s = %g a" % (nX, ok_grid, a.rcmp_rs, a.rcmp_rs * rs))
    # --- gate quantities
    six = GC.load_sixterm(a.fm_dir)
    bX, bP = GC.birth(JX, rs), GC.birth(JP, rs)
    gX = GC.gate_profile(JX, FX, rs, a.sigma, bX, sixterm=six) if FX is not None else None
    gP = GC.gate_profile(JP, FP, rs, a.sigma, bP, sixterm=six) if FP is not None else None
    # --- verdict per gated record
    print("-- per record (gated clocks marked *): E/E0 |d|, roots |d|, depth |d|, R_sh, named rel, decisions, conv X/P, R_t X/P, "
          "site fields |d| r<=rcmp (all common sites), sKop X P |d|")
    gX_by = {r[0]: r for r in (gX or {}).get("rows", [])}; gP_by = {r[0]: r for r in (gP or {}).get("rows", [])}
    worst = {"E_over_E0": 0.0, "roots": 0.0, "depth": 0.0, "fields": 0.0}
    for t in common:
        p = per[t]; rr = rowsres.get(t, {})
        fin = max([rr[k][0] for k in SITE_FIELDS if k in rr] or [0.0]); fall = max([rr[k][1] for k in SITE_FIELDS if k in rr] or [0.0])
        g = t in gated
        bad = []
        if g:
            for c, v in (("E_over_E0", p.get("E_over_E0", 0.0)), ("roots", p.get("roots", 0.0)), ("depth", p.get("depth", 0.0)), ("fields", fin)):
                worst[c] = max(worst[c], v)
                if v > LINE[c]:
                    bad.append(c)
            if p.get("R_sh", 0.0) != 0.0:
                bad.append("R_sh")
            if p["conv"][0] != p["conv"][1]:
                bad.append("converged")
            if p["Rt_presence"][0] != p["Rt_presence"][1]:
                bad.append("R_t presence")
            if bad:
                fails.append((t, bad))
        kx = gX_by.get(t); kp = gP_by.get(t)
        ks = ("%.4f %.4f %.1e" % (kx[2], kp[2], abs(kx[2] - kp[2]))) if (kx and kp) else "-"
        print(" %s%.4f  %.2e  %.2e  %.2e  %s  %.2e  %d  %s/%s  %s/%s  %.2e (%.2e)  %s%s" % (
            "*" if g else " ", t, p.get("E_over_E0", 0.0), p.get("roots", 0.0), p.get("depth", 0.0), p.get("R_sh", 0.0), p["named_rel"], p["n_decisions"],
            p["conv"][0], p["conv"][1], p["Rt_presence"][0], p["Rt_presence"][1], fin, fall, ks, ("  FAIL " + ",".join(bad)) if bad else ""))
    print("-- decisions differing (reported, not gated; chain-sized labels expected): %d keys" % len(dec_keys))
    for k, n in sorted(dec_keys.items(), key=lambda kv: -kv[1])[:30]:
        print("   %-70s %d records" % (k[:70], n))
    print("-- npz rows, max over the gated clocks: |d| on r <= rcmp (on all common sites)")
    for k in SITE_FIELDS + ROW_REPORT + ("N_nodes", "v_nodes"):
        v = [rowsres[t][k] for t in gated if t in rowsres and k in rowsres[t]]
        if v:
            print("   %-10s %.3e (%.3e)%s" % (k, max(x[0] for x in v), max(x[1] for x in v), "  gated at %.1e" % LINE["fields"] if k in SITE_FIELDS else ""))
    print("-- gate quantities (reviewer A sec. 3.1, 3.2; gate_common_v1)")
    for nm, b, gg in (("X", bX, gX), ("P", bP, gP)):
        if b:
            print("   %s birth: bracket [%.4f, %.4f] m %.3e / %.3e; T_birth %.5f (quadratic %s, err %s); R_birth %s a = %s of 2GE0 %.3f; later crossings %d" % (
                nm, b["T_lo"], b["T_hi"], b["m_lo"], b["m_hi"], b["T_birth"], GC.fmt(b["T_quad"], "%.5f"), GC.fmt(b["T_err"], "%.1e"), GC.fmt(b["R_birth"], "%.4f"),
                GC.fmt(b["R_birth_over_2gE"], "%.5f"), b["two_gE0"], b["later_crossings"]))
        else:
            print("   %s birth: no crossing of min(N - |v|) in the records" % nm)
        if gg:
            print("   %s gate: T_gate %s (first record from T_birth - 1 with sigma ||K||_op > 0.1%s); window T_gate - T_birth %s" % (
                nm, GC.fmt(gg["T_gate"], "%.4f"), " or the six-term on the nodes r >= sigma > 0.1" if gg["sixterm"] else "; six-term OPEN (no --fm-dir)", GC.fmt(gg["window"], "%.4f")))
    if gX and gP:
        print("   per record six-term (nodes r >= sigma | r < sigma reported) X / P:")
        for t in sorted(set(r[0] for r in gX["rows"]) & set(r[0] for r in gP["rows"])):
            x = [r for r in gX["rows"] if r[0] == t][0]; y = [r for r in gP["rows"] if r[0] == t][0]
            print("     %.4f  K_op %.4f / %.4f  six %s / %s  centre %s / %s" % (t, x[2], y[2], GC.fmt(x[5], "%.4f"), GC.fmt(y[5], "%.4f"), GC.fmt(x[9], "%.4f"), GC.fmt(y[9], "%.4f")))
    if bX and bP:
        print("   birth X - P: dT %.3e r_s, dR %s a" % (bX["T_birth"] - bP["T_birth"], GC.fmt((bX["R_birth"] - bP["R_birth"]) if (bX["R_birth"] is not None and bP["R_birth"] is not None) else None, "%.3e")))
    print("-- worst over the gated clocks: %s against %s" % (json.dumps(worst), json.dumps(LINE)))
    verdict = "PASS" if not fails else "FAIL"
    print("CHAIN VALIDATION %s at the record floor over %d gated clocks (%.4f to %.4f)%s" % (
        verdict, len(gated), gated[0], gated[-1], "" if not fails else "; first failing records: %s" % fails[:5]))
    ungated = [t for t in common if t not in gated] + [t for t in RX if t not in RP]
    if ungated:
        print("not gated (one side uncertified, or past P's records): %s" % ", ".join("%.4f" % t for t in sorted(set(ungated))))
    if a.out:
        json.dump({"verdict": verdict, "fails": fails, "worst": worst, "line": LINE, "gated": gated, "per_record": {"%.5f" % t: per[t] for t in common},
                   "rows": {"%.5f" % t: rowsres.get(t) for t in common}, "birth": {"X": bX, "P": bP},
                   "gate": {"X": gX, "P": gP}, "decision_keys": dec_keys, "args_differ": adiff}, open(a.out, "w"), indent=1, default=str)
        print("written %s" % a.out)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
