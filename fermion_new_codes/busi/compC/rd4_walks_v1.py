#!/usr/bin/env python
"""rd4_walks_v1.py (busi lane, PERF RD-4 self-test (b)): engine_twofield_v9c6's light_sector_walk_v9 replayed per walk on the stored R2 C0
segment (C64 perf_ab_R2 segment ..._v9c3_..._091433Z, the records 13.0 -> 14.0 and their light channels kappa 1, 2 and the guard 3) in three
arms on the same inputs: LAD the ladder route (--light-walk ladder: v9c5), TAB --light-walk tabulated, AUD tabulated with the audit (every
tier-1/tier-2 verdict also factored by the Cholesky). The inputs per walk as light_continue_sector_v9 forms them: the path from the previous
record's applied fields to the record's (LightPathV9), the start spectrum and sectors at the previous fields and the trial's at the record's
(SYM-C1's sector eigh, as the card A/B's arms run), the start occupation the anchor mask of the previous record's stored anchors (checked
against the stored mask_L), mu_prev the previous record's mu, n_sea the record's; SYM-C2 on (the ladders on K_+), --light-d-min 1e-7,
--light-max-pieces 8192, --light-tol-block 1e-8 (the segment's args). The raw route (the v9c prim wrapper's __wrapped__) is called.

Per walk: the certificate and sector entries of TAB and AUD against LAD (every key but the seconds and the Cholesky counts: pieces, the
piece list, d_min_certified, the crossings with their failure probes s, pieces [s_lo, s_hi], reopening probes, swaps, the occupations before
and after, the anchors at the end, the end occupation), the verdict tiers, the audit's disagreements, the table end checks, the walls; and
LAD against the stored record (pieces, Cholesky count, the walk's own crossings), which is information (the stored run is v9c3 on its own
window entries).

Usage. python rd4_walks_v1.py STAGE OUT_DIR [--gpu 0] [--records all|13.34375,13.625] [--channels 1,2,3] [--grid 2048] [--table-after 0]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SEG = "<internal>/segment_c64s8L_lagged_sector_v9c3_anchor_passage_lz_20261020T091433Z"
SKIP = {"seconds", "cholesky_calls", "cholesky"}


def strip(d):
    return {k: v for k, v in d.items() if k not in SKIP} if isinstance(d, dict) else d


def same(x, y, path, out):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            out.append("%s keys %s" % (path, sorted(set(x) ^ set(y)))); return
        for k in x:
            if k in SKIP:
                continue
            same(x[k], y[k], path + "/" + str(k), out)
        return
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            out.append("%s length %d %d" % (path, len(x), len(y))); return
        for q, (u, v) in enumerate(zip(x, y)):
            same(u, v, "%s[%d]" % (path, q), out)
        return
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            out.append("%s array differs" % path)
        return
    if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
        return
    if x != y:
        out.append("%s %r %r" % (path, x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage"); ap.add_argument("out")
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--records", default="all"); ap.add_argument("--channels", default="1,2,3")
    ap.add_argument("--grid", type=int, default=2048); ap.add_argument("--table-after", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.stage)
    import engine_twofield_v9c6 as E
    E.C5.backend(a.gpu)
    torch = E.BK["torch"]; dev = E.BK["dev"]
    E._SYMC.update({"light": True, "ladder": True, "tol_block": 1e-8})
    E._RD4.update({"grid": int(a.grid), "table_after": int(a.table_after), "device": (int(dev.index or 0) if dev.type == "cuda" else None), "threads": int(os.environ.get("OMP_NUM_THREADS") or 1)})
    t_ = time.time(); worst = E._rd4_selfcheck()
    print("kernels: self-check levels to %.2e in %.1f s on %s" % (worst, time.time() - t_, torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"), flush=True)
    walk = getattr(E.light_sector_walk_v9, "__wrapped__", E.light_sector_walk_v9)
    os.makedirs(a.out, exist_ok=True)
    J = json.load(open(glob.glob(os.path.join(SEG, "*.json"))[0])); R = J["records"]
    Z = np.load(glob.glob(os.path.join(SEG, "*_fields.npz"))[0], allow_pickle=True)
    SP = np.load(glob.glob(os.path.join(SEG, "*_spectra.npz"))[0], allow_pickle=True)
    T = np.asarray([r["T_over_rs"] for r in R], float)
    nsites = int(Z["N_applied"].shape[1])
    sgn_dev, S_host = E.light_S_v36(nsites)
    kap = [int(x) for x in a.channels.split(",")]
    chans = {k: E.LightChan(k, nsites, False, sourcing=(k < 3)) for k in kap}
    if a.records == "all":
        idx = [i for i in range(1, len(R)) if T[i] > 12.99 and R[i].get("E") is not None]
    else:
        idx = [int(np.argmin(np.abs(T - float(x)))) for x in a.records.split(",")]
    rows = []; lines = []; tot = {"walks": 0, "identical": 0, "audit_disagree": 0, "lad_s": 0.0, "tab_s": 0.0, "verdicts": 0}
    for i in idx:
        fp = (np.asarray(Z["N_applied"][i - 1], float), np.asarray(Z["v_applied"][i - 1], float))
        fc = (np.asarray(Z["N_applied"][i], float), np.asarray(Z["v_applied"][i], float))
        for k in kap:
            ch = chans[k]
            c0 = R[i - 1]["light_v36"]["channels"][str(k)]; c1 = R[i]["light_v36"]["channels"][str(k)]
            sec0 = c0.get("sector") or {}; sec1 = c1.get("sector") or {}
            anchors0 = list(sec0.get("anchors") or [c0.get("mu")])
            e0t, U0 = E._symc_eig_light(ch, fp[0], fp[1]); sg0, _n, _i = E.light_sectorize_v36(e0t, U0, sgn_dev, 1e-8)
            e0 = e0t.cpu().numpy().astype(float); sig0 = E.sector_e_v9(sg0); del U0
            e1t, U1 = E._symc_eig_light(ch, fc[0], fc[1]); sg1, _n, _i = E.light_sectorize_v36(e1t, U1, sgn_dev, 1e-8)
            e1 = e1t.cpu().numpy().astype(float); sig1 = E.sector_e_v9(sg1)
            trial = (e1, sig1, U1)
            mask0, _g = E.light_select_sector_v9(e0, sig0, anchors0)
            stored = np.unpackbits(np.asarray(SP["mask_L%d" % k][i - 1], np.uint8))[: 2 * nsites].astype(bool)
            mask_mis = int(np.sum(stored != mask0))
            n_sea = int(c1.get("n_sea") or int(mask0.sum()))
            res = {}
            for arm, mode, audit in (("LAD", "ladder", False), ("TAB", "tabulated", False), ("AUD", "tabulated", True)):
                E._RD4["walk"] = mode; E._RD4["audit"] = audit
                path = E.LightPathV9(ch, fp, fc, sgn_dev, 1e-8, trial)
                t0 = time.time()
                cert, sector, anch, occ = walk(path, S_host, e0, sig0, mask0, trial, n_sea, 1e-7, 8192, 1e-8, mu_prev=c0.get("mu"))
                wall = time.time() - t0
                res[arm] = {"cert": cert, "sector": sector, "anchors_end": anch, "occ_end": (None if occ is None else np.asarray(occ, bool)), "wall": wall,
                            "stats": dict(E._RD4_LAST) if mode == "tabulated" else {}, "eigh": int(path.eigh_n)}
            E._RD4["walk"] = "ladder"; E._RD4["audit"] = False
            L = res["LAD"]; row = {"T": float(T[i]), "kappa": k, "mask0_vs_stored_mismatch": mask_mis, "anchors_start": anchors0}
            for arm in ("TAB", "AUD"):
                diffs = []
                same(L["cert"], res[arm]["cert"], "cert", diffs); same(L["sector"], res[arm]["sector"], "sector", diffs)
                same(L["anchors_end"], res[arm]["anchors_end"], "anchors_end", diffs); same(L["occ_end"], res[arm]["occ_end"], "occ_end", diffs)
                row[arm] = {"differences": diffs[:20], "n_differences": len(diffs), "wall": res[arm]["wall"], "cholesky": res[arm]["cert"].get("cholesky_calls"), "stats": res[arm]["stats"]}
            st = res["TAB"]["stats"]
            nver = int(st.get("verdict_table", 0) + st.get("verdict_count", 0) + st.get("verdict_cholesky", 0))
            cr_l = L["sector"].get("crossings") or []
            cw = sec1.get("crossings_walk") if sec1.get("crossings_walk") is not None else sec1.get("crossings")
            row.update({"LAD": {"pieces": L["cert"].get("pieces"), "cholesky": L["cert"].get("cholesky_calls"), "certified": L["cert"].get("certified"), "why": L["cert"].get("why"),
                                "crossings": [{"s": c_["s"], "piece": c_["piece"], "s_reopen": c_["s_reopen"], "swap": c_["swap"], "flagged": c_["flagged"], "pair_gap_end": c_["pair_gap_end"]} for c_ in cr_l],
                                "anchors_end": len(L["anchors_end"] or []), "wall": L["wall"], "eigh": L["eigh"]},
                        "verdicts_total_tab": nver, "ladder_cholesky_equals_tab_verdicts_plus_reopen": None,
                        "stored": {"pieces": sec1.get("pieces"), "cholesky": sec1.get("cholesky"), "route": sec1.get("route"),
                                   "crossings_walk": [{"s": c_.get("s"), "piece": c_.get("piece"), "s_reopen": c_.get("s_reopen"), "swap": c_.get("swap")} for c_ in (cw or [])]}})
            ident = row["TAB"]["n_differences"] == 0 and row["AUD"]["n_differences"] == 0
            aud = int(sum(v for kk, v in res["AUD"]["stats"].items() if kk.startswith("audit_disagree")))
            tot["walks"] += 1; tot["identical"] += int(ident); tot["audit_disagree"] += aud; tot["lad_s"] += L["wall"]; tot["tab_s"] += res["TAB"]["wall"]; tot["verdicts"] += nver
            line = ("T %.5f k%d: LAD pieces %s chol %s crossings %s (%s) %.2f s | TAB %s chol %d, verdicts table %d count %d cholesky %d, tables %d (%.2f s), end-check fails %d, %.2f s | AUD %s, disagreements %d | stored pieces %s chol %s crossings_walk %s | mask0 vs stored %d" % (
                T[i], k, row["LAD"]["pieces"], row["LAD"]["cholesky"], len(cr_l), ["%.5f swap %s" % (c_["s"], c_["swap"]) for c_ in cr_l], L["wall"],
                "IDENTICAL" if row["TAB"]["n_differences"] == 0 else "DIFFERS %s" % row["TAB"]["differences"][:3], int(res["TAB"]["cert"].get("cholesky_calls") or 0),
                st.get("verdict_table", 0), st.get("verdict_count", 0), st.get("verdict_cholesky", 0), st.get("tables", 0), float(st.get("table_s", 0.0)), st.get("table_end_check_fail", 0), res["TAB"]["wall"],
                "IDENTICAL" if row["AUD"]["n_differences"] == 0 else "DIFFERS %s" % row["AUD"]["differences"][:3], aud,
                row["stored"]["pieces"], row["stored"]["cholesky"], ["%.5f swap %s" % (c_["s"], c_["swap"]) for c_ in row["stored"]["crossings_walk"]], mask_mis))
            print(line, flush=True); lines.append(line); rows.append(row)
            del U1, trial, res
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    summ = ("SUMMARY: %d walks, %d identical in TAB and AUD (every key but seconds and Cholesky counts), audit disagreements %d; ladder walls %.1f s, tabulated walls %.1f s (%.2fx); %d tabulated verdicts" % (
        tot["walks"], tot["identical"], tot["audit_disagree"], tot["lad_s"], tot["tab_s"], tot["lad_s"] / max(tot["tab_s"], 1e-9), tot["verdicts"]))
    print(summ, flush=True); lines.append(summ)
    print("  run counters: %s" % json.dumps({k: v for k, v in sorted(E._RD4_STATS.items())}), flush=True)
    lines.append("run counters: %s" % json.dumps({k: v for k, v in sorted(E._RD4_STATS.items())}))
    with open(os.path.join(a.out, "rd4_walks_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "rd4_walks_v1.json"), "w") as f:
        json.dump({"rows": rows, "total": tot, "stats": E._RD4_STATS}, f, indent=1, default=float)
    ok = tot["identical"] == tot["walks"] and tot["audit_disagree"] == 0
    print("%s rd4_walks_v1" % ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
