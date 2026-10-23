#!/usr/bin/env python
"""cvread_ii_v1.py (busi lane, inbox #424; ReviewA_cvread_v1.md sec. 4 (ii)): the Landau-Zener sub-step counts per channel and per record,
as the engine logs them (record["continuation"][kappa]["nsub"], with tau_LZ, norm_dH_G, mode; record["nsub_max"]), on both sides
(X = CV2048, P = job (b)), beside the decisions: per channel n_pocket, inversions, mask_equals_negative_set, verified, rotated_blocks,
partly_occupied_blocks, n_intermediate; per record inversions_total, n_pocket, mask_equals_negative_set; and the comparator's count of
differing decision leaves (chainval_cmp_v1 per_record n_decisions, from its json).
The engine's rule (engine_ops_v3.continue_sea_v3): nsub = max(1, round(c_LZ sqrt(tau ||dH||_G))) when tau_LZ = 1/(c_LZ sqrt(||dH||_G/tau))
<= tau, else 1. The column 'arg' is c_LZ sqrt(tau ||dH||_G) (the number rounded), so a flip near a half-integer shows.

  python cvread_ii_v1.py X_DIR/X_TAG P_DIR/P_TAG --rs 128 --kits KITS_CODE_DIR --cmp CMP_JSON --nodes-npz CVREAD_I_NPZ --out OUT.json
"""
import argparse
import json
import math
import sys

import numpy as np

CH_KEYS = ("n_pocket", "inversions", "mask_equals_negative_set", "verified", "rotated_blocks", "partly_occupied_blocks", "n_intermediate")
REC_KEYS = ("inversions_total", "n_pocket", "mask_equals_negative_set", "nsub_max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("X"); ap.add_argument("P")
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--kits", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--nodes-npz", dest="nodes_npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.kits)
    import gate_common_v1 as GC
    rs = a.rs
    xd, xt = GC.split_spec(a.X); pd, pt = GC.split_spec(a.P)
    JX, _ = GC.load_run(xd, xt); JP, _ = GC.load_run(pd, pt)
    tau = float(JX["args"]["tau"]); cLZ = float(JX["args"]["c_LZ"])
    assert tau == float(JP["args"]["tau"]) and cLZ == float(JP["args"]["c_LZ"])
    C = json.load(open(a.cmp))
    ndec = {round(float(k), 6): v.get("n_decisions") for k, v in C["per_record"].items()}
    S = np.load(a.nodes_npz)["sweep"]
    dN_in = {round(float(r[0]), 6): float(r[2]) for r in S}
    RX, RP = GC.recs_by_T(JX, rs), GC.recs_by_T(JP, rs)
    common = sorted(set(RX) & set(RP))
    print("== cvread_ii_v1: X %s, P %s; tau %g a, c_LZ %g; %d common records" % (xt, pt, tau, cLZ, len(common)))
    rows = []; tot = {"X": 0, "P": 0}; nrec_diff = 0; nch_diff = 0; dec_match_where_diff = []
    for t in common:
        x, p = RX[t], RP[t]
        cx, cp = x.get("continuation") or {}, p.get("continuation") or {}
        chs = sorted(set(cx) | set(cp), key=lambda s: int(s))
        per = []
        for k in chs:
            u, w = cx.get(k) or {}, cp.get(k) or {}
            nx, np_ = u.get("nsub"), w.get("nsub")
            tot["X"] += int(nx or 0); tot["P"] += int(np_ or 0)
            ax = cLZ * math.sqrt(tau * u["norm_dH_G"]) if u.get("norm_dH_G") is not None else None
            aP = cLZ * math.sqrt(tau * w["norm_dH_G"]) if w.get("norm_dH_G") is not None else None
            per.append({"ch": int(k), "nsub": (nx, np_), "arg": (ax, aP), "tau_LZ": (u.get("tau_LZ"), w.get("tau_LZ")), "mode": (u.get("mode"), w.get("mode")),
                        "dec": {q: (u.get(q), w.get(q)) for q in CH_KEYS}})
        diff = [c for c in per if c["nsub"][0] != c["nsub"][1]]
        recd = {q: (x.get(q), p.get(q)) for q in REC_KEYS}
        row = {"T": t, "channels": per, "nsub_differs": [c["ch"] for c in diff], "record": recd, "n_decisions": ndec.get(t),
               "certified": (GC.certified(x), GC.certified(p)), "nodes_dN_in2R": dN_in.get(t)}
        rows.append(row)
        if diff:
            nrec_diff += 1; nch_diff += len(diff)
            same = all(c["dec"][q][0] == c["dec"][q][1] for c in diff for q in CH_KEYS if q not in ("verified",))
            dec_match_where_diff.append((t, same))
    # print: every record, compact; then the differing ones in full
    print("-- per record: T | nsub per channel X (P where different) | nsub_max X/P | sum nsub X/P | inversions X/P | pockets X/P | mask=neg X/P | decisions (cmp) | |dN| nodes in 2R")
    for r in rows:
        nx = [c["nsub"][0] for c in r["channels"]]; npp = [c["nsub"][1] for c in r["channels"]]
        s = " ".join(("%d" % a_) if a_ == b_ else ("%d(%d)" % (a_, b_)) for a_, b_ in zip(nx, npp))
        rd = r["record"]
        print(" %s%.4f | %s | %s/%s | %d/%d | %s/%s | %s/%s | %s/%s | %s | %s%s" % (
            "*" if all(r["certified"]) else " ", r["T"], s, rd["nsub_max"][0], rd["nsub_max"][1], sum(nx), sum(npp), rd["inversions_total"][0], rd["inversions_total"][1],
            rd["n_pocket"][0], rd["n_pocket"][1], rd["mask_equals_negative_set"][0], rd["mask_equals_negative_set"][1], r["n_decisions"],
            ("%.2e" % r["nodes_dN_in2R"]) if r["nodes_dN_in2R"] is not None else "-", "   <- nsub differs in channel(s) %s" % r["nsub_differs"] if r["nsub_differs"] else ""))
    print("-- the records where nsub differs, channel by channel (arg = c_LZ sqrt(tau ||dH||_G), the number the engine rounds):")
    for r in rows:
        for c in r["channels"]:
            if c["ch"] in r["nsub_differs"]:
                print("  T %.4f ch %d: nsub X %s P %s; arg X %.4f P %.4f; tau_LZ X %.3f P %.3f a; mode X '%s' P '%s'; decisions X/P: %s" % (
                    r["T"], c["ch"], c["nsub"][0], c["nsub"][1], c["arg"][0], c["arg"][1], c["tau_LZ"][0], c["tau_LZ"][1], c["mode"][0], c["mode"][1],
                    ", ".join("%s %s/%s" % (q, c["dec"][q][0], c["dec"][q][1]) for q in CH_KEYS)))
    # whole-run decision agreement on the comparable per-channel counts
    agree = {q: sum(1 for r in rows for c in r["channels"] if c["dec"][q][0] == c["dec"][q][1]) for q in CH_KEYS}
    nchan = sum(len(r["channels"]) for r in rows)
    print("-- totals: records %d, channel-records %d; nsub summed X %d P %d; records with any nsub difference %d (channel-records %d)" % (len(rows), nchan, tot["X"], tot["P"], nrec_diff, nch_diff))
    print("   per-channel decision counts equal X/P (of %d channel-records): %s" % (nchan, ", ".join("%s %d" % (q, agree[q]) for q in CH_KEYS)))
    print("   record-level equal: %s" % ", ".join("%s %d/%d" % (q, sum(1 for r in rows if r["record"][q][0] == r["record"][q][1]), len(rows)) for q in REC_KEYS))
    print("   at the nsub-differing records the per-channel decisions (pockets, inversions, mask = negative set, rotated, partly occupied, intermediate) are equal: %s" % dec_match_where_diff)
    json.dump({"rows": rows, "totals": tot, "n_records_nsub_differs": nrec_diff, "n_channel_records_nsub_differs": nch_diff, "agree": agree,
               "dec_match_where_diff": dec_match_where_diff}, open(a.out, "w"), indent=1, default=str)
    print("written %s" % a.out)


if __name__ == "__main__":
    main()
