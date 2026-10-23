#!/usr/bin/env python
"""time_symw1_replay_v1.py (busi SYM-W1): the replaced step's time per record from the two arms' jsons. For each record: the dg step's wall, its
iterations and eigendecompositions, and from the sub-steps' hbar_prop_v36w2 entries the propagation blocks (one per fixed-point iterate, all 10
channels), their wall and the Chebyshev orders. Then per arm the means, the block's factor (eigh / cheb), and the per-record saving.
Usage: [SYMW1_REPLAY=<dir>] time_symw1_replay_v1.py"""
import json
import os

TAG = "lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc"
RP = os.environ.get("SYMW1_REPLAY", "<internal>/replay")


def rows(arm):
    R = json.load(open(os.path.join(RP, arm, "collapse_%s.json" % TAG)))["records"]
    out = []
    for r in R:
        if r["T_over_rs"] <= 38.0 + 1e-9:
            continue
        dg = r.get("dg_v36f") or {}
        subs = [s for a in (dg.get("attempt_list") or []) for s in (a.get("substeps") or [])] or (dg.get("substeps") or [])
        hb = [s.get("hbar_prop_v36w2") for s in subs if s.get("hbar_prop_v36w2")]
        out.append({"T": r["T_over_rs"], "rec_s": r.get("seconds"), "dg_s": dg.get("seconds"), "it": dg.get("iterations"), "eigh": dg.get("eigendecompositions"),
                    "blocks": sum(h["blocks"] for h in hb), "hb_s": sum(h["seconds"] for h in hb), "calls": sum(h["cheb_calls"] for h in hb),
                    "Ksum": sum(h["cheb_K_sum"] for h in hb), "Kmax": max([h["cheb_K_max"] for h in hb] or [0]), "tail": max([h["cheb_tail_max"] for h in hb] or [0.0]),
                    "nsub": dg.get("nsub"), "attempts": dg.get("attempts")})
    return out


def main():
    A = {arm: rows(arm) for arm in ("eigh", "cheb")}
    for arm in A:
        print("== %s" % arm)
        for q in A[arm]:
            print("  T/rs %.4f: record %.1f s, dg step %.1f s (nsub %s, attempts %s, %s iterations, %s eigh); H-bar blocks %d in %.2f s (%.3f s a block)%s" % (
                q["T"], q["rec_s"] or 0, q["dg_s"] or 0, q["nsub"], q["attempts"], q["it"], q["eigh"], q["blocks"], q["hb_s"], q["hb_s"] / max(q["blocks"], 1),
                ("; Chebyshev calls %d, order mean %.1f max %d, tail max %.1e" % (q["calls"], q["Ksum"] / q["calls"], q["Kmax"], q["tail"])) if q["calls"] else ""))
    ts = sorted(set(q["T"] for q in A["eigh"]) & set(q["T"] for q in A["cheb"]))
    if not ts:
        return
    S = {arm: {k: sum(q[k] or 0 for q in A[arm] if q["T"] in ts) for k in ("dg_s", "rec_s", "blocks", "hb_s", "eigh", "it")} for arm in A}
    n = len(ts)
    be, bc = S["eigh"]["hb_s"] / S["eigh"]["blocks"], S["cheb"]["hb_s"] / S["cheb"]["blocks"]
    print("over %d common records: the propagation block (10 channels) %.3f s (eigh route) vs %.3f s (Chebyshev): factor %.1fx; per channel call %.4f vs %.4f s" % (n, be, bc, be / bc, be / 10, bc / 10))
    print("  per record: H-bar blocks %.1f (eigh arm) / %.1f (cheb arm); propagation %.1f s vs %.1f s; dg step %.1f s vs %.1f s (%+.1f percent); record wall %.1f vs %.1f s (%+.1f percent); eigh %.0f vs %.0f" % (
        S["eigh"]["blocks"] / n, S["cheb"]["blocks"] / n, S["eigh"]["hb_s"] / n, S["cheb"]["hb_s"] / n, S["eigh"]["dg_s"] / n, S["cheb"]["dg_s"] / n,
        100 * (S["cheb"]["dg_s"] / S["eigh"]["dg_s"] - 1), S["eigh"]["rec_s"] / n, S["cheb"]["rec_s"] / n, 100 * (S["cheb"]["rec_s"] / S["eigh"]["rec_s"] - 1), S["eigh"]["eigh"] / n, S["cheb"]["eigh"] / n))
    print("  the same-iterate saving (eigh arm's blocks x (block_eigh - block_cheb)) per record: %.1f s = %.1f percent of the eigh arm's dg step" % (
        S["eigh"]["blocks"] / n * (be - bc), 100 * S["eigh"]["blocks"] * (be - bc) / S["eigh"]["dg_s"]))


if __name__ == "__main__":
    main()
