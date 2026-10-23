#!/usr/bin/env python
"""c6_0_readout_v1.py (busi f52b1152, 2026-10-20, inbox #323): C6-0 of REVIEWA_write_DISSOLVE_v1 (MERGED SPECIFICATION), busi's own readout of
the stored arrays of the three massless histories (16, 2), (32, 2), (32, 4), written from the spec's definitions (CHECKS item 2), not from the
desk's script: the deepest record (the smallest dense min(N - |v|) among records with a trapping zero), R* its dense outer root, the first
record without a zero after it; the mass function M(R*) (s11 npz, node grid, linear interpolation) and its fall; the ball's (eps_tot -
eps_pairs) and the pairs' (eps_pairs) bare shell energy inside R* (sites R < R*) and their Killing form (N_applied eps + v_applied J); the
ball's and the pairs' shell current at each rooted record's own dense outer root; the share of the ball's bare energy inside R* at the deepest
record, the first root-free one, T/r_s = 3 and 5; the per-record wall seconds. Read-only on the inputs; numpy only, seconds.
The added columns of the spec (the number inside the root, the momentum content, the rows' residual at the roots) need the regenerated slices
of C6-1 and are read by dissolve_frozen_c2_v1.py at C6-1's slice clocks; they are not in the stored files.
Usage: c6_0_readout_v1.py --out DIR [--hist name=dir/tag ...]"""
import argparse, json, os
import numpy as np

DEF = {"rs16_ell2": "<internal>/collapse_s11_ball_rs16_ell2_tau2_v35from20",
       "rs32_ell2": "<internal>/collapse_pair_rs32_ell2a_v36",
       "rs32_ell4": "<internal>/collapse_pair_rs32_ell4a_v36b"}


def one(name, base, pr):
    J = json.load(open(base + ".json")); F = np.load(base + "_fields.npz", allow_pickle=True); S = np.load(base + "_s11.npz", allow_pickle=True)
    rs = float(J["args"]["rs"]); recs = [r for r in J["records"] if r.get("E") is not None]
    Tf = np.asarray(F["T"], float); R = np.asarray(F["R"], float); Ts = np.asarray(S["T"], float)
    row = lambda T: int(np.argmin(np.abs(Tf - T)))  # noqa: E731
    srow = lambda T: int(np.argmin(np.abs(Ts - T)))  # noqa: E731
    rooted = [r for r in recs if r.get("trapping_zeros")]
    if not rooted:
        pr("%s: no record with a trapping zero" % name); return
    deep = min(rooted, key=lambda r: r["min_N_minus_absv"])
    after = next(r for r in recs if r["T"] > deep["T"] and not r.get("trapping_zeros"))
    last_rooted = max(r["T"] for r in rooted)
    Rs = float(deep["trapping_zeros"][-1])
    pr("\n== %s (%s): r_s/a %.0f; %d records, %d with a zero (T/r_s %.4f to %.4f); deepest T/r_s %.4f, min(N-|v|) %.4f, roots %s; R* = %.3f a; first root-free after it %.4f; T_c (midpoint) %.5f" % (
        name, base, rs, len(recs), len(rooted), rooted[0]["T_over_rs"], last_rooted / rs, deep["T_over_rs"], deep["min_N_minus_absv"],
        ", ".join("%.3f" % z for z in deep["trapping_zeros"]), Rs, after["T_over_rs"], 0.5 * (last_rooted / rs + after["T_over_rs"])))

    def parts(T, Rc):
        i = row(T); m = R < Rc
        eb = F["eps_tot"][i] - F["eps_pairs"][i]; Jb = F["J_tot"][i] - F["J_pairs"][i]; ep = F["eps_pairs"][i]; Jp = F["J_pairs"][i]
        Na = F["N_applied"][i]; va = F["v_applied"][i]
        j = srow(T); M = float(np.interp(Rc, S["r_nodes"][j], S["M"][j]))
        return {"M": M, "ball_bare": float(eb[m].sum()), "ball_kill": float((Na * eb + va * Jb)[m].sum()), "pairs_bare": float(ep[m].sum()),
                "pairs_kill": float((Na * ep + va * Jp)[m].sum()), "ball_total": float(eb.sum())}
    a_, b_ = parts(deep["T"], Rs), parts(after["T"], Rs)
    dM = a_["M"] - b_["M"]
    pr("mass fall inside R* from %.4f to %.4f: dM %.4f (%.1f percent of %.3f)" % (deep["T_over_rs"], after["T_over_rs"], dM, 100 * dM / a_["M"], a_["M"]))
    for k in ("ball_bare", "ball_kill", "pairs_bare", "pairs_kill"):
        d = a_[k] - b_[k]
        pr("  %-10s inside R*: %.4f -> %.4f, fall %.4f (%.2f of dM)" % (k, a_[k], b_[k], d, d / dM))
    pr("current at each rooted record's own dense outer root (ball; pairs), outward positive as stored:")
    cur = []
    for r in rooted:
        i = row(r["T"]); Rt = float(r["trapping_zeros"][-1])
        jb = float(np.interp(Rt, R, F["J_tot"][i] - F["J_pairs"][i])); jp = float(np.interp(Rt, R, F["J_pairs"][i])); cur.append((r["T_over_rs"], jb, jp))
    pr("  " + "; ".join("%.4f %+.3f (%+.1e)" % c for c in cur))
    pr("  ball current positive on %d of %d rooted records; smallest %+.3f; pairs' |current| at most %.1e" % (sum(1 for c in cur if c[1] > 0), len(cur), min(c[1] for c in cur), max(abs(c[2]) for c in cur)))
    shares = []
    for T in (deep["T"], after["T"], 3.0 * rs, 5.0 * rs):
        if T <= Tf.max() + 1e-9:
            p = parts(T, Rs); shares.append("%.4f: %.3f" % (T / rs, p["ball_bare"] / p["ball_total"]))
    pr("share of the ball's bare energy inside R* (T/r_s: share): " + ", ".join(shares))
    pe = [(r["T_over_rs"], parts(r["T"], float(r["trapping_zeros"][-1]))) for r in rooted]
    worst = max(pe, key=lambda x: abs(x[1]["pairs_bare"]) / x[1]["M"])
    pr("pairs' bare energy inside the record's own outer root: largest |.|/M %.4f at %.4f (%.4f against M %.3f)" % (abs(worst[1]["pairs_bare"]) / worst[1]["M"], worst[0], worst[1]["pairs_bare"], worst[1]["M"]))
    sec = np.array([r.get("seconds") or np.nan for r in recs], float)
    w = np.array([1.0 <= r["T_over_rs"] <= 2.5 for r in recs])
    pr("pace (records' own seconds): mean %.1f median %.1f max %.1f; mean over 1.0 to 2.5: %.1f" % (np.nanmean(sec), np.nanmedian(sec), np.nanmax(sec), np.nanmean(sec[w])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--hist", nargs="*", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    H = dict(x.split("=", 1) for x in a.hist) if a.hist else DEF
    L = open(os.path.join(a.out, "c6_0_readout_v1.log"), "w")
    pr = lambda s: (L.write(s + "\n"), print(s, flush=True))  # noqa: E731
    for n, b in H.items():
        try:
            one(n, b, pr)
        except Exception as e:  # noqa: BLE001
            pr("== %s FAILED: %r" % (n, e))
    L.close()


if __name__ == "__main__":
    main()
