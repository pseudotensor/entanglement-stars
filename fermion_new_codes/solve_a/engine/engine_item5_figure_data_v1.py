#!/usr/bin/env python
"""engine_item5_figure_data_v1.py (builder analyst; inbox #9 item 4: "the item 5 figure file from the delivered records, no new run: the
production's clock and flow profiles at 3 clock times with the trapping zero, and the first surface's radius against the clock at infinity
with the stepping ladder's members and the 2GE line, to the clock the ladders reproduce and no further"). The data half (the figure is
engine_item5_figure_v1.py, which only loads this npz). Inputs: the production's delivered records data (COLLAPSE_records_v1.1_c19_rs64_data.npz:
the sites R, the record times T and the solved lapse N and flow v at the sites per record) and the stepping ladder's members' run jsons
(label=json|tau, the production first). Choices fixed here (builder's, labelled in the npz and the summary): the reproduced clock range
is from the earliest first formation among the members (the first record holding a trapping zero) to the earliest end of a first episode
among them (the first record after the formation holding none), the members' R_t and 2 g E plotted on it only; the three profile times are
the production's records at the first formation, at the record nearest the middle of the reproduced range and at the last record before
the range's end, each holding a trapping zero; the profiles are drawn to --rmax-over-rs. The clock at infinity is the records' flat-slice
time T (the exterior's Killing time). Every number in the summary is printed by the script. Args: --data --member (repeated:
label=json|tau) --rmax-over-rs (3) --out-npz --summary."""
import argparse
import json

import numpy as np


def first_episode(R):
    f = next((r for r in R if r.get("trapping_zeros")), None)
    if f is None:
        return None, None
    e = next((r["T_over_rs"] for r in R if r["T_over_rs"] > f["T_over_rs"] and not r.get("trapping_zeros")), None)
    return f["T_over_rs"], e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--rmax-over-rs", type=float, default=3.0)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    Z = np.load(a.data, allow_pickle=True)
    Rs, T, N, v, g, rs = np.asarray(Z["R"], float), np.asarray(Z["T"], float), np.asarray(Z["N"], float), np.asarray(Z["v"], float), float(Z["g"]), float(Z["rs"])
    mem = []
    for s in a.member:
        lab, rest = s.split("=", 1)
        p, tau = rest.split("|")
        J = json.load(open(p))
        R = sorted([r for r in J["records"] if "E" in r and "T_over_rs" in r], key=lambda r: r["T_over_rs"])
        f, e = first_episode(R)
        mem.append({"label": lab, "path": p, "tau": float(tau), "R": R, "form": f, "end": e, "script": J.get("script")})
    t0 = min(m["form"] for m in mem if m["form"] is not None)
    ends = [m["end"] for m in mem if m["end"] is not None]
    t1 = min(ends) if ends else min(m["R"][-1]["T_over_rs"] for m in mem)
    Tr = T / rs
    P = mem[0]
    prodR = {round(r["T_over_rs"], 6): r for r in P["R"]}
    cand = [t for t in Tr if P["form"] - 1e-9 <= t < t1 - 1e-9 and round(t, 6) in prodR and prodR[round(t, 6)].get("trapping_zeros")]
    assert cand, "no production record with a trapping zero in the reproduced range"
    mid = 0.5 * (t0 + t1)
    picks = [cand[0], min(cand, key=lambda t: abs(t - mid)), cand[-1]]
    out = {"R_sites": Rs, "rs": rs, "g": g, "t_range": np.array([t0, t1]), "rmax_over_rs": a.rmax_over_rs}
    lines = ["## The item 5 figure's data (engine_item5_figure_data_v1.py; inbox #9 item 4; from the delivered records, no new run)", "",
             "- the production's records: %s (r_s %g, g %g; %d records, the solved lapse N and flow v at %d sites)" % (a.data.split("/")[-1], rs, g, len(T), len(Rs)),
             "- the reproduced clock range (builder's rule: the earliest first formation to the earliest first-episode end among the stepping ladder's members): T/rs %.3f to %.3f" % (t0, t1)]
    for i, t in enumerate(picks):
        k = int(np.argmin(np.abs(Tr - t)))
        rec = prodR[round(t, 6)]
        sel = Rs <= a.rmax_over_rs * rs
        out["profile_T_%d" % i] = t
        out["profile_N_%d" % i] = N[k][sel]
        out["profile_v_%d" % i] = v[k][sel]
        out["profile_R_%d" % i] = Rs[sel]
        out["profile_Rt_%d" % i] = float(rec["R_t"])
        out["profile_rin_%d" % i] = float(rec["r_in"]) if rec.get("r_in") is not None else np.nan
        lines.append("- profile %d: T/rs %.3f (the record's R_t %.3f, r_in %s, 2 g E %.3f, min(N - |v|) %.4f at r %.2f)" % (
            i + 1, t, rec["R_t"], "%.3f" % rec["r_in"] if rec.get("r_in") is not None else "none", 2 * g * rec["E"], rec["min_N_minus_absv"], rec["r_min_N_minus_absv"]))
    for j, m in enumerate(mem):
        rr = [r for r in m["R"] if t0 - 1e-9 <= r["T_over_rs"] < t1 - 1e-9]
        Tm = np.array([r["T_over_rs"] for r in rr]); Rt = np.array([r["R_t"] if r.get("trapping_zeros") and r.get("R_t") is not None else np.nan for r in rr])
        E2 = np.array([2 * g * r["E"] for r in rr])
        out["member_label_%d" % j] = m["label"]; out["member_tau_%d" % j] = m["tau"]; out["member_T_%d" % j] = Tm; out["member_Rt_%d" % j] = Rt; out["member_2gE_%d" % j] = E2
        lines.append("- member %s (tau %g, %s): first formation T/rs %s, first episode ends %s; %d records in the range, R_t from %s to %s, 2 g E from %.3f to %.3f" % (
            m["label"], m["tau"], m["script"], "%.3f" % m["form"] if m["form"] is not None else "none", "%.3f" % m["end"] if m["end"] is not None else "none",
            len(rr), "%.3f" % np.nanmin(Rt) if np.isfinite(Rt).any() else "none", "%.3f" % np.nanmax(Rt) if np.isfinite(Rt).any() else "none", E2.min(), E2.max()))
    out["n_members"] = len(mem); out["n_profiles"] = len(picks)
    np.savez(a.out_npz, **out)
    lines.append("- every number above is printed by the script; the figure is drawn by engine_item5_figure_v1.py from %s." % a.out_npz.split("/")[-1])
    open(a.summary, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
