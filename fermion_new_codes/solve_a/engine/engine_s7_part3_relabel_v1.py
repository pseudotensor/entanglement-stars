#!/usr/bin/env python
"""engine_s7_part3_relabel_v1.py (builder analyst; inbox #41, the revised S4 part C, verbatim: "C. The delivered part 3 relabelled. From the
lane's own s7_part3_builder_report.md and the window definition of engine spec 36.3: the window's plateau shells n_L and n_R and the 17 to 19
shell bands against the dense and applied outer roots at T/r_s 18 on the r_s/a 32 record; ||Delta C(0)||_op of the windowed rotation (Delta C(0)
= sum_{xy} P_{xy}|x><y| (x) (e^{i theta (chi_x - chi_y) sigma_z} - I), supported on pairs straddling the ramp); the normalized transfer
D_det/||Delta C(0)||_op per theta; the statement whether the bands lie inside the outer root. If inside, the number prints as 'a perturbation of
the correlations across shells n_L to n_R inside the root reached the collar with covariance witness D_det >= 5e-4 to 1.5e-3, a fraction [value]
of its size'; if not, it is a record and nothing from part 3 prints."). Record-only (the CPU, seconds); no mutual information.

Inputs, the lane's own files: engine_s7_twocopy_v2's member json (its head: the window X, w, the run's theta, the label's source bands; its
records: the exact bound beyond R_t(t), the collar D_det) and its stored T0 state (the states npz: the sea, the dust and their occupations per
kappa); the member's applied fields at T0; part A's roots of the same record (engine_single_object_A_v1's json: the dense and applied roots).
Computed: the window chi_X (version 2's function) on the sites, the plateau shells n_L (the last site with chi = 1) and n_R (the first with chi =
0) and the ramp shells; the source bands of the rotated two-copy state against the unrotated (version 2's twocopy_cols and cdens2 on the stored
T0 state, the copy-summed site energy and momentum densities, the shells where each difference exceeds 1e-15 of its raw density's largest
value, the energy and the momentum band apart; P08 sec. 5's count, w + 2 and w + 4 shells for a w-shell ramp, beside them); the margins to the
dense and applied inner and outer roots; ||Delta C(0)||_op = max over the two copy blocks of ||C o (e^{+-i theta Delta chi} - 1)||_op (the
task's sigma_z form; o the entrywise product, Delta chi_xy = chi_x - chi_y; C the channel's state at T0) and, the same object as version 2's
D_det at T0, ||C_+(0) - C_-(0)||_op = ||C o 2 sin(theta Delta chi)||_op (version 2's +-theta pair, sigma_y in copy space), at the run's theta and
at the law's thetas; the normalized transfer, version 2's exact bounds beyond R_t(t) at its records and its collar D_det over each norm.
Args: --v2-json --v2-states --member --a-json --md --out-json. Every number is printed by the script."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import math
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "flow"))
import engine_s7_twocopy_v2 as V2  # noqa: E402

C5 = V2.C5


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def band(diff, raw, R):
    idx = np.nonzero(np.abs(diff) > 1e-15 * max(float(np.max(np.abs(raw))), 1e-300))[0]
    return {"shells": int(len(idx)), "first_R": float(R[idx[0]]) if len(idx) else None, "last_R": float(R[idx[-1]]) if len(idx) else None,
            "R_list": [float(R[i]) for i in idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-json", required=True); ap.add_argument("--v2-states", required=True); ap.add_argument("--member", required=True)
    ap.add_argument("--a-json", required=True); ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    C5.backend(None)
    torch = C5.BK["torch"]
    J2 = json.load(open(a.v2_json)); h2 = J2["head"]
    S = np.load(a.v2_states)
    d, tag = os.path.split(a.member)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag))); ra = J["args"]
    n = int(ra["nsites"]); m = float(ra["m"]); rs = float(ra["rs"])
    R = np.arange(n, dtype=float) + (0.5 if ra.get("half_offset") else 1.0)
    X = float(h2["window"]["X"]); w = float(h2["window"]["width"]); th = float(h2["theta_main"])
    law = [float(x) for x in h2["law_thetas"]]
    chi = V2.window(R, X, w)
    ones = np.nonzero(chi >= 1.0 - 1e-15)[0]; zeros = np.nonzero(chi <= 1e-15)[0]
    nL = float(R[ones[-1]]); nR = float(R[zeros[0]])
    ramp = [float(R[i]) for i in range(len(R)) if 1e-15 < chi[i] < 1.0 - 1e-15]
    A = json.load(open(a.a_json)); rec = A["records"][0]
    roots = {"dense": rec["outer_inner"]["dense"], "applied": rec["outer_inner"]["applied"]}
    out = {"script": "engine_s7_part3_relabel_v1.py", "T0_over_rs": h2["T0_over_rs"], "window": {"X": X, "width": w, "form": h2["window"]["form"], "n_L": nL, "n_R": nR,
           "ramp_shells": ramp, "chi_15_to_23": {("%.0f" % R[i]): float(chi[i]) for i in range(len(R)) if 15 <= R[i] <= 23}}, "theta_run": th, "law_thetas": law,
           "roots": roots, "channels": [], "no_mutual_information": "D15"}
    print("window X %.4f w %g: n_L R %.0f (the last site with chi = 1), n_R R %.0f (the first with chi = 0), the ramp shells %s; roots at T/rs %.3f: dense %s, applied %s" % (
        X, w, nL, nR, ramp, h2["T0_over_rs"], roots["dense"], roots["applied"]), flush=True)
    # the stored label's band (version 2's, the run's theta) per kappa
    lab = {int(l_["kappa"]): l_ for l_ in h2["labels"] if l_.get("role") == "run"}
    for kap in (1, 2):
        ch = C5.Chan(kap, m, n, bool(ra.get("half_offset", 0)))
        sea = torch.as_tensor(S["T0_k%d_sea" % kap]); dust = torch.as_tensor(S["T0_k%d_dust" % kap]); fd = np.asarray(S["T0_k%d_f" % kap], float)
        Phi = torch.cat([sea, dust], dim=1); f = np.concatenate([np.ones(sea.shape[1]), fd]); f2 = np.concatenate([f, f])
        # the source bands of the rotated two-copy state (version 2's construction at the run's theta) against the unrotated
        c = torch.as_tensor(np.repeat(np.cos(th * chi), 2), dtype=torch.complex128)[:, None]; s_ = torch.as_tensor(np.repeat(np.sin(th * chi), 2), dtype=torch.complex128)[:, None]
        Y = V2.twocopy_cols(Phi, c, s_, +1)
        e_rot, j_rot = V2.cdens2(ch, Y, f2)
        e0, j0 = V2.cdens(ch, [(Phi, f)])
        raw_e = 2.0 * e0; raw_j = 2.0 * j0
        be = band(e_rot - 2.0 * e0, raw_e, R); bj = band(j_rot - 2.0 * j0, raw_j, R)
        del Y
        # Delta C(0) of the windowed rotation (sigma_z form, +-theta copy blocks) and version 2's C_+ - C_- (the same object as D_det at T0)
        C = (Phi * torch.as_tensor(f, dtype=torch.complex128)[None, :]) @ Phi.conj().T
        Cn = C.numpy()
        chis = np.repeat(chi, 2)
        dchi = chis[:, None] - chis[None, :]
        norms = []
        for thq in [th] + law:
            Mp = Cn * (np.exp(1j * thq * dchi) - 1.0); Mm = Cn * (np.exp(-1j * thq * dchi) - 1.0)
            np_ = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (Mp + Mp.conj().T))))); nm_ = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (Mm + Mm.conj().T)))))
            Nq = 1j * Cn * (2.0 * np.sin(thq * dchi))
            npm = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (Nq + Nq.conj().T)))))
            herm = float(np.max(np.abs(Mp - Mp.conj().T)))
            norms.append({"theta": thq, "DeltaC0_op": max(np_, nm_), "copy_blocks_op": [np_, nm_], "Cplus_minus_Cminus_op": npm, "hermiticity_defect": herm})
        # version 2's transfer readouts for this kappa
        rows = [r for r in J2["s7"]]
        cf = [q for q in rows[0]["configs"] if int(q["kappa"]) == kap]
        i_cf = rows[0]["configs"].index(cf[0])
        sel = {}
        for T_ in (18.25, 18.5):
            rr = next((r for r in rows if abs(r["T_over_rs"] - T_) < 1e-9), None)
            if rr is not None:
                sel["%.3f" % T_] = rr["configs"][i_cf]["bound_Rt"]
        pers = [(r["T_over_rs"], r["configs"][i_cf]["bound_Rt"]) for r in rows if r["persists"] and r["configs"][i_cf]["bound_Rt"] is not None]
        mx = max(pers, key=lambda q: q[1]) if pers else None
        dcol = max(r["configs"][i_cf]["D_det"] for r in rows)
        n0 = norms[0]
        ratios = {k_: {"over_DeltaC0": v_ / n0["DeltaC0_op"], "over_Cplus_minus_Cminus": v_ / n0["Cplus_minus_Cminus_op"]} for k_, v_ in sel.items()}
        lb = lab.get(kap, {}).get("unrotated", {})
        rowk = {"kappa": kap, "energy_band": be, "momentum_band": bj, "v2_label_band": {"shells_above_1e-15": lb.get("shells_above_1e-15"), "first_last": lb.get("shells_first_last")},
                "norms": norms, "bound_beyond_Rt": sel, "bound_beyond_Rt_max_persisting": list(mx) if mx else None, "collar_D_det_max": dcol, "ratios": ratios,
                "ratio_max_persisting": ({"over_DeltaC0": mx[1] / n0["DeltaC0_op"], "over_Cplus_minus_Cminus": mx[1] / n0["Cplus_minus_Cminus_op"]} if mx else None),
                "ratio_collar": {"over_DeltaC0": dcol / n0["DeltaC0_op"]}}
        out["channels"].append(rowk)
        print("kappa %d: the energy band %d shells (R %s to %s), the momentum band %d shells (R %s to %s); version 2's label: %s shells above 1e-15 (%s); ||Delta C(0)||_op at the run's theta %.6e (copy blocks %s), ||C_+ - C_-||_op %.6e; the bounds beyond R_t(t) %s, the largest while held %s; the collar D_det at most %.3e; ratios %s" % (
            kap, be["shells"], fmt(be["first_R"], "%.0f"), fmt(be["last_R"], "%.0f"), bj["shells"], fmt(bj["first_R"], "%.0f"), fmt(bj["last_R"], "%.0f"),
            lb.get("shells_above_1e-15"), lb.get("shells_first_last"), n0["DeltaC0_op"], ", ".join("%.6e" % x for x in n0["copy_blocks_op"]), n0["Cplus_minus_Cminus_op"],
            sel, mx, dcol, {k_: {kk: "%.4e" % vv for kk, vv in v_.items()} for k_, v_ in ratios.items()}), flush=True)
    # inside the outer root?
    allR = sorted(set(sum([c_["energy_band"]["R_list"] + c_["momentum_band"]["R_list"] for c_ in out["channels"]], [])))
    ins = {}
    for kind, (Ro, Ri) in roots.items():
        if Ro is None:
            ins[kind] = None; continue
        ins[kind] = {"R_outer": Ro, "R_inner": Ri, "bands_inside_outer_root": bool(max(allR) < Ro) if allR else None, "margin_to_outer": (Ro - max(allR)) if allR else None,
                     "ramp_inside_outer_root": bool(max(ramp) < Ro), "bands_outside_inner_root": (bool(min(allR) > Ri) if (Ri is not None and allR) else None),
                     "margin_to_inner": (min(allR) - Ri) if (Ri is not None and allR) else None}
    out["inside"] = ins
    both_inside = all(v_ and v_["bands_inside_outer_root"] for v_ in ins.values() if v_ is not None)
    out["prints"] = bool(both_inside)
    c1 = next(c_ for c_ in out["channels"] if c_["kappa"] == 1)
    L = ["## S4 (revised) part C: the delivered part 3 relabelled (engine_s7_part3_relabel_v1.py; inbox #41; the lane's own files: engine_s7_twocopy_v2's member json and stored T0 state, part A's roots of the same record)", "",
         "- the window (engine spec 36.3; version 2's): %s, X %.4f, w %g; the plateau shells n_L = R %.0f (the last site with chi = 1) and n_R = R %.0f (the first with chi = 0); the ramp shells %s; chi on R 15 to 23: %s." % (
             out["window"]["form"], X, w, nL, nR, ", ".join("%.0f" % x for x in ramp), ", ".join("R %s %.6f" % (k_, v_) for k_, v_ in out["window"]["chi_15_to_23"].items())),
         "- the roots at T/r_s %.3f (part A): dense outer %.4f, inner %.4f; applied outer %.4f, inner %.4f." % (h2["T0_over_rs"], roots["dense"][0], roots["dense"][1], roots["applied"][0], roots["applied"][1]),
         "- the source bands of the rotated preparation against the unrotated state (version 2's construction on the stored T0 state at the run's theta %.6f; the shells where the copy-summed difference exceeds 1e-15 of the raw density's largest value): %s. P08 sec. 5's count for a w-shell ramp is w + 2 (energy) and w + 4 (momentum); the \"17 to 19 shell bands\" are P08's numbers for a 15-shell transition and not part 3's window." % (
             th, "; ".join("kappa %d: energy %d shells (R %s to %s), momentum %d shells (R %s to %s), version 2's stored label %s shells (%s)" % (
                 c_["kappa"], c_["energy_band"]["shells"], fmt(c_["energy_band"]["first_R"], "%.0f"), fmt(c_["energy_band"]["last_R"], "%.0f"), c_["momentum_band"]["shells"],
                 fmt(c_["momentum_band"]["first_R"], "%.0f"), fmt(c_["momentum_band"]["last_R"], "%.0f"), c_["v2_label_band"]["shells_above_1e-15"], c_["v2_label_band"]["first_last"]) for c_ in out["channels"])),
         "- inside the outer root: %s." % "; ".join("%s root %.4f: the bands %s (margin %s sites), the ramp %s; outside the inner root %.4f: %s (margin %s)" % (
             k_, v_["R_outer"], "inside" if v_["bands_inside_outer_root"] else "NOT inside", fmt(v_["margin_to_outer"], "%.4f"), "inside" if v_["ramp_inside_outer_root"] else "NOT inside",
             v_["R_inner"], v_["bands_outside_inner_root"], fmt(v_["margin_to_inner"], "%.4f")) for k_, v_ in ins.items() if v_), ""]
    L += ["| kappa | theta | Delta C(0) op (the task's sigma_z form; copy blocks) | C_+(0) - C_-(0) op (version 2's +-theta pair, the same object as its D_det) |", "|---|---|---|---|"]
    for c_ in out["channels"]:
        for q in c_["norms"]:
            L.append("| %d | %.6f%s | %.6e (%s) | %.6e |" % (c_["kappa"], q["theta"], " (the run's)" if q["theta"] == th else " (the law's)", q["DeltaC0_op"], ", ".join("%.6e" % x for x in q["copy_blocks_op"]), q["Cplus_minus_Cminus_op"]))
    L += ["", "| kappa | version 2's exact bound beyond R_t(t) (T/r_s: value) | over Delta C(0) op | over C_+(0) - C_-(0) op | the largest while the surface persists (T/r_s, value; over the two norms) | the collar D_det at most (over Delta C(0) op) |", "|---|---|---|---|---|---|"]
    for c_ in out["channels"]:
        mp = c_["bound_beyond_Rt_max_persisting"]; rm = c_["ratio_max_persisting"]
        L.append("| %d | %s | %s | %s | %s | %.3e (%.3e) |" % (
            c_["kappa"], "; ".join("%s: %.3e" % kv for kv in c_["bound_beyond_Rt"].items()), "; ".join("%s: %.4e" % (k_, v_["over_DeltaC0"]) for k_, v_ in c_["ratios"].items()),
            "; ".join("%s: %.4e" % (k_, v_["over_Cplus_minus_Cminus"]) for k_, v_ in c_["ratios"].items()),
            ("%.3f, %.3e; %.4e, %.4e" % (mp[0], mp[1], rm["over_DeltaC0"], rm["over_Cplus_minus_Cminus"])) if mp else "-", c_["collar_D_det_max"], c_["ratio_collar"]["over_DeltaC0"]))
    if both_inside:
        r1 = c1["ratios"]
        L += ["", "- the bands lie inside the outer root on both roots: the number prints in C's form. The lane's own record measured 5e-4 to 1.5e-3 as the exact bound beyond R_t(t) (part 3's D_det beyond the surface, at T/r_s 18.250 and 18.500), not at the collar, whose D_det stayed at the rounding floor (at most %.1e, out of causal reach): the sentence's \"reached the collar\" therefore reads \"reached beyond the outer root\" for these numbers (flagged for the lead). Its [value], kappa 1: %s of ||Delta C(0)||_op (the task's normalization), %s of ||C_+(0) - C_-(0)||_op (version 2's, the same object as its D_det)." % (
            c1["collar_D_det_max"], " to ".join("%.4e" % v_["over_DeltaC0"] for v_ in r1.values()), " to ".join("%.4e" % v_["over_Cplus_minus_Cminus"] for v_ in r1.values()))]
    else:
        L += ["", "- the bands do not lie inside the outer root on every root: part 3 is a record and nothing from it prints."]
    L += ["- ITEM7_s7_part3_sentence_v2.md stays withdrawn from the paper in its 2-preparation form (inbox #41); no mutual information (D15); every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L[-3:]), flush=True)


if __name__ == "__main__":
    main()
