#!/usr/bin/env python3
"""d1_control_data_v1.py (solve lane, PAGE VERIFY D1): the conformal control (page_history_v6s --kappa-zero on B2's
fields, run v6s_B2_kz) against the exact 1+1 Gaussian formula on the model's ray map, per copy (one sector), per cut
and record clock; the production kappa = 1 run (v6o_B2_k1) beside it.

Inputs: the control's D13 json, the production's D13 json, d1_raymap_v1.npz (d1_raymap_data_v1.py: the map, the
sheet's line formula reproduced, the finite-chain circle form) and the sheet's a2_raymap_B2.npz (its truncated curves).
Per copy = half the printed chain number (S_X = 2 S_Xc(C_+); each S_stag sector at kappa = 0 is one c = 1 Dirac
fermion, busi ready_jobs_v3 sec. 16). Two readings of the control are tabulated:
  growth  = [S_X(t) - S_X(0)]/2, the state's own entropy growth (the quantity the in-state formula predicts; the
            reference's lattice behaviour does not enter);
  excess  = [S_X(t) - S_X^vac(t)]/2, the printed excess over the held Killing vacuum of the record (the sheet's
            PASS criterion is written on this; at kappa = 0 the held vacuum of a trapped record is not the conformal
            u-vacuum and jumps between adjacent clocks).
Outputs: d1_conformal_control_v1.npz, _summary.json, _tables.md.
"""
import argparse, hashlib, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--control", required=True, help="v6s_B2_kz_k1_d13.json")
ap.add_argument("--production", required=True, help="v6o_B2_k1_d13.json")
ap.add_argument("--raymap", required=True, help="d1_raymap_v1.npz")
ap.add_argument("--sheet", required=True, help="a2_raymap_B2.npz (REVIEWA_write_A2_scripts)")
ap.add_argument("--out", default="d1_conformal_control_v1")
args = ap.parse_args()
RS = 64.0


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_d13(path):
    j = json.load(open(path))
    c = j["clocks"]
    out = {}
    for k in sorted(c):
        d = c[k]["per_channel"]; ch = list(d.keys())[0]; cuts = d[ch]["cuts"]
        t = float(c[k]["t_over_rs"])
        out[t] = {cut: dict(S=cuts[cut]["S_X"], V=cuts[cut]["S_X_vac"], X=cuts[cut]["excess"],
                            fh=cuts[cut]["excess_floor_high"], fl=cuts[cut]["excess_floor_low"]) for cut in cuts}
    args_ = j["settings"]["args"]
    return out, args_


ctrl, ctrl_args = load_d13(args.control)
prod, prod_args = load_d13(args.production)
assert ctrl_args.get("kappa_zero") is True and not prod_args.get("kappa_zero", False)
rm = np.load(args.raymap)
sh = np.load(args.sheet)
t_rm = rm["t_rec"] / RS
t_sh = sh["t_rec"] / RS
clocks = sorted(set(ctrl) & set(prod))
cuts = ["2", "16", "24", "30", "48"]
# the sheet's lattice-scale boundaries (e-foldings of blueshift at which the thermal partner reaches c/a)
EF = {"c1": 6.6, "cpi2": 7.05, "cpi": 7.7}

res = {"clocks": np.array(clocks)}
summary = {"files": {"control": [args.control, md5(args.control)], "production": [args.production, md5(args.production)],
                     "raymap": [args.raymap, md5(args.raymap)], "sheet": [args.sheet, md5(args.sheet)]},
           "control_args": {k: ctrl_args[k] for k in ("kappa_zero", "kappas", "nsites", "interp", "interp_sub", "cheb_tol",
                                                      "sector", "sector_propagate", "cuts_rs", "T_end_rs", "tag", "record", "fields")},
           "per_copy": "half the printed chain number (S_X = 2 S_Xc(C_+))", "cuts": {}}
lines = []
lines.append("# D1 conformal control: v6s_B2_kz (kappa = 0 chain on B2's fields) against the 1+1 Gaussian formula, per copy\n")
lines.append("growth = [S_X(t) - S_X(0)]/2 (the state's own entropy growth per copy); excess = [S_X(t) - S_X^vac(t)]/2 (the printed excess over the held vacuum, per copy); "
             "circle = the finite-chain form S(t) - S(0) on the traced map (d1_raymap_data_v1); sheet = REVIEWA_write_A2_v1's line formula dS_wall (reproduced to 4 digits); "
             "semi = -(1/12) ln p'(u_c); trunc(pi) = the sheet's truncated curve at c = pi (a2_raymap_B2.npz); Lam = e-foldings of blueshift at the cut's outgoing endpoint; "
             "k1 = the production kappa = 1 chain (v6o_B2_k1), same conventions. floor = the control's propagation floor (high) per copy.\n")
for cut in cuts:
    tag = "R%s" % cut
    g = lambda key: np.interp(np.array(clocks), t_rm, rm["%s_%s" % (key, tag)])
    Lam = g("Lam_out"); dS_circ = g("dS_circ"); dS_wall = g("dS_wall"); dS_semi = g("dS_semi"); dS_wall_in = g("dS_wall_in")
    circ_term = g("S_circ_term") - rm["S_circ_term_%s" % tag][0]; cut_term = g("S_cut_term") - rm["S_cut_term_%s" % tag][0]
    Lam_in = g("Lam_in"); l_int = g("l_int") / RS
    trunc_pi = np.interp(np.array(clocks), t_sh, sh["dS_trunc_c3.142_%s" % tag])
    trunc_1 = np.interp(np.array(clocks), t_sh, sh["dS_trunc_c1.000_%s" % tag])
    wall_sheet = np.interp(np.array(clocks), t_sh, sh["dS_wall_%s" % tag])
    S0 = ctrl[clocks[0]][cut]["S"]; P0 = prod[clocks[0]][cut]["S"]
    growth = np.array([(ctrl[t][cut]["S"] - S0) / 2 for t in clocks])
    excess = np.array([ctrl[t][cut]["X"] / 2 for t in clocks])
    vac = np.array([ctrl[t][cut]["V"] / 2 for t in clocks])
    floor = np.array([ctrl[t][cut]["fh"] / 2 for t in clocks])
    k1_growth = np.array([(prod[t][cut]["S"] - P0) / 2 for t in clocks])
    k1_excess = np.array([prod[t][cut]["X"] / 2 for t in clocks])
    k1_vac = np.array([prod[t][cut]["V"] / 2 for t in clocks])
    d_circ = growth - dS_circ
    d_sheet_growth = growth - dS_wall
    d_sheet_excess = excess - dS_wall
    for k_, arr in (("Lam", Lam), ("Lam_in", Lam_in), ("l_int_rs", l_int), ("dS_circ", dS_circ), ("dS_wall", dS_wall), ("dS_semi", dS_semi),
                    ("dS_wall_in", dS_wall_in), ("circ_term", circ_term), ("cut_term", cut_term), ("trunc_pi", trunc_pi), ("trunc_1", trunc_1),
                    ("wall_sheet_npz", wall_sheet), ("growth", growth), ("excess", excess), ("vac", vac), ("floor", floor),
                    ("k1_growth", k1_growth), ("k1_excess", k1_excess), ("k1_vac", k1_vac), ("d_circ", d_circ),
                    ("d_sheet_growth", d_sheet_growth), ("d_sheet_excess", d_sheet_excess)):
        res["%s_%s" % (k_, tag)] = arr
    # regimes on the cut's own blueshift
    cl = np.array(clocks)
    I = (Lam < 3.0) & (cl > 0); II = (Lam >= 3.0) & (Lam < EF["cpi"]); III = Lam >= EF["cpi"]
    win_sheet = (Lam < EF["c1"]) & (cl > 0)
    def stat(mask, arr):
        if not mask.any(): return None
        a = arr[mask]; i = np.argmax(np.abs(a))
        return {"max_abs": float(np.abs(a).max()), "at_t": float(cl[mask][i]), "mean": float(a.mean()), "n": int(mask.sum()),
                "t_range": [float(cl[mask].min()), float(cl[mask].max())]}
    # saturation: the plateau of the growth over the last 40 r_s and the onset (first clock with circle - growth > 0.1)
    late = cl >= 60.0
    onset = [float(t) for t, d in zip(cl, d_circ) if d < -0.1]
    cs = {"regime_I_Lam_lt_3": {"d_circ": stat(I, d_circ), "d_sheet_growth": stat(I, d_sheet_growth), "d_sheet_excess": stat(I, d_sheet_excess)},
          "regime_II_3_le_Lam_lt_7.7": {"d_circ": stat(II, d_circ), "d_sheet_growth": stat(II, d_sheet_growth), "d_sheet_excess": stat(II, d_sheet_excess)},
          "regime_III_Lam_ge_7.7": {"d_circ": stat(III, d_circ), "d_sheet_growth": stat(III, d_sheet_growth), "d_sheet_excess": stat(III, d_sheet_excess)},
          "sheet_window_Lam_lt_6.6": {"d_circ": stat(win_sheet, d_circ), "d_sheet_excess": stat(win_sheet, d_sheet_excess), "d_sheet_growth": stat(win_sheet, d_sheet_growth)},
          "efold_clocks_t_rs": {k: (float(np.interp(v, Lam, cl)) if Lam.max() > v else None) for k, v in (("1", 1.0), ("3", 3.0), ("6.6", 6.6), ("7.05", 7.05), ("7.7", 7.7), ("10", 10.0), ("20", 20.0))},
          "plateau_growth_60_100": {"mean": float(growth[late].mean()), "std": float(growth[late].std()), "min": float(growth[late].min()), "max": float(growth[late].max())} if late.any() else None,
          "plateau_excess_60_100": {"mean": float(excess[late].mean()), "std": float(excess[late].std()), "min": float(excess[late].min()), "max": float(excess[late].max())} if late.any() else None,
          "held_vacuum_per_copy": {"t0": float(vac[0]), "min": float(vac.min()), "max": float(vac.max()), "max_jump_between_adjacent_clocks": float(np.abs(np.diff(vac)).max())},
          "k1_held_vacuum_per_copy": {"t0": float(k1_vac[0]), "min": float(k1_vac.min()), "max": float(k1_vac.max()), "max_jump_between_adjacent_clocks": float(np.abs(np.diff(k1_vac)).max())},
          "onset_circle_minus_growth_gt_0.1_t_rs": (onset[0] if onset else None),
          "growth_at_100": float(growth[-1]), "excess_at_100": float(excess[-1]), "circle_at_100": float(dS_circ[-1]), "sheet_at_100": float(dS_wall[-1]),
          "k1_growth_at_100": float(k1_growth[-1]), "k1_excess_at_100": float(k1_excess[-1]),
          "floor_max_per_copy": float(floor.max()), "sheet_wall_reproduction_max_abs_diff": float(np.abs(dS_wall - wall_sheet).max())}
    summary["cuts"][tag] = cs
    lines.append("\n## cut %s r_s\n" % cut)
    lines.append("| t (r_s) | Lam | growth | circle | growth - circle | excess | sheet | excess - sheet | growth - sheet | semi | trunc(pi) | vac | floor | k1 growth | k1 excess | k1 vac |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(clocks):
        lines.append("| %g | %.2f | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f | %+.4f | %.4f | %.4f | %.4f | %.1e | %.4f | %.4f | %.4f |" % (
            t, Lam[i], growth[i], dS_circ[i], d_circ[i], excess[i], dS_wall[i], d_sheet_excess[i], d_sheet_growth[i], dS_semi[i], trunc_pi[i], vac[i], floor[i],
            k1_growth[i], k1_excess[i], k1_vac[i]))
    lines.append("\nregimes (on Lam at this cut): I Lam < 3: max|growth - circle| %s; II 3 <= Lam < 7.7: %s; III Lam >= 7.7: %s. Sheet's window (Lam < 6.6): max|excess - sheet| %s, max|growth - circle| %s. "
                 "Plateau of the growth over 60-100 r_s: %s. Held vacuum per copy: %s (k1: %s). e-fold clocks: %s." % (
                     cs["regime_I_Lam_lt_3"]["d_circ"], cs["regime_II_3_le_Lam_lt_7.7"]["d_circ"], cs["regime_III_Lam_ge_7.7"]["d_circ"],
                     cs["sheet_window_Lam_lt_6.6"]["d_sheet_excess"], cs["sheet_window_Lam_lt_6.6"]["d_circ"], cs["plateau_growth_60_100"],
                     cs["held_vacuum_per_copy"], cs["k1_held_vacuum_per_copy"], cs["efold_clocks_t_rs"]))

# the static-vacuum check at T = 0: the circle form's prediction for S_vac(R_c) - S_vac(2 r_s) against the lattice
S0_lat = np.array([ctrl[clocks[0]][c]["S"] / 2 for c in cuts])
S0_pred = rm["S_pred0_cuts"]
summary["static_vacuum_check_T0"] = {"cuts_rs": [float(c) for c in cuts], "lattice_per_copy_minus_2rs": (S0_lat - S0_lat[0]).tolist(),
                                     "circle_form_minus_2rs": (S0_pred - S0_pred[0]).tolist(), "residual": (S0_lat - S0_lat[0] - (S0_pred - S0_pred[0])).tolist()}
lines.append("\n## static-vacuum check at T = 0 (per copy): lattice S_vac(R_c) - S_vac(2 r_s) = %s; circle form = %s; residual = %s\n" % (
    np.round(S0_lat - S0_lat[0], 4).tolist(), np.round(S0_pred - S0_pred[0], 4).tolist(), np.round(S0_lat - S0_lat[0] - (S0_pred - S0_pred[0]), 4).tolist()))
summary["raymap"] = {"pH_rs": float(rm["pH"] / RS), "const0_rs": float(rm["const0"] / RS), "L_circle_rs": float(rm["L_circle"] / RS), "dt_a": float(rm["dt"])}
np.savez(args.out + ".npz", **res)
json.dump(summary, open(args.out + "_summary.json", "w"), indent=1)
open(args.out + "_tables.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(json.dumps(summary["cuts"]["R2"], indent=1))
print("wrote", args.out + ".npz / _summary.json / _tables.md")
