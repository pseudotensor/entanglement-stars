#!/usr/bin/env python
"""a1_rates_curve.py (PAGE VERIFY A1, reviewer A write lane): from the transmission table of a1_greybody.py,
the model's conditional luminosity, number and marginal-entropy rates, beta = T_H Sdot/L, the known-answer
checks, the evaporation and the two conditional Page curves (typical-state purifier; Gaussian-typical
purifier with the Bianchi-Hackl-Kieburg plateau), the measured-beta continuation, and the numbers at the
formation charge and at one solar mass.  Units: r_s = 1 for the rates (x = omega r_s, T_H = 1/(4 pi) r_s^-1,
n_F = 1/(e^{4 pi x} + 1)); lattice units a = 1, g = G/a^2 = 1/4 for the object.

Per channel (one 2-component radial problem, particles and antiparticles inside the measure dx/pi):
  l_k   = (1/pi) int dx  x Gamma_k n_F        (energy rate  = sum_k w_k l_k / r_s^2)
  nu_k  = (1/pi) int dx    Gamma_k n_F        (number rate  = sum_k w_k nu_k / r_s)
  sig_k = (1/pi) int dx  s(Gamma_k n_F)       (entropy rate = sum_k w_k sig_k / r_s),  s(n) = -n ln n - (1-n) ln(1-n)
  beta  = T_H Sdot / L = sum w sig / (4 pi sum w l).
Weights: the declared field (eight valleys, both parities, 2k magnetic labels) w_k = 32 k, which is the paper's
64 k per |k| in the measure d omega/(2 pi) per particle type, and the production's 16 k chains with the two
sectors each chain carries.  Alternatives printed beside: 4k (one continuum Dirac species), 8k (two species,
the staggered-copy option), 16k (the plan's literal reading of the chain weight).
"""
import argparse, json, math
import numpy as np

LN2 = math.log(2.0)


def s_ent(n):
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    m = (n > 0) & (n < 1)
    out[m] = -n[m] * np.log(n[m]) - (1 - n[m]) * np.log1p(-n[m])
    return out


def nF(y):
    return 1.0 / (np.exp(np.clip(y, -700, 700)) + 1.0)


def channel_rates(x, w, G, lam=1.0):
    """lam = T/T_H (1 for the Hawking state)."""
    n = G * nF(4 * math.pi * x / lam)
    return (np.sum(w * x * n) / math.pi, np.sum(w * n) / math.pi, np.sum(w * s_ent(n)) / math.pi)


def s_bhk(f):
    """Bianchi-Hackl-Kieburg leading term per total mode, symmetric about 1/2."""
    f = np.asarray(f, float)
    g = np.minimum(f, 1 - f)
    out = np.zeros_like(g)
    m = g > 0
    out[m] = (LN2 - 1) * g[m] + (g[m] - 1) * np.log1p(-g[m])
    return out


def page_curves(beta, npts=200001):
    """Curves on y = S_BH/S0 in [0,1] (y falls in time: t/t_evap = 1 - y^{3/2}).  Returns dict of arrays."""
    y = np.linspace(1.0, 0.0, npts)
    t = 1.0 - y ** 1.5
    S_th = beta * (1 - y)
    S_page = np.minimum(S_th, y)
    N = (S_th + y) / LN2
    f = np.where(N > 0, S_th / np.maximum(S_th + y, 1e-300), 0.0)
    S_G = N * s_bhk(f)
    return dict(y=y, t=t, S_th=S_th, S_page=S_page, S_G=S_G, f=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="a1_greybody.npz")
    ap.add_argument("--table2", default=None, help="second table (other node count) for the quadrature control")
    ap.add_argument("--out", default="a1_curve")
    a = ap.parse_args()
    d = np.load(a.table)
    meta = json.loads(str(d["meta"]))
    kmax = meta["kmax"]
    R = {}

    # ---- per-channel rates on the table's domain, with the tails ----
    rows = []
    for k in range(1, kmax + 1):
        x, w, G = d["x_%d" % k], d["w_%d" % k], d["G_%d" % k]
        l, nu, sg = channel_rates(x, w, G)
        rows.append((k, l, nu, sg, sg / (4 * math.pi * l) if l > 0 else np.nan))
    rows = np.array(rows)
    xmax = max(meta["panels"]); xmin = min(meta["panels"])
    # tails: above xmax with Gamma <= 1 (bound); below xmin with the threshold law (kappa = 1 dominates)
    xx = np.linspace(xmax, xmax + 20, 400001)
    n = nF(4 * math.pi * xx)
    tail_hi = dict(l=np.trapz(xx * n, xx) / math.pi, nu=np.trapz(n, xx) / math.pi, sig=np.trapz(s_ent(n), xx) / math.pi)
    xx = np.linspace(0, xmin, 20001)
    n = (xx ** 2 / 4) * nF(4 * math.pi * xx)
    tail_lo = dict(l=np.trapz(xx * n, xx) / math.pi, nu=np.trapz(n, xx) / math.pi, sig=np.trapz(s_ent(n), xx) / math.pi)
    R["per_channel"] = dict(columns="kappa, l_k, nu_k, sigma_k, beta_k", rows=rows.tolist())
    R["tails_per_unit_weight"] = dict(above_xmax_bound=tail_hi, below_xmin_kappa1=tail_lo)

    # ---- weighted sums in the conventions ----
    conv = {"declared_field_32k": 32.0, "one_Dirac_species_4k": 4.0, "two_species_staggered_8k": 8.0, "plan_literal_16k": 16.0}
    sums = {}
    for name, c in conv.items():
        wk = c * rows[:, 0]
        L = float(np.sum(wk * rows[:, 1])); Nn = float(np.sum(wk * rows[:, 2])); S = float(np.sum(wk * rows[:, 3]))
        sums[name] = dict(weight_per_kappa=c, Lcal=L, ndot_rs=Nn, Sdot_rs=S, beta=S / (4 * math.pi * L))
    R["sums"] = sums
    # cumulative in kappa for the declared field
    cum = []
    for K in range(1, kmax + 1):
        wk = 32.0 * rows[:K, 0]
        L = float(np.sum(wk * rows[:K, 1])); S = float(np.sum(wk * rows[:K, 3])); Nn = float(np.sum(wk * rows[:K, 2]))
        cum.append((K, L, Nn, S, S / (4 * math.pi * L)))
    R["cumulative_declared"] = dict(columns="K, Lcal(K), ndot(K), Sdot(K), beta(K)", rows=cum)
    Lc = sums["declared_field_32k"]["Lcal"]; Sc = sums["declared_field_32k"]["Sdot_rs"]; Nc = sums["declared_field_32k"]["ndot_rs"]
    beta = sums["declared_field_32k"]["beta"]
    # tail beyond kmax by the threshold-law ratio of the last two channels (geometric estimate, bound-like)
    r_last = rows[-1, 1] * rows[-1, 0] / (rows[-2, 1] * rows[-2, 0])
    R["tail_beyond_kmax_estimate"] = dict(ratio_last_two_weighted=float(r_last), Lcal_tail=float(32 * rows[-1, 0] * rows[-1, 1] * r_last / (1 - r_last)))

    # ---- known-answer checks ----
    chk = {}
    # (a) one channel, Gamma = 1: l = 1/(192 pi), sigma = 1/24, beta = 2 (on the table's nodes of kappa 1 plus the tails)
    x, w = d["x_1"], d["w_1"]
    l1, nu1, s1 = channel_rates(x, w, np.ones_like(x))
    l1 += tail_hi["l"]; nu1 += tail_hi["nu"]; s1 += tail_hi["sig"]
    xx = np.linspace(0, xmin, 20001); n = nF(4 * math.pi * xx)
    l1 += np.trapz(xx * n, xx) / math.pi; nu1 += np.trapz(n, xx) / math.pi; s1 += np.trapz(s_ent(n), xx) / math.pi
    chk["one_channel_Gamma1"] = dict(l=l1, l_exact=1 / (192 * math.pi), nu=nu1, nu_exact=LN2 / (4 * math.pi ** 2), sigma=s1, sigma_exact=1 / 24,
                                     beta=s1 / (4 * math.pi * l1), beta_exact=2.0)
    # (b) 3+1 black body: int y^2 s(n_F) / int y^3 n_F = 4/3; and the hot-hole limit with geometric-optics Gamma
    yy = np.linspace(0, 60, 600001); n = nF(yy)
    bb = np.trapz(yy ** 2 * s_ent(n), yy) / np.trapz(yy ** 3 * n, yy)
    chk["blackbody_3p1"] = dict(beta=bb, exact=4 / 3, int_y3_nF=np.trapz(yy ** 3 * n, yy), exact_7pi4_120=7 * math.pi ** 4 / 120)
    bc = 3 * math.sqrt(3) / 2
    hot = []
    for lam in (1.0, 3.0, 10.0, 100.0, 1000.0):
        T = lam / (4 * math.pi)
        xg = np.linspace(0, 40 * T, 400001)
        Lsum = 0.0; Ssum = 0.0
        for k in range(1, int(bc * xg.max()) + 2):
            th = (bc * xg >= k).astype(float)
            n = th * nF(xg / T)
            Lsum += 2 * k * np.trapz(xg * n, xg); Ssum += 2 * k * np.trapz(s_ent(n), xg)
        hot.append((lam, T * Ssum / Lsum))
    chk["hot_hole_geometric_optics"] = dict(columns="T/T_H, beta", rows=hot, limit=4 / 3)
    # (c) the production comparator at the record's charge at 100 r_s (kappa 1, 2; weights 64 k/(2 pi); 48 GL nodes on [0, 1.5])
    E100 = 106.05330786; g = 0.25; rs100 = 2 * g * E100; TH100 = 1 / (8 * math.pi * g * E100)
    LH_stored = 1.86122701e-06
    wk = 32.0 * rows[:2, 0]
    L12 = float(np.sum(wk * rows[:2, 1]))
    # emulate the production quadrature on my Gamma by interpolation (log Gamma in x)
    xs, ws = np.polynomial.legendre.leggauss(48); xs = 0.75 * (xs + 1); ws = 0.75 * ws
    L12_prod = 0.0
    for k in (1, 2):
        Gi = np.exp(np.interp(xs, d["x_%d" % k], np.log(np.maximum(d["G_%d" % k], 1e-300))))
        L12_prod += 64.0 * k / (2 * math.pi) * float(np.sum(ws * xs * Gi * nF(4 * math.pi * xs)))
    chk["comparator_100rs"] = dict(E=E100, rs=rs100, T_H=TH100, LH12_TH_stored=LH_stored, Lcal12_from_stored=LH_stored * rs100 ** 2,
                                   Lcal12_mine_full_domain=L12, Lcal12_mine_production_quadrature=L12_prod,
                                   rel_dev_full=L12 / (LH_stored * rs100 ** 2) - 1, rel_dev_prodquad=L12_prod / (LH_stored * rs100 ** 2) - 1,
                                   LH12_TH_mine=L12 / rs100 ** 2)
    # (d) the paper's Eq. sc-luminosity (|k| <= 8): Lcal 0.005237, ndot 0.01554, Sdot 0.1079, R 1.639; Page 1976 per Weyl species 8.183e-5
    K8 = [r for r in cum if r[0] == 8][0]
    chk["paper_eq_sc_luminosity"] = dict(Lcal_k8=K8[1], paper=0.005237, ndot_k8=K8[2], paper_ndot=0.01554, Sdot_k8=K8[3], paper_Sdot=0.1079,
                                         R_k8=K8[4], paper_R=1.639, Page1983_spin_half=1.639,
                                         alpha_per_Weyl_species=Lc / 4 / 16, Page1976_per_two_component_species=8.183e-5,
                                         alpha_declared_field=Lc / 4)
    # (e) the first law at the formation charge: S_BH = 4 pi g E^2, dS/dE = 1/T_H
    Ef = 116.6; rs0 = 2 * g * Ef; TH0 = 1 / (8 * math.pi * g * Ef)
    S_of_E = lambda E: 4 * math.pi * g * E * E
    h = 1e-4
    chk["first_law"] = dict(E_f=Ef, S_BH=S_of_E(Ef), T_H=TH0, dS_dE_fd=(S_of_E(Ef + h) - S_of_E(Ef - h)) / (2 * h), one_over_TH=1 / TH0,
                            residual_rel=(S_of_E(Ef + h) - S_of_E(Ef - h)) / (2 * h) * TH0 - 1)
    # (f) the Page-time algebra: closed form against the numerical crossing, and Page's 0.54 at beta 1.5
    C = page_curves(beta)
    i = np.argmax(C["S_th"] >= C["y"])
    tP_num = C["t"][i]; SP_num = C["S_page"][i]
    tP_cf = 1 - (beta / (1 + beta)) ** 1.5; SP_cf = beta / (1 + beta)
    chk["page_time_algebra"] = dict(beta=beta, tP_over_tevap_closed=tP_cf, tP_numeric=tP_num, SP_over_S0_closed=SP_cf, SP_numeric=SP_num,
                                    E_P_over_E0=math.sqrt(SP_cf), beta_1p5_tP=1 - (1.5 / 2.5) ** 1.5, beta_2_tP=1 - (2 / 3) ** 1.5, beta_2_SP=2 / 3)
    # (g) convergence in kappa and in the quadrature
    conv_k = dict(Lcal_K=[(int(r[0]), r[1]) for r in cum], increment_k3_rel=(cum[2][1] - cum[1][1]) / cum[1][1],
                  beyond_k8_rel=(cum[-1][1] - cum[7][1]) / cum[7][1] if kmax >= 8 else None, tail_est_rel=R["tail_beyond_kmax_estimate"]["Lcal_tail"] / Lc,
                  tails_rel=dict(above=32 * sum(k for k in range(1, kmax + 1)) * tail_hi["l"] / Lc, below=32 * tail_lo["l"] / Lc))
    if a.table2:
        d2 = np.load(a.table2); m2 = json.loads(str(d2["meta"]))
        L2 = 0.0; S2 = 0.0
        for k in range(1, m2["kmax"] + 1):
            l, nu, sg = channel_rates(d2["x_%d" % k], d2["w_%d" % k], d2["G_%d" % k])
            L2 += 32 * k * l; S2 += 32 * k * sg
        K2 = [r for r in cum if r[0] == m2["kmax"]][0]
        conv_k["quadrature_control"] = dict(nodes_main=meta["nodes"], nodes_control=m2["nodes"], kmax_control=m2["kmax"],
                                            Lcal_rel_diff=L2 / K2[1] - 1, beta_control=S2 / (4 * math.pi * L2), beta_main_sameK=K2[4])
    chk["convergence"] = conv_k
    R["checks"] = chk

    # ---- the object at the formation charge, the curves, the variants ----
    obj = {}
    for label, E in (("formation_charge", Ef), ("record_100rs", E100)):
        rs = 2 * g * E; TH = 1 / (8 * math.pi * g * E); S0 = 4 * math.pi * g * E * E
        t_evap = rs ** 3 / (6 * g * Lc)
        L = Lc / rs ** 2
        Sdot = beta * L / TH
        o = dict(E=E, rs_a=rs, T_H=TH, S0_nats=S0, L_per_a=L, Sdot_per_a=Sdot, Sdot_per_64a=Sdot * 64, Sdot_per_own_rs=Sdot * rs,
                 t_evap_a=t_evap, t_evap_over_64a=t_evap / 64, t_evap_over_own_rs=t_evap / rs,
                 t_P_a=tP_cf * t_evap, t_P_over_64a=tP_cf * t_evap / 64, S_P_nats=SP_cf * S0,
                 late_slope_at_tP_per_64a=-4 * math.pi * Lc / (rs * math.sqrt(SP_cf)) * 64,
                 computed_epoch_6400a_over_t_evap=6400 / t_evap, computed_epoch_over_tP=6400 / (tP_cf * t_evap),
                 S_th_at_6400a=Sdot * 6400, S_th_from_formation_12p125_to_100=Sdot * 64 * (100 - 12.125))
        # Gaussian-typical purifier
        iG = int(np.argmax(C["S_G"]))
        o["gaussian_max_over_S0"] = float(C["S_G"][iG]); o["gaussian_max_nats"] = float(C["S_G"][iG] * S0)
        o["gaussian_tmax_over_tevap"] = float(C["t"][iG]); o["gaussian_tmax_over_64a"] = float(C["t"][iG] * t_evap / 64)
        o["gaussian_y_at_max"] = float(C["y"][iG])
        o["gaussian_at_crossing_over_S0"] = float(SP_cf * (2 - 1 / LN2)); o["gaussian_at_crossing_nats"] = float(SP_cf * (2 - 1 / LN2) * S0)
        o["gaussian_over_page_at_crossing"] = 2 - 1 / LN2
        o["gaussian_per_mode_at_crossing"] = 2 * (LN2 - 0.5)
        # the late branch's deficit under the Gaussian purifier: S_G/S_BH at y = 0.3, 0.1
        for yy_ in (0.3, 0.1, 0.03):
            j = int(np.argmin(np.abs(C["y"] - yy_))); o["gaussian_over_area_at_y%.2f" % yy_] = float(C["S_G"][j] / C["y"][j])
        obj[label] = o
    R["object"] = obj
    # the measured-beta continuation (variant ii of the plan's step 8), at the formation charge
    meas = []
    S0 = obj["formation_charge"]["S0_nats"]; t_evap = obj["formation_charge"]["t_evap_a"]
    for b in (0.05, 0.08, 0.14, 0.20, 0.38):
        meas.append(dict(beta_eff=b, S_P_over_S0=b / (1 + b), S_P_nats=b / (1 + b) * S0, t_P_over_tevap=1 - (b / (1 + b)) ** 1.5,
                         t_P_over_64a=(1 - (b / (1 + b)) ** 1.5) * t_evap / 64, early_slope_per_64a=b * 4 * math.pi * Lc / obj["formation_charge"]["rs_a"] * 64,
                         early_slope_record_charge_per_64a=b * 4 * math.pi * Lc / obj["record_100rs"]["rs_a"] * 64))
    R["measured_beta_continuation"] = meas
    # where the computed segment sits
    seg = dict(computed_S_100rs_2rs_cut=9.50449, computed_late_slope_per_rs=(0.030, 0.040), beta_eff_46_100=(0.05, 0.08), beta_eff_80_100=(0.14, 0.20),
               beta_eff_window_max=0.38, thermal_slope_formation_charge=obj["formation_charge"]["Sdot_per_64a"],
               thermal_slope_record_charge=obj["record_100rs"]["Sdot_per_64a"])
    seg["late_slope_over_thermal_record_charge"] = (0.030 / seg["thermal_slope_record_charge"], 0.040 / seg["thermal_slope_record_charge"])
    seg["late_slope_over_thermal_formation_charge"] = (0.030 / seg["thermal_slope_formation_charge"], 0.040 / seg["thermal_slope_formation_charge"])
    seg["thermal_law_0_to_100_formation_charge"] = obj["formation_charge"]["S_th_at_6400a"]
    seg["thermal_law_formation_to_100"] = obj["formation_charge"]["S_th_from_formation_12p125_to_100"]
    seg["computed_over_thermal_from_formation"] = 9.50449 / obj["formation_charge"]["S_th_from_formation_12p125_to_100"]
    seg["beta_thermal_k12"] = cum[1][4]
    seg["beta_eff_over_thermal_k12"] = (0.05 / cum[1][4], 0.08 / cum[1][4], 0.38 / cum[1][4])
    R["computed_segment"] = seg

    # ---- one solar mass (SI), conditional on the same Lcal ----
    G_SI = 6.67430e-11; c = 2.99792458e8; hbar = 1.054571817e-34; kB = 1.380649e-23; Msun = 1.98847e30; yr = 3.15576e7
    lP = math.sqrt(hbar * G_SI / c ** 3)
    rs = 2 * G_SI * Msun / c ** 2
    TH_K = hbar * c ** 3 / (8 * math.pi * G_SI * Msun * kB)
    S0 = math.pi * rs ** 2 / lP ** 2
    L_W = Lc * hbar * c ** 2 / rs ** 2
    t_evap_s = rs ** 3 / (6 * lP ** 2 * Lc) / c
    sol = dict(rs_m=rs, T_H_K=TH_K, S0_nats=S0, L_W=L_W, Sdot_nats_per_s=beta * L_W / (kB * TH_K), ndot_per_s=Nc * c / rs,
               t_evap_yr=t_evap_s / yr, t_P_yr=tP_cf * t_evap_s / yr, S_P_nats=SP_cf * S0,
               gaussian_max_nats=obj["formation_charge"]["gaussian_max_over_S0"] * S0, gaussian_tmax_yr=obj["formation_charge"]["gaussian_tmax_over_tevap"] * t_evap_s / yr,
               t_evap_over_rs_c=t_evap_s * c / rs, paper_table_t_star_yr=3.3e65, paper_table_T_H_K=6.17e-8, paper_table_t_star_over_rs=1.06e78,
               entropy_per_particle_nats=Sc / Nc)
    R["solar_mass"] = sol

    # ---- save ----
    np.savez(a.out + ".npz", y=C["y"], t_over_tevap=C["t"], S_th_over_S0=C["S_th"], S_page_over_S0=C["S_page"], S_G_over_S0=C["S_G"], f=C["f"],
             beta=beta, Lcal=Lc, per_channel=rows, cumulative=np.array(cum))
    json.dump(R, open(a.out + ".json", "w"), indent=1, default=float)

    # ---- print ----
    p = print
    p("# A1 rates and curves (table: %s, kmax %d, nodes %d)" % (a.table, kmax, meta["nodes"]))
    p("\n## per channel (one radial problem, p + anti-p): kappa  l_k  nu_k  sigma_k  beta_k   | 32k l_k  cumulative Lcal")
    for r, cr in zip(rows, cum):
        p("%2d  %.6e  %.6e  %.6e  %.4f   | %.6e  %.7e" % (r[0], r[1], r[2], r[3], r[4], 32 * r[0] * r[1], cr[1]))
    p("\n## weighted sums: convention  weight/kappa  Lcal  ndot*rs  Sdot*rs  beta")
    for k_, v in sums.items():
        p("%-26s %5.1f  %.6e  %.6e  %.6e  %.5f" % (k_, v["weight_per_kappa"], v["Lcal"], v["ndot_rs"], v["Sdot_rs"], v["beta"]))
    p("\n## checks")
    for k_, v in chk.items():
        p("- %s: %s" % (k_, json.dumps(v, default=float)))
    p("\n## object")
    for k_, v in obj.items():
        p("- %s: %s" % (k_, json.dumps(v, default=float)))
    p("\n## measured-beta continuation")
    for m in meas:
        p("- %s" % json.dumps(m, default=float))
    p("\n## computed segment: %s" % json.dumps(seg, default=float))
    p("\n## solar mass: %s" % json.dumps(sol, default=float))


if __name__ == "__main__":
    main()
