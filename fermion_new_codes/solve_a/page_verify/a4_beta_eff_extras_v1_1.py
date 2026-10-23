"""a4_beta_eff_extras_v1_1.py (solve, page_verify A4 v1.1): the additional readings the cold reading
deliveries/lead_gaps/ReviewA_reading_A4_beta_v1.md (md5 82104620) asks for, computed by this lane's own code from the same stored arrays.
Called by a4_beta_eff_data_v1_1.py with its loaded arrays (ctx); returns a json-able dict. Blocks:
 all_windows   (reviewer A R2, reviewer B R5): every window [A, B] on the 2 r_s checkpoint grid with 40 <= A < B <= 100 and B - A >= 6 (406 per set and cut):
               max and min with location, max per length class (= 6, = 10, >= 20), counts above 0.252 and 0.5; both cuts (2 r_s lagged), 4 sets.
 formation     (reviewer A R1, P3; reviewer B R7): the 2 r_s cut's excess over 14-40 per set, the Hawking comparator L_H(T_H) of kappa 1, 2 (flux.LH12_TH) integrated
               over the same window (trapezoid on the checkpoints) and its entropy at beta = 2, the ratio, and the collar flow over 36-62.
 post_burst    (reviewer A R4; reviewer B R6): the occupation detector over the shells' 56-92 (the collar's 46-82 shifted by 10 r_s): totals, band shares, per-band beta.
 increments    (reviewer A R3): the occupation estimator on a thermal increment n = g0 + chi (f - g0) of the stored modes, chi = 0.3, 0.1, 0.03, median over clocks >= 40.
 upper_bound   (reviewer A R5; reviewer B R3): S_occ (diagonal excess) against the stored von Neumann excess beyond the same 24 r_s cut (B2 cut 24), per clock >= 16
               and per window of >= 6 r_s from 56 on the shells' clock (increments).
 alternatives  (reviewer A sec. 1; reviewer B R8): fixed-charge lag; slope clock kappa_T/(2 pi); exterior inventory denominator (flux.E_out_per_channel);
               outward-only check (collar_split in/out); raw cut entropy (B2.S_X_raw_w); per channel kappa 2, 3; channel energy shares; other cuts 24, 30, 48
               read forward by their ray-map lags; window-difference check of the occupation energy against the inventory (reviewer B minor 4).
 margins       (reviewer B R4): occupation values against the thermal fills on every tabulated window (shifted where available, own otherwise).
"""
import numpy as np


def _beta_window(TH, S_at_A, S_at_B, E_A, E_B):
    dE = E_B - E_A
    return (TH * (S_at_B - S_at_A) / dE) if dE != 0 else None, dE


def run(ctx):
    clk = ctx["clk"]; TH = ctx["TH"]; RS = ctx["RS"]
    sets = ctx["sets"]; emit_of_arrival = ctx["emit_of_arrival"]; S_at = ctx["S_at"]
    out = {}
    # ---------------- all windows
    grid = [t for t in clk if 40.0 - 1e-9 <= t <= 100.0 + 1e-9 and abs(t - round(t / 2) * 2) < 1e-9]
    idx = {t: int(np.argmin(np.abs(clk - t))) for t in grid}
    aw = {}
    for setname, st in sets.items():
        for cut in ("16", "2"):
            ci = 1 if cut == "16" else 0
            ser = st["S"][:, ci]
            vals = []
            for i, A in enumerate(grid):
                for B in grid[i + 1:]:
                    if B - A < 6 - 1e-9:
                        continue
                    iA, iB = idx[A], idx[B]
                    THw = 0.5 * (TH[iA] + TH[iB]); dE = st["flowE"][iB] - st["flowE"][iA]
                    if cut == "16":
                        dS = ser[iB] - ser[iA]
                    else:
                        dS = S_at(ser, emit_of_arrival(B)) - S_at(ser, emit_of_arrival(A))
                    if dE == 0 or not np.isfinite(dS):
                        continue
                    vals.append((THw * dS / dE, A, B, dE))
            b = np.array([v[0] for v in vals]); L = np.array([v[2] - v[1] for v in vals])
            def mx(m):
                if not m.any():
                    return None
                j = int(np.argmax(np.where(m, b, -np.inf))); return {"beta": float(b[j]), "A": vals[j][1], "B": vals[j][2], "dE": float(vals[j][3])}
            jmin = int(np.argmin(b))
            aw["%s_cut%s" % (setname, cut)] = {"n_windows": len(vals), "max": mx(np.ones_like(b, bool)), "min": {"beta": float(b[jmin]), "A": vals[jmin][1], "B": vals[jmin][2]},
                                              "max_len6": mx(np.abs(L - 6) < 1e-9), "max_len10": mx(np.abs(L - 10) < 1e-9), "max_len_ge20": mx(L >= 20 - 1e-9),
                                              "n_above_0.252": int((b > 0.252).sum()), "n_above_0.5": int((b > 0.5).sum()),
                                              "max_from46_len6_len10": mx((np.array([v[1] for v in vals]) >= 46) & ((np.abs(L - 6) < 1e-9) | (np.abs(L - 10) < 1e-9)))}
    out["all_windows"] = aw
    # ---------------- the formation window against the Hawking comparator
    F = ctx["flux"]
    LH = np.array(F["LH12_TH"], float)
    i14, i40 = int(np.argmin(np.abs(clk - 14))), int(np.argmin(np.abs(clk - 40)))
    tt = clk[i14:i40 + 1] * RS; ll = LH[i14:i40 + 1]
    E_cmp = float(np.trapezoid(ll, tt)) if hasattr(np, "trapezoid") else float(np.trapz(ll, tt))
    THm = float(np.mean(TH[i14:i40 + 1]))
    S_cmp = 2.0 * E_cmp / THm
    fo = {"window": [14.0, 40.0], "comparator_energy_LH12_TH": E_cmp, "comparator_entropy_at_beta2": S_cmp, "T_H_mean": THm}
    for setname in ("k1", "k12", "K18"):
        st = sets[setname]; dS = float(st["S"][i40, 0] - st["S"][i14, 0])
        fo["excess_2rs_" + setname] = dS
    fo["ratio_k12_to_comparator"] = fo["excess_2rs_k12"] / S_cmp
    i36, i62 = int(np.argmin(np.abs(clk - 36))), int(np.argmin(np.abs(clk - 62)))
    fo["collar_flow12_36_62"] = float(sets["k12"]["flowE"][i62] - sets["k12"]["flowE"][i36])
    fo["collar_flow12_over_comparator"] = fo["collar_flow12_36_62"] / E_cmp
    out["formation"] = fo
    # ---------------- occupations: post-burst, increments, upper bound, margins
    agg = ctx["agg"]; tocc = ctx["tocc"]; TH_of = ctx["TH_of"]; bands = ctx["bands"]
    def occ_window(nm, a_, b_):
        g = agg[nm]; ia = int(np.argmin(np.abs(tocc - a_))); ib = int(np.argmin(np.abs(tocc - b_)))
        THw = 0.5 * (TH_of(a_) + TH_of(b_))
        dS = g["S_tot"][ib] - g["S_tot"][ia]; dE = g["E_tot"][ib] - g["E_tot"][ia]
        dSb = np.array(g["S_band"][ib]) - np.array(g["S_band"][ia]); dEb = np.array(g["E_band"][ib]) - np.array(g["E_band"][ia])
        return {"A": a_, "B": b_, "dS": float(dS), "dE": float(dE), "beta": float(THw * dS / dE),
                "band_share": (dEb / dE).tolist(), "beta_band": [float(THw * s / e) if e != 0 else None for s, e in zip(dSb, dEb)]}
    out["post_burst"] = {nm: occ_window(nm, 56.0, 92.0) for nm in ("K18", "k12", "k1")}
    out["post_burst"]["bands"] = bands
    out["burst_inclusive_46_100_own"] = {nm: occ_window(nm, 46.0, 100.0) for nm in ("K18", "k12", "k1")}
    out["burst_only_50_56"] = {nm: occ_window(nm, 50.0, 56.0) for nm in ("K18", "k12", "k1")}
    out["increments"] = ctx["increments"]
    vn = {"K18": ctx["SUM"][:, 2], "k1": ctx["WX0"][:, 2]}   # the von Neumann excess at the 24 r_s cut (index 2)
    ub = {}
    for nm in ("K18", "k1"):
        so = np.interp(clk, tocc, agg[nm]["S_tot"])
        m = clk >= 16 - 1e-9
        per_clock = so[m] - vn[nm][m]
        gw = [t for t in tocc if 56 - 1e-9 <= t <= 100 + 1e-9]
        nwin = 0; nok = 0; worst = None
        for i, A in enumerate(gw):
            for B in gw[i + 1:]:
                if B - A < 6 - 1e-9:
                    continue
                iA, iB = int(np.argmin(np.abs(clk - A))), int(np.argmin(np.abs(clk - B)))
                d_occ = float(np.interp(B, tocc, agg[nm]["S_tot"]) - np.interp(A, tocc, agg[nm]["S_tot"])); d_vn = float(vn[nm][iB] - vn[nm][iA])
                nwin += 1; nok += int(d_occ > d_vn)
                if worst is None or d_occ - d_vn < worst[0]:
                    worst = (d_occ - d_vn, A, B, d_occ, d_vn)
        ub[nm] = {"min_clock_margin_from16": float(per_clock.min()), "clock_of_min": float(clk[m][int(np.argmin(per_clock))]),
                  "at_100": [float(np.interp(100, tocc, agg[nm]["S_tot"])), float(vn[nm][-1])],
                  "windows_from56": nwin, "windows_occ_above_vn": nok, "smallest_window_margin": worst}
    out["upper_bound"] = ub
    cal = ctx["cal"]
    rows_occ = ctx["rows_occ"]
    mg = []
    for r in rows_occ:
        if not (r["window"].startswith("late6") or r["window"].startswith("late10")) or r["beta_occ"] is None or r["beta_occ"] <= 0:
            continue
        sh = r["shifted_by"] > 0
        # shifted where available, own otherwise
        twin = [q for q in rows_occ if q["window"] == r["window"] and q["set"] == r["set"] and q["shifted_by"] > 0]
        if not sh and twin:
            continue
        mg.append({"window": r["window"], "set": r["set"], "clock": "shifted" if sh else "own", "beta_occ": r["beta_occ"], "fill": cal[r["set"]], "margin": cal[r["set"]] / r["beta_occ"]})
    out["margins"] = {"rows": mg, "smallest": {nm: min((x for x in mg if x["set"] == nm), key=lambda x: x["margin"]) for nm in ("K18", "k1", "k12")}}
    # ---------------- alternatives
    alt = {}
    i46, i100, i80, i70, i76 = (int(np.argmin(np.abs(clk - t))) for t in (46, 100, 80, 70, 76))
    K = sets["K18"]
    def b16(st, iA, iB, TH_used=None, Eser=None):
        THw = 0.5 * (TH[iA] + TH[iB]) if TH_used is None else 0.5 * (TH_used[iA] + TH_used[iB])
        Es = st["flowE"] if Eser is None else Eser
        return float(THw * (st["S"][iB, 1] - st["S"][iA, 1]) / (Es[iB] - Es[iA]))
    for lagname, lagv in (("fixed_charge_formation", ctx["lag_fixed_form"]), ("fixed_charge_late", ctx["lag_fixed_late"])):
        dS = S_at(K["S"][:, 0], 100 - lagv) - S_at(K["S"][:, 0], 46 - lagv)
        alt["lag_" + lagname] = {"lag": lagv, "beta_K18_2rs_46_100": float(0.5 * (TH[i46] + TH[i100]) * dS / (K["flowE"][i100] - K["flowE"][i46]))}
    Tsl = np.array(F["kappa_T"], float) / (2 * np.pi)
    alt["slope_clock"] = {"T_slope_over_T_H_at_100": float(Tsl[i100] / TH[i100]), "beta_K18_16_46_100": b16(K, i46, i100, Tsl), "beta_K18_16_70_76": b16(K, i70, i76, Tsl)}
    Eout = np.array(F["E_out_per_channel"], float).sum(0)
    alt["inventory_denominator"] = {"beta_K18_16_46_100": b16(K, i46, i100, None, Eout), "dE_inventory": float(Eout[i100] - Eout[i46]), "dE_flow": float(K["flowE"][i100] - K["flowE"][i46])}
    cs = ctx["collar_split"]
    o = np.array(cs["E_flux_out_12"], float); n_ = np.array(cs["E_flux_in_12"], float); cc = np.array(cs["clocks"], float)
    m = (cc >= 46 - 1e-9) & (np.abs(o) > 0)
    rat = np.abs(n_[m]) / np.abs(o[m])
    alt["outward_only"] = {"in_over_out_max": float(rat.max()), "in_over_out_median": float(np.median(rat)), "checkpoints": int(m.sum())}
    raw = np.array(ctx["B2"]["S_X_raw_w"], float)
    alt["raw_cut_entropy_16_46_100"] = {"dS_raw": float(raw[i100, 1] - raw[i46, 1]), "dS_excess": float(K["S"][i100, 1] - K["S"][i46, 1])}
    WX = ctx["WX"]; flow = ctx["flow"]
    pc = {}
    for kap in (1, 2, 3):
        st = {"S": WX[kap - 1], "flowE": flow[kap - 1]}
        for (a, b_) in ((46, 100), (80, 100)):
            iA, iB = int(np.argmin(np.abs(clk - a))), int(np.argmin(np.abs(clk - b_)))
            dE = flow[kap - 1][iB] - flow[kap - 1][iA]
            pc["k%d_%d_%d" % (kap, a, b_)] = {"beta16": float(0.5 * (TH[iA] + TH[iB]) * (WX[kap - 1][iB, 1] - WX[kap - 1][iA, 1]) / dE),
                                              "beta2_lagged": float(0.5 * (TH[iA] + TH[iB]) * (S_at(WX[kap - 1][:, 0], emit_of_arrival(b_)) - S_at(WX[kap - 1][:, 0], emit_of_arrival(a))) / dE),
                                              "dS16": float(WX[kap - 1][iB, 1] - WX[kap - 1][iA, 1]), "dE": float(dE)}
    tot = flow.sum(0)[i100] - flow.sum(0)[i46]
    pc["energy_share_46_100"] = {"k1": float((flow[0][i100] - flow[0][i46]) / tot), "k2": float((flow[1][i100] - flow[1][i46]) / tot),
                                 "k1_2": float((flow[0][i100] - flow[0][i46] + flow[1][i100] - flow[1][i46]) / tot), "k3_18": float((flow[2:].sum(0)[i100] - flow[2:].sum(0)[i46]) / tot)}
    alt["per_channel"] = pc
    oc = {}
    for cut, ci, R1 in (("24", 2, 24.0), ("30", 3, 30.0), ("48", 4, 48.0)):
        lag = ctx["lag_16_to"](R1)   # function: collar arrival clock -> arrival clock at R1
        bs = []
        for a in range(46, 100, 6):
            A, B = float(a), float(a + 6)
            tA, tB = lag(A), lag(B)
            if not (np.isfinite(tA) and np.isfinite(tB)) or tB > 100:
                continue
            iA, iB = int(np.argmin(np.abs(clk - A))), int(np.argmin(np.abs(clk - B)))
            dE = K["flowE"][iB] - K["flowE"][iA]
            bs.append((A, B, float(0.5 * (TH[iA] + TH[iB]) * (S_at(K["S"][:, ci], tB) - S_at(K["S"][:, ci], tA)) / dE)))
        oc[cut] = bs
    alt["other_cuts_K18_6rs"] = oc
    # occupation energy window differences against the inventory window differences (reviewer B minor 4)
    wd = {}
    for nm in ("K18", "k1"):
        g = agg[nm]; ref = g["Eout_P_check"]
        rel = []
        for r in rows_occ:
            if r["set"] != nm or not (r["window"].startswith("late6") or r["window"].startswith("late10") or r["window"] in ("burst", "late_46_100")):
                continue
            ia = int(np.argmin(np.abs(tocc - r["A"]))); ib = int(np.argmin(np.abs(tocc - r["B"])))
            dref = ref[ib] - ref[ia]
            if abs(dref) > 1e-4:
                rel.append(abs((g["E_tot"][ib] - g["E_tot"][ia]) - dref) / abs(dref))
        wd[nm] = {"windows": len(rel), "max_rel": float(max(rel)) if rel else None, "median_rel": float(np.median(rel)) if rel else None}
    alt["occ_energy_window_diff_vs_inventory"] = wd
    out["alternatives"] = alt
    return out
