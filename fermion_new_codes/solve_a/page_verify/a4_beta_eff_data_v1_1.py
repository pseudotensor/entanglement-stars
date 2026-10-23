#!/usr/bin/env python3
"""a4_beta_eff_data_v1_1.py (solve, page_verify A4 v1.1): version 1 (a4_beta_eff_data_v1.py, md5 d5c83963) with the cold reading's code items
(ReviewA_reading_A4_beta_v1.md, reviewer B minor 2: exact T_H assertion with atol 0, W = 16 kappa asserted, a missing channel file fails) and the
reading's additional readings (a4_beta_eff_extras_v1_1.py, block "v1_1"); every version-1 number is computed by the same code. Version 1's docstring:
beta_eff(t) per window from stored data.

Method 1 (cut-based): beta_cut = T_H * Delta S_cut / Delta E_collar over a window [A, B] on the collar's record clock, where
  Delta E_collar = flow(B) - flow(A) is the accumulated signed collar energy flow (16 r_s) of the channel set, and
  Delta S_cut = S(B - lag(B)) - S(A - lag(A)) is the change of the stored exterior-cut excess S_X(C) - S_X(P_-) at the cut;
  lag = 0 at the 16 r_s cut (it is the collar's radius); at the 2 r_s cut lag(t) is the outgoing light travel time 2 -> 16 r_s for
  the ray arriving at 16 r_s at t, integrated in the B2 record's applied fields (dR/dt = N - |v|, fields linear in T between records).
  T_H = 1/(8 pi G E) at the record (G = g = 1/4), the mean of the window's two end values (the Page deliveries' L_H(T_H) comparator).
  Channel sets: kappa 1 alone (16 x_1 against kappa 1's weighted flow), kappa 1+2 (against L12, the printed collar flux), the K = 18
  sum (against the K = 18 flow, and against L12 beside).
Method 2 (occupation-based): the exterior Killing shells beyond R_cut = 24 r_s (page_d13_readout_v1d killing_shells; per clock the
  block's eigen-energies e_j, the state's diagonal occupations n_j = <u_j|C|u_j> and the instantaneous vacuum's nvac_j = <u_j|P_-|u_j>):
  S_occ(t) = sum_j [s(n_j) - s(nvac_j)], E_occ(t) = sum_j e_j (n_j - nvac_j), s(n) = -n ln n - (1-n) ln(1-n), per band of |e_j|/T_H^ref
  and in total, weighted 16 kappa over the channels; beta_occ = T_H * Delta S_occ / Delta E_occ over a window on the 24 r_s clock.
  (The diagonal entropy is the mode-by-mode detector's coarse entropy, an upper bound on the block's von Neumann entropy.)
Calibration: the same estimator on a Fermi-Dirac fill at T_H of the stored e_j (beta of a thermal increment, expected near 2).
Reads stored files only (paths and md5 in the output's sources). Usage: a4_beta_eff_data_v1.py OUTDIR
"""
import hashlib, json, os, sys, time
import numpy as np

PV4 = "<internal>/page_v4_data_v1.json"
FLD = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz"
REC = "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json"
RS = 64.0          # nominal r_s in a (the clocks' unit)
G = 0.25
CUTS = ["2", "16", "24", "30", "48"]
BANDS_TH = [0.0, 1.0, 3.0, 10.0, 25.0, None]   # |e|/T_H^ref edges; None = the band top 1/a; then 'above'


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def s_bin(n):
    n = np.clip(np.asarray(n, float), 0.0, 1.0)
    out = np.zeros_like(n)
    m = (n > 0) & (n < 1)
    out[m] = -n[m] * np.log(n[m]) - (1 - n[m]) * np.log1p(-n[m])
    return out


# ------------------------------------------------------------------------------------------------ the ray map (lag)
class Fields:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.R = np.asarray(z["R"], float); self.T = np.asarray(z["T"], float)
        self.N = np.asarray(z["N_applied"], float); self.v = np.asarray(z["v_applied"], float)

    def speed(self, R, t):
        """outgoing light speed N - |v| at radius R (a) and record time t (a), linear in R and in T between records."""
        k = np.clip(np.searchsorted(self.T, t) - 1, 0, len(self.T) - 2)
        w = (t - self.T[k]) / (self.T[k + 1] - self.T[k]); w = min(max(w, 0.0), 1.0)
        c0 = np.interp(R, self.R, self.N[k] - np.abs(self.v[k])); c1 = np.interp(R, self.R, self.N[k + 1] - np.abs(self.v[k + 1]))
        return (1 - w) * c0 + w * c1

    def travel(self, R0, R1, t0, dt=1.0):
        """RK4 in t of dR/dt = N - |v| from (R0, t0) to R1; returns the arrival time (a)."""
        R, t = R0, t0
        while R < R1:
            k1 = self.speed(R, t)
            if k1 <= 0:
                return np.nan
            k2 = self.speed(R + 0.5 * dt * k1, t + 0.5 * dt); k3 = self.speed(R + 0.5 * dt * k2, t + 0.5 * dt); k4 = self.speed(R + dt * k3, t + dt)
            dR = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if R + dR >= R1:
                return t + dt * (R1 - R) / dR
            R += dR; t += dt
        return t


def fixed_charge_lag(r_s_f, R0, R1):
    """the deliveries' fixed-charge flight comparison 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2)."""
    F = lambda x: x * x / 2 + x + np.log(x - 1)
    return 2 * r_s_f * (F(np.sqrt(R1 / r_s_f)) - F(np.sqrt(R0 / r_s_f)))


# ------------------------------------------------------------------------------------------------ the windows
def windows():
    W = [("formation", 14.0, 40.0), ("formation_a", 14.0, 26.0), ("formation_b", 26.0, 36.0), ("formation_c", 36.0, 40.0),
         ("burst", 40.0, 46.0), ("burst_42_46", 42.0, 46.0)]
    for a in range(46, 100, 6):
        W.append(("late6_%d_%d" % (a, a + 6), float(a), float(a + 6)))
    for a in range(46, 96, 10):
        W.append(("late10_%d_%d" % (a, a + 10), float(a), float(a + 10)))
    W.append(("late10_90_100", 90.0, 100.0))
    W += [("late_46_100", 46.0, 100.0), ("check_80_100", 80.0, 100.0), ("late4_48_52", 48.0, 52.0)]
    for a in range(52, 100, 4):
        W.append(("late4_%d_%d" % (a, a + 4), float(a), float(a + 4)))
    return W


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    src = {}
    d = json.load(open(PV4)); src[PV4] = md5(PV4)
    B = d["B2"]; Fx = d["flux"]
    clk = np.array(B["clocks"], float)
    W = np.array(B["W"], float)
    x = np.array(B["x"], float)                     # (18, 52, 5) unweighted per-channel excess
    WX = W[:, None, None] * x
    SUM = np.array(B["SUM"], float)                 # (52, 5)
    assert np.allclose(WX.sum(0), SUM, atol=1e-9 * (1 + np.abs(SUM)).max()), "SUM != sum 16 kappa x_kappa"
    FMW = np.array(B["FMW"], float); SHARE = np.array(B["SHARE"], float)
    fmeas_w = np.array(B["fmeas_w"], float)         # (18, 52, 5) weighted floors
    flow = np.array(Fx["flow_per_channel"], float)  # (18, 52) weighted accumulated collar flow
    flow12 = np.array(Fx["flow12"], float); flow18 = np.array(Fx["flow18"], float); flow123 = np.array(Fx["flow123"], float)
    assert np.allclose(flow[:2].sum(0), flow12) and np.allclose(flow.sum(0), flow18)
    TH = np.array(Fx["T_H"], float); E = np.array(Fx["E"], float)
    assert np.allclose(TH, 1.0 / (8 * np.pi * G * E), rtol=1e-10, atol=0.0), "T_H != 1/(8 pi G E)"
    assert np.array_equal(W, 16.0 * np.arange(1, 19)), "W != 16 kappa"
    ic = d["interp_control"]
    err_ic = np.array([[np.nan if v is None else v for v in row] for row in ic["error_prod_minus_ctrl"]], float)  # (52, 5) prod - ctrl, kappa 1-8
    sub16 = d["sub16_B2"]

    # ---------------- the lag 2 -> 16 r_s and 16 -> 24 r_s in the record's fields
    src[FLD] = md5(FLD); src[REC] = md5(REC)
    Fd = Fields(FLD)
    t_em = np.arange(10.0, 100.01, 0.5)
    arr16 = np.array([Fd.travel(2 * RS, 16 * RS, t * RS) / RS for t in t_em])
    lag2_16_em = arr16 - t_em                       # by emission clock
    t_em16 = np.arange(30.0, 100.01, 0.5)
    arr24 = np.array([Fd.travel(16 * RS, 24 * RS, t * RS) / RS for t in t_em16])
    lag16_24_em = arr24 - t_em16
    ok = np.isfinite(arr16)
    def emit_of_arrival(tA):
        """the emission clock at 2 r_s of the ray arriving at 16 r_s at tA (r_s), by inverting arr16(t_em) (monotone)."""
        return float(np.interp(tA, arr16[ok], t_em[ok], left=np.nan, right=np.nan))
    rsf = d["marks"]["two_gE_formation"]
    lag_fixed_form = fixed_charge_lag(rsf, 2 * RS, 16 * RS) / RS
    lag_fixed_late = fixed_charge_lag(2 * G * E[-1], 2 * RS, 16 * RS) / RS
    lag_fixed_16_24_form = fixed_charge_lag(rsf, 16 * RS, 24 * RS) / RS

    def S_at(series, t):                            # linear interpolation of a 2 r_s-spaced readout series
        return float(np.interp(t, clk, series, left=np.nan, right=np.nan))

    def S_near(series, t):                          # the nearest stored readout clock
        return float(series[int(np.argmin(np.abs(clk - t)))])

    def idx(t):
        i = int(np.argmin(np.abs(clk - t))); assert abs(clk[i] - t) < 1e-9, t; return i

    sets = {
        "k1": {"S": WX[0], "flowE": flow[0], "floor": fmeas_w[0], "Kc": None},
        "k12": {"S": WX[0] + WX[1], "flowE": flow12, "floor": fmeas_w[0] + fmeas_w[1], "Kc": None},
        "K18": {"S": SUM, "flowE": flow18, "floor": FMW, "Kc": SHARE},
        "K18_vs_L12": {"S": SUM, "flowE": flow12, "floor": FMW, "Kc": SHARE},
    }
    rows = []
    for (name, A, Bc) in windows():
        iA, iB = idx(A), idx(Bc)
        TH_w = 0.5 * (TH[iA] + TH[iB])
        for setname, st in sets.items():
            dE = st["flowE"][iB] - st["flowE"][iA]
            dE_k3 = (flow123[iB] - flow123[iA]) if setname == "k12" else None
            for ci, cut in ((1, "16"), (0, "2")):
                ser = st["S"][:, ci]
                rec = {"window": name, "A": A, "B": Bc, "set": setname, "cut": cut, "T_H": TH_w, "dE_collar": dE}
                if cut == "16":
                    a_, b_ = A, Bc
                    dS = ser[iB] - ser[iA]; dS_near = dS
                    floor = st["floor"][iB, ci] + st["floor"][iA, ci]
                    kc = (st["Kc"][iB, ci] + st["Kc"][iA, ci]) if st["Kc"] is not None else None
                    ric = abs(err_ic[iB, ci] - err_ic[iA, ci]) if setname in ("K18", "K18_vs_L12") else None
                    rec.update({"lag_A": 0.0, "lag_B": 0.0})
                else:
                    a_, b_ = emit_of_arrival(A), emit_of_arrival(Bc)
                    dS = S_at(ser, b_) - S_at(ser, a_)
                    dS_near = S_near(ser, b_) - S_near(ser, a_)
                    fl = st["floor"][:, ci]
                    floor = S_at(fl, b_) + S_at(fl, a_)
                    kc = (S_at(st["Kc"][:, ci], b_) + S_at(st["Kc"][:, ci], a_)) if st["Kc"] is not None else None
                    ric = None                      # the record-interpolation control has no 2 r_s cut
                    rec.update({"lag_A": A - a_, "lag_B": Bc - b_})
                    # unlagged beside (the formation signature)
                    dS_un = ser[iB] - ser[iA]
                    rec["dS_unlagged"] = dS_un
                    rec["beta_unlagged"] = TH_w * dS_un / dE if dE != 0 else None
                rec.update({"cut_window_A": a_, "cut_window_B": b_, "dS_cut": dS, "dS_cut_nearest_clock": dS_near,
                            "beta": TH_w * dS / dE if dE != 0 else None,
                            "beta_nearest_clock": TH_w * dS_near / dE if dE != 0 else None,
                            "err_dS_floor": floor, "err_dS_Kc": kc, "err_dS_interp_ctrl": ric})
                errs = [e for e in (floor, kc, ric) if e is not None and np.isfinite(e)]
                rec["err_beta_from_dS"] = TH_w * float(np.sqrt(np.sum(np.square(errs)))) / abs(dE) if dE != 0 and errs else None
                if dE_k3 is not None:
                    rec["beta_with_kappa3_energy"] = TH_w * dS / dE_k3 if dE_k3 != 0 else None
                rows.append(rec)

    # ---------------- the 2 r_s cut's emission windows mapped forward to the collar (the formation window read at the cut's own clock)
    fwd_rows = []
    for (name, a_, b_) in (("formation_emit", 14.0, 40.0), ("formation_emit_a", 14.0, 26.0), ("formation_emit_b", 26.0, 36.0),
                           ("formation_emit_c", 36.0, 40.0), ("emit_40_60", 40.0, 60.0), ("emit_60_78", 60.0, 78.0)):
        Aa = float(np.interp(a_, t_em, arr16)); Ba = float(np.interp(b_, t_em, arr16))
        if not (np.isfinite(Aa) and np.isfinite(Ba)) or Ba > clk[-1]:
            continue
        THw = 0.5 * (float(np.interp(Aa, clk, TH)) + float(np.interp(Ba, clk, TH)))
        for setname, st in sets.items():
            dS = st["S"][idx(b_), 0] - st["S"][idx(a_), 0]
            dE = float(np.interp(Ba, clk, st["flowE"]) - np.interp(Aa, clk, st["flowE"]))
            fl = st["floor"][idx(b_), 0] + st["floor"][idx(a_), 0]
            kc = (st["Kc"][idx(b_), 0] + st["Kc"][idx(a_), 0]) if st["Kc"] is not None else None
            fwd_rows.append({"window": name, "cut": "2", "set": setname, "emit_A": a_, "emit_B": b_, "collar_A": Aa, "collar_B": Ba,
                             "T_H": THw, "dS_cut": dS, "dE_collar_interp": dE, "beta": THw * dS / dE if dE != 0 else None,
                             "err_dS_floor": fl, "err_dS_Kc": kc})

    # ---------------- method 2: the exterior shells' occupations (beyond 24 r_s)
    man = {m["label"]: m["realpath"] for m in d["manifest"]}
    occ = {}
    THref = None
    edges_lab = []
    for kap in range(1, 19):
        pj = man["B2 production kappa %d d13.json" % kap]
        pz = pj[:-5] + ".npz"
        if not os.path.exists(pz):
            raise FileNotFoundError("channel %d's d13.npz is missing: %s (v1.1: no silent relabelling of K18)" % (kap, pz))
        src[pz] = md5(pz)
        z = np.load(pz, allow_pickle=True)
        J = json.load(open(pj))
        ser = {"t": [], "S_tot": [], "E_tot": [], "S_band": [], "E_band": [], "S_raw": [], "S_vac": [], "Eout_P_check": [], "S_cal": [], "E_cal": [], "S_cal_band": [], "E_cal_band": [],
               "S_inc_0.3": [], "E_inc_0.3": [], "S_inc_0.1": [], "E_inc_0.1": [], "S_inc_0.03": [], "E_inc_0.03": []}
        for t in clk:
            key = "T%010.4f_k%d" % (t, kap)
            if key + "_shell_n_j" not in z.files or key + "_shell_nvac_j" not in z.files:
                continue
            e = np.asarray(z[key + "_shell_e"], float); n = np.asarray(z[key + "_shell_n_j"], float); nv = np.asarray(z[key + "_shell_nvac_j"], float)
            ck = J["clocks"]["%010.4f" % t]["per_channel"][str(kap)]
            if THref is None:
                THref = float(ck["T_H_ref"])
            band_top = 1.0
            sx = s_bin(n) - s_bin(nv); ex = e * (n - nv)
            ae = np.abs(e) / THref
            Sb, Eb = [], []
            for b in range(len(BANDS_TH) - 1):
                lo = BANDS_TH[b]; hi = BANDS_TH[b + 1] if BANDS_TH[b + 1] is not None else band_top / THref
                m = (ae >= lo) & (ae < hi)
                Sb.append(float(sx[m].sum())); Eb.append(float(ex[m].sum()))
            m = ae >= band_top / THref
            Sb.append(float(sx[m].sum())); Eb.append(float(ex[m].sum()))
            ser["t"].append(float(t)); ser["S_tot"].append(float(sx.sum())); ser["E_tot"].append(float(ex.sum()))
            ser["S_band"].append(Sb); ser["E_band"].append(Eb)
            ser["S_raw"].append(float(s_bin(n).sum())); ser["S_vac"].append(float(s_bin(nv).sum()))
            inv = (ck.get("inventory") or {}).get("24") or {}
            ser["Eout_P_check"].append(inv.get("E_D"))   # the stored Killing energy beyond 24 r_s on D = C - P_- (page_d13 inventory)
            # calibration: a Fermi-Dirac fill at the clock's T_H over the same e_j, against the ground 1(e < 0)
            TH_now = float(ck["T_H_now"])
            f = 1.0 / (np.exp(np.clip(e / TH_now, -700, 700)) + 1.0)
            g0 = (e < 0).astype(float)
            ser["S_cal"].append(float(s_bin(f).sum())); ser["E_cal"].append(float((e * (f - g0)).sum()))
            scb, ecb = [], []
            sf = s_bin(f); ef = e * (f - g0)
            for b in range(len(BANDS_TH) - 1):
                lo = BANDS_TH[b]; hi = BANDS_TH[b + 1] if BANDS_TH[b + 1] is not None else band_top / THref
                m = (ae >= lo) & (ae < hi)
                scb.append(float(sf[m].sum())); ecb.append(float(ef[m].sum()))
            m = ae >= band_top / THref
            scb.append(float(sf[m].sum())); ecb.append(float(ef[m].sum()))
            ser["S_cal_band"].append(scb); ser["E_cal_band"].append(ecb)
            for chi in (0.3, 0.1, 0.03):   # v1.1 (reviewer A R3): a thermal increment filling a fraction chi of each mode's weight
                ni = g0 + chi * (f - g0)
                ser["S_inc_%g" % chi].append(float(s_bin(ni).sum())); ser["E_inc_%g" % chi].append(float((e * (ni - g0)).sum()))
        for k_ in ser:
            ser[k_] = np.array(ser[k_], dtype=float) if k_ != "Eout_P_check" else np.array([np.nan if v is None else v for v in ser[k_]], float)
        occ[kap] = ser
    if not edges_lab:
        edges_lab = ["%g-%s" % (BANDS_TH[b], ("%g" % BANDS_TH[b + 1]) if BANDS_TH[b + 1] is not None else "top") for b in range(len(BANDS_TH) - 1)] + ["above"]
    kaps = [k for k in range(1, 19) if occ.get(k) is not None]
    tocc = occ[kaps[0]]["t"]
    for k in kaps:
        assert np.array_equal(occ[k]["t"], tocc), "clock sets differ across channels"
    wk = {k: 16.0 * k for k in kaps}
    agg = {}
    for nm, sel in (("k1", [1]), ("k12", [1, 2]), ("K18", kaps)):
        sel = [k for k in sel if k in kaps]
        agg[nm] = {key: sum(wk[k] * occ[k][key] for k in sel) for key in ("S_tot", "E_tot", "S_band", "E_band", "S_cal", "E_cal", "S_raw", "S_vac", "S_cal_band", "E_cal_band",
                                                                    "S_inc_0.3", "E_inc_0.3", "S_inc_0.1", "E_inc_0.1", "S_inc_0.03", "E_inc_0.03")}
        agg[nm]["Eout_P_check"] = sum(wk[k] * occ[k]["Eout_P_check"] for k in sel)
        agg[nm]["channels"] = sel
    # the energy check: sum_j e_j (n_j - nvac_j) against the stored class totals (soft + hard + above, out + in, _P)
    chk_rel = {}
    for nm in agg:
        ref = agg[nm]["Eout_P_check"]; m = np.abs(ref) > 1e-3
        chk_rel[nm] = {"max_rel_where_abs_gt_1e-3": float(np.nanmax(np.abs(agg[nm]["E_tot"][m] - ref[m]) / np.abs(ref[m]))) if m.any() else None,
                       "max_abs": float(np.nanmax(np.abs(agg[nm]["E_tot"] - ref))),
                       "label": "sum_j e_j (n_j - nvac_j) over the shells against the stored inventory E_D beyond 24 r_s, weighted 16 kappa"}
    lag1624 = float(np.nanmedian(lag16_24_em[(t_em16 >= 36) & (t_em16 <= 90)]))
    shift = 2.0 * round(lag1624 / 2.0)
    occ_rows = []
    TH_of = lambda t: float(np.interp(t, clk, TH))
    for (name, A, Bc) in windows():
        for shifted in (False, True):
            a_, b_ = (A + shift, Bc + shift) if shifted else (A, Bc)
            if b_ > tocc[-1] + 1e-9 or a_ < tocc[0] - 1e-9:
                continue
            ia = int(np.argmin(np.abs(tocc - a_))); ib = int(np.argmin(np.abs(tocc - b_)))
            if abs(tocc[ia] - a_) > 1e-9 or abs(tocc[ib] - b_) > 1e-9:
                continue
            THw = 0.5 * (TH_of(a_) + TH_of(b_))
            for nm in agg:
                g = agg[nm]
                dS = g["S_tot"][ib] - g["S_tot"][ia]; dE = g["E_tot"][ib] - g["E_tot"][ia]
                dSb = (g["S_band"][ib] - g["S_band"][ia]).tolist(); dEb = (g["E_band"][ib] - g["E_band"][ia]).tolist()
                dSc = g["S_cal"][ib] - g["S_cal"][ia]; dEc = g["E_cal"][ib] - g["E_cal"][ia]
                occ_rows.append({"window": name, "A_collar": A, "B_collar": Bc, "shifted_by": shift if shifted else 0.0, "A": a_, "B": b_, "set": nm,
                                 "T_H": THw, "dS_occ": dS, "dE_occ": dE, "beta_occ": THw * dS / dE if dE != 0 else None,
                                 "dS_band": dSb, "dE_band": dEb,
                                 "beta_band": [THw * s_ / e_ if e_ != 0 else None for s_, e_ in zip(dSb, dEb)],
                                 "S_occ_A": g["S_tot"][ia], "S_occ_B": g["S_tot"][ib], "E_occ_A": g["E_tot"][ia], "E_occ_B": g["E_tot"][ib]})
    cal = {}
    # the calibration's level: T_H S_cal / E_cal at single clocks (a thermal fill against the ground), expected 2 for a flat mode density
    for nm in agg:
        g = agg[nm]
        THc = np.interp(tocc, clk, TH)
        cal[nm] = {"t": tocc.tolist(), "beta_fill": (THc * g["S_cal"] / g["E_cal"]).tolist(),
                   "beta_fill_band": (THc[:, None] * g["S_cal_band"] / np.where(g["E_cal_band"] == 0, np.nan, g["E_cal_band"])).tolist(),
                   "label": "T_H S/E of a Fermi-Dirac fill at the clock's T_H of the stored e_j against the block's ground 1(e < 0), weighted 16 kappa; per band of |e|/T_H^ref as rows_occ"}

    # ---------------- v1.1: the reading's additional readings
    import a4_beta_eff_extras_v1_1 as X
    THc = np.interp(tocc, clk, TH); m40 = tocc >= 40 - 1e-9
    increments = {nm: {"chi_%g" % chi: float(np.median((THc * agg[nm]["S_inc_%g" % chi] / agg[nm]["E_inc_%g" % chi])[m40])) for chi in (0.3, 0.1, 0.03)}
                  for nm in agg}
    def lag_16_to(R1):
        def f_(tA):
            return Fd.travel(16 * RS, R1 * RS, tA * RS) / RS
        return f_
    ctx = {"clk": clk, "TH": TH, "RS": RS, "sets": sets, "emit_of_arrival": emit_of_arrival, "S_at": S_at, "flux": Fx, "agg": agg, "tocc": tocc,
           "TH_of": TH_of, "bands": edges_lab, "increments": increments, "SUM": SUM, "WX0": WX[0], "WX": WX, "flow": flow, "B2": B,
           "cal": {nm: float(np.median(np.array(cal[nm]["beta_fill"])[m40])) for nm in cal}, "rows_occ": occ_rows,
           "lag_fixed_form": lag_fixed_form, "lag_fixed_late": lag_fixed_late, "collar_split": d["collar_split"], "lag_16_to": lag_16_to}
    v11 = X.run(ctx)

    res = {"script": os.path.abspath(__file__), "label": __doc__.split("\n")[0], "sources_md5": src,
           "rs_a": RS, "G": G, "cuts": CUTS, "clocks": clk.tolist(), "T_H": TH.tolist(), "E": E.tolist(),
           "lag": {"t_emit_2rs": t_em.tolist(), "t_arrive_16rs": arr16.tolist(), "lag_2_16": lag2_16_em.tolist(),
                   "t_emit_16rs": t_em16.tolist(), "lag_16_24": lag16_24_em.tolist(),
                   "fixed_charge_2_16_formation": lag_fixed_form, "fixed_charge_2_16_late_charge": lag_fixed_late,
                   "fixed_charge_16_24_formation": lag_fixed_16_24_form, "shells_shift_used": shift, "lag_16_24_median_36_90": lag1624,
                   "definition": "outgoing light travel time (r_s) in the B2 record's applied fields, dR/dt = N - |v|, RK4 dt = 1 a, fields linear in T between records and in R between sites"},
           "rows_cut": rows, "rows_cut_forward": fwd_rows, "rows_occ": occ_rows, "occ_bands_TH": edges_lab, "T_H_ref": THref, "occ_channels": kaps,
           "occ_energy_check_max_rel": chk_rel, "occ_calibration": cal, "v1_1": v11,
           "series": {"S_cut_K18": SUM.tolist(), "S_cut_k1": WX[0].tolist(), "S_cut_k12": (WX[0] + WX[1]).tolist(),
                      "flow_k1": flow[0].tolist(), "flow12": flow12.tolist(), "flow18": flow18.tolist(), "flow123": flow123.tolist(),
                      "FMW": FMW.tolist(), "SHARE": SHARE.tolist(),
                      "occ_t": tocc.tolist(), "S_occ_K18": agg["K18"]["S_tot"].tolist(), "E_occ_K18": agg["K18"]["E_tot"].tolist(),
                      "S_occ_k1": agg["k1"]["S_tot"].tolist(), "E_occ_k1": agg["k1"]["E_tot"].tolist(),
                      "S_occ_k12": agg["k12"]["S_tot"].tolist(), "E_occ_k12": agg["k12"]["E_tot"].tolist()},
           "seconds": time.time() - t_start}
    json.dump(res, open(os.path.join(out, "A4_beta_eff_v1.1.json"), "w"), indent=1, default=float)
    # npz for the figure
    np.savez(os.path.join(out, "A4_beta_eff_v1.1.npz"),
             clocks=clk, T_H=TH, S_K18=SUM, S_k1=WX[0], S_k12=WX[0] + WX[1], flow_k1=flow[0], flow12=flow12, flow18=flow18,
             t_em=t_em, lag2_16=lag2_16_em, t_em16=t_em16, lag16_24=lag16_24_em,
             occ_t=tocc, S_occ_K18=agg["K18"]["S_tot"], E_occ_K18=agg["K18"]["E_tot"], S_occ_k1=agg["k1"]["S_tot"], E_occ_k1=agg["k1"]["E_tot"],
             v11_json=np.array(json.dumps(v11, default=float)), rows_cut_json=np.array(json.dumps(rows, default=float)), rows_occ_json=np.array(json.dumps(occ_rows, default=float)),
             rows_fwd_json=np.array(json.dumps(fwd_rows, default=float)),
             occ_bands=np.array(edges_lab), cal_t=tocc,
             cal_K18=np.array(cal["K18"]["beta_fill"]), cal_k1=np.array(cal["k1"]["beta_fill"]),
             cal_K18_band=np.array(cal["K18"]["beta_fill_band"]), cal_k1_band=np.array(cal["k1"]["beta_fill_band"]))
    print("done in %.1f s; lag 2->16 at arrival 46, 70, 100: %s; fixed-charge %.3f (formation), %.3f (late)" % (
        time.time() - t_start, [round(t - emit_of_arrival(t), 3) for t in (46.0, 70.0, 100.0)], lag_fixed_form, lag_fixed_late))
    print("lag 16->24 median %.3f -> shells shift %.1f r_s; occ energy check max rel %s" % (lag1624, shift, chk_rel))


if __name__ == "__main__":
    main()
