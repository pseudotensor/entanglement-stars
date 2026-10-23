#!/usr/bin/env python
"""pt_testA_run_v1.py (busi, inbox #428, Test A: reviewer A's prescribed-recession chain runs, ReviewA_pc6_v1.md sec. 4, observables (i)
and (iii)). One light chain (kappa 1 or 2) on the record's formation fields to T_rec and then held (L_eff = 0, the control) or with the
enclosed charge lowered at dE/dt = -L_eff (pt_fieldsA_v1). Observable: f_out(omega, u) at the collar in both bands, read the paper's way:
fixed outgoing packets of Killing frequency omega (sign = band) whose centre crosses the collar R_col = 16 nominal r_s at the clock
T_rec + u, back-propagated through the receding (or held) fields and the stored formation to T = 0, and occupied there by the smoothed
negative spectral occupation f(h(0)) of the initial Hamiltonian (J3P's state: the continued sea at T = 0 carried through the formation),
n = <chi|f_ws(h(0))|chi>, ws = 1/(80 r_s,nom) (J3P's width) and its ladder ws/2, ws/4, the reading at ws/4 (the ladder
converges there: ws/2 against ws/4 is the projector error printed); the transmission-weighted single-packet Fermi fit
n = sum rho Gamma_kappa f(omega; T) (holes 1 - n for omega < 0) against T_H(E(T_rec + u)) and kappa_T/2 pi.

The chain is long enough that nothing returns: its sites reach beyond every packet's back-propagated front at T = 0 (the front speed
bound 1.08 (1 + |v| beyond the packets) plus a margin of 2048 sites; the weight in the last 512 active rows is monitored). The propagation: CF4 steps of --dt-form
inside each record interval (fields linear between records), CF4 steps of --dt-post through the recession (exact exponentials on held
fields), Chebyshev at tol 1e-15 (the engine's cheb_order), on the device (cuda:1 or cuda:2 only, checked free)."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "code_v6"))
import pt_lib_v1 as L  # noqa: E402
import pt_fieldsA_v1 as FA  # noqa: E402
import page_ops_v6 as OPS  # noqa: E402

log = L.log
FRONT = 1.08          # the back-propagated front speed bound beyond the packets (1 + |v| < 1.06 there); the edge weight monitors it


def parse_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="P64_s4_B2n4096")
    ap.add_argument("--T-rec-rs", type=float, default=100.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--Leff", type=float, default=0.0, help="L_eff in units of L_H (0 = the held control)")
    ap.add_argument("--prep", required=True, help="the prep json (L_H, the table beside it)")
    ap.add_argument("--omegas", default="-6,-3,-2,-1,-0.5,0.5,1,2,3,6", help="omega/T_H(E_rec); sign = band")
    ap.add_argument("--u-rs", default="25,50,100,150,200,300,400,600,800,1000,1250,1500,1750,2000", help="collar clocks u (nominal r_s) after T_rec")
    ap.add_argument("--R-col-rs", type=float, default=16.0)
    ap.add_argument("--dt-form", type=float, default=4.0)
    ap.add_argument("--dt-post", type=float, default=64.0)
    ap.add_argument("--held-chunk", type=float, default=4096.0)
    ap.add_argument("--ws-rs", type=float, default=1.0 / 80.0)
    ap.add_argument("--ws-factors", default="0.25,0.5,1", help="the first is the reading, the rest the projector ladder")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-packets", type=int, default=0, help="debug: keep the first n packets")
    ap.add_argument("--u-max-rs", type=float, default=0.0, help="debug: drop packets beyond this u")
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    P = json.load(open(a.prep))
    H = FA.load_record(a.label, a.T_rec_rs)
    g, E_rec, rsn = H["g"], H["E_rec"], H["rs_nom"]
    rsE = 2 * g * E_rec
    TH = 1.0 / (4 * math.pi * rsE)
    LH = P["L_H"]
    Labs = a.Leff * LH
    tabz = np.load(a.prep.replace(".json", "_table.npz"))
    table = {"lam": tabz["lam"], "N": tabz["N"], "v": tabz["v"], "E": tabz["E"]}
    kap = a.kappa
    Rcol = a.R_col_rs * rsn
    T_rec = H["T_rec"]
    kT = H["rec"]["kappa_T"]
    log("Test A run %s: record %s at T/r_s %g (E %.6f, 2gE %.4f, T_H %.6e, kappa_T %.6e), kappa %g, L_eff %g L_H = %.6e, device %s" % (
        a.tag, a.label, a.T_rec_rs, E_rec, rsE, TH, kT, kap, a.Leff, Labs, a.device))
    # ---------------------------------------------------------------- the packets
    oms = parse_list(a.omegas); us = parse_list(a.u_rs)
    if a.u_max_rs > 0:
        us = [u for u in us if u <= a.u_max_rs]
    specs = []
    for om_th in oms:
        om = om_th * TH
        sig = abs(om) / 6.0 if abs(om_th) >= 1.0 else abs(om) / 2.0
        sd = 1.0 / (2.0 * sig)
        x0 = max(Rcol + 7.4 * sd, 3.0 * kap / abs(om) + 7.4 * sd)
        for u in us:
            specs.append({"omega_over_TH": om_th, "omega": om, "sig_om": sig, "sd_t": sd, "x0": float(round(x0)), "u_rs": u})
    if a.max_packets:
        specs = specs[: a.max_packets]
    # the travel from the collar to x0 by the continuum outgoing ray on the record's exterior (the u label)
    vf0 = lambda R: -math.sqrt(E_rec) * np.sqrt(2 * g / np.asarray(R, float))
    for s in specs:
        s["travel"] = L.retarded_travel(vf0, Rcol, s["x0"])
        s["T_obs"] = T_rec + s["u_rs"] * rsn + s["travel"]
        s["hi_est"] = s["x0"] + 7.4 * s["sd_t"] * 1.05
    t_max = max(s["T_obs"] for s in specs)
    front0 = max(s["hi_est"] + FRONT * s["T_obs"] for s in specs)
    n = int(math.ceil((front0 + 2048) / 1024.0)) * 1024
    log("%d packets, t_max %.1f (u_max %.0f r_s), chain %d sites (%.1f nominal r_s)" % (len(specs), t_max, max(us), n, n / rsn))
    F = FA.FieldsA(H, n, a.device, L_eff=Labs, table=table if Labs > 0 else None)
    log("charge at t_max %.6f (lambda %.6f); E(1) re-solved %s" % (F.charge(t_max), F.lam(t_max), getattr(F, "E1", None)))
    ch = L.Chain(n, kap, a.device)
    # build packets on the fields at their T_obs
    order = sorted(range(len(specs)), key=lambda j: -specs[j]["T_obs"])
    specs = [specs[j] for j in order]
    K = len(specs)
    PR = L.Prop(ch.rows, K, a.device)
    X = torch.zeros((ch.rows, K), dtype=L.CD, device=a.device)
    psis = []
    for j, s in enumerate(specs):
        Nn, vn = F.numpy_fields(s["T_obs"])
        Nf = lambda R, Nn=Nn: Nn[np.asarray(R, int) - 1]
        vfn = lambda R, vn=vn: vn[np.asarray(R, int) - 1]
        psi, info = L.wkb_packet(Nf, vfn, kap, s["omega"], s["sig_om"], s["x0"], n)
        s["info"] = info
        s["hi"] = info["hi"]
        if j < 3 or j % 20 == 0:
            e1, esd = L.packet_energy_check(ch, Nn, vn, psi)
            s["energy_check"] = {"mean_over_omega": e1 / s["omega"], "sd_over_sig": esd / s["sig_om"]}
            log("  packet %d: omega/T_H %+g u %g r_s x0 %.0f T_obs %.1f: <h>/omega %.5f, sd/sig %.4f, edge %.1e, dropped %.1e" % (
                j, s["omega_over_TH"], s["u_rs"], s["x0"], s["T_obs"], e1 / s["omega"], esd / s["sig_om"], info["edge_weight"], info["dropped_weight"]))
        psis.append(psi)
    psis = np.array(psis).T
    # ---------------------------------------------------------------- the backward sweep
    pending = list(range(K))
    Ka = 0
    def front_rows(t):
        if Ka == 0:
            return 0
        f = max(specs[j]["hi"] + FRONT * (specs[j]["T_obs"] - t) for j in range(Ka)) + 2048
        return 2 * int(min(n, math.ceil(f)))
    def add_due(t):
        nonlocal Ka
        while pending and specs[pending[0]]["T_obs"] >= t - 1e-9:
            j = pending.pop(0)
            assert j == Ka
            X[:, j] = torch.as_tensor(psis[:, j], dtype=L.CD, device=a.device)
            Ka += 1
    # the clock grid: post-record then formation
    Tobs_set = sorted({s["T_obs"] for s in specs}, reverse=True)
    grid = set(Tobs_set)
    grid.add(T_rec)
    if Labs > 0:
        t = T_rec
        while t < t_max:
            grid.add(t); t += a.dt_post
    else:
        pts = sorted(grid)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            m = int(math.ceil((p1 - p0) / a.held_chunk))
            for i in range(1, m):
                grid.add(p0 + i * (p1 - p0) / m)
    t = T_rec
    while t > 1e-9:
        t -= a.dt_form
        grid.add(max(t, 0.0))
    grid.add(0.0)
    grid = sorted(grid, reverse=True)
    grid = [x for x in grid if x <= t_max + 1e-9]
    log("clock grid: %d steps (post %d, formation %d)" % (len(grid) - 1, sum(1 for x in grid if x > T_rec), sum(1 for x in grid if x < T_rec)))
    c1, c2 = OPS.CF4_C; a1, a2 = OPS.CF4_ALPHA
    Dcache = {}
    def diag_at(tt):
        N_, v_ = F.device_fields(tt)
        return N_, v_
    t_cur = grid[0]
    add_due(t_cur)
    tick = time.time(); n_steps = len(grid) - 1
    mon = {"edge_max": 0.0}
    for i in range(n_steps):
        t1, t0 = grid[i], grid[i + 1]
        D = t1 - t0
        rows = front_rows(t0)
        held_step = (t0 >= T_rec - 1e-9 and Labs == 0.0)
        if held_step:
            N_, v_ = F.device_fields(0.5 * (t0 + t1))
            Dh = ch.diags(N_, v_)
            PR.expm(Dh, X, -D, rows, Ka)
        else:
            Na, va = F.device_fields(t0 + c1 * D); Nb, vb = F.device_fields(t0 + c2 * D)
            DA = ch.diags(2 * (a1 * Na + a2 * Nb), 2 * (a1 * va + a2 * vb))
            DB = ch.diags(2 * (a2 * Na + a1 * Nb), 2 * (a2 * va + a1 * vb))
            PR.expm(DB, X, -0.5 * D, rows, Ka)
            PR.expm(DA, X, -0.5 * D, rows, Ka)
        add_due(t0)
        if i % 200 == 0 or i == n_steps - 1:
            r = front_rows(t0)
            edge = float((X[max(0, r - 1024): r, :Ka].abs() ** 2).sum(dim=0).max()) if Ka else 0.0
            mon["edge_max"] = max(mon["edge_max"], edge)
            el = time.time() - tick
            log("  step %d/%d t %.1f (T/r_s %.3f) active %d cols, %d rows; terms %d; tail acc %.2e; edge weight %.1e; %.0f s, eta %.0f s" % (
                i + 1, n_steps, t0, t0 / rsn, Ka, r, L.STATS["terms"], L.STATS["tail_acc"], edge, el, el / (i + 1) * (n_steps - i - 1)))
    assert Ka == K
    # ---------------------------------------------------------------- occupations at T = 0
    N0, v0 = F.device_fields(0.0)
    D0 = ch.diags(N0, v0)
    rows0 = front_rows(0.0)
    w0 = L.Chain.bound(D0[:, :rows0])
    ws = a.ws_rs / rsn
    wsf = parse_list(a.ws_factors)
    M = L.step_order(w0, ws * min(wsf))
    log("moments of h(0): w %.5f, ws %.4e, M %d" % (w0, ws, M))
    mu = PR.moments(D0, X, M, rows0, K, w0).cpu().numpy()
    occ = {}
    for f in wsf:
        o, last = L.occupation_from_moments(mu, w0, ws * f)
        occ[f] = o
        log("  ws x %g: last coefficient %.1e" % (f, last))
    norm0 = mu[0]
    # ---------------------------------------------------------------- readouts
    rows_out = []
    for j, s in enumerate(specs):
        info = s["info"]
        Eobs = F.charge(s["T_obs"] - s["travel"])
        rs_u = 2 * g * Eobs
        TH_u = 1.0 / (4 * math.pi * rs_u)
        n_main = float(occ[wsf[0]][j])
        part = n_main if s["omega"] > 0 else 1.0 - n_main
        nH = L.hawking_n(info, kap, rs_u, TH_u)
        nT = L.hawking_n(info, kap, rs_u, kT / (2 * math.pi))
        Tf = L.fit_T(info, kap, rs_u, part, TH_u)
        row = {"omega_over_TH": s["omega_over_TH"], "band": info["band"], "u_rs": s["u_rs"], "T_obs": s["T_obs"], "x0": s["x0"], "sig_om": s["sig_om"],
               "n": n_main, "n_ws": {str(f): float(occ[f][j]) for f in wsf}, "particle_or_hole": part,
               "ws_spread": float(abs(occ[wsf[1]][j] - occ[wsf[0]][j])) if len(wsf) > 1 else None,
               "E_at_u": Eobs, "T_H_at_u": TH_u, "n_Hawking_TH": nH, "ratio_TH": part / nH if nH > 0 else None,
               "n_Hawking_kappaT": nT, "ratio_kappaT": part / nT if nT > 0 else None,
               "T_fit_over_TH": (Tf / TH_u) if Tf else None, "T_fit_over_kappaT_2pi": (Tf / (kT / (2 * math.pi))) if Tf else None,
               "norm_defect_T0": float(norm0[j] - 1.0), "Gamma_w": float(np.sum(info["rho"] * L.gamma_k(kap, np.abs(info["oms"]) * rs_u))),
               "packet": {k: info[k] for k in ("vg0", "edge_weight", "dropped_weight", "n_om_used", "lo", "hi")}, "energy_check": s.get("energy_check")}
        rows_out.append(row)
    rows_out.sort(key=lambda r: (r["omega_over_TH"], r["u_rs"]))
    log("table (particle or hole occupation / Hawking at T_H(u); T_fit/T_H):")
    for r in rows_out:
        log("  w/T_H %+5.1f u %6.0f r_s: n %.6e  p|h %.6e  n_H %.6e  ratio %s  T/T_H %s  ws-spread %s" % (
            r["omega_over_TH"], r["u_rs"], r["n"], r["particle_or_hole"], r["n_Hawking_TH"],
            ("%.4f" % r["ratio_TH"]) if r["ratio_TH"] is not None else "-", ("%.4f" % r["T_fit_over_TH"]) if r["T_fit_over_TH"] else "-", ("%.1e" % r["ws_spread"]) if r["ws_spread"] is not None else "-"))
    J = {"script": "pt_testA_run_v1.py", "args": vars(a), "tag": a.tag, "record": {k: H[k] for k in ("label", "npz_md5", "json_md5", "T_rec", "E_rec", "g", "ell", "tau")},
         "record_json": H["rec"], "rs_E": rsE, "T_H": TH, "L_H": LH, "L_eff_abs": Labs, "kappa": kap, "nsites": n, "R_col": Rcol,
         "charge_t_max": F.charge(t_max), "lambda_t_max": F.lam(t_max), "stats": dict(L.STATS), "monitor": mon, "moments_M": M, "w0": w0, "ws": ws,
         "engine_md5s": L.engine_md5s(), "rows": rows_out, "seconds": time.time() - t_start}
    stem = os.path.join(a.out, a.tag)
    json.dump(J, open(stem + ".json", "w"), indent=1, default=float)
    np.savez(stem + "_occ.npz", omega_over_TH=np.array([s["omega_over_TH"] for s in specs]), u_rs=np.array([s["u_rs"] for s in specs]),
             T_obs=np.array([s["T_obs"] for s in specs]), **{"occ_ws%g" % f: occ[f] for f in wsf}, norm0=norm0)
    log("wrote %s.json (%.0f s)" % (stem, time.time() - t_start))


if __name__ == "__main__":
    main()
