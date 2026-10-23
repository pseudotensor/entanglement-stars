#!/usr/bin/env python
"""pt3_stage2_held_audit.py (busi, PAGE design v3.1 stage 2: the native held joint-output audit, F9, A8, A12.2; also the L_eff = 0 row of
sec. 4). The massless k = 1 probe (v1's Test A chain, page_ops_v6.chain_sparse at m = 0) on the record P64_s4_B2n4096's formation fields to
T/r_s 100 and then held (pt_fieldsA_v1, L_eff = 0), on a self-adjoint parent long enough that nothing returns before the last readout
(>= 1e4 r_s; certified from the symbol's fastest carrier, pt3_output.certify_exterior).

Start (b) (v3.1 sec. 1): C_0 = theta(-h(0)), approximated by f_ws(h(0)) on the ws ladder ws, ws/2, ws/4 (ws = 1/(80 r_s)); the ladder's
spread on the output block certifies the approximation (eps of the entropy intervals). Start (a) is the same contraction at ws itself (the
tie to v1).

Output algebra: outgoing Killing-frequency packets in both bands at the supplying line |omega| = omega_line (default 3.95 T_H, the k = 1
inverted pair), Gaussian in omega with temporal width sd_t, whose centre crosses the collar R_col = 16 r_s at u_k = u_0 + k du (k = 0 ..
n_bins - 1), back-propagated with the fields to T = 0 (v1's method: CF4 through the formation, exact exponentials on the held fields), so
each column is the Heisenberg mode at T = 0; their Gram G = X^dag X is the anticommutator matrix of the output modes. A second family at
twice du (the binning convergence, --family2). Inventory columns: the held operator's interior-localized window eigenvectors (w_in(R <= R_in)
> 1/2, |e| < 10 T_H, from the 4096-site domain's sector eigensolve, embedded), back-propagated from the last readout clock: the retained
excitation-bearing content at the end of the readout.
At T = 0: one Chebyshev recursion of h(0) with three accumulators gives Y_f = f_ws(h(0)) X at the three widths; stored are G and
Cf = X^dag Y_f (K x K each, complex), the norm defects and the edge weights. The entropies are read by pt3_stage2_read.py.
Device: cuda:1 or cuda:2 only (checked free by the queue), never cuda:0."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PT)
sys.path.insert(0, os.path.join(PT, "code_v6"))
import pt_lib_v1 as L  # noqa: E402
import pt_fieldsA_v1 as FA  # noqa: E402
import page_ops_v6 as OPS  # noqa: E402
import pt3_core as K3  # noqa: E402
import pt3_output as OUT  # noqa: E402

log = L.log
FRONT = 1.08


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="P64_s4_B2n4096")
    ap.add_argument("--T-rec-rs", type=float, default=100.0)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--omega-line", type=float, default=3.95, help="|omega|/T_H of the bins (the supplying line)")
    ap.add_argument("--bands", default="1,-1")
    ap.add_argument("--u0-rs", type=float, default=50.0)
    ap.add_argument("--du-rs", type=float, default=250.0)
    ap.add_argument("--u-max-rs", type=float, default=8350.0, help=">= 3/Gamma_res = 8.3e3 r_s (F9)")
    ap.add_argument("--sdt-over-du", type=float, default=0.4, help="the packet's temporal standard deviation over the bin spacing")
    ap.add_argument("--family2", type=int, default=1, help="1: also the family at 2 du (binning convergence)")
    ap.add_argument("--R-in", type=float, default=64.0, help="the inventory's interior radius (sites)")
    ap.add_argument("--R-col-rs", type=float, default=16.0)
    ap.add_argument("--dt-form", type=float, default=4.0)
    ap.add_argument("--held-chunk", type=float, default=4096.0)
    ap.add_argument("--ws-rs", type=float, default=1.0 / 80.0)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-cols", type=int, default=0, help="debug: keep the first n columns")
    ap.add_argument("--no-bins", action="store_true", help="pre-launch print: inventory columns only")
    ap.add_argument("--baseline-vacuum", action="store_true", help="reviewer A pst1 item 4: the probe's held Killing vacuum P_-(h(T_rec)): back-propagate "
                    "to T_rec only and occupy with f_ws(h(T_rec)) (stationary: the numerical floor and sec. 10's baseline)")
    ap.add_argument("--inv-u-rs", type=float, default=None, help="the inventory's clock u (r_s after T_rec); default the last readout")
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    H = FA.load_record(a.label, a.T_rec_rs)
    g, E_rec, rsn = H["g"], H["E_rec"], H["rs_nom"]
    TH = 1.0 / (8 * math.pi * g * E_rec)
    kap = a.kappa
    Rcol = a.R_col_rs * rsn
    T_rec = H["T_rec"]
    om_line = a.omega_line * TH
    log("stage 2 held audit %s: record %s T/r_s %g, E %.6f, T_H %.6e, omega_line %.4f T_H, device %s" % (a.tag, a.label, a.T_rec_rs, E_rec, TH, a.omega_line, a.device))
    vf0 = lambda R: -math.sqrt(E_rec) * np.sqrt(2 * g / np.asarray(R, float))
    # ------------------------------------------------------------------ the output bins
    specs = []
    fams = [(1, a.du_rs)] + ([(2, 2 * a.du_rs)] if a.family2 else [])
    for fam, du in fams:
        sd = a.sdt_over_du * du * rsn
        sig = 1.0 / (2.0 * sd)
        nb = int(math.floor((a.u_max_rs - a.u0_rs) / du)) + 1
        for b in [int(x) for x in a.bands.split(",")]:
            om = b * om_line
            x0 = max(Rcol + 7.4 * sd, 3.0 * kap / abs(om) + 7.4 * sd)
            for k in range(nb):
                u = a.u0_rs + k * du
                specs.append({"kind": "bin", "family": fam, "band": b, "bin": k, "omega": om, "sig_om": sig, "sd_t": sd, "x0": float(round(x0)), "u_rs": u})
    if a.no_bins:
        specs = []
    for s in specs:
        s["travel"] = L.retarded_travel(vf0, Rcol, s["x0"])
        s["T_obs"] = T_rec + s["u_rs"] * rsn + s["travel"]
        s["hi_est"] = s["x0"] + 7.4 * s["sd_t"] * 1.05
    t_last = max(s["T_obs"] for s in specs) if specs else T_rec
    t_inv = t_last if a.inv_u_rs is None else T_rec + a.inv_u_rs * rsn
    # ------------------------------------------------------------------ the inventory columns (held operator's interior window levels)
    ne = len(H["R_sites"])
    ops4 = K3.ChainOps(ne, kap)
    h4 = ops4.h(H["N_app"][H["k_rec"]], H["v_app"][H["k_rec"]])
    e4, U4, s4 = K3.full_spectrum_from_sector(h4)
    w4 = np.real((np.abs(U4[ops4.site_mask(a.R_in), :]) ** 2).sum(axis=0))
    inv = np.where((np.abs(e4) < 10 * TH) & (w4 > 0.5))[0]
    log("inventory: %d interior window levels (|e| < 10 T_H, w_in(R <= %g) > 1/2): e/T_H %s, sectors %s, w_in %s" % (
        len(inv), a.R_in, np.round(e4[inv] / TH, 3).tolist(), s4[inv].tolist(), np.round(w4[inv], 4).tolist()))
    for j in inv:
        specs.append({"kind": "inventory", "e_over_TH": float(e4[j] / TH), "w_in": float(w4[j]), "sector": int(s4[j]), "T_obs": t_inv, "hi_est": float(ne), "x0": None, "u_rs": None})
    if a.max_cols:
        specs = specs[: a.max_cols]
    front0 = max(s["hi_est"] + FRONT * s["T_obs"] for s in specs)
    n = int(math.ceil((front0 + 2048) / 1024.0)) * 1024
    log("%d columns (%d bins, %d inventory), t_last %.1f (T/r_s %.2f), chain %d sites (%.1f r_s)" % (len(specs), sum(s["kind"] == "bin" for s in specs),
                                                                                                  sum(s["kind"] == "inventory" for s in specs), t_last, t_last / rsn, n, n / rsn))
    F = FA.FieldsA(H, n, a.device, L_eff=0.0)
    # the exterior certificate on the actual parent at the held clock and at T = 0
    Nh, vh = F.numpy_fields(T_rec); N0h, v0h = F.numpy_fields(0.0)
    R = np.arange(1, n + 1, dtype=float)
    cert = {"held": OUT.certify_exterior(Nh, vh, kap, R, Rcol, t_last, rsn), "T0": OUT.certify_exterior(N0h, v0h, kap, R, Rcol, t_last, rsn)}
    log("certificate: L %.0f r_s, v_max %.4f (bound %.4f), first return %.0f r_s (bound route %.0f r_s) against %.0f needed: %s" % (
        cert["held"]["L_rs"], cert["held"]["v_max"], cert["held"]["v_bound"], cert["held"]["T_return_rs"], cert["held"]["T_return_bound_rs"],
        cert["held"]["T_needed_rs"], cert["held"]["certified"]))
    ch = L.Chain(n, kap, a.device)
    order = sorted(range(len(specs)), key=lambda j: -specs[j]["T_obs"])
    specs = [specs[j] for j in order]
    Kc = len(specs)
    PR = L.Prop(ch.rows, Kc, a.device)
    X = torch.zeros((ch.rows, Kc), dtype=L.CD, device=a.device)
    psis = np.zeros((ch.rows, Kc), complex)
    for j, s in enumerate(specs):
        if s["kind"] == "bin":
            Nn, vn = F.numpy_fields(s["T_obs"])
            Nf = lambda R_, Nn=Nn: Nn[np.asarray(R_, int) - 1]
            vfn = lambda R_, vn=vn: vn[np.asarray(R_, int) - 1]
            psi, info = L.wkb_packet(Nf, vfn, kap, s["omega"], s["sig_om"], s["x0"], n)
            s["info"] = {k: (v if not isinstance(v, np.ndarray) else None) for k, v in info.items()}
            s["hi"] = info["hi"]
            if j < 2 or s["bin"] in (0, 10):
                e1, esd = L.packet_energy_check(ch, Nn, vn, psi)
                s["energy_check"] = {"mean_over_omega": e1 / s["omega"], "sd_over_sig": esd / s["sig_om"]}
            psis[:, j] = psi
        else:
            jj = inv[np.argmin(np.abs(e4[inv] / TH - s["e_over_TH"]))]
            psis[: 2 * ne, j] = U4[:, jj]
            s["hi"] = ne
    # ------------------------------------------------------------------ the backward sweep (v1's)
    pending = list(range(Kc))
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
            X[:, j] = torch.as_tensor(psis[:, j], dtype=L.CD, device=a.device)
            Ka += 1
    grid = set(s["T_obs"] for s in specs)
    grid.add(T_rec)
    pts = sorted(grid)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        m = int(math.ceil((p1 - p0) / a.held_chunk))
        for i in range(1, m):
            grid.add(p0 + i * (p1 - p0) / m)
    t_stop = T_rec if a.baseline_vacuum else 0.0
    if not a.baseline_vacuum:
        t = T_rec
        while t > 1e-9:
            t -= a.dt_form
            grid.add(max(t, 0.0))
        grid.add(0.0)
    grid = sorted([x for x in grid if x >= t_stop - 1e-9], reverse=True)
    log("clock grid: %d steps" % (len(grid) - 1))
    c1, c2 = OPS.CF4_C; a1, a2 = OPS.CF4_ALPHA
    add_due(grid[0])
    tick = time.time(); n_steps = len(grid) - 1
    mon = {"edge_max": 0.0}
    for i in range(n_steps):
        t1, t0 = grid[i], grid[i + 1]
        D = t1 - t0
        rows = front_rows(t0)
        if t0 >= T_rec - 1e-9:
            N_, v_ = F.device_fields(0.5 * (t0 + t1))
            PR.expm(ch.diags(N_, v_), X, -D, rows, Ka)
        else:
            Na, va = F.device_fields(t0 + c1 * D); Nb, vb = F.device_fields(t0 + c2 * D)
            DA = ch.diags(2 * (a1 * Na + a2 * Nb), 2 * (a1 * va + a2 * vb))
            DB = ch.diags(2 * (a2 * Na + a1 * Nb), 2 * (a2 * va + a1 * vb))
            PR.expm(DB, X, -0.5 * D, rows, Ka)
            PR.expm(DA, X, -0.5 * D, rows, Ka)
        add_due(t0)
        if i % 100 == 0 or i == n_steps - 1:
            r = front_rows(t0)
            edge = float((X[max(0, r - 1024): r, :Ka].abs() ** 2).sum(dim=0).max()) if Ka else 0.0
            mon["edge_max"] = max(mon["edge_max"], edge)
            el = time.time() - tick
            log("  step %d/%d t %.1f (T/r_s %.2f) active %d, rows %d; terms %d; tail acc %.2e; edge %.1e; %.0f s, eta %.0f s" % (
                i + 1, n_steps, t0, t0 / rsn, Ka, r, L.STATS["terms"], L.STATS["tail_acc"], edge, el, el / (i + 1) * (n_steps - i - 1)))
    assert Ka == Kc
    # ------------------------------------------------------------------ T = 0: G and Cf at the ws ladder
    N0, v0 = F.device_fields(t_stop)
    D0 = ch.diags(N0, v0)
    rows0 = front_rows(t_stop)
    w0 = L.Chain.bound(D0[:, :rows0])
    ws = a.ws_rs / rsn
    wsl = [ws, ws / 2, ws / 4]
    M = L.step_order(w0, wsl[-1])
    coefs = [L.step_coeffs(w0, x, M) for x in wsl]
    log("T = 0: w %.5f, ws %.4e, M %d, rows %d" % (w0, ws, M, rows0))
    G = (X[:rows0].conj().T @ X[:rows0]).cpu().numpy()
    Ys = [torch.zeros((rows0, Kc), dtype=L.CD, device=a.device) for _ in wsl]
    T0b = X.clone(); T1b, T2b = PR.B[1], PR.B[2]
    for y, c in zip(Ys, coefs):
        y += complex(c[0]) * T0b[:rows0]
    L._launch(None, T0b, T1b, None, D0, rows0, Kc, 1.0 / w0, 0j, 1, 0)
    for y, c in zip(Ys, coefs):
        y += complex(c[1]) * T1b[:rows0]
    Ta, Tb, Tc = T0b, T1b, T2b
    tk = time.time()
    for k in range(2, M):
        L._launch(Ta, Tb, Tc, None, D0, rows0, Kc, 2.0 / w0, 0j, 0, 0)
        for y, c in zip(Ys, coefs):
            y.add_(Tc[:rows0], alpha=complex(c[k]))
        Ta, Tb, Tc = Tb, Tc, Ta
        if k % 50000 == 0:
            log("  series term %d/%d (%.0f s)" % (k, M, time.time() - tk))
    Cf = [(X[:rows0].conj().T @ y).cpu().numpy() for y in Ys]
    lastc = [float(abs(c[-1])) for c in coefs]
    meta = [{k: v for k, v in s.items() if k not in ("info",)} for s in specs]
    for s, m_ in zip(specs, meta):
        if "info" in s:
            m_["packet"] = {k: s["info"][k] for k in ("vg0", "edge_weight", "dropped_weight", "n_om_used", "lo", "hi")}
    J = {"script": "pt3_stage2_held_audit.py", "args": vars(a), "tag": a.tag, "record": {k: H[k] for k in ("label", "npz_md5", "json_md5", "T_rec", "E_rec", "g")},
         "T_H": TH, "omega_line": om_line, "nsites": n, "R_col": Rcol, "certificate": cert, "stats": dict(L.STATS), "monitor": mon, "M": M, "w0": w0,
         "ws_ladder": wsl, "last_coefficients": lastc, "occupied_at": ("f_ws(h(T_rec)): the held Killing vacuum baseline" if a.baseline_vacuum else "f_ws(h(0)): start (b)"), "columns": meta, "norm_defect_max": float(np.max(np.abs(np.real(np.diag(G)) - 1.0))),
         "inventory_e_over_TH": [float(x) for x in e4[inv] / TH], "seconds": time.time() - t_start, "own_md5": K3.own_md5s(), "engine_md5": L.engine_md5s()}
    stem = os.path.join(a.out, a.tag)
    K3.dump_json(J, stem + ".json")
    np.savez(stem + "_blocks.npz", G=G, Cf_ws=Cf[0], Cf_ws2=Cf[1], Cf_ws4=Cf[2])
    log("wrote %s.json and _blocks.npz (%.0f s); norm defect max %.2e" % (stem, time.time() - t_start, J["norm_defect_max"]))


if __name__ == "__main__":
    main()
