#!/usr/bin/env python
"""pt_testB_run_v1.py (busi, inbox #428, Test B: reviewer B's state-resolved outgoing-packet transfer audit, ReviewB_pc6_v1.md Round 4).

One massless k = 1 light chain (page_ops_v6.chain_sparse, the paper's channel operator h[N, v]) on a fully specified smooth prescribed
collapse with a regular interior and no thermal inner boundary, at r_s/a = 32, 64, 128 (the same background in units of r_s).

The background (BUILDER CHOICES, stated): N = 1 (the paper's marginally bound dust control, eq. star-dust-control); the flow is the
paper's declared exterior-plus-ball flow (page_ops_v6.flow_dense: v = -(r_s/r)^(1/2) outside the ball, -(r_s/R_b^3)^(1/2) r inside)
back-averaged by the paper's kernel (page_ops_v6.FlowAverager, ell = r_s/16 at every size, so the ladder holds the background fixed);
the ball edge R_b(t): static at R_b0 = 2 r_s for t <= 0 (its Killing ground state is the initial state, no trapping: max |v| = 0.71),
then the dust law R_b^(3/2) = R_b0^(3/2) - (3/2) r_s^(1/2) tau(t) on a dust clock tau with d tau/dt a C^2 quintic ramp from 0 to 1 over
0.25 r_s, decelerated smoothly onto the regular static core R_bf (C^infinity: R_b^(3/2) = R_bf^(3/2) + Delta softplus(x/Delta),
x the dust law's R_b^(3/2) - R_bf^(3/2), Delta = R_bf^(3/2)/2, the deceleration's time constant 0.118 r_s; --delta-factor 2 the
slow-deceleration control, 0.47 r_s); completion A R_bf = 0.5 r_s (inner root 0.354 r_s), completion B
R_bf = 0.3 r_s (the interior-completion control). The exterior beyond R_b0 + 8 ell never changes and the charge is constant, so the
late exterior is the constant-charge Schwarzschild exterior of the Hawking comparator.

Three readings of the same outgoing packets psi_i (Killing frequency omega, both bands, centre crossing the collar R_col = 16 r_s at
t_col), C^out_ij = <chi_i|C_0|chi_j>, chi_i = U(T_obs_i, 0)^dagger psi_i:
 (1) native: C_0 = f_ws(h(0)), the smoothed negative spectral occupation of the initial static star (ws = 1/(80 r_s), ws/2 the
     control), the evolved state never refilled;
 (2) reset: C(t_d) -> f_ws(h(t_d)) at t_d = 8 and 20 r_s (the disputed closure, a control only): C^reset_ij = <chi_i(t_d)|f(h(t_d))|chi_j(t_d)>
     for the packets observed after t_d;
 (3) continuum QFT: the late-time stationary Hawking prediction of the Unruh state on the constant-charge exterior,
     C^H_ij = sum_omega conj(g_i) g_j e^{i omega (T_i - T_j)} Gamma_1(omega r_s) f_H(omega) within a band (holes 1 - ...), T_H = kappa/2 pi
     with kappa the averaged flow's surface slope (1/(2 r_s) to O(ell^2/r_s^2)); beside it the adiabatic ray-map reading with the
     continuum outgoing/ingoing null rays of the same averaged flow, kappa(t_col) = -d ln(dt_in/dt_col)/dt_col, where resolvable.
Errors printed: the ws ladder (projector), the Chebyshev tail and the norm defect (propagation), the edge weight of the active rows
(outer return), the CF4 step (--dt-collapse halved in a control run)."""
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
import page_ops_v6 as OPS  # noqa: E402

log = L.log
FRONT = 1.08


def softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


class Background:
    """the prescribed collapse in lattice units: r_s, ell = r_s/16, R_b(t), the averaged flow at the chain sites."""

    def __init__(self, rs, n, Rb0_rs=2.0, Rbf_rs=0.5, ramp_rs=0.25, ell_rs=1.0 / 16.0, delta_factor=0.5):
        self.rs, self.n = float(rs), int(n)
        self.Rb0, self.Rbf = Rb0_rs * rs, Rbf_rs * rs
        self.ramp = ramp_rs * rs
        self.ell = ell_rs * rs
        self.Delta = delta_factor * self.Rbf ** 1.5          # the deceleration's scale: time constant Delta/(1.5 r_s^(1/2))
        R = np.arange(1, self.n + 1, dtype=float)
        self.R = R
        self.n_in = int(min(self.n, math.ceil(self.Rb0 + 12 * self.ell)))
        self.avg_in = OPS.FlowAverager(R[: self.n_in], self.ell, "averaged")
        # the static exterior (never changes): the averaged vacuum flow on all sites (the ball's edge inside R_b0 does not reach it)
        self.v_static = OPS.FlowAverager(R, self.ell, "averaged")(OPS.flow_dense(self.rs, self.Rb0))
        x0 = self.Rb0 ** 1.5 - self.Rbf ** 1.5
        # the clock at which the background is static to rounding: softplus(x/Delta) < 1e-14 Delta
        self.t_static = self.dust_clock_inverse((x0 + 32.0 * self.Delta) / (1.5 * math.sqrt(self.rs)))
        self.cache = {}

    def dust_clock(self, t):
        if t <= 0:
            return 0.0
        r = self.ramp
        if t >= r:
            return t - r / 2.0
        s = t / r
        # integral of the quintic ramp 10 s^3 - 15 s^4 + 6 s^5
        return r * (2.5 * s ** 4 - 3.0 * s ** 5 + s ** 6)

    def dust_clock_inverse(self, tau):
        return tau + self.ramp / 2.0

    def Rb(self, t):
        x = self.Rb0 ** 1.5 - self.Rbf ** 1.5 - 1.5 * math.sqrt(self.rs) * self.dust_clock(t)
        return float((self.Rbf ** 1.5 + self.Delta * softplus(np.array(x / self.Delta))) ** (2.0 / 3.0))

    def v_sites(self, t):
        if t >= self.t_static:
            t = self.t_static
        key = round(t, 9)
        if key in self.cache:
            return self.cache[key]
        v = self.v_static.copy()
        v[: self.n_in] = self.avg_in(OPS.flow_dense(self.rs, self.Rb(t)))
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[key] = v
        return v

    def roots(self, t):
        v = self.v_sites(t)
        r_t, r_in = OPS.trapping_zeros(v[: self.n_in + 200], self.R[: self.n_in + 200])
        return r_t, r_in


def kappa_at_root(bg, t):
    v = bg.v_sites(t)
    r_t, _ = bg.roots(t)
    if r_t is None:
        return None, None
    i = int(math.floor(r_t)) - 1
    sl = slice(max(i - 3, 0), i + 4)
    p = np.polyfit(bg.R[sl] - r_t, 1.0 + v[sl], 3)
    return r_t, float(p[-2])


def ray_map(bg, t_cols, R_col, fine=8):
    """the continuum null rays of the averaged flow (site values, linear in R, the flow's time dependence by bg.v_sites): the outgoing
    ray through (R_col, t_col) traced back to the centre, then the ingoing ray traced back out to R_col: t_in(t_col)."""
    from scipy.integrate import solve_ivp
    Rgrid = bg.R

    def vint(r, t):
        v = bg.v_sites(t)
        return float(np.interp(r, Rgrid, v, left=0.0))
    out = []
    for tc in t_cols:
        def ev1(s_, y):
            return y[0] - 0.5
        ev1.terminal = True
        s1 = solve_ivp(lambda s_, y: [-(1.0 + vint(y[0], tc - s_))], (0.0, tc + 4 * R_col + 40 * bg.rs), [R_col], events=ev1,
                       rtol=1e-12, atol=1e-12, max_step=bg.rs / fine)
        if not s1.t_events[0].size:
            out.append(None); continue
        t0 = tc - float(s1.t_events[0][0])
        def ev2(s_, y):
            return y[0] - R_col
        ev2.terminal = True
        s2 = solve_ivp(lambda s_, y: [-(-1.0 + vint(y[0], t0 - s_))], (0.0, 4 * R_col + 4 * bg.rs), [0.5], events=ev2,
                       rtol=1e-12, atol=1e-12, max_step=bg.rs / fine)
        if not s2.t_events[0].size:
            out.append(None); continue
        out.append(t0 - float(s2.t_events[0][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, required=True, help="r_s/a")
    ap.add_argument("--Rbf-rs", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--delta-factor", type=float, default=0.5, help="Delta/R_bf^(3/2): the deceleration onto the core (0.5: time constant 0.118 r_s)")
    ap.add_argument("--omegas-rs", default="-1,-0.5,-0.25,0.25,0.5,1", help="omega r_s; sign = band")
    ap.add_argument("--tcol-rs", default="5,10,15,20,25,30,40,50,60,80,100,125,150")
    ap.add_argument("--td-rs", default="8,20")
    ap.add_argument("--R-col-rs", type=float, default=16.0)
    ap.add_argument("--dt-collapse-rs", type=float, default=1.0 / 64.0)
    ap.add_argument("--held-chunk-rs", type=float, default=16.0)
    ap.add_argument("--ws-rs", type=float, default=1.0 / 80.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-raymap", action="store_true")
    a = ap.parse_args()
    if a.device.startswith("cuda") and a.device not in ("cuda:1", "cuda:2"):
        raise SystemExit("only cuda:1 or cuda:2")
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    rs = a.rs; kap = a.kappa
    Rcol = a.R_col_rs * rs
    oms = [float(x) / rs for x in a.omegas_rs.split(",")]
    tcols = [float(x) * rs for x in a.tcol_rs.split(",")]
    tds = [float(x) * rs for x in a.td_rs.split(",")]
    # packets: one x0 per band, every collar clock
    vac = lambda R: -np.sqrt(rs / np.asarray(R, float))
    specs = []
    for om in oms:
        sig = abs(om) / 6.0
        sd = 1.0 / (2 * sig)
        x0 = float(round(max(Rcol + 7.4 * sd, 3 * kap / abs(om) + 7.4 * sd)))
        trav = L.retarded_travel(vac, Rcol, x0)
        for tc in tcols:
            specs.append({"omega": om, "omega_rs": om * rs, "sig_om": sig, "sd_t": sd, "x0": x0, "t_col": tc, "travel": trav, "T_obs": tc + trav,
                          "hi_est": x0 + 7.4 * sd * 1.05})
    t_max = max(s["T_obs"] for s in specs)
    front0 = max(s["hi_est"] + FRONT * s["T_obs"] for s in specs)
    n = int(math.ceil((front0 + 2048) / 1024.0)) * 1024
    bg = Background(rs, n, Rbf_rs=a.Rbf_rs, delta_factor=a.delta_factor)
    r_t_f, kap_f = kappa_at_root(bg, bg.t_static + 1.0)
    TH = kap_f / (2 * math.pi)
    log("Test B %s: r_s/a %g, R_bf %g r_s, chain %d sites (%.1f r_s), %d packets, t_max %.1f r_s; static from t %.3f r_s; final root %.4f r_s, "
        "kappa r_s %.6f (vacuum 0.5), T_H r_s %.6f" % (a.tag, rs, a.Rbf_rs, n, n / rs, len(specs), t_max / rs, bg.t_static / rs, r_t_f / rs, kap_f * rs, TH * rs))
    # the formation clock: the first root
    tf = None
    for tt in np.arange(0.0, bg.t_static, rs / 256.0):
        if bg.roots(tt)[0] is not None:
            tf = tt; break
    log("first root at t %.4f r_s; R_b(t_static) %.4f r_s" % ((tf or float("nan")) / rs, bg.Rb(bg.t_static) / rs))
    ch = L.Chain(n, kap, a.device)
    order = sorted(range(len(specs)), key=lambda j: -specs[j]["T_obs"])
    specs = [specs[j] for j in order]
    K = len(specs)
    PR = L.Prop(ch.rows, K, a.device)
    X = torch.zeros((ch.rows, K), dtype=L.CD, device=a.device)
    ones = torch.ones(n, dtype=L.RD, device=a.device)
    vlate = bg.v_sites(bg.t_static)
    for j, s in enumerate(specs):
        vt = bg.v_sites(min(s["T_obs"], bg.t_static))
        psi, info = L.wkb_packet(lambda R: np.ones_like(np.asarray(R, float)), lambda R, vt=vt: vt[np.asarray(R, int) - 1], kap, s["omega"], s["sig_om"], s["x0"], n)
        s["info"] = info; s["hi"] = info["hi"]
        X[:, j] = torch.as_tensor(psi, dtype=L.CD, device=a.device)
        if j % 13 == 0:
            e1, esd = L.packet_energy_check(ch, np.ones(n), vt, psi)
            s["energy_check"] = {"mean_over_omega": e1 / s["omega"], "sd_over_sig": esd / s["sig_om"]}
            log("  packet %d omega r_s %+g t_col %g r_s: <h>/omega %.5f sd/sig %.4f edge %.1e dropped %.1e" % (j, s["omega_rs"], s["t_col"] / rs, e1 / s["omega"], esd / s["sig_om"],
                info["edge_weight"], info["dropped_weight"]))
    psis0 = X.clone()
    Ka = 0
    pending = list(range(K))

    def front_rows(t):
        if Ka == 0:
            return 0
        f = max(specs[j]["hi"] + FRONT * (specs[j]["T_obs"] - t) for j in range(Ka)) + 2048
        return 2 * int(min(n, math.ceil(f)))
    # the clock grid
    grid = set(s["T_obs"] for s in specs) | set(tds) | {0.0, bg.t_static}
    pts = sorted(x for x in grid if x >= bg.t_static)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        m = int(math.ceil((p1 - p0) / (a.held_chunk_rs * rs)))
        for i in range(1, m):
            grid.add(p0 + i * (p1 - p0) / m)
    t = 0.0
    while t < bg.t_static:
        grid.add(t); t += a.dt_collapse_rs * rs
    grid = sorted((x for x in grid if x <= t_max + 1e-9), reverse=True)
    c1, c2 = OPS.CF4_C; a1, a2 = OPS.CF4_ALPHA
    snaps = {}
    tick = time.time()

    def add_due(tt):
        nonlocal Ka
        while pending and specs[pending[0]]["T_obs"] >= tt - 1e-9:
            pending.pop(0); Ka += 1
    # the columns not yet launched must be zero during the sweep: zero them and restore at launch
    X[:, :] = 0
    def launch(tt):
        nonlocal Ka
        k0 = Ka
        add_due(tt)
        if Ka > k0:
            X[:, k0:Ka] = psis0[:, k0:Ka]
    launch(grid[0])
    nst = len(grid) - 1
    for i in range(nst):
        t1, t0 = grid[i], grid[i + 1]
        D = t1 - t0
        rows = front_rows(t0)
        if t0 >= bg.t_static - 1e-9:
            Dh = ch.diags(ones, torch.as_tensor(vlate, dtype=L.RD, device=a.device))
            PR.expm(Dh, X, -D, rows, Ka)
        else:
            va = torch.as_tensor(bg.v_sites(t0 + c1 * D), dtype=L.RD, device=a.device)
            vb = torch.as_tensor(bg.v_sites(t0 + c2 * D), dtype=L.RD, device=a.device)
            DA = ch.diags(ones, 2 * (a1 * va + a2 * vb)); DB = ch.diags(ones, 2 * (a2 * va + a1 * vb))
            PR.expm(DB, X, -0.5 * D, rows, Ka)
            PR.expm(DA, X, -0.5 * D, rows, Ka)
        launch(t0)
        for td in tds:
            if abs(t0 - td) < 1e-9:
                snaps[td] = (X[:, :Ka].clone(), Ka, front_rows(t0))
        if i % 100 == 0 or i == nst - 1:
            r = front_rows(t0)
            edge = float((X[max(0, r - 1024): r, :Ka].abs() ** 2).sum(dim=0).max()) if Ka else 0.0
            el = time.time() - tick
            log("  step %d/%d t %.3f r_s cols %d rows %d terms %d tail %.1e edge %.1e %.0f s eta %.0f s" % (
                i + 1, nst, t0 / rs, Ka, r, L.STATS["terms"], L.STATS["tail_acc"], edge, el, el / (i + 1) * (nst - i - 1)))
    assert Ka == K
    # ---------------------------------------------------------------- the covariances
    def cov(Xb, Kb, rows, vfield, ws):
        Dq = ch.diags(ones, torch.as_tensor(vfield, dtype=L.RD, device=a.device))
        w = L.Chain.bound(Dq[:, :rows])
        Mq = L.step_order(w, ws)
        c = L.step_coeffs(w, ws, Mq)
        PRb = L.Prop(ch.rows, Kb, a.device)
        Xf = torch.zeros((ch.rows, Kb), dtype=L.CD, device=a.device); Xf[:, :Kb] = Xb[:, :Kb]
        Y = PRb.series(Dq, Xf, c, rows, Kb, w)
        C = (Xf[:rows, :Kb].conj().T @ Y).cpu().numpy()
        del PRb, Xf, Y
        return C, Mq, w
    ws = a.ws_rs / rs
    rows0 = front_rows(0.0)
    v0 = bg.v_sites(0.0)
    C0, M0, w0 = cov(X, K, rows0, v0, ws)
    C0h, M0h, _ = cov(X, K, rows0, v0, ws / 2)
    log("native covariance: M %d (ws), %d (ws/2), w %.4f; Hermiticity defect %.1e" % (M0, M0h, w0, float(np.max(np.abs(C0 - C0.conj().T)))))
    norm0 = (X[:rows0, :K].abs() ** 2).sum(dim=0).cpu().numpy()
    reset = {}
    for td in tds:
        if td not in snaps:
            continue
        Xd, Kd, rd = snaps[td]
        Xfull = torch.zeros((ch.rows, Kd), dtype=L.CD, device=a.device); Xfull[:, :Kd] = Xd
        Cr, Mr, wr = cov(Xfull, Kd, rd, bg.v_sites(td), ws)
        reset[td] = {"C": Cr, "K": Kd}
        log("reset at t_d %.1f r_s: %d packets after it, M %d" % (td / rs, Kd, Mr))
    # ---------------------------------------------------------------- the QFT comparator
    omg = {}
    rows_out = []
    for j, s in enumerate(specs):
        info = s["info"]
        n_nat = float(np.real(C0[j, j])); n_nat_h = float(np.real(C0h[j, j]))
        part = n_nat if s["omega"] > 0 else 1.0 - n_nat
        nH = L.hawking_n(info, kap, rs, TH)
        Tf = L.fit_T(info, kap, rs, part, TH)
        row = {"omega_rs": s["omega_rs"], "t_col_rs": s["t_col"] / rs, "T_obs_rs": s["T_obs"] / rs, "n_native": n_nat, "n_native_ws_half": n_nat_h,
               "particle_or_hole_native": part, "n_QFT_stationary": nH, "ratio_native_QFT": part / nH if nH > 0 else None,
               "T_fit_over_TH": (Tf / TH) if Tf else None, "norm_defect": float(norm0[j] - 1.0), "Gamma_w": float(np.sum(info["rho"] * L.gamma_k(kap, np.abs(info["oms"]) * rs))),
               "energy_check": s.get("energy_check"), "packet": {k: info[k] for k in ("edge_weight", "dropped_weight", "vg0")}}
        for td, R_ in reset.items():
            if j < R_["K"]:
                nr = float(np.real(R_["C"][j, j]))
                row["n_reset_td%g" % (td / rs)] = nr
                row["particle_or_hole_reset_td%g" % (td / rs)] = nr if s["omega"] > 0 else 1.0 - nr
        rows_out.append(row)
    # within-band off-diagonals: native against the stationary QFT kernel and the reset
    offd = []
    for b in sorted(set(s["omega"] for s in specs)):
        idx = [j for j, s in enumerate(specs) if s["omega"] == b]
        info = specs[idx[0]]["info"]
        om = info["oms"]; rho = info["rho"]
        G = L.gamma_k(kap, np.abs(om) * rs); f = L.fermi(np.abs(om), TH)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                p, q = idx[ii], idx[jj]
                dT = specs[p]["T_obs"] - specs[q]["T_obs"]
                kern = np.sum(rho * G * f * np.exp(1j * om * dT))
                cq = kern if b > 0 else -kern
                offd.append({"omega_rs": b * rs, "t_col_i": specs[p]["t_col"] / rs, "t_col_j": specs[q]["t_col"] / rs, "C_native": complex(C0[p, q]),
                             "abs_native": float(abs(C0[p, q])), "abs_QFT": float(abs(cq)),
                             "abs_reset_td%g" % (tds[0] / rs): (float(abs(reset[tds[0]]["C"][p, q])) if tds[0] in reset and max(p, q) < reset[tds[0]]["K"] else None)})
    # cross-band blocks (stationary prediction: zero)
    bands = np.array([s["omega"] for s in specs])
    cross = float(np.max(np.abs(C0[np.ix_(bands > 0, bands < 0)]))) if (bands > 0).any() and (bands < 0).any() else None
    # the ray map (adiabatic kappa(t_col)) where resolvable
    rm = None
    if not a.no_raymap:
        tcg = np.array(sorted(set(s["t_col"] for s in specs)))
        dens = []
        for tc in tcg:
            h_ = 0.05 * rs
            tin = ray_map(bg, [tc - h_, tc, tc + h_], Rcol)
            if None in tin:
                dens.append(None); continue
            d1 = (tin[2] - tin[0]) / (2 * h_)
            d2 = (tin[2] - 2 * tin[1] + tin[0]) / (h_ * h_)
            ok = d1 > 1e-6          # below that the ray's distance from the horizon is under rounding: kappa unresolved (the stationary value holds)
            dens.append({"t_col_rs": tc / rs, "t_in_rs": tin[1] / rs, "dtin_dtcol": d1, "kappa_rs": (-d2 / d1) * rs if ok else None, "resolved": bool(ok)})
        rm = dens
        log("ray map: %s" % "; ".join(("t_col %.0f: dt_in/dt_col %.3e kappa r_s %s" % (d["t_col_rs"], d["dtin_dtcol"], ("%.4f" % d["kappa_rs"]) if d["kappa_rs"] else "-")) if d else "-" for d in dens))
    rows_out.sort(key=lambda r: (r["omega_rs"], r["t_col_rs"]))
    log("table: native particle|hole occupation, the reset at t_d, the stationary QFT, ratio, T_fit/T_H")
    for r in rows_out:
        log("  w r_s %+5.2f t_col %5.0f r_s: native %.6e (ws/2 %.6e)  reset8 %s reset20 %s  QFT %.6e  ratio %s  T/T_H %s" % (
            r["omega_rs"], r["t_col_rs"], r["particle_or_hole_native"], r["n_native_ws_half"] if r["omega_rs"] > 0 else 1 - r["n_native_ws_half"],
            ("%.3e" % r["particle_or_hole_reset_td8"]) if "particle_or_hole_reset_td8" in r else "-", ("%.3e" % r["particle_or_hole_reset_td20"]) if "particle_or_hole_reset_td20" in r else "-",
            r["n_QFT_stationary"], ("%.4f" % r["ratio_native_QFT"]) if r["ratio_native_QFT"] is not None else "-", ("%.4f" % r["T_fit_over_TH"]) if r["T_fit_over_TH"] else "-"))
    J = {"script": "pt_testB_run_v1.py", "args": vars(a), "tag": a.tag, "rs": rs, "nsites": n, "t_static_rs": bg.t_static / rs, "first_root_rs": (tf / rs) if tf else None,
         "final_root_rs": r_t_f / rs, "kappa_rs": kap_f * rs, "T_H_rs": TH * rs, "stats": dict(L.STATS), "engine_md5s": L.engine_md5s(), "rows": rows_out,
         "offdiag": offd, "cross_band_max_abs_native": cross, "raymap": rm, "moments": {"M_ws": M0, "M_ws_half": M0h, "w0": w0}, "seconds": time.time() - t_start}
    stem = os.path.join(a.out, a.tag)
    json.dump(J, open(stem + ".json", "w"), indent=1, default=lambda o: [o.real, o.imag] if isinstance(o, complex) else float(o))
    np.savez(stem + "_cov.npz", C_native=C0, C_native_ws_half=C0h, **{"C_reset_td%g" % (td / rs): R_["C"] for td, R_ in reset.items()},
             omega_rs=np.array([s["omega_rs"] for s in specs]), t_col_rs=np.array([s["t_col"] / rs for s in specs]), norm0=norm0)
    log("wrote %s.json (%.0f s)" % (stem, time.time() - t_start))


if __name__ == "__main__":
    main()
