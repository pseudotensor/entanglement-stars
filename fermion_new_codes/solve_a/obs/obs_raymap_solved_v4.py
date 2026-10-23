#!/usr/bin/env python3
"""obs_raymap_solved_v4.py (builder analyst; engine spec 19.2: the production's consumers propagate on the applied (kernel-averaged) site
fields: version 4 = version 3 with --fields-key stored | applied (default applied): applied reads N_applied, v_applied, the fields of the
h_m that propagated the run's state (engine_collapse_v10 and later; the v17 production's kernel averages), stored reads N, v (the solve from
the accepted sources, version 3's choice); the record line and the info name the key. Nothing else changes. Version 3's docstring follows.
obs_raymap_solved_v3.py (builder analyst; B7 of SPEC_final4b_observables.md sec. 11.6: J3' per sec. 11.3, after the reviewer B red team
redteam_JPL4 of 2026-10-15 (its defects 5 and 6); version 3 of obs_raymap_solved_v2.py).

Version 3: (a) the chain reaches max(128 r_s, x_0 + 4 sigma_x (spec) + 8 r_s, x_0 + 5 s_x + 8 r_s) (the red team: the spec's 128 r_s leaves
the omega/kappa = 1/2 packet 1.29 sigma_x of room), the Gaussian tail beyond the chain and the packet's temporal width s_t = s_x/v_g printed
(the T_surf series are overlapping temporal averages of width s_t, labelled); (b) the frozen extension's Duhamel estimate per column,
B = int_{T_end}^{T_obs} ((t - T_end)/dt_rate) ||D phi(t)|| dt with D = h(T_end) - h(T_end - dt_rate) (the last stored interval's field
change extrapolated linearly; dt_rate = --drift-window-over-rs, 4 nominal r_s), sampled every --duhamel-dt-over-rs in the frozen region:
|dn_w| <= 2B + B^2, and d theta = theta^2 (dn/n)/(2 pi r (1 - n/Gamma)) printed beside the end-norm check (the red team: one overlap at
T_end does not bound the occupation error); the quasi-frozen check's relative difference also as d theta; (c) the first-passage share's
time series kept (on the fixture the back-traced packet ends at R > R_t on k < 0 with share 0.99995: the transmitted part returns through
the core's lattice-scale conversion, so the share has no first-passage plateau when the return overlaps the reflection), its minimum and
maximum printed and no plateau read unless the series shows one; (d) theta (fit) labelled the reading, theta (T_conv) the narrow-band
comparator. Version 2's docstring follows.

obs_raymap_solved_v2.py (builder analyst; B7 of SPEC_final4b_observables.md sec. 11.6: J3' per sec. 11.3; version 2 of
obs_raymap_solved_v1.py).

(1) The chain: the stored history on the engine's sites (to R_max = the engine's last site, 32 r_s at the primary), continued beyond by the
analytic exterior N = 1, v = -sqrt(2 g E_k/R) at each slice's solved E_k, to R_end = max(--rend-over-rs r_s, the largest x_0 + 5 s_x + 8 r_s)
(the packet must fit: the spec's 128 r_s holds the omega/kappa = 1/2 packet's centre but not its tail); the junction at the engine's last site
printed per slice (|N - 1|, |v + sqrt(2 g E_k/R)|). The packets: sigma_omega = omega/6 at omega/kappa in {1/2, 1, 2}; the spec's sigma_x =
v_g/sigma_omega (24, 12, 6 r_s) sets x_0 = max(8 r_s, r_s + 4 sigma_x); the packet's density standard deviation is s_x = v_g/(2 sigma_omega)
(printed beside). Here r_s is the late slice's 2 g E (the geometry); times are printed in units of the engine's nominal r_s (its T/r_s).
(2) The times: T_surf in {12, 16, 20, 24, 28} (nominal r_s); T_obs = T_surf + the symbol's travel time to x_0 (BUILDER, labelled: the axial
symbol omega = v sin k cos k + N sqrt(m^2 + sin^2 k) on the slice nearest T_surf, the ray of the packet's central omega integrated
dt = dR/v_g from the radius where its momentum is k_* = 0.3 (the near-surface window's edge of sec. 8.1, below which the continuum symbol
holds to 6 percent) out to x_0; the angular term is left out of the ray because the j = 1/2 ray below its barrier has no classical
continuation). Beyond the stored history the last slice is frozen; the control: the back-propagated packet's norm inside R_t(T_end) + 5 ell
at T_end, required below 1e-3 (printed per row, flagged). The quasi-frozen check (sec. 3.5): at T_surf = --hold-tsurf (24) the n_w with every
slice from T_hold = T_surf on replaced by the slice at T_hold (the frozen hold) against the stored propagation, the relative difference
printed against 1e-3. (3) The initial state P_0: the probe's sea at t = 0, the Chebyshev erf step at width w_s = kappa_T/40 (w_s/2 the
control) of the probe's operator on the first slice; n_w = <phi|f|phi> and ||f phi||^2 with phi = U(T_obs)^dagger w, the backward
propagation by the engine's own step (each slice's operator held over [T_k, T_k + tau)) evaluated by Chebyshev exponentials, the five T_surf
columns propagated together. BUILDER, labelled: the dust orbitals belong to the engine's field (mass m = the engine's, 25.6 kappa at the
primary), not to the probe (m_p = 0 or kappa/4), and the stored history holds N, v only; P_0 is the probe's sea. (4) Gamma at each T_surf by
the stationary scattering of the frozen slice nearest T_surf (obs_plateau_v5's dense import, site order 5; R_cut = R_t/4 and R_t/2, the
cut check; the outer saturation at 64 r_s), averaged over the packet's spectral measure; beside it the stationary occupation of the same
scattering (n_stat = the packet average of n_H) and the first-passage share G2 (1 - the reflected share at R > R_t + 2 on the ingoing branch at
its plateau, obs_raymap_pg_v2's plateau_window, the packet back-propagated on the frozen slice at T_surf on a chain long enough for the
reflected part). theta = T/(kappa_T/2 pi) with kappa_T of the record nearest T_surf, T from the transmission-weighted fit
n_w = sum rho_w Gamma f(omega; T) (sec. 8.1) and T_conv = omega/ln(Gamma_w/n_w - 1) (sec. 10.2) both printed. (5) Per T_surf the record's
R_t, r_in, R_sh, the late-surface flag, M8's pairs' energy inside R_t and the pocket count; per (m_p, omega) the log slope of n_w per nominal
r_s. (6) One sign per run (--sign; the kappa_j = -1 check is a second run). Args: --record --fields | --synthetic-npz/--synthetic-json (a
history in the engine's layout); --g --j --sign --mp-over-kappa --omega-over-kappa --tsurf-list --hold-tsurf --rend-over-rs --ws-over-kappa
--no-fp --gpu --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc, jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as V4  # noqa: E402
import obs_greybody_radial_v1 as GR  # noqa: E402
import obs_plateau_v4 as P4  # noqa: E402
import obs_plateau_v5 as P5  # noqa: E402
import obs_raymap_pg_v2 as RPG  # noqa: E402

BW = 5


class Diags2:
    """the probe's operator as its component diagonals; apply() takes a vector or a matrix of columns (numpy, or torch on a GPU)."""

    def __init__(self, N, v, m, y, torch=None, dev=None):
        ab = GR.band_matrix_y(np.asarray(v, float), np.asarray(N, float), m, np.asarray(y, float))
        dim = ab.shape[1]
        self.n, self.torch, self.dev = dim, torch, dev
        self.d = []
        g = np.zeros(dim)
        for off in range(-BW, BW + 1):
            dg = ab[BW - off, off:] if off >= 0 else ab[BW - off, :dim + off]
            if np.max(np.abs(dg)) == 0:
                continue
            if off >= 0:
                g[:dim - off] += np.abs(dg)
            else:
                g[-off:] += np.abs(dg)
            self.d.append((off, torch.as_tensor(dg, device=dev) if torch else dg))
        self.bound = float(g.max()) * 1.0001

    def apply(self, X):
        Y = self.torch.zeros_like(X) if self.torch else np.zeros_like(X)
        n = self.n
        two = X.ndim == 2
        for off, D in self.d:
            Dc = D[:, None] if two else D
            if off >= 0:
                Y[:n - off] += Dc * X[off:]
            else:
                Y[-off:] += Dc * X[:n + off]
        return Y


def cheb_expm2(B, X, tau, tol=1e-15):
    """exp(-i tau h) X for a vector or a matrix of columns (tau < 0: the backward step)."""
    if tau == 0:
        return X
    r = B.bound
    a = r * tau
    K = int(abs(a) + 12)
    while abs(jv(K, abs(a))) > tol or K < abs(a) + 4:
        K += 4
    T0, T1 = X, B.apply(X) / r
    Y = jv(0, a) * T0 + 2 * (-1j) * jv(1, a) * T1
    for k in range(2, K + 1):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + 2 * ((-1j) ** k) * jv(k, a) * T2
        T0, T1 = T1, T2
    return Y


def cheb_step_coeffs(r, ws, nmax=400000):
    M = 1
    while M < 8 * r / ws:
        M *= 2
    M = min(M, nmax)
    th = (np.arange(M) + 0.5) * math.pi / M
    fx = 0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws))
    from scipy.fft import dct
    c = dct(fx, type=2) / M
    c[0] /= 2.0
    keep = np.nonzero(np.abs(c) > 1e-13)[0]
    K = int(keep[-1]) + 1 if len(keep) else 1
    return c[:K]


def cheb_step_occupation2(B, X, ws):
    """per column: <X| f(h) |X> and ||f(h) X||^2, f(E) = erfc(E/(sqrt 2 ws))/2 (Chebyshev on [-r, r])."""
    c = cheb_step_coeffs(B.bound, ws)
    r = B.bound
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + (c[1] * T1 if len(c) > 1 else 0 * T0)
    for k in range(2, len(c)):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    if B.torch:
        occ = (X.conj() * Y).sum(dim=0).real.cpu().numpy(); occ2 = (Y.conj() * Y).sum(dim=0).real.cpu().numpy()
    else:
        occ = np.real(np.sum(X.conj() * Y, axis=0)); occ2 = np.real(np.sum(Y.conj() * Y, axis=0))
    return np.atleast_1d(occ), np.atleast_1d(occ2), len(c)


def history(record, fields, g, nsites, synthetic=False, fields_key="applied"):
    D = json.load(open(record))
    recs = [r for r in D["records"] if "E" in r]
    Z = np.load(fields, allow_pickle=True)
    Re = np.asarray(Z["R"], float)
    T = np.asarray(Z["T"], float)
    R = np.arange(1, nsites + 1, dtype=float)
    ne = min(len(Re), nsites)
    slices = []
    for k in range(len(T)):
        rec = min(recs, key=lambda q: abs(q["T"] - T[k]))
        E = float(rec["E"])
        N = np.ones(nsites); v = -np.sqrt(2 * g * E / R)
        if fields_key == "applied":
            Nk = np.asarray(Z["N_applied"][k], float); vk = np.asarray(Z["v_applied"][k], float)
        else:
            Nk = np.asarray(Z["N"][k], float); vk = np.asarray(Z["v"][k], float)
        N[:ne] = Nk[:ne]; v[:ne] = vk[:ne]
        jN = abs(Nk[ne - 1] - 1.0); jv_ = abs(vk[ne - 1] + math.sqrt(2 * g * E / Re[ne - 1]))
        slices.append({"T": float(T[k]), "N": N, "v": v, "E": E, "rec": rec, "junction_N": jN, "junction_v": jv_})
    args = D.get("args", {})
    info = {"source": ("synthetic history %s" if synthetic else "collapse record %s") % os.path.basename(record), "fields_key": fields_key, "tau": float(Z["tau"]), "ell": float(Z["ell"]),
            "label": str(Z["label"]), "engine_nsites": int(len(Re)), "rs_nominal": float(args.get("rs", 2 * g * slices[-1]["E"])),
            "junction_max_N": max(s["junction_N"] for s in slices), "junction_max_v": max(s["junction_v"] for s in slices)}
    return R, slices, info


def axial_travel(N, v, R, m, omega, x0, kstar=0.3):
    """the axial symbol's ray at omega from k = kstar near the surface out to x0: (travel time, R_*), or (None, None)."""
    kk = np.linspace(1e-7, math.pi / 2 - 1e-7, 4001)
    Ni = lambda x: float(np.interp(x, R, N)); vi = lambda x: float(np.interp(x, R, v))
    om_k = lambda k, n_, v_: v_ * np.sin(k) * np.cos(k) + n_ * np.sqrt(m * m + np.sin(k) ** 2)
    vg_k = lambda k, n_, v_: v_ * np.cos(2 * k) + n_ * np.sin(k) * np.cos(k) / np.sqrt(m * m + np.sin(k) ** 2)
    xs = np.arange(x0, 1.0, -0.25)
    t, Rstar, kprev = 0.0, None, None
    for i, x in enumerate(xs):
        n_, v_ = Ni(x), vi(x)
        if abs(v_) >= n_:
            break
        om = om_k(kk, n_, v_)
        imax = int(np.argmax(om))
        if om[imax] < omega or om[0] > omega:
            break
        k = brentq(lambda q: om_k(q, n_, v_) - omega, kk[0], kk[imax], xtol=1e-14)
        vg = vg_k(k, n_, v_)
        if vg <= 0:
            break
        if i > 0:
            t += 0.25 / vg
        if k >= kstar:
            Rstar = x
            break
        kprev = k
    return (t, Rstar) if Rstar is not None else (None, kprev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--synthetic", action="store_true", help="the record/fields are a synthetic history in the engine's layout (the control)")
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--j", type=float, default=0.5)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--mp-over-kappa", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--omega-over-kappa", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--tsurf-list", type=float, nargs="+", default=[12.0, 16.0, 20.0, 24.0, 28.0], help="units of the nominal r_s")
    ap.add_argument("--hold-tsurf", type=float, default=24.0)
    ap.add_argument("--rend-over-rs", type=float, default=128.0)
    ap.add_argument("--ws-over-kappa", type=float, default=1.0 / 40)
    ap.add_argument("--kstar", type=float, default=0.3)
    ap.add_argument("--no-fp", action="store_true")
    ap.add_argument("--fp-dt-over-rs", type=float, default=0.5)
    ap.add_argument("--gamma-ngrid", type=int, default=24)
    ap.add_argument("--drift-window-over-rs", type=float, default=4.0, help="the Duhamel estimate's rate window before T_end (nominal r_s)")
    ap.add_argument("--duhamel-dt-over-rs", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fields-key", default="applied", choices=["applied", "stored"], help="engine spec 19.2: the applied (averaged) site fields")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    torch = dev = None
    if a.gpu is not None:
        import torch as _t
        torch, dev = _t, _t.device("cuda:%d" % a.gpu)
    t_start = time.time()
    # a first pass at the engine's size for the geometry, then the chain at the packets' length
    R0, sl0, hinfo = history(a.record, a.fields, a.g, 64, a.synthetic, a.fields_key)
    E_late = sl0[-1]["E"]
    rs = 2 * a.g * E_late
    rsn = hinfo["rs_nominal"]
    kT_late = sl0[-1]["rec"].get("kappa_T") or 1.0 / (2 * rs)
    kappa = kT_late
    kj = a.sign * (a.j + 0.5)
    ell = hinfo["ell"]
    # the packets' geometry (the far exterior at E_late)
    packs = []
    for mk in a.mp_over_kappa:
        m = mk * kappa
        for r_ in a.omega_over_kappa:
            omega = r_ * kappa
            sg = omega / 6.0
            below = 0.0
            if m > 0:
                below = 0.5 * erfc((omega - m) / (math.sqrt(2) * sg))
                if omega - 3 * sg < m * (1 - 1e-9):
                    packs.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "skipped": "the packet's 3 sigma window reaches below the threshold m"})
                    continue
            x0 = 8 * rs
            for _ in range(8):
                vext = -math.sqrt(rs / x0); me = math.sqrt(m * m + (kj / x0) ** 2)
                k, om, a2 = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
                vg0 = abs(float(np.sum(a2 * V4.group_velocity(k, me, vext) * np.gradient(k))))
                sx_lead = vg0 / sg
                x1 = max(8 * rs, rs + 4 * sx_lead)
                if abs(x1 - x0) < 1e-9:
                    break
                x0 = x1
            packs.append({"mp_over_kappa": mk, "omega_over_kappa": r_, "m": m, "omega": omega, "sigma_omega": sg, "x0": x0, "sigma_x_spec": sx_lead,
                          "s_x_density": vg0 / (2 * sg), "vg_far": vg0, "k": k, "om": om, "a2": a2, "me": me, "below_threshold_measure": below})
    Rend = max([a.rend_over_rs * rs] + [max(p["x0"] + 4 * p["sigma_x_spec"], p["x0"] + 5 * p["s_x_density"]) + 8 * rs for p in packs if "x0" in p])
    nsites = int(math.ceil(Rend / 512.0)) * 512
    for p in packs:
        if "x0" in p:
            p["gaussian_tail_beyond_chain"] = 0.5 * erfc((nsites - p["x0"]) / (math.sqrt(2.0) * p["s_x_density"]))
            p["s_t_over_rs"] = p["s_x_density"] / p["vg_far"] / rsn
    R, slices, hinfo = history(a.record, a.fields, a.g, nsites, a.synthetic, a.fields_key)
    y = kj / R
    times = np.array([s_["T"] for s_ in slices])
    T_end = float(times[-1])
    tau = hinfo["tau"]
    print("history %s: %d slices to T_end/r_s %.2f (nominal r_s %g; late 2gE %.4f; kappa_T 4gE %.5f); chain %d sites (%.1f r_s); junction max |N-1| %.2e |v - ext| %.2e" % (
        hinfo["source"], len(slices), T_end / rsn, rsn, rs, kappa * 4 * a.g * E_late, nsites, nsites / rs, hinfo["junction_max_N"], hinfo["junction_max_v"]), flush=True)
    ops = {}

    def op(k, m, held=None):
        kk = held if (held is not None and k >= held) else k
        key = (kk, m)
        if key not in ops:
            if len(ops) > 48:
                ops.clear()
            s_ = slices[kk]
            ops[key] = Diags2(s_["N"], s_["v"], m, y, torch, dev)
        return ops[key]

    def slice_at(t):
        """the slice governing time t (held over [T_k, T_k + tau))."""
        return max(i for i, Ti in enumerate(times) if Ti <= t + 1e-9) if t >= times[0] else 0

    def slice_below(t):
        """the slice governing the interval just below t."""
        c = [i for i, Ti in enumerate(times) if Ti < t - 1e-9]
        return max(c) if c else 0

    # version 3: the drift operator for the frozen extension's Duhamel estimate (the last stored interval's rate, extrapolated linearly)
    k_rate = slice_at(T_end - a.drift_window_over_rs * rsn)
    dt_rate = T_end - float(times[k_rate])
    dN_rate = slices[-1]["N"] - slices[k_rate]["N"]; dv_rate = slices[-1]["v"] - slices[k_rate]["v"]

    def colnorms(Y):
        return ((Y.conj() * Y).real.sum(dim=0).sqrt().cpu().numpy()) if torch else np.sqrt(np.sum(np.abs(Y) ** 2, axis=0))

    def back_propagate(w, Tobs_list, m, held=None, check_end=True):
        """columns for each T_obs (descending insertion): phi_j = U(T_obs_j)^dagger w; returns phi (2n x J) at t = 0, the per-column norm
        inside R_t(T_end) + 5 ell at T_end (None for columns with T_obs <= T_end), and (version 3) the per-column Duhamel estimate
        B_j = int_{T_end}^{T_obs_j} ((t - T_end)/dt_rate) ||D phi_j(t)|| dt with D = h(T_end) - h(T_end - dt_rate), sampled at sub-steps of
        --duhamel-dt-over-rs in the frozen region (None for columns with T_obs <= T_end or when the hold is in use)."""
        order = np.argsort(-np.asarray(Tobs_list))
        cols = []
        t = float(max(Tobs_list))
        X = None
        endnorm = [None] * len(Tobs_list)
        duh = [None] * len(Tobs_list)
        last = {}
        Rt_end = slices[-1]["rec"].get("R_t") or rs
        inner = np.repeat(R < Rt_end + 5 * ell, 2)
        inner_t = torch.as_tensor(inner, device=dev) if torch else inner
        pending = [int(j) for j in order]
        wt = torch.as_tensor(w, device=dev) if torch else w
        Dop = Diags2(dN_rate, dv_rate, m, y, torch, dev) if (check_end and held is None and dt_rate > 0) else None
        dsub = a.duhamel_dt_over_rs * rsn

        def duhamel_sample(tt):
            if Dop is None or X is None or tt < T_end - 1e-9:
                return
            f = colnorms(Dop.apply(X)) * (tt - T_end) / dt_rate
            for c, j in enumerate(cols):
                if Tobs_list[j] <= T_end + 1e-9:
                    continue
                if j in last:
                    t0_, f0_ = last[j]
                    duh[j] = (duh[j] or 0.0) + 0.5 * (f0_ + float(f[c])) * (t0_ - tt)
                else:
                    duh[j] = 0.0
                last[j] = (tt, float(f[c]))
        while True:
            added = False
            while pending and Tobs_list[pending[0]] >= t - 1e-9:
                j = pending.pop(0)
                col = wt.clone() if torch else wt.copy()
                X = col[:, None] if X is None else (torch.cat([X, col[:, None]], dim=1) if torch else np.concatenate([X, col[:, None]], axis=1))
                cols.append(j)
                added = True
            if added:
                duhamel_sample(t)
            if t <= 1e-9:
                break
            # the next event below t: a slice boundary, a pending T_obs, T_end (the frozen-extension check), a Duhamel sub-step
            k = slice_below(t)
            t_lo = float(times[k]) if t > times[0] + 1e-9 else 0.0
            nxt = [t_lo]
            if pending:
                nxt.append(Tobs_list[pending[0]])
            if check_end and t > T_end + 1e-9:
                nxt.append(T_end)
                if Dop is not None:
                    nxt.append(max(T_end, t - dsub))
            t_next = max(x for x in nxt if x < t - 1e-9) if any(x < t - 1e-9 for x in nxt) else 0.0
            if X is not None:
                X = cheb_expm2(op(k, m, held), X, -(t - t_next))
            t = t_next
            if t >= T_end - 1e-9:
                duhamel_sample(t)
            if check_end and abs(t - T_end) < 1e-9 and X is not None:
                nin = ((X.conj() * X).real * inner_t[:, None]).sum(dim=0).cpu().numpy() if torch else np.sum(np.abs(X) ** 2 * inner[:, None], axis=0)
                for c, j in enumerate(cols):
                    if Tobs_list[j] > T_end + 1e-9 and endnorm[j] is None:
                        endnorm[j] = float(nin[c])
            if t <= 1e-9 and not pending:
                break
        return X, cols, endnorm, duh

    rows = []
    for p in packs:
        if "skipped" in p:
            rows.append(p)
            continue
        m, omega = p["m"], p["omega"]
        w = V4.build_packet(nsites, int(round(p["x0"] - 1)), p["k"], p["a2"], p["me"], m=m, a=kj / p["x0"])
        rho_w = p["a2"] * np.gradient(p["k"]); rho_w = rho_w / rho_w.sum()
        series = []
        Tobs = []
        for ts in a.tsurf_list:
            T_surf = ts * rsn
            ks = slice_at(T_surf)
            trav, Rstar = axial_travel(slices[ks]["N"], slices[ks]["v"], R, m, omega, p["x0"], a.kstar)
            if trav is None:
                trav = (p["x0"] - rs) / p["vg_far"]
                Rstar = None
            Tobs.append(T_surf + trav)
            series.append({"T_surf_over_rs": ts, "T_surf": T_surf, "travel": trav, "R_star": Rstar, "T_obs": T_surf + trav,
                           "T_obs_over_rs": (T_surf + trav) / rsn, "frozen_beyond_history": T_surf + trav > T_end + 1e-9, "slice": ks})
        X, cols, endnorm, duh = back_propagate(w, Tobs, m)
        B0 = Diags2(slices[0]["N"], slices[0]["v"], m, y, torch, dev)
        occ, occ2, Kd = cheb_step_occupation2(B0, X, a.ws_over_kappa * kappa)
        occh, _, _ = cheb_step_occupation2(B0, X, 0.5 * a.ws_over_kappa * kappa)
        nrm = ((X.conj() * X).real.sum(dim=0).cpu().numpy()) if torch else np.sum(np.abs(X) ** 2, axis=0)
        for c, j in enumerate(cols):
            s_ = series[j]
            s_.update({"n_w": float(occ[c]), "n_w_fsq": float(occ2[c]), "n_w_ws_half": float(occh[c]), "norm_at_T0": float(nrm[c]), "cheb_degree": Kd,
                       "endnorm_inside_Rt_plus_5ell": endnorm[j], "endnorm_ok": (endnorm[j] is None or endnorm[j] < 1e-3),
                       "duhamel_B": duh[j], "duhamel_dn_bound": (2 * duh[j] + duh[j] ** 2) if duh[j] is not None else None})
        for s_ in series:
            rec = slices[s_["slice"]]["rec"]
            s_.update({"R_t": rec.get("R_t"), "r_in": rec.get("r_in"), "R_sh": rec.get("R_sh"), "late_surface": rec.get("late_surface"), "kappa_T": rec.get("kappa_T"),
                       "pairs_inside_Rt": rec.get("pairs_inside_Rt"), "n_pocket": (sum(rec["n_pocket"].values()) if isinstance(rec.get("n_pocket"), dict) else None),
                       "E": rec.get("E")})
            print("m_p/kappa %g omega/kappa %g T_surf/r_s %g: T_obs/r_s %.2f n_w %.6e (w_s/2 %.6e, ||f phi||^2 %.6e) norm %.10f end-norm %s (%.0f s)" % (
                p["mp_over_kappa"], p["omega_over_kappa"], s_["T_surf_over_rs"], s_["T_obs_over_rs"], s_["n_w"], s_["n_w_ws_half"], s_["n_w_fsq"], s_["norm_at_T0"],
                ("%.2e" % s_["endnorm_inside_Rt_plus_5ell"]) if s_["endnorm_inside_Rt_plus_5ell"] is not None else "-", time.time() - t_start), flush=True)
        # the quasi-frozen check at the hold time
        hold = None
        if a.hold_tsurf in a.tsurf_list:
            j = a.tsurf_list.index(a.hold_tsurf)
            kh = slice_at(series[j]["T_surf"])
            Xh, _, _, _ = back_propagate(w, [series[j]["T_obs"]], m, held=kh, check_end=False)
            oh, _, _ = cheb_step_occupation2(B0, Xh, a.ws_over_kappa * kappa)
            hold = {"T_hold_over_rs": float(times[kh]) / rsn, "n_w_stored": series[j]["n_w"], "n_w_hold": float(oh[0]), "omega_over_kappa": p["omega_over_kappa"],
                    "rel_diff": float(abs(oh[0] - series[j]["n_w"]) / max(abs(series[j]["n_w"]), 1e-300))}
            print("  quasi-frozen check at T_surf/r_s %g: hold from %.2f r_s: n_w %.6e against the stored %.6e (relative %.2e; 1e-3 the bar)" % (
                a.hold_tsurf, hold["T_hold_over_rs"], hold["n_w_hold"], hold["n_w_stored"], hold["rel_diff"]), flush=True)
        # Gamma, n_stat and the first-passage share per T_surf
        om_grid = p["om"][:: max(1, len(p["om"]) // a.gamma_ngrid)]
        for s_ in series:
            ks = s_["slice"]
            fn, sinfo, aux = P5.slice_fields_v5(a.fields, a.record, ks, a.g, "dense", 5)
            Rt = s_["R_t"] or rs
            kT = s_["kappa_T"] or kappa
            gam = {}
            for lab, Rcut in (("R_t/4", Rt / 4), ("R_t/2", Rt / 2)):
                RL = 64 * rs
                Rg, vg_, Ng_, gi = P4.blended_chain(fn, rs, Rcut, 4.0, RL, 0.05, kT)
                rho_g = Rcut + 4.0 * np.logaddexp(0.0, (Rg - Rcut) / 4.0)
                rho_g = RL - 0.05 * RL * np.logaddexp(0.0, (RL - rho_g) / (0.05 * RL))
                yg = kj / rho_g
                yg[:4] = yg[4]; yg[-4:] = yg[-5]
                G, NH = [], []
                for o in om_grid:
                    sc = GR.scatter_y(vg_, Ng_, m, yg, float(o))
                    rd = P4.hawking_reading(sc, float(o), kT / (2 * math.pi))
                    G.append(rd.get("Gamma", np.nan)); NH.append(rd.get("n_H", np.nan))
                Gi = np.interp(p["om"], om_grid, np.array(G)); Ni_ = np.interp(p["om"], om_grid, np.array(NH))
                gam[lab] = {"Gamma_w": float(np.sum(rho_w * Gi)), "n_stat": float(np.sum(rho_w * Ni_)), "R_cut": Rcut, "Gi": Gi}
            Gw = gam["R_t/4"]["Gamma_w"]; Gi = gam["R_t/4"]["Gi"]
            s_["Gamma_w"] = Gw; s_["Gamma_w_cut2"] = gam["R_t/2"]["Gamma_w"]; s_["n_stat"] = gam["R_t/4"]["n_stat"]; s_["n_stat_cut2"] = gam["R_t/2"]["n_stat"]
            nw = s_["n_w"]
            fT = lambda T: float(np.sum(rho_w * Gi / (np.exp(np.clip(p["om"] / T, -700, 700)) + 1.0))) - nw
            try:
                Tfit = brentq(fT, 1e-6 * kT, 100 * kT, xtol=1e-14)
                s_["theta_fit"] = Tfit / (kT / (2 * math.pi))
            except ValueError:
                s_["theta_fit"] = None
            s_["theta_conv"] = (omega / math.log(Gw / nw - 1.0)) / (kT / (2 * math.pi)) if (nw > 0 and Gw / nw > 2.0) else None
            # the first-passage share (G2) on the frozen slice at T_surf
            if not a.no_fp:
                Lfp = p["x0"] + 5 * p["s_x_density"]
                tfp = 2.0 * Lfp / max(p["vg_far"], 0.2)
                nfp = int(math.ceil((Lfp + tfp + 8 * rs) / 512.0)) * 512
                Rf = np.arange(1, nfp + 1, dtype=float)
                Nf = np.ones(nfp); vf = -np.sqrt(2 * a.g * slices[ks]["E"] / Rf)
                nl = min(nsites, nfp)
                Nf[:nl] = slices[ks]["N"][:nl]; vf[:nl] = slices[ks]["v"][:nl]
                Bf = Diags2(Nf, vf, m, kj / Rf, torch, dev)
                wf = V4.build_packet(nfp, int(round(p["x0"] - 1)), p["k"], p["a2"], p["me"], m=m, a=kj / p["x0"])
                X1 = torch.as_tensor(wf, device=dev) if torch else wf
                dt = a.fp_dt_over_rs * rs
                tt, share = [], []
                outside = Rf > Rt + 2
                t = 0.0
                while t < tfp:
                    X1 = cheb_expm2(Bf, X1, -dt)
                    t += dt
                    ph = X1.cpu().numpy() if torch else X1
                    psi = ph.reshape(-1, 2).copy()
                    psi[~outside] = 0.0
                    nf_ = 4 * nfp
                    kq = 2 * np.pi * np.fft.fftfreq(nf_)
                    pk = (np.abs(np.fft.fft(psi, n=nf_, axis=0)) ** 2).sum(axis=1)
                    tot = float(pk.sum())
                    out_norm = float((np.abs(psi) ** 2).sum())
                    share.append(out_norm * float(pk[kq < 0].sum()) / tot if tot > 0 else 0.0)
                    tt.append(t)
                win = RPG.plateau_window(np.array(tt), np.array(share), rs)
                s_["G2"] = {"series": [[tt[i_] / rsn, share[i_]] for i_ in range(0, len(tt), 4)], "share_min": float(min(share)), "share_max": float(max(share)),
                            "Gamma": (1.0 - win["median"]) if win else None, "spread": win["spread"] if win else None, "window": ([win["start"], win["end"]] if win else None),
                            "t_fp": tfp, "chain": nfp, "final_reflected_share": share[-1]}
            print("  T_surf/r_s %g: Gamma_w %.6f (R_t/2 cut %.6f) n_stat %.6e (cut2 %.6e) G2 %s theta (fit) %s theta (T_conv) %s kappa_T 4gE %s" % (
                s_["T_surf_over_rs"], Gw, s_["Gamma_w_cut2"], s_["n_stat"], s_["n_stat_cut2"], (("%.6f" % s_["G2"]["Gamma"]) if s_.get("G2") and s_["G2"]["Gamma"] is not None else "-"),
                ("%.5f" % s_["theta_fit"]) if s_.get("theta_fit") else "-", ("%.5f" % s_["theta_conv"]) if s_.get("theta_conv") else "-",
                ("%.5f" % (kT * 4 * a.g * s_["E"])) if s_.get("E") else "-"), flush=True)
        for s_ in series:
            th_ = s_.get("theta_fit") or 1.0
            q_ = s_["n_w"] / s_["Gamma_w"] if s_.get("Gamma_w") else 0.0
            den = 2 * math.pi * p["omega_over_kappa"] * max(1 - q_, 1e-12)
            s_["dtheta_duhamel"] = (th_ ** 2 * (s_["duhamel_dn_bound"] / s_["n_w"]) / den) if (s_.get("duhamel_dn_bound") is not None and s_["n_w"] > 0) else None
        if hold is not None:
            sh = [s_ for s_ in series if s_["T_surf_over_rs"] == a.hold_tsurf][0]
            th_ = sh.get("theta_fit") or 1.0
            hold["dtheta"] = th_ ** 2 * hold["rel_diff"] / (2 * math.pi * p["omega_over_kappa"])
        Ts = np.array([s_["T_surf_over_rs"] for s_ in series]); ns = np.array([s_["n_w"] for s_ in series])
        slope = float(np.polyfit(Ts, np.log(np.maximum(ns, 1e-300)), 1)[0]) if len(ns) > 1 and np.all(ns > 0) else None
        pr = {k: v for k, v in p.items() if k not in ("k", "om", "a2")}
        pr.update({"series": series, "log_slope_per_rs": slope, "hold_check": hold})
        rows.append(pr)
        json.dump({"script": "obs_raymap_solved_v3.py", "args": vars(a), "history": hinfo, "kappa_T_late": kappa, "E_late": E_late, "rs_geo": rs, "nsites": nsites, "rows": rows},
                  open(a.out_json, "w"), indent=1, default=float)
    json.dump({"script": "obs_raymap_solved_v3.py", "args": vars(a), "history": hinfo, "kappa_T_late": kappa, "E_late": E_late, "rs_geo": rs, "nsites": nsites, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    L = ["## J3' (obs_raymap_solved_v3.py): the late conversion through the core on the stored history", "",
         "History: %s (%s); tau %g, ell %g; nominal r_s %g, late 2 g E %.4f, kappa_T of the late slice %.6e (kappa_T 4 g E %.5f); j %g, kappa_j %+g; chain %d sites (%.1f r_s); the engine chain's junction max |N - 1| %.2e, |v - exterior| %.2e; the Duhamel rate window %.2f r_s before T_end." % (
             hinfo["source"], hinfo["label"].split(";")[0], hinfo["tau"], hinfo["ell"], rsn, rs, kappa, kappa * 4 * a.g * E_late, a.j, kj, nsites, nsites / rs,
             hinfo["junction_max_N"], hinfo["junction_max_v"], dt_rate / rsn), "",
         "### The packets", "", "| m_p/kappa | omega/kappa | x_0/r_s | s_x/r_s (density) | sigma_x/r_s (spec) | temporal width s_t/r_s | Gaussian tail beyond the chain | below-threshold measure |", "|---|---|---|---|---|---|---|---|"]
    f_ = lambda x, fmt: (fmt % x) if isinstance(x, (int, float)) and x is not None else "-"
    for row in rows:
        if "series" in row:
            L.append("| %g | %g | %.2f | %.2f | %.2f | %.2f | %.1e | %.1e |" % (row["mp_over_kappa"], row["omega_over_kappa"], row["x0"] / rs, row["s_x_density"] / rs,
                     row["sigma_x_spec"] / rs, row["s_t_over_rs"], row["gaussian_tail_beyond_chain"], row["below_threshold_measure"]))
    L += ["", "### The series (theta (fit) is the reading; theta (T_conv) the narrow-band comparator; overlapping temporal averages of width s_t, not independent instants)", "",
          "| m_p/kappa | omega/kappa | T_surf/r_s | T_obs/r_s | frozen beyond | n_w | n_w (w_s/2) | norm at T_0 | end-norm inside R_t + 5 ell | Duhamel B (bound on |dn_w|; d theta) | Gamma_w (R_t/4; R_t/2) | n_stat | theta (fit) | theta (T_conv) | R_t, r_in, R_sh | late | pairs inside R_t | pockets | G2 share (min, max over the back-trace) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "series" not in row:
            L.append("| %g | %g | %s | | | | | | | | | | | | | | | | |" % (row["mp_over_kappa"], row["omega_over_kappa"], row["skipped"]))
            continue
        for s_ in row["series"]:
            g2 = s_.get("G2") or {}
            L.append("| %g | %g | %g | %.2f | %s | %.6e | %.6e | %.10f | %s | %s | %s; %s | %s | %s | %s | %s, %s, %s | %s | %s | %s | %s |" % (
                row["mp_over_kappa"], row["omega_over_kappa"], s_["T_surf_over_rs"], s_["T_obs_over_rs"],
                "yes" if s_["frozen_beyond_history"] else "no", s_["n_w"], s_["n_w_ws_half"], s_["norm_at_T0"],
                (("%.2e" % s_["endnorm_inside_Rt_plus_5ell"]) + ("" if s_["endnorm_ok"] else " FAIL")) if s_["endnorm_inside_Rt_plus_5ell"] is not None else "-",
                ("%.2e (%.2e; %.1e)" % (s_["duhamel_B"], s_["duhamel_dn_bound"], s_["dtheta_duhamel"] or float("nan"))) if s_.get("duhamel_B") is not None else "-",
                f_(s_.get("Gamma_w"), "%.6f"), f_(s_.get("Gamma_w_cut2"), "%.6f"), f_(s_.get("n_stat"), "%.6e"),
                f_(s_.get("theta_fit"), "%.5f"), f_(s_.get("theta_conv"), "%.5f"), f_(s_.get("R_t"), "%.3f"), f_(s_.get("r_in"), "%.3f"), f_(s_.get("R_sh"), "%.1f"),
                s_.get("late_surface"), f_(s_.get("pairs_inside_Rt"), "%.3e"), s_.get("n_pocket"),
                ("%.4f, %.4f" % (g2["share_min"], g2["share_max"])) if g2 else "-"))
        hc = row.get("hold_check")
        L.append("| | | log slope of n_w per r_s | %s | quasi-frozen check | %s | | | | | | | | | | | | | |" % (
            f_(row.get("log_slope_per_rs"), "%.3e"), ("hold from %.2f r_s: %.6e against %.6e, relative %.2e (d theta %.1e)" % (hc["T_hold_over_rs"], hc["n_w_hold"], hc["n_w_stored"], hc["rel_diff"], hc.get("dtheta", float("nan")))) if hc else "-"))
    L += ["", "- every number above is printed by the script; theta (fit) is sec. 8.1's transmission-weighted form over the packet's spectral measure, theta (T_conv) sec. 10.2's omega/ln(Gamma_w/n_w - 1);",
          "  kappa_T per row from the record nearest T_surf; n_stat is the stationary scattering's occupation on the frozen slice (the two-lead control of 10.2 (3));",
          "  the Duhamel B extrapolates the last stored interval's field change linearly beyond T_end (a bound on |dn_w| of 2B + B^2; d theta = theta^2 (dn/n)/(2 pi r (1 - n/Gamma)));",
          "  G2 is the share of the back-traced packet at R > R_t + 2 on k < 0 on the frozen slice at T_surf (its time series in the JSON): it reaches about 1 when the",
          "  transmitted part returns through the core's lattice-scale conversion, so no first-passage plateau is read unless the series shows one;",
          "  P_0 is the probe's sea at t = 0 (the dust orbitals belong to the engine's field of mass m and are not in the stored history)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("done (%.0f s)" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
