#!/usr/bin/env python3
"""obs_raymap_solved_v1.py (builder analyst; J3' of SPEC_final4b_observables.md sec. 10.2, B4: the model's own late conversion through its
core, on the collapse's stored history; sec. 8.1's estimator).

The history: the collapse's re-solves (engine_collapse_v4's JSON records and fields npz): slice k holds (N, v) on the engine's sites at time
T_k, frozen over [T_k, T_{k+1}) (the engine's quasi-static step); beyond the engine's chain the analytic exterior N = 1, v = -sqrt(2 g E_k/R)
at the slice's collar charge, so the probe's chain can be longer than the engine's (--nsites); beyond the stored history the last slice is
frozen (the spec's "the late fields frozen beyond the stored history"). The probe: the radial channel kappa_j = sign (j + 1/2) of a fermion of
mass m_p (the test's declared parameter, --mp-over-kappa) on the model's operator with the lapse hook (on-site N (m_p sz + (kappa_j/R) sy), the
sine hop with the averaged N, the range-two flow hop); its initial state P_0 = the negative-energy projector of h(T_0) (no surface at T_0, so
the transported reference coincides with the instantaneous ground state there), by the Chebyshev expansion of the erf-smoothed step at
width w_s = kappa/40 (w_s/2 the control; sec. 8.1). The observable: a Gaussian packet w at frequency omega on the outgoing branch of the far
exterior (flow_chart_v4's construction at the late slice's exterior; sigma_omega = omega/6, sec. 8.1), observed at T_obs; its occupation
n_w = <phi| f(h_0) |phi> with phi = U(T_obs)^dagger w, the backward propagation through the history by Chebyshev exponentials per frozen
interval (BUILDER: every interval exact to 1e-14 in the expansion). The surface time of the packet T_surf = T_obs - (x_0 - R_t)/v_g (its
back-traced arrival at the surface); the series: T_surf at 5 times over the last 10 r_s of the stored history (so T_obs runs beyond it on the
frozen late fields, labelled). The transmission: Gamma_j(omega) on the late slice's exterior by the stationary scattering of
obs_greybody_radial_v1 (the slice imported beyond R_cut with obs_plateau_v4's blend; BUILDER, labelled: R_cut = R_core + 5 ell), averaged over
the packet's spectral measure; the temperature from n_w = int d omega rho_w(omega) Gamma(omega) f(omega; T) (sec. 8.1's transmission-weighted
form), theta = T/(kappa_T/2 pi) with kappa_T = d_R(N + v) at R_t of the late slice (engine_collapse_v4's record). Printed per (omega, T_surf):
n_w, its w_s/2 control, the packet's norm on the chain at T_0 (leak check), Gamma, T, theta; and the drain-or-growth slope of n_w over
T_surf. --synthetic builds a history without a collapse record: the no-trap exterior v_0 ramped to the exact profile over t_r (the control on
which J3' must return the stationary conversion).  Args: --record --fields | --synthetic rs; --j --sign --mp-over-kappa --omega-over-kappa
--nsites --tsurf-span --nseries --ws-over-kappa --gpu --out-json --md."""
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
import obs_plateau_v4 as P  # noqa: E402

BW = 5


class Diags:
    """the probe's operator as its 11 component diagonals (numpy, or torch on a GPU)."""

    def __init__(self, N, v, m, y, torch=None, dev=None):
        ab = GR.band_matrix_y(np.asarray(v, float), np.asarray(N, float), m, np.asarray(y, float))
        dim = ab.shape[1]
        self.n, self.torch, self.dev = dim, torch, dev
        self.d = []
        g = np.zeros(dim)
        for off in range(-BW, BW + 1):
            if off >= 0:
                dg = ab[BW - off, off:]
            else:
                dg = ab[BW - off, :dim + off]
            if np.max(np.abs(dg)) == 0:
                continue
            if off >= 0:
                g[:dim - off] += np.abs(dg)
            else:
                g[-off:] += np.abs(dg)
            self.d.append((off, torch.as_tensor(dg, device=dev) if torch else dg))
        self.bound = float(g.max()) * 1.0001

    def apply(self, X):
        Y = (self.torch.zeros_like(X) if self.torch else np.zeros_like(X))
        n = self.n
        for off, D in self.d:
            if off >= 0:
                Y[:n - off] += D * X[off:]
            else:
                Y[-off:] += D * X[:n + off]
        return Y


def cheb_expm(B, X, tau, tol=1e-15):
    """exp(-i tau h) X (tau may be negative: the backward step)."""
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


def cheb_step_occupation(B, X, ws, nmax=400000):
    """<X| f(h) |X> and <X| f(h)^2 |X> with f(E) = erfc(E/(sqrt 2 ws))/2, by the Chebyshev expansion on [-r, r] (the coefficients by the
    discrete cosine sums; the degree grown until the coefficients fall below 1e-13)."""
    r = B.bound
    M = 1
    while M < 8 * r / ws:
        M *= 2
    M = min(M, nmax)
    th = (np.arange(M) + 0.5) * math.pi / M
    fx = 0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws))
    c = np.array([2.0 / M * np.sum(fx * np.cos(k * th)) for k in range(M)]) if M <= 4096 else None
    if c is None:
        from scipy.fft import dct
        c = dct(fx, type=2) / M
    c[0] /= 2.0
    keep = np.nonzero(np.abs(c) > 1e-13)[0]
    K = int(keep[-1]) + 1 if len(keep) else 1
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + (c[1] * T1 if K > 1 else 0 * T0)
    for k in range(2, K):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    dot = lambda u, w: (complex((u.conj() * w).sum().item()) if hasattr(u, "conj") else complex(np.vdot(u, w)))
    return float(dot(X, Y).real), float(dot(Y, Y).real), K


def history_from_record(record, fields, g, nsites):
    D = json.load(open(record))
    recs = [r for r in D["records"] if "E" in r]
    Z = np.load(fields, allow_pickle=True)
    Re = np.asarray(Z["R"], float)
    T = np.asarray(Z["T"], float)
    R = np.arange(1, nsites + 1, dtype=float)
    slices = []
    for k in range(len(T)):
        rec = min(recs, key=lambda q: abs(q["T"] - T[k]))
        E = float(rec["E"])
        N = np.ones(nsites); v = -np.sqrt(2 * g * E / R)
        ne = min(len(Re), nsites)
        N[:ne] = np.asarray(Z["N"][k], float)[:ne]; v[:ne] = np.asarray(Z["v"][k], float)[:ne]
        slices.append({"T": float(T[k]), "N": N, "v": v, "E": E, "rec": rec})
    info = {"source": "collapse record %s" % os.path.basename(record), "tau": float(Z["tau"]), "ell": float(Z["ell"]), "label": str(Z["label"]),
            "engine_nsites": int(len(Re))}
    return R, slices, info


def history_synthetic(rs, nsites, tr_over_rs, dt_over_rs, nfrozen):
    """the control: v_0 (the no-trap form, |v_0| <= 0.73) ramped to the exact profile over t_r with s = sin^2, then frozen."""
    R = np.arange(1, nsites + 1, dtype=float)
    x = R / (2 * rs)
    A, Bc = -1.2374368670764582, 0.5303300858899106
    v0 = np.where(x >= 1.0, -np.sqrt(rs / R), A * x + Bc * x ** 3)
    Rm = rs / 8
    v1 = np.where(R >= Rm, -np.sqrt(rs / R), -np.sqrt(rs / Rm) * R / Rm)
    slices = []
    t, dt, tr = 0.0, dt_over_rs * rs, tr_over_rs * rs
    while t < tr + nfrozen * dt:
        s = 1.0 if t >= tr else math.sin(math.pi * (t + dt / 2) / (2 * tr)) ** 2
        slices.append({"T": t, "N": np.ones(nsites), "v": v0 + s * (v1 - v0), "E": rs / 0.5, "rec": {}})
        t += dt
    return R, slices, {"source": "synthetic ramp v_0 -> exact (r_s %g, t_r %g r_s, step %g r_s)" % (rs, tr_over_rs, dt_over_rs), "tau": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None)
    ap.add_argument("--fields", default=None)
    ap.add_argument("--synthetic", type=float, default=None, help="r_s of the synthetic control history")
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--j", type=float, default=0.5)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--mp-over-kappa", type=float, nargs="+", default=[0.0])
    ap.add_argument("--omega-over-kappa", type=float, nargs="+", default=[1.0])
    ap.add_argument("--nsites", type=int, default=16384)
    ap.add_argument("--tsurf-span", type=float, default=10.0, help="units of r_s: the series' span before the history's end")
    ap.add_argument("--nseries", type=int, default=5)
    ap.add_argument("--tsurf-list", type=float, nargs="+", default=None, help="explicit T_surf values in units of r_s (overrides the span)")
    ap.add_argument("--ws-over-kappa", type=float, default=1.0 / 40)
    ap.add_argument("--Rcut-over-rs", type=float, default=None, help="Gamma's import cut (default: R_core + 5 ell from the late record)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    torch = dev = None
    if a.gpu is not None:
        import torch as _t
        torch, dev = _t, _t.device("cuda:%d" % a.gpu)
    if a.synthetic:
        R, slices, hinfo = history_synthetic(a.synthetic, a.nsites, 2.0, 0.0625, 8)
    else:
        R, slices, hinfo = history_from_record(a.record, a.fields, a.g, a.nsites)
    late = slices[-1]
    E_late = late["E"]
    rs_late = 2 * a.g * E_late
    kT = late["rec"].get("kappa_T") if late["rec"] else None
    if a.synthetic:
        kT = 1.0 / (2 * a.synthetic)                                     # the exact profile's surface gravity (the control)
    if kT is None:
        fs = late["N"] + late["v"]
        i = int(np.nonzero(fs[:-1] * fs[1:] <= 0)[0][-1])
        kT = float(fs[i + 1] - fs[i])
    kappa = kT
    kj = a.sign * (a.j + 0.5)
    y = kj / R
    T_end = slices[-1]["T"]
    tau_end = hinfo["tau"]
    ops = {}

    def op(k, m):
        key = (k, m)
        if key not in ops:
            if len(ops) > 64:
                ops.clear()
            s_ = slices[k]
            ops[key] = Diags(s_["N"], s_["v"], m, y, torch, dev)
        return ops[key]
    times = [s_["T"] for s_ in slices]
    rows = []
    t0 = time.time()
    for mk in a.mp_over_kappa:
        m = mk * kappa
        B0 = Diags(slices[0]["N"], slices[0]["v"], m, y, torch, dev)
        for r_ in a.omega_over_kappa:
            omega = r_ * kappa
            sg = omega / 6.0
            if m > 0 and omega - 3 * sg <= m:
                rows.append({"omega_over_kappa": r_, "mp_over_kappa": mk, "skipped": "below the threshold window"})
                continue
            x0 = 4 * rs_late
            for _ in range(6):
                vext = -math.sqrt(rs_late / x0); me = math.sqrt(m * m + (kj / x0) ** 2)
                k, om, a2 = V4.branch_momenta(omega, sg, me, vext, outgoing=True)
                vg0 = abs(float(np.sum(a2 * V4.group_velocity(k, me, vext) * np.gradient(k))))
                sx = vg0 / (2 * sg)
                x1 = max(4 * rs_late, rs_late + 4 * sx)
                if abs(x1 - x0) < 1e-9:
                    break
                x0 = x1
            if x0 + 5 * sx > a.nsites - 8:
                rows.append({"omega_over_kappa": r_, "mp_over_kappa": mk, "skipped": "the packet does not fit (x0 %.0f, sigma_x %.0f)" % (x0, sx)})
                continue
            w = V4.build_packet(a.nsites, int(round(x0 - 1)), k, a2, me, m=m, a=kj / x0)
            travel = (x0 - rs_late) / max(vg0, 1e-6)
            series = []
            tlist = [x * rs_late for x in a.tsurf_list] if a.tsurf_list else [T_end - a.tsurf_span * rs_late * (1 - q / max(a.nseries - 1, 1)) for q in range(a.nseries)]
            for T_surf in tlist:
                T_obs = T_surf + travel
                phi = torch.as_tensor(w, device=dev) if torch else w.copy()
                t = T_obs
                while t > 1e-9:
                    kk = max(i for i, Ti in enumerate(times) if Ti <= t - 1e-9) if t > times[0] + 1e-9 else 0
                    t_lo = times[kk]
                    dtb = t - t_lo
                    phi = cheb_expm(op(kk, m), phi, -dtb)
                    t = t_lo
                    if kk == 0:
                        break
                occ, occ2, Kd = cheb_step_occupation(B0, phi, a.ws_over_kappa * kappa)
                occ_c, _, _ = cheb_step_occupation(B0, phi, 0.5 * a.ws_over_kappa * kappa)
                norm = float((phi.conj() * phi).sum().real) if torch else float(np.vdot(phi, phi).real)
                series.append({"T_surf": T_surf, "T_obs": T_obs, "frozen_beyond_history": T_obs > T_end + tau_end, "n_w": occ, "n_w_sq": occ2, "n_w_ws_half": occ_c,
                               "norm_at_T0": norm, "cheb_degree": Kd})
                print("m_p/kappa %g omega/kappa %g T_surf/r_s %.2f: n_w %.6e (w_s/2 %.6e) norm %.12f (%.0f s)" % (mk, r_, T_surf / rs_late, occ, occ_c, norm, time.time() - t0), flush=True)
            Ts = np.array([s_["T_surf"] for s_ in series]); ns = np.array([s_["n_w"] for s_ in series])
            slope = float(np.polyfit(Ts / rs_late, np.log(np.maximum(ns, 1e-300)), 1)[0]) if len(ns) > 1 and np.all(ns > 0) else None
            rows.append({"omega_over_kappa": r_, "mp_over_kappa": mk, "omega": omega, "x0": x0, "sigma_x": sx, "travel": travel, "series": series,
                         "log_slope_per_rs": slope, "packet_k": k.tolist(), "packet_a2": a2.tolist(), "packet_om": om.tolist()})
    # the transmission on the late exterior and the temperature
    for row in rows:
        if "series" not in row:
            continue
        m = row["mp_over_kappa"] * kappa
        om = np.array(row["packet_om"]); a2 = np.array(row["packet_a2"]); kk = np.array(row["packet_k"])
        rho_w = a2 * np.gradient(kk); rho_w = rho_w / rho_w.sum()
        if a.synthetic:
            fn = lambda rho: (np.ones_like(np.asarray(rho, float)), -np.sqrt(a.synthetic / np.maximum(np.asarray(rho, float), 1e-12)))
            Rcut = a.synthetic / 16
        else:
            fn, sinfo = P.slice_fields(a.fields, a.record, len(slices) - 1, a.g)
            Rc_late = late["rec"].get("R_c") or late["rec"].get("R_sh") or rs_late / 8
            Rcut = (a.Rcut_over_rs * rs_late) if a.Rcut_over_rs else (Rc_late + 5 * hinfo.get("ell", 4))
        RL = 64 * rs_late
        Rg, vg_, Ng_, gi = P.blended_chain(fn, rs_late, Rcut, 4.0, RL, 0.05, kappa)
        rho_g = Rcut + 4.0 * np.logaddexp(0.0, (Rg - Rcut) / 4.0)
        rho_g = RL - 0.05 * RL * np.logaddexp(0.0, (RL - rho_g) / (0.05 * RL))          # the same map as blended_chain's
        yg = kj / rho_g
        yg[:4] = yg[4]; yg[-4:] = yg[-5]
        Gs = []
        for o in om[:: max(1, len(om) // 24)]:
            sc = GR.scatter_y(vg_, Ng_, m, yg, float(o))
            rd = P.hawking_reading(sc, float(o), kappa / (2 * math.pi))
            Gs.append((float(o), rd.get("Gamma")))
        oG = np.array([q[0] for q in Gs]); gG = np.array([q[1] if q[1] is not None else np.nan for q in Gs])
        Gam = np.interp(om, oG, gG)
        row["Gamma_packet"] = float(np.sum(rho_w * Gam)); row["Gamma_grid"] = Gs; row["Gamma_Rcut"] = Rcut
        for s_ in row["series"]:
            nw = s_["n_w"]
            fT = lambda T: float(np.sum(rho_w * Gam / (np.exp(np.clip(om / T, -700, 700)) + 1.0))) - nw
            try:
                Tfit = brentq(fT, 1e-6 * kappa, 100 * kappa, xtol=1e-14)
                s_["T"] = Tfit; s_["theta"] = Tfit / (kappa / (2 * math.pi))
            except ValueError:
                s_["T"] = None; s_["theta"] = None
    L = ["## J3' (obs_raymap_solved_v1.py): the late conversion through the core on the stored history", "",
         "History: %s; kappa_T of the late slice %.6e (kappa_T 4 g E %.5f); j %g, kappa_j %+g." % (hinfo["source"], kappa, kappa * 4 * a.g * E_late, a.j, kj), "",
         "| m_p/kappa | omega/kappa | T_surf/r_s | T_obs beyond the history | n_w | n_w (w_s/2) | norm at T_0 | Gamma (packet) | theta |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if "series" not in row:
            L.append("| %g | %g | - | - | %s | | | | |" % (row["mp_over_kappa"], row["omega_over_kappa"], row["skipped"]))
            continue
        for s_ in row["series"]:
            L.append("| %g | %g | %.2f | %s | %.6e | %.6e | %.10f | %.6f | %s |" % (row["mp_over_kappa"], row["omega_over_kappa"], s_["T_surf"] / rs_late,
                     "yes" if s_["frozen_beyond_history"] else "no", s_["n_w"], s_["n_w_ws_half"], s_["norm_at_T0"], row["Gamma_packet"],
                     ("%.5f" % s_["theta"]) if s_.get("theta") else "-"))
        L.append("| | | log slope of n_w per r_s | %s | | | | | |" % (("%.3e" % row["log_slope_per_rs"]) if row["log_slope_per_rs"] is not None else "-"))
    L += ["", "- every number above is printed by the script; theta is the transmission-weighted fit of sec. 8.1 against kappa_T/(2 pi)."]
    json.dump({"script": "obs_raymap_solved_v1.py", "args": vars(a), "history": hinfo, "kappa_T": kappa, "E_late": E_late, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
