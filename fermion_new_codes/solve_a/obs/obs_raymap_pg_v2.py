#!/usr/bin/env python3
"""obs_raymap_pg_v2.py (builder analyst; version 2 of the lead's draft as completed by the builder, after the reviewer B red team of the
observables (reviewb/review_output/redteam_final4b_observables): (1) the plateau temperature is withheld (theta_r = None, the reason
recorded) whenever the ray-map exponent is unresolved: version 1 fell back to kappa = 1/(2 r_s) when the k-route fit was missing, which
substitutes the control value for the measured quantity, and used any fitted slope however far from the exponential regime; (2) the
outgoing-branch roots of F(k) = sqrt(M_T^2 + sin^2 k) - sin k cos k - omega are all found (a grid of 4001 points on (0, pi/2) and
brentq on every sign change; version 1 called brentq only when the endpoint signs differed and missed the two-root channels); every root is
reported with its group velocity at the surface and the k-route window uses the smallest root with outward group velocity (labelled).
The windows themselves are the lead's (sec. 3.2) and are the subject of the next specification turn.  Version 1 of the docstring:
obs_raymap_pg_v1.py (lead's first draft; SPEC_final4b_observables.md sec. 3; J3 on the exact Painleve-Gullstrand radial profile, control).

Imports flow_chart_v4 (the chain, the profile, the packets) and flow_page_v2 (the CF4 formation ramp, the hold, the far packets).  Per channel
(r_s, m, j, sign): the frozen final background h_f on N = 8192 sites (GPU eigh), the far packets at omega/kappa in {1/2, 1, 2, 4}, the
back-trace split by region and branch (R < R_b; R > r_s with k < 0; R > r_s with k > 0), the exponent kappa_p by the k-route and the x-route,
the greybody Gamma_j(omega) by G2 (the back-trace's reflected share) and G1 (a near-surface forward launch), the symbol's backward rays, the
occupations n_w from the ramp at the late holds and theta_r from n_w/Gamma.  BUILDER: the plateau-window detection (the curve's derivative),
the massive k_s root, the planar control run (the same functions on planar_fields with the tanh profile), the writer."""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flow"))
import flow_chart_v4 as V4  # noqa: E402
import flow_page_v2 as FP  # noqa: E402

SX, SY, SZ, I2 = V4.SX, V4.SY, V4.SZ, V4.I2


def chain_hamiltonian_N(mz, yy, v, N):
    """flow_chart_v4.chain_hamiltonian with the lapse: N_i on the on-site block, (N_i + N_{i+1})/2 on the sine hop, the flow hop unweighted
    by N (H_m[N, v] = sum N Hcal + (1/2){v, P}); N = 1 gives the version-4 operator exactly."""
    mz, yy, v, N = (np.asarray(x, float) for x in (mz, yy, v, N))
    n = len(mz)
    h = np.zeros((2 * n, 2 * n), complex)
    for i in range(n):
        h[2 * i:2 * i + 2, 2 * i:2 * i + 2] = N[i] * (mz[i] * SZ + yy[i] * SY)
    for i in range(n - 1):
        h[2 * i:2 * i + 2, 2 * i + 2:2 * i + 4] = -0.5j * 0.5 * (N[i] + N[i + 1]) * SX
    for i in range(n - 2):
        h[2 * i:2 * i + 2, 2 * i + 4:2 * i + 6] = -0.25j * 0.5 * (v[i] + v[i + 2]) * I2
    return np.triu(h) + np.triu(h, 1).conj().T


def split_state(psi, R, rs, Rb):
    """the state split by region and branch: (norm R < R_b, centre and log-mean |k| of that part, short-wave share there;
    weight at R > r_s with k < 0 [the ingoing branch, the reflected part in a back-trace]; weight at R > r_s with k > 0)."""
    a = psi.reshape(-1, 2)
    n = 4 * len(a)
    k = 2 * np.pi * np.fft.fftfreq(n)
    def kspec(mask):
        b = np.where(mask[:, None], a, 0.0)
        tot = float(np.sum(np.abs(b) ** 2))
        if tot < 1e-30:
            return tot, None
        pk = (np.abs(np.fft.fft(b, n=n, axis=0)) ** 2).sum(1)
        return tot, pk / pk.sum() * tot
    nin, pin = kspec(R < Rb)
    row = {"norm_in": nin}
    if pin is not None:
        rho = (np.abs(a) ** 2).sum(1) * (R < Rb)
        row["x_in"] = float((rho * R).sum() / rho.sum())
        pos = np.abs(k) > 0
        row["k_in_logmean"] = float(np.exp((pin[pos] * np.log(np.abs(k[pos]))).sum() / pin[pos].sum()))
        sw = (np.abs(k) >= np.pi / 4) & (np.abs(k) <= 3 * np.pi / 4)
        row["short_wave_share_in"] = float(pin[sw].sum() / nin)
    nout, pout = kspec(R >= rs)
    if pout is not None:
        row["w_out_kneg"] = float(pout[k < 0].sum())
        row["w_out_kpos"] = float(pout[k > 0].sum())
    return row


def backtrace(F, w, taus, R, rs, Rb):
    rows = []
    for tau in taus:
        psi = F.evolve_f(w, -tau)
        rows.append(dict(tau=float(tau), **split_state(psi, R, rs, Rb)))
    return rows


def fit_log(rows, key, tkey, lo, hi, offset=0.0, sign=1.0):
    """slope of ln(y - offset) against tau over the rows with lo < y - offset < hi."""
    pts = [(r[tkey], r[key] - offset) for r in rows if r.get(key) is not None and lo < r[key] - offset < hi]
    if len(pts) < 4:
        return {"n": len(pts)}
    t = np.array([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    c = np.polyfit(t, y, 1)
    return {"n": len(pts), "slope": float(sign * c[0]), "rms": float(np.sqrt(np.mean((np.polyval(c, t) - y) ** 2))), "window": [lo, hi]}


def exponents(rows, kappa, k0, omega, rs, m=0.0, kj=None):
    """k-route: d ln k_in/d tau over 1.5 k0 < k_in < k_s/3; x-route: -d ln(x_in - r_s)/d tau over 1.5 k_s^2/kappa < x_in - r_s < r_s/4.
    k_s: the outgoing branch's momentum at Killing frequency omega on the surface (w = 1): sin k (1 - cos k) = omega massless;
    BUILDER: the massive root of sqrt(m^2 + sin^2 k) - sin k cos k = omega with the centrifugal term."""
    a2s = 0.0 if kj is None else (kj / rs) ** 2      # the root with the channel's angular term at the surface (w = 1)
    fks = lambda k: np.sqrt(m * m + a2s + np.sin(k) ** 2) - np.sin(k) * np.cos(k) - omega
    # version 2: every root on (0, pi/2) (grid and brentq), each with the surface group velocity d/dk [E - sin k cos k] (N = 1, v = -1)
    kg = np.linspace(1e-9, np.pi / 2 - 1e-9, 4001)
    fg = np.array([fks(x) for x in kg])
    roots = [brentq(fks, kg[i], kg[i + 1], xtol=1e-14) for i in range(len(kg) - 1) if fg[i] * fg[i + 1] < 0]
    vg = lambda k: np.sin(k) * np.cos(k) / np.sqrt(m * m + a2s + np.sin(k) ** 2) - np.cos(2 * k)
    root_info = [{"k": float(r), "group_velocity_at_surface": float(vg(r))} for r in roots]
    outward = [r for r in roots if vg(r) > 0]
    ks = min(outward) if outward else None
    out = {"k_s": ks, "roots": root_info}
    if ks:
        out["k_route"] = fit_log(rows, "k_in_logmean", "tau", 1.5 * k0, ks / 3)
        out["x_route"] = fit_log(rows, "x_in", "tau", 1.5 * ks * ks / kappa, rs / 4, offset=rs, sign=-1.0)
        for r_ in ("k_route", "x_route"):
            if "slope" in out[r_]:
                out[r_]["kappa_p_over_kappa"] = out[r_]["slope"] / kappa
    return out


def plateau_window(t, y, rs, falling=False):
    """BUILDER: the plateau between the first change of y (the reflected part forming, or crossing inward when falling) and the next change
    (the centre's return): the smoothed derivative's main peak, the first time after it where |dy/dt| < 0.1 of the peak (the window's
    start), the first later time where the derivative regains 0.2 of the peak in either direction or start + 2 r_s (the window's end)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if len(t) < 7:
        return None
    d = np.gradient(y, t)
    d = np.convolve(d, np.ones(3) / 3, mode="same")
    s = -d if falling else d
    i_pk = int(np.argmax(s))
    peak = s[i_pk]
    if peak <= 0:
        return None
    after = np.nonzero(np.abs(d[i_pk:]) < 0.1 * peak)[0]
    if not len(after):
        return None
    i0 = i_pk + int(after[0])
    end = t[i0] + 2 * rs
    later = np.nonzero((np.abs(d[i0:]) > 0.2 * peak) | (t[i0:] > end))[0]
    i1 = i0 + int(later[0]) if len(later) else len(t)
    if i1 - i0 < 2:
        return None
    sel = slice(i0, i1)
    return {"start": float(t[i0]), "end": float(t[i1 - 1]), "median": float(np.median(y[sel])), "spread": float(np.ptp(y[sel])), "n": i1 - i0,
            "peak_rate": float(peak)}


def greybody_G2(rows, rs):
    """1 - (the reflected share at R > r_s on the ingoing branch) at its plateau (BUILDER: plateau_window), the spread as the uncertainty."""
    w = np.array([r.get("w_out_kneg", np.nan) for r in rows])
    t = np.array([r["tau"] for r in rows])
    ok = np.isfinite(w)
    win = plateau_window(t[ok], w[ok], rs) if ok.sum() >= 7 else None
    if not win:
        return {"n": int(ok.sum()), "window": None}
    return {"Gamma": float(1 - win["median"]), "spread": win["spread"], "window": [win["start"], win["end"]], "n_window": win["n"]}


def near_surface_packet(N, R, rs, Rm, m, kj, omega, x0_offset=24.0, sigma_x=8.0):
    """G1's launch: the packet on the local outgoing branch at omega, centred at x0 = r_s + x0_offset, width sigma_x (sigma_omega = v_g/(2 sigma_x))."""
    x0 = rs + x0_offset
    vloc, aloc = float(V4.pg_profile(x0, rs, Rm)), kj / x0
    me = float(np.sqrt(m * m + aloc * aloc))
    k, om, a2 = V4.branch_momenta(omega, omega / 8, me, vloc, outgoing=True)
    vg = abs(float(np.sum(a2 * V4.group_velocity(k, me, vloc) * np.gradient(k))))
    sg = vg / (2 * sigma_x)
    k, om, a2 = V4.branch_momenta(omega, sg, me, vloc, outgoing=True)
    w = V4.build_packet(N, int(round(x0 - 1)), k, a2, me, m=m, a=aloc)
    return w, {"x0": x0, "sigma_x": sigma_x, "sigma_omega": sg, "vg_launch": vg, "k_mean": float(np.sum(a2 * k * np.gradient(k)))}


def greybody_G1(F, w, R, rs, times):
    """the norm at R >= r_s forward in time; Gamma = its plateau after the reflected part has crossed inward and before the return
    (BUILDER: plateau_window on the falling curve)."""
    rows = []
    for t in times:
        psi = F.evolve_f(w, t)
        rho = V4.density(psi)
        rows.append({"t": float(t), "norm_out": float(rho[R >= rs].sum()), "norm_in": float(rho[R < rs].sum())})
    win = plateau_window([r["t"] for r in rows], [r["norm_out"] for r in rows], rs, falling=True)
    return rows, ({"Gamma": win["median"], "spread": win["spread"], "window": [win["start"], win["end"]], "n_window": win["n"]} if win else None)


def symbol_backtrace(x0, k0, m, kj, rs, Rm, taus):
    """Hamilton's equations of the symbol backward from (x0, k0): the symbol's kappa_p from the same fits."""
    Rs, Ps = V4.symbol_ode(x0, k0, m, kj, rs, Rm, -np.asarray(taus))
    rows = [{"tau": float(t), "x_in": float(r), "k_in_logmean": float(abs(p))} for t, r, p in zip(taus, Rs, Ps)]
    return rows


def plateau_theta(F, w, pk, holds, kappa, Gamma, kappa_p):
    """n_w at the holds; theta_r from n_w/Gamma = F_w(theta) with the Fermi function at theta kappa_p/(2 pi) and the packet's measure."""
    out = {"hold": {}}
    for th in holds:
        _, n = F.apply_C(w, th / kappa)
        out["hold"][str(th)] = n
    late = [out["hold"][str(th)] for th in holds if th >= 48.0]
    nw = float(np.mean(late))
    out.update(n_plateau=nw, n_spread=float(np.ptp(late)), Gamma=Gamma)
    try:
        out["theta_r"] = V4.invert_theta(nw / Gamma, pk["k"], pk["om"], pk["a2"], kappa_p, lo=0.05, hi=5.0)
    except ValueError:
        out["theta_r"] = None
    return out


def run_channel(a, m, j, sign, gpu):
    kj = sign * (j + 0.5)
    rs, Rm = a.rs, a.Rm
    F = FP.Formation(a, m, kj, gpu)  # the ramp, h_f's eigenbasis, apply_C, evolve_f (BUILDER: Formation reads a.ramp_over_rs, a.dt, a.nsites, a.holds)
    R, N = F.R, F.N
    kappa = 1.0 / (2 * rs)
    out = {"label": "exact Painleve-Gullstrand profile (control); PG radial; v declared; formation by the ramp", "rs": rs, "Rm": Rm, "mass": m,
           "j": j, "kappa_j": kj, "N": N, "kappa": kappa, "packets": []}
    Rb = 1.5 * rs
    for r_ in a.omega_over_kappa:
        omega = r_ * kappa
        pk, why = FP.outgoing_packet(N, R, rs, Rm, m, kj, omega)
        if pk is None:
            out["packets"].append({"omega_over_kappa": r_, "skipped": why})
            continue
        w = pk["w"]
        taus = np.arange(0, (pk["x0"] - rs) / 0.5 + 16 * rs, 0.25 / kappa)
        rows = backtrace(F, w, taus, R, rs, Rb)
        k0 = float(np.sum(pk["a2"] * pk["k"] * np.gradient(pk["k"])))
        row = {"omega_over_kappa": r_, "omega": omega, "x0": pk["x0"], "sigma_x": pk["sigma_x"], "k0": k0, "backtrace": rows,
               "exponents": exponents(rows, kappa, k0, omega, rs, m, kj), "G2": greybody_G2(rows, rs),
               "symbol": exponents(symbol_backtrace(pk["x0"], k0, m, kj, rs, Rm, taus), kappa, k0, omega, rs, m, kj)}
        try:
            wn, info = near_surface_packet(N, R, rs, Rm, m, kj, omega)
            trows, g1 = greybody_G1(F, wn, R, rs, np.arange(0, 8 * rs, 4.0))
            row["G1"] = {"launch": info, "rows": trows, "plateau": g1}
        except ValueError as ex:
            row["G1"] = {"skipped": str(ex)}
        G = row["G2"].get("Gamma")
        kp = row["exponents"].get("k_route", {}).get("slope")
        if G and G > 1e-6 and kp is not None and kp > 0:
            row["plateau"] = plateau_theta(F, w, pk, [float(h) for h in a.holds], kappa, G, kp)
            row["plateau"]["kappa_p_used"] = "the k-route's fitted slope (its window is the lead's sec. 3.2; resolution not established)"
        else:
            # version 2: no substitution of the control value; the occupations are still recorded for a later analysis
            row["plateau"] = {"theta_r": None, "withheld": "exponent or transmission unresolved (k-route slope %s, Gamma %s)" % (kp, G),
                              "hold": {str(float(h)): F.apply_C(w, float(h) / kappa)[1] for h in a.holds}}
        out["packets"].append(row)
        print("rs %g m %g j %g omega/kappa %g: kappa_p/kappa k-route %s, x-route %s, Gamma(G2) %s" % (
            rs, m, j, r_, row["exponents"].get("k_route", {}).get("kappa_p_over_kappa"), row["exponents"].get("x_route", {}).get("kappa_p_over_kappa"), G), flush=True)
    F.free()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rs", type=float, default=64.0)
    p.add_argument("--Rm", type=float, default=16.0)
    p.add_argument("--nsites", type=int, default=8192)
    p.add_argument("--ramp-over-rs", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--holds", type=float, nargs="+", default=[48.0, 64.0, 80.0, 96.0])
    p.add_argument("--channels", required=True, help="m:j:sign,... e.g. 0:0.5:-1,0.0039:0.5:-1")
    p.add_argument("--omega-over-kappa", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res = {"script": os.path.basename(__file__), "args": vars(a), "channels": []}
    for spec in a.channels.split(","):
        m, j, sg = (float(x) for x in spec.split(":"))
        res["channels"].append(run_channel(a, m, j, sg, a.gpu))
        json.dump(res, open(os.path.join(a.outdir, "raymap_%s.json" % a.tag), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
