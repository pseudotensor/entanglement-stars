#!/usr/bin/env python3
"""d1_raymap_data_v1.py (solve lane, PAGE VERIFY D1): the ray map of the B2 history and the conformal-copy
prediction for the massless chain's exterior-cut entropy, per copy (one sector), on the record clock.

Independent of (but following the same recipe as) deliveries/lead_gaps/REVIEWA_write_A2_scripts/a2_raymap_B2.py:
the stored applied fields (N, v) of the B2 record (801 records, tau = 8 a, 4096 sites), bilinear in (t, R); the
regular centre unfolded (x < 0 is the ingoing leg); the appended exterior beyond 4096 a Painleve-Gullstrand at the
record's charge (N = 1, v = -sqrt(r_s(t)/R), r_s(t) = 4096 v(t, 4096)^2, the production's rule) to the chain's wall
at 8192 a; the fields frozen before T = 0 and after the last record. Right-moving null rays dx/dt = f(x, t) =
N(|x|, t) + sgn(x) v(|x|, t); the separation equation d(delta x)/dt = f_x delta x gives every Jacobian without
small differences.

Two predictions are evaluated per cut R_c and record clock t:

(1) The sheet's line formula (REVIEWA_write_A2_v1 RESULT), reproduced as a check of the map:
      dS_wall = (1/12) ln[(p(u_c) - p(u_w))^2 / ((2 D)^2 p'(u_c) p'(u_w))],  u_w = t - D - D_ref (the sheet's u_0),
    with p(u), p'(u) from the family of rays crossing +R_ref (4096 a) outbound at u, and the semi-infinite form
    dS_semi = -(1/12) ln p'(u_c).

(2) The finite-chain (circle) form of the in-state entropy with the local cutoff factors at the cut:
      S(t) = (1/6) ln[(L/pi) sin(pi l_int(t)/L)] + (1/12)[Lambda_out(t) + Lambda_in(t)]
             + (1/12) ln[f_out(R_c, t) f_in(R_c, t)] + const,
    L the initial static chain's round-trip time (the circumference, in the advanced label p, of the ray circle on
    which the initial Killing vacuum is the conformal vacuum), l_int = p_in - p_out the p-arc of the rays inside the
    cut at t (p = crossing time of x = -R_ref), Lambda = -ln(dp/dt) for the outgoing and the ingoing endpoint from
    the separation equation, f_out = N - |v| and f_in = N + |v| at the cut (the lattice spacing's width in p at each
    endpoint is a (dp/dt)/f). The prediction for the state's entropy growth per copy is dS_circ = S(t) - S(0);
    the reference's (held-vacuum) lattice behaviour does not enter it.

Also written: the static-vacuum check at T = 0 (the circle form's prediction for the differences of the vacuum cut
entropy between the five cuts), the e-folding markers, and the frozen-fields travel times.
Usage (box, nice 10): python d1_raymap_data_v1.py --out d1_raymap_v1.npz
"""
import argparse, json, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fields", default="<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz")
ap.add_argument("--out", default="d1_raymap_v1.npz")
ap.add_argument("--dt", type=float, default=0.25, help="RK4 step in a")
ap.add_argument("--t-step", type=float, default=0.5, help="record-clock grid step in r_s")
ap.add_argument("--t-max", type=float, default=100.0)
ap.add_argument("--nsites", type=int, default=8192)
ap.add_argument("--Rref", type=float, default=4096.0, help="reference radius in a (the last stored site)")
ap.add_argument("--du", type=float, default=0.125, help="u grid step of the reference family in r_s")
ap.add_argument("--umin", type=float, default=-40.0)
ap.add_argument("--umax", type=float, default=190.0)
args = ap.parse_args()

RS = 64.0
t_start = time.time()
z = np.load(args.fields)
T = np.asarray(z["T"], float); Rg = np.asarray(z["R"], float)
N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
nT, nR = N.shape; dT = T[1] - T[0]
assert np.allclose(np.diff(T), dT) and np.allclose(Rg, np.arange(1, nR + 1))
N0 = np.concatenate([N[:, :1], N], axis=1)            # column R = 0: N(0) = N(1), v(0) = 0 (regular centre)
v0 = np.concatenate([np.zeros((nT, 1)), v], axis=1)
rs_t = nR * v[:, -1] ** 2                               # the record's charge seen at the last site
Rmax = float(nR); Rwall = float(args.nsites); Rref = float(args.Rref)


def fields(t, r):
    """N, v, dN/dR, dv/dR at (t, r >= 0): bilinear in the records; PG beyond Rmax; frozen outside [0, T_end]."""
    t = np.clip(t, 0.0, T[-1])
    it = np.minimum((t / dT).astype(int), nT - 2)
    wt = np.clip((t - T[it]) / dT, 0.0, 1.0)
    rr = np.minimum(r, Rmax)
    ir = np.minimum(rr.astype(int), nR - 1)
    wr = rr - ir

    def bil(F):
        F00 = F[it, ir]; F01 = F[it, ir + 1]; F10 = F[it + 1, ir]; F11 = F[it + 1, ir + 1]
        val = (1 - wt) * ((1 - wr) * F00 + wr * F01) + wt * ((1 - wr) * F10 + wr * F11)
        der = (1 - wt) * (F01 - F00) + wt * (F11 - F10)
        return val, der

    Nv, Nd = bil(N0); vv, vd = bil(v0)
    out = r > Rmax
    if np.any(out):
        rs = (1 - wt[out]) * rs_t[it[out]] + wt[out] * rs_t[it[out] + 1]
        ro = r[out]; q = np.sqrt(rs / ro)
        Nv[out] = 1.0; Nd[out] = 0.0; vv[out] = -q; vd[out] = 0.5 * q / ro
    return Nv, vv, Nd, vd


def f_df(x, t):
    """the right-moving null speed on the unfolded line and its x derivative."""
    s = np.where(x >= 0, 1.0, -1.0)
    r = np.abs(x)
    Nv, vv, Nd, vd = fields(t, r)
    return Nv + s * vv, s * Nd + vd


def trace(x0, t0, h, mode, X, maxsteps):
    """vectorized RK4 of (x, G), dx/dt = f, dG/dt = f_x, from (x0, t0) with the signed step h until x crosses X
    ('le': x <= X, 'ge': x >= X). Returns the crossing times, G at the crossing (linear within the step), done."""
    n = x0.size
    x = x0.astype(float).copy(); t = t0.astype(float).copy(); G = np.zeros(n)
    done = np.zeros(n, bool); tc = np.full(n, np.nan); Gc = np.full(n, np.nan)
    for _ in range(maxsteps):
        act = np.where(~done)[0]
        if act.size == 0:
            break
        xa = x[act]; ta = t[act]; Ga = G[act]
        k1, g1 = f_df(xa, ta)
        k2, g2 = f_df(xa + 0.5 * h * k1, ta + 0.5 * h)
        k3, g3 = f_df(xa + 0.5 * h * k2, ta + 0.5 * h)
        k4, g4 = f_df(xa + h * k3, ta + h)
        xn = xa + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Gn = Ga + h / 6 * (g1 + 2 * g2 + 2 * g3 + g4)
        tn = ta + h
        cross = (xn <= X) if mode == "le" else (xn >= X)
        if cross.any():
            w = (X - xa[cross]) / (xn[cross] - xa[cross])
            tc[act[cross]] = ta[cross] + w * h
            Gc[act[cross]] = Ga[cross] + w * (Gn[cross] - Ga[cross])
            done[act[cross]] = True
        x[act] = xn; t[act] = tn; G[act] = Gn
    return tc, Gc, done


def f_at(x, t):
    return f_df(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(t, float)))[0]


def travel(x1, x2, t, n=40000):
    """frozen-fields travel time of a right-mover from x1 to x2 (> x1) on the unfolded line at the record clock t."""
    xx = np.linspace(x1, x2, n + 1)
    f, _ = f_df(xx, np.full(xx.size, float(t)))
    return float(np.trapezoid(1.0 / f, xx))


dt = args.dt
maxsteps = int((400.0 * RS) / dt)

# ---------------------------------------------------------------- (A) the reference family: p(u), ln p'(u)
u_grid = np.arange(args.umin, args.umax + 1e-9, args.du) * RS
p_grid, G_A, okA = trace(np.full(u_grid.size, Rref), u_grid, -dt, "le", -Rref, maxsteps)
fo = f_at(np.full(u_grid.size, Rref), u_grid); fi = f_at(np.full(u_grid.size, -Rref), p_grid)
lnp1 = np.log(fo / fi) + G_A                            # ln p'(u) (G from the backward trace is minus the integral)
Lam_ref = -lnp1
pH = float(np.nanmax(p_grid))
aff = u_grid < 0
const0 = float(np.nanmedian((u_grid - p_grid)[aff]))    # u - p in the affine (static) region
print("family A: %d rays, %d traced, %.1f s; p_H = %.4f r_s; affine offset u - p = %.4f r_s" % (
    u_grid.size, okA.sum(), time.time() - t_start, pH / RS, const0 / RS), flush=True)


def P_aff(uu):
    out = np.interp(uu, u_grid, p_grid, left=np.nan)
    early = uu < u_grid[0]; out[early] = uu[early] - const0
    return out


def LNP1_aff(uu):
    out = np.interp(uu, u_grid, lnp1, left=np.nan)
    out[uu < u_grid[0]] = 0.0
    return out


# ---------------------------------------------------------------- (C) the initial static circle
xx = np.arange(-Rwall, Rwall + 1e-9, 0.05)
f_static, _ = f_df(xx, np.zeros(xx.size))
L_circle = float(np.trapezoid(1.0 / f_static, xx))     # the round trip of the initial static chain (in p)
print("initial static circle L = %.4f r_s (= %.1f a)" % (L_circle / RS, L_circle), flush=True)

# ---------------------------------------------------------------- (B) the cuts
cuts_rs = [2.0, 16.0, 24.0, 30.0, 48.0]
t_rec = np.arange(0.0, args.t_max + 1e-9, args.t_step) * RS
res = {"u_grid": u_grid, "p_grid": p_grid, "lnp1": lnp1, "Lam_ref": Lam_ref, "pH": pH, "const0": const0,
       "L_circle": L_circle, "t_rec": t_rec, "rs_t": rs_t, "T": T, "Rref": Rref, "Rwall": Rwall, "dt": dt,
       "cuts_rs": np.array(cuts_rs)}
summary = {"pH_rs": pH / RS, "const0_rs": const0 / RS, "L_circle_rs": L_circle / RS, "dt_a": dt, "cuts": {}}
nt = t_rec.size
for Rc_rs in cuts_rs:
    Rc = Rc_rs * RS
    tag = "R%g" % Rc_rs
    # (B1) the cut's retarded time: out-ray forward to +Rref
    u_c, _, ok1 = trace(np.full(nt, Rc), t_rec, +dt, "ge", Rref, maxsteps)
    # (B2) the out-ray at the cut traced back through the centre to -Rref: p_out, Lambda_out
    p_out, G_out, ok2 = trace(np.full(nt, Rc), t_rec, -dt, "le", -Rref, maxsteps)
    f_out_c = f_at(np.full(nt, Rc), t_rec)
    lnpdot_out = np.log(f_out_c / f_at(np.full(nt, -Rref), p_out)) + G_out
    Lam_out = -lnpdot_out
    # (B3) the in-ray at the cut traced back to -Rref: p_in, Lambda_in
    p_in, G_in, ok3 = trace(np.full(nt, -Rc), t_rec, -dt, "le", -Rref, maxsteps)
    f_in_c = f_at(np.full(nt, -Rc), t_rec)
    lnpdot_in = np.log(f_in_c / f_at(np.full(nt, -Rref), p_in)) + G_in
    Lam_in = -lnpdot_in
    # the circle form
    l_int = p_in - p_out
    S_circ_term = (1.0 / 6) * np.log((L_circle / np.pi) * np.sin(np.pi * l_int / L_circle))
    S_cut_term = (1.0 / 12) * np.log(f_out_c * f_in_c)
    S_pred = S_circ_term + (1.0 / 12) * (Lam_out + Lam_in) + S_cut_term
    dS_circ = S_pred - S_pred[0]
    # the sheet's line formula, reproduced (frozen travel times at each record, u_w = t - D - D_ref)
    D_out = np.array([travel(Rc, Rwall, tt) for tt in t_rec])
    D_in = np.array([travel(-Rwall, -Rc, tt) for tt in t_rec])
    D_ref = np.array([travel(Rref, Rwall, tt) for tt in t_rec])
    u_w = t_rec - D_out - D_ref
    pc = P_aff(u_c); pw = P_aff(u_w); lc = LNP1_aff(u_c); lw = LNP1_aff(u_w)
    dS_wall = (1.0 / 12) * (2 * np.log(np.abs(pc - pw)) - 2 * np.log(2 * D_out) - lc - lw)
    dS_semi = -lc / 12
    # the sheet's formula with the actual ingoing leg (u_0 = t - D_in - D_ref) instead of 2D: a variant for the budget
    u_w2 = t_rec - D_in - D_ref
    pw2 = P_aff(u_w2); lw2 = LNP1_aff(u_w2)
    dS_wall_in = (1.0 / 12) * (2 * np.log(np.abs(pc - pw2)) - 2 * np.log(D_out + D_in) - lc - lw2)
    # e-folding markers on the cut's own Lambda_out
    marks = {}
    for s_ in (0.1, 1.0, 3.0, 6.6, 7.05, 7.7, 10.0, 20.0):
        j = np.where(Lam_out > s_)[0]
        marks["%g" % s_] = float(t_rec[j[0]] / RS) if j.size else None
    for k_, arr in (("u_c", u_c), ("p_out", p_out), ("Lam_out", Lam_out), ("p_in", p_in), ("Lam_in", Lam_in),
                    ("f_out_c", f_out_c), ("f_in_c", f_in_c), ("l_int", l_int), ("S_circ_term", S_circ_term),
                    ("S_cut_term", S_cut_term), ("S_pred", S_pred), ("dS_circ", dS_circ), ("D_out", D_out),
                    ("D_in", D_in), ("D_ref", D_ref), ("u_w", u_w), ("dS_wall", dS_wall), ("dS_semi", dS_semi),
                    ("dS_wall_in", dS_wall_in), ("Lam_ref_at_uc", -lc)):
        res["%s_%s" % (k_, tag)] = arr
    summary["cuts"][tag] = {"traced": [int(ok1.sum()), int(ok2.sum()), int(ok3.sum())], "efold_marks_t_rs": marks,
                            "l_int0_rs": float(l_int[0] / RS), "S_pred0": float(S_pred[0]),
                            "D_out_rs_0_100": [float(D_out[0] / RS), float(D_out[-1] / RS)],
                            "D_in_rs_0_100": [float(D_in[0] / RS), float(D_in[-1] / RS)],
                            "lag_uc_minus_t_rs_0_50_100": [float((u_c[i] - t_rec[i]) / RS) for i in (0, nt // 2, nt - 1)]}
    print("\ncut %g r_s (%d/%d/%d traced; %.0f s): l_int(0) = %.3f r_s, D_out = %.2f..%.2f r_s, D_in = %.2f..%.2f r_s, lag u_c - t = %.2f (t=0) %.2f (t=50) %.2f (t=100) r_s; e-folds 1, 3, 6.6, 7.7, 20 at t = %s" % (
        Rc_rs, ok1.sum(), ok2.sum(), ok3.sum(), time.time() - t_start, l_int[0] / RS, D_out[0] / RS, D_out[-1] / RS,
        D_in[0] / RS, D_in[-1] / RS, (u_c[0] - t_rec[0]) / RS, (u_c[nt // 2] - t_rec[nt // 2]) / RS, (u_c[-1] - t_rec[-1]) / RS,
        [marks[k] for k in ("1", "3", "6.6", "7.7", "20")]), flush=True)
    print("   t/r_s  u_c/r_s  Lam_out  Lam_in   l_int/r_s  circ_term-0  cut_term-0  | dS_circ | dS_wall(sheet) dS_semi dS_wall_in")
    for tt in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 50, 60, 70, 80, 90, 100):
        i = int(round(tt / args.t_step))
        if i >= nt: continue
        print("   %5.1f  %7.2f  %7.3f  %6.3f  %8.3f   %8.4f   %8.4f   | %7.4f | %7.4f  %7.4f  %7.4f" % (
            tt, u_c[i] / RS, Lam_out[i], Lam_in[i], l_int[i] / RS, S_circ_term[i] - S_circ_term[0], S_cut_term[i] - S_cut_term[0],
            dS_circ[i], dS_wall[i], dS_semi[i], dS_wall_in[i]), flush=True)

# ---------------------------------------------------------------- (D) the static-vacuum check at T = 0
S0 = np.array([res["S_pred_R%g" % c][0] for c in cuts_rs])
summary["static_vacuum_check"] = {"cuts_rs": cuts_rs, "S_pred0_minus_S_pred0_at_2rs": (S0 - S0[0]).tolist(),
                                  "note": "compare with half the chain's S_X held vacuum at T = 0 minus its 2 r_s value"}
print("\nstatic vacuum check (T = 0): predicted S(R_c) - S(2 r_s) per copy:", np.round(S0 - S0[0], 4).tolist())
res["S_pred0_cuts"] = S0
np.savez(args.out, **res)
with open(args.out.replace(".npz", "_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print("\nwrote %s and its summary json (%.0f s)" % (args.out, time.time() - t_start))
