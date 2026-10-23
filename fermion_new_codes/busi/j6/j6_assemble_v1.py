#!/usr/bin/env python3
"""
j6_assemble_v1.py -- assemble the formed planar coefficient of the J6 control from the run dumps (no evolution):
  route B (mdensity json): every G7/K15 node direct on [M_LO, sqrt 2];
  the light json: direct channels below M_LO (and the massless channel if run), on the same parent;
  eta_formed(hold 0) = eta_flat(exact 1D quadrature)
                       + sum over panels of rho (S4_hold0 - S4_flat_chain)              [every node direct]
                       + trapezoid over the light channels of rho (S4_hold0 - S4_flat_chain) on [0, M_LO]
                         (the piece below the lightest computed mass is an ESTIMATE unless M = 0 was run; rho(0) = 0),
  with the error table (per-panel K15-G7 of the excess, the trapezoid against a cubic spline, the estimate's bracket,
  the parent dependence from the second light parent, the time step, the purity floor) and the C6 verdict;
  the wall-free hold growth per channel on the rate parent against the thermal estimate at T = kappa/2pi;
  the rho-weighted growth, its hold series (linear = stationary) and R with its comparators.
The dumps of the v1 code (j6_transverse_chain.py) and of v2 have the same format.
Usage: j6_assemble_v1.py --mdensity mdensity.json --light light.json --parent 1024 --rate-parent 2048 --tau 12
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j6_transverse_chain_v2 import (rho_mass, s4_exact, eta_flat_1d, quad_rho, trapezoid, gk15, thermal_growth,
                                    KAPPA, T_HAWK, V_UP, ETA_FLAT_REF, C6_TARGET, C6_TOL, M_THRESHOLD)

def load(path):
    with open(path) as f: return json.load(f)

def S4(r, tau):
    """4-component cut entropy at the hold label tau from a result dict (json keys are str(float))."""
    for k, v in r['series'].items():
        if abs(float(k) - tau) < 1e-9: return 2.0*v[0]
    raise KeyError(f"hold {tau:g} not in {list(r['series'])}")

def light_parent(lt, Nx):
    key = str(Nx)
    if key not in lt: raise KeyError(f"parent {Nx} not in the light json (parents {list(lt)})")
    return sorted(lt[key], key=lambda r: r['M'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mdensity', required=True)
    ap.add_argument('--light', required=True, help='light json containing the parent --parent (the mdensity parent) and the rate parent')
    ap.add_argument('--parent', type=int, default=1024)
    ap.add_argument('--rate-parent', type=int, default=2048, dest='rate_parent')
    ap.add_argument('--tau', type=float, default=12.0, help='hold label of the growth readout')
    ap.add_argument('--dt-spread', type=float, default=3e-8, dest='dt_spread',
                    help='per-channel 2-component hold-0 spread of the converged time-step ladder (ladders mode: 2.6e-8 at M 0.1)')
    args = ap.parse_args()
    md = load(args.mdensity); lt = load(args.light)
    nodes = np.array(md['nodes'], float); breaks = [float(b) for b in md['breaks']]; res = md['results']
    Nx_md = int(res[0]['Nx'])
    if Nx_md != args.parent:
        print(f"WARNING: the mdensity parent is {Nx_md}, --parent is {args.parent}; the light tail must be on the mdensity parent")
    M_lo = breaks[0]
    rho = rho_mass(nodes)
    exc0 = np.array([S4(r, 0.0) - 2.0*r['S_flat'] for r in res])
    flat_chain = np.array([2.0*r['S_flat'] for r in res])
    purity_max = max(float(r['purity']) for r in res)
    nzero = max(int(r['n_zero_modes']) for r in res); nz0 = max(int(r['nz0']) for r in res)

    # ---- route B: panel sums of the excess with the K15 and G7 rules (weights rebuilt from the breaks)
    exc_K = exc_G = flat_K = 0.0; exc_hi = exc_mid = 0.0; straddle = None
    print("panel            excess (K15)    |K15-G7|")
    for i, (a, b) in enumerate(zip(breaks[:-1], breaks[1:])):
        x, wk, wg = gk15(a, b); sl = slice(15*i, 15*i + 15)
        if np.max(np.abs(x - nodes[sl])) > 1e-9: raise ValueError(f"panel {i} nodes do not match the json")
        fe = rho[sl]*exc0[sl]; ff = rho[sl]*flat_chain[sl]
        IK, IG = float(np.sum(wk*fe)), float(np.sum(wg*fe))
        exc_K += IK; exc_G += IG; flat_K += float(np.sum(wk*ff))
        if a >= M_THRESHOLD: exc_hi += IK
        elif b <= M_THRESHOLD: exc_mid += IK
        else: straddle = IK
        print(f"[{a:.5f},{b:.5f}]  {IK:+.4e}    {abs(IK-IG):.1e}")
    tot = md['totals']
    check = tot['hold0'][0] - tot['flat_chain'][0]
    print(f"\nexcess on the panels [{M_lo}, {breaks[-1]:.5f}]: {exc_K:+.5e}  (json totals hold0 - flat_chain: {check:+.5e}; difference {exc_K-check:+.1e})")
    print(f"  |K15 - G7| of the excess, summed: {abs(exc_K-exc_G):.1e} (resolution diagnostic; the van Hove panels' errors cancel in the excess)")
    print(f"  gapped channels (panels with a >= {M_THRESHOLD:.4f}): {exc_hi:+.4e};  pocket channels (b <= threshold): {exc_mid:+.4e};"
          f"  straddling panel: {straddle if straddle is not None else 0.0:+.4e}")
    print(f"  zero modes: max n_zero_modes {nzero}, max nz0 {nz0} (must both be 0);  max |S_A - S_B| over the nodes: {purity_max:.1e}")

    # ---- the exact flat and the panel set's design error
    eta_flat, eta_flat_err = eta_flat_1d()
    tail_flat = eta_flat_1d(0.0, M_lo)[0]
    design = (flat_K + tail_flat) - eta_flat
    print(f"\neta_flat exact: {eta_flat:.10f} (quad err {eta_flat_err:.0e}; ref {ETA_FLAT_REF});  panels + exact tail on the chain's flat: {flat_K+tail_flat:.10f}"
          f"  (design error {design:+.2e}, removed by using the exact flat; its residual in the excess is of order 1e-11)")

    # ---- the light channels on the mdensity parent: the excess below M_LO
    L = light_parent(lt, args.parent)
    Ml = np.array([r['M'] for r in L]); rl = rho_mass(Ml)
    excl = np.array([S4(r, 0.0) - 2.0*r['S_flat'] for r in L])
    sel = Ml <= M_lo*(1.0 + 1e-9)
    if not sel.any() or abs(Ml[sel].max() - M_lo) > 1e-6*M_lo:
        print(f"WARNING: the light set does not reach M_LO = {M_lo}; the junction is not covered")
    Ms, fs = Ml[sel], (rl*excl)[sel]
    light_direct = trapezoid(fs, Ms)
    cs_int = float(CubicSpline(Ms, fs).integrate(float(Ms[0]), float(Ms[-1]))) if len(Ms) >= 4 else light_direct
    light_err = abs(cs_int - light_direct)
    print(f"\nlight channels on parent {args.parent}, M <= {M_lo}:")
    for m, e, f in zip(Ms, excl[sel], fs): print(f"  M {m:.5f}  excess {e:.5f}  rho*excess {f:.4e}")
    print(f"  trapezoid of rho*excess over [{Ms[0]:g},{Ms[-1]:g}]: {light_direct:.4e}  (cubic spline {cs_int:.4e}; difference {light_err:.1e})")
    if Ms[0] > 0.0:
        held = 0.5*Ms[0]*fs[0]
        if len(Ms) > 1:
            slope = (excl[sel][1] - excl[sel][0])/(Ms[1] - Ms[0]); e_at0 = max(excl[sel][0] - slope*Ms[0], 0.0)
            lin = quad_rho(lambda M: e_at0 + slope*M, 0.0, float(Ms[0]))[0]
        else:
            lin, e_at0 = held, excl[sel][0]
        below = 0.5*(held + lin); below_err = 0.5*abs(lin - held) + 0.2*below
        below_note = f"ESTIMATE: excess held at {excl[sel][0]:.3f} gives {held:.2e}; linear in M to {e_at0:.2f} nats at M 0 gives {lin:.2e}"
    else:
        below, below_err, below_note = 0.0, 0.0, "direct (the massless channel was run; rho(0) = 0)"
    print(f"  piece [0,{Ms[0]:g}]: {below:.2e} +- {below_err:.1e}  ({below_note})")

    # ---- parent dependence of the hold-0 light channels
    parent_err = 0.0
    for key in lt:
        Nx2 = int(key)
        if Nx2 == args.parent: continue
        L2 = light_parent(lt, Nx2)
        common = [(a, b) for a in L for b in L2 if abs(a['M'] - b['M']) < 1e-12 and a['M'] <= M_lo*(1.0 + 1e-9)]
        if len(common) >= 2:
            Mc = np.array([a['M'] for a, b in common]); d = np.array([abs(S4(a, 0.0) - S4(b, 0.0)) for a, b in common])
            pe = trapezoid(rho_mass(Mc)*d, Mc); parent_err = max(parent_err, pe)
            print(f"  parent dependence of hold 0 (rho-weighted |S4({Nx2}) - S4({args.parent})| over [{Mc[0]:g},{Mc[-1]:g}]): {pe:.1e};"
                  f" largest per-channel difference {d.max():.1e} at M {Mc[int(np.argmax(d))]:g}")
    if Ms[0] > 0.0: parent_err += 0.2*below      # the uncomputed piece's parent sensitivity, allowed at 20 percent

    # ---- assembly and the error table
    excess_total = exc_K + light_direct + below
    eta_formed = eta_flat + excess_total
    dt_err = 2.0*args.dt_spread
    cheb_err = 2.0*purity_max
    err_items = [('panels (K15-G7 of the excess)', min(abs(exc_K - exc_G), 5e-7) if abs(exc_K - exc_G) < 5e-7 else abs(exc_K - exc_G)),
                 ('light trapezoid vs cubic spline', light_err),
                 (f'piece below {Ms[0]:g}', below_err),
                 ('finite parent', parent_err),
                 ('time step', dt_err),
                 ('purity floor', cheb_err)]
    err_total = sum(v for _, v in err_items)
    print("\n=== assembly (formation end, hold 0) ===")
    print(f"  eta_flat (exact)                         {eta_flat:.7f}")
    print(f"  excess on the panels [{M_lo}, sqrt 2]      {exc_K:+.4e}")
    print(f"  excess of the light channels [{Ms[0]:g},{M_lo}]  {light_direct:+.4e}")
    print(f"  excess below {Ms[0]:g}                        {below:+.4e}  ({'estimate' if Ms[0] > 0 else 'direct'})")
    print(f"  formation offset Delta                    {excess_total:+.4e}  (relative {excess_total/eta_flat:+.3e})")
    print(f"  eta_formed(hold 0)                        {eta_formed:.7f} +- {err_total:.1e}")
    print("  error table (linear sum of measured spreads and labelled estimates; not a certified bound):")
    for name, v in err_items: print(f"    {name:40s} {v:.1e}")
    diff = eta_formed - C6_TARGET
    verdict = 'PASS' if abs(diff) + err_total < C6_TOL else 'NOT PASSED'
    print(f"  C6: eta_formed - {C6_TARGET} = {diff:+.3e}; |diff| + error = {abs(diff)+err_total:.2e} against {C6_TOL}: {verdict}"
          f"{'' if Ms[0] == 0.0 else ' (the piece below ' + f'{Ms[0]:g}' + ' is an estimate; run the light set to M = 0 to remove it)'}")

    # ---- the wall-free hold growth on the rate parent and R
    key2 = str(args.rate_parent)
    if key2 not in lt:
        print(f"\n(rate parent {args.rate_parent} not in the light json: no growth readout)"); return
    R2 = light_parent(lt, args.rate_parent)
    M2 = np.array([r['M'] for r in R2]); r2 = rho_mass(M2); tau = args.tau
    try:
        g = np.array([S4(r, tau) - S4(r, 0.0) for r in R2])
    except KeyError as e:
        print(f"\n(hold {tau:g} not in the light json: {e})"); return
    th = np.array([thermal_growth(M, tau) for M in M2])
    print(f"\n=== wall-free hold growth on parent {args.rate_parent}, hold 0 -> {tau:g} (4-component) against the thermal estimate ===")
    print("  (CONDITIONAL estimate: 2 copies x 2 valleys x (particle + hole), T = kappa/2pi, escape gap 0.8 M, unit transmission)")
    for m, gg, tt in zip(M2, g, th):
        print(f"  M {m:.5f}  growth {gg:+.4e}  thermal {tt:.4e}  ratio {gg/tt if tt > 0 else float('nan'):.3f}")
    if abs(g[-1]) > 1e-6:
        print(f"  WARNING: the heaviest light channel (M {M2[-1]:g}) still grows by {g[-1]:.1e}: channels above the light set contribute to R")
    G_direct = trapezoid(r2*g, M2)
    if M2[0] > 0.0:
        scale = g[0]/th[0] if th[0] > 0 else 1.0
        G_below = quad_rho(lambda M: scale*thermal_growth(M, tau), 0.0, float(M2[0]))[0]
        below_note2 = f"thermal shape scaled by the ratio {scale:.3f} at M {M2[0]:g} (estimate)"
    else:
        G_below, below_note2 = 0.0, "direct (rho(0) = 0)"
    G = G_direct + G_below
    t_hold = tau/KAPPA
    R = G/t_hold
    R_rest = 7.0/(180.0*math.pi)*KAPPA**3
    R_prof = R_rest/(1.0 - V_UP**2)
    print(f"  rho-weighted growth: direct {G_direct:.3e} + below {M2[0]:g}: {G_below:.3e} ({below_note2}) = {G:.3e}  (relative {G/eta_flat:.2e})")
    print(f"  R (interval average over {t_hold:g} time units) = {R:.3e} per a^3 per unit time = {R/KAPPA**3:.4f} kappa^3")
    print(f"  comparators: transparent thermal for this profile's upstream flow {V_UP} (thresholds 0.8 M): {R_prof:.3e} ({R_prof/KAPPA**3:.4f} kappa^3), ratio {R/R_prof:.3f};"
          f"  exterior at rest (7/(180 pi)) kappa^3: {R_rest:.3e}, ratio {R/R_rest:.3f}")
    print(f"  equality of the linear and vacuum terms: kappa t = {eta_flat*KAPPA/R:.3e} = {eta_flat*KAPPA**3/R:.1f}/(kappa a)^2  (63.5/(kappa a)^2 for the exterior-at-rest comparator)")
    taus_avail = sorted({float(k) for k in R2[0]['series']})
    if len(taus_avail) > 2:
        print("  hold series of the rho-weighted growth over the computed light channels (direct part; a linear series is a stationary rate):")
        for t in taus_avail[1:]:
            gt = np.array([S4(r, t) - S4(r, 0.0) for r in R2]); Gt = trapezoid(r2*gt, M2)
            print(f"    hold {t:4g}: growth {Gt:.4e}  per label {Gt/t:.4e}")

if __name__ == '__main__':
    main()
