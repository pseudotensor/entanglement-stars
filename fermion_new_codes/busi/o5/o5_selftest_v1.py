#!/usr/bin/env python
"""o5_selftest_v1.py -- manufactured checks of O5's code with no data: (1) the spinor spherical harmonics (orthonormality by quadrature,
sigma.rhat Omega_kappa = -Omega_{-kappa}, the Condon-Shortley sign); (2) the chain operator (Hermitian, anticommutes with (-1)^j sigma_y K,
local symbol on a uniform chain); (3) the chain's fixed-cut identity (the strict weight change equals the exact integrated current) to
rounding on a held h, the Simpson integral to quadrature accuracy; (4) the lift of a smooth column on a fine cube reproduces the radial
weights within its Gram defect, the reduction check picks the written convention and rejects the 3 alternatives, the adjoint lift
returns the column; (5) the comparator's arithmetic on synthetic inputs (verdicts, clocks, the files). CPU, about 1 to 2 minutes.
Prints PASS/FAIL lines and ends with '[selftest] N failures'; exit code 0 when none.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_chain_export_v1 as CH   # noqa: E402
import o5_compare_v1 as CM        # noqa: E402
import o5_lift_v1 as LF           # noqa: E402

FAILS = []


def say(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def test_profile(n=240, rs=40.0, Rb=24.0):
    r = np.arange(1.0, n + 1.0)
    u = np.where(r < Rb, math.sqrt(rs / Rb) * r / Rb, np.sqrt(rs / r))
    N = 1.0 - 0.4 * np.exp(-(r / 10.0) ** 2)
    return r, N, -u


def t1_harmonics():
    xg, wg = np.polynomial.legendre.leggauss(48)
    phi = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    TH, PH = np.meshgrid(np.arccos(xg), phi, indexing='ij')
    Wq = (wg[:, None] * np.full(phi.shape, 2.0 * math.pi / phi.size)[None, :]).ravel()
    th, ph = TH.ravel(), PH.ravel()
    Y11 = LF.sph_Y(1, 1, np.array([math.pi / 2]), np.array([0.0]))[0]
    say(abs(Y11 + math.sqrt(3.0 / (8.0 * math.pi))) < 1e-12, f"Y_11 at the equator is -sqrt(3/8pi) (Condon-Shortley): {Y11:.6f}")
    funcs = []
    for kap in (-2, -1, 1, 2):
        for mj in np.arange(-(abs(kap) - 0.5), abs(kap) - 0.5 + 1e-9, 1.0):
            funcs.append((kap, float(mj), LF.spinor_harmonic(kap, mj, th, ph)))
    worst = 0.0
    for i, (ka, ma, Oa) in enumerate(funcs):
        for j, (kb, mb, Ob) in enumerate(funcs):
            g = np.sum(Wq * np.sum(np.conj(Oa) * Ob, 0))
            worst = max(worst, abs(g - (1.0 if i == j else 0.0)))
    say(worst < 1e-12, f"spinor harmonics kappa -2..2 orthonormal by quadrature (worst {worst:.1e})")
    rh = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 0)
    worst = 0.0
    for kap in (-2, -1, 1, 2):
        O = LF.spinor_harmonic(kap, 0.5, th, ph)
        Om = LF.spinor_harmonic(-kap, 0.5, th, ph)
        srO = np.stack([rh[2] * O[0] + (rh[0] - 1j * rh[1]) * O[1], (rh[0] + 1j * rh[1]) * O[0] - rh[2] * O[1]], 0)
        worst = max(worst, float(np.max(np.abs(srO + Om))))
    say(worst < 1e-12, f"sigma.rhat Omega_kappa = -Omega_(-kappa) pointwise for kappa -2..2 (worst {worst:.1e})")


def t2_chain():
    r, N, v = test_profile()
    ch = CH.RadialChain(r, N, v, 0.2, 1)
    say(ch.hermiticity_defect() < 1e-14, f"chain h Hermitian ({ch.hermiticity_defect():.1e})")
    say(ch.particle_hole_defect() < 1e-13, f"chain h anticommutes with (-1)^j sigma_y K ({ch.particle_hole_defect():.1e})")
    ch.diagonalize()
    say(ch.eps_res < 1e-11 and ch.eps_orth < 1e-11, f"eigendecomposition residual {ch.eps_res:.1e}, orthonormality defect {ch.eps_orth:.1e}")
    roots = CH.table_roots(r, N, v)
    say(len(roots) == 2 and roots[0][0] < roots[1][0] and roots[0][1] < 0 < roots[1][1], f"2 roots of N - |v| on the test table with the expected slope signs: {roots}")
    # the local symbol on a uniform chain: a wide packet far out, N 1, v -0.3, kappa 1
    n = 400
    ru = np.arange(1.0, n + 1.0)
    chu = CH.RadialChain(ru, np.ones(n), -0.3 * np.ones(n), 0.2, 1)
    k0, Rc = 0.5, 250.0
    u, E = CH.particle_spinor(k0, Rc, 0.2, 1)
    psi, _ = CH.gaussian_column(ru, Rc, 20.0, k0, 5.0, u)
    e, e2 = chu.expectation(psi.reshape(-1))
    sym = float(CH.chain_omega(k0, 1.0, -0.3, 0.2, 1, Rc))
    say(abs(e - sym) < 2e-3, f"<h> of a wide packet {e:.5f} against the local symbol N E(k) + (v/2) sin 2k = {sym:.5f}")


def t3_identity():
    r, N, v = test_profile()
    m, kap = 0.2, 1
    ch = CH.RadialChain(r, N, v, m, kap)
    ch.diagonalize()
    roots = CH.table_roots(r, N, v)
    r_h = roots[-1][0]
    u, _ = CH.particle_spinor(0.3, 30.0, m, kap)
    psi0, _ = CH.gaussian_column(r, 30.0, 6.0, 0.3, 5.0, u)
    psi0f = psi0.reshape(-1)
    dt, sub, S, nread = 0.5, 8, 3, 40
    times = dt * np.arange(0, nread * S + 1)
    states, c0 = ch.propagate(psi0f, times)
    dens = np.sum(np.abs(states) ** 2, 2)
    p_strict = CH.beyond_weights(r, dens, [r_h], 'strict')[:, 0]
    p_mid = CH.beyond_weights(r, dens, [r_h], 'midpoint')[:, 0]
    Q = ch.cut_operator(r > r_h)
    sweep_t = times[nread * np.arange(0, S + 1)]
    Jint = CH.integrated_current_exact(ch.w, ch.V, Q, c0, sweep_t)
    dp = p_strict[nread * np.arange(0, S + 1)] - p_strict[0]
    say(float(np.max(np.abs(Jint - dp))) < 1e-10, f"fixed-cut identity: strict weight change minus the exact integrated current, max {np.max(np.abs(Jint - dp)):.1e} (rounding); dp at the end {dp[-1]:+.4f}")
    t_fine = (dt / sub) * np.arange(0, nread * S * sub + 1)
    sf, _ = ch.propagate(psi0f, t_fine)
    ff = sf.reshape(len(t_fine), ch.dim)
    Jf = np.real(np.sum(np.conj(ff) * (ff @ Q.T), 1))
    Jsimp = CH.cumulative_simpson(Jf, dt / sub, sub)
    say(float(np.max(np.abs(Jsimp - (p_strict - p_strict[0])))) < 1e-5, f"Simpson integral of the current on dt/{sub} against the weight change at every readout time: max {np.max(np.abs(Jsimp - (p_strict - p_strict[0]))):.1e}")
    flat = states.reshape(len(times), ch.dim)
    J = np.real(np.sum(np.conj(flat) * (flat @ Q.T), 1))
    say(abs(J[0] - float(np.real(np.vdot(psi0f, Q @ psi0f)))) < 1e-12 and float(np.max(np.abs(np.imag(np.sum(np.conj(flat) * (flat @ Q.T), 1))))) < 1e-12, "the current is real and the direct and spectral forms agree at t 0")
    amb = np.max(np.abs(p_mid - p_strict))
    say(amb < 0.5 * np.max(dens), f"cell-rule against strict rule: the difference is at most half a shell weight (max {amb:.2e})")
    nrm = np.sum(np.abs(flat) ** 2, 1)
    say(float(np.max(np.abs(nrm - 1.0))) < 1e-12, f"norm kept to {np.max(np.abs(nrm - 1.0)):.1e} over {len(times)} readout times")
    clk = CH.half_weight_clock(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.2, 0.6, 0.8]))
    say(abs(clk - 1.5) < 1e-12, f"half-weight clock on a sampled ramp: {clk} (expected 1.5)")


def t4_lift():
    r, N, v = test_profile(n=120)
    m, kap, Rc, sigma, k0, cut, H = 0.2, 2, 30.0, 4.0, 0.2, 4.0, 48
    u, E = CH.particle_spinor(k0, Rc, m, kap)
    psi0, info = CH.gaussian_column(r, Rc, sigma, k0, cut, u)
    p_chain = np.sum(np.abs(psi0) ** 2, 1)
    S = LF.lift_shell_column(Rc, sigma, k0, cut, u, info['Z'], H, kap, 0.5, 1.0, 1, p_chain=p_chain)
    dG = S['gram_defect']
    say(abs(S['W_lift'] - 1.0) <= dG + 2e-3, f"lift weight before renormalization {S['W_lift']:.6f}: |W - 1| = {abs(S['W_lift'] - 1.0):.2e} within the Gram defect {dG:.2e} (+2e-3 radial quadrature)")
    say(abs(float(np.sum(np.abs(S['values']) ** 2)) - 1.0) < 1e-12, "lifted column renormalized to 1 on the cube")
    dens0 = p_chain[None, :]
    worst = 0.0
    for R in (20.0, 26.5, 33.0, 40.0):
        pm = CH.beyond_weights(r, dens0, [R], 'midpoint')[0, 0]
        pl = LF.weight_beyond(S['r'], S['values'], R)
        worst = max(worst, abs(pl - pm))
    say(worst <= dG + 2e-3, f"lifted weights beyond 4 radii reproduce the chain's cell-rule weights: worst difference {worst:.2e} against the Gram defect {dG:.2e} (+2e-3)")
    say(S['gram']['dev'][30] < 0.02 and S['gram']['dev'][2] > S['gram']['dev'][30], f"shell Gram deviation small on a large shell ({S['gram']['dev'][30]:.1e} at j 30) and larger on a small one ({S['gram']['dev'][2]:.1e} at j 2)")
    # the reduction check on a flat field (N 1, v 0): the chain with N 1, v 0 against the numpy cubic operator
    rt = np.arange(1.0, 201.0)
    ch = CH.RadialChain(rt, np.ones(rt.size), np.zeros(rt.size), m, kap)
    psi_flat, info_f = CH.gaussian_column(rt, Rc, sigma, k0, cut, u)
    e_chain, _ = ch.expectation(psi_flat.reshape(-1))
    Oa, Ob = LF.angular_pair(S['coords'], S['r'], kap, 0.5, 1)
    lifts = {}
    for ls_ in (1.0, -1.0):
        for ka in (1, -1):
            pair = (Oa, Ob) if ka > 0 else (Ob, Oa)
            lifts[f"lower{ls_:+g}_kappa{ka:+d}"] = LF.lift_shell_column(Rc, sigma, k0, cut, u, info_f['Z'], H, kap, 0.5, ls_, ka, pair=pair)
    Hc = int(math.ceil(Rc + cut * sigma)) + 3
    Nflat = np.ones(rt.size)
    rc = LF.reduction_check(lifts, rt, Nflat, 1e-9 * np.ones(rt.size), m, Hc)   # u ~ 0: a flat field
    d0 = abs(rc['lower+1_kappa+1']['E_mean'] - e_chain)
    others = [abs(v_['E_mean'] - e_chain) for k_, v_ in rc.items() if k_ != 'lower+1_kappa+1']
    say(d0 < 0.02 * abs(e_chain), f"reduction: the written convention's <h> on the cube {rc['lower+1_kappa+1']['E_mean']:.5f} against the chain's {e_chain:.5f} (|diff| {d0:.1e}, 2 percent allowed)")
    say(min(others) > 3.0 * d0, f"reduction: the 3 alternative conventions differ more (min {min(others):.2e} against {d0:.2e}): " + ', '.join(f"{k_} {v_['E_mean']:.4f}" for k_, v_ in rc.items()))
    # the adjoint lift returns the column (G ~ I)
    g = LF.channel_projection(S['coords'], S['values_raw'], kap, 0.5, r.size, 1.0, 1)
    rel = float(np.linalg.norm(g - psi0) / np.linalg.norm(psi0))
    say(rel < 0.05, f"the channel projection of the lifted column returns the chain column: relative difference {rel:.3e}")
    # the detector lift and the GAP10 axis packet replica
    nhat = LF.direction_vector('100')
    pos, rr = LF.site_on_direction(2.0 * 20.0, '100')
    cc, vv = LF.gap10_axis_packet(H, pos, nhat, 0.3, LF.helicity_spinors(0.3, nhat, m)[0], 3.0, 4.0)
    say(abs(float(np.sum(np.abs(vv) ** 2)) - 1.0) < 1e-12 and cc.shape[0] > 0, f"GAP10 axis packet replica normalized on {cc.shape[0]} sites at {pos.tolist()}")
    spn = LF.helicity_spinors(0.3, nhat, m)
    say(abs(np.vdot(spn[0], spn[1])) < 1e-12 and all(abs(np.linalg.norm(s_) - 1.0) < 1e-12 for s_ in spn), "helicity spinors orthonormal")


def t5_comparator():
    rs = 29.0
    S = 4
    sweep_t = rs * np.arange(0, S + 1)
    times = 0.5 * np.arange(0, 58 * S + 1)
    p_ch = {'r_h': np.array([0.0, 0.3, 0.6, 0.8, 0.9]), '1.5rs': np.array([0.0, 0.1, 0.3, 0.5, 0.6]), '2rs': np.array([0.0, 0.05, 0.2, 0.4, 0.5]), '3rs': np.array([0.0, 0.0, 0.1, 0.2, 0.3])}
    d_ch = 1e-9 * np.ones(S + 1)
    d_cu = np.array([0.0, 1e-5, 1e-4, 5e-4, 2e-3])
    dG = 1e-3
    p_cu = {}
    for nm, p in p_ch.items():
        p_cu[nm] = p.copy()
        for s in range(1, S + 1):
            tot = CM.weight_band(p[s], d_ch[s]) + CM.weight_band(p[s], d_cu[s]) + dG
            p_cu[nm][s] = p[s] + (0.3 * tot if nm != '2rs' else 3.0 * tot)
    amp_ch = np.exp(-(times - 40.0) ** 2 / 50.0)
    amp_cu = 1.001 * np.exp(-(times - 45.0) ** 2 / 50.0)
    J_ch = np.array([0.0, 1e-2, 2e-2, 1e-2, 5e-3])
    pair = dict(name='synthetic', chain_json='c.json', lift_json='l.json', cube_json='k.json', rs=rs, sweep_t=sweep_t, radii=CM.RADII, times=times,
                column=dict(Rc=20.0, sigma=8.0, ak0=0.1, kappa=1, E_mean=0.15, E_rms=0.05, spinor_mode='particle'),
                chain=dict(p=p_ch, p_alt={nm: p + 2e-3 for nm, p in p_ch.items()}, delta=d_ch, J=J_ch, Jint_exact=p_ch['r_h'], dp_strict=p_ch['r_h'], wmax=2.0, amp_delta=1e-9 * np.ones(times.size),
                           lattice_scale=0.01, killing=None, r_h=22.25, identity_defect_exact_max=1e-13, prop_err_last=1e-9),
                cube=dict(p=p_cu, delta=d_cu, J=J_ch + 1e-6, bound=2.1, amp_delta=1e-4 * np.ones(times.size), lattice_scale=0.012, killing=None, r_h=22.25, r_h_direction=22.274, H=232,
                          napply=None, wall=None, energy_mean=0.15, energy_std=0.05, norm_drift=1e-13),
                lift=dict(gram_defect=dG, amp_lift_error=2e-3, W_lift=1.0, reduction=dict(default_is_closest=True), det_gram=[1e-4, 1e-4]),
                amps=[dict(label='o5det|s=+', sign=1, k=0.2, chain=amp_ch, cube=amp_cu), dict(label='o5det|s=-', sign=-1, k=-0.2, chain=amp_ch, cube=None)], axis=[])
    opts = dict(chain_rule='midpoint', cert_max=1e-3, clock_tol_rs=1.0, lattice_scale_min=0.05)
    R = CM.compare_pair(pair, opts)
    v = {nm: [row['verdict'] for row in R['shells'][nm]] for nm in CM.RADII}
    say(all(x == 'agree' for x in v['r_h'] + v['1.5rs'] + v['3rs']), f"comparator: residuals at 0.3 of the error sum agree ({v['r_h']})")
    say(all(x.startswith('physics') for x in v['2rs']), f"comparator: residuals at 3 times the error sum are labelled physics ({v['2rs']})")
    say(R['shells']['2rs'][1]['verdict'] == 'physics (within the chain cell-rule ambiguity)' and R['shells']['2rs'][3]['verdict'] == 'physics',
        f"comparator: the cell-rule ambiguity (2e-3) qualifies a small physics residual and not a large one ({R['shells']['2rs'][1]['verdict']}; {R['shells']['2rs'][3]['verdict']})")
    say([row['certified'] for row in R['shells']['r_h']] == [True, True, True, False], "comparator: the sweep end with certificate 2e-3 is uncertified")
    say(R['summary']['certified_sweep_end_readings'] == 12 and R['summary']['physics'] == 3, f"comparator: counts {R['summary']['certified_sweep_end_readings']} certified, {R['summary']['physics']} physics")
    c = R['clocks']['half_weight_1p5rs']
    say(abs(c['chain_rs'] - 2.0) < 1e-9 and c['agree'], f"comparator: half-weight clock of p beyond 1.5 r_s = {c['chain_rs']:.3f} r_s (expected 2.0), agreement {c['agree']}")
    d = R['detectors'][0]
    say(abs(d['peak_clock_difference_rs'] - 5.0 / rs) < 1e-9 and d['peak_clock_agree'] and d['peak_modulus_verdict'] == 'agree', f"comparator: detector peak clocks differ by {d['peak_clock_difference_rs']:.3f} r_s, moduli agree within the error")
    say(R['detectors'][1].get('cube_peak') is None, "comparator: a detector member absent from the cube run is reported without a verdict")
    say(all(row['verdict'] == 'agree' for row in R['current']), "comparator: the current rows agree when the cube's current equals the chain's")
    say(R['lattice_scale']['label'] is not None and 'angular channel' in R['lattice_scale']['label'], f"comparator: a physics residual on a long-wavelength column is labelled '{R['lattice_scale']['label']}'")
    say(R['summary']['fallback_indicated'].startswith('no'), "comparator: no fallback when the Gram defect is below the certificate bands")
    pair2 = json.loads(json.dumps(pair, default=CM._json_default))
    tmpd = tempfile.mkdtemp(prefix='o5_selftest_')
    out = os.path.join(tmpd, 'cmp')
    with open(out + '.json', 'w') as fh:
        json.dump(dict(results=[R]), fh, default=CM._json_default)
    text = CM.write_md([R], out + '.md', opts)
    say(os.path.exists(out + '.md') and 'labelled physics' in text and 'weight beyond 2rs' in text, f"comparator: md written ({len(text)} characters) with the physics label and the shell tables")
    low = text.lower()
    say(all(w not in low for w in ('escape', 'frozen', 'prediction')), "comparator: the md carries none of the ruled-out words")
    del pair2


def main():
    t1_harmonics()
    t2_chain()
    t3_identity()
    t4_lift()
    t5_comparator()
    print(f"[selftest] {len(FAILS)} failures" + ('' if not FAILS else ': ' + '; '.join(FAILS)), flush=True)
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
