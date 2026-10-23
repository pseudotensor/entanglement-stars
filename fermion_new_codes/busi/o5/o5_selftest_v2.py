#!/usr/bin/env python
"""o5_selftest_v2.py -- O5's manufactured checks, version 2 (o5_selftest_v1 is imported, not edited): its checks (1) to (4) run unchanged (the
harmonics, the reconstructed chain operator, the fixed-cut identity, the lift); its comparator checks are rewritten on o5_compare_v2 with the 2
wrong premises corrected (the cell-rule qualification built from explicit residuals on both sides of the chain-rule interval; the fallback both
ways against the largest certified band); the declared-profile conventions are checked (o5_lift_v2 with a v1 profile reproduces o5_lift_v1 to
rounding; the v1 profile on the chain grid reproduces o5_chain_export_v1.gaussian_column); and, when --solve_path (or the environment's
O5_SOLVE_PATH) names the directory with page_ops_v6.py and engine_single_object_B_v1.py, the solve-operator path: o5_chain_export_v2's chain
(page_ops_v6.clocked_csr) equals v1's reconstruction to rounding on the test profile, its number-current operator equals
page_ops_v6.number_current_csr, HORIZON's launch column is unit, carries the unit +E spinor of the local symbol, the envelope
exp(-(r - R_c)^2/(2 sigma^2)) and the carrier exp(i k_0 r), the declared profile reproduces it, the KS override is reported as such, the lift of
it is consistent, and o5_operator_check_v1's functions return zero differences. Without the path those checks are SKIP lines.
Prints PASS/FAIL/SKIP lines and ends with '[selftest v2] N failures, K skipped'; exit code 0 when none of either (1 with failures, 2 with skips).
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_selftest_v1 as T1        # noqa: E402
import o5_chain_export_v1 as CH1   # noqa: E402
import o5_chain_export_v2 as CH2   # noqa: E402
import o5_compare_v2 as CM2        # noqa: E402
import o5_lift_v1 as LF1           # noqa: E402
import o5_lift_v2 as LF2           # noqa: E402
import o5_operator_check_v1 as OC  # noqa: E402

FAILS, SKIPS = [], []


def say(ok, msg):
    print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def skip(msg):
    print('SKIP ' + msg, flush=True)
    SKIPS.append(msg)


def run_v1_suite():
    """o5_selftest_v1's checks (1) to (4), unchanged; its comparator checks (5) are replaced below."""
    T1.FAILS.clear()
    T1.t1_harmonics()
    T1.t2_chain()
    T1.t3_identity()
    T1.t4_lift()
    FAILS.extend(T1.FAILS)


def _synthetic_pair(dG, amb=2e-3, operator=None):
    """Synthetic readings: every shell but 2 r_s at 0.3 of the error sum (agree); at 2 r_s the sweep ends 1, 2, 3 carry a residual beyond the error
    sum on the side of the chain's alternative rule (within the ambiguity), the same residual on the far side (not within it) and 4 error sums."""
    rs, S = 29.0, 4
    sweep_t = rs * np.arange(0, S + 1)
    times = 0.5 * np.arange(0, 58 * S + 1)
    p_ch = {'r_h': np.array([0.0, 0.3, 0.6, 0.8, 0.9]), '1.5rs': np.array([0.0, 0.1, 0.3, 0.5, 0.6]), '2rs': np.array([0.0, 0.05, 0.2, 0.4, 0.5]), '3rs': np.array([0.0, 0.0, 0.1, 0.2, 0.3])}
    d_ch = 1e-9 * np.ones(S + 1)
    d_cu = np.array([0.0, 1e-5, 1e-4, 5e-4, 2e-3])
    p_alt = {nm: p + amb for nm, p in p_ch.items()}
    p_cu = {}
    for nm, p in p_ch.items():
        q = p.copy()
        for s in range(1, S + 1):
            tot = CM2.weight_band(p[s], d_ch[s]) + CM2.weight_band(p[s], d_cu[s]) + dG
            if nm != '2rs':
                q[s] = p[s] + 0.3 * tot
            elif s == 1:
                q[s] = p[s] + tot + 0.5 * amb
            elif s == 2:
                q[s] = p[s] - tot - 0.5 * amb
            else:
                q[s] = p[s] + 4.0 * tot
        p_cu[nm] = q
    amp_ch = np.exp(-(times - 40.0) ** 2 / 50.0)
    amp_cu = 1.001 * np.exp(-(times - 45.0) ** 2 / 50.0)
    J_ch = np.array([0.0, 1e-2, 2e-2, 1e-2, 5e-3])
    op = operator or dict(source='page_ops_v6.clocked_csr (synthetic)', solve_code=True, page_ops_v6_md5='synthetic')
    return dict(name='synthetic', chain_json='c.json', lift_json='l.json', cube_json='k.json', rs=rs, sweep_t=sweep_t, radii=CM2.RADII, times=times,
                column=dict(Rc=20.0, sigma=8.0, ak0=0.1, kappa=1, E_mean=0.15, E_rms=0.05, spinor_mode='particle', profile=None, source='synthetic'),
                chain=dict(p=p_ch, p_alt=p_alt, delta=d_ch, J=J_ch, Jint_exact=p_ch['r_h'], dp_strict=p_ch['r_h'], wmax=2.0, amp_delta=1e-9 * np.ones(times.size),
                           lattice_scale=0.01, killing=None, r_h=22.25, identity_defect_exact_max=1e-13, prop_err_last=1e-9),
                cube=dict(p=p_cu, delta=d_cu, J=J_ch + 1e-6, bound=2.1, amp_delta=1e-4 * np.ones(times.size), lattice_scale=0.012, killing=None, r_h=22.25, r_h_direction=22.274, H=232,
                          napply=None, wall=None, energy_mean=0.15, energy_std=0.05, norm_drift=1e-13),
                lift=dict(gram_defect=dG, amp_lift_error=2e-3, W_lift=1.0, reduction=dict(default_is_closest=True), det_gram=[1e-4, 1e-4]),
                amps=[dict(label='o5det|s=+', sign=1, k=0.2, chain=amp_ch, cube=amp_cu), dict(label='o5det|s=-', sign=-1, k=-0.2, chain=amp_ch, cube=None)], axis=[],
                provenance=dict(operator=op, chain_script='o5_chain_export_v2.py', lift_script='o5_lift_v2.py', profile=dict(form='gaussian'), lift_has_profile=True, reflection_flagged=0))


def t5_comparator():
    opts = dict(chain_rule='midpoint', cert_max=1e-3, clock_tol_rs=1.0, lattice_scale_min=0.05, operator_check=None)
    R = CM2.compare_pair(_synthetic_pair(5e-4), opts)
    v = {nm: [row['verdict'] for row in R['shells'][nm]] for nm in CM2.RADII}
    say(all(x == 'agree' for x in v['r_h'] + v['1.5rs'] + v['3rs']), f"comparator v2: residuals at 0.3 of the error sum agree ({v['r_h']})")
    say(all(x.startswith('physics') for x in v['2rs']), f"comparator v2: residuals beyond the error sum are labelled physics ({v['2rs']})")
    say(v['2rs'][0] == 'physics (within the chain cell-rule ambiguity)' and v['2rs'][1] == 'physics' and v['2rs'][2] == 'physics',
        f"comparator v2: a residual beyond the error sum on the side of the chain's alternative rule is qualified by the cell-rule ambiguity; the same residual on the far side and a residual "
        f"of 4 error sums are not ({v['2rs'][0]}; {v['2rs'][1]}; {v['2rs'][2]})")
    rows = R['shells']['2rs']
    say(rows[0]['distance_to_chain_rule_interval'] <= rows[0]['error_sum'] and rows[1]['distance_to_chain_rule_interval'] > rows[1]['error_sum'],
        f"comparator v2: the stored distance to the chain-rule interval decides the qualification ({rows[0]['distance_to_chain_rule_interval']:.2e} <= {rows[0]['error_sum']:.2e}; "
        f"{rows[1]['distance_to_chain_rule_interval']:.2e} > {rows[1]['error_sum']:.2e})")
    say([row['certified'] for row in R['shells']['r_h']] == [True, True, True, False], "comparator v2: the sweep end with certificate 2e-3 is uncertified")
    say(R['summary']['certified_sweep_end_readings'] == 12 and R['summary']['agree'] == 9 and R['summary']['physics'] == 3,
        f"comparator v2: counts {R['summary']['certified_sweep_end_readings']} certified, {R['summary']['agree']} agree, {R['summary']['physics']} physics")
    c = R['clocks']['half_weight_1p5rs']
    say(abs(c['chain_rs'] - 2.0) < 1e-9 and c['agree'], f"comparator v2: half-weight clock of p beyond 1.5 r_s = {c['chain_rs']:.3f} r_s (expected 2.0), agreement {c['agree']}")
    d = R['detectors'][0]
    say(abs(d['peak_clock_difference_rs'] - 5.0 / 29.0) < 1e-9 and d['peak_clock_agree'] and d['peak_modulus_verdict'] == 'agree',
        f"comparator v2: detector peak clocks differ by {d['peak_clock_difference_rs']:.3f} r_s, moduli agree within the error")
    say(R['detectors'][1].get('cube_peak') is None, "comparator v2: a detector member absent from the cube run is reported without a verdict")
    say(all(row['verdict'] == 'agree' for row in R['current']), "comparator v2: the current rows agree when the cube's current equals the chain's")
    say('angular channel' in (R['lattice_scale']['label'] or ''), f"comparator v2: a physics residual on a long-wavelength column is labelled '{R['lattice_scale']['label']}'")
    sm = R['summary']
    cb = sm['certificate_band_certified']
    say(sm['fallback_indicated'].startswith('no') and sm['gram_defect_exceeds_every_certificate_band'] is False,
        f"comparator v2: no fallback when the Gram defect (5e-4) lies below the largest certified certificate band ({cb['max']:.2e})")
    say(cb['min'] < 5e-4 < cb['max'] and abs(cb['fraction_below_gram_defect'] - 0.75) < 1e-12 and cb['n_certified'] == 12,
        f"comparator v2: certified bands from {cb['min']:.2e} to {cb['max']:.2e}; the Gram defect exceeds the band at a fraction {cb['fraction_below_gram_defect']:.2f} of the 12 certified readings "
        f"(9: the first 2 sweep ends and 3 r_s at the third)")
    R2 = CM2.compare_pair(_synthetic_pair(5e-3), opts)
    sm2 = R2['summary']
    say(sm2['fallback_indicated'].startswith('yes') and sm2['gram_defect_exceeds_every_certificate_band'] is True and sm2['certificate_band_certified']['fraction_below_gram_defect'] == 1.0,
        f"comparator v2: the fallback is indicated when the Gram defect (5e-3) exceeds every certified certificate band (largest {sm2['certificate_band_certified']['max']:.2e})")
    say(sm['result_eligible'] is True and 'solve code' in sm['chain_operator'], "comparator v2: a chain export on the solve operator is result eligible")
    R3 = CM2.compare_pair(_synthetic_pair(5e-4, operator=dict(source=CM2.RECONSTRUCTED, solve_code=False)), opts)
    say(R3['summary']['result_eligible'] is False and 'not a result' in R3['summary']['sentence'], "comparator v2: a chain export on the reconstructed operator is not a result without the operator check")
    opts2 = dict(opts, operator_check=dict(file='check.json', max_entry_diff=1e-16, equal_to_rounding=True))
    R4 = CM2.compare_pair(_synthetic_pair(5e-4, operator=dict(source=CM2.RECONSTRUCTED, solve_code=False)), opts2)
    say(R4['summary']['result_eligible'] is True and 'shown equal' in R4['summary']['chain_operator'], "comparator v2: with the operator check's equality the reconstructed operator's export is result eligible")
    tmpd = tempfile.mkdtemp(prefix='o5_selftest_v2_')
    out = os.path.join(tmpd, 'cmp')
    with open(out + '.json', 'w') as fh:
        json.dump(dict(results=[R, R2]), fh, default=CM2._json_default)
    text = CM2.write_md([R, R2, R3], out + '.md', opts)
    say(os.path.exists(out + '.md') and 'labelled physics' in text and 'weight beyond 2rs' in text and 'distance to the chain-rule interval' in text and 'not a result' in text,
        f"comparator v2: md written ({len(text)} characters) with the physics label, the shell tables, the chain-rule distance column and the provenance")
    low = text.lower()
    say(all(w not in low for w in ('escape', 'frozen', 'prediction')), "comparator v2: the md carries none of the ruled-out words")


def t6_profile_conventions():
    r, N, v = T1.test_profile(n=120)
    m, kap, Rc, sigma, k0, cut, H = 0.2, 2, 30.0, 4.0, 0.2, 4.0, 48
    u, E = CH1.particle_spinor(k0, Rc, m, kap)
    psi0, info = CH1.gaussian_column(r, Rc, sigma, k0, cut, u)
    p_chain = np.sum(np.abs(psi0) ** 2, 1)
    prof = CH2.profile_lane(Rc, sigma, k0, info['Z'], cut)
    d = float(np.max(np.abs(CH2.radial_profile(r, prof)[:, None] * u[None, :] - psi0)))
    say(d <= 1e-15, f"the v1 profile dict evaluated on the chain grid reproduces o5_chain_export_v1.gaussian_column ({d:.1e})")
    S1 = LF1.lift_shell_column(Rc, sigma, k0, cut, u, info['Z'], H, kap, 0.5, 1.0, 1, p_chain=p_chain)
    S2 = LF2.lift_profile_column(prof, u, H, kap, 0.5, 1.0, 1, p_chain=p_chain)
    same = (np.array_equal(S1['coords'], S2['coords']) and float(np.max(np.abs(S1['values'] - S2['values']))) <= 1e-15
            and abs(S1['gram_defect'] - S2['gram_defect']) <= 1e-15 and abs(S1['W_lift'] - S2['W_lift']) <= 1e-14)
    say(same, "o5_lift_v2 with the v1 profile reproduces o5_lift_v1's lift to rounding (coords, values, W_lift, Gram defect)")
    ph = CH2.profile_horizon(20.0, 8.0, 2.0 * math.pi / 64.0, 0.05)
    say(abs(CH2.density_std(ph) - 8.0 / math.sqrt(2.0)) < 1e-12 and abs(CH2.density_std(prof) - sigma) < 1e-12,
        "density width: sigma/sqrt 2 for HORIZON's amplitude-width profile, sigma for v1's")
    lo, hi, half = LF2.support_bounds(ph, 1e-10)
    say(abs(half - 8.0 * math.sqrt(2.0 * math.log(1e10))) < 1e-9 and lo == max(0.0, 20.0 - half) and hi == 20.0 + half,
        f"the uncut profile's lift support ends where the amplitude falls below 1e-10 of its peak: half-width {half:.2f} a")
    w = CH2.envelope_weights(ph, np.arange(1.0, 101.0))
    expected = 0.5 * (1.0 + math.erf((0.5 - 20.0) / (8.0 / math.sqrt(2.0)) / math.sqrt(2.0)))
    say(abs(w['gaussian_weight_below_grid'] - expected) < 1e-12 and w['gaussian_weight_outside_cut'] == 0.0,
        f"the envelope weights of HORIZON's profile: density below r = 1/2 at R_c 20, sigma 8: {w['gaussian_weight_below_grid']:.2e}; outside the (absent) cut 0")
    w14 = CH2.envelope_weights(CH2.profile_horizon(14.0, 16.0, 2.0 * math.pi / 64.0, 0.05), np.arange(1.0, 101.0))
    say(w14['gaussian_weight_below_grid'] > 0.1, f"HORIZON's own sigma 16 at R_c 14 a would put {w14['gaussian_weight_below_grid']:.3f} of its density below r = 1/2 (the ruling against it on this record)")


def t7_operator_path(solve_path):
    try:
        OPS, ESB, info = CH2.load_solve_modules(solve_path)
    except Exception as e:  # noqa: BLE001
        say(False, f"solve code import from {solve_path}: {type(e).__name__}: {e}")
        return
    print(f"[selftest v2] solve code: page_ops_v6 md5 {info['page_ops_v6']['md5']}, engine_single_object_B_v1 md5 {info['engine_single_object_B_v1']['md5']}, "
          f"page_history_v6b placeholder {info['page_history_v6b_placeholder']}, HORIZON's pairs {[(round(k, 6), s) for k, s, _ in info['KS']]}", flush=True)
    r, N, v = T1.test_profile()
    m = 0.2
    for kap in (1, 2):
        c1 = CH1.RadialChain(r, N, v, m, kap)
        c2 = CH2.SolveChain(r, N, v, m, kap, OPS)
        d = float(np.max(np.abs(c1.H - c2.H)))
        say(d <= 1e-14, f"v2's chain (page_ops_v6.clocked_csr(chain_sparse)) equals v1's reconstruction to rounding, kappa {kap}: max entry difference {d:.1e}")
        say(c2.hermiticity_defect() == 0.0 and c2.particle_hole_defect() < 1e-13, f"v2's chain Hermitian ({c2.hermiticity_defect():.1e}) and particle-hole symmetric ({c2.particle_hole_defect():.1e}), kappa {kap}")
    roots = CH1.table_roots(r, N, v)
    r_h = roots[-1][0]
    c2 = CH2.SolveChain(r, N, v, m, 1, OPS)
    Qs, qd = c2.number_current(r_h)
    say(qd <= 1e-15 and np.count_nonzero(Qs) > 0, f"page_ops_v6.number_current_csr equals v1's i[h, P_>] at the cut {r_h:.3f} a: max entry difference {qd:.1e} ({np.count_nonzero(Qs)} nonzeros)")
    res, _, _ = OC.compare_operators(r, N, v, m, 1, OPS, R_c=r_h)
    say(res['max_entry_diff'] <= 1e-14 and res['lowest_positive_diff_max'] <= 1e-12 and res['number_current_max_entry_diff'] <= 1e-15,
        f"o5_operator_check_v1.compare_operators on the test profile: max entry difference {res['max_entry_diff']:.1e}, the 5 lowest positive eigenvalues differ by {res['lowest_positive_diff_max']:.1e}")
    k0, Rc = 2.0 * math.pi / 64.0, 30.0
    col = CH2.horizon_column(ESB, r, m, 1, Rc, k0, 8.0, N, v)
    psi0 = col['psi0']
    say(abs(col['norm'] - 1.0) < 1e-12 and psi0.shape == (r.size, 2), f"HORIZON's column is unit on the chain grid ({col['norm']:.12f}), site-major {psi0.shape}")
    say(col['ks_override'] is True and col['n_columns_built'] == 4 and col['info']['depth'] == 'annulus middle' and abs(col['info']['R_c'] - Rc) < 1e-12,
        f"(2 pi/64, sigma 8) is not one of HORIZON's pairs: KS override {col['ks_override']}, {col['n_columns_built']} columns built, the member at R_c {col['info']['R_c']:g}")
    col16 = CH2.horizon_column(ESB, r, m, 1, Rc, k0, 16.0, N, v)
    say(col16['ks_override'] is False and col16['n_columns_built'] == 12, f"(2 pi/64, sigma 16) is HORIZON's own pair: KS override {col16['ks_override']}, {col16['n_columns_built']} columns built")
    say(ESB.KS == tuple(tuple(x) for x in ESB.KS) and len(ESB.KS) == 3, f"the engine's KS tuple is restored after the calls ({len(ESB.KS)} pairs)")
    u = col['spinor']
    D = m * CH1.SZ + math.sin(k0) * CH1.SX + (1.0 / Rc) * CH1.SY
    E = math.sqrt(m * m + math.sin(k0) ** 2 + (1.0 / Rc) ** 2)
    say(float(np.linalg.norm(D @ u - E * u)) < 1e-12 and abs(np.linalg.norm(u) - 1.0) < 1e-12, "HORIZON's spinor is the unit +E eigenvector of the local symbol")
    u1, _ = CH1.particle_spinor(k0, Rc, m, 1)
    say(abs(abs(np.vdot(u1, u)) - 1.0) < 1e-12, f"v1's closed-form particle spinor equals the engine's up to a phase (|overlap| {abs(np.vdot(u1, u)):.12f})")
    dens = np.sum(np.abs(psi0) ** 2, 1)
    j = np.arange(20, 40)
    lhs = 64.0 * np.log(dens[j] / dens[j + 1])
    rhs = 2.0 * (r[j] - Rc) + 1.0
    say(float(np.max(np.abs(lhs - rhs))) < 1e-8, f"HORIZON's envelope is exp(-(r - R_c)^2/(2 sigma^2)), sigma the amplitude width (density-ratio check, max defect {np.max(np.abs(lhs - rhs)):.1e}); "
        "v1's default made sigma the density width (a factor sqrt 2 in width)")
    phs = np.angle(psi0[j + 1, 0] / psi0[j, 0])
    ref = np.angle(np.exp(-1.0j * k0 * r[j]) * psi0[j, 0] / u[0])
    say(float(np.max(np.abs(phs - k0))) < 1e-10 and float(np.max(np.abs(ref))) < 1e-10, f"the carrier advances k_0 per site and is referenced to r = 0 (exp(i k_0 r)): defects {np.max(np.abs(phs - k0)):.1e}, {np.max(np.abs(ref)):.1e}")
    say(col['declared_profile_max_diff'] <= 1e-13, f"the declared profile reproduces the engine's column to {col['declared_profile_max_diff']:.1e}")
    prof = col['profile']
    f_chain = CH2.radial_profile(r, prof)
    say(float(np.max(np.abs(f_chain[:, None] * u[None, :] - psi0))) <= 1e-13 and prof.get('cut') is None and prof['envelope_denominator'] == 2.0 and prof['carrier_reference'] == 0.0,
        "o5_lift_v2's radial_profile (the lift's evaluation) reproduces HORIZON's column on the chain grid; no cut, denominator 2 sigma^2, carrier referenced to the origin")
    S = LF2.lift_profile_column(prof, u, 72, 1, 0.5, 1.0, 1, p_chain=dens)
    say(abs(S['W_lift'] - 1.0) <= S['gram_defect'] + 2e-3, f"the lift of HORIZON's column (H 72): weight before renormalization {S['W_lift']:.6f}, |W - 1| = {abs(S['W_lift'] - 1.0):.2e} within the Gram defect {S['gram_defect']:.2e} (+2e-3)")
    cres = OC.compare_columns(r, N, v, m, 1, Rc, k0, 8.0, 5.0, ESB, CH1.RadialChain(r, N, v, m, 1), c2)
    say(abs(cres['spinor']['overlap_abs'] - 1.0) < 1e-12 and all(abs(row['E_mean_operator_diff']) < 1e-12 for row in cres['columns']) and len(cres['columns']) == 4,
        f"o5_operator_check_v1.compare_columns: 4 columns, every <h> equal on both operators to {max(abs(row['E_mean_operator_diff']) for row in cres['columns']):.1e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="O5 self-test v2")
    ap.add_argument('--solve_path', default=os.environ.get('O5_SOLVE_PATH', ''), help="the directory with page_ops_v6.py and engine_single_object_B_v1.py (the operator-path checks)")
    ap.add_argument('--skip_v1', action='store_true', help="do not rerun o5_selftest_v1's checks (1) to (4)")
    args = ap.parse_args(argv)
    if not args.skip_v1:
        run_v1_suite()
    t5_comparator()
    t6_profile_conventions()
    if args.solve_path:
        t7_operator_path(args.solve_path)
    else:
        skip("operator path: give --solve_path <dir with page_ops_v6.py and engine_single_object_B_v1.py> (or O5_SOLVE_PATH); the checks of v2's chain against the solve operator and of HORIZON's column did not run")
    print(f"[selftest v2] {len(FAILS)} failures, {len(SKIPS)} skipped" + ('' if not FAILS else ': ' + '; '.join(FAILS)), flush=True)
    return 0 if (not FAILS and not SKIPS) else (1 if FAILS else 2)


if __name__ == '__main__':
    sys.exit(main())
