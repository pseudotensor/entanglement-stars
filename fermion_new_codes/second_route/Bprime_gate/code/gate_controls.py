"""Guarded analytic controls for the own saved-Gram gate reader."""
from __future__ import annotations
import argparse
from pathlib import Path
from gate_runtime import admit, identity, require_identity, code_bindings, verify_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Bprime saved-output gate manufactured controls')
    spec = require_identity(args.spec, args.spec_sha256)
    import numpy as np
    import gate_algebra as algebra
    from gate_io import save_record
    verify_threads(runtime, 'after_imports')
    source = code_bindings()
    records, checks = {}, []

    def close(label, observed, expected):
        observed, expected = np.asarray(observed), np.asarray(expected)
        residual = observed-expected
        scale = max(1.0, float(np.max(np.abs(expected))))
        maximum = float(np.max(np.abs(residual)))
        checks.append({'label': label, 'passed': bool(np.all(np.isfinite(residual)) and maximum <= 5e-12*scale),
                       'observed': observed, 'expected': expected, 'signed_residual': residual,
                       'maximum': maximum, 'threshold': 5e-12*scale})

    def exact(label, observed, expected):
        checks.append({'label': label, 'passed': bool(observed == expected),
                       'observed': observed, 'expected': expected})

    def refused(label, function):
        try:
            function()
        except ValueError as error:
            checks.append({'label': label, 'passed': True, 'reason': str(error)})
        else:
            checks.append({'label': label, 'passed': False, 'reason': 'Expected refusal absent'})

    cross = np.diag([1, 1j])/4
    gram = np.block([[np.eye(2)/2, cross], [cross.conj().T, np.eye(2)/2]])
    blocks = algebra.partition_stats(gram, 2)
    records['complex_joint'] = {'G': gram, 'statistics': blocks}
    close('complex joint eigenvalues', blocks['joint']['eigenvalues'], [.25, .25, .75, .75])
    close('complex supply eigenvalues', blocks['supply']['eigenvalues'], [.5, .5])
    close('complex exit eigenvalues', blocks['exit']['eigenvalues'], [.5, .5])
    close('complex cross singular values', blocks['cross']['singular_values'], [.25, .25])
    close('complex joint trace', blocks['joint']['trace'], 2)
    close('joint differs from block union', blocks['joint_union_maximum']['max_abs'], .25)
    exact('top cluster counts both values', blocks['cross']['top_cluster_count'], 2)
    broken = gram.copy(); broken[0, 2] += .125j
    defective = algebra.spectral(broken)
    records['broken_hermiticity'] = {'G': broken, 'statistics': defective}
    exact('unequal triangle refused', defective['hermitian_admitted'], False)
    exact('failed triangle has no accepted spectrum', 'eigenvalues' in defective, False)

    epsilon = 2.0**-15
    q = np.array([1, 1j, -1, -1j])
    base = np.eye(4)/2
    shifted = base+epsilon*np.outer(q, q.conj())
    difference = algebra.partition_compare(base, shifted, 2)
    records['rank_one'] = {'base': base, 'comparison': shifted, 'statistics': difference}
    close('rank one entry maximum', difference['joint']['difference_maximum']['max_abs'], epsilon)
    close('rank one operator norm', difference['joint']['opnorm'], 4*epsilon)
    close('rank one complex delta', difference['joint']['delta'], epsilon*np.array(
        [[1,-1j,-1,1j],[1j,1,-1j,-1],[-1,1j,1,-1j],[-1j,-1,1j,1]]))
    close('rank one diagonal', difference['joint']['diagonal_delta'], np.full(4, epsilon))
    close('rank one joint dp', difference['joint']['dp'], [0,0,0,4*epsilon])
    for label in ('supply', 'exit'):
        close('rank one '+label+' dp', difference[label]['dp'], [0,2*epsilon])
        exact('rank one '+label+' threshold count', difference[label]['directions_within_1e4'], 2)
    close('rank one signed median', difference['joint']['signed_median_dp'], 0)
    exact('joint detects 3 of 4 within target', difference['joint']['directions_within_1e4'], 3)
    negative = algebra.compare(base, base-epsilon*np.eye(4))
    records['negative_uniform'] = negative
    close('negative signed median', negative['signed_median_dp'], -epsilon)
    for power, passed in ((34, True), (33, False)):
        perturbation = np.diag([2.0**(-power), 0, 0, 0])
        item = algebra.compare(base, base+perturbation)
        records['regression_power_'+str(power)] = item
        exact('regression threshold '+str(power), item['regression_1e10'], passed)

    boundary_delta = np.diag([1e-10, 0., 0., 0.])
    boundary = algebra.compare(np.zeros((4,4)), boundary_delta)
    boundary['inclusive_leq_indicator'] = boundary['difference_maximum']['max_abs'] <= 1e-10
    records['strict_regression_boundary'] = boundary
    exact('boundary inclusive indicator', boundary['inclusive_leq_indicator'], True)
    exact('exact regression boundary is refused', boundary['regression_1e10'], False)
    exact('boundary maximum equals stored threshold', boundary['difference_maximum']['max_abs'], 1e-10)
    imaginary_delta = 1j*2.0**-20*np.eye(4)
    anti = algebra.compare(np.zeros((4,4)), imaginary_delta)
    records['raw_antihermitian'] = {'comparison': imaginary_delta, 'statistics': anti,
                                    'hermitian_delta': (imaginary_delta+imaginary_delta.conj().T)/2}
    close('raw antiHermitian difference norm', anti['opnorm'], 2.0**-20)
    close('Hermitian part difference norm', algebra.opnorm((imaginary_delta+imaginary_delta.conj().T)/2), 0)
    exact('antiHermitian spectrum withheld', 'spectral_unavailable' in anti, True)

    dense_defect = .5*np.eye(4)+1j*2.0**-40*np.outer(q, q.conj())
    dense_stats = algebra.spectral(dense_defect)
    records['entry_small_norm_large_defect'] = {'G': dense_defect, 'statistics': dense_stats}
    close('dense defect entry maximum', dense_stats['hermiticity_max']['max_abs'], 2.0**-39)
    close('dense defect induced norm', dense_stats['hermiticity_opnorm'], 2.0**-37)
    exact('entrywise-small defect still refused', dense_stats['hermitian_admitted'], False)

    fractions = np.array([.01, .5, 1, .75])
    zero_errors = algebra.residuals(fractions, np.zeros(4))
    records['thresholds'] = zero_errors
    exact('dark equality unresolved', bool(zero_errors['dark_unresolved'][0]), True)
    exact('dark equality is not strictly dark', bool(zero_errors['dark'][0]), False)
    exact('half equality unresolved', bool(zero_errors['half_unresolved'][1]), True)
    exact('half equality not strictly above', bool(zero_errors['above_half'][1]), False)
    exact('zero residual/bar not resolved', bool(zero_errors['above_bar'][2]), False)
    broad = algebra.residuals(np.array([.75]), np.array([.25/1.34]))
    records['threshold_interval'] = broad
    for threshold in ('1e2','1e3','1e4'):
        exact('residual interval crosses '+threshold, bool(broad['r_unresolved_'+threshold][0]), True)
    ratios = algebra.onset_ratios(np.array([0., .125, .25]), np.array([0., 0., .125]))
    records['ratio_branches'] = ratios
    exact('zero over zero unavailable', bool(ratios['zero_over_zero'][0]), True)
    exact('positive over zero status', bool(ratios['positive_over_zero'][1]), True)
    close('finite ratio', ratios['ratio'][2], 2/1.34)

    # Repeated energy labels and identity qF cannot certify a phased basis.
    basis = np.diag([1, 1j, -1, -1j]).astype(complex)
    indices = np.array([2,0], dtype=np.int64)
    selected = basis[:, indices].copy()
    pairing = algebra.exact_basis(basis, selected, indices)
    phased = selected.copy(); phased[:,0] *= 1j
    wrong = algebra.exact_basis(basis, phased, indices)
    records['basis_mapping'] = {'basis': basis, 'selected': selected, 'phased': phased,
                                'repeated_energies': np.zeros(4), 'qF': np.eye(4),
                                'declared_map': pairing, 'undeclared_phase': wrong}
    exact('declared subset map passes', pairing['exact_equal'], True)
    exact('undeclared phase fails despite equal labels', wrong['exact_equal'], False)
    refused('duplicate subset map', lambda: algebra.exact_basis(basis, selected, np.array([0,0])))
    times = np.array([1.,2.,4.], dtype=np.float64)
    exact('exact clock', algebra.exact_clock(times, 2.), 1)
    refused('one ulp neighbour', lambda: algebra.exact_clock(times, np.nextafter(2., 3.)))
    refused('duplicate clock', lambda: algebra.exact_clock(np.array([2.,2.]), 2.))
    refused('missing endpoint', lambda: algebra.exact_clock(times, 8.))
    refused('changed file hash', lambda: require_identity(args.spec, '0'*64))
    if code_bindings() != source:
        raise ValueError('Source changed during controls')
    require_identity(args.spec, args.spec_sha256)
    verify_threads(runtime, 'before_save')
    report = {'stage': 'controls', 'spec': spec, 'source': source, 'runtime': runtime,
              'records': records, 'checks': checks, 'all_passed': all(c['passed'] for c in checks),
              'original_inputs_opened': False}
    save_record(args.outdir, 'controls', report)
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
