#!/usr/bin/env python3
"""Recover only the missing HZ5 fixture report; reuse every saved array."""
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from hold_endpoint_control import artifact_identity, tree_sum
from hold_error_bounds import downwards, gamma_rational, inflated_residuals, upwards
from hold_exact import held_operator, resource_check
from hold_precision_checks import matrix_bound
from hold_spectral_checks import gram_error
from horizon_bins import cut_intervals, local_symbol_extrema, weight_intervals
from horizon_exact import squared_norm
from hreduce_horizon_json import write_new_json


BASE = Path('<internal>/second_route_bulk')
OUT = BASE / 'horizon/hreduce_v1/fixture'
SAVED = OUT / 'horizon_fixture_arrays.npz'
ORIGINAL = BASE / 'checks/held_readouts/fixture_arrays.npz'
SAVED_SHA256 = '0dde937fbd8a24bf76f9bf6175921912fc34f887b5e191f66b9112ada6de2458'


def bins_from_saved(saved, prefix, norm):
    def array(name):
        return saved[prefix + name]
    raw = array('raw_bins')
    conditional = array('conditional_bins')
    report = dict(raw_bins=raw.tolist(), conditional_bins=conditional.tolist(),
        saved_weight_summation_upper=array('bin_weight_sum_error_upper').tolist(),
        raw_sum_minus_direct_norm=float(tree_sum(raw) - norm),
        conditional_sum_minus_one=float(tree_sum(conditional) - 1), buffers={})
    for factor in (1, 2):
        stem = f'buffer_{factor}_'
        union = float(array(stem + 'threshold_union_weight'))
        conditional_union = float(array(stem + 'threshold_conditional_union_weight'))
        mask = array(stem + 'union_mask')
        depth = math.ceil(math.log2(max(1, int(np.count_nonzero(mask)))))
        allowance = (gamma_rational(depth) * Fraction(union) / (1 - gamma_rational(depth))
                     if not prefix else None)
        report['buffers'][str(factor)] = dict(
            raw_bins=array(stem + 'raw_bins').tolist(),
            conditional_bins=array(stem + 'conditional_bins').tolist(),
            saved_weight_summation_upper=array(stem + 'bin_weight_sum_error_upper').tolist(),
            per_edge_threshold_weights=array(stem + 'threshold_edge_weights').tolist(),
            per_edge_conditional_threshold_weights=(array(stem + 'threshold_edge_weights') / norm).tolist(),
            threshold_union_weight=union, conditional_threshold_union_weight=conditional_union,
            union_weight_summation_upper=upwards(allowance) if allowance is not None else None,
            union_allowance_status=('recovered primary binary64 bound' if not prefix else
                'unrecovered and unused; original longdouble-to-binary64 conversion was not saved'),
            raw_closure=float(tree_sum(array(stem + 'raw_bins')) + union - norm),
            conditional_closure=float(tree_sum(array(stem + 'conditional_bins')) + conditional_union - 1))
    report['buffer_change'] = dict(nested_removal=bool(np.all(
        array('buffer_1_union_mask') <= array('buffer_2_union_mask'))))
    return report


def recovered_edge_scalars(eigenvalues, intervals, distances, spectral_error):
    if distances.shape != (len(intervals), len(eigenvalues)):
        raise ValueError('Saved edge-distance layout differs from the unchanged intervals')
    records = []
    for interval, distance in zip(intervals, distances, strict=True):
        low, high = Fraction(interval['lower']), Fraction(interval['upper'])
        center = float((low + high) / 2)
        cut_error = max(abs(Fraction(center) - low), abs(high - Fraction(center)))
        distance_error = gamma_rational(2) * (
            Fraction(float(np.max(np.abs(eigenvalues)))) + abs(Fraction(center)))
        rho = max(Fraction(0), Fraction(float(np.min(distance))) - cut_error - distance_error)
        error = Fraction(spectral_error)
        projector = min(Fraction(1), 2 * error / (rho - error)) if rho > error else Fraction(1)
        records.append(dict(**interval, distance_center=center,
            rho_effective_lower=downwards(rho),
            cut_expression_radius_upper=upwards(cut_error),
            distance_arithmetic_upper=upwards(distance_error),
            resolved_spectral_gap=bool(rho > error),
            halfline_projector_error_upper=upwards(projector)))
    return records


def lost_small_construction_scalars():
    # Exactly the original absent scalar controls; no new cases or resolution.
    radius = np.arange(100., 149.)
    clock = .9 + .0002 * radius
    flow = -.4 + .0001 * radius
    full = held_operator(radius, clock, flow, 1, .2)
    site_select = radius > 125.
    component_select = np.repeat(site_select, 2)
    direct = held_operator(radius[site_select], clock[site_select], flow[site_select], 1, .2)
    principal_error = float(sp.linalg.norm(direct - full[component_select][:, component_select]))
    sy = sp.csr_matrix([[0, -1j], [1j, 0]])
    anti = sp.kron(sp.diags((-1.)**np.arange(125, 148)), sy, format='csr')
    symmetry_error = float(sp.linalg.norm(anti @ direct.conjugate() @ anti.getH() + direct))
    q = .73
    n = 64
    mass, angular, local_clock, local_flow = .2, 1 / 3, 1.2, -.7
    energy = np.sqrt(mass**2 + np.sin(q)**2 + angular**2)
    spinor = np.array([1., (energy - mass) / (np.sin(q) - 1j * angular)])
    spinor /= np.linalg.norm(spinor)
    wave = (np.exp(1j * q * np.arange(n))[:, None] * spinor).reshape(-1)
    stencil = held_operator(np.full(n, 3.), np.full(n, local_clock), np.full(n, local_flow), 1, mass)
    symbol_energy = local_clock * energy + local_flow * np.sin(2 * q) / 2
    symbol_error = float(np.max(np.abs((stencil @ wave - symbol_energy * wave)[4:-4])))
    momentum = np.linspace(0, 2 * np.pi, 100001)
    extrema_errors, branch_errors, records = [], [], []
    for mass_value, angular_value, flow_value in ((.2, .03, -.3), (.2, .03, 0.), (0., 0., -.3), (0., 0., 0.)):
        result = local_symbol_extrema(mass_value, angular_value, 1.1, flow_value)
        positive = flow_value * np.sin(2 * momentum) / 2 + 1.1 * np.sqrt(
            mass_value**2 + angular_value**2 + np.sin(momentum)**2)
        negative = flow_value * np.sin(2 * momentum) / 2 - 1.1 * np.sqrt(
            mass_value**2 + angular_value**2 + np.sin(momentum)**2)
        extrema_errors.append(float(max(abs(result['maximum'] - max(positive.max(), negative.max())),
                                       abs(result['minimum'] - min(positive.min(), negative.min())))))
        branch_errors.append([float(abs(result['branches']['1']['minimum'] - positive.min())),
            float(abs(result['branches']['1']['maximum'] - positive.max())),
            float(abs(result['branches']['-1']['minimum'] - negative.min())),
            float(abs(result['branches']['-1']['maximum'] - negative.max()))])
        records.append(result)
    return dict(absolute_radius_principal_block_error=principal_error,
        antiunitary_error=symmetry_error, massive_plane_wave_symbol_error=symbol_error,
        local_symbol_extrema_grid_errors=extrema_errors,
        branchwise_symbol_extrema_grid_errors=branch_errors, local_symbol_records=records)


def main():
    if (OUT / 'horizon_fixture_checks.json').exists():
        raise FileExistsError('The missing report has already been recovered')
    if hashlib.sha256(SAVED.read_bytes()).hexdigest() != SAVED_SHA256:
        raise ValueError('The surviving failed-attempt arrays changed')
    paths = [SAVED, ORIGINAL, OUT.parent / 'execution/fixture_exit.json', Path(__file__),
             Path(__file__).with_name('hreduce_horizon_json.py')]
    paths += [BASE / 'code' / name for name in ('horizon_exact.py', 'horizon_bins.py',
        'hold_exact.py', 'hold_error_bounds.py', 'hold_precision_checks.py',
        'hold_endpoint_control.py', 'hold_spectral_checks.py')]
    bindings = [artifact_identity(path) for path in paths]
    write_new_json(OUT / 'fixture_recovery_input.json', dict(inputs=bindings,
        scope='HZ5 report-only recovery; no new transform, eigensolve, binning or array writes'))
    with np.load(SAVED, allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    with np.load(ORIGINAL, allow_pickle=False) as archive:
        e, u, column = archive['eigenvalues'], archive['eigenvectors'], archive['occupied'][:, 0]
        h = sp.csr_matrix(archive['h'])
    resource_check(len(e), dense=True)
    if len(e) != 96 or len(saved) != 69 or not np.array_equal(e, saved['eigenvalues']):
        raise ValueError('The inherited fixture or surviving array inventory differs')
    norm, norm_error = squared_norm(column)
    alpha_norm, alpha_error = squared_norm(saved['alpha'])
    residual = float(np.linalg.norm(h @ u - u * e))
    gram, _, _ = gram_error(u)
    bounds = inflated_residuals(len(e), int(np.max(np.diff(h.indptr))), matrix_bound(h),
                               float(np.max(np.abs(e))), residual, gram)
    intervals = cut_intervals(.2, 128., .25, 8192.)
    edges = recovered_edge_scalars(e, intervals, saved['edge_distances'], bounds['polar_spectral_error_upper'])
    bins = bins_from_saved(saved, '', norm)
    alternate_bins = bins_from_saved(saved, 'alternate_', norm)
    formal = weight_intervals(bins, edges, bounds['gram_F_upper'], len(e), norm, norm_error,
                              alpha_norm, alpha_error)
    cuts, widths = np.array([-2., -.5, .5, 2.]), np.array([.125, .0625, .0625, .125])
    boundary = saved['buffer_boundary_energies']
    selected, equality = saved['exact_buffer_selected'], saved['exact_buffer_equalities']
    exact_boundary = True
    for index in range(len(boundary)):
        for edge_index, (edge, width) in enumerate(zip(cuts, widths, strict=True)):
            distance = abs(Fraction(float(boundary[index])) - Fraction(float(edge)))
            exact_boundary = exact_boundary and bool(selected[edge_index, index]) == (distance < Fraction(float(width)))
            exact_boundary = exact_boundary and bool(equality[edge_index, index]) == (distance == Fraction(float(width)))
    cut_enclosures = True
    for row in intervals:
        low, high = Fraction(row['lower']), Fraction(row['upper'])
        raw = Fraction(row['raw_decision_boundary_numerator'], row['raw_decision_boundary_denominator'])
        cut_enclosures = cut_enclosures and low <= raw <= high
        cut_enclosures = cut_enclosures and low <= Fraction(row['ideal_lower']) <= Fraction(row['ideal_upper']) <= high
    alternate_error = float(np.max(np.abs(saved['conditional_bins'] - saved['alternate_conditional_bins'])))
    closures = [abs(bins['raw_sum_minus_direct_norm']), abs(bins['conditional_sum_minus_one'])]
    for factor in (1, 2):
        closures.extend((abs(bins['buffers'][str(factor)]['raw_closure']),
                         abs(bins['buffers'][str(factor)]['conditional_closure'])))
    nominal_contained = all(row['conditional_interval'][0] <= value <= row['conditional_interval'][1]
        for row, value in zip(formal['unbuffered'], bins['conditional_bins'], strict=True))
    small = lost_small_construction_scalars()
    write_new_json(OUT / 'fixture_recovery_scalars.json', dict(**small, direct_norm=norm,
        direct_norm_error_upper=norm_error, saved_alpha_norm=alpha_norm,
        saved_alpha_norm_error_upper=alpha_error, residual_F=residual, gram_F=gram,
        inflated_bounds=bounds, recovered_edge_scalar_records=edges,
        saved_unresolved_masks_retained=True, bins_from_saved=bins,
        alternate_bins_from_saved=alternate_bins, formal_weight_controls=formal))
    assignments_correct = np.array_equal(saved['synthetic_assignments'], np.array([0, 1, 2, 2, 3, 4, 4]))
    passed = bool(assignments_correct and exact_boundary and equality.sum() == 8
        and small['absolute_radius_principal_block_error'] < 1e-13
        and small['antiunitary_error'] < 1e-13 and small['massive_plane_wave_symbol_error'] < 1e-12
        and max(small['local_symbol_extrema_grid_errors']) < 1e-8
        and max(value for row in small['branchwise_symbol_extrema_grid_errors'] for value in row) < 1e-8
        and alternate_error < 1e-12 and max(closures) < 1e-12
        and bins['buffer_change']['nested_removal'] and nominal_contained and cut_enclosures)
    if bindings != [artifact_identity(path) for path in paths]:
        raise RuntimeError('A recovery input changed during use')
    write_new_json(OUT / 'horizon_fixture_checks.json', dict(passed=passed,
        exact_cut_assignments_correct=bool(assignments_correct),
        exact_buffer_boundary_comparisons_correct=bool(exact_boundary), exact_boundary_atoms=int(equality.sum()),
        ideal_and_binary_cut_enclosures_correct=bool(cut_enclosures), **small,
        alternate_conditional_bin_difference=alternate_error, normalization_errors=closures,
        formal_weight_controls=formal, input_bindings=bindings,
        recovery='saved arrays reused verbatim; only HZ5 lost scalar report calculations recovered',
        scope='massive stencil/symbol plus exact binary synthetic edges and existing saved-eigenpair transform; no HORIZON production result'))
    print('Recovered fixture report; passed=' + str(passed))
    if not passed:
        raise RuntimeError('The original fixture predicate failed on recovered report values')


if __name__ == '__main__':
    main()
