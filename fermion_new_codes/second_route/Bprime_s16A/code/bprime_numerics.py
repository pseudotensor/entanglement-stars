"""Own degree-32 coupled Taylor action and independent eight-node sink integral."""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange


DEGREE = 32


def gauss8():
    k = np.arange(1, 8, dtype=np.float64)
    off = k/np.sqrt(4*k*k-1)
    matrix = np.diag(off, 1)+np.diag(off, -1)
    values, vectors = np.linalg.eigh(matrix)
    return (values+1)/2, vectors[0]**2


def gram(states, weight=None):
    states = np.asarray(states, dtype=np.complex128)
    flat = states.reshape(len(states), -1)
    if weight is None:
        return flat.conj()@flat.T
    weight = np.asarray(weight)
    if weight.ndim == 2 and states.ndim == 2:
        return flat.conj()@weight@flat.T
    weighted = states*np.broadcast_to(weight, states.shape[1:])[None]
    return flat.conj()@weighted.reshape(len(states), -1).T


def spectrum(value, Q0=None):
    value = np.asarray(value, dtype=np.complex128)
    anti = float(np.max(np.abs(value-value.conj().T)))
    hermitian = (value+value.conj().T)/2
    raw = np.linalg.eigvalsh(hermitian)
    result = {'anti_Hermitian_maximum_entry': anti, 'raw_Hermitian_part_eigenvalues': raw}
    if Q0 is not None:
        qvalues, qvectors = np.linalg.eigh((Q0+Q0.conj().T)/2)
        if np.any(qvalues <= 0):
            raise ValueError('Initial Gram is not positive definite')
        inverse_half = (qvectors/np.sqrt(qvalues)[None])@qvectors.conj().T
        transformed = inverse_half@hermitian@inverse_half
        result.update(Q0_eigenvalues=qvalues, Q0_inverse_half=inverse_half,
                      orthonormal_representation_eigenvalues=np.linalg.eigvalsh((transformed+transformed.conj().T)/2))
    return result


@njit(parallel=True, cache=True, fastmath=False)
def _accumulate(endpoint, quadrature, term, powers):
    # Disjoint column/entry tiles change scheduling only, with no reduction.
    entries = endpoint.shape[1]
    tile_size = 4096
    tiles_per_column = (entries+tile_size-1)//tile_size
    for tile in prange(endpoint.shape[0]*tiles_per_column):
        column = tile//tiles_per_column
        start = (tile % tiles_per_column)*tile_size
        stop = min(start+tile_size, entries)
        for entry in range(start, stop):
            value = term[column, entry]
            endpoint[column, entry] += value
            for node in range(quadrature.shape[0]):
                quadrature[node, column, entry] += powers[node]*value


def taylor_step(operator, states, delta, integrate=False, nodes=None, weights=None):
    states = np.ascontiguousarray(states, dtype=np.complex128)
    endpoint, term, scratch = states.copy(), states.copy(), np.empty_like(states)
    count = len(states)
    entries = states[0].size
    if integrate:
        if nodes is None or weights is None:
            nodes, weights = gauss8()
        quadrature = np.broadcast_to(states, (8,)+states.shape).copy()
        powers = np.ones(8)
    else:
        quadrature = np.empty((0,)+states.shape, dtype=np.complex128)
        powers = np.empty(0)
    for degree in range(1, DEGREE+1):
        operator.apply(term, out=scratch)
        scratch *= delta/degree
        term, scratch = scratch, term
        if integrate:
            powers *= nodes
        _accumulate(endpoint.reshape(count, entries), quadrature.reshape(len(quadrature), count, entries),
                    term.reshape(count, -1), powers)
    if not np.all(np.isfinite(endpoint)):
        raise ValueError('Taylor action produced nonfinite states')
    sample_grams = np.asarray([2*gram(value, operator.eta) for value in quadrature])
    increment = delta*np.tensordot(weights, sample_grams, axes=(0, 0)) if integrate else None
    return endpoint, increment, sample_grams


def truncation_bound(intervals, B, initial):
    z = np.asarray(intervals, dtype=np.float64)*B
    if np.any(z < 0) or not np.all(np.isfinite(z)):
        raise ValueError('Invalid Taylor interval or operator bound')
    remainder = np.asarray([math.exp(float(value))*float(value)**33/math.factorial(33) for value in z])
    initial_frobenius = float(np.linalg.norm(np.asarray(initial).ravel()))
    operator_error = math.expm1(float(np.sum(np.log1p(remainder))))
    epsilon = initial_frobenius*operator_error
    loss_bound = (2*initial_frobenius+epsilon)*epsilon
    return {'step_z': z, 'one_step_remainders': remainder, 'initial_Frobenius_norm': initial_frobenius,
            'endpoint_operator_error_bound': operator_error,
            'endpoint_block_error_bound': epsilon, 'loss_Gram_error_bound': loss_bound,
            'loss_Gram_bound_threshold': 1e-10, 'loss_Gram_bound_pass': bool(loss_bound < 1e-10),
            'scope': 'Analytic exact-arithmetic truncation bound; evaluated in binary64, not a floating-point interval certificate.'}


def step_plan(T, B, steps=None):
    T, B = float(T), float(B)
    if not np.isfinite(T) or T <= 0 or not np.isfinite(B) or B < 0:
        raise ValueError('Invalid positive endpoint or norm bound')
    n0 = max(1, math.ceil(T*B/4))
    chosen = n0 if steps is None else steps
    if type(chosen) is not int or chosen < 1:
        raise ValueError('Step count must be a positive integer')
    while True:
        times = T*np.arange(chosen+1, dtype=np.float64)/chosen
        times[0], times[-1] = 0., T
        intervals = np.diff(times)
        valid = np.all(intervals > 0) and np.all(intervals*B <= 4)
        if valid:
            return {'n0': n0, 'steps': chosen, 'step_adjustment': chosen-n0,
                    'times': times, 'intervals': intervals}
        if steps is not None:
            raise ValueError('Fixed refinement step count violates an actual interval gate')
        chosen += 1


def propagate(operator, initial, T, steps=None, integrate=True, progress=None):
    T, B = float(T), float(operator.B)
    plan = step_plan(T, B, steps)
    times, intervals, steps = plan['times'], plan['intervals'], plan['steps']
    states = np.ascontiguousarray(initial, dtype=np.complex128).copy()
    Q0 = gram(states)
    norms = [Q0]
    increments, samples, elapsed = [], [], []
    nodes, weights = gauss8()
    bound = truncation_bound(intervals, B, initial)
    if np.any(bound['step_z'] > 4):
        raise ValueError('Actual Taylor interval exceeds the frozen z<=4 limit')
    if not bound['loss_Gram_bound_pass']:
        raise ValueError('Analytic loss-Gram truncation bound is not below 1e-10')
    for index, delta in enumerate(intervals):
        started = time.perf_counter()
        states, increment, sample = taylor_step(operator, states, float(delta), integrate, nodes, weights)
        norms.append(gram(states))
        if integrate:
            increments.append(increment)
            samples.append(sample)
        elapsed.append(time.perf_counter()-started)
        if progress is not None:
            progress(index+1, steps, times[index+1], elapsed[-1])
    loss = Q0-norms[-1]
    sink = np.sum(increments, axis=0) if integrate else None
    return {'final_columns': states, 'Q0': Q0, 'loss_Gram': loss,
            'times': times, 'actual_intervals': intervals, 'step_remaining_norm_Grams': np.asarray(norms),
            'step_sink_Gram_increments': np.asarray(increments), 'gauss_sink_Gram_samples': np.asarray(samples),
            'integrated_sink_Gram': sink, 'gauss_nodes': nodes, 'gauss_weights': weights,
            'step_elapsed_seconds': np.asarray(elapsed), 'bound': bound,
            'method': {'degree': DEGREE, 'B': B, 'initial_n0': plan['n0'], 'step_adjustment': plan['step_adjustment'], 'steps': steps, 'nominal_delta': T/steps,
                       'endpoint': T, 'generator_applications': DEGREE*steps,
                       'maximum_step_z': float(np.max(bound['step_z'])), 'integrated_sink': integrate},
            'integral_minus_deficit': None if sink is None else sink-loss,
            'integral_closure_maximum_entry': None if sink is None else float(np.max(np.abs(sink-loss)))}


def protocol_plans(operator, initial, T, reference_norm_bound):
    if not np.isfinite(reference_norm_bound) or reference_norm_bound < 0:
        raise ValueError('Invalid reference-basis norm bound')
    coarse = step_plan(T, operator.B)
    fine = step_plan(T, operator.B, 2*coarse['steps'])
    plans = {}
    for name, plan in (('coarse', coarse), ('fine', fine)):
        bound = truncation_bound(plan['intervals'], operator.B, initial)
        d = bound['endpoint_operator_error_bound']
        product = d*(1+d)+d
        initial_norm_bound = bound['initial_Frobenius_norm']
        action_bound = float(reference_norm_bound)*initial_norm_bound*product
        if not bound['loss_Gram_bound_pass'] or not np.isfinite(action_bound) or action_bound >= 1e-10:
            raise ValueError('Pre-loop forward Gram or complete-action truncation bound failed')
        plans[name] = {**plan, 'forward_bound': bound,
                       'adjoint_operator_error_bound': d, 'product_operator_error_bound': product,
                       'full_coordinate_action_error_bound': action_bound,
                       'reference_norm_bound': float(reference_norm_bound),
                       'initial_norm_bound': initial_norm_bound,
                       'norm_bound_convention': 'Spectral norms conservatively bounded by supplied reference bound and actual initial Frobenius norm.'}
    return plans


def run_actions(operator, initial, T, reference_norm_bound, after_action=None, progress=None):
    """All four actions preflighted before the first physical recurrence."""
    plans = protocol_plans(operator, initial, T, reference_norm_bound)
    results = {'plans': plans}
    for pass_name in ('coarse', 'fine'):
        count = plans[pass_name]['steps']
        for direction in ('forward', 'adjoint'):
            name = pass_name+'_'+direction
            state = initial if direction == 'forward' else results[pass_name+'_forward']['final_columns']
            generator = operator if direction == 'forward' else operator.adjoint()
            callback = None if progress is None else lambda *args, label=name: progress(label, *args)
            result = propagate(generator, state, T, steps=count,
                               integrate=pass_name == 'coarse' and direction == 'forward', progress=callback)
            results[name] = result
            if after_action is not None:
                after_action(name, result)
    return results


def project_action(Q, U, reference_columns, returned_adjoint_rows):
    Q, U = np.asarray(Q), np.asarray(U)
    reference_columns, returned_adjoint_rows = np.asarray(reference_columns), np.asarray(returned_adjoint_rows)
    if Q.shape != (reference_columns.shape[1],)*2 or U.shape != (Q.shape[0], len(returned_adjoint_rows)) or reference_columns.shape[0] != returned_adjoint_rows.shape[1]:
        raise ValueError('Full-coordinate projection dimensions disagree')
    return Q@U-reference_columns.conj().T@returned_adjoint_rows.T


def projection_diagnostics(Y, U, forward_loss):
    Y, U = np.asarray(Y), np.asarray(U)
    raw = U.conj().T@Y
    hermitian = (raw+raw.conj().T)/2
    residual = Y-U@raw
    return {'B_raw': raw, 'B_Hermitian_part': hermitian, 'R': residual,
            'B_anti_Hermitian_maximum_entry': float(np.max(np.abs(raw-raw.conj().T))),
            'forward_Gram_anti_Hermitian_maximum_entry': float(np.max(np.abs(forward_loss-forward_loss.conj().T))),
            'projection_minus_direct_forward_Gram': raw-forward_loss,
            'projection_closure_maximum_entry': float(np.max(np.abs(raw-forward_loss)))}


def action_diagnostics(Y, U, eigenvalues, forward_loss):
    Y, U, eigenvalues = np.asarray(Y), np.asarray(U), np.asarray(eigenvalues)
    result = projection_diagnostics(Y, U, forward_loss)
    raw, hermitian, residual = result['B_raw'], result['B_Hermitian_part'], result['R']
    target_residual = Y-U*eigenvalues[None]
    rayleigh = Y-U*np.real(np.diag(raw))[None]
    block_error = float(np.linalg.norm(hermitian-np.diag(eigenvalues), 2))
    off_span = float(np.linalg.norm(residual, 2))
    result.update({
            'block_operator_norm_error': block_error, 'off_span_operator_norm': off_span,
            'beta': block_error+off_span, 'beta_pass': block_error+off_span <= 1e-4,
            'source_eigenpair_residual': target_residual,
            'source_eigenpair_residual_operator_norm': float(np.linalg.norm(target_residual, 2)),
            'source_eigenpair_residual_column_norms': np.linalg.norm(target_residual, axis=0),
            'own_Rayleigh_residual': rayleigh, 'own_Rayleigh_residual_column_norms': np.linalg.norm(rayleigh, axis=0),
            'signed_diagonal_loss_minus_source': np.real(np.diag(forward_loss))-eigenvalues})
    return result


def refinement_diagnostics(coarse_loss, fine_loss, coarse_Y, fine_Y, Q0):
    coarse, fine = spectrum(coarse_loss, Q0), spectrum(fine_loss, Q0)
    metrics = {'loss_Gram_maximum_entry_difference': float(np.max(np.abs(fine_loss-coarse_loss))),
               'raw_ordered_spectrum_maximum_difference': float(np.max(np.abs(fine['raw_Hermitian_part_eigenvalues']-coarse['raw_Hermitian_part_eigenvalues']))),
               'orthonormal_spectrum_maximum_difference': float(np.max(np.abs(fine['orthonormal_representation_eigenvalues']-coarse['orthonormal_representation_eigenvalues']))),
               'full_coordinate_action_operator_norm_difference': float(np.linalg.norm(fine_Y-coarse_Y, 2))}
    return {**metrics, 'all_passed': all(np.isfinite(value) and value <= 1e-8 for value in metrics.values()),
            'coarse_spectrum': coarse, 'fine_spectrum': fine,
            'signed_Gram_fine_minus_coarse': fine_loss-coarse_loss, 'action_fine_minus_coarse': fine_Y-coarse_Y}
