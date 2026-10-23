"""Own fixed-degree Taylor evolution, exact requested clocks and coherent norms."""
from __future__ import annotations

import math
import time

import numpy as np

from lead_operator import cross_gram


DEGREE = 32
_ACCUMULATE = None


def block_norm(states):
    """Induced state-block norm using its small complete column Gram."""
    gram = cross_gram(states)
    values = np.linalg.eigvalsh((gram+gram.conj().T)/2)
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite complete-Gram norm')
    return float(np.sqrt(max(0., float(values[-1])))) if len(values) else 0.


def matrix_norm(value):
    return float(np.linalg.norm(np.asarray(value), 2))


def metric_factor(initial):
    Q = cross_gram(initial)
    values, vectors = np.linalg.eigh((Q+Q.conj().T)/2)
    admitted = bool(np.all(np.isfinite(values)) and values[-1] > 0 and values[0] > 1e-12*values[-1])
    result = {'Q': Q, 'eigenvalues': values, 'admitted': admitted,
              'anti_Hermitian_maximum_entry': float(np.max(np.abs(Q-Q.conj().T)))}
    if admitted:
        result['inverse_half'] = (vectors/np.sqrt(values)[None])@vectors.conj().T
    return result


def normalize_comparison(states, factor):
    if not factor['admitted']:
        raise ValueError('Metric-normalized comparison is withheld; no column may be removed')
    return factor['inverse_half'].T@states


def make_plan(B, readout_times):
    B, requested = float(B), np.asarray(readout_times, dtype=np.float64)
    if not np.isfinite(B) or B < 0 or requested.ndim != 1 or not len(requested) or not np.all(np.isfinite(requested)) or requested[0] < 0 or requested[-1] <= 0 or np.any(np.diff(requested) <= 0):
        raise ValueError('Finite nonnegative bound and ordered exact clocks with positive endpoint required')
    times, counts = [0.], []
    left = 0.
    for right in requested:
        if right == 0:
            counts.append(0)
            continue
        count = max(1, math.ceil(float(right-left)*B/4))
        while True:
            local = left+(right-left)*np.arange(count+1, dtype=np.float64)/count
            local[0], local[-1] = left, right
            delta = np.diff(local)
            if np.all(delta > 0) and np.all(delta*B <= 4):
                break
            count += 1
        times.extend(local[1:].tolist()); counts.append(count)
        left = right
    times = np.asarray(times)
    return {'B': B, 'times': times, 'intervals': np.diff(times), 'readout_times': requested.copy(),
            'readout_indices': np.asarray([np.flatnonzero(times == value)[0] for value in requested], dtype=np.int64),
            'segment_step_counts': np.asarray(counts, np.int64), 'degree': DEGREE}


def bisect_plan(plan):
    coarse = np.asarray(plan['times'])
    middle = coarse[:-1]+np.diff(coarse)/2
    if np.any(middle <= coarse[:-1]) or np.any(middle >= coarse[1:]):
        raise ValueError('An actual interval has no representable midpoint')
    times = np.empty(2*len(coarse)-1)
    times[::2], times[1::2] = coarse, middle
    result = {**plan, 'times': times, 'intervals': np.diff(times),
              'readout_indices': 2*np.asarray(plan['readout_indices']),
              'segment_step_counts': 2*np.asarray(plan['segment_step_counts']),
              'refinement': 'Every actual coarse interval is bisected once.'}
    if np.any(result['intervals']*result['B'] > 4):
        raise ValueError('Bisected plan violates the strict interval gate')
    return result


def truncation_bound(plan, initial):
    z = np.asarray(plan['intervals'])*float(plan['B'])
    if np.any(z > 4) or np.any(z < 0) or not np.all(np.isfinite(z)):
        raise ValueError('Every actual Taylor interval must obey 0<=z<=4')
    remainders = np.asarray([math.exp(float(v))*float(v)**33/math.factorial(33) for v in z])
    delta = math.expm1(float(np.sum(np.log1p(remainders))))
    norm_bound = float(np.linalg.norm(np.asarray(initial).ravel()))
    gram_error = norm_bound**2*(2*delta+delta**2)
    return {'step_z': z, 'one_step_remainders': remainders, 'operator_error_bound': delta,
            'input_norm_upper_bound': norm_bound, 'input_norm_convention': 'Frobenius upper bound on induced norm',
            'state_block_error_bound': norm_bound*delta, 'complete_Gram_error_bound': gram_error,
            'bound_pass': bool(np.isfinite(gram_error) and gram_error <= 1e-10),
            'scope': 'Analytic exact-arithmetic truncation only; not a floating-point interval bound.'}


def _accumulator():
    global _ACCUMULATE
    if _ACCUMULATE is None:
        from numba import njit, prange

        @njit(parallel=True, cache=True, fastmath=False)
        def accumulate(endpoint, term):
            entries = endpoint.shape[1]
            tiles = (entries+4095)//4096
            for tile in prange(endpoint.shape[0]*tiles):
                column, start = tile//tiles, (tile % tiles)*4096
                for entry in range(start, min(start+4096, entries)):
                    endpoint[column, entry] += term[column, entry]
        _ACCUMULATE = accumulate
    return _ACCUMULATE


def taylor_step(operator, states, delta):
    if not np.isfinite(delta) or delta <= 0 or delta*operator.B > 4:
        raise ValueError('Frozen Taylor action requires positive actual delta with delta*B<=4')
    states = np.ascontiguousarray(states, dtype=np.complex128)
    if states.ndim != 2 or not len(states):
        raise ValueError('Taylor input requires a nonempty flat column block')
    endpoint, term, scratch = states.copy(), states.copy(), np.empty_like(states)
    for degree in range(1, DEGREE+1):
        operator.apply(term, out=scratch)
        scratch *= float(delta)/degree
        term, scratch = scratch, term
        _accumulator()(endpoint, term)
    if not np.all(np.isfinite(endpoint)):
        raise ValueError('Taylor evolution produced nonfinite amplitudes')
    return endpoint


def _step_record(operator, states, Q0, energy0, Hstates=None):
    Hstates = operator.h_apply(states) if Hstates is None else Hstates
    gram, energy = cross_gram(states), cross_gram(states, Hstates)
    record = {'Gram': gram, 'energy_Gram': energy,
              'Gram_difference_from_initial': gram-Q0,
              'Gram_difference_norm': matrix_norm(gram-Q0),
              'energy_difference_from_initial': energy-energy0,
              'scaled_energy_defect': matrix_norm(energy-energy0)/max(1., matrix_norm(energy0))}
    if not getattr(operator, 'is_unitary', True):
        record['signed_CAP_energy_rate'] = operator.energy_balance(states, hstates=Hstates)
        record['CAP_probability_loss_rate'] = 2*cross_gram(states*np.sqrt(np.repeat(operator.eta, 2))[None]) if hasattr(operator, 'N') else 2*cross_gram(states*np.sqrt(operator.eta)[None])
    indices = getattr(operator, 'end_collar_indices', None)
    if indices is not None:
        record['end_collar_Gram'] = cross_gram(states[:, indices])
    return record


def propagate(operator, F, readout_times, interval_plan=None, checkpoint=None, progress=None):
    """Checkpoint(clock, full_states, record) saves every requested complex block.

    Without a checkpoint callback, small controls retain readout_states in memory.
    Actual producers must provide their on-box streaming checkpoint callback.
    """
    states = np.ascontiguousarray(F, dtype=np.complex128)
    if states.ndim != 2 or not len(states) or not np.all(np.isfinite(states)):
        raise ValueError('Initial block must be nonempty finite complex columns')
    plan = make_plan(operator.B, readout_times) if interval_plan is None else interval_plan
    if plan['degree'] != DEGREE or plan['B'] != operator.B or not np.array_equal(plan['readout_times'], np.asarray(readout_times)):
        raise ValueError('Plan differs from the frozen degree, operator bound or exact requested clocks')
    if plan['times'][0] != 0 or np.any(np.diff(plan['times']) <= 0) or not np.array_equal(plan['intervals'], np.diff(plan['times'])):
        raise ValueError('Malformed actual interval plan')
    if not np.array_equal(plan['times'][plan['readout_indices']], np.asarray(readout_times)):
        raise ValueError('A requested clock is not an exact plan node')
    bound = truncation_bound(plan, states)
    if not bound['bound_pass']:
        raise ValueError('Accumulated pre-loop complete-Gram truncation bound exceeds1e-10')
    Hinitial = operator.h_apply(states)
    Q0, energy0 = cross_gram(states), cross_gram(states, Hinitial)
    records, readouts, snapshots, elapsed = [], [], [], []
    requested = {int(index): float(clock) for index, clock in zip(plan['readout_indices'], plan['readout_times'])}
    if 0 in requested:
        initial_record = _step_record(operator, states, Q0, energy0, Hstates=Hinitial)
        initial_record['clock'] = 0.
        initial_readout = {'clock': 0., 'diagnostics': initial_record}
        if checkpoint is None:
            snapshots.append(states.copy())
        else:
            initial_readout['checkpoint'] = checkpoint(0., states, initial_record)
        readouts.append(initial_readout)
    for index, delta in enumerate(plan['intervals'], 1):
        started = time.perf_counter()
        states = taylor_step(operator, states, float(delta))
        record = _step_record(operator, states, Q0, energy0)
        record['clock'] = float(plan['times'][index])
        records.append(record)
        if index in requested:
            readout = {'clock': requested[index], 'diagnostics': record}
            if checkpoint is None:
                snapshots.append(states.copy())
            else:
                readout['checkpoint'] = checkpoint(requested[index], states, record)
            readouts.append(readout)
        elapsed.append(time.perf_counter()-started)
        if progress is not None:
            progress(index, len(plan['intervals']), float(plan['times'][index]), elapsed[-1])
    unitary = getattr(operator, 'is_unitary', True)
    gram_defect = max(record['Gram_difference_norm'] for record in records)
    energy_defect = max(record['scaled_energy_defect'] for record in records)
    return {'final_columns': states, 'Q0': Q0, 'initial_energy_Gram': energy0, 'plan': plan,
            'bound': bound, 'step_diagnostics': records, 'readouts': readouts,
            'readout_states': snapshots if checkpoint is None else None,
            'step_elapsed_seconds': np.asarray(elapsed), 'is_unitary': unitary,
            'maximum_Gram_conservation_defect': gram_defect,
            'maximum_scaled_unitary_energy_defect': energy_defect if unitary else None,
            'unitary_numerical_gates_pass': bool(gram_defect <= 1e-8 and energy_defect <= 1e-8) if unitary else None,
            'generator_applications': DEGREE*len(plan['intervals']),
            'diagnostic_H_applications': len(plan['intervals'])+1,
            'total_Taylor_and_diagnostic_operator_applications': (DEGREE+1)*len(plan['intervals'])+1,
            'CAP_energy_conservation_test': 'not_applicable; signed anticommutator rate saved' if not unitary else None}


def compare_blocks(coarse, fine, energy_coarse=None, energy_fine=None, energy_initial=None):
    difference = fine-coarse
    record = {'state_error_operator_norm': block_norm(difference),
              'complete_Gram_error_operator_norm': matrix_norm(cross_gram(fine)-cross_gram(coarse))}
    record['state_and_Gram_pass'] = bool(record['state_error_operator_norm'] <= 1e-8 and record['complete_Gram_error_operator_norm'] <= 1e-8)
    if energy_coarse is not None:
        if energy_fine is None or energy_initial is None:
            raise ValueError('Energy refinement requires both energies and their common initial scale')
        record['scaled_energy_difference'] = matrix_norm(energy_fine-energy_coarse)/max(1., matrix_norm(energy_initial))
        record['energy_refinement_pass'] = record['scaled_energy_difference'] <= 1e-8
    return record


def finite_end_certificate(T, end_radius, Nmax, vmax, input_norm_bound=1., L=16., spacing=1.):
    """Use the single outward-rounded, reviewed endpoint certificate helper."""
    if L != 16. or spacing != 1.:
        raise ValueError('The frozen finite-end contract fixes L16 and a1')
    from lead_references import endpoint_bounds
    return endpoint_bounds(end_radius, T, Nmax, vmax, input_norm_bound, native=False)
