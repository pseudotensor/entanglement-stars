"""Own small-matrix reductions of saved Bprime Grams; no propagation."""
from __future__ import annotations

import numpy as np


ALGEBRA_TOL = 5e-12


def opnorm(matrix):
    values = np.linalg.svd(matrix, compute_uv=False)
    return float(values[0]) if values.size else 0.0


def maximum(values):
    """Lexicographically first absolute maximum, preserving complex values."""
    values = np.asarray(values)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Finite nonempty values required')
    magnitudes = np.abs(values)
    value = float(np.max(magnitudes))
    flat = int(np.argmax(magnitudes))
    index = tuple(int(x) for x in np.unravel_index(flat, values.shape))
    return {'max_abs': value, 'index': list(index), 'value': values[index].item(),
            'exact_ties': int(np.count_nonzero(magnitudes == value))}


def spectral(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Square Gram required')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Nonfinite Gram')
    defect = matrix-matrix.conj().T
    hermitian = (matrix+matrix.conj().T)/2
    scale = max(1.0, opnorm(matrix))
    defect_norm = opnorm(defect)
    passed = defect_norm <= ALGEBRA_TOL*scale
    result = {'shape': list(matrix.shape), 'finite_entries': int(matrix.size),
              'trace': np.trace(matrix).item(),
              'imaginary_diagonal_max': maximum(np.imag(np.diag(matrix))),
              'hermiticity_max': maximum(defect), 'hermiticity_opnorm': defect_norm,
              'hermitian_part_change_max': maximum(hermitian-matrix),
              'hermiticity_threshold': ALGEBRA_TOL*scale, 'hermitian_admitted': bool(passed)}
    if passed:
        eigenvalues = np.linalg.eigvalsh(hermitian)
        result.update(eigenvalues=eigenvalues, below_zero=eigenvalues[eigenvalues < 0],
                      below_zero_indices=np.flatnonzero(eigenvalues < 0),
                      below_zero_distances=-eigenvalues[eigenvalues < 0],
                      above_one=eigenvalues[eigenvalues > 1],
                      above_one_indices=np.flatnonzero(eigenvalues > 1),
                      above_one_distances=eigenvalues[eigenvalues > 1]-1,
                      trace_spectral_residual=np.trace(matrix)-np.sum(eigenvalues),
                      min_p=float(eigenvalues[0]), max_p=float(eigenvalues[-1]),
                      sum_p=float(np.sum(eigenvalues)), sum_one_minus_p=float(np.sum(1-eigenvalues)),
                      max_one_minus_p=float(np.max(1-eigenvalues)),
                      median_one_minus_p=float(np.median(1-eigenvalues)),
                      eigenvalues_below_1e2_count=int(np.count_nonzero(eigenvalues < 1e-2)))
    return result


def cross_stats(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError('Finite rectangular matrix required')
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0])
    resolution = ALGEBRA_TOL*max(1.0, largest)
    count = int(np.count_nonzero(largest-singular <= resolution))
    return {'singular_values': singular, 'opnorm': largest, 'maximum': maximum(matrix),
            'top_cluster_resolution': resolution, 'top_cluster_count': count,
            'top_unresolved': largest <= resolution,
            'last_included_gap': float(largest-singular[count-1]),
            'next_gap': float(largest-singular[count]) if count < len(singular) else None,
            'next_singular_value': float(singular[count]) if count < len(singular) else None}


def compare(reference, comparison, save_delta=True, hermitian=True):
    reference, comparison = np.asarray(reference), np.asarray(comparison)
    if reference.shape != comparison.shape:
        raise ValueError('Aligned matrices must have identical shapes')
    difference = comparison-reference
    peak = maximum(difference)
    loc = tuple(peak['index'])
    result = {'shape': list(difference.shape), 'finite_entries': int(difference.size), 'difference_maximum': peak, 'reference_at_max': reference[loc].item(),
              'comparison_at_max': comparison[loc].item(), 'opnorm': opnorm(difference),
              'frobenius_norm': float(np.linalg.norm(difference)),
              'regression_1e10': peak['max_abs'] < 1e-10,
              'elementwise_1e4': peak['max_abs'] <= 1e-4,
              'opnorm_1e4_indicator': opnorm(difference) <= 1e-4}
    if save_delta:
        result['delta'] = difference
    if not hermitian or difference.shape[0] != difference.shape[1]:
        result['singular_values'] = np.linalg.svd(difference, compute_uv=False)
        return result
    result['diagonal_delta'] = np.diag(difference).copy()
    first, second = spectral(reference), spectral(comparison)
    result['reference_raw_statistics'] = {k:v for k,v in first.items() if k != 'eigenvalues'}
    result['comparison_raw_statistics'] = {k:v for k,v in second.items() if k != 'eigenvalues'}
    result['reference_hermitian_admitted'] = first['hermitian_admitted']
    result['comparison_hermitian_admitted'] = second['hermitian_admitted']
    if first['hermitian_admitted'] and second['hermitian_admitted']:
        dp = second['eigenvalues']-first['eigenvalues']
        result.update(reference_p=first['eigenvalues'], comparison_p=second['eigenvalues'], dp=dp,
                      dp_within_1e4=(np.abs(dp) <= 1e-4), dp_maximum=maximum(dp), signed_median_dp=float(np.median(dp)),
                      directions_within_1e4=int(np.count_nonzero(np.abs(dp) <= 1e-4)),
                      spectral_1e4=bool(np.all(np.abs(dp) <= 1e-4)))
    else:
        result['spectral_unavailable'] = 'Raw Hermiticity admission failed'
    return result


def partition_stats(matrix, supply):
    n = len(matrix)
    if not 0 < supply < n:
        raise ValueError('Both window blocks required')
    pieces = {'supply': matrix[:supply, :supply], 'exit': matrix[supply:, supply:], 'joint': matrix}
    result = {key: spectral(value) for key, value in pieces.items()}
    result['cross'] = cross_stats(matrix[:supply, supply:])
    if all(result[key]['hermitian_admitted'] for key in pieces):
        union = np.sort(np.concatenate((result['supply']['eigenvalues'], result['exit']['eigenvalues'])))
        delta = result['joint']['eigenvalues']-union
        result['block_union'] = union
        result['joint_minus_union'] = delta
        result['joint_union_maximum'] = maximum(delta)
    return result


def partition_compare(reference, comparison, supply):
    result = {'joint': compare(reference, comparison)}
    for label, selection in (('supply', slice(None, supply)), ('exit', slice(supply, None))):
        result[label] = compare(reference[selection, selection], comparison[selection, selection], False)
    result['cross'] = compare(reference[:supply, supply:], comparison[:supply, supply:], False, False)
    return result


def residuals(p, dp):
    p, dp = np.asarray(p), np.asarray(dp)
    if p.shape != dp.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(dp)):
        raise ValueError('Finite same-rank spectra/differences required')
    r, bar = 1-p, 1.34*np.abs(dp)
    result = {'r': r, 'bar_estimate': bar, 'above_bar': r > bar, 'above_10bar': r > 10*bar,
              'dark': p < 1e-2, 'above_half': p > .5,
              'dark_unresolved': (p-bar <= 1e-2)&(p+bar >= 1e-2),
              'half_unresolved': (p-bar <= .5)&(p+bar >= .5)}
    for threshold, label in ((1e-2, '1e2'), (1e-3, '1e3'), (1e-4, '1e4')):
        result['r_above_'+label] = r > threshold
        result['r_unresolved_'+label] = (r-bar <= threshold)&(r+bar >= threshold)
    result['counts'] = {key: int(np.count_nonzero(value)) for key, value in result.items() if value.dtype.kind == 'b'}
    result.update(max_r=float(np.max(r)), median_r=float(np.median(r)), sum_r=float(np.sum(r)),
                  max_bar=float(np.max(bar)))
    return result


def onset_ratios(onset_dp, timestep_dp):
    numerator, denominator = np.abs(onset_dp), 1.34*np.abs(timestep_dp)
    if numerator.shape != denominator.shape or not np.all(np.isfinite(numerator+denominator)):
        raise ValueError('Finite same-rank differences required')
    valid = denominator > 0
    ratio = np.full(numerator.shape, np.nan)
    ratio[valid] = numerator[valid]/denominator[valid]
    result = {'numerator': numerator, 'denominator': denominator, 'ratio': ratio,
              'finite_ratio': valid, 'positive_over_zero': (~valid)&(numerator > 0),
              'zero_over_zero': (~valid)&(numerator == 0)}
    result['counts'] = {key: int(np.count_nonzero(value)) for key, value in result.items() if value.dtype.kind == 'b'}
    result['finite_count'] = int(np.count_nonzero(valid))
    if np.any(valid):
        result['finite_quantiles'] = np.quantile(ratio[valid], [0,.1,.5,.9,1])
        result['above_one_count'] = int(np.count_nonzero(ratio[valid] > 1))
        result['above_ten_count'] = int(np.count_nonzero(ratio[valid] > 10))
    return result


def exact_clock(times, target):
    times = np.asarray(times)
    if times.dtype != np.dtype('float64') or times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError('Finite binary64 clock vector required')
    positions = np.flatnonzero(times == target)
    if len(positions) != 1:
        raise ValueError('Required clock must have exactly one exact match')
    return int(positions[0])


def exact_basis(reference, comparison, indices):
    reference, comparison, indices = np.asarray(reference), np.asarray(comparison), np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices):
        raise ValueError('Distinct integer column mapping required')
    if np.any(indices < 0) or np.any(indices >= reference.shape[1]):
        raise ValueError('Column mapping out of range')
    selected = reference[:, indices]
    if selected.shape != comparison.shape or not np.all(np.isfinite(selected)) or not np.all(np.isfinite(comparison)):
        raise ValueError('Finite equally embedded basis columns required')
    return {'exact_equal': bool(np.array_equal(selected, comparison)), 'indices': indices.copy(),
            'finite_entries': int(selected.size), 'difference_maximum': maximum(comparison-selected)}
