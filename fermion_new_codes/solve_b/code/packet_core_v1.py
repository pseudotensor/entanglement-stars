"""Own finite radial packet operators; recipe-controlled execution only on remote CPU."""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, jv
from scipy.sparse import diags, eye, kron

S1 = np.array([[0, 1], [1, 0]], dtype=complex)
S2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
S3 = np.diag([1., -1.]).astype(complex)


def operators(radius, lapse, flow, kappa, mass=.2, spacing=1.):
    """Open-end two-component chain in site-major component order."""
    if any(np.iscomplexobj(x) for x in (radius, lapse, flow)):
        raise ValueError('real fields required')
    r, n, v = (np.asarray(x, dtype=float) for x in (radius, lapse, flow))
    if r.ndim != 1 or n.shape != r.shape or v.shape != r.shape:
        raise ValueError('one-dimensional matched radius/lapse/flow arrays required')
    if not all(np.isfinite(x).all() for x in (r, n, v)):
        raise ValueError('finite fields required')
    if not np.all(r > 0) or not np.all(n > 0):
        raise ValueError('positive radii and lapse required')
    if not np.isfinite(mass) or mass < 0 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('nonnegative finite mass and positive spacing required')
    if not np.allclose(np.diff(r), spacing, atol=0., rtol=0.):
        raise ValueError('declared exact uniform radial grid required')
    if np.ndim(kappa) != 0 or np.iscomplexobj(kappa) or not np.isfinite(kappa):
        raise ValueError('finite real scalar kappa required')
    size = len(r)
    if not size:
        raise ValueError('empty grid')
    shift = diags(np.ones(max(0, size-1)), 1, shape=(size, size), dtype=complex).tocsr()
    p = (shift-shift.getH())/(2j*spacing)
    shift2 = shift@shift
    qsite = (shift2-shift2.getH())/(4j*spacing)
    h0 = (kron(eye(size, format='csr'), mass*S3, format='csr')
          +kron(diags(kappa/r), S2, format='csr')+kron(p, S1, format='csr'))
    q = kron(qsite, eye(2, format='csr'), format='csr')
    nd = diags(np.repeat(n, 2), format='csr')
    vd = diags(np.repeat(v, 2), format='csr')
    hn = (nd@h0+h0@nd)/2
    hv = (vd@q+q@vd)/2
    return {'h0': h0, 'q': q, 'hn': hn.tocsr(), 'hv': hv.tocsr(),
            'h': (hn+hv).tocsr()}


def lower_band(matrix):
    """Hermitian bandwidth-four lower storage; reject unexpected couplings."""
    matrix = matrix.tocsr()
    rows, cols = matrix.nonzero()
    if np.any(np.abs(rows-cols) > 4):
        raise ValueError('unexpected operator bandwidth')
    band = np.zeros((5, matrix.shape[0]), dtype=complex)
    for off in range(5):
        diagonal = matrix.diagonal(-off)
        band[off, :len(diagonal)] = diagonal
    return band


def bare_packet(radius, centre, momentum, sigma=4., mass=.2, spacing=1., convention='amplitude'):
    """Finite Gaussian and homogeneous positive spinor before radial projection."""
    r = np.asarray(radius, dtype=float)
    if not all(np.isfinite(x) for x in (centre, momentum, sigma, mass, spacing)) or sigma <= 0:
        raise ValueError('finite preparation parameters and positive width required')
    if convention not in ('amplitude', 'probability'):
        raise ValueError('unknown Gaussian convention')
    denominator = (2 if convention == 'amplitude' else 4)*sigma*sigma
    p = np.sin(spacing*momentum)/spacing
    energy = np.hypot(mass, p)
    spinor = np.array([energy+mass, p], dtype=complex)/np.sqrt(2*energy*(energy+mass))
    scalar = np.exp(-(r-centre)**2/denominator+1j*momentum*(r-centre))
    raw = (scalar[:, None]*spinor[None, :]).reshape(-1)
    norm = float(np.vdot(raw, raw).real)
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError('unresolved raw packet norm')
    return raw/np.sqrt(norm), norm


def tail_degree(z, target=1e-14):
    if not np.isfinite(z) or z < 0:
        raise ValueError('finite nonnegative Chebyshev argument required')
    if z == 0:
        return 0, 0.
    for degree in range(4097):
        if degree+2 <= z/2:
            continue
        logtail = (np.log(2.)+z*z/(4*(degree+2))+(degree+1)*np.log(z/2)
                   -gammaln(degree+2)-np.log1p(-z/(2*(degree+2))))
        if logtail <= np.log(target):
            return degree, float(np.exp(logtail))
    raise ValueError('Chebyshev degree cap exceeded')


def exponential_action(h, states, duration, scale):
    if not np.isfinite(duration) or not np.isfinite(scale) or scale < 0:
        raise ValueError('invalid exponential parameters')
    states = np.asarray(states, dtype=complex)
    if not np.isfinite(states).all():
        raise ValueError('nonfinite states')
    if scale == 0:
        if h.nnz and np.any(h.data != 0):
            raise ValueError('nonzero H with zero scale')
        return states.copy(), {'L': 0., 'z': 0., 'degree': 0, 'tail': 0.}
    z = scale*abs(duration)
    degree, tail = tail_degree(z)
    sign = 1 if duration >= 0 else -1
    coefficients = jv(np.arange(degree+1), z)*(-1j*sign)**np.arange(degree+1)
    old = states
    result = coefficients[0]*old
    if degree:
        current = (h@old)/scale
        result = result+2*coefficients[1]*current
        for index in range(2, degree+1):
            new = 2*(h@current)/scale-old
            result += 2*coefficients[index]*new
            old, current = current, new
    if not np.isfinite(result).all():
        raise ValueError('nonfinite exponential action')
    return result, {'L': float(scale), 'z': float(z), 'degree': degree, 'tail': tail}


def construction_scale(radius, lapse, flow, kappa, mass=.2, spacing=1.):
    return float(2*(np.max(lapse)*(abs(mass)+abs(kappa)/np.min(radius)+1/spacing)
                    +np.max(np.abs(flow))/(2*spacing)))


def projected_packet(raw, positive_basis):
    norm = float(np.vdot(raw, raw).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('invalid preprojection norm')
    projected = positive_basis@(positive_basis.conj().T@raw)
    retained = float(np.vdot(projected, projected).real)/norm
    if not np.isfinite(retained) or retained <= 1e-12 or retained > 1+1e-10:
        raise ValueError('unresolved retained projection')
    return projected/np.sqrt(norm*retained), retained


def pure_distance_details(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('finite matched state vectors required')
    ln, rn = float(np.vdot(left, left).real), float(np.vdot(right, right).real)
    if not np.isfinite(ln) or not np.isfinite(rn) or ln <= 0 or rn <= 0:
        raise ValueError('positive finite state norms required')
    left_unit=left/np.sqrt(ln)
    right_unit=right/np.sqrt(rn)
    overlap = np.vdot(left_unit,right_unit)
    radicand = float(1-abs(overlap)**2)
    residual = right_unit-left_unit*overlap
    squared = float(np.vdot(residual, residual).real)
    if not np.isfinite(radicand) or not np.isfinite(squared) or radicand < -1e-12 or squared < -1e-12 or squared > 1+1e-12:
        raise ValueError('unavailable pure-state distance')
    return {'left_norm': ln, 'right_norm': rn, 'overlap_real': float(overlap.real),
        'overlap_imag': float(overlap.imag), 'raw_radicand': radicand,
        'raw_roundoff_clipped': radicand < 0, 'raw_derived_distance': float(np.sqrt(max(0., radicand))),
        'stable_squared_raw': squared, 'stable_roundoff_clipped': squared < 0 or squared > 1,
        'stable_distance': float(np.sqrt(min(1., max(0., squared))))}


def pure_distance(left, right):
    return pure_distance_details(left, right)['stable_distance']


def root_partition(radius, roots):
    r = np.asarray(radius)
    if roots is None or len(roots) != 2 or not np.isfinite(roots).all() or not roots[0] < roots[1]:
        return None
    return np.stack((r <= roots[0], (r > roots[0]) & (r <= roots[1]), r > roots[1]))


def applied_roots(radius, lapse, flow):
    r, n, v = (np.asarray(x) for x in (radius, lapse, flow))
    if any(np.iscomplexobj(x) or x.ndim != 1 or not np.isfinite(x).all() for x in (r,n,v)) or not (r.shape == n.shape == v.shape) or not len(r) or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(n <= 0):
        raise ValueError('invalid root fields')
    magnitude=abs(v)
    signs=np.where(n>magnitude,1,np.where(n<magnitude,-1,0))
    f = n-magnitude
    crossings, ambiguities = [], []
    for i in range(len(f)-1):
        if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1]:
            scale=max(abs(f[i]),abs(f[i+1]))
            left=abs(f[i])/scale;right=abs(f[i+1])/scale
            root=float(r[i]+left/(left+right)*(r[i+1]-r[i]))
            if not np.isfinite(root) or not r[i] <= root <= r[i+1]:
                ambiguities.append({'indices':[i,i+1],'radii':[float(r[i]),float(r[i+1])],
                    'kind':'unresolved_interpolation','left_sign':int(signs[i]),'right_sign':int(signs[i+1])})
                continue
            crossings.append({'radius':root,
                'bracket': [i,i+1], 'values': [float(f[i]),float(f[i+1])],
                'direction': 'outer' if signs[i] < 0 else 'inner'})
    zeros = np.flatnonzero(signs == 0)
    groups = np.split(zeros, np.flatnonzero(np.diff(zeros) != 1)+1) if len(zeros) else []
    for group in groups:
        left, right = int(group[0]), int(group[-1])
        if left == right and left > 0 and right+1 < len(f) and (signs[left-1] != 0 and signs[right+1] != 0 and signs[left-1] != signs[right+1]):
            crossings.append({'radius': float(r[left]), 'bracket': [left-1,right+1],
                'values': [float(f[left-1]),float(f[right+1])], 'direction': 'outer' if signs[left-1] < 0 else 'inner'})
        else:
            ambiguities.append({'indices': group.tolist(), 'radii': r[group].tolist(),
                'kind': 'plateau' if len(group) > 1 else 'tangency_or_endpoint',
                'left_sign': None if left == 0 else int(signs[left-1]),
                'right_sign': None if right+1 == len(f) else int(signs[right+1])})
    crossings.sort(key=lambda x:x['radius'])
    candidates = [x for x in crossings if x['direction'] == 'outer']
    selected = candidates[-1] if candidates else None
    reason = None
    interval = None
    if selected is None:
        reason = 'no resolved outer boundary'
    else:
        start = selected['bracket'][0]
        while start > 0 and signs[start-1] < 0:
            start -= 1
        interval = [start, selected['bracket'][-1]]
        if any(max(x['radii']) >= selected['radius'] for x in ambiguities):
            reason = 'ambiguous zero at or outside selected boundary'
        elif signs[-1] < 0:
            reason = 'outer trapped interval reaches wall'
        elif np.any(v[interval[0]:interval[1]+1] > 0):
            reason = 'outward flow in selected bracket or adjacent trapped interval'
    if signs[-1] < 0:
        reason='outer trapped interval reaches wall'
    valid_outer = selected is not None and reason is None
    pair = (len(crossings) == 2 and crossings[0]['direction'] == 'inner' and crossings[1]['direction'] == 'outer'
            and valid_outer and not ambiguities and signs[0] > 0 and signs[-1] > 0)
    return {'roots': [x['radius'] for x in crossings], 'crossings': crossings,
        'plateaus': [x['indices'] for x in ambiguities if x['kind'] == 'plateau'],
        'ambiguities': ambiguities, 'status': 'resolved_pair' if pair else 'ambiguous' if ambiguities or len(crossings) > 2 else 'absent_or_unpaired',
        'endpoint_values':[float(f[0]),float(f[-1])], 'endpoint_flow':[float(v[0]),float(v[-1])],
        'outer_available': bool(valid_outer), 'outer_root': selected['radius'] if valid_outer else None,
        'selected_outer': selected, 'selected_trapped_interval': interval, 'outer_reason': reason,
        'interior_ambiguity_irrelevant_to_outer': bool(valid_outer and ambiguities),
        'precision_policy': 'represented input signs; unrounded storage stability unestablished'}


def row_constant(radius, kappa, flow, mass=.2, spacing=1.):
    r = np.asarray(radius)
    ids = np.arange(len(r))
    d1 = (ids > 0).astype(int)+(ids+1 < len(r)).astype(int)
    d2 = (ids > 1).astype(int)+(ids+2 < len(r)).astype(int)
    gram = mass*mass+(kappa/r)**2+d1/(4*spacing*spacing)+d2*np.asarray(flow)**2/(16*spacing*spacing)
    return 2*np.sqrt(gram), d1, d2


def validate_history(records, fields):
    times = np.asarray(fields['T'])
    if np.iscomplexobj(times) or times.shape != (321,) or not np.array_equal(times, np.arange(321)*8.):
        raise ValueError('complete exact record clock required')
    if (len(records) != 321 or not np.array_equal([r['T'] for r in records], times)
        or not np.array_equal([r.get('T_over_rs') for r in records], times/64)):
        raise ValueError('JSON/fields clock association failed')
    radius = np.asarray(fields['R'])
    if np.iscomplexobj(radius) or not np.array_equal(radius, np.arange(1., 2049.)):
        raise ValueError('exact integer radial grid required')
    for key in ('N_applied', 'v_applied', 'eps_tot', 'J_tot'):
        x = fields[key]
        if np.iscomplexobj(x) or x.shape != (321, 2048) or not np.isfinite(x).all():
            raise ValueError('invalid applied/source field '+key)
    if not np.all(fields['N_applied'] > 0):
        raise ValueError('nonpositive applied lapse')
    for key, expected in [('g', .25), ('ell', 4), ('tau', 8.)]:
        if np.asarray(fields[key]).shape != () or np.iscomplexobj(fields[key]) or fields[key].item() != expected:
            raise ValueError('invalid scalar '+key)
    if records[97]['trapping_zeros'] != [records[97]['r_in'], records[97]['R_t']]:
        raise ValueError('formation root metadata mismatch')
    if root_partition(radius, records[97]['trapping_zeros']) is None:
        raise ValueError('formation root pair unavailable')
    return True


def variance_value(second, mean):
    raw = np.asarray(second)-np.asarray(mean)**2
    tolerance = 1e-12*np.maximum(1., np.asarray(second))
    return raw, np.where(raw >= -tolerance, np.maximum(raw, 0.), np.nan), raw < 0


def validate_roster(roster, expected):
    if roster != expected or len(set(tuple(x) for x in roster)) != len(roster):
        raise ValueError('packet roster mismatch')
    return True


def contraction_status(matrix):
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0):
        return 'invalid'
    return 'conditional_contraction' if np.linalg.norm(matrix, 2) <= 1+1e-12 else 'noncontraction'


def midpoint_fractions(count):
    if not isinstance(count, int) or count < 1:
        raise ValueError('positive integer midpoint count required')
    return (np.arange(count)+.5)/count


def fourier_diagnostics(states, spacing=1.):
    # Input shape M x 2 x packets. FFT order is retained; Nyquist is positive.
    size = states.shape[0]
    bins = np.rint(np.fft.fftfreq(size)*size).astype(int)
    if size % 2 == 0:
        bins[bins == -size//2] = size//2
    grid = 2*np.pi*bins/(size*spacing)
    transformed = np.fft.fft(states, axis=0, norm='ortho')
    probability = np.sum(abs(transformed)**2, axis=1)
    qsymbol = np.sin(2*grid*spacing)/(2*spacing)
    cosine = np.cos(2*grid*spacing)
    qsymbol[(4*bins) % size == 0] = 0.
    cosine[(8*abs(bins) == size) | (8*abs(bins) == 3*size)] = 0.
    masks = {'high': abs(grid) > 1.2/spacing,
             'negative_cosine': (size < 8*abs(bins)) & (8*abs(bins) < 3*size),
             'opposed_current': (size < 4*abs(bins)) & (2*abs(bins) < size),
             'Q_symbol': qsymbol, 'cosine_symbol': cosine}
    return grid, bins, probability, masks



def velocity_parts(op, radius):
    position = diags(np.repeat(radius, 2), format='csr')
    return tuple((1j*(op[key]@position-position@op[key])).tocsr() for key in ('hn', 'hv'))


PACKET_CODE_NAMES = ('packet_core_v1.py', 'control_PACKET_v1.py', 'run_PACKET_secondroute_v1.py',
    'read_PACKET_secondroute_v1.py', 'read_PACKET_sources_v1.py', 'summarize_PACKET_secondroute_v1.py', 'cpu_runtime.py')


def file_hash(path):
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_snapshot(spec):
    from pathlib import Path
    directory = Path(__file__).parent
    return {'spec_sha256': file_hash(spec), 'code_sha256': {name: file_hash(directory/name) for name in PACKET_CODE_NAMES}}


def verified_controls(path, spec):
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    if data.get('passed') is not True or data.get('source_bindings') != source_snapshot(spec):
        raise ValueError('controlled source/spec identity mismatch')
    return file_hash(path)


def propagation_step(hleft, hright, states, duration, substeps, scale_at_fraction, held=False):
    fractions = [0.] if held else midpoint_fractions(substeps)
    output = np.asarray(states).copy()
    metadata = []
    for index, fraction in enumerate(fractions):
        h = hleft if held else (1-fraction)*hleft+fraction*hright
        before_norm=np.sum(abs(output)**2,axis=0)
        output, row = exponential_action(h, output, duration if held else duration/substeps, scale_at_fraction(fraction))
        after_norm=np.sum(abs(output)**2,axis=0)
        metadata.append({'substep': index, 'fraction': float(fraction), 'before_norm':np.asarray(before_norm).tolist(),
            'after_norm':np.asarray(after_norm).tolist(), **row})
    return output, metadata


def held_ledger(before, after, hbefore, hafter):
    def quadratic(states, h):
        raw = np.sum(states.conj()*(h@states), axis=0)
        norm = np.sum(abs(states)**2, axis=0)
        if not np.isfinite(raw).all() or not np.isfinite(norm).all() or np.any(norm <= 0):
            raise ValueError('invalid held-work state')
        return raw, raw.real/norm, norm
    eraw, e, bn = quadratic(before, hbefore)
    fraw, f, an = quadratic(after, hbefore)
    nraw, ne, _ = quadratic(after, hafter)
    wraw, work, _ = quadratic(after, hafter-hbefore)
    return {'before_norm': bn, 'after_norm': an, 'frozen_energy_before': e, 'frozen_energy_after': f,
        'frozen_energy_defect': f-e, 'next_energy': ne, 'field_work': work, 'work_defect': ne-e-work,
        'frozen_before_raw_real': eraw.real, 'frozen_before_raw_imag': eraw.imag,
        'frozen_after_raw_real': fraw.real, 'frozen_after_raw_imag': fraw.imag,
        'next_raw_real': nraw.real, 'next_raw_imag': nraw.imag,
        'work_raw_real': wraw.real, 'work_raw_imag': wraw.imag,
        'work_raw_defect_real': (nraw-eraw-wraw).real, 'work_raw_defect_imag': (nraw-eraw-wraw).imag}


def spectral_admission(h, e, u, mass):
    residual = float(np.linalg.norm(h@u-u*e[None, :]))
    gram = u.conj().T@u
    gram.flat[::len(e)+1] -= 1
    orth = float(np.linalg.norm(gram))
    lam = float(np.max(abs(e)))
    bound = (np.sqrt(1+orth)*lam+residual)/np.sqrt(1-orth) if orth < 1 else None
    beta = residual+(bound+lam)*orth/(1+np.sqrt(1-orth)) if bound is not None else None
    tolerance = 64*len(e)*np.finfo(float).eps
    minimum = float(np.min(abs(e)))
    if (orth > tolerance or beta is None or not np.isfinite(beta) or minimum <= beta
        or minimum+beta < abs(mass) or np.count_nonzero(e > 0) != len(e)//2
        or np.count_nonzero(e < 0) != len(e)//2):
        raise ValueError('free projector/band gap unresolved')
    return {'residual_frobenius': residual, 'orthogonality_frobenius': orth, 'isometry_tolerance': tolerance,
        'edge': lam, 'edge_allowance': beta, 'norm_estimate': bound, 'nearest_zero': minimum,
        'negative_count': int(np.count_nonzero(e < 0)), 'positive_count': int(np.count_nonzero(e > 0)),
        'gap_consistent_with_mass': True, 'pairing_max_defect': float(np.max(abs(e+e[::-1]))), 'interval_certificate': False}


def comparison_association(left, right, left_labels, right_labels, primary_only=False):
    validate_observable_schema(left,len(left_labels))
    validate_observable_schema(right,len(right_labels))
    for x in (left, right):
        if not np.array_equal(x['record_indices'], np.arange(97,321)) or not np.array_equal(x['T'], np.arange(97,321)*8.):
            raise ValueError('comparison record/clock coverage mismatch')
    count = 16 if primary_only else len(left_labels)
    if len(right_labels) != count or len(left_labels) not in ((16,32) if primary_only else (count,)):
        raise ValueError('comparison cohort count mismatch')
    if left_labels[:count] != right_labels:
        raise ValueError('comparison label mismatch')
    return count


def readout_admission(values, allowed_missing=False):
    bad = ~np.isfinite(np.asarray(values))
    return {'admitted': bool(allowed_missing or not bad.any()), 'nonfinite_count': int(np.count_nonzero(bad)),
            'missingness_allowed': bool(allowed_missing)}


def admit_record_observables(values, position_probability, momentum_probability,
                            dense_available, applied_pair_available, applied_outer_available):
    count = position_probability.shape[1]
    failures = {}
    def add(name, bad):
        bad = np.asarray(bad, dtype=bool)
        if bad.any():
            failures[name] = np.argwhere(bad).tolist()
    for name, value in values.items():
        bad = ~np.isfinite(value)
        if name == 'dense_shares' and not dense_available:
            bad = np.zeros_like(bad)
        elif name == 'applied_shares' and not applied_pair_available:
            bad = bad.copy()
            bad[:2] = False
            if not applied_outer_available:
                bad[2] = False
        add('nonfinite_'+name, bad)
        if name in ('negative_band_weight','inside_12','first_four','wall_64','fixed_outer_share') or name.endswith('_share'):
            add('probability_range_'+name, (np.asarray(value)<-1e-10)|(np.asarray(value)>1+1e-10))
    for name, pp in [('position', position_probability), ('momentum', momentum_probability)]:
        add(name+'_range', ~np.isfinite(pp) | (pp < -1e-10) | (pp > 1+1e-10))
        add(name+'_normalization', abs(pp.sum(axis=0)-1) > 1e-10)
    for name, available in [('dense_shares', dense_available), ('applied_shares', applied_pair_available)]:
        if available:
            pp = values[name]
            add(name+'_range', (pp < -1e-10) | (pp > 1+1e-10))
            add(name+'_closure', abs(pp.sum(axis=0)-1) > 1e-10)
    if applied_outer_available:
        pp = values['applied_shares'][2]
        add('applied_outer_range', (pp < -1e-10) | (pp > 1+1e-10) | ~np.isfinite(pp))
    admitted = np.ones(count, dtype=bool)
    for locators in failures.values():
        for loc in locators:
            admitted[loc[-1]] = False
    return {'admitted_columns': admitted.tolist(), 'failures': failures,
        'dense_partition_available': bool(dense_available), 'applied_pair_available': bool(applied_pair_available),
        'applied_outer_available': bool(applied_outer_available)}


def validate_completed_states(complete, shape, labels):
    if (complete.get('records') != 224 or complete.get('intervals') != 223
        or complete.get('start_record') != 97 or complete.get('last_record') != 320
        or complete.get('T_first') != 776. or complete.get('T_last') != 2560.
        or complete.get('labels') != labels or complete.get('columns') != len(labels)
        or tuple(shape) != (224,4096,len(labels))):
        raise ValueError('saved state label/clock/shape association failed')
    return True


def packet_labels(channel, mode):
    if channel not in (1, 5) or mode not in ('E','I16','I32'):
        raise ValueError('unknown packet channel or rule')
    conventions = ['amplitude', 'probability'] if channel == 1 and mode != 'I32' else ['amplitude']
    return [[c, site, k] for c in conventions for site in ('centre','inner','mid','outer') for k in (-.45,-.3,.3,.45)]


def validate_step_association(record, next_record, h_record, next_h_record):
    if not (isinstance(record,int) and 97 <= record < 320 and next_record == record+1 and h_record == record and next_h_record == next_record):
        raise ValueError('shifted record-H association')
    return True


OBSERVABLE_SCALARS = tuple("""norm energy_H energy_0 energy_lapse energy_flow second_H second_0 variance_H_raw variance_H variance_H_clipped variance_0_raw variance_0 variance_0_clipped current_Q velocity_lapse velocity_flow velocity_total centroid position_variance_raw position_variance position_variance_clipped mean_N mean_v mean_N_minus_abs_v negative_band_weight fixed_outer_share inside_12 first_four wall_64 parseval_defect Q_symbol_mean cosine_mean packet_row_max_excess packet_cauchy_max_excess packet_row_max_adjusted_excess packet_cauchy_max_adjusted_excess radial_mean_band_excess radial_second_band_excess homogeneous_mean_excess H_mean_band_excess H_second_band_excess H_finer_bound H_finer_mean_excess H_finer_second_excess H_construction_bound high_share negative_cosine_share opposed_current_share""".split())+tuple(name+'_raw_'+part for name in ('energy_H','energy_0','energy_lapse','energy_flow','current_Q','velocity_lapse','velocity_flow') for part in ('real','imag'))


def validate_observable_schema(data,count):
    shapes={key:(224,count) for key in OBSERVABLE_SCALARS}
    shapes.update({'dense_shares':(224,3,count),'applied_shares':(224,3,count),
        'record_indices':(224,),'T':(224,),'momentum_grid':(2048,),'momentum_bins':(2048,)})
    if set(data) != set(shapes):
        raise ValueError('missing or unknown observable keys')
    for key,shape in shapes.items():
        if np.asarray(data[key]).shape != shape:
            raise ValueError('wrong observable shape '+key)
    return True
