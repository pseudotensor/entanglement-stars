"""Independent native free Fourier oracle and input-only endpoint certificates.

No foreign lane solver is used. Runtime admission precedes import. Functions
do no I/O; admitted entrypoints save every requested array on the box.
"""
from __future__ import annotations
import math
from decimal import Decimal, localcontext, ROUND_CEILING
import numpy as np


def _sinh_upper(x):
    return (x.exp().next_plus()-(-x).exp().next_minus())/Decimal(2)


def _decimal_upper(value):
    result = float(value)
    return None if not math.isfinite(result) else float(np.nextafter(result, math.inf))


def free_symbols(side):
    if type(side) is not int or side < 4 or side % 2:
        raise ValueError('Even periodic side at least four required')
    sine = np.sin(2*np.pi*np.fft.fftfreq(side))
    sine[0] = sine[side//2] = 0.  # exact mathematical sine zeros
    sx, sy, sz = sine[:, None, None], sine[None, :, None], sine[None, None, :]
    energy = np.sqrt(sx*sx+sy*sy+sz*sz)
    return sx, sy, sz, energy


def free_h_symbol(fourier, sine_axes):
    """Canonical tau_x tensor sigma_i action, independently expanded."""
    f = np.asarray(fourier)
    sx, sy, sz = sine_axes
    result = np.empty_like(f)
    result[0] = (sx-1j*sy)*f[3]+sz*f[2]
    result[1] = (sx+1j*sy)*f[2]-sz*f[3]
    result[2] = (sx-1j*sy)*f[1]+sz*f[0]
    result[3] = (sx+1j*sy)*f[0]-sz*f[1]
    return result


def fourier_reference(initial, clocks, observe=None, spectral_callback=None):
    """Yield exact-symbol, floating FFT states for one four-spinor column.

    FFT uses exp(-ikx) forward and exp(+ikx) inverse. Spectral measure is
    saved as exact mode energies and positive/negative weights, with the
    zero-energy subspace separately inventoried, rather than chosen bins.
    """
    initial = np.asarray(initial)
    if initial.ndim != 4 or initial.shape[0] != 4 or len(set(initial.shape[1:])) != 1:
        raise ValueError('One canonical four-spinor periodic cube required')
    if initial.dtype != np.complex128 or not np.all(np.isfinite(initial)):
        raise ValueError('Finite complex128 Fourier input required')
    clocks = np.asarray(clocks, dtype=float)
    if clocks.ndim != 1 or np.any(~np.isfinite(clocks)) or np.any(clocks < 0) or np.any(np.diff(clocks) <= 0):
        raise ValueError('Nonnegative strictly increasing clocks required')
    side = initial.shape[-1]
    sx, sy, sz, energy = free_symbols(side)
    f = np.fft.fftn(initial, axes=(1, 2, 3))
    hf = free_h_symbol(f, (sx, sy, sz))
    if spectral_callback is not None:
        density = np.sum(np.abs(f)**2, axis=0)/(side**3)
        signed = np.sum(np.conj(f)*hf, axis=0).real/(side**3)
        quotient = np.divide(signed, energy, out=np.zeros_like(signed), where=energy > 0)
        positive = (density+quotient)/2
        negative = (density-quotient)/2
        zero = energy == 0
        zero_indices = np.argwhere(zero)
        zero_weights = density[zero].copy()
        positive[zero] = negative[zero] = 0.
        spectral_callback({'energy': energy, 'positive_weight': positive,
                           'negative_weight': negative, 'zero_indices': zero_indices,
                           'zero_weight': zero_weights,
                           'initial_norm': np.asarray(np.sum(np.abs(initial)**2))})
        del density, signed, quotient, positive, negative, zero
    for clock in clocks:
        coefficient = np.empty_like(energy)
        np.divide(np.sin(energy*clock), energy, out=coefficient, where=energy > 0)
        coefficient[energy == 0] = clock
        evolved = np.cos(energy*clock)[None]*f-1j*coefficient[None]*hf
        state = np.fft.ifftn(evolved, axes=(1, 2, 3))
        result = state if observe is None else observe(state)
        yield float(clock), np.array(result, copy=True), {
            'clock': float(clock), 'periodic_side': side,
            'initial_norm': float(np.sum(np.abs(initial)**2)),
            'periodic_norm': float(np.sum(np.abs(state)**2)),
            'method': 'independent analytic free symbol and floating FFT; no phase alignment'}


def free_image_bound(side, input_radius, final_clock, block_norm=1.):
    if side < 256 or side & (side-1) or not 0 <= input_radius <= 32. or final_clock < 0 or block_norm < 0:
        raise ValueError('Frozen free-image bound domain required')
    q = math.exp(-float(side))
    factor = 24*(1+q)/(1-q)**3+2/(1-q)
    log_bound = input_radius+40.+final_clock*math.sinh(1.)-side+math.log(factor)
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        d = lambda x: Decimal.from_float(float(x))
        # q<=exp(-256)<1e-3 implies the displayed geometric factor<27.
        exponent = d(input_radius)+Decimal(40)+d(final_clock)*_sinh_upper(Decimal(1))-Decimal(side)
        upper = Decimal(27)*exponent.exp().next_plus()*d(block_norm)
        bound = _decimal_upper(upper)
        passed = upper <= Decimal('1e-10')
    return {'side': side, 'mu_a': 1., 'R_in': float(input_radius), 'R_obs': 40.,
            'clock': float(final_clock), 'input_block_norm': float(block_norm),
            'log_bound_unit_input_estimate': log_bound, 'bound': bound,
            'decimal_upper': str(upper), 'geometric_factor_upper': 27,
            'passed': bool(passed), 'roundoff_included': False}


def native_ball_observer(side, halfside=40):
    if side < 2*halfside+1:
        raise ValueError('Reference box does not contain common native ball')
    coordinates = np.arange(-halfside, halfside+1)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    site_mask = x*x+y*y+z*z <= halfside**2
    sl = slice(side//2-halfside, side//2+halfside+1)
    def observe(state):
        return state[:, sl, sl, sl][:, site_mask].reshape(-1)
    return observe, np.flatnonzero(site_mask.ravel())


def embed_periodic(native_column, side=256):
    native = np.asarray(native_column).reshape(4, 81, 81, 81)
    if side < 256 or side & (side-1):
        raise ValueError('Frozen Fourier reference size required')
    result = np.zeros((4, side, side, side), dtype=np.complex128)
    sl = slice(side//2-40, side//2+41)
    result[:, sl, sl, sl] = native
    return result


def endpoint_bounds(end_radius, final_clock, n_upper, v_abs_upper, block_norm=1., native=False):
    """Frozen weighted-link and complete range-two cut majorants.

    Supplied N/v maxima must be valid over every changing-weight bond in
    both compared operators; sample maxima alone are not certificates.
    """
    if min(end_radius, n_upper) <= 0 or min(final_clock, v_abs_upper, block_norm) < 0:
        raise ValueError('Finite nonnegative endpoint-bound data required')
    if not all(math.isfinite(x) for x in (end_radius, final_clock, n_upper, v_abs_upper, block_norm)):
        raise ValueError('Finite endpoint-bound data required')
    origin = 40. if native else 800.
    distance = end_radius-2.-origin
    if distance <= 0:
        raise ValueError('Complete endpoint alteration collar must be beyond flat-weight subspace')
    mus = (.25, .5, 1.) if native else (.0625, .125, .25, .5, 1.)
    rows, one_values, local_values = [], [], []
    with localcontext() as ctx:
        ctx.prec = 80
        ctx.rounding = ROUND_CEILING
        d = lambda x: Decimal.from_float(float(x))
        cut_decimal = d(n_upper)/2+d(v_abs_upper)/4
        cut_norm = _decimal_upper(cut_decimal)
        for mu in mus:
            md = d(mu)
            b = d(n_upper)*_sinh_upper(md)+d(v_abs_upper)*_sinh_upper(2*md)/2
            prefactor = d(final_clock)*cut_decimal*d(block_norm)
            one = prefactor*(-md*d(distance)+b*d(final_clock)).exp().next_plus()
            local = prefactor*(-2*md*d(distance)+b*d(final_clock)).exp().next_plus()
            one_values.append(one)
            local_values.append(local)
            rows.append({'mu_a': mu, 'b_mu_upper': _decimal_upper(b), 'distance': distance,
                         'one_way_state_bound': _decimal_upper(one),
                         'one_way_decimal_upper': str(one),
                         'local_round_trip_bound': _decimal_upper(local),
                         'local_round_trip_decimal_upper': str(local)})
        best_local_decimal = Decimal(6 if native else 1)*min(local_values)
        best_full_decimal = Decimal(6 if native else 1)*min(one_values)
        used_decimal = best_local_decimal if native else best_full_decimal
        best_local, best_full, used = map(_decimal_upper, (best_local_decimal, best_full_decimal, used_decimal))
    return {'native_six_face_sum': bool(native), 'end_radius': float(end_radius),
            'clock': float(final_clock), 'N_upper': float(n_upper),
            'v_abs_upper': float(v_abs_upper), 'cut_norm_upper': cut_norm,
            'input_block_norm': float(block_norm), 'candidates': rows,
            'best_local_round_trip_bound': best_local, 'best_one_way_state_bound': best_full,
            'used_bound': used, 'used_decimal_upper': str(used_decimal),
            'passed': bool(used_decimal <= Decimal('1e-10')),
            'meaning': 'exact-arithmetic weighted-generator bound; ordinary floating evaluation not included'}
