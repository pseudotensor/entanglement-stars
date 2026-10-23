"""Own manufactured channel-lead controls; no original numerical input is read."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import time
import traceback

from lead_runtime import (SPEC_SHA256, DERIVATION_SHA256, admit, check_memory,
                          code_bindings, core_bindings, identity, require_fresh,
                          require_identity, verify_threads)


class Recorder:
    def __init__(self, np):
        self.np, self.arrays, self.cases = np, {}, []

    def encode(self, value, key):
        np = self.np
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Manufactured records may not serialize objects')
            self.arrays[key] = value
            return {'array_key': key, 'shape': list(value.shape), 'dtype': str(value.dtype)}
        if isinstance(value, np.generic):
            return self.encode(value.item(), key)
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, dict):
            return {str(k): self.encode(v, key+'__'+str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(v, key+'__'+str(i)) for i, v in enumerate(value)]
        if isinstance(value, float) and not np.isfinite(value):
            return {'nonfinite_literal': repr(value)}
        return value

    def case(self, name, inputs, expected, observed, checks):
        row = {'name': name, 'inputs': self.encode(inputs, name+'__input'),
               'expected': self.encode(expected, name+'__expected'),
               'observed': self.encode(observed, name+'__observed'),
               'checks': {key: bool(value) for key, value in checks.items()}}
        row['passed'] = all(row['checks'].values())
        self.cases.append(row)
        return row

    def preserve(self, outdir, metadata):
        raw = outdir/'controls_partial_arrays.npz'
        self.np.savez_compressed(raw, **self.arrays)
        report = {**metadata, 'status': 'incomplete', 'cases': self.cases,
                  'output_bindings': {'partial_arrays': identity(raw)}}
        (outdir/'controls_partial.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def independent_native(np, N, flow, mass=0., mask=None, periodic=False):
    """Dense anticommutator construction, independent of own link caches/kernel."""
    n, sites = N.shape[0], N.size
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1., -1.])
    alpha = [np.kron(sx, s) for s in (sx, sy, sz)]
    beta = np.kron(sz, np.eye(2))
    result = (mass*np.kron(beta, np.diag(N.ravel()))).astype(np.complex128)
    ids = np.arange(sites).reshape(N.shape)
    for axis in range(3):
        shift = np.zeros((sites, sites), dtype=np.complex128)
        for coordinate in np.ndindex(N.shape):
            target = list(coordinate); target[axis] += 1
            if target[axis] == n:
                if not periodic:
                    continue
                target[axis] = 0
            shift[ids[coordinate], ids[tuple(target)]] = 1.
        p = (shift-shift.conj().T)/(2j)
        q = (shift@shift-shift.conj().T@shift.conj().T)/(4j)
        lapse, velocity = np.diag(N.ravel()), np.diag(flow[axis].ravel())
        result += np.kron(alpha[axis], (lapse@p+p@lapse)/2)
        result += np.kron(np.eye(4), (velocity@q+q@velocity)/2)
    if mask is not None:
        keep = np.tile(mask.ravel(), 4)
        result = keep[:, None]*result*keep[None]
    return result


def independent_chain(np, k, N, v, mass=0.):
    n = len(N)
    shift = np.diag(np.ones(n-1), 1)
    p, q = (shift-shift.T)/(2j), (shift@shift-shift.T@shift.T)/(4j)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1., -1.])
    h0 = np.kron(p, sx)+np.kron(np.diag(k/np.arange(1, n+1)), sy)+mass*np.kron(np.eye(n), sz)
    lapse, velocity = np.kron(np.diag(N), np.eye(2)), np.kron(np.diag(v), np.eye(2))
    return (lapse@h0+h0@lapse)/2+(velocity@np.kron(q, np.eye(2))+np.kron(q, np.eye(2))@velocity)/2


def fourier_oracle_case(r, np, num, expm):
    """Author/repair fixed side4 actual-oracle path against dense definitions."""
    from lead_references import fourier_reference, free_symbols, free_h_symbol
    n, clocks = 4, np.array([0., .125, .7])
    N, flow = np.ones((n,)*3), np.zeros((3, n, n, n))
    H = independent_native(np, N, flow, periodic=True)
    columns = np.zeros((4, 4, n, n, n), dtype=np.complex128)
    coordinates = np.indices((n,)*3)
    columns[0, 0] = 1/8
    columns[1, 1] = (-1.)**(coordinates[0]+coordinates[2])/8
    columns[2, :, 0, 0, 0] = np.array([1, 1j, 1, -1j])/2
    columns[3] = (columns[0]+1j*columns[2])/np.sqrt(2)
    F = columns.reshape(4, -1)
    exact = np.array([(expm(-1j*clock*H)@F.T).T for clock in clocks])
    observed, spectra = [], []
    for column in columns:
        measures = []
        trajectory = list(fourier_reference(column, clocks, spectral_callback=measures.append))
        if len(measures) != 1 or [item[0] for item in trajectory] != clocks.tolist():
            raise ValueError('Manufactured Fourier callback/clock shape changed')
        observed.append([item[1].reshape(-1) for item in trajectory])
        spectra.append(measures[0])
    observed = np.asarray(observed).transpose(1, 0, 2)
    energies, vectors = np.linalg.eigh(H)
    coefficients = F@vectors.conj()
    branch_masks = (energies > 5e-12, energies < -5e-12, np.abs(energies) <= 5e-12)
    dense_weights = np.array([np.sum(np.abs(coefficients[:, mask])**2, axis=1) for mask in branch_masks]).T
    actual_weights = np.array([[np.sum(s['positive_weight']), np.sum(s['negative_weight']),
                                np.sum(s['zero_weight'])] for s in spectra])
    analytic_weights = np.array([[0., 0., 1.], [0., 0., 1.],
                                  [7/16, 7/16, 1/8], [7/32, 7/32, 9/16]])
    sx, sy, sz, energy = free_symbols(n)
    symbol_HF = np.array([np.fft.ifftn(free_h_symbol(np.fft.fftn(column, axes=(1, 2, 3)),
                                                  (sx, sy, sz)), axes=(1, 2, 3)).reshape(-1)
                         for column in columns])
    exact_HF = (H@F.T).T
    Q = F.conj()@F.T
    observed_Grams = np.array([block.conj()@block.T for block in observed])
    exact_Grams = np.array([block.conj()@block.T for block in exact])
    dense_bound = np.linalg.norm(H, 2)
    state_threshold = 5e-12*max(1., dense_bound*num.block_norm(F))
    symbol_threshold = 5e-12*max(1., np.linalg.norm(H, 2)*num.block_norm(F))
    gram_threshold = 5e-12
    selected_modes = np.array([[0, 0, 1], [0, 0, 3]])
    actual_mode_weights = np.array([[spectra[2]['positive_weight'][tuple(mode)],
                                     spectra[2]['negative_weight'][tuple(mode)]] for mode in selected_modes])
    expected_mode_weights = np.array([[1/64, 0.], [0., 1/64]])
    actual_mode_energies = np.array([energy[tuple(mode)] for mode in selected_modes])
    r.case('actual_free_Fourier_oracle_dense_zero_modes_and_coherence',
           {'side': n, 'clocks': clocks, 'columns': columns, 'H_dense': H,
            'zero_projector_cutoff': 5e-12, 'exact_nonzero_spectral_gap': 1.,
            'selected_modes': selected_modes},
           {'states': exact, 'H_initial': exact_HF, 'Grams': exact_Grams,
            'branch_weights': analytic_weights, 'dense_projector_branch_weights': dense_weights,
            'initial_Gram': Q, 'state_threshold': state_threshold,
            'selected_mode_weights': expected_mode_weights, 'selected_mode_energies': np.ones(2),
            'dense_branch_ranks': np.array([112, 112, 32]),
            'symbol_threshold': symbol_threshold, 'Gram_threshold': gram_threshold},
           {'states': observed, 'H_initial': symbol_HF, 'Grams': observed_Grams,
            'branch_weights': actual_weights, 'mode_measures': spectra,
            'selected_mode_weights': actual_mode_weights, 'selected_mode_energies': actual_mode_energies,
            'dense_branch_ranks': np.array([np.sum(mask) for mask in branch_masks]),
            'dense_eigenvalues': energies, 'dense_eigenvectors': vectors,
            'dense_coefficients': coefficients, 'initial_Gram': Q,
            'state_residuals': observed-exact, 'symbol_residual': symbol_HF-exact_HF},
           {'complete_complex_states_without_phase_alignment': max(num.block_norm(a-b) for a, b in zip(observed, exact)) <= state_threshold,
            'actual_free_h_symbol': num.block_norm(symbol_HF-exact_HF) <= symbol_threshold,
            'complete_coherent_Grams': max(num.matrix_norm(a-b) for a, b in zip(observed_Grams, exact_Grams)) <= gram_threshold,
            'dense_projector_analytic_weights': np.max(np.abs(dense_weights-analytic_weights)) <= 5e-12,
            'actual_callback_branch_weights': np.max(np.abs(actual_weights-dense_weights)) <= 5e-12,
            'dense_branch_ranks': [int(np.sum(mask)) for mask in branch_masks] == [112, 112, 32],
            'signed_mode_labels': np.max(np.abs(actual_mode_weights-expected_mode_weights)) <= 5e-12,
            'signed_mode_energies': np.max(np.abs(actual_mode_energies-1.)) <= 5e-12,
            'Parseval': np.max(np.abs(np.sum(actual_weights, axis=1)-np.real(np.diag(Q)))) <= 5e-12,
            'exact_zero_mode_indices': all(len(s['zero_indices']) == 8 for s in spectra),
            'both_signed_branches_excited': bool(np.all(actual_weights[2:, :2] > .2)),
            'coherent_metric_not_discarded': bool(np.abs(Q[0, 2]) > .01)})


def algebra_cases(r, np, native, channel, num):
    n = 5
    points = np.indices((n,)*3)-n//2
    N = 1.+(points[0]+2*points[1]-points[2])/64
    v = np.array([points[1]/32, -points[2]/64, points[0]/64])
    mask = np.sum(points*points, axis=0) <= 4
    op = native.NativeHamiltonian(N, v, mass=.25, mask=mask)
    H = independent_native(np, N, v, .25, mask)
    observed = op.h_apply(np.eye(4*n**3, dtype=np.complex128)).T
    probes = np.random.default_rng(1701).normal(size=(3, len(H)))+1j*np.random.default_rng(1702).normal(size=(3, len(H)))
    valley_residuals, spin_residuals = [], []
    alpha, beta, _ = native.spin_matrices()
    for nu in itertools.product((0, 1), repeat=3):
        image = op.valley_image(probes, nu)
        valley_residuals.append(op.h_apply(image)-op.valley_image(op.h_apply(probes), nu))
        S = native.valley_matrix(nu)
        spin_residuals.extend([S.conj().T@alpha[i]@S-(-1)**nu[i]*alpha[i] for i in range(3)])
        spin_residuals.append(S.conj().T@beta@S-beta)
    threshold = 5e-12*max(1., op.B*num.block_norm(probes))
    r.case('native_open_variable_bonds_and_eight_valleys', {'N': N, 'v': v, 'mask': mask, 'mass': .25, 'probes': probes},
           {'H_dense': H, 'valley_residual': 0., 'scaled_threshold': threshold},
           {'H_apply': observed, 'valley_residuals': np.asarray(valley_residuals), 'spin_residuals': np.asarray(spin_residuals)},
           {'dense_H': np.linalg.norm(observed-H) <= 5e-12*max(1., op.B),
            'eight_native_valleys': max(num.block_norm(x) for x in valley_residuals) <= threshold,
            'canonical_spin': np.max(np.abs(spin_residuals)) <= 5e-12})
    n = 6  # Even periodic side is required by checkerboard valley wrap phases.
    points = np.indices((n,)*3)-n//2
    velocity = np.array([1/8, -1/4, 1/16])
    free = native.NativeHamiltonian(np.ones((n,)*3), np.broadcast_to(velocity[:, None, None, None], (3, n, n, n)), mass=.25, periodic=True)
    momenta = 2*np.pi*np.array([1, 2, 0])/n
    spinor = np.array([1, 1j, -.5, .25j], complex); spinor /= np.linalg.norm(spinor)
    wave = np.exp(1j*np.einsum('i,ixyz->xyz', momenta, points))/np.sqrt(n**3)
    initial = (spinor[:, None, None, None]*wave[None]).reshape(1, -1)
    symbol = .25*beta+sum(alpha[i]*np.sin(momenta[i]) for i in range(3))+np.eye(4)*np.sum(velocity*np.sin(2*momenta)/2)
    target = ((symbol@spinor)[:, None, None, None]*wave[None]).reshape(1, -1)
    observed = free.h_apply(initial)
    wraps = [free.h_apply(free.valley_image(initial, nu))-free.valley_image(observed, nu) for nu in itertools.product((0, 1), repeat=3)]
    r.case('periodic_sine_two_step_symbol_and_even_valley_wrap', {'p': momenta, 'velocity': velocity, 'initial': initial},
           {'symbol': symbol, 'H_initial': target}, {'H_initial': observed, 'valley_residuals': np.asarray(wraps)},
           {'symbol': num.block_norm(observed-target) <= 5e-12*max(1., free.B),
            'even_periodic_valleys': max(num.block_norm(x) for x in wraps) <= 5e-12*max(1., free.B)})
    radial_N = 1+np.arange(10)/64
    radial_v = np.arange(10)/128-.04
    h = independent_chain(np, 1, radial_N, radial_v, mass=.25)
    opposite = independent_chain(np, -1, radial_N, radial_v, mass=.25)
    parity = np.kron(np.diag((-1.)**np.arange(1, 11)), np.diag([1., -1.]))
    blocks = channel.folded_blocks(1, radial_N, radial_v, first_odd=1, end=10, mass=.25)
    W = np.kron(np.eye(5), channel.fold_matrix())
    expected = W@h@W.conj().T
    assembled = np.zeros_like(expected)
    for c in range(5):
        assembled[4*c:4*c+4, 4*c:4*c+4] = blocks['onsite'][0, c]
        if c < 4:
            assembled[4*c:4*c+4, 4*c+4:4*c+8] = blocks['forward'][0, c]
            assembled[4*c+4:4*c+8, 4*c:4*c+4] = blocks['forward'][0, c].conj().T
    dropped = h.copy(); dropped[4:6, 8:10] = 0; dropped[8:10, 4:6] = 0
    wrong_W = W.copy(); wrong_W[2] *= -1
    r.case('radial_parity_and_every_folded_crossing', {'N': radial_N, 'v': radial_v, 'raw_H': h, 'W': W},
           {'opposite_k_H': opposite, 'folded_H': expected},
           {'parity_H': parity@h@parity.conj().T, 'folded_H': assembled,
            'dropped_two_link_fold': W@dropped@W.conj().T, 'wrong_parity_fold': wrong_W@h@wrong_W.conj().T},
           {'radial_opposite_k': np.linalg.norm(parity@h@parity.conj().T-opposite) <= 5e-12,
            'complete_fold': np.linalg.norm(assembled-expected) <= 5e-12,
            'missing_two_link_detected': np.linalg.norm(W@dropped@W.conj().T-expected) > 1e-6,
            'wrong_parity_detected': np.linalg.norm(wrong_W@h@wrong_W.conj().T-expected) > 1e-6})


def split_and_hybrid_cases(r, np, native, channel, hybrid, num, expm):
    n, sites = 5, 125
    points = np.indices((n,)*3)-2
    N = 1+(points[0]+points[1])/128
    v = np.array([points[1]/128, points[2]/128, points[0]/128])
    h = independent_native(np, N, v)
    active = np.flatnonzero(np.tile(np.sum(points*points, axis=0).ravel() <= 4, 4))
    small = h[np.ix_(active, active)]
    core_count = len(active)//2
    outside = len(active)-core_count
    unitary = np.exp(2j*np.pi*np.arange(outside)[:, None]*np.arange(outside)[None]/outside)/np.sqrt(outside)
    change = np.eye(len(active), dtype=complex); change[core_count:, core_count:] = unitary
    transformed = change.conj().T@small@change
    F = np.zeros((2, len(active)), complex); F[0, 0], F[0, 1], F[1, 0], F[1, 2] = 1., .5j, .25j, 1.
    fixture = native.DenseHamiltonian(transformed)
    own = num.propagate(fixture, F, [.25])
    expected = (expm(-.25j*small)@F.T).T
    restored = own['final_columns']@change.T
    mask = np.arange(len(active)) < core_count
    P = np.diag(mask)
    current_oracle = F.conj()@(1j*(transformed@P-P@transformed))@F.T
    energy_oracle = F.conj()@(.5j*(transformed@transformed@P-P@transformed@transformed))@F.T
    r.case('complete_native_exterior_unitary_coordinates', {'H_unsplit': small, 'basis_change': change, 'F': F, 'P': P},
           {'final': expected, 'current': current_oracle, 'energy_current': energy_oracle},
           {'final_restored': restored, 'current': hybrid.projector_current(fixture, F, mask),
            'energy_current': hybrid.projector_current(fixture, F, mask, energy=True)},
           {'propagation': num.block_norm(restored-expected) <= 5e-12,
            'complete_Gram': num.matrix_norm(hybrid.cross_gram(restored)-hybrid.cross_gram(F)) <= 5e-12,
            'signed_current': num.matrix_norm(hybrid.projector_current(fixture, F, mask)-current_oracle) <= 5e-12,
            'energy_current': num.matrix_norm(hybrid.projector_current(fixture, F, mask, energy=True)-energy_oracle) <= 5e-12})
    # A small four-shell engineering fixture runs the actual completed apply.
    # Every shell complement is kept. Known Fourier columns fix its metric.
    native_op = native.NativeHamiltonian(N, v)
    shell_sites = [np.flatnonzero(np.indices((n,)*3)[0].ravel() == c) for c in range(4)]
    dofs = [(np.arange(4)[:, None]*sites+selected[None]).reshape(-1) for selected in shell_sites]
    E = [np.exp(2j*np.pi*np.arange(len(d))[:, None]*np.arange(64)[None]/len(d))/np.sqrt(len(d)) for d in dofs]
    radial_N, radial_v = 1+np.arange(44)/512, np.arange(44)/2048-.03
    paper = channel.folded_blocks(1, radial_N, radial_v, first_odd=33, end=44)
    raw = independent_chain(np, 1, radial_N, radial_v)
    W = np.kron(np.eye(6), channel.fold_matrix())
    folded_oracle = W@raw[64:88, 64:88]@W.conj().T
    corrections, dg = {}, {}
    matrix = np.zeros((500+128,)*2, complex); matrix[:500, :500] = h
    s = (0., .25, .75, 1.)
    for c in range(4):
        for d in range(c, min(4, c+2)):
            value = E[c].conj().T@h[np.ix_(dofs[c], dofs[d])]@E[d]
            dg[c, d] = value
            ideal = channel.ideal_cell_block(1, paper, c, d)
            corrections[c, d] = (s[c]+s[d])/2*(ideal-value)
            target = np.kron(np.eye(16), folded_oracle[4*c:4*c+4, 4*d:4*d+4])
            delta = (s[c]+s[d])/2*(target-value)
            block = E[c]@delta@E[d].conj().T
            matrix[np.ix_(dofs[c], dofs[d])] += block
            if c != d:
                matrix[np.ix_(dofs[d], dofs[c])] += block.conj().T
    port = channel.ideal_cell_block(1, paper, 3, 4)
    port_oracle = np.kron(np.eye(16), folded_oracle[12:16, 16:20])
    # Lead state order is chain first, then its two folded cells.
    matrix[500:, 500:] = np.kron(np.eye(16), folded_oracle[16:24, 16:24])
    lead_first = 500+(np.arange(16)[:, None]*8+np.arange(4)[None]).reshape(-1)
    upper = E[3]@port_oracle
    matrix[np.ix_(dofs[3], lead_first)] += upper
    matrix[np.ix_(lead_first, dofs[3])] += upper.conj().T
    setup = {'K': 1, 'E': E, 'shell_dofs': dofs, 'E_norm_bounds': [1+1e-12]*4,
             'Dg': dg, 'corrections': corrections, 'port': port}
    coupled = hybrid.HybridOperator(native_op, setup, radial_N, radial_v, 44)
    applied = coupled.h_apply(np.eye(len(matrix), dtype=np.complex128)).T
    component_timings = {}
    profiled = (1j*coupled.apply(np.eye(len(matrix), dtype=np.complex128), timings=component_timings)).T
    probes = np.random.default_rng(1917).normal(size=(2, len(matrix)))+1j*np.random.default_rng(1918).normal(size=(2, len(matrix)))
    probes /= np.linalg.norm(probes, axis=1)[:, None]
    evolved = num.propagate(coupled, probes, [.125])
    oracle = (expm(-.125j*matrix)@probes.T).T
    diagnostic = coupled.diagnostics(evolved['final_columns'])
    wrong = matrix.copy(); wrong[np.ix_(lead_first, dofs[3])] = upper.T
    r.case('actual_four_shell_completed_hybrid_apply', {'N': N, 'v': v, 'frames': E, 'dofs': dofs,
           'radial_N': radial_N, 'radial_v': radial_v, 'probes': probes},
           {'H_dense': matrix, 'final': oracle}, {'H_apply': applied, 'H_apply_profiled': profiled,
           'component_timings_seconds': component_timings, 'final': evolved['final_columns'],
           'diagnostics': diagnostic, 'wrong_adjoint_H': wrong},
           {'actual_apply': np.linalg.norm(applied-matrix) <= 5e-12*max(1., coupled.B),
            'profiled_apply_dense_oracle': np.linalg.norm(profiled-matrix) <= 5e-12*max(1., coupled.B),
            'profiled_apply_identical_arithmetic': np.array_equal(profiled, applied),
            'all_component_timings_recorded': set(component_timings) == {'native_generator', 'lead_generator', 'shell_projection', 'collar_multiply', 'port', 'shell_reconstruction'} and all(np.isfinite(value) and value >= 0 for value in component_timings.values()),
            'dense_propagation': num.block_norm(evolved['final_columns']-oracle) <= 5e-12,
            'complete_inventory': num.matrix_norm(diagnostic['native_Gram']+diagnostic['lead_Gram']-hybrid.cross_gram(evolved['final_columns'])) <= 5e-12,
            'nonzero_complement': sum(np.linalg.norm(q) for q in diagnostic['complement_amplitudes']) > 1e-3,
            'wrong_adjoint_detected': np.linalg.norm(wrong-wrong.conj().T) > 1e-6,
            'wrong_metric_detected': np.linalg.norm((2**.5*E[0]).conj().T@(2**.5*E[0])-np.eye(64)) > 1.})


def angular_cases(r, np, native, channel, hybrid, num, expm):
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 2]], dtype=np.int64)
    explicit = np.full((len(points), 2), 0j)
    explicit[:, 0] = 1/np.sqrt(4*np.pi)
    plus = channel.spinor_harmonic(1, 1, points)
    minus = channel.spinor_harmonic(-1, 1, points)
    sigma = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
    direction = points/np.linalg.norm(points, axis=1)[:, None]
    sigma_r = np.einsum('ni,ijk->njk', direction, sigma)
    omega_residual = np.einsum('nij,nj->ni', sigma_r, plus)+minus
    r.case('explicit_k1_extremal_spinor_harmonics', {'points': points},
           {'Omega_plus_k1_mhalf': explicit, 'sigma_r_identity': 0.},
           {'Omega_plus': plus, 'Omega_minus': minus, 'identity_residual': omega_residual},
           {'explicit_plus': np.max(np.abs(plus-explicit)) <= 5e-12,
            'Omega_pair_identity': np.max(np.abs(omega_residual)) <= 5e-12,
            'negative_m_extremal_finite': np.all(np.isfinite(channel.spinor_harmonic(1, -1, points)))})
    H = 3
    all_points = native.coordinates(H)
    squared = np.sum(all_points*all_points, axis=1)
    sites = np.flatnonzero((squared > 1) & (squared <= 9))
    shell_points = all_points[sites]
    A = channel.sampled_frame(shell_points, 1)
    G = A.conj().T@A
    eigenvalues, vectors = np.linalg.eigh((G+G.conj().T)/2)
    if eigenvalues[0] <= 1e-12*eigenvalues[-1]:
        r.case('actual_angular_rank_whitening', {'points': shell_points, 'A': A},
               {'full_rank': True}, {'G': G, 'eigenvalues': eigenvalues}, {'full_rank': False})
        return
    E = A@((vectors/np.sqrt(eigenvalues)[None])@vectors.conj().T)
    covariance = []
    for mu in itertools.product((0, 1), repeat=3):
        target, phases = channel.valley_permutation(1, mu)
        transformed = np.einsum('ab,bnd->and', native.valley_matrix(mu), E.reshape(4, len(sites), 64))
        transformed *= np.where((shell_points@np.asarray(mu)) % 2, -1., 1.)[None, :, None]
        covariance.append(transformed.reshape(E.shape)-E[:, target]*phases[None])
    c_target, c_phases = channel.antiunitary_permutation(1)
    _, _, uc = native.spin_matrices()
    c_image = np.einsum('ab,bnd->and', uc, E.reshape(4, len(sites), 64).conj()).reshape(E.shape)
    c_residual = c_image-E[:, c_target]*c_phases[None]
    n = 7
    lapse = np.ones((n,)*3)
    flow = np.zeros((3, n, n, n))
    mask = (squared <= 9).reshape((n,)*3)
    operator = native.NativeHamiltonian(lapse, flow, mask=mask)
    dofs = (np.arange(4)[:, None]*n**3+sites[None]).reshape(-1)
    active = np.flatnonzero(np.tile(mask.ravel(), 4))
    dense = independent_native(np, lapse, flow, mask=mask)[np.ix_(active, active)]
    initial = np.zeros((32, 4*n**3), complex)
    for nu in range(8):
        plus, minus = E[:, 8*nu], E[:, 8*nu+2]
        for j, vector in enumerate((plus, minus, (plus+1j*minus)/np.sqrt(2), (plus-1j*minus)/np.sqrt(2))):
            initial[4*nu+j, dofs] = vector
    # Add a fixed actual angular complement, not a random abstract isometry.
    seed = np.exp(1j*np.arange(len(dofs))/7)
    complement = seed-E@(E.conj().T@seed)
    initial[0, dofs] += complement/np.linalg.norm(complement)
    evolved = num.propagate(operator, initial, [.125])
    expected = (expm(-.125j*dense)@initial[:, active].T).T
    own = evolved['final_columns']
    q = own[:, dofs]-hybrid.shell_projection(own, dofs, E)@E.T
    shell_active = np.searchsorted(active, dofs)
    embedding = np.zeros((len(active), 64), complex); embedding[shell_active] = E
    P_shell = np.zeros((len(active), len(active)), complex); P_shell[shell_active, shell_active] = 1
    Q = P_shell-embedding@embedding.conj().T
    q_oracle = expected@Q.T
    q_active = np.zeros((len(initial), len(active)), complex); q_active[:, shell_active] = q
    # Reversing the held Hamiltonian recovers the retained full P/Q state.
    reversed_operator = native.DenseHamiltonian(-dense)
    returned = num.propagate(reversed_operator, own[:, active], [.125])['final_columns']
    r.case('actual_angular_rank_whitening_PQ_and_valley_dynamics', {'points': shell_points, 'A': A, 'G': G, 'initial': initial},
           {'dense_final': expected, 'Q_final': q_oracle, 'returned': initial[:, active]},
           {'eigenvalues': eigenvalues, 'E': E, 'orthogonality': E.conj().T@E,
            'valley_residuals': np.asarray(covariance), 'antiunitary_frame_residual': c_residual,
            'antiunitary_target': c_target, 'antiunitary_phases': c_phases,
            'final': own[:, active], 'Q_final': q_active, 'returned': returned},
           {'actual_rank': eigenvalues[0] > 1e-12*eigenvalues[-1],
            'actual_whitening': np.max(np.abs(E.conj().T@E-np.eye(64))) <= 5e-12,
            'eight_frame_valleys': max(np.linalg.norm(x) for x in covariance) <= 5e-12,
            'actual_frame_antiunitary': np.linalg.norm(c_residual) <= 5e-12,
            'antiunitary_frame_square': np.array_equal(c_target[c_target], np.arange(64)) and np.array_equal(c_phases*c_phases[c_target], np.ones(64)),
            'both_sectors_coherent_dynamics': num.block_norm(own[:, active]-expected) <= 5e-12*max(1., operator.B*num.block_norm(initial)),
            'complement_retained': np.linalg.norm(q) > .1,
            'actual_Q_dense_oracle': num.block_norm(q_active-q_oracle) <= 5e-12,
            'full_shell_reversibility': num.block_norm(returned-initial[:, active]) <= 5e-12})


def inventory_cases(r, np, native, hybrid, num, expm):
    h = np.array([[0, .5j], [-.5j, 0]], complex)
    F = np.array([[0, 1], [0, 1j]], complex)
    operator = native.DenseHamiltonian(h)
    result = num.propagate(operator, F, [.25, .5])
    before, after = result['readout_states']
    delta = hybrid.cross_gram(after[:, 1:])-hybrid.cross_gram(before[:, 1:])
    mask = np.array([True, False])
    inward = hybrid.projector_current(operator, np.array([[1, 1]], complex)/np.sqrt(2), mask)
    outward_lead = hybrid.projector_current(operator, np.array([[1, 1]], complex)/np.sqrt(2), ~mask)
    expected_inward = (np.array([[1, 1]], complex)/np.sqrt(2)).conj()@(1j*(h@np.diag(mask)-np.diag(mask)@h))@(np.array([[1, 1]], complex)/np.sqrt(2)).T
    # Same separate bin weights, different complex cross-bin covariance.
    x = np.array([[1, 1], [1j, 1]], complex)/np.sqrt(2)
    y = x.copy(); y[1, 1] *= 1j
    cov_x, cov_y = x.T@x.conj(), y.T@y.conj()
    damping = native.DenseHamiltonian(np.zeros((2, 2)), eta=np.array([.25, .75]))
    initial = np.array([[1, 1j], [1j, 1]], complex)/np.sqrt(2)
    damped = num.propagate(damping, initial, [2.])
    exact = initial*np.exp(-2*np.array([.25, .75]))[None]
    cap_H = np.array([[.25, .125j], [-.125j, -.5]], complex)
    noncommuting_cap = native.DenseHamiltonian(cap_H, eta=damping.eta)
    eta_matrix = np.diag(damping.eta)
    cap_rate_oracle = -initial.conj()@(eta_matrix@cap_H+cap_H@eta_matrix)@initial.T
    cap_rate = noncommuting_cap.energy_balance(initial)
    full = result['final_columns']
    partitions = hybrid.cross_gram(full[:, :1])+hybrid.cross_gram(full[:, 1:])
    r.case('reversible_inventory_return_phase_and_CAP_rate', {'H': h, 'F': F, 'covariance_states': x,
           'changed_cross_bin_phase': y, 'damping_eta': damping.eta, 'damping_initial': initial},
           {'negative_inventory_increment': True, 'inward_current': expected_inward,
            'damped_final': exact, 'CAP_rate_initial': 2*hybrid.cross_gram(initial*np.sqrt(damping.eta)[None])},
           {'inventory_increment': delta, 'inward_current': inward, 'outward_lead_current': outward_lead, 'covariance': cov_x,
            'changed_covariance': cov_y, 'damped_final': damped['final_columns'],
            'noncommuting_CAP_energy_rate': cap_rate, 'CAP_energy_rate_oracle': cap_rate_oracle,
            'partition_Gram': partitions, 'total_Gram': hybrid.cross_gram(full)},
           {'negative_inventory_increment': np.linalg.eigvalsh((delta+delta.conj().T)/2)[0] < -.01,
            'complete_partition': num.matrix_norm(partitions-hybrid.cross_gram(full)) <= 5e-12,
            'inward_current_sign': inward[0, 0].real > .1,
            'return_is_negative_outward_lead_current': outward_lead[0, 0].real < -.1,
            'inward_current_dense': num.matrix_norm(inward-expected_inward) <= 5e-12,
            'cross_bin_phase_matters': np.linalg.norm(cov_x-cov_y) > .1,
            'pure_damping_amplitude_rate': num.block_norm(damped['final_columns']-exact) <= 5e-12,
            'signed_CAP_energy_anticommutator': num.matrix_norm(cap_rate-cap_rate_oracle) <= 5e-12,
            'CAP_energy_is_not_conserved': num.matrix_norm(cap_rate_oracle) > .01})


def propagation_and_guard_cases(r, np, native, channel, hybrid, num, expm, outdir, runtime):
    from lead_setup import hermitian_realization
    h = np.array([[.25, .5+.125j], [.5-.125j, -.125]], complex)
    op = native.DenseHamiltonian(h)
    F = np.array([[1, .5j], [.25j, 1]], complex)
    clocks = np.array([0., .3, .9])
    plan = num.make_plan(op.B, clocks)
    fine = num.bisect_plan(plan)
    coarse_result = num.propagate(op, F, clocks, interval_plan=plan)
    fine_result = num.propagate(op, F, clocks, interval_plan=fine)
    exact = (expm(-.9j*h)@F.T).T
    refused = {}
    for name, call in {
        'too_large_z': lambda: num.taylor_step(op, F, 5/op.B),
        'duplicate_clock': lambda: num.make_plan(op.B, [.3, .3]),
        'nonpositive_clock': lambda: num.make_plan(op.B, [0.]),
        'wrong_degree': lambda: num.propagate(op, F, clocks, interval_plan={**plan, 'degree': 31}),
        'preloop_large_block_truncation': lambda: num.propagate(op, F*1e100, [4/op.B]),
        'metric_rank': lambda: num.normalize_comparison(F, num.metric_factor(np.zeros_like(F))),
        'unbounded_workspace': lambda: check_memory(runtime, 'expected_huge_workspace_refusal', workspace_bytes=1 << 100),
    }.items():
        try:
            call(); refused[name] = False
        except (ValueError, RuntimeError):
            refused[name] = True
    fixture = outdir/'hash_fixture.bin'; fixture.write_bytes(b'own fixed fixture\n')
    original = identity(fixture); fixture.write_bytes(b'own changed fixture\n')
    try:
        require_identity(fixture, original['sha256']); refused['hash_mutation'] = False
    except ValueError:
        refused['hash_mutation'] = True
    try:
        require_fresh(outdir); refused['existing_outputs'] = False
    except RuntimeError:
        refused['existing_outputs'] = True
    previous = os.environ['OMP_NUM_THREADS']
    os.environ['OMP_NUM_THREADS'] = '2'
    try:
        verify_threads(runtime, 'expected_bad_environment'); refused['changed_thread_environment'] = False
    except RuntimeError:
        refused['changed_thread_environment'] = True
    finally:
        os.environ['OMP_NUM_THREADS'] = previous
    comparison = num.compare_blocks(coarse_result['final_columns'], fine_result['final_columns'])
    good_raw = h+1e-15j*np.eye(2)
    bad_raw = h+.01j*np.eye(2)
    small_realization = hermitian_realization(good_raw, op.B)
    large_realization = hermitian_realization(bad_raw, op.B)
    # Full off-batch coherence is recovered from retained common-clock amplitudes.
    batch_parts = [num.propagate(op, F[j:j+1], clocks, interval_plan=fine)['final_columns'] for j in range(2)]
    stacked = np.concatenate(batch_parts, axis=0)
    batched_gram = np.block([[hybrid.cross_gram(stacked[:1]), hybrid.cross_gram(stacked[:1], stacked[1:])],
                            [hybrid.cross_gram(stacked[1:], stacked[:1]), hybrid.cross_gram(stacked[1:])]])
    # A deliberately close reflecting endpoint differs visibly at a late clock.
    short = channel.RadialHamiltonian(1, np.ones(6), np.zeros(6))
    long = channel.RadialHamiltonian(1, np.ones(24), np.zeros(24))
    pulse = np.zeros((1, 12), complex); pulse[0, 2:4] = np.array([1, 1j])/np.sqrt(2)
    extended = np.zeros((1, 48), complex); extended[:, :12] = pulse
    short_state = num.propagate(short, pulse, [12.])['final_columns']
    long_state = num.propagate(long, extended, [12.])['final_columns']
    tail = num.finite_end_certificate(10000., 900., 1., 0.)
    cap_op = native.DenseHamiltonian(h, eta=np.array([.125, .5]))
    cap_plan = num.make_plan(cap_op.B, clocks)
    cap_coarse = num.propagate(cap_op, F, clocks, interval_plan=cap_plan)
    cap_fine = num.propagate(cap_op, F, clocks, interval_plan=num.bisect_plan(cap_plan))
    cap_comparison = num.compare_blocks(cap_coarse['final_columns'], cap_fine['final_columns'],
        cap_coarse['step_diagnostics'][-1]['energy_Gram'], cap_fine['step_diagnostics'][-1]['energy_Gram'], cap_coarse['initial_energy_Gram'])
    r.case('fixed_degree_refinement_bound_guards_batching_and_end_return', {'H': h, 'F': F, 'clocks': clocks,
           'short_pulse': pulse, 'long_pulse': extended, 'good_raw_H': good_raw, 'bad_raw_H': bad_raw}, {'exact_final': exact, 'all_refusals': True,
           'late_short_end_difference': True, 'off_batch_Gram': hybrid.cross_gram(fine_result['final_columns'])},
           {'coarse': coarse_result, 'fine': fine_result, 'refinement': comparison, 'refusals': refused,
            'batched_Gram': batched_gram, 'independent_batches': batch_parts, 'short_final': short_state,
            'long_common_final': long_state[:, :12], 'failed_end_certificate': tail,
            'small_raw_realization': small_realization, 'large_raw_realization': large_realization,
            'CAP_coarse': cap_coarse, 'CAP_fine': cap_fine, 'CAP_refinement': cap_comparison},
           {**refused, 'exact_degree32': num.block_norm(coarse_result['final_columns']-exact) <= 5e-12,
            'actual_interval_bisection': np.array_equal(fine['times'][::2], plan['times']),
            'exact_initial_readout': np.array_equal(coarse_result['readout_states'][0], F) and coarse_result['readouts'][0]['clock'] == 0.,
            'preloop_truncation': coarse_result['bound']['bound_pass'],
            'refinement': comparison['state_and_Gram_pass'],
            'complete_off_batch_coherence': num.matrix_norm(batched_gram-hybrid.cross_gram(fine_result['final_columns'])) <= 5e-12,
            'discarding_off_batch_coherence_detected': np.linalg.norm(batched_gram-np.diag(np.diag(batched_gram))) > .1,
            'small_raw_roundoff_realization': small_realization['passed'] and np.array_equal(small_realization['value'], h),
            'large_raw_defect_not_averaged': not large_realization['passed'] and large_realization['value'] is None,
            'CAP_same_clock_refinement': cap_comparison['state_and_Gram_pass'] and cap_comparison['energy_refinement_pass'],
            'CAP_unitary_energy_gate_inapplicable': cap_coarse['maximum_scaled_unitary_energy_defect'] is None and cap_coarse['unitary_numerical_gates_pass'] is None,
            'short_end_return_detected': num.block_norm(short_state-long_state[:, :12]) > 1e-3,
            'bad_end_bound_withheld': not tail['passed']})


def run(args, runtime):
    import numpy as np
    import numba
    from scipy.linalg import expm
    import lead_native as native
    import lead_channels as channel
    import lead_operator as hybrid
    import lead_numerics as num
    from lead_field_controls import run_field_controls
    verify_threads(runtime, 'after_numerical_imports')
    check_memory(runtime, 'before_manufactured_dense', workspace_bytes=2*1024**3)
    bindings = {'spec': require_identity(args.spec, SPEC_SHA256),
                'derivation': require_identity(args.derivation, DERIVATION_SHA256), 'code': code_bindings()}
    recorder = Recorder(np)
    families = (('actual_Fourier_oracle', lambda: fourier_oracle_case(recorder, np, num, expm)),
                ('native_fold', lambda: algebra_cases(recorder, np, native, channel, num)),
                ('split_completed_hybrid', lambda: split_and_hybrid_cases(recorder, np, native, channel, hybrid, num, expm)),
                ('angular_complement', lambda: angular_cases(recorder, np, native, channel, hybrid, num, expm)),
                ('inventory_CAP', lambda: inventory_cases(recorder, np, native, hybrid, num, expm)),
                ('propagation_guards', lambda: propagation_and_guard_cases(recorder, np, native, channel, hybrid, num, expm, args.outdir, runtime)))
    timings = {}
    try:
        for name, callback in families:
            started = time.perf_counter(); callback(); timings[name] = time.perf_counter()-started
            verify_threads(runtime, 'after_'+name)
            check_memory(runtime, 'after_'+name)
        fields, arrays = run_field_controls()
        recorder.arrays.update({'field_'+key: value for key, value in arrays.items()})
        recorder.case('field_reconstruction_and_back_average', {'fixture_arrays': ['field_'+key for key in arrays]},
                      {'all_passed': True}, fields, {row['name']: row['passed'] for row in fields['checks']})
        verify_threads(runtime, 'after_fields')
    except Exception:
        recorder.preserve(args.outdir, {'runtime': runtime, 'bindings': bindings,
                                       'family_elapsed_seconds': timings})
        raise
    if code_bindings() != bindings['code']:
        raise RuntimeError('Staged source changed during manufactured controls')
    require_identity(args.spec, SPEC_SHA256); require_identity(args.derivation, DERIVATION_SHA256)
    raw = args.outdir/'controls_arrays.npz'
    np.savez_compressed(raw, **recorder.arrays)
    output = {'all_passed': all(row['passed'] for row in recorder.cases),
              'bindings': bindings, 'spec_binding': bindings['spec'], 'code_bindings': bindings['code'],
              'core_bindings': core_bindings(), 'runtime': runtime, 'cases': recorder.cases,
              'family_elapsed_seconds': timings, 'numba_version': numba.__version__,
              'numba_threading_layer': numba.threading_layer(),
              'output_bindings': {str(path.relative_to(args.outdir)): identity(path) for path in args.outdir.rglob('*') if path.is_file()},
              'original_inputs_opened': False, 'expected_negative_controls_are_physical_failures': False}
    (args.outdir/'controls.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('Manufactured cases saved; all_passed='+str(output['all_passed']), flush=True)
    return 0 if output['all_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--derivation', type=Path, required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Own channel-lead manufactured controls', 16, 15, mode='mixed')
    require_fresh(args.outdir)
    try:
        return run(args, runtime)
    except Exception as error:
        (args.outdir/'controls_failure.json').write_text(json.dumps({'runtime': runtime, 'status': 'incomplete',
            'exception': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
