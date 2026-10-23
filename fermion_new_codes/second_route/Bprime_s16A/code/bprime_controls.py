"""Manufactured-only Bprime entrypoint; original scientific inputs are not opened."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from bprime_runtime import admit, check_memory, core_bindings, identity, require_identity, verify_threads


SPEC_SHA256 = 'ca1071223753f58cd8a1afe664e6369af69a0f803808fcf54e71f1351a24d755'


class Recorder:
    def __init__(self, np, output):
        self.np, self.output, self.rows, self.arrays = np, Path(output), [], {}

    def encode(self, value, key):
        np = self.np
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('No object fixture serialization')
            self.arrays[key] = value.copy()
            return {'array_key': key, 'shape': list(value.shape), 'dtype': str(value.dtype)}
        if isinstance(value, np.generic):
            return self.encode(value.item(), key)
        if isinstance(value, float) and not np.isfinite(value):
            self.arrays[key] = np.asarray(value)
            return {'nonfinite_scalar_array_key': key}
        if isinstance(value, complex):
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, dict):
            return {str(k): self.encode(v, key+'__'+str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(v, key+'__'+str(i)) for i, v in enumerate(value)]
        return value

    def case(self, name, family, inputs, expected, run, valid, rejection=False):
        try:
            observed = run()
            passed = not rejection and bool(valid(observed))
        except Exception as error:
            observed = {'exception': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
            passed = rejection and isinstance(error, ValueError)
        key = 'case'+str(len(self.rows))
        self.rows.append({'name': name, 'family': family, 'passed': passed,
                          'inputs': self.encode(inputs, key+'__input'), 'expected': self.encode(expected, key+'__expected'),
                          'observed': self.encode(observed, key+'__observed')})

    def save(self, execution, final=False):
        raw = self.output/('controls_arrays.npz' if final else 'controls_partial_arrays.npz')
        self.np.savez_compressed(raw, **self.arrays)
        output = {'execution': execution, 'core_bindings': execution['core_bindings'],
                  'all_passed': bool(self.rows) and all(x['passed'] for x in self.rows),
                  'control_count': len(self.rows), 'checks': self.rows,
                  'failed_check_names': [x['name'] for x in self.rows if not x['passed']],
                  'output_bindings': {'arrays': identity(raw)}}
        output['output_bindings'].update({str(path.relative_to(self.output)): identity(path)
                                          for path in sorted((self.output/'fixtures').glob('*')) if path.is_file()})
        path = self.output/('controls.json' if final else 'controls_partial.json')
        path.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
        return output


def dense_native(np, op, lapse, flow, eta, mass, periodic):
    """Independent Kronecker/shift construction, not the native link tables."""
    n = lapse.shape[0]
    shift = np.diag(np.ones(n-1), 1).astype(np.complex128)
    if periodic:
        shift[-1, 0] = 1
    eye = np.eye(n)
    shifts = [np.kron(np.kron(shift, eye), eye), np.kron(np.kron(eye, shift), eye), np.kron(np.kron(eye, eye), shift)]
    alpha, beta, _ = op.spin_matrices()
    spatial = n**3
    h0 = mass*np.kron(beta, np.eye(spatial))
    currents = []
    for matrix, spin in zip(shifts, alpha):
        h0 += np.kron(spin, (matrix-matrix.conj().T)/(2j))
        currents.append(np.kron(np.eye(4), (matrix@matrix-matrix.conj().T@matrix.conj().T)/(4j)))
    nf = np.tile(lapse.ravel(), 4)
    h = (nf[:, None]*h0+h0*nf[None])/2
    for vf, current in zip(flow, currents):
        vf = np.tile(vf.ravel(), 4)
        h += (vf[:, None]*current+current*vf[None])/2
    return h, -1j*h-np.diag(np.tile(eta.ravel(), 4))


def native_cases(r, np, op, num, expm):
    x, y, z = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    lapse = 1+x/20+y/30+z/40
    flow = np.asarray([.1+x/100, -1/12+y/120, 1/14+z/140])
    eta = .2+(x+2)/100
    rng = np.random.default_rng(1701)
    for periodic in (False, True):
        native = op.NativeOperator(lapse, flow, eta, .2, periodic=periodic)
        h, A = dense_native(np, op, lapse, flow, eta, .2, periodic)
        states = np.eye(500, dtype=np.complex128).reshape(500, 4, 5, 5, 5)
        def run(native=native, states=states, h=h, A=A):
            matrix = native.apply(states).reshape(500, 500).T
            adjoint = native.adjoint().apply(states).reshape(500, 500).T
            return {'A_native': matrix, 'A_adjoint_native': adjoint,
                    'forward_maximum_entry': float(np.max(np.abs(matrix-A))),
                    'adjoint_maximum_entry': float(np.max(np.abs(adjoint-A.conj().T))),
                    'h_Hermiticity': float(np.max(np.abs(h-h.conj().T))),
                    'B': native.B, 'dense_norm2': float(np.linalg.norm(A, 2))}
        r.case('native_dense_'+str(periodic), 1, {'lapse': lapse, 'flow': flow, 'eta': eta, 'mass': .2, 'periodic': periodic},
               {'A_dense': A, 'A_adjoint_dense': A.conj().T, 'threshold': 5e-12}, run,
               lambda v: v['forward_maximum_entry'] <= 5e-12 and v['adjoint_maximum_entry'] <= 5e-12 and v['h_Hermiticity'] <= 5e-12 and v['dense_norm2'] <= v['B']*(1+1e-14))
        probes = (rng.normal(size=(2, 4, 5, 5, 5))+1j*rng.normal(size=(2, 4, 5, 5, 5))).astype(np.complex128)/32
        reference = (probes.reshape(2, -1)@expm(A*.125).T).reshape(probes.shape)
        def forward_backward(native=native, probes=probes, reference=reference):
            primary = num.propagate(native, probes, .125, integrate=True)
            fine = num.propagate(native, probes, .125, steps=2*primary['method']['steps'], integrate=False)
            adjoint = num.propagate(native.adjoint(), fine['final_columns'], .125, steps=fine['method']['steps'], integrate=False)
            expected_adjoint = (reference.reshape(2, -1)@expm(A.conj().T*.125).T).reshape(probes.shape)
            return {'primary': primary, 'fine': fine, 'adjoint': adjoint,
                    'reference': reference, 'expected_adjoint': expected_adjoint,
                    'forward_error': float(np.max(np.abs(fine['final_columns']-reference))),
                    'adjoint_error': float(np.max(np.abs(adjoint['final_columns']-expected_adjoint))),
                    'action_Gram_closure': probes.reshape(2, -1).conj()@(probes-adjoint['final_columns']).reshape(2, -1).T-fine['loss_Gram']}
        r.case('native_Taylor_forward_finer_adjoint_no_integral_'+str(periodic), 2,
               {'initial': probes, 'T': .125}, {'dense_endpoint': reference, 'threshold': 5e-12}, forward_backward,
               lambda v: v['forward_error'] <= 5e-12 and v['adjoint_error'] <= 5e-12 and
                   v['primary']['integral_closure_maximum_entry'] <= 5e-12 and np.max(np.abs(v['action_Gram_closure'])) <= 5e-12)
    constant = op.NativeOperator(np.full((5, 5, 5), 1.3), np.asarray([np.full((5, 5, 5), value) for value in (.2, -.1, .15)]), np.zeros((5, 5, 5)), .2, periodic=True)
    alpha, beta, _ = op.spin_matrices()
    k = 2*np.pi*np.asarray([1, 2, 1])/5
    plane = np.exp(1j*(k[0]*x+k[1]*y+k[2]*z))
    states = np.eye(4, dtype=np.complex128)[:, :, None, None, None]*plane[None, None]
    symbol = 1.3*(.2*beta+sum(alpha[i]*np.sin(k[i]) for i in range(3)))+sum(v*np.sin(2*kk)/2 for v, kk in zip((.2, -.1, .15), k))*np.eye(4)
    expected = symbol.T[:, :, None, None, None]*plane[None, None]
    r.case('periodic_Fourier_one_and_two_step_orientation', 1, {'k': k, 'states': states}, {'h_symbol_action': expected},
           lambda: constant.h_apply(states), lambda v: np.max(np.abs(v-expected)) <= 5e-12)


def symmetry_cases(r, np, op, num, expm):
    alpha, beta, uc = op.spin_matrices()
    residuals = np.asarray([uc@value.conj()@uc.conj().T-value for value in alpha])
    r.case('canonical_spin_and_antiunitary_square', 2, {'alpha': alpha, 'beta': beta, 'UC': uc},
           {'alpha_residual': np.zeros_like(alpha), 'beta_residual': np.zeros_like(beta)},
           lambda: {'alpha': residuals, 'beta': uc@beta.conj()@uc.conj().T+beta, 'square': uc@uc.conj()-np.eye(4)},
           lambda v: max(np.max(np.abs(v[key])) for key in ('alpha', 'beta', 'square')) <= 5e-12)
    x, y, z = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    native = op.NativeOperator(1+x/20+y/30+z/40, np.asarray([.1+x/100, -1/12+y/120, 1/14+z/140]), .2+(x+2)/100, .2)
    probes = (np.sin(np.arange(1000)*.17)+1j*np.cos(np.arange(1000)*.13)).reshape(2, 4, 5, 5, 5).astype(np.complex128)
    def run():
        hres = native.h_apply(op.antiunitary(probes))+op.antiunitary(native.h_apply(probes))
        ares = native.apply(op.antiunitary(probes))-op.antiunitary(native.apply(probes))
        sinkres = native.eta[None, None]*op.antiunitary(probes)-op.antiunitary(native.eta[None, None]*probes)
        forward = num.propagate(native, probes, .125)
        conjugate = num.propagate(native, op.antiunitary(probes), .125)
        return {'h': hres, 'A': ares, 'eta': sinkres, 'endpoint': conjugate['final_columns']-op.antiunitary(forward['final_columns']),
                'Gram': conjugate['loss_Gram']-forward['loss_Gram'].conj()}
    r.case('nonconstant_massive_native_antiunitary_propagator_and_Gram', 2, {'states': probes}, {'all_residuals': 0., 'threshold': 5e-12}, run,
           lambda v: all(np.max(np.abs(value)) <= 5e-12*max(1., native.B) for value in v.values()))
    sx = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.diag([1., -1.]).astype(np.complex128)
    h, eta = sx/2, np.diag([.25, .75])
    base, shifted = expm(-1j*h-eta), expm(-1j*(h+np.eye(2)/8)-eta)
    r.case('scalar_energy_shift_breaks_operator_but_not_loss', 2, {'h': h, 'eta': eta, 'shift': .125},
           {'operator_residual': np.eye(2)/4, 'loss_difference': np.zeros((2, 2))},
           lambda: {'operator_residual': sz@(h+np.eye(2)/8).conj()@sz+h+np.eye(2)/8,
                    'loss_difference': base.conj().T@base-shifted.conj().T@shifted},
           lambda v: np.max(np.abs(v['operator_residual']-np.eye(2)/4)) <= 5e-12 and np.max(np.abs(v['loss_difference'])) <= 5e-12 and np.max(np.abs(v['operator_residual'])) > .1)
    broken = eta+sx/8
    r.case('positive_spin_sink_breaks_antiunitary', 2, {'eta': broken}, {'sink_residual': -sx/4},
           lambda: sz@broken.conj()@sz-broken, lambda v: np.max(np.abs(v+sx/4)) <= 5e-12 and np.max(np.abs(v)) > .1)


def analytic_cases(r, np, op, num, expm):
    F = np.asarray([[1, 1j], [1j, 1]], dtype=np.complex128)/np.sqrt(2)
    states = F.T.copy()
    d1, d2 = 1-np.exp(-1), 1-np.exp(-3)
    exact = np.asarray([[(d1+d2)/2, 1j*(d1-d2)/2], [-1j*(d1-d2)/2, (d1+d2)/2]])
    pure = op.DenseGenerator(np.zeros((2, 2)), [.25, .75])
    r.case('pure_damping_factor_two_complex_offdiagonal', 3, {'F': F, 'eta': [.25, .75], 'T': 2.},
           {'loss': exact, 'eigenvalues': np.asarray([d1, d2])}, lambda: num.propagate(pure, states, 2.),
           lambda v: np.max(np.abs(v['loss_Gram']-exact)) <= 5e-12 and v['integral_closure_maximum_entry'] <= 5e-12 and np.max(np.abs(np.linalg.eigvalsh(v['loss_Gram'])-[d1, d2])) <= 5e-12)
    sx = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.diag([1., -1.]).astype(np.complex128)
    zero = op.DenseGenerator(sx/2, [0., 0.])
    unitary_endpoint = (F.T@(np.cos(1.)*np.eye(2)-1j*np.sin(1.)*sx).T)
    r.case('zero_sink_zero_loss', 3, {'F': F, 'h': sx/2, 'T': 2.}, {'loss': np.zeros((2, 2)), 'endpoint': unitary_endpoint},
           lambda: num.propagate(zero, states, 2.), lambda v: np.max(np.abs(v['loss_Gram'])) <= 5e-12 and
               v['integral_closure_maximum_entry'] <= 5e-12 and np.max(np.abs(v['final_columns']-unitary_endpoint)) <= 5e-12)
    generator = op.DenseGenerator(sx/2, [.25, .75])
    B0 = -1j*sx/2+sz/4
    V = np.exp(-.5)*(np.cos(np.sqrt(3)/4)*np.eye(2)+4/np.sqrt(3)*np.sin(np.sqrt(3)/4)*B0)
    exact = np.eye(2)-V.conj().T@V
    r.case('noncommuting_exact_exponential_and_sink_integral', 3, {'h': sx/2, 'eta': [.25, .75], 'T': 1.},
           {'V': V, 'loss': exact}, lambda: num.propagate(generator, np.eye(2, dtype=np.complex128), 1.),
           lambda v: np.max(np.abs(v['final_columns'].T-V)) <= 5e-12 and np.max(np.abs(v['loss_Gram']-exact)) <= 5e-12 and v['integral_closure_maximum_entry'] <= 5e-12 and v['bound']['loss_Gram_bound_pass'])
    nodes, weights = num.gauss8()
    moments = np.asarray([np.dot(weights, nodes**degree) for degree in range(16)])
    r.case('Gauss8_positive_normalized_monomials_through15', 3, {'nodes': nodes, 'weights': weights},
           {'moments': 1/(np.arange(16)+1)}, lambda: moments,
           lambda v: np.all(weights > 0) and abs(weights.sum()-1) <= 5e-12 and np.max(np.abs(v-1/(np.arange(16)+1))) <= 5e-12)


def association_cases(r, np, op, num, output):
    F = np.eye(3, dtype=np.complex128)
    loss = np.diag([.125, .5, .875]).astype(np.complex128)
    rotation = np.asarray([[1, 1j, 0], [1j, 1, 0], [0, 0, np.sqrt(2)]], dtype=np.complex128)/np.sqrt(2)
    rotated = rotation.conj().T@loss@rotation
    r.case('complex_basis_rotation_spectra_diagonals_and_principal_block', 4, {'loss': loss, 'rotation': rotation, 'indices': [0, 2]},
           {'spectrum': np.asarray([.125, .5, .875]), 'principal': loss[np.ix_([0, 2], [0, 2])]},
           lambda: {'rotated': rotated, 'spectrum': np.linalg.eigvalsh(rotated), 'principal': loss[np.ix_([0, 2], [0, 2])],
                    'wrong_map_difference': np.max(np.abs(loss-rotated))},
           lambda v: np.max(np.abs(v['spectrum']-[.125, .5, .875])) <= 5e-12 and v['wrong_map_difference'] > .1 and not np.array_equal(np.diag(v['rotated']), np.diag(loss)))
    data = (np.arange(108*3).reshape(108, 3)+1j*np.arange(108*3).reshape(108, 3)/7).astype(np.complex128)
    fixture = output/'fixtures'
    fixture.mkdir(exist_ok=True)
    c_path, f_path = fixture/'C.npz', fixture/'Fortran.npz'
    np.savez(c_path, U=data)
    np.savez(f_path, U=np.asfortranarray(data))
    def logical():
        with np.load(c_path, allow_pickle=False) as ca, np.load(f_path, allow_pickle=False) as fa:
            a, b = ca['U'], fa['U']
            op.require_exact(a, b, 'C/Fortran same logical columns')
            return {'C': op.canonical_identity(a), 'Fortran': op.canonical_identity(b),
                    'payload_orders_differ': a.flags.c_contiguous and b.flags.f_contiguous}
    r.case('logical_C_Fortran_identity_and_hash', 4, {'U': data, 'files': [str(c_path), str(f_path)]}, {'logical_hashes_equal': True}, logical,
           lambda v: v['C'] == v['Fortran'] and v['payload_orders_differ'])
    r.case('wrong_source_column_map_rejected', 4, {'U': data, 'wrong': data[:, ::-1]}, {'rejection': 'ValueError'},
           lambda: op.require_exact(data, data[:, ::-1], 'wrong columns'), lambda v: False, rejection=True)
    sparse = np.zeros((108, 1), dtype=np.complex128)
    sparse[2*27+1*9+2*3+0, 0] = 1+2j
    expected = np.zeros((1, 4, 5, 5, 5), dtype=np.complex128)
    expected[0, 2, 2, 3, 1] = 1+2j
    r.case('exact_spin_major_centered_subcube_embedding', 4, {'subcube': sparse, 'H': 2, 'Hs': 1}, {'embedded': expected},
           lambda: op.embed_subcube(sparse, 2, 1), lambda v: np.array_equal(v, expected))
    wrong = sparse.reshape(27, 4).T.reshape(108, 1)
    r.case('wrong_site_spin_map_detected', 4, {'wrong': wrong}, {'rejection': 'ValueError'},
           lambda: op.require_exact(op.embed_subcube(wrong, 2, 1), expected, 'wrong site/spin ordering'), lambda v: False, rejection=True)
    ratios = np.asarray([6.931471805599452, 69.31471805599453, 69.32])
    times = ratios*16
    r.case('exact_clock_endpoint', 4, {'T': times, 'ratios': ratios, 'L': 16.}, {'endpoint': float(times[0])},
           lambda: op.exact_clock_association(times, ratios, 16.), lambda v: v == times[0])
    for name, bad in (('missing', times[:-1]), ('duplicate', np.asarray([times[0], times[0], times[2]])),
                      ('rounded', np.asarray([6.93*16, times[1], times[2]]))):
        r.case('clock_'+name+'_rejected', 4, {'T': bad, 'ratios': ratios}, {'rejection': 'ValueError'},
               lambda bad=bad: op.exact_clock_association(bad, ratios, 16.), lambda v: False, rejection=True)
    mutation = fixture/'mutable_source.txt'
    mutation.write_text('before')
    before = identity(mutation)
    mutation.write_text('after')
    r.case('source_hash_mutation_rejected', 4, {'before_text': 'before', 'after_text': 'after', 'expected_sha256': before['sha256'], 'mutated_file': str(mutation)}, {'rejection': 'ValueError'},
           lambda: require_identity(mutation, before['sha256']), lambda v: False, rejection=True)


def bound_cases(r, np, op, num):
    h = np.asarray([[0, .5], [.5, 0]], dtype=np.complex128)
    generator = op.DenseGenerator(h, [.25, .75])
    initial = np.eye(2, dtype=np.complex128)
    def run():
        coarse = num.propagate(generator, initial, 1., integrate=True)
        fine = num.propagate(generator, initial, 1., steps=2*coarse['method']['steps'], integrate=False)
        adjoint = num.propagate(generator.adjoint(), fine['final_columns'], 1., steps=fine['method']['steps'], integrate=False)
        action = initial-adjoint['final_columns']
        return {'coarse': coarse, 'fine': fine, 'adjoint': adjoint,
                'complete_action_Gram': initial.conj()@action.T,
                'fine_Gram': fine['loss_Gram']}
    r.case('actual_no_integral_fine_and_adjoint_path_and_analytic_bounds', 5, {'initial': initial, 'T': 1.},
           {'degree': 32, 'maximum_z': 4., 'Gram_closure_threshold': 5e-12}, run,
           lambda v: all(v[key]['method']['degree'] == 32 and v[key]['method']['maximum_step_z'] <= 4. and
                         v[key]['bound']['loss_Gram_bound_pass'] for key in ('coarse', 'fine', 'adjoint')) and
               np.max(np.abs(v['complete_action_Gram']-v['fine_Gram'])) <= 5e-12 and
               np.max(np.abs(v['coarse']['loss_Gram']-v['fine']['loss_Gram'])) <= 5e-12)
    from scipy.linalg import expm
    h4 = np.asarray([[0, .2, .03j, 0], [.2, 0, .15, 0], [-.03j, .15, 0, .1], [0, 0, .1, 0]], dtype=np.complex128)
    generator4 = op.DenseGenerator(h4, [.125, .25, .5, .75])
    V = expm(generator4.matrix)
    source = np.eye(4)-V.conj().T@V
    eigenvalues, all_U = np.linalg.eigh((source+source.conj().T)/2)
    ranks = [0, 3]
    U, target = all_U[:, ranks], eigenvalues[ranks]
    W = U.T.copy()
    def four_actions():
        actions = num.run_actions(generator4, W, 1., reference_norm_bound=1.)
        Ys, diagnostics = {}, {}
        for label in ('coarse', 'fine'):
            Ys[label] = num.project_action(np.eye(4), U, np.eye(4), actions[label+'_adjoint']['final_columns'])
            diagnostics[label] = num.action_diagnostics(Ys[label], U, target, actions[label+'_forward']['loss_Gram'])
        refinement = num.refinement_diagnostics(actions['coarse_forward']['loss_Gram'], actions['fine_forward']['loss_Gram'],
                                                 Ys['coarse'], Ys['fine'], num.gram(W))
        adjoint_probe = num.propagate(generator4.adjoint(), W[[0]], 1., steps=actions['fine_forward']['method']['steps'], integrate=False)
        inner_left = np.vdot(W[0], actions['fine_forward']['final_columns'][1])
        inner_right = np.vdot(adjoint_probe['final_columns'][0], W[1])
        wrong = U*target[None]+.125*all_U[:, 1, None]@np.asarray([[1., 0.]])
        wrong_diagnostics = num.action_diagnostics(wrong, U, target, np.diag(target))
        U_coord = np.eye(4, dtype=np.complex128)[:, :2]
        coordinate_actions = num.run_actions(generator4, U_coord.T.copy(), 1., reference_norm_bound=1.)
        coordinate_Y = num.project_action(np.eye(4), U_coord, np.eye(4), coordinate_actions['fine_adjoint']['final_columns'])
        coordinate_diagnostics = num.projection_diagnostics(coordinate_Y, U_coord, coordinate_actions['fine_forward']['loss_Gram'])
        coordinate_expected = source@U_coord
        coordinate_expected_R = coordinate_expected-U_coord@(U_coord.conj().T@coordinate_expected)
        return {'actions': actions, 'full_Y': Ys, 'diagnostics': diagnostics, 'refinement': refinement,
                'dense_expected_action': source@U,
                'full_action_error': float(np.max(np.abs(Ys['fine']-source@U))),
                'adjoint_inner_left': inner_left, 'adjoint_inner_right': inner_right,
                'deliberately_omitted_coordinate_action': wrong, 'wrong_diagnostics': wrong_diagnostics,
                'noninvariant_coordinate_actions': coordinate_actions, 'noninvariant_U': U_coord,
                'noninvariant_full_action': coordinate_Y, 'noninvariant_dense_expected': coordinate_expected,
                'noninvariant_diagnostics': coordinate_diagnostics, 'noninvariant_expected_R': coordinate_expected_R}
    r.case('four_actions_source_spectral_directions_full_coordinate_and_offspan_detection', 5,
           {'h': h4, 'eta': generator4.eta, 'source_loss': source, 'source_U': U, 'source_eigenvalues': target,
            'ranks': ranks, 'physical_initial': W},
           {'full_action': source@U, 'false_compression_offspan_norm': .125, 'threshold': 5e-12}, four_actions,
           lambda v: v['full_action_error'] <= 5e-12 and v['refinement']['all_passed'] and
               abs(v['adjoint_inner_left']-v['adjoint_inner_right']) <= 5e-12 and
               all(v['diagnostics'][label]['projection_closure_maximum_entry'] <= 5e-12 and
                   v['diagnostics'][label]['source_eigenpair_residual_operator_norm'] <= 5e-12 for label in ('coarse', 'fine')) and
               v['wrong_diagnostics']['block_operator_norm_error'] <= 5e-12 and
               abs(v['wrong_diagnostics']['off_span_operator_norm']-.125) <= 5e-12 and not v['wrong_diagnostics']['beta_pass'] and
               np.max(np.abs(v['noninvariant_full_action']-v['noninvariant_dense_expected'])) <= 5e-12 and
               np.max(np.abs(v['noninvariant_diagnostics']['R']-v['noninvariant_expected_R'])) <= 5e-12 and
               np.linalg.norm(v['noninvariant_expected_R'], 2) > 5e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 16), default=15)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Bprime manufactured controls only; no original input values', thread_cap=16, numba_workers=args.workers)
    if any(args.outdir.iterdir()):
        raise RuntimeError('Preserve existing control artifacts; use a fresh empty output directory')
    import numpy as np
    import numba
    from scipy.linalg import expm
    import bprime_operator as op
    import bprime_numerics as num
    verify_threads(runtime, 'after_numerical_imports')
    check_memory(runtime, 'before_dense_manufactured', workspace_bytes=128*1024**2)
    spec = require_identity(args.spec, SPEC_SHA256)
    source = core_bindings()
    execution = {'runtime': runtime, 'spec_binding': spec, 'core_bindings': source,
                 'versions': {'numpy': np.__version__, 'numba': numba.__version__},
                 'custody': 'All manufactured JSON/NPZ and fixture files remain on the box.'}
    recorder = Recorder(np, args.outdir)
    for name, callback in (
        ('native', lambda: native_cases(recorder, np, op, num, expm)),
        ('symmetry', lambda: symmetry_cases(recorder, np, op, num, expm)),
        ('analytic', lambda: analytic_cases(recorder, np, op, num, expm)),
        ('association', lambda: association_cases(recorder, np, op, num, args.outdir)),
        ('bounds', lambda: bound_cases(recorder, np, op, num))):
        callback()
        verify_threads(runtime, 'after_'+name+'_controls')
        recorder.save(execution)
        print('Completed manufactured family '+name+'.', flush=True)
    if core_bindings() != source:
        raise RuntimeError('Own admitted source changed during controls')
    require_identity(args.spec, spec['sha256'])
    execution['numba_threads'] = numba.get_num_threads()
    execution['numba_threading_layer'] = numba.threading_layer()
    execution['all_source_and_spec_identities_rechecked_unchanged'] = True
    verify_threads(runtime, 'after_all_controls')
    result = recorder.save(execution, final=True)
    print('Manufactured control count '+str(result['control_count'])+'; failures '+str(len(result['failed_check_names'])), flush=True)
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
