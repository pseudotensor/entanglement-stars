"""Exact CAR primitives for the author-specified vertex gate; unrun draft.

The mathematical definitions and conventions are in VERTEX_gate_author_v1.tex.
Execution is restricted to the reviewed, fresh directory on the CPU box.
"""
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import sympy as sp


def scalar(value):
    return sp.simplify(sp.expand(value))


def matrices():
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    identity = sp.eye(2)
    kron = sp.kronecker_product
    alpha = [kron(sx, s) for s in (sx, sy, sz)]
    beta = kron(sz, identity)
    gamma5 = kron(sx, identity)
    charge_conjugation = -kron(sx, sy)
    gamma = [beta] + [sp.I * kron(sy, s) for s in (sx, sy, sz)]
    pair = [charge_conjugation * g * gamma5 for g in gamma]
    return dict(pauli=[sx, sy, sz], identity=identity, alpha=alpha,
                beta=beta, gamma5=gamma5, charge_conjugation=charge_conjugation,
                gamma=gamma, pair=pair, time_reversal=kron(identity, sp.I * sy))


def basis(modes, particles=None):
    if particles is None:
        return list(range(1 << modes))
    return sorted(sum(1 << j for j in occupied)
                  for occupied in combinations(range(modes), particles))


def act(word, state):
    """Apply a left-to-right operator word; (True,j) creates mode j."""
    sign = 1
    for create, mode in reversed(word):
        occupied = bool(state & (1 << mode))
        if occupied == create:
            return None
        if (state & ((1 << mode) - 1)).bit_count() % 2:
            sign = -sign
        state ^= 1 << mode
    return state, sign


def operator(terms, states):
    index = {state: j for j, state in enumerate(states)}
    entries = {}
    for column, state in enumerate(states):
        for coefficient, word in terms:
            answer = act(word, state)
            if answer is None:
                continue
            target, sign = answer
            if target not in index:
                raise ValueError('Specified operator leaves the chosen sector')
            key = index[target], column
            entries[key] = entries.get(key, 0) + sign * coefficient
    entries = {key: scalar(value) for key, value in entries.items()}
    return sp.SparseMatrix(len(states), len(states),
                           {key: value for key, value in entries.items() if value != 0})


def adjoint_terms(terms):
    return [(sp.conjugate(coefficient),
             tuple((not create, mode) for create, mode in reversed(word)))
            for coefficient, word in terms]


def multiply_terms(left, right):
    return [(a * b, wa + wb) for a, wa in left for b, wb in right]


def pair_terms(matrix, offset):
    return [(matrix[i, j], ((False, offset + i), (False, offset + j)))
            for i in range(4) for j in range(4) if matrix[i, j] != 0]


def onsite_terms(pair, site=0):
    """W=O+O†, with site modes ordered H0..3,L0..3 and no pair 1/2."""
    result = []
    for mu, matrix in enumerate(pair):
        metric = 1 if mu == 0 else -1
        light = pair_terms(matrix, 8 * site + 4)
        heavy = pair_terms(matrix, 8 * site)
        result.extend((metric * coefficient, word)
                      for coefficient, word in multiply_terms(adjoint_terms(light), heavy))
    return result + adjoint_terms(result)


def bilinear_terms(matrix, offset=0):
    return [(matrix[i, j], ((True, offset + i), (False, offset + j)))
            for i in range(matrix.rows) for j in range(matrix.cols)
            if matrix[i, j] != 0]


def bond_terms(alpha, spacing=sp.Integer(1)):
    """One open bond: -i c0† alpha c1/(2a) plus its adjoint, both species."""
    forward = []
    for species in (0, 4):
        for i in range(4):
            for j in range(4):
                if alpha[i, j] != 0:
                    forward.append((-sp.I * alpha[i, j] / (2 * spacing),
                                    ((True, species + i), (False, 8 + species + j))))
    return forward + adjoint_terms(forward)


def commutator(left, right):
    return left * right - right * left


def exact_entries(matrix):
    return [[int(i), int(j), str(scalar(value))]
            for (i, j), value in sorted(matrix.todok().items()) if scalar(value) != 0]


def matrix_record(matrix):
    return dict(shape=list(matrix.shape), entries=exact_entries(matrix))


ROOT = Path('<internal>/vertex_gate_v1')
SPEC = 'VERTEX_gate_author_v1.tex'
SPEC_SHA = '7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c'
INPUT_PINS = {
    '03_model.tex': 'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb',
    'JOINT_VISION_page_mock_signed_v1.md': 'fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279',
    'ReviewA_pcq4_v1.md': '1a4818713fb720ce41e3a90e0aa6d9a7e16c92b351d4bd718e9434371848dcb2',
    'ReviewA_pcq5_v1.md': 'eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2',
    'ReviewB_pcq5_v1.md': '2bacd47b093cdef79448dd5e4a6c3c4923c7829cfc5855d2ea117bb43630db9f',
    'ReviewB_pcq6_v1.md': '9279dacb90acc5bd804629e6058e52765469181e3b823e6dadcdc02be6c149af',
}
STAGES = ['spinor_algebra', 'on_site', 'bond_current', 'amplitudes']
STOP = []
EXACT_CACHE = {}
SPINOR_CACHE = {}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def exact(value):
    if value in EXACT_CACHE:
        return EXACT_CACHE[value]
    original = value
    value = scalar(value)
    result = dict(expression=str(value), srepr=sp.srepr(value))
    real, imag = value.as_real_imag()
    if real.is_Rational and imag.is_Rational:
        result['gaussian_rational'] = dict(real=[int(real.p), int(real.q)],
                                           imag=[int(imag.p), int(imag.q)])
    EXACT_CACHE[original] = result
    return result


def encode(value):
    if isinstance(value, sp.MatrixBase):
        entries = [[int(i), int(j), exact(v)] for (i, j), v in sorted(value.todok().items())
                   if v != 0]
        return dict(shape=list(value.shape), entries=entries)
    if isinstance(value, sp.Basic):
        return exact(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(encode(data), stream, separators=(',', ':'), sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def boundary():
    if STOP:
        raise TimeoutError('Handled stop: ' + ','.join(STOP))


class Record:
    def __init__(self):
        self.report = dict(status='running', spec_sha256=SPEC_SHA, input_sha256=INPUT_PINS,
                           source_sha256=digest(Path(__file__)), sympy_version=sp.__version__,
                           python_version=sys.version, stage_order=STAGES,
                           paper_commit='daaf6fbd05ff029e771f8fc1ebcd8f24ccf17dca',
                           units='a=1, U=lambda_X/a^3; no half in pairing; CAR-normalized spinors',
                           resources=dict(threads=1, address_space_bytes=2*2**30,
                                          handled_wall_seconds=110, external_wall_seconds=120,
                                          cpu_soft_seconds=100, cpu_hard_seconds=110),
                           stages={s: dict(saved=False, raw_complete=False, validated=False,
                                           validation_complete=False, all_equal=None) for s in STAGES},
                           missing_raw_products=[s+'.json' for s in STAGES], incomplete_raw_stages=STAGES.copy(),
                           target_complete=False,
                           validated_semantics='All specified equality decisions returned; all_equal is separate.',
                           missing_validations=STAGES.copy(), scientific_errors=[], operational_error=None,
                           per_item_findings={str(i): dict(status='pending') for i in range(1, 5)},
                           scope_qualifications=[
                               'Dirac-mass protection is in the symmetric phase with a preserving reference; the common-spinor Lorentz contraction does not establish common multivalley boost symmetry.',
                               'Amplitude tensors are not rates or occupied-state integrals.',
                               'An interacting total current does not define the later flow allocation.',
                               'Zero product Hartree field does not imply unchanged exact evolution.'])
        self.raw = None
        self.stage = None
        self.comparisons = []
        self.save_report()

    def save_report(self):
        stages = self.report['stages']
        self.report['missing_raw_products'] = [s+'.json' for s in STAGES if not stages[s]['saved']]
        self.report['incomplete_raw_stages'] = [s for s in STAGES if not stages[s]['raw_complete']]
        self.report['missing_validations'] = [s for s in STAGES if not stages[s]['validation_complete']]
        self.report['target_complete'] = all(stages[s]['raw_complete'] and stages[s]['validation_complete'] for s in STAGES)
        dependencies = {'1': ['spinor_algebra', 'on_site'], '2': ['amplitudes'],
                        '3': ['bond_current'], '4': ['on_site']}
        for item, names in dependencies.items():
            complete = all(stages[s]['validation_complete'] for s in names)
            self.report['per_item_findings'][item] = dict(
                status='comparisons_returned' if complete else 'pending', stage_dependencies=names,
                all_specified_comparisons_equal=all(stages[s]['all_equal'] is True for s in names) if complete else None)
        atomic(ROOT/'vertex_gate_result.json', self.report)

    def begin(self, name):
        boundary()
        self.stage = name
        self.raw = dict(stage=name, constructed={}, comparisons=[], status='building')
        self.comparisons = []
        self.save_report()

    def persist(self, **state_updates):
        atomic(ROOT/(self.stage+'.json'), self.raw)
        self.report['stages'][self.stage].update(saved=True, sha256=digest(ROOT/(self.stage+'.json')), **state_updates)
        self.save_report()

    def put(self, name, value):
        self.raw['constructed'][name] = encode(value)
        # A returned object is saved before a subsequent dependent operation.
        self.persist()
        boundary()

    def compare(self, name, actual, expected):
        saved = encode(dict(name=name, actual=actual, expected=expected,
                            comparison_status='residual_pending'))
        self.raw['comparisons'].append(saved)
        self.persist()
        boundary()
        residual = actual - expected
        if isinstance(residual, sp.MatrixBase):
            residual = residual.applyfunc(scalar)
        else:
            residual = scalar(residual)
        self.comparisons.append(dict(name=name, actual=actual, expected=expected, residual=residual))
        saved.update(encode(self.comparisons[-1]))
        saved['comparison_status'] = 'residual_returned'
        self.persist()
        boundary()

    def finish(self):
        # Persist actual and expected separately before evaluating identities.
        self.raw['status'] = 'returned'
        self.persist(raw_complete=True)
        boundary()
        for index, check in enumerate(self.comparisons):
            residual = check['residual']
            passed = all(v == 0 for v in residual) if isinstance(residual, sp.MatrixBase) else residual == 0
            check['equal'] = bool(passed)
            self.raw['comparisons'][index]['equal'] = bool(passed)
            if not passed:
                self.report['scientific_errors'].append(self.stage+':'+check['name'])
        self.raw['validation_complete'] = True
        self.persist(validated=True, validation_complete=True,
                     all_equal=all(c['equal'] for c in self.comparisons), checks=len(self.comparisons))
        boundary()


def capture(rec, name, value):
    rec.put(name, value)
    return value


def zero_checks(rec, name, matrix):
    rec.compare(name, matrix, sp.SparseMatrix.zeros(*matrix.shape))


def stage_a(rec):
    rec.begin('spinor_algebra')
    rec.put('basis_conventions', dict(spinor=['upper_up', 'upper_down', 'lower_up', 'lower_down'],
                                      spatial=['x', 'y', 'z'], dirac_metric=[1, -1, -1, -1]))
    mat = capture(rec, 'matrices', matrices())
    alpha, beta, g5 = mat['alpha'], mat['beta'], mat['gamma5']
    gamma, pair = mat['gamma'], mat['pair']
    cd, ut = mat['charge_conjugation'], mat['time_reversal']
    metric = [1, -1, -1, -1]
    for mu in range(4):
        for nu in range(4):
            expected = 2*metric[mu]*sp.eye(4) if mu == nu else sp.zeros(4)
            rec.compare(f'clifford_{mu}_{nu}', gamma[mu]*gamma[nu]+gamma[nu]*gamma[mu], expected)
        zero_checks(rec, f'charge_conjugation_{mu}', cd*gamma[mu]*cd.inv()+gamma[mu].T)
        zero_checks(rec, f'pair_antisymmetry_{mu}', pair[mu].T+pair[mu])
        zero_checks(rec, f'light_chiral_{mu}', g5.T*pair[mu]+pair[mu]*g5)
        rec.compare(f'inversion_pair_{mu}', beta.T*pair[mu]*beta, metric[mu]*pair[mu])
        rec.compare(f'time_reversal_pair_{mu}', ut.T*pair[mu].conjugate()*ut,
                    (-1 if mu == 0 else 1)*pair[mu])
    for i, a in enumerate(alpha):
        rec.compare(f'inversion_alpha_{i}', beta.H*a*beta, -a)
        rec.compare(f'time_reversal_alpha_{i}', ut.H*a.conjugate()*ut, -a)
        zero_checks(rec, f'free_chiral_{i}', commutator(a, g5))
    rec.compare('inversion_beta', beta.H*beta*beta, beta)
    rec.compare('time_reversal_beta', ut.H*beta.conjugate()*ut, beta)
    for axis in (0, 2):
        u = (sp.eye(2)-sp.I*mat['pauli'][axis])/sp.sqrt(2)
        s = sp.kronecker_product(sp.eye(2), u)
        rotation = sp.Matrix(3, 3, lambda i, j: scalar(sp.trace(mat['pauli'][j]*u.H*mat['pauli'][i]*u)/2))
        rec.put(f'cubic_generator_{axis}', dict(u=u, S=s, R=rotation))
        zero_checks(rec, f'rotation_real_{axis}', rotation.conjugate()-rotation)
        rec.compare(f'rotation_orthogonal_{axis}', rotation.T*rotation, sp.eye(3))
        rec.compare(f'rotation_temporal_{axis}', s.T*pair[0]*s, pair[0])
        for i in range(3):
            rec.compare(f'rotation_spatial_{axis}_{i}', s.T*pair[i+1]*s,
                        sum((rotation[i, j]*pair[j+1] for j in range(3)), sp.zeros(4)))
    for i in range(3):
        rec.compare(f'boost_temporal_{i}', alpha[i].T*pair[0]+pair[0]*alpha[i], 2*pair[i+1])
        for j in range(3):
            rec.compare(f'boost_spatial_{i}_{j}', alpha[i].T*pair[j+1]+pair[j+1]*alpha[i],
                        2*pair[0] if i == j else sp.zeros(4))
    pauli_basis = [sp.kronecker_product(a, b)
                   for a in [sp.eye(2)]+mat['pauli'] for b in [sp.eye(2)]+mat['pauli']]
    rec.put('pauli_mass_basis', pauli_basis)
    anti = []
    for pindex, p in enumerate(pauli_basis):
        anti.append([])
        for aindex, a in enumerate(alpha):
            anti[-1].append(capture(rec, f'mass_anticommutator_{pindex}_{aindex}', a*p+p*a))
    rec.put('mass_map', dict(basis_order=['0', 'x', 'y', 'z'],
                              pauli_tensors=pauli_basis, anticommutators=anti))
    gram = sp.Matrix(16, 16, lambda i, j: scalar(sum(sp.trace(anti[i][k].H*anti[j][k]) for k in range(3))))
    rec.put('mass_gram', gram)
    rec.compare('complete_mass_gram', gram, sp.diag(48, 16, 16, 16, 48, 16, 16, 16,
                                                  0, 32, 32, 32, 0, 32, 32, 32))
    rec.compare('excluded_mass_y', commutator(g5, pauli_basis[8]), 2*sp.I*pauli_basis[12])
    rec.compare('excluded_mass_z', commutator(g5, pauli_basis[12]), -2*sp.I*pauli_basis[8])
    rec.finish()
    return mat


def ordered_tensor(pair):
    pairs = list(combinations(range(4), 2))
    tensor = sp.Matrix(6, 6, lambda row, col: scalar(4*sum(
        (1 if mu == 0 else -1)*sp.conjugate(pair[mu][pairs[row][0], pairs[row][1]])
        *pair[mu][pairs[col][0], pairs[col][1]] for mu in range(4))))
    return pairs, tensor


def ordered_terms(pairs, tensor, site=0):
    result = []
    h, l = 8*site, 8*site+4
    for row, (a, b) in enumerate(pairs):
        for col, (c, d) in enumerate(pairs):
            if tensor[row, col] != 0:
                result.append((tensor[row, col], ((True, l+a), (True, l+b),
                                                  (False, h+d), (False, h+c))))
    return result+adjoint_terms(result)


def expectation_word(vector, word, states):
    index = {state: i for i, state in enumerate(states)}
    value = 0
    for col, state in enumerate(states):
        if vector[col] == 0:
            continue
        answer = act(word, state)
        if answer is not None:
            target, sign = answer
            if target in index:
                value += sp.conjugate(vector[index[target]])*sign*vector[col]
    return scalar(value)


def contractions(vector, states, rec, name):
    covariance = capture(rec, name+'_covariance', sp.Matrix(8, 8, lambda i, j: expectation_word(vector, ((True, j), (False, i)), states)))
    anomalous = capture(rec, name+'_anomalous', sp.Matrix(8, 8, lambda i, j: expectation_word(vector, ((False, i), (False, j)), states)))
    return dict(covariance=covariance, anomalous=anomalous)


def stage_b(rec, mat):
    rec.begin('on_site')
    states = basis(8)
    rec.put('basis', states)
    pairs, tensor = ordered_tensor(mat['pair'])
    rec.put('ordered_pair_definition', dict(pair_indices=pairs, tensor=tensor))
    phi_h, phi_l = [], []
    for mu, m in enumerate(mat['pair']):
        phi_h.append(operator(pair_terms(m, 0), states))
        rec.put(f'phi_H_{mu}', phi_h[-1])
        phi_l.append(operator(pair_terms(m, 4), states))
        rec.put(f'phi_L_{mu}', phi_l[-1])
    rec.put('pair_annihilators', dict(heavy=phi_h, light=phi_l, pair_indices=pairs, tensor=tensor))
    o = capture(rec, 'O', sum(((1 if mu == 0 else -1)*phi_l[mu].H*phi_h[mu] for mu in range(4)), sp.zeros(256, cls=sp.SparseMatrix)))
    w = capture(rec, 'W', o+o.H)
    w_ordered = capture(rec, 'W_ordered', operator(ordered_terms(pairs, tensor), states))
    nh = capture(rec, 'N_H', sp.SparseMatrix(256, 256, {(i, i): (s & 15).bit_count() for i, s in enumerate(states)}))
    nl = capture(rec, 'N_L', sp.SparseMatrix(256, 256, {(i, i): (s >> 4).bit_count() for i, s in enumerate(states)}))
    q5 = capture(rec, 'Q5_L', operator(bilinear_terms(mat['gamma5'], 4), states))
    ph = capture(rec, 'parity_H', sp.SparseMatrix(256, 256, {(i, i): (-1)**(s & 15).bit_count() for i, s in enumerate(states)}))
    pl = capture(rec, 'parity_L', sp.SparseMatrix(256, 256, {(i, i): (-1)**(s >> 4).bit_count() for i, s in enumerate(states)}))
    rec.put('operators', dict(O=o, W=w, W_ordered=w_ordered, N_H=nh, N_L=nl, Q5_L=q5, parity_H=ph, parity_L=pl))
    zero_checks(rec, 'hermiticity', w-w.H)
    zero_checks(rec, 'independent_W', w-w_ordered)
    zero_checks(rec, 'heavy_charge_change', commutator(nh, o)+2*o)
    zero_checks(rec, 'light_charge_change', commutator(nl, o)-2*o)
    for name, charge in [('total_charge', nh+nl), ('light_axial', q5), ('heavy_parity', ph), ('light_parity', pl)]:
        zero_checks(rec, name, commutator(w, charge))
    known_k = sp.Matrix([[4, 0, 0, 0, 0, -4], [0, -8, 0, 0, 0, 0],
                         [0, 0, -4, -4, 0, 0], [0, 0, -4, -4, 0, 0],
                         [0, 0, 0, 0, -8, 0], [-4, 0, 0, 0, 0, 4]])
    rec.compare('known_pair_tensor', tensor, known_k)
    wedge_canonical = basis(4, 2)
    wedge_ordered = [sum(1 << i for i in occupied) for occupied in pairs]
    permutation = [wedge_canonical.index(v) for v in wedge_ordered]
    b_canonical = capture(rec, 'B3_canonical', operator(bilinear_terms(mat['alpha'][2]), wedge_canonical))
    b3 = capture(rec, 'B3_ordered', b_canonical.extract(permutation, permutation))
    v0, v3 = sp.Matrix([1, 0, 0, 0, 0, -1]), sp.Matrix([0, 0, 1, 1, 0, 0])
    common = capture(rec, 'common_boost_residual', b3*tensor+tensor*b3)
    mixed_boost = capture(rec, 'mixed_boost_residual', -b3*tensor+tensor*b3)
    b_v0 = capture(rec, 'B3_v0', b3*v0)
    k_v0 = capture(rec, 'K_v0', tensor*v0)
    k_v3 = capture(rec, 'K_v3', tensor*v3)
    mixed_action = capture(rec, 'mixed_boost_action', mixed_boost*v0)
    rec.put('boost_scope', dict(canonical_basis=wedge_canonical, ordered_basis=wedge_ordered,
                                permutation=permutation, B3_canonical=b_canonical, B3=b3,
                                K=tensor, common_residual=common, mixed_residual=mixed_boost,
                                v0=v0, v3=v3, B3v0=b_v0, Kv0=k_v0, Kv3=k_v3, mixed_action=mixed_action))
    zero_checks(rec, 'common_spinor_boost', common)
    rec.compare('B3_v0', b_v0, -2*v3)
    rec.compare('K_v0', k_v0, 8*v0)
    rec.compare('K_v3', k_v3, -8*v3)
    rec.compare('mixed_valley_boost_obstruction', mixed_action, 32*v3)
    product = sp.zeros(256, 1)
    product[3] = 1
    mixed = sp.zeros(256, 1)
    for h in (0, 4):
        for j in (1, 5):
            answer = act(((True, h), (True, j)), 0)
            target, sign = answer
            mixed[target] += sp.Rational(sign, 2)
    expected = sp.zeros(256, 1)
    expected[(1 << 4)+(1 << 5)] = 4
    expected[(1 << 6)+(1 << 7)] = -4
    rec.put('fixture_vectors', dict(product=product, mixed_slater=mixed, product_W_expected=expected))
    double = capture(rec, 'double_commutator', commutator(w, commutator(w, nl)))
    for name, vector in [('product', product), ('mixed_slater', mixed)]:
        data = contractions(vector, states, rec, name)
        rec.put(name+'_contractions', data)
        applied = capture(rec, name+'_W_applied', w*vector)
        norm = capture(rec, name+'_norm', (vector.H*vector)[0])
        mean = capture(rec, name+'_expectation_W', (vector.H*applied)[0])
        data.update(vector=vector, W_applied=applied, norm=norm, expectation_W=mean)
        rec.put(name, data)
        rec.compare(name+'_norm', data['norm'], sp.Integer(1))
        rec.compare(name+'_expectation', data['expectation_W'], sp.Integer(0 if name == 'product' else 2))
        if name == 'product':
            rec.compare('product_W_vector', data['W_applied'], expected)
            zero_checks(rec, 'product_mixed_covariance', data['covariance'][:4, 4:])
            zero_checks(rec, 'product_anomalous', data['anomalous'])
    squared = capture(rec, 'W_squared', w*w)
    product_squared = capture(rec, 'product_W_squared', (product.H*squared*product)[0])
    product_double = capture(rec, 'product_double', (product.H*double*product)[0])
    rec.put('connected_controls', dict(W_squared=squared, double_commutator=double,
                                      product_W_squared=product_squared, product_double=product_double))
    rec.compare('product_W_squared', product_squared, sp.Integer(32))
    rec.compare('product_double', product_double, sp.Integer(-128))
    coordinates = sp.symbols('wick0:56', real=True)
    mixed_block = sp.Matrix(4, 4, lambda i, j: coordinates[2*(4*i+j)]+sp.I*coordinates[2*(4*i+j)+1])
    fl = [coordinates[32+2*i]+sp.I*coordinates[33+2*i] for i in range(6)]
    fh = [coordinates[44+2*i]+sp.I*coordinates[45+2*i] for i in range(6)]
    polynomial = 0
    for row, (a, b) in enumerate(pairs):
        for col, (c, d) in enumerate(pairs):
            polynomial += tensor[row, col]*(mixed_block[a, c]*mixed_block[b, d]
                                           -mixed_block[a, d]*mixed_block[b, c]+fl[row]*fh[col])
    rec.put('wick_unexpanded', dict(coordinates=list(coordinates), B=mixed_block, F_L=fl, F_H_dc=fh, complex_polynomial=polynomial))
    polynomial = capture(rec, 'wick_polynomial', sp.expand(polynomial+sp.conjugate(polynomial)))
    origin = dict.fromkeys(coordinates, 0)
    wick_origin = capture(rec, 'wick_origin', polynomial.subs(origin))
    derivatives = []
    for i, v in enumerate(coordinates):
        derivatives.append(capture(rec, f'wick_derivative_{i}', sp.diff(polynomial, v).subs(origin)))
    rec.put('wick', dict(coordinates=list(coordinates), B=mixed_block, F_L=fl, F_H_dc=fh,
                         polynomial=polynomial, value_at_zero=wick_origin,
                         first_derivatives_at_zero=derivatives))
    rec.compare('wick_origin', wick_origin, sp.Integer(0))
    for i, v in enumerate(derivatives):
        rec.compare(f'wick_first_{i}', v, sp.Integer(0))
    rec.finish()
    return pairs, tensor


def mass_terms(beta, site, mass):
    return [(mass*c, word) for c, word in bilinear_terms(beta, 8*site)]


def stage_c(rec, mat, pairs, tensor):
    rec.begin('bond_current')
    states = basis(16, 2)
    rec.put('basis', states)
    t = capture(rec, 'T', operator(bond_terms(mat['alpha'][2]), states))
    mass0 = capture(rec, 'mass0', operator(mass_terms(mat['beta'], 0, sp.Rational(3, 5)), states))
    mass1 = capture(rec, 'mass1', operator(mass_terms(mat['beta'], 1, sp.Rational(3, 5)), states))
    h0 = capture(rec, 'H0', t+mass0+mass1)
    w0 = capture(rec, 'W0', operator(ordered_terms(pairs, tensor, 0), states))
    w1 = capture(rec, 'W1', operator(ordered_terms(pairs, tensor, 1), states))
    w = capture(rec, 'W', w0+w1)
    d0 = capture(rec, 'D0', t/2+mass1)
    dw = capture(rec, 'DW', w1)
    rec.put('operands', dict(T=t, mass0=mass0, mass1=mass1, H0=h0, W0=w0, W1=w1, W=w, D0=d0, DW=dw))
    q0 = sp.I*commutator(h0, d0)
    rec.put('Q0', q0)
    q1 = sp.I*commutator(h0, dw)+sp.I*commutator(w, d0)
    rec.put('Q1', q1)
    q2 = sp.I*commutator(w, dw)
    rec.put('Q2', q2)
    direct = sp.I*commutator(h0+w, d0+dw)
    rec.put('Q_direct', direct)
    bond = sp.I*commutator(t, w1-w0)/2
    rec.put('Q_bond', bond)
    zero_checks(rec, 'dimer_Q0', q0)
    zero_checks(rec, 'no_quadratic_interaction', q2)
    zero_checks(rec, 'bond_identity', q1-bond)
    zero_checks(rec, 'direct_identity', direct-q0-q1)
    row, col = states.index((1 << 4)+(1 << 7)), states.index((1 << 0)+(1 << 9))
    rec.put('witness', dict(row=row, column=col, bra=[4, 7], ket=[0, 9], value=q1[row, col]))
    rec.compare('nonzero_current_witness', q1[row, col], sp.Integer(1))
    e1 = capture(rec, 'e1', d0+dw)
    e0 = capture(rec, 'e0', h0+w-e1)
    current = capture(rec, 'local_j', sp.I*commutator(e0, e1))
    continuity0 = capture(rec, 'local_continuity_0', sp.I*commutator(h0+w, e0)+current)
    continuity1 = capture(rec, 'local_continuity_1', sp.I*commutator(h0+w, e1)-current)
    moment_residual = capture(rec, 'local_current_moment', current-direct)
    rec.put('local_continuity', dict(e0=e0, e1=e1, j=current,
                                    residual0=continuity0, residual1=continuity1,
                                    current_residual=moment_residual))
    zero_checks(rec, 'local_continuity_0', continuity0)
    zero_checks(rec, 'local_continuity_1', continuity1)
    zero_checks(rec, 'local_current_moment', moment_residual)
    h = sp.zeros(24)
    x = sp.diag(*[site for site in range(3) for j in range(8)])
    for site in range(3):
        h[8*site:8*site+4, 8*site:8*site+4] = sp.Rational(3, 5)*mat['beta']
    for site in range(2):
        for species in (0, 4):
            i, j = 8*site+species, 8*(site+1)+species
            h[i:i+4, j:j+4] = -sp.I*mat['alpha'][2]/2
            h[j:j+4, i:i+4] = sp.I*mat['alpha'][2]/2
    rec.put('free_chain_H0', h)
    rec.put('free_chain_X', x)
    dipole = capture(rec, 'free_chain_dipole', (x*h+h*x)/2)
    q_direct = capture(rec, 'free_chain_Q_direct', sp.I*commutator(h, dipole))
    q_squared = capture(rec, 'free_chain_Q_squared', commutator(x, h*h)/(2*sp.I))
    q_twostep = sp.zeros(24)
    for component in range(8):
        q_twostep[component, 16+component] = -sp.I/4
        q_twostep[16+component, component] = sp.I/4
    rec.put('free_chain_Q_two_step', q_twostep)
    rec.put('free_chain', dict(basis=[[site, 'H' if j<4 else 'L', j%4]
                                     for site in range(3) for j in range(8)],
                                X=x, H0=h, energy_dipole=dipole, Q_direct=q_direct,
                                Q_squared=q_squared, Q_two_step=q_twostep))
    zero_checks(rec, 'chain_squared_identity', q_direct-q_squared)
    zero_checks(rec, 'chain_two_step', q_direct-q_twostep)
    for component in range(8):
        rec.compare(f'chain_entry_{component}', q_direct[component, 16+component], -sp.I/4)
    rec.finish()


def spinors(mass, sine, mat, rec, label):
    sine = sp.Matrix(sine)
    key = mass, tuple(sine)
    if key in SPINOR_CACHE:
        return capture(rec, label+'_reused', SPINOR_CACHE[key])
    rec.put(label+'_inputs', dict(mass=mass, sine_momentum=sine))
    energy = capture(rec, label+'_energy', sp.sqrt(mass**2+(sine.T*sine)[0]))
    sigma_d = sum((sine[i]*mat['pauli'][i] for i in range(3)), sp.zeros(2))
    top = sp.sqrt((energy+mass)/(2*energy))
    result = capture(rec, label+'_spinors', sp.Matrix.vstack(top*sp.eye(2), top*sigma_d/(energy+mass)).applyfunc(scalar))
    h = capture(rec, label+'_H', mass*mat['beta']+sum((sine[i]*mat['alpha'][i] for i in range(3)), sp.zeros(4)))
    eigen = capture(rec, label+'_eigen_residual', (h*result-energy*result).applyfunc(scalar))
    ortho = capture(rec, label+'_orthonormality_residual', (result.H*result-sp.eye(2)).applyfunc(scalar))
    data = dict(mass=mass, sine_momentum=sine, energy=energy, spinors=result, H=h,
                eigen_residual=eigen, orthonormality_residual=ortho)
    rec.put(label+'_leg', data)
    SPINOR_CACHE[key] = data
    return data


def amplitude_case(rec, name, heavy, light, mat, expected_average, expected_tensor=None, metadata=None):
    rec.put(name+'_external_legs', dict(heavy=heavy, light=light, metadata=metadata))
    currents_h, currents_l = [], []
    for mu, m in enumerate(mat['pair']):
        currents_h.append(capture(rec, f'{name}_JH_{mu}', (heavy[0]['spinors'].T*m*heavy[1]['spinors']).applyfunc(scalar)))
        currents_l.append(capture(rec, f'{name}_JL_{mu}', (light[0]['spinors'].T*m*light[1]['spinors']).applyfunc(scalar)))
    tensor = capture(rec, name+'_tensor', sp.Matrix(4, 4, lambda t, s: scalar(4*sum(
        (1 if mu == 0 else -1)*sp.conjugate(currents_l[mu][t//2, t%2])
        *currents_h[mu][s//2, s%2] for mu in range(4)))))
    average = capture(rec, name+'_spin_average', scalar(sum(sp.conjugate(v)*v for v in tensor)/4))
    conventions = dict(incoming_valleys=[[0, 0, 0], [0, 0, 0]],
                       outgoing_valleys=[[0, 0, 0], [0, 0, 0]],
                       spin_pair_order=['up_up', 'up_down', 'down_up', 'down_down'],
                       tensor_rows='outgoing light ordered spins', tensor_columns='incoming heavy ordered spins',
                       stripped_prefactor='U/V_s times crystal momentum delta')
    conventions.update(metadata or {})
    energy_residual = capture(rec, name+'_energy_residual', scalar(sum(v['energy'] for v in heavy)-sum(v['energy'] for v in light)))
    rec.put(name, dict(heavy=heavy, light=light, metadata=conventions, pair_currents_H=currents_h,
                       pair_currents_L=currents_l, tensor=tensor, spin_average=average,
                       energy_conservation_residual=energy_residual))
    for family, legs in [('H', heavy), ('L', light)]:
        for leg, data in enumerate(legs):
            zero_checks(rec, f'{name}_{family}{leg}_eigen', data['eigen_residual'])
            zero_checks(rec, f'{name}_{family}{leg}_orthonormal', data['orthonormality_residual'])
    rec.compare(name+'_energy_conservation', energy_residual, sp.Integer(0))
    rec.compare(name+'_spin_average', average, expected_average)
    if expected_tensor is not None:
        rec.compare(name+'_tensor', tensor, expected_tensor)
    return tensor


def stage_d(rec, mat):
    rec.begin('amplitudes')
    mass = sp.Rational(3, 5)
    axes = [sp.Matrix([int(i == j) for i in range(3)]) for j in range(3)]
    leg_count = 0
    def leg(m, sine):
        nonlocal leg_count
        leg_count += 1
        return spinors(m, sine, mat, rec, f'leg_{leg_count}')
    rest = leg(mass, sp.zeros(3, 1))
    light_z = [leg(0, sign*axes[2]) for sign in (1, -1)]
    h = [leg(mass, sign*sp.Rational(4, 5)*axes[2]) for sign in (1, -1)]
    amplitude_case(rec, 'same_valley_threshold', [rest, rest],
                   [leg(0, sign*mass*axes[2]) for sign in (1, -1)], mat, sp.Integer(0), sp.zeros(4),
                   dict(valleys=[[0, 0, 0]]*4))
    expected = sp.diag(-sp.Rational(32, 5), 0, 0, -sp.Rational(32, 5))
    amplitude_case(rec, 'collinear', h, light_z, mat, sp.Rational(512, 25), expected)
    perpendicular = amplitude_case(rec, 'perpendicular', h,
                                    [leg(0, sign*axes[0]) for sign in (1, -1)], mat, sp.Rational(256, 25))
    # Independent pair-current formula from the contract (d_H along z, n_L along x).
    def cross_sigma(v):
        return [v[(i+1)%3]*mat['pauli'][(i+2)%3]-v[(i+2)%3]*mat['pauli'][(i+1)%3] for i in range(3)]
    jh = [-sp.I*mat['pauli'][1]*v for v in cross_sigma(sp.Rational(4, 5)*axes[2])]
    rec.put('perpendicular_JH_formula', jh)
    jl = [-sp.I*mat['pauli'][1]*v for v in cross_sigma(axes[0])]
    rec.put('perpendicular_JL_formula', jl)
    formula = sp.Matrix(4, 4, lambda t, s: scalar(-4*sum(
        sp.conjugate(jl[i][t//2, t%2])*jh[i][s//2, s%2] for i in range(3))))
    rec.put('perpendicular_formula', dict(J_H_spatial=jh, J_L_spatial=jl, tensor=formula))
    rec.compare('perpendicular_tensor_formula', perpendicular, formula)
    r = sp.Symbol('r', real=True)
    energy = mass*(1+r*r)/(1-r*r)
    dz = 2*mass*r/(1-r*r)
    rec.put('symbolic_family_parameters', dict(r=r, mass=mass, energy=energy, d_z=dz))
    symbolic, symbolic_light = [], []
    for sign in (1, -1):
        u = capture(rec, f'symbolic_H_{sign}_spinors', sp.Matrix.vstack(sp.eye(2), sign*r*mat['pauli'][2])/sp.sqrt(1+r*r))
        hh = capture(rec, f'symbolic_H_{sign}_H', mass*mat['beta']+sign*dz*mat['alpha'][2])
        eigen_h = capture(rec, f'symbolic_H_{sign}_eigen_residual', (hh*u-energy*u).applyfunc(scalar))
        ortho_h = capture(rec, f'symbolic_H_{sign}_orthonormality_residual', (u.H*u-sp.eye(2)).applyfunc(scalar))
        symbolic.append(dict(mass=mass, sine_momentum=sp.Matrix([0, 0, sign*dz]), energy=energy,
                             spinors=u, H=hh, eigen_residual=eigen_h, orthonormality_residual=ortho_h))
        rec.put(f'symbolic_heavy_leg_{sign}', symbolic[-1])
        ul = capture(rec, f'symbolic_L_{sign}_spinors', sp.Matrix.vstack(sp.eye(2), sign*mat['pauli'][2])/sp.sqrt(2))
        hl = capture(rec, f'symbolic_L_{sign}_H', sign*energy*mat['alpha'][2])
        eigen_l = capture(rec, f'symbolic_L_{sign}_eigen_residual', (hl*ul-energy*ul).applyfunc(scalar))
        ortho_l = capture(rec, f'symbolic_L_{sign}_orthonormality_residual', (ul.H*ul-sp.eye(2)).applyfunc(scalar))
        symbolic_light.append(dict(mass=sp.Integer(0), sine_momentum=sp.Matrix([0, 0, sign*energy]),
                                   energy=energy, spinors=ul, H=hl,
                                   eigen_residual=eigen_l, orthonormality_residual=ortho_l))
        rec.put(f'symbolic_light_leg_{sign}', symbolic_light[-1])
    sym_expected = sp.diag(-16*r/(1+r*r), 0, 0, -16*r/(1+r*r))
    sym_tensor = amplitude_case(rec, 'symbolic_collinear', symbolic, symbolic_light, mat,
                                128*r*r/(1+r*r)**2, sym_expected,
                                dict(domain='real |r|<1; physical on-shell axis at a=1 has |r|<=1/2'))
    derivative = sym_tensor.diff(r).subs(r, 0)
    rec.put('symbolic_derivative', derivative)
    rec.compare('first_nonzero_derivative', derivative, sp.diag(-16, 0, 0, -16))
    directions = axes+[sp.ones(3, 1)/sp.sqrt(3)]
    for bits in range(8):
        c = [(bits >> i) & 1 for i in range(3)]
        dc = sp.diag(*[(-1)**b for b in c])
        for nindex, n in enumerate(directions):
            b = mat['pauli'][1]*(sp.eye(2)-sum((n[i]*mat['pauli'][i] for i in range(3)), sp.zeros(2))
                                       *sum(((dc*n)[i]*mat['pauli'][i] for i in range(3)), sp.zeros(2)))/2
            rec.put(f'xor_{bits}_direction_{nindex}_B_c', b)
            expected = sp.Matrix(4, 4, lambda t, s: scalar(4*sp.conjugate(b[t//2, t%2])*mat['pauli'][1][s//2, s%2]))
            rec.put(f'xor_{bits}_direction_{nindex}_expected_tensor', expected)
            f = sum(c[i]*n[i]**2 for i in range(3))
            k = [sp.asin(mass*n[i]) for i in range(3)]
            incoming_momenta = [sp.zeros(3, 1), sp.pi*sp.Matrix(c)]
            outgoing_momenta = [sp.Matrix(k), sp.pi*sp.Matrix(c)-sp.Matrix(k)]
            momentum_residual = (sum(incoming_momenta, sp.zeros(3, 1))
                                 -sum(outgoing_momenta, sp.zeros(3, 1))).applyfunc(scalar)
            data = dict(incoming_valleys=[[0, 0, 0], c], outgoing_valleys=[[0, 0, 0], c],
                        residual_k=k, incoming_momenta=incoming_momenta, outgoing_momenta=outgoing_momenta,
                        crystal_momentum_residual=momentum_residual, D_c=dc, n=n, B_c=b, f_c=f)
            tensor = amplitude_case(rec, f'xor_{bits}_direction_{nindex}', [rest, rest],
                                    [leg(0, mass*n), leg(0, -mass*dc*n)],
                                    mat, scalar(16*f), expected, data)
            zero_checks(rec, f'xor_{bits}_direction_{nindex}_momentum', momentum_residual)
            if c == [0, 0, 1] and nindex == 2:
                rec.compare('intervalley_nonzero_entry', tensor[1, 1], sp.Integer(4))
    rec.finish()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Reviewed box directory required')
    if SPEC_SHA == 'PENDING_FINAL_CONTRACT':
        raise RuntimeError('Final mathematical contract has not been frozen')
    for name, pin in {**INPUT_PINS, SPEC: SPEC_SHA}.items():
        if digest(ROOT/name) != pin:
            raise RuntimeError('Input/spec binding mismatch: '+name)
    if (ROOT/'vertex_gate_result.json').exists():
        raise FileExistsError('Existing allocation; no retry')
    allowed = {'execution.json', 'execution.json.tmp', 'stdout.log', 'stderr.log'}
    if any(p.suffix in ('.json', '.jsonl', '.log', '.tmp', '.npz') and p.name not in allowed
           for p in ROOT.iterdir()):
        raise FileExistsError('Existing scientific product; no retry')
    resource.setrlimit(resource.RLIMIT_AS, (2*2**30, 2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (100, 110))
    def stop(sig, frame):
        STOP.append(signal.Signals(sig).name)
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGXCPU, stop)
    signal.alarm(110)
    rec = Record()
    started = time.monotonic()
    try:
        mat = stage_a(rec)
        pairs, tensor = stage_b(rec, mat)
        stage_c(rec, mat, pairs, tensor)
        stage_d(rec, mat)
    except BaseException as exc:
        rec.report['status'] = 'incomplete'
        rec.report['operational_error'] = type(exc).__name__+': '+str(exc)
        if rec.raw is not None and not rec.report['stages'][rec.stage].get('validated'):
            try:
                rec.raw['operational_error'] = rec.report['operational_error']
                if not rec.report['stages'][rec.stage]['raw_complete']:
                    rec.raw['status'] = 'partial_exception'
                rec.persist()
            except BaseException as preservation:
                rec.report['preservation_error'] = type(preservation).__name__+': '+str(preservation)
    finally:
        signal.alarm(0)
        rec.report['elapsed_seconds'] = time.monotonic()-started
        rec.report['handled_signals'] = STOP
        complete = all(rec.report['stages'][s]['raw_complete'] and
                       rec.report['stages'][s]['validation_complete'] for s in STAGES)
        if complete:
            rec.report['status'] = 'complete_checks_failed' if rec.report['scientific_errors'] else 'complete'
            if rec.report['operational_error']:
                rec.report['status'] += '_with_operational_error'
        else:
            rec.report['status'] = 'incomplete'
        rec.save_report()
    print(json.dumps(dict(status=rec.report['status'], scientific_errors=rec.report['scientific_errors'],
                          missing_raw_products=rec.report['missing_raw_products'])))
    if not rec.report['target_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

