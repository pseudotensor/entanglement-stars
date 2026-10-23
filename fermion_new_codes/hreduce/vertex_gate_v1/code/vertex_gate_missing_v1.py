"""Missing-only exact vertex completion; predecessor records are immutable inputs."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time

ROOT = Path('<internal>/continuation_v1')
OLD = ROOT.parent
CONTRACT = 'VERTEX_gate_missing_contract_v1.tex'
SPEC_SHA = '1c002616f0711c78687163bf3dbebb9b31e22c600773c7447321d2cf8ce357eb'
ORIGINAL_SPEC_SHA = '7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c'
STAGES = ['spinor_algebra', 'on_site', 'bond_current', 'amplitudes']
PATHS = {'spinor_algebra': OLD/'spinor_algebra.json', 'on_site': ROOT/'on_site_completion.json',
         'bond_current': ROOT/'bond_current.json', 'amplitudes': ROOT/'amplitudes.json'}
STOP = []
MODE_COUNT = 8
BASE = None
ADMISSION = {'admitted': [], 'decoded': [], 'status': 'admitting'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def plain_atomic(path, data):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        json.dump(data, stream, separators=(',', ':'), sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def boundary():
    if STOP:
        raise TimeoutError('Handled stop: '+','.join(STOP))


def identity(s):
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def read_bound(name, pin):
    path = OLD/name
    before_path = identity(path.lstat())
    descriptor = os.open(path, os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        s = os.fstat(stream.fileno())
        if not stat.S_ISREG(s.st_mode):
            raise ValueError('Regular predecessor file required: '+name)
        before = identity(s)
        body = stream.read()
        after = identity(os.fstat(stream.fileno()))
    after_path = identity(path.lstat())
    actual = hashlib.sha256(body).hexdigest()
    if not before_path == before == after == after_path or actual != pin:
        raise ValueError('Changed predecessor: '+name)
    ADMISSION['admitted'].append(dict(name=name, path=str(path), expected_sha256=pin,
                                      actual_sha256=actual, before=before, after=after))
    plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
    boundary()
    return body


def checked_int(value):
    if type(value) is not int:
        raise ValueError('JSON integer required')
    return value


def decode_scalar(data):
    if set(data) != {'expression', 'srepr', 'gaussian_rational'}:
        raise ValueError('Restricted exact scalar schema')
    gr = data['gaussian_rational']
    if set(gr) != {'real', 'imag'}:
        raise ValueError('Gaussian rational coordinates required')
    values = []
    for key in ('real', 'imag'):
        pair = gr[key]
        if type(pair) is not list or len(pair) != 2:
            raise ValueError('Rational pair required')
        p, q = map(checked_int, pair)
        if q <= 0 or math.gcd(p, q) != 1:
            raise ValueError('Canonical rational required')
        values.append(sp.Rational(p, q))
    value = values[0]+sp.I*values[1]
    if type(data['expression']) is not str or type(data['srepr']) is not str:
        raise ValueError('Opaque encoding strings required')
    BASE.EXACT_CACHE[value] = copy.deepcopy(data)
    return value


def decode_matrix(data, shape, pointer, source):
    if set(data) != {'shape', 'entries'} or data['shape'] != list(shape):
        raise ValueError('Unexpected matrix schema/shape: '+pointer)
    if any(type(v) is not int for v in data['shape']):
        raise ValueError('Integer shape required')
    if type(data['entries']) is not list:
        raise ValueError('Sparse entries must be a JSON list')
    entries = {}
    previous = (-1, -1)
    for row in data['entries']:
        if type(row) is not list or len(row) != 3:
            raise ValueError('Sparse triple required')
        i, j = checked_int(row[0]), checked_int(row[1])
        if not (0 <= i < shape[0] and 0 <= j < shape[1]) or (i, j) <= previous:
            raise ValueError('Sorted unique in-range indices required')
        value = decode_scalar(row[2])
        if value == 0:
            raise ValueError('Canonical sparse nonzero required')
        entries[i, j] = value
        previous = (i, j)
    result = sp.SparseMatrix(shape[0], shape[1], entries)
    ADMISSION['decoded'].append(dict(source=str(OLD/source), sha256=PINS[source],
                                     pointer=pointer, shape=list(shape)))
    plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
    boundary()
    return result


def native_act(word, state):
    def native(value):
        if type(value) is int or isinstance(value, sp.Integer):
            return int(value)
        raise TypeError('Exact non-Boolean integer CAR operand required')
    state = native(state)
    if not 0 <= state < 1 << MODE_COUNT:
        raise ValueError('Occupation mask outside active problem')
    normalized = []
    for create, mode in word:
        if type(create) is not bool:
            raise TypeError('Boolean creation flag required')
        mode = native(mode)
        if not 0 <= mode < MODE_COUNT:
            raise ValueError('Mode outside active problem')
        normalized.append((create, mode))
    return ORIGINAL_ACT(tuple(normalized), state)


def capture(rec, name, value):
    rec.put(name, value)
    return value


class Record:
    def __init__(self, predecessor):
        self.report = dict(status='running', spec_sha256=SPEC_SHA, input_sha256=INPUT_PINS,
                           source_sha256=digest(Path(__file__)), sympy_version=sp.__version__,
                           python_version=sys.version, stage_order=STAGES,
                           paper_commit='daaf6fbd05ff029e771f8fc1ebcd8f24ccf17dca',
                           units='a=1, U=lambda_X/a^3; no half in pairing; CAR-normalized spinors',
                           resources=dict(threads=1, address_space_bytes=2*2**30,
                                          handled_wall_seconds=94, external_wall_seconds=104,
                                          cpu_soft_seconds=84, cpu_hard_seconds=94),
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
        self.inherited_names = set()
        self.report['stages']['spinor_algebra'] = copy.deepcopy(predecessor['stages']['spinor_algebra'])
        self.report.update(original_spec_sha256=ORIGINAL_SPEC_SHA, predecessor_root=str(OLD),
                           predecessor_result_sha256=PINS['vertex_gate_result.json'],
                           predecessor_execution_sha256=PINS['execution.json'],
                           predecessor_error=predecessor['operational_error'],
                           original_attempt=predecessor,
                           component_source_sha256=PINS['vertex_gate_exact_v1.py'],
                           reuse_manifest=str(ROOT/'reuse_manifest.json'),
                           reused_stage_A=dict(path=str(PATHS['spinor_algebra']), sha256=PINS['spinor_algebra.json'], checks=74))
        self.save_report()

    def save_report(self):
        stages = self.report['stages']
        self.report['missing_raw_products'] = [PATHS[s].name for s in STAGES if not stages[s]['saved']]
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
        self.report['remaining_named_outputs'] = {
            s: [k for k in REQUIRED[s] if k not in (self.raw['constructed'] if self.stage == s and self.raw else {})]
            for s in STAGES[1:] if not stages[s]['raw_complete']}
        self.report['missing_equality_decisions'] = {}
        for stage in STAGES[1:]:
            saved = self.raw if self.stage == stage and self.raw else {}
            decided = {c['name'] for c in saved.get('comparisons', []) if type(c.get('equal')) is bool}
            self.report['missing_equality_decisions'][stage] = ([] if stages[stage]['validation_complete']
                else [name for name in PLANS[stage] if name not in decided])
        self.report['remaining_leg_outputs'] = ([] if stages['amplitudes']['raw_complete'] else
            remaining_legs(self.raw if self.stage == 'amplitudes' and self.raw else {}))
        atomic(ROOT/'vertex_gate_missing_result.json', self.report)

    def begin(self, name):
        boundary()
        self.stage = name
        self.raw = dict(stage=name, constructed={}, comparisons=[], status='building')
        if name == 'amplitudes':
            self.raw['leg_obligations'] = {label: dict(branch='unresolved', complete=False) for label in LEG_LABELS}
        self.inherited_names = set()
        self.comparisons = []
        self.save_report()

    def persist(self, **state_updates):
        atomic(PATHS[self.stage], self.raw)
        self.report['stages'][self.stage].update(saved=True, sha256=digest(PATHS[self.stage]), **state_updates)
        self.save_report()

    def put(self, name, value):
        if name in self.inherited_names:
            raise ValueError('Inherited object must not be overwritten: '+name)
        self.raw['constructed'][name] = encode(value)
        self.raw.setdefault('newly_constructed', []).append(name)
        if self.stage == 'amplitudes':
            for label, obligation in self.raw['leg_obligations'].items():
                branch = obligation['branch']
                if branch != 'unresolved':
                    obligation['full_leg_saved'] = obligation['first_saved_name'] in self.raw['constructed']
                    obligation['complete'] = all(label+suffix in self.raw['constructed'] for suffix in LEG_SUFFIXES[branch])
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
        if ([c['name'] for c in self.raw['comparisons']] != PLANS[self.stage]
                or any(c.get('comparison_status') != 'residual_returned' for c in self.raw['comparisons'])):
            raise ValueError('Incomplete prescribed residual-name plan')
        if self.stage == 'amplitudes' and remaining_legs(self.raw):
            raise ValueError('Incomplete prescribed spinor-leg obligations')
        if any(k not in self.raw['constructed'] for k in REQUIRED[self.stage]):
            raise ValueError('Incomplete prescribed primary raw keys')
        self.raw['status'] = 'returned' 
        self.persist(raw_complete=True)
        boundary()
        for index, check in enumerate(self.comparisons):
            if 'equal' in check:
                continue
            residual = check['residual']
            passed = all(v == 0 for v in residual) if isinstance(residual, sp.MatrixBase) else residual == 0
            check['equal'] = bool(passed)
            self.raw['comparisons'][index]['equal'] = bool(passed)
            if not passed:
                self.report['scientific_errors'].append(self.stage+':'+check['name'])
            if index == len(PLANS[self.stage])-1:
                self.raw['validation_complete'] = True
                self.persist(validated=True, validation_complete=True,
                             all_equal=all(c['equal'] for c in self.comparisons), checks=len(self.comparisons))
            else:
                self.persist()
            boundary()
        self.raw['validation_complete'] = True
        self.persist(validated=True, validation_complete=True,
                     all_equal=all(c['equal'] for c in self.comparisons), checks=len(self.comparisons))
        boundary()

def stage_b_tail(rec, saved):
    states, pairs, tensor, w, product, mixed, expected, double = saved
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


PINS = {'03_model.tex': 'c4ccb5d297e3b42097cb2aaa5be9087ca8c99d8d4d8ac29ed4380b491b615bbb', 'ReviewB_pcq5_v1.md': '2bacd47b093cdef79448dd5e4a6c3c4923c7829cfc5855d2ea117bb43630db9f', 'ReviewB_pcq6_v1.md': '9279dacb90acc5bd804629e6058e52765469181e3b823e6dadcdc02be6c149af', 'ReviewA_pcq4_v1.md': '1a4818713fb720ce41e3a90e0aa6d9a7e16c92b351d4bd718e9434371848dcb2', 'ReviewA_pcq5_v1.md': 'eafb1c6487282f65bf3edd5e3416643d2b2451a3e2a6ea39e267a5578c1167b2', 'JOINT_VISION_page_mock_signed_v1.md': 'fe8648e986ce8382b22c13a3ba6b27289d49229ae32f034182e1098428253279', 'VERTEX_gate_author_v1.tex': '7b32632b79d26b771192a86fb721eaeb85326938ba589fb953c502a594943d7c', 'execution.json': '4f48ecac00cb4591c85fb5911ddbf456246e4aff70148aaf8abf29e6675e8497', 'on_site.json': '37155ac55d9ba1ce59d456afcb3bee2060c8784d42ba6060a83a1005cbc5c5f8', 'run_vertex_gate_exact_v1.py': '61b8b82354a9fcbac56e641239c44301c888f25c23055d553235fd3e17704a41', 'spinor_algebra.json': 'af73d5cf386c0787076310488b55d8d0d9b47002a56b31a97665a8307630fabb', 'stderr.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'stdout.log': '3300b3f59deb0433fad1338d9c67953b4d666b491e5b3ba3bb4fbbcce1c65654', 'vertex_gate_exact_v1.py': '50fa4a6c44b91295a59ac3d91181902aa963166bc328fca6c616ecc6a901f26b', 'vertex_gate_result.json': 'e8898fbdda68834f507b55ee94232deab6791139e225c25ef1d35ccec4c59547'}
EXPECTED_COUNTS = {'on_site': 80, 'bond_current': 18, 'amplitudes': 430}
REQUIRED = {
    'on_site': ['product_covariance', 'product_anomalous', 'product_contractions',
                'product_W_applied', 'product_norm', 'product_expectation_W', 'product',
                'mixed_slater_covariance', 'mixed_slater_anomalous', 'mixed_slater_contractions',
                'mixed_slater_W_applied', 'mixed_slater_norm', 'mixed_slater_expectation_W',
                'mixed_slater', 'W_squared', 'product_W_squared', 'product_double',
                'connected_controls', 'wick_unexpanded', 'wick_polynomial', 'wick_origin', 'wick']
               + ['wick_derivative_'+str(i) for i in range(56)],
    'bond_current': ['basis', 'T', 'mass0', 'mass1', 'H0', 'W0', 'W1', 'W', 'D0', 'DW',
                     'operands', 'Q0', 'Q1', 'Q2', 'Q_direct', 'Q_bond', 'witness', 'e0', 'e1',
                     'local_j', 'local_continuity_0', 'local_continuity_1', 'local_current_moment',
                     'local_continuity', 'free_chain_H0', 'free_chain_X', 'free_chain_dipole',
                     'free_chain_Q_direct', 'free_chain_Q_squared', 'free_chain_Q_two_step', 'free_chain'],
    'amplitudes': ['same_valley_threshold', 'collinear', 'perpendicular', 'symbolic_collinear',
                   'perpendicular_JH_formula', 'perpendicular_JL_formula', 'perpendicular_formula',
                   'symbolic_family_parameters', 'symbolic_derivative']
                  + ['xor_'+str(b)+'_direction_'+str(n) for b in range(8) for n in range(4)]}


# Literal operational name plans from the sole paired source comparison.
PLANS = {
    'on_site': ['hermiticity','independent_W','heavy_charge_change','light_charge_change',
                'total_charge','light_axial','heavy_parity','light_parity','known_pair_tensor',
                'common_spinor_boost','B3_v0','K_v0','K_v3','mixed_valley_boost_obstruction',
                'product_norm','product_expectation','product_W_vector','product_mixed_covariance',
                'product_anomalous','mixed_slater_norm','mixed_slater_expectation',
                'product_W_squared','product_double','wick_origin']+['wick_first_'+str(i) for i in range(56)],
    'bond_current': ['dimer_Q0','no_quadratic_interaction','bond_identity','direct_identity',
                     'nonzero_current_witness','local_continuity_0','local_continuity_1',
                     'local_current_moment','chain_squared_identity','chain_two_step']
                    + ['chain_entry_'+str(i) for i in range(8)]}
INITIAL_CASES = ['same_valley_threshold','collinear','perpendicular','symbolic_collinear']
XOR_CASES = ['xor_'+str(b)+'_direction_'+str(n) for b in range(8) for n in range(4)]
CASES = INITIAL_CASES+XOR_CASES

def case_decisions(q, tensor=True):
    return [q+'_'+family+str(leg)+'_'+kind for family in ('H','L') for leg in range(2)
            for kind in ('eigen','orthonormal')] + [q+'_energy_conservation',q+'_spin_average'] + ([q+'_tensor'] if tensor else [])

PLANS['amplitudes'] = (case_decisions(INITIAL_CASES[0])+case_decisions(INITIAL_CASES[1])
    +case_decisions(INITIAL_CASES[2],False)+['perpendicular_tensor_formula']
    +case_decisions(INITIAL_CASES[3])+['first_nonzero_derivative'])
for q in XOR_CASES:
    PLANS['amplitudes'] += case_decisions(q)+[q+'_momentum']
    if q == 'xor_4_direction_2':
        PLANS['amplitudes'].append('intervalley_nonzero_entry')
REQUIRED['amplitudes'] = [q+suffix for q in CASES for suffix in
    (['_external_legs']+[f'_J{family}_{mu}' for mu in range(4) for family in ('H','L')]
     +['_tensor','_spin_average','_energy_residual',''])]
REQUIRED['amplitudes'] += ['perpendicular_JH_formula','perpendicular_JL_formula',
                          'perpendicular_formula','symbolic_family_parameters','symbolic_derivative']
for sign in ('1','-1'):
    for family, full in (('H','heavy'),('L','light')):
        REQUIRED['amplitudes'] += ['symbolic_'+family+'_'+sign+suffix for suffix in
            ('_spinors','_H','_eigen_residual','_orthonormality_residual')]
        REQUIRED['amplitudes'].append('symbolic_'+full+'_leg_'+sign)
REQUIRED['amplitudes'] += [q+suffix for q in XOR_CASES for suffix in ('_B_c','_expected_tensor')]
LEG_LABELS = ['leg_'+str(i) for i in range(1,74)]
LEG_SUFFIXES = {'compute': ['_inputs','_energy','_spinors','_H','_eigen_residual',
                           '_orthonormality_residual','_leg','_cache_origin'],
                'reuse': ['_cache_origin','_reused']}


def remaining_legs(raw):
    result = []
    constructed = raw.get('constructed', {})
    obligations = raw.get('leg_obligations', {})
    for label in LEG_LABELS:
        obligation = obligations.get(label, dict(branch='unresolved',complete=False))
        if obligation['branch'] == 'unresolved':
            result.append(dict(label=label, branch='unresolved', allowed_suffixes=LEG_SUFFIXES))
        else:
            missing = [label+suffix for suffix in LEG_SUFFIXES[obligation['branch']]
                       if label+suffix not in constructed]
            if missing:
                result.append(dict(label=label, **obligation, missing=missing))
    return result


def decoded_sequence(value, expected, pointer, source):
    if value != expected:
        raise ValueError('Unexpected saved interface: '+pointer)
    def only_ints(v):
        if type(v) is list:
            return all(only_ints(w) for w in v)
        return type(v) is int
    if not only_ints(value):
        raise ValueError('Exact JSON integer interface required')
    answer = copy.deepcopy(value)
    ADMISSION['decoded'].append(dict(source=str(OLD/source), sha256=PINS[source], pointer=pointer))
    plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
    boundary()
    return answer


def restore_operands(a, b):
    m = a['constructed']['matrices']
    matrices = {}
    for key in ('pauli', 'identity', 'alpha', 'beta', 'gamma5', 'charge_conjugation',
                'gamma', 'pair', 'time_reversal'):
        shape = (2, 2) if key in ('pauli', 'identity') else (4, 4)
        if key in ('pauli', 'alpha', 'gamma', 'pair'):
            count = 3 if key in ('pauli', 'alpha') else 4
            if type(m[key]) is not list or len(m[key]) != count:
                raise ValueError('Matrix-family interface mismatch')
            matrices[key] = [decode_matrix(v, shape, '/constructed/matrices/'+key+'/'+str(i),
                                           'spinor_algebra.json') for i, v in enumerate(m[key])]
        else:
            matrices[key] = decode_matrix(m[key], shape, '/constructed/matrices/'+key,
                                           'spinor_algebra.json')
    c = b['constructed']
    states = decoded_sequence(c['basis'], list(range(256)), '/constructed/basis', 'on_site.json')
    pairs = decoded_sequence(c['ordered_pair_definition']['pair_indices'],
                             [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],
                             '/constructed/ordered_pair_definition/pair_indices', 'on_site.json')
    tensor = decode_matrix(c['ordered_pair_definition']['tensor'], (6,6),
                           '/constructed/ordered_pair_definition/tensor', 'on_site.json')
    w = decode_matrix(c['W'], (256,256), '/constructed/W', 'on_site.json')
    fixtures = []
    for name in ('product', 'mixed_slater', 'product_W_expected'):
        fixtures.append(decode_matrix(c['fixture_vectors'][name], (256,1),
                                       '/constructed/fixture_vectors/'+name, 'on_site.json'))
    double = decode_matrix(c['double_commutator'], (256,256),
                            '/constructed/double_commutator', 'on_site.json')
    return matrices, (states, pairs, tensor, w, *fixtures, double)


def start_inherited_b(rec, b):
    rec.begin('on_site')
    rec.raw['constructed'] = copy.deepcopy(b['constructed'])
    rec.raw['comparisons'] = copy.deepcopy(b['comparisons'])
    rec.inherited_names = set(b['constructed'])
    rec.raw.update(inherited_names=sorted(rec.inherited_names), newly_constructed=[],
                   predecessor=dict(path=str(OLD/'on_site.json'), sha256=PINS['on_site.json'],
                                    status=b['status'], operational_error=b.get('operational_error'),
                                    constructed_pointer='/constructed', comparisons_pointer='/comparisons'))
    rec.persist()
    boundary()
    for check in rec.raw['comparisons']:
        residual = check['residual']
        if (check.get('comparison_status') != 'residual_returned' or 'equal' in check
                or set(residual) != {'shape','entries'} or residual['entries'] != []
                or type(residual['shape']) is not list or len(residual['shape']) != 2
                or any(type(v) is not int or v < 1 for v in residual['shape'])):
            raise ValueError('Saved residual is not the admitted empty sparse matrix')
        check['equal'] = True
        check['decision_origin'] = 'previously unreturned decision from saved empty exact residual'
        rec.comparisons.append(dict(name=check['name'], equal=True))
        rec.persist()
        boundary()


def main():
    global BASE, sp, ORIGINAL_ACT, scalar, encode, atomic, contractions, zero_checks, INPUT_PINS, MODE_COUNT
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Reviewed continuation box directory required')
    allowed = {'execution.json', 'execution.json.tmp', 'stdout.log', 'stderr.log'}
    if any(p.suffix in ('.json','.log','.tmp','.npz') and p.name not in allowed for p in ROOT.iterdir()):
        raise FileExistsError('Used continuation; no retry')
    if digest(ROOT/CONTRACT) != SPEC_SHA:
        raise ValueError('Continuation contract mismatch')
    resource.setrlimit(resource.RLIMIT_AS, (2*2**30, 2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (84,94))
    def stop(sig, frame):
        STOP.append(signal.Signals(sig).name)
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGXCPU, stop)
    signal.alarm(94)
    started = time.monotonic()
    ADMISSION.update(original_mathematical_sha256=ORIGINAL_SPEC_SHA,
                     original_root=str(OLD), current_source_sha256=digest(Path(__file__)),
                     current_wrapper_sha256=digest(ROOT/'run_vertex_gate_missing_v1.py'),
                     continuation_contract_sha256=SPEC_SHA, original_stage_states=None,
                     reused_pointers={'spinor_algebra.json': ['/constructed/matrices','/comparisons'],
                                      'on_site.json': ['/constructed','/comparisons']})
    plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
    rec = None
    try:
        bodies = {name: read_bound(name, pin) for name, pin in PINS.items()}
        a = json.loads(bodies['spinor_algebra.json'])
        b = json.loads(bodies['on_site.json'])
        predecessor = json.loads(bodies['vertex_gate_result.json'])
        execution = json.loads(bodies['execution.json'])
        if (predecessor['source_sha256'] != PINS['vertex_gate_exact_v1.py']
                or predecessor['spec_sha256'] != ORIGINAL_SPEC_SHA
                or predecessor['status'] != 'incomplete' or predecessor['target_complete'] is not False
                or predecessor['operational_error'] != "AttributeError: 'Zero' object has no attribute 'bit_count'"
                or len(a['comparisons']) != 74 or not all(c.get('equal') is True for c in a['comparisons'])
                or not predecessor['stages']['spinor_algebra']['validation_complete']
                or predecessor['stages']['spinor_algebra']['sha256'] != PINS['spinor_algebra.json']
                or predecessor['stages']['on_site']['sha256'] != PINS['on_site.json']
                or len(b['comparisons']) != 14 or b['status'] != 'partial_exception'
                or execution['exit_code'] != 1 or not execution['child_terminal_before_inventory']):
            raise ValueError('Predecessor status bindings differ from the admitted record')
        for name in ('vertex_gate_exact_v1.py', 'VERTEX_gate_author_v1.tex'):
            if execution['source_sha256'][name] != PINS[name]:
                raise ValueError('Executed source/spec binding mismatch')
        for name in ('spinor_algebra.json','on_site.json'):
            if execution['surviving_files'][name]['sha256'] != PINS[name]:
                raise ValueError('Execution raw binding mismatch')
        for stage in ('bond_current','amplitudes'):
            if (OLD/(stage+'.json')).exists() or predecessor['stages'][stage]['saved']:
                raise ValueError('Absent predecessor stage unexpectedly present')
        ADMISSION.update(original_stage_states=predecessor['stages'],
                         predecessor_status=predecessor['status'], predecessor_error=predecessor['operational_error'],
                         status='admitted')
        plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
        # Import pinned own definitions only. No guarded main, original Record, A/B or old builders run.
        module_spec = importlib.util.spec_from_file_location('vertex_owned_predecessor', OLD/'vertex_gate_exact_v1.py')
        BASE = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(BASE)
        sp = BASE.sp
        ORIGINAL_ACT = BASE.act
        BASE.act = native_act
        BASE.ROOT = ROOT
        BASE.STOP = STOP
        scalar, encode, atomic = BASE.scalar, BASE.encode, BASE.atomic
        contractions, zero_checks, INPUT_PINS = BASE.contractions, BASE.zero_checks, BASE.INPUT_PINS
        if BASE.SPINOR_CACHE:
            raise ValueError('New spinor cache must start empty')
        rec = Record(predecessor)
        mat, saved = restore_operands(a, b)
        ADMISSION['status'] = 'decoded'
        plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
        start_inherited_b(rec, b)
        stage_b_tail(rec, saved)
        MODE_COUNT = 16
        BASE.stage_c(rec, mat, saved[1], saved[2])
        original_spinors = BASE.spinors
        origins = {}
        def tracked_spinors(mass, sine, matrix_dictionary, recorder, label):
            key = mass, tuple(sine)
            reuse = key in BASE.SPINOR_CACHE
            first = origins[key] if reuse else label+'_leg'
            provenance = dict(first_saved_name=first, pointer='/constructed/'+first,
                              cache_key=dict(mass=mass, sine_momentum=list(sine)))
            recorder.raw['leg_obligations'][label] = encode(dict(branch='reuse' if reuse else 'compute',
                complete=False, full_leg_saved=first in recorder.raw['constructed'], **provenance))
            recorder.persist()
            boundary()
            if reuse:
                recorder.put(label+'_cache_origin', provenance)
                return original_spinors(mass, sine, matrix_dictionary, recorder, label)
            value = original_spinors(mass, sine, matrix_dictionary, recorder, label)
            origins[key] = first
            recorder.put(label+'_cache_origin', provenance)
            return value
        BASE.spinors = tracked_spinors
        BASE.stage_d(rec, mat)
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
        ADMISSION['operational_error'] = error
        if rec is not None:
            rec.report['operational_error'] = error
            if rec.raw is not None and not rec.report['stages'][rec.stage]['validation_complete']:
                try:
                    rec.raw['operational_error'] = error
                    if not rec.report['stages'][rec.stage]['raw_complete']:
                        rec.raw['status'] = 'partial_exception'
                    rec.persist()
                except BaseException as preservation:
                    rec.report['preservation_error'] = type(preservation).__name__+': '+str(preservation)
        else:
            plain_atomic(ROOT/'vertex_gate_missing_result.json', dict(status='admission_incomplete',
                         operational_error=error, target_complete=False, reuse_manifest=str(ROOT/'reuse_manifest.json')))
    finally:
        signal.alarm(0)
        ADMISSION.update(elapsed_seconds=time.monotonic()-started, handled_signals=STOP)
        plain_atomic(ROOT/'reuse_manifest.json', ADMISSION)
        if rec is not None:
            rec.report.update(elapsed_seconds=time.monotonic()-started, handled_signals=STOP)
            rec.save_report()
            complete = rec.report['target_complete']
            rec.report['status'] = ('complete_checks_failed' if rec.report['scientific_errors'] else 'complete') if complete else 'incomplete'
            if complete and rec.report['operational_error']:
                rec.report['status'] += '_with_operational_error'
            rec.save_report()
    if rec is None or not rec.report['target_complete']:
        raise SystemExit(1)
    print(json.dumps(dict(status=rec.report['status'], scientific_errors=rec.report['scientific_errors'])))


if __name__ == '__main__':
    main()
