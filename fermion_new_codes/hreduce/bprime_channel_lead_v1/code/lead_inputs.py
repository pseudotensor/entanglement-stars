"""Input-only binding of native F1 columns and six independent B11 levels.

Callers admit the permitted-box runtime and bind exact source files first.
This module never loads a source transport Gram or propagation consequence.
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

SUPPLY_INDICES = np.arange(0, 128, 8, dtype=np.int64)
DFT_ROWS = np.array([0, 1, 4, 7], dtype=np.int64)
PREP_ROOT = Path('<internal>/prep_v5')
PREP_PATHS = {role: PREP_ROOT/(stem+'_prep.npz') for role, stem in (
    ('full', 'F1_s16_d0'), ('supply', 'F1_s16_d0_sup'), ('exit', 'F1_s16_d0_exit'))}
PREP_SHA256 = {
    'full': '312aba4d605e39970407357da3cf1937d0846fea3dcf8e7bf2d716c095a26c92',
    'supply': 'ac9e2d4737ed7a8e93dbcc920bdf95ef284962cd6b423aaf9fe1a2b808668b3e',
    'exit': '76fe487140dc809e443ad0bb2a49bbddfa606c25bc7f12243e183893f0cfccab'}
B11_JSON_SHA256 = {
    'F1': '93b36ecfe12de5dd07f48221052b011b2035d2516ce5bf8fff5d803c67b02d9c',
    'F2': 'df283b7aa671d85effb4fccbda38062bc6ed93f1adbf166dbb72175613ba62c0'}


def opnorm(value):
    """Induced Euclidean norm for a finite-column state or small matrix."""
    x = np.asarray(value)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError('Finite matrix required for operator norm')
    gram = x@x.conj().T if x.shape[0] <= x.shape[1] else x.conj().T@x
    return float(np.sqrt(max(0., float(np.linalg.eigvalsh(gram)[-1]))))


def metric_normalizer(states):
    """States are row columns; return common right-factor in column notation."""
    x = np.asarray(states)
    q = x.conj()@x.T
    eig, vec = np.linalg.eigh(q)
    if not np.all(np.isfinite(eig)) or eig[0] <= 0 or eig[0] <= 1e-12*eig[-1]:
        raise ValueError('Frozen full-rank input metric gate failed; no column is removed')
    factor = (vec/np.sqrt(eig)[None, :])@vec.conj().T
    normalized = factor.T@x
    return normalized, {'metric': q, 'eigenvalues': eig, 'right_factor': factor}


def embed_subcube(u, halfside=40, subhalf=12):
    u = np.asarray(u)
    if u.ndim != 2 or u.shape[0] != 4*(2*subhalf+1)**3:
        raise ValueError('Spin-major subcube columns required')
    n, s = 2*halfside+1, 2*subhalf+1
    result = np.zeros((u.shape[1], 4, n, n, n), dtype=np.complex128)
    sl = slice(halfside-subhalf, halfside+subhalf+1)
    result[:, :, sl, sl, sl] = u.T.reshape(u.shape[1], 4, s, s, s)
    return result.reshape(u.shape[1], -1)


def load_native_inputs(pinned, native_class):
    """Pinned full/supply/exit roles must have path and admitted SHA256."""
    from lead_runtime import require_identity
    bindings, blocks = {}, {}
    names = ('N', 'vx', 'vy', 'vz', 'U_sub', 'e', 'H', 'Hs', 'L', 'T0', 'R_I', 'n_supply')
    for role, path in PREP_PATHS.items():
        entry = pinned[role]
        if Path(entry['path']).resolve() != path.resolve() or entry['sha256'] != PREP_SHA256[role]:
            raise ValueError('Wrong original F1 input role/path')
        bindings[role] = require_identity(path, entry['sha256'])
        with np.load(path, allow_pickle=False) as archive:
            blocks[role] = {name: archive[name] for name in names}
    full = blocks['full']
    config = tuple(float(full[k]) for k in ('H', 'Hs', 'L', 'T0', 'R_I'))
    if config != (40., 12., 16., .0049735919716217296, 9.246593046624408):
        raise ValueError('Wrong native F1 basis configuration')
    for role, block in blocks.items():
        count = 256 if role == 'full' else 128
        if block['U_sub'].shape != (62500, count) or block['U_sub'].dtype != np.complex128:
            raise ValueError('Wrong original basis shape/dtype')
        if block['e'].shape != (count,) or block['e'].dtype != np.float64:
            raise ValueError('Wrong original eigenvalue metadata')
        if not np.all(np.isfinite(block['U_sub'])) or not np.all(np.isfinite(block['e'])):
            raise ValueError('Nonfinite original basis')
        for name in names:
            if name not in ('U_sub', 'e', 'n_supply') and not np.array_equal(block[name], full[name]):
                raise ValueError('Split/full input convention differs: '+role+' '+name)
    for key in ('N', 'vx', 'vy', 'vz'):
        if full[key].shape != (81, 81, 81) or full[key].dtype != np.float64 or not np.all(np.isfinite(full[key])):
            raise ValueError('Malformed given native field '+key)
    if np.any(full['N'] <= 0):
        raise ValueError('Nonpositive given native lapse')
    exact_columns = {}
    for index, column in enumerate(full['U_sub'].T):
        digest = hashlib.sha256(np.ascontiguousarray(column).tobytes()).hexdigest()
        if digest in exact_columns:
            raise ValueError('Duplicate full-basis column')
        exact_columns[digest] = index
    arrays, report = {}, {'bindings': bindings, 'checks': [], 'pilot_indices': SUPPLY_INDICES.tolist()}
    def check(name, passed, measured=None):
        report['checks'].append({'name': name, 'passed': bool(passed), 'measured': measured})
    mappings = []
    for role in ('supply', 'exit'):
        block = blocks[role]
        mapping = []
        for column, energy in zip(block['U_sub'].T, block['e']):
            digest = hashlib.sha256(np.ascontiguousarray(column).tobytes()).hexdigest()
            index = exact_columns.get(digest)
            if index is None or not np.array_equal(column, full['U_sub'][:, index]) or energy != full['e'][index]:
                raise ValueError('Split/full exact input association failed')
            mapping.append(index)
        mappings.extend(mapping)
        arrays[role+'_full_indices'] = np.asarray(mapping)
        q = block['U_sub'].conj().T@block['U_sub']
        arrays[role+'_metric'] = q
        eig = np.linalg.eigvalsh(q)
        arrays[role+'_metric_spectrum'] = eig
        check(role+'_metric_rank', eig[0] > 1e-12*eig[-1] and eig[0] > 0, float(eig[0]))
        check(role+'_metric_max_entry', np.max(np.abs(q-np.eye(128))) <= 1e-10,
              float(np.max(np.abs(q-np.eye(128)))))
    check('full256_exact_partition', sorted(mappings) == list(range(256)))
    t0 = float(full['T0'])
    check('supply_open_window', np.all((blocks['supply']['e'] > 0)&(blocks['supply']['e'] < t0)))
    check('exit_open_window', np.all((blocks['exit']['e'] < 0)&(blocks['exit']['e'] > -t0)))
    coordinates = np.arange(-12, 13)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    mask = x*x+y*y+z*z < float(full['R_I'])**2
    sl = slice(28, 53)
    flow = np.array([full['vx'], full['vy'], full['vz']])
    native = native_class(full['N'][sl, sl, sl], flow[:, sl, sl, sl], mask=mask)
    for role in ('supply', 'exit'):
        block = blocks[role]
        states = block['U_sub'].T
        outside = np.linalg.norm((states.reshape(128, 4, 25, 25, 25)*(~mask)[None, None]).reshape(128, -1), axis=1)
        residual = native.h_apply(states)-states*block['e'][:, None]
        residual_norm = np.linalg.norm(residual, axis=1)
        arrays[role+'_retained_residual'] = residual
        arrays[role+'_outside_norm'] = outside
        check(role+'_retained_support', np.max(outside) <= 1e-12, float(np.max(outside)))
        # Same native input convention gate as the completed task17, retained.
        check(role+'_retained_eigenpair', np.max(residual_norm) <= 1e-8, float(np.max(residual_norm)))
        arrays[role+'_subcube'] = block['U_sub']
        arrays[role+'_energies'] = block['e']
    supply = blocks['supply']['U_sub'][:, SUPPLY_INDICES]
    dft = np.exp(2j*np.pi*np.arange(16)[:, None]*DFT_ROWS[None, :]/16)/4
    arrays['pilot_native'] = embed_subcube(supply)
    arrays['variable_probe_native'] = embed_subcube(supply@dft)
    arrays['pilot_indices'], arrays['DFT_rows'], arrays['DFT_coefficients'] = SUPPLY_INDICES, DFT_ROWS, dft
    arrays['native_N'], arrays['native_flow'] = full['N'], flow
    report['all_passed'] = all(row['passed'] for row in report['checks'])
    report['unavailable_standing_checks'] = ['full-operator spectrum extrema', 'physical prepared E_inv', 'stored first collar flux']
    return report, arrays


def select_b11_inputs(family, kappa, radial_n, radial_v, radial_class, source_energies):
    """Independent retained eigensystem before source loss values are opened."""
    radius = {'F1': 9.246593046299292, 'F2': 15.403124155906166}[family]
    nsites = len(np.arange(1., math_ceil(radius)))
    if 2*nsites != {'F1': 18, 'F2': 30}[family]:
        raise ValueError('Wrong B11 retained site count')
    operator = radial_class(kappa, radial_n[:nsites], radial_v[:nsites])
    matrix = operator.h_apply(np.eye(2*nsites, dtype=np.complex128)).T
    hermitian_defect = opnorm(matrix-matrix.conj().T)/max(1., operator.B)
    energies, vectors = np.linalg.eigh(matrix)
    positive = np.flatnonzero(energies > 1e-10)
    negative = np.flatnonzero(energies < -1e-10)
    negative = negative[np.argsort(np.abs(energies[negative]), kind='stable')]
    if len(positive) < 3 or len(negative) < 3:
        raise ValueError('Missing B11 retained levels')
    for sign in (positive, negative):
        if len(sign) > 3 and abs(energies[sign[2]]-energies[sign[3]]) <= 1e-10:
            raise ValueError('B11 boundary eigencluster requires source selected subspace')
    selected = np.r_[positive[:3], negative[:3]]
    for column in range(len(energies)):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        vectors[:, column] *= np.conj(vectors[pivot, column])/abs(vectors[pivot, column])
    residual = matrix@vectors-vectors*energies[None, :]
    defect = opnorm(residual)/max(1., operator.B)
    orthogonality = opnorm(vectors.conj().T@vectors-np.eye(2*nsites))
    source_energies = np.asarray(source_energies, dtype=float)
    if source_energies.shape != (6,):
        raise ValueError('Exactly six declared source input energies required')
    differences = np.abs(energies[selected, None]-source_energies[None, :])
    admissible = [order for order in itertools.permutations(range(6))
                  if all(differences[i, j] <= 1e-8 for i, j in enumerate(order))]
    clusters = []
    for index in selected:
        members = np.flatnonzero(np.abs(energies-energies[index]) <= 1e-10)
        clusters.append(members)
    individual = np.array([len(members) == 1 for members in clusters])
    if admissible:
        representative = np.asarray(admissible[0])
        difference = energies[selected]-source_energies[representative]
        column_map = np.full(6, -1, dtype=np.int64)
        ambiguity_in_cluster = True
        for i in range(6):
            options = sorted({order[i] for order in admissible})
            if len(options) == 1 and individual[i]:
                column_map[i] = options[0]
            else:
                individual[i] = False
                if np.ptp(source_energies[options]) > 1e-10:
                    ambiguity_in_cluster = False
    else:
        representative = np.full(6, -1, dtype=np.int64)
        column_map = representative.copy()
        difference = np.full(6, np.nan)
        individual[:] = False
        ambiguity_in_cluster = False
    selected_set = set(map(int, selected))
    full_cluster_bound = all(set(map(int, members)).issubset(selected_set) for members in clusters)
    checks = {'eigenpair': defect <= 5e-12, 'orthogonality': orthogonality <= 5e-12,
              'raw_Hermiticity': hermitian_defect <= 5e-12,
              'source_input_energy': bool(admissible),
              'source_association_ambiguity_only_in_included_cluster': ambiguity_in_cluster,
              'selected_clusters_complete': full_cluster_bound}
    return {'family': family, 'kappa': kappa, 'R_I': radius, 'checks': checks,
            'all_passed': all(checks.values()), 'scaled_eigenpair_defect': defect,
            'orthogonality_defect': orthogonality, 'raw_Hermiticity_defect': hermitian_defect,
            'source_energy_permutation_count': len(admissible),
            'source_association_unique': len(admissible) == 1,
            'individual_column_comparison_available': individual.tolist(),
            'scope': 'Included eigenclusters compare as subspaces; ambiguous individual columns are withheld'}, {
            'retained_matrix': matrix, 'energies': energies, 'vectors': vectors,
            'residual': residual, 'selected_indices': selected, 'initial': vectors[:, selected].T,
            'source_input_energy': source_energies, 'source_column_map': column_map,
            'energy_multiset_representative_permutation': representative,
            'individual_comparison_available': individual,
            'selected_cluster_membership': np.array([np.isin(np.arange(len(energies)), members) for members in clusters]),
            'source_energy_feasible_mask': differences <= 1e-8,
            'source_energy_feasible_permutations': np.asarray(admissible, dtype=np.int64).reshape(-1, 6),
            'energy_difference': difference}


def math_ceil(value):
    # Local integer helper avoids a roundoff-dependent np.arange endpoint.
    import math
    return math.ceil(value)


def b11_declared_inputs(family):
    """Convert only declared input numbers, leaving result literals lexical.

    Source JSON is parsed with numeric tokens represented as strings; only
    this explicit input allowlist is converted. No source NPZ is opened.
    """
    from lead_runtime import require_identity
    path = Path('<internal>/b11')/('b11_'+family+'_s16.json')
    binding = require_identity(path, B11_JSON_SHA256[family])
    raw = json.loads(path.read_text(), parse_int=str, parse_float=str,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite source JSON token')))
    expected = {'L': 16., 'sigma': 2., 'nearest': 3., 'win': 3.,
                'T_over_L': 100., 'dt_over_L': .25, 'bin_over_L': 10., 'lead': 200., 'rc': 50.}
    args = raw['args']
    if args['family'] != family or any(float(args[key]) != value for key, value in expected.items()):
        raise ValueError('Wrong B11 declared experiment configuration')
    t0 = float(raw['T0'])
    if t0 != 1/(4*np.pi*16):
        raise ValueError('Wrong declared B11 input energy unit')
    inputs = {}
    for kappa in (1, 2, 3):
        case = raw['kappa'][str(kappa)]
        ratios = np.asarray([float(x) for x in case['e_over_T0']])
        if ratios.shape != (6,) or not np.all(np.isfinite(ratios)):
            raise ValueError('Malformed B11 source input energies')
        expected_radius = {'F1': 9.246593046299292, 'F2': 15.403124155906166}[family]
        if (float(case['R_I']) != expected_radius or
                int(case['retained_dim']) != {'F1': 18, 'F2': 30}[family] or
                int(case['n_supply']) != 3 or int(case['n_exit']) != 3):
            raise ValueError('B11 declared retained geometry/input counts differ')
        inputs[kappa] = {'energies': ratios*t0, 'e_over_T0': ratios,
                         'R_I': float(case['R_I']), 'retained_dim': int(case['retained_dim']),
                         'n_supply': int(case['n_supply']), 'n_exit': int(case['n_exit'])}
    return {'binding': binding, 'configuration': expected, 'T0': t0,
            'input_cases': inputs, 'result_numeric_tokens_converted': False,
            'result_npz_opened': False}
