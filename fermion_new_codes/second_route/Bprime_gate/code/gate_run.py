"""Staged basis admission, then independent reductions of immutable saved Grams."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from gate_runtime import admit, code_bindings, require_identity, verify_threads, check_memory, SPEC_SHA256


def recheck_inventory(inventory):
    for group in ('runs', 'preps'):
        for item in inventory[group].values():
            for key in ('json', 'npz', 'manifest'):
                if key in item:
                    require_identity(item[key]['path'], item[key]['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('admission', 'reduce'))
    parser.add_argument('--outdir', type=Path, required=True)
    for name in ('spec', 'intake', 'controls'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Bprime saved-Gram '+args.stage)
    if args.spec_sha256 != SPEC_SHA256:
        raise ValueError('Final specification identity required')
    bindings = {key: require_identity(getattr(args, key), getattr(args, key+'_sha256'))
                for key in ('spec','intake','controls')}
    source = code_bindings()
    inventory = json.loads(args.intake.read_text())
    if (not inventory['all_passed'] or inventory['code_bindings'] != source or
            inventory['spec_binding']['sha256'] != SPEC_SHA256):
        raise ValueError('Accepted current metadata-only intake required')
    recheck_inventory(inventory)
    import numpy as np
    import gate_algebra as algebra
    from gate_io import save_record, restore_record, logical_identity
    verify_threads(runtime, 'after_imports')
    check_memory(runtime, 'basis_workspace', 4*1024**3)
    controls = restore_record(args.controls)
    if not controls['all_passed'] or controls['source'] != source or controls['spec']['sha256'] != SPEC_SHA256:
        raise ValueError('Accepted exact-source manufactured controls required')
    report = {'stage': args.stage, 'bindings': bindings, 'source': source, 'runtime': runtime,
              'checks': [], 'historical_prep_bytes': 'unavailable: inherited profile hash is not a prep hash',
              'claim_scope': 'Own arithmetic in the recorded coefficient order; current prep values separately checked',
              'foreign_code_imported': False, 'physical_propagation': False}

    def check(label, passed, **evidence):
        report['checks'].append({'label': label, 'passed': bool(passed), **evidence})

    def load(role, kind, names):
        binding = inventory[kind][role]['npz']
        require_identity(binding['path'], binding['sha256'])
        with np.load(binding['path'], allow_pickle=False) as z:
            return {name: z[name] for name in names}

    try:
        if args.stage == 'admission':
            prep_names = ('U_sub','e','qF','n_supply','Hs','L')
            preps = {role: load(role, 'preps', prep_names) for role in inventory['preps']}
            basis_records = {}
            for role, data in preps.items():
                U, hs, L = data['U_sub'], int(data['Hs']), float(data['L'])
                finite = bool(np.all(np.isfinite(U)))
                expected_columns = 128 if role in ('prep_sup','prep_exit') else 256
                check(role+' basis shape/dtype', U.shape == (4*(2*hs+1)**3,expected_columns) and U.dtype == np.complex128)
                check(role+' finite basis', finite, finite_entries=int(np.count_nonzero(np.isfinite(U))), total_entries=int(U.size))
                check(role+' scale', L == 16.)
                if not finite:
                    raise ValueError('Nonfinite current prep basis')
                Q = U.conj().T@U
                basis_records[role] = {'logical_basis': logical_identity(U),
                    'column_identities': [logical_identity(U[:,j]) for j in range(U.shape[1])],
                    'Hs': hs, 'L': L, 'Q': Q, 'Q_minus_I_maximum': algebra.maximum(Q-np.eye(len(Q))),
                    'norm_squared': np.real(np.diag(Q)).copy()}
            report['current_preps'] = basis_records

            def common_rows(data, hs):
                own = int(data['Hs'])
                n, m = 2*own+1, 2*hs+1
                if own > hs:
                    raise ValueError('Common embedding must include every original row')
                U = data['U_sub'].reshape(4,n,n,n,-1)
                expanded = np.zeros((4,m,m,m,U.shape[-1]), dtype=np.complex128)
                offset = hs-own
                expanded[:,offset:offset+n,offset:offset+n,offset:offset+n,:] = U
                return expanded.reshape(4*m**3,-1)

            full = preps['kit_full']
            for role, indices in (('prep_full', np.arange(256)), ('prep_sup', np.arange(128)),
                                  ('prep_exit', np.arange(128,256)), ('kit_onset225', np.arange(256)),
                                  ('prep_onset225', np.arange(256))):
                hs = max(int(full['Hs']), int(preps[role]['Hs']))
                comparison = algebra.exact_basis(common_rows(full, hs), common_rows(preps[role], hs), indices)
                check(role+' exact current embedded columns', comparison['exact_equal'], **comparison)
            expected = {
                'baseline': ('kit_full', np.arange(256),128,.5,2.,0),
                '2c': ('kit_full', np.arange(256),128,.25,2.,0),
                '2b': ('kit_onset225', np.arange(256),128,.5,2.25,0),
                'x_full': ('kit_full', np.arange(256),128,.5,2.,0),
                'x_n32': ('kit_full', np.array(list(range(16))+list(range(128,144))),16,.5,2.,32),
                'run1_sup': ('prep_sup', np.arange(128),128,.5,2.,0),
                'run1_exit': ('prep_exit', np.arange(128),0,.5,2.,0)}
            run_records = {}
            for role, (prep_role, indices, ns, dt, onset, max_columns) in expected.items():
                data = load(role, 'runs', ('T','e','qF','C_in','n_supply'))
                meta = inventory['runs'][role]['input_metadata']
                prep = preps[prep_role]
                arguments, declaration = meta['args'], meta['prep_manifest']
                check(role+' declared timestep', arguments['dt'] == dt)
                check(role+' declared selection', arguments['max_columns'] == max_columns)
                check(role+' declared source family', declaration['family'] == 'F1')
                check(role+' declared scale/onset', declaration['L'] == 16. and declaration['onset_L'] == onset)
                check(role+' declared prep basename', Path(arguments['prep']).name == Path(inventory['preps'][prep_role]['npz']['path']).name)
                check(role+' supply count', int(data['n_supply']) == ns)
                n = len(indices)
                for name, shape, dtype in (('e',(n,),np.float64), ('qF',(n,n),np.complex128), ('C_in',(n,n),np.complex128)):
                    value = data[name]
                    check(role+' '+name+' shape/dtype/finite', value.shape == shape and value.dtype == dtype and np.all(np.isfinite(value)))
                wanted_e, wanted_q = prep['e'][indices], prep['qF'][np.ix_(indices,indices)]
                check(role+' exact declared energies', np.array_equal(data['e'], wanted_e), maximum=algebra.maximum(data['e']-wanted_e))
                check(role+' exact declared qF values', np.array_equal(data['qF'], wanted_q), maximum=algebra.maximum(data['qF']-wanted_q))
                for name in ('G','R'):
                    if name == 'R' and name not in inventory['runs'][role]['array_headers']:
                        continue
                    header = inventory['runs'][role]['array_headers'][name]
                    check(role+' '+name+' header', tuple(header['shape']) == (len(data['T']),n,n) and header['descr'] == '<c16')
                targets = [6.931471805599452*16]
                if role != 'x_full':
                    targets.append(69.31471805599453*16)
                if role in ('baseline','2c','2b'):
                    targets.append(1600.)
                matches = [algebra.exact_clock(data['T'], time) for time in targets]
                run_records[role] = {'prep_role': prep_role, 'indices': indices, 'n_supply': ns,
                    'T': data['T'], 'clock_hex': [float(t).hex() for t in data['T']],
                    'required_clock_indices': matches, 'e': data['e'],
                    'qF_identity': logical_identity(data['qF']), 'C_in_identity': logical_identity(data['C_in']),
                    'C_in_meaning': 'stored declared covariance; no entropy inference',
                    'qF_meaning': 'stored tagged current contraction; not the input norm metric'}
            report['runs'] = run_records
            report['scientific_G_R_payloads_opened'] = False
        else:
            if args.admission is None or args.admission_sha256 is None:
                raise ValueError('Exact separately accepted basis-admission record required')
            bindings['admission'] = require_identity(args.admission, args.admission_sha256)
            admission = restore_record(args.admission)
            if (not admission['all_passed'] or admission['stage'] != 'admission' or admission['source'] != source or
                    admission['bindings']['intake']['sha256'] != args.intake_sha256 or
                    admission['bindings']['controls']['sha256'] != args.controls_sha256):
                raise ValueError('Basis admission does not bind this exact source/input set')
            runs = {}
            for role in inventory['runs']:
                names = ('T','G')+ (('R',) if 'R' in inventory['runs'][role]['array_headers'] else ())
                runs[role] = load(role, 'runs', names)
                for name in names[1:]:
                    if not np.all(np.isfinite(runs[role][name])):
                        raise ValueError('Nonfinite original '+role+' '+name)
            from gate_reduce import reduce_grams
            report['results'] = reduce_grams(runs, 16.)
            check('required regression', report['results']['regression_passed'])
            check('endpoint timestep', report['results']['endpoint_passed'])
            report['scientific_G_R_payloads_opened'] = True
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        check('stage completed without exception', False)
    report['all_passed'] = bool(report['checks']) and all(item['passed'] for item in report['checks'])
    recheck_inventory(inventory)
    for binding in bindings.values():
        require_identity(binding['path'], binding['sha256'])
    if code_bindings() != source:
        raise ValueError('Source changed during saved-data stage')
    verify_threads(runtime, 'before_save')
    save_record(args.outdir, args.stage, report)
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
