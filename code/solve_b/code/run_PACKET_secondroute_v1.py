"""Own packet propagation and full-state retention; formatting/readout separate."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from scipy.linalg import eig_banded
from numpy.lib.format import open_memmap
from cpu_runtime import require_remote_cpu
from packet_core_v1 import (operators, lower_band, bare_packet, projected_packet,
    construction_scale, exponential_action, validate_history, pure_distance,
    midpoint_fractions, validate_roster, verified_controls, pure_distance_details, propagation_step, held_ledger, spectral_admission, source_snapshot, packet_labels, validate_step_association)

BASE = '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11'
PINNED = {'.json': '79b79bd15e5ba5ab3cb467af985c165c39c6a69c6bfe9ca360d438811a7d6372',
          '_fields.npz': 'a7d74d1b5e8516caa68f8249e7794572ffa315ed92eccde824552e9b8d3f32bb'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def expectation(states, matrix):
    norms = np.sum(abs(states)**2, axis=0)
    return np.real(np.sum(states.conj()*(matrix@states), axis=0))/norms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--channel', type=int, choices=(1, 5), required=True)
    args = ap.parse_args()
    runtime = require_remote_cpu(args.output)
    if any(args.output.glob('*_states.npy')):
        raise RuntimeError('existing propagation outputs; no silent overwrite')
    controls_sha = verified_controls(args.output.parent/'known_answers.json', args.spec)
    bindings = {BASE+suffix: digest(BASE+suffix) for suffix in PINNED}
    if any(bindings[BASE+s] != h for s, h in PINNED.items()):
        raise RuntimeError('source identity changed')
    raw = json.loads(Path(BASE+'.json').read_text())
    if raw['args']['kappa_sea'] != 10 or raw['args']['m'] != .2 or raw['args']['rs'] != 64:
        raise RuntimeError('source parameter/channel contract mismatch')
    with np.load(BASE+'_fields.npz', allow_pickle=False) as archive:
        fields = {key: archive[key] for key in ('T', 'R', 'N_applied', 'v_applied', 'eps_tot', 'J_tot', 'g', 'ell', 'tau')}
    records = raw['records']
    validate_history(records, fields)
    r, n, v = fields['R'], fields['N_applied'], fields['v_applied']
    kappa = args.channel
    ones, zero = np.ones(len(r)), np.zeros(len(r))
    op = operators(r, ones, zero, kappa)
    started = time.monotonic()
    e, u = eig_banded(lower_band(op['h0']), lower=True, check_finite=False)
    free = spectral_admission(op['h0'], e, u, .2)
    free.update({'coarse_bound': .2+kappa+1., 'seconds': time.monotonic()-started})
    np.save(args.output/'free_eigenvalues.npy', e)
    np.save(args.output/'free_eigenvectors.npy', u)
    write(args.output/'free_projector.json', free)
    positive, negative = u[:, e > 0], u[:, e < 0]
    centres = [('centre', 4.), ('inner', records[97]['r_in']), ('mid', (records[97]['r_in']+records[97]['R_t'])/2), ('outer', records[97]['R_t'])]
    rounded = {'centre': 4., 'inner': 23.24, 'mid': 30.9, 'outer': 38.56}
    momentum = [-.45, -.3, .3, .45]
    conventions = ['amplitude', 'probability'] if kappa == 1 else ['amplitude']
    labels, states, raw_states, preparations = [], [], [], []
    for convention in conventions:
        for label, centre in centres:
            for k in momentum:
                raw_state, raw_norm = bare_packet(r, centre, k, convention=convention)
                state, retained = projected_packet(raw_state, positive)
                rounded_raw, _ = bare_packet(r, rounded[label], k, convention=convention)
                rounded_state, _ = projected_packet(rounded_raw, positive)
                negative_weight = float(np.linalg.norm(negative.conj().T@state)**2)
                if not np.isfinite(negative_weight) or not -1e-10 <= negative_weight <= 1+1e-10:
                    raise RuntimeError('preparation negative-weight range failure')
                labels.append([convention, label, k])
                states.append(state); raw_states.append(raw_state)
                preparations.append({'convention': convention, 'site_label': label, 'centre': float(centre), 'momentum': k,
                    'unprojected_norm': raw_norm, 'retained_fraction': retained,
                    'projected_norm': float(np.vdot(state, state).real),
                    'negative_band_weight': negative_weight,
                    'rounded_centre': rounded[label], 'rounded_overlap_real': float(np.vdot(state, rounded_state).real),
                    'rounded_overlap_imag': float(np.vdot(state, rounded_state).imag),
                    'placement_pure_distance': pure_distance(state, rounded_state),
                    'placement_distance_diagnostics': pure_distance_details(state, rounded_state)})
    expected = [[c, label, k] for c in conventions for label, _ in centres for k in momentum]
    validate_roster(labels, expected)
    validate_roster(labels, packet_labels(kappa, 'E'))
    states = np.array(states).T
    np.save(args.output/'initial_states.npy', states)
    np.save(args.output/'initial_unprojected.npy', np.array(raw_states).T)
    write(args.output/'preparation.json', {'kappa': kappa, 'labels': labels, 'packets': preparations})
    # All complete free eigenpairs are saved, so no projection/eigensolver rerun is needed.
    del positive, negative, u
    write(args.output/'input_bindings.json', {'controls_sha256': controls_sha, 'sources': bindings, 'spec_sha256': digest(args.spec),
        'code': {str(Path(__file__).name): digest(__file__), 'packet_core_v1.py': digest(Path(__file__).with_name('packet_core_v1.py'))},
        'source_snapshot': source_snapshot(args.spec),
        'free_projector_sha256': digest(args.output/'free_projector.json'),
        'free_eigenvalues_sha256': digest(args.output/'free_eigenvalues.npy'),
        'free_eigenvectors_sha256': digest(args.output/'free_eigenvectors.npy'),
        'preparation_sha256': digest(args.output/'preparation.json'),
        'runtime': runtime, 'numpy': np.__version__, 'scipy': scipy.__version__, 'python': sys.version,
        'platform': platform.platform(), 'channel': kappa,
        'formation_context': [{'record': j, 'T': records[j]['T'], 'T_over_rs': records[j]['T_over_rs'],
            'trapping_zeros': records[j].get('trapping_zeros'),
            'dense_pair_present': len(records[j].get('trapping_zeros',[])) == 2} for j in (95,96,97)], 'start_record': 97, 'last_record': 320})
    for mode, substeps in [('E', 1), ('I16', 16), ('I32', 32)]:
        count = 16 if mode == 'I32' else states.shape[1]
        current = states[:, :count].copy()
        saved = open_memmap(args.output/(mode+'_states.npy'), mode='w+', dtype=np.complex128, shape=(224, len(current), count))
        saved[0] = current
        ledger = []
        exponentials = []
        for j in range(97, 320):
            validate_step_association(j, j+1, j, j+1)
            old_op = operators(r, n[j], v[j], kappa)
            next_op = operators(r, n[j+1], v[j+1], kappa)
            before = current.copy()
            def step_scale(fraction):
                return construction_scale(r, (1-fraction)*n[j]+fraction*n[j+1],
                    (1-fraction)*v[j]+fraction*v[j+1], kappa)
            current, metas = propagation_step(old_op['h'], next_op['h'], current, 8., substeps, step_scale, held=mode == 'E')
            exponentials.extend({'record': j, **meta} for meta in metas)
            row = held_ledger(before, current, old_op['h'], next_op['h'])
            ledger.append({'record': j, **{key: value.tolist() for key, value in row.items()},
                'E_conservation_applicable': mode == 'E'})
            saved[j-96] = current
            saved.flush()
            if j % 16 == 0:
                print(json.dumps({'channel': kappa, 'mode': mode, 'record': j+1, 'norm_drift': float(np.max(abs(np.sum(abs(current)**2, axis=0)-1)))}), flush=True)
        write(args.output/(mode+'_ledger.json'), ledger)
        write(args.output/(mode+'_exponentials.json'), exponentials)
        write(args.output/(mode+'_complete.json'), {'start_record': 97, 'last_record': 320, 'T_first': 776., 'T_last': 2560.,
            'records': 224, 'intervals': 223, 'columns': count,
            'state_sha256': digest(args.output/(mode+'_states.npy')), 'labels': labels[:count],
            'ledger_sha256': digest(args.output/(mode+'_ledger.json')),
            'exponentials_sha256': digest(args.output/(mode+'_exponentials.json')),
            'ideal_tail_sum': sum(row['tail'] for row in exponentials)})
        del saved
    write(args.output/'producer_complete.json', {'channel': kappa, 'seconds': time.monotonic()-started, 'runtime': runtime})
    print(json.dumps({'channel': kappa, 'complete': True}), flush=True)

if __name__ == '__main__':
    main()
