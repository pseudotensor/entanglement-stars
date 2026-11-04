"""Read saved packet states into complete diagnostic arrays; no propagation."""
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from cpu_runtime import require_remote_cpu
from packet_core_v1 import (operators, row_constant, root_partition, applied_roots,
    velocity_parts, fourier_diagnostics, variance_value, validate_history, construction_scale, verified_controls, source_snapshot, admit_record_observables, validate_completed_states, packet_labels, validate_roster, validate_observable_schema)
from run_PACKET_secondroute_v1 import BASE, PINNED, digest, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--channel', type=int, choices=(1, 5), required=True)
    args = ap.parse_args()
    runtime = require_remote_cpu(args.output)
    controls_sha = verified_controls(args.output.parent/'known_answers.json', args.spec)
    producer = json.loads((args.output/'producer_complete.json').read_text())
    if producer['channel'] != args.channel:
        raise RuntimeError('wrong producer channel')
    if any(digest(BASE+suffix) != expected for suffix, expected in PINNED.items()):
        raise RuntimeError('source binding changed')
    records = json.loads(Path(BASE+'.json').read_text())['records']
    with np.load(BASE+'_fields.npz', allow_pickle=False) as archive:
        fields = {key: archive[key] for key in ('T', 'R', 'N_applied', 'v_applied', 'eps_tot', 'J_tot', 'g', 'ell', 'tau')}
    validate_history(records, fields)
    r = fields['R']; n = fields['N_applied']; v = fields['v_applied']
    kappa = args.channel
    bindings = json.loads((args.output/'input_bindings.json').read_text())
    if bindings['controls_sha256'] != controls_sha or bindings['source_snapshot'] != source_snapshot(args.spec):
        raise RuntimeError('producer source/control binding mismatch')
    for name in ('free_eigenvalues','free_eigenvectors','preparation','free_projector'):
        suffix = '.json' if name in ('preparation','free_projector') else '.npy'
        if digest(args.output/(name+suffix)) != bindings[name+'_sha256']:
            raise RuntimeError('free basis/preparation binding mismatch')
    free_e = np.load(args.output/'free_eigenvalues.npy')
    free_u = np.load(args.output/'free_eigenvectors.npy', mmap_mode='r')
    negative = np.asarray(free_u[:, free_e < 0])
    free_meta = json.loads((args.output/'free_projector.json').read_text())
    h0 = operators(r, np.ones(len(r)), np.zeros(len(r)), kappa)['h0']
    q = operators(r, np.ones(len(r)), np.zeros(len(r)), kappa)['q']
    roots = []
    for j in range(321):
        dense = records[j].get('trapping_zeros')
        pair = dense if root_partition(r, dense) is not None else None
        if pair is not None and (records[j].get('r_in') != pair[0] or records[j].get('R_t') != pair[1]):
            pair = None
        roots.append({'record': j, 'T': float(fields['T'][j]), 'dense': pair,
                      'applied': applied_roots(r, n[j], v[j]),
                      'converged': records[j].get('converged'),
                      'indicator': records[j].get('indicator'), 'regime': records[j].get('regime'),
                      'record_label': records[j].get('record_label'), 'solver': records[j].get('solver')})
    write(args.output/'roots_and_quality.json', roots)
    for mode in ('E', 'I16', 'I32'):
        complete = json.loads((args.output/(mode+'_complete.json')).read_text())
        path = args.output/(mode+'_states.npy')
        if digest(path) != complete['state_sha256']:
            raise RuntimeError('state binding changed')
        states = np.load(path, mmap_mode='r')
        count = complete['columns']
        prep = json.loads((args.output/'preparation.json').read_text())
        validate_roster(prep['labels'], packet_labels(kappa, 'E'))
        validate_completed_states(complete, states.shape, packet_labels(kappa, mode))
        for key in ('ledger','exponentials'):
            if digest(args.output/(mode+'_'+key+'.json')) != complete[key+'_sha256']:
                raise RuntimeError('step ledger/exponential binding mismatch')
        density = open_memmap(args.output/(mode+'_position_probability.npy'), mode='w+', dtype=float, shape=(224, 2048, count))
        momentum = open_memmap(args.output/(mode+'_momentum_probability.npy'), mode='w+', dtype=float, shape=(224, 2048, count))
        local_source = open_memmap(args.output/(mode+'_packet_source.npy'), mode='w+', dtype=float, shape=(224, 2048, count, 5))
        collected = {}
        def save(key, value):
            collected.setdefault(key, []).append(np.asarray(value))
        imaginary = []
        admissions = []
        for index, j in enumerate(range(97, 321)):
            psi = np.asarray(states[index])
            if not np.isfinite(psi).all():
                raise RuntimeError('nonfinite saved state')
            norm = np.sum(abs(psi)**2, axis=0)
            if np.any(norm <= 0):
                raise RuntimeError('zero saved norm')
            op = operators(r, n[j], v[j], kappa)
            veln, velv = velocity_parts(op, r)
            def expval(matrix, name):
                product = matrix@psi
                raw = np.sum(psi.conj()*product, axis=0)
                save(name+'_raw_real', raw.real)
                save(name+'_raw_imag', raw.imag)
                imaginary.append(float(np.max(abs(raw.imag)/norm)))
                return raw.real/norm, product
            eh, hpsi = expval(op['h'], 'energy_H')
            e0, h0psi = expval(h0, 'energy_0')
            en, _ = expval(op['hn'], 'energy_lapse')
            ev, _ = expval(op['hv'], 'energy_flow')
            current, qpsi = expval(q, 'current_Q')
            vn, _ = expval(veln, 'velocity_lapse')
            vv, _ = expval(velv, 'velocity_flow')
            second_h = np.sum(abs(hpsi)**2, axis=0)/norm
            second_0 = np.sum(abs(h0psi)**2, axis=0)/norm
            rawvh, vh, cliph = variance_value(second_h, eh)
            rawv0, v0, clip0 = variance_value(second_0, e0)
            pp = np.sum(abs(psi.reshape(2048, 2, count))**2, axis=1)/norm
            density[index] = pp
            grid, bins, pm, masks = fourier_diagnostics(psi.reshape(2048, 2, count))
            pm = pm/norm
            momentum[index] = pm
            source_e = np.real(np.sum(psi.reshape(2048, 2, count).conj()*h0psi.reshape(2048, 2, count), axis=1))/norm
            source_j = np.real(np.sum(psi.reshape(2048, 2, count).conj()*qpsi.reshape(2048, 2, count), axis=1))/norm
            constant, _, _ = row_constant(r, kappa, v[j])
            signed = source_e+v[j, :, None]*source_j
            cauchy = np.sqrt(pp)*constant[:, None]/2
            local_source[index] = np.stack((source_e, source_j, signed, np.broadcast_to(constant[:, None], signed.shape), cauchy), axis=-1)
            dense_mask = root_partition(r, roots[j]['dense'])
            applied = roots[j]['applied']
            applied_mask = root_partition(r, applied['roots']) if applied['status'] == 'resolved_pair' else None
            dense_share = dense_mask@pp if dense_mask is not None else np.full((3, count), np.nan)
            applied_share = applied_mask@pp if applied_mask is not None else np.full((3, count), np.nan)
            if applied_mask is None and applied['outer_available']:
                applied_share[2] = pp[r > applied['outer_root']].sum(axis=0)
            fixed_share = np.sum(pp[r > records[97]['R_t']], axis=0)
            centroid = r@pp
            second_r = r*r@pp
            rawvr, vr, clipr = variance_value(second_r, centroid)
            negative_weight = np.sum(abs(negative.conj().T@psi)**2, axis=0)/norm
            values = {'norm': norm, 'energy_H': eh, 'energy_0': e0, 'energy_lapse': en, 'energy_flow': ev,
                'second_H': second_h, 'second_0': second_0, 'variance_H_raw': rawvh, 'variance_H': vh,
                'variance_H_clipped': cliph, 'variance_0_raw': rawv0, 'variance_0': v0, 'variance_0_clipped': clip0,
                'current_Q': current, 'velocity_lapse': vn, 'velocity_flow': vv, 'velocity_total': vn+vv,
                'centroid': centroid, 'position_variance_raw': rawvr, 'position_variance': vr,
                'position_variance_clipped': clipr, 'mean_N': n[j]@pp, 'mean_v': v[j]@pp,
                'mean_N_minus_abs_v': (n[j]-abs(v[j]))@pp, 'negative_band_weight': negative_weight,
                'dense_shares': dense_share, 'applied_shares': applied_share, 'fixed_outer_share': fixed_share,
                'inside_12': np.sum(pp[r < 12], axis=0), 'first_four': np.sum(pp[:4], axis=0),
                'wall_64': np.sum(pp[-64:], axis=0), 'parseval_defect': pm.sum(axis=0)-1,
                'Q_symbol_mean': masks['Q_symbol']@pm, 'cosine_mean': masks['cosine_symbol']@pm,
                'packet_row_max_excess': np.max(abs(signed)-constant[:, None], axis=0),
                'packet_cauchy_max_excess': np.max(abs(signed)-cauchy, axis=0),
                'packet_row_max_adjusted_excess': np.max(abs(signed)-constant[:, None]-1e-10*np.maximum(1., constant[:, None]), axis=0),
                'packet_cauchy_max_adjusted_excess': np.max(abs(signed)-cauchy-1e-10*np.maximum(1., cauchy), axis=0),
                'radial_mean_band_excess': abs(e0)-free_meta['edge'],
                'radial_second_band_excess': second_0-free_meta['edge']**2,
                'homogeneous_mean_excess': e0-np.hypot(.2, 1.),
                'H_mean_band_excess': abs(eh)-construction_scale(r, n[j], v[j], kappa)/2,
                'H_second_band_excess': second_h-(construction_scale(r, n[j], v[j], kappa)/2)**2,
                'H_finer_bound': np.full(count, np.max(n[j])*(free_meta['edge']+free_meta['edge_allowance'])+np.max(abs(v[j]))/2),
                'H_finer_mean_excess': abs(eh)-(np.max(n[j])*(free_meta['edge']+free_meta['edge_allowance'])+np.max(abs(v[j]))/2),
                'H_finer_second_excess': second_h-(np.max(n[j])*(free_meta['edge']+free_meta['edge_allowance'])+np.max(abs(v[j]))/2)**2,
                'H_construction_bound': np.full(count, construction_scale(r, n[j], v[j], kappa)/2)}
            values.update({key+'_share': pm[mask].sum(axis=0) for key, mask in masks.items() if key in ('high','negative_cosine','opposed_current')})
            admissions.append({'record': j, **admit_record_observables(values, pp, pm, dense_mask is not None, applied_mask is not None, applied['outer_available'])})
            for label,bound in [('row',np.broadcast_to(constant[:,None],signed.shape)),('cauchy',cauchy)]:
                bad=(abs(signed)-bound)>1e-10*np.maximum(1.,bound)
                if bad.any():
                    admissions[-1]['failures']['packet_'+label+'_site_column']=np.argwhere(bad).tolist()
                    for site,column in np.argwhere(bad):
                        admissions[-1]['admitted_columns'][int(column)]=False
            for key, value in values.items():
                save(key, value)
            if index % 32 == 0:
                print(json.dumps({'readout': mode, 'channel': kappa, 'record': j}), flush=True)
        arrays = {key: np.array(value) for key, value in collected.items()}
        arrays.update({'record_indices': np.arange(97, 321), 'T': fields['T'][97:], 'momentum_grid': grid, 'momentum_bins': bins})
        validate_observable_schema(arrays,count)
        np.savez(args.output/(mode+'_observables.npz'), **arrays)
        density.flush(); momentum.flush(); local_source.flush()
        write(args.output/(mode+'_admission.json'), admissions)
        write(args.output/(mode+'_readout_complete.json'), {'controls_sha256': controls_sha, 'source_snapshot': source_snapshot(args.spec), 'records': 224, 'columns': count,
            'max_imaginary_expectation': max(imaginary), 'observables_sha256': digest(args.output/(mode+'_observables.npz')),
            'pointwise_source_columns': ['epsilon', 'J', 'epsilon_plus_WJ', 'row_bound', 'cauchy_bound'],
            'saved_state_sha256': complete['state_sha256'],
            'free_projector_sha256': digest(args.output/'free_projector.json'),
            'free_eigenvalues_sha256': digest(args.output/'free_eigenvalues.npy'),
            'free_eigenvectors_sha256': digest(args.output/'free_eigenvectors.npy'),
            'preparation_sha256': digest(args.output/'preparation.json'),
            'root_quality_sha256': digest(args.output/'roots_and_quality.json'),
            'admission_sha256': digest(args.output/(mode+'_admission.json')),
            'root_missing_representation': 'NaN in numerical arrays; unavailable in presentation',
            'rank_one_source': 'normalized P=0 radial probe; no background sea claim', 'runtime': runtime})
    write(args.output/'readout_complete.json', {'channel': kappa, 'runtime': runtime})

if __name__ == '__main__':
    main()
