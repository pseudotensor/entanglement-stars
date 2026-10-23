"""Saved-output reductions only; no propagation, diagonalization or source loading."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from packet_core_v1 import verified_controls, pure_distance_details, comparison_association, readout_admission, source_snapshot, packet_labels, validate_roster, validate_completed_states, validate_observable_schema
from run_PACKET_secondroute_v1 import digest


def clean(obj):
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, np.generic): return clean(obj.item())
    if isinstance(obj, float): return obj if np.isfinite(obj) else None
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(x) for x in obj]
    return obj

def extreme(values, records, maximum=True):
    values = np.asarray(values)
    valid = np.isfinite(values)
    if not valid.any(): return {'value': None, 'record_indices': [], 'unavailable_count': int(values.size)}
    value = float(np.max(values[valid]) if maximum else np.min(values[valid]))
    ids = np.flatnonzero(valid & (values == value))
    return {'value': value, 'record_indices': records[ids].tolist(), 'unavailable_count': int(np.count_nonzero(~valid))}

def reversals(values, records, intervals=False):
    x = np.asarray(values)
    sign = np.where(abs(x) <= 1e-10, 0, np.sign(x))
    valid = np.flatnonzero(np.isfinite(x) & (sign != 0))
    brackets, gaps = [], []
    for a,b in zip(valid[:-1],valid[1:]):
        if sign[a] != sign[b]:
            pair = [[int(records[a]),int(records[a])+1],[int(records[b]),int(records[b])+1]] if intervals else [int(records[a]),int(records[b])]
            (gaps if not np.isfinite(x[a:b+1]).all() else brackets).append(pair)
    return {'brackets':brackets,'gap_spanning_unresolved':gaps,
        'zero_unresolved_records':records[(abs(x)<=1e-10)&np.isfinite(x)].tolist(),
        'unavailable_records':records[~np.isfinite(x)].tolist(), 'bracket_role':'pair of centroid-increment endpoint intervals' if intervals else 'record endpoints'}


def located_differences(values, records):
    output=[]
    for c in range(values.shape[-1]):
        x=np.asarray(values[...,c]);valid=np.isfinite(x)
        if not valid.any():
            output.append({'maximum_absolute':None,'locators':[], 'finite_count':0,'unavailable_count':int(x.size)})
            continue
        maximum=float(np.max(abs(x[valid])))
        locs=np.argwhere(valid & (abs(x)==maximum))
        mapped=[[int(records[loc[0]]),*[int(y) for y in loc[1:]]] for loc in locs]
        output.append({'maximum_absolute':maximum,'locators':mapped,'finite_count':int(valid.sum()),
            'unavailable_count':int(np.count_nonzero(~valid)),'expected_count':int(x.size)})
    return output


def batch_distance(left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError('comparison state shape mismatch')
    return np.array([pure_distance_details(left[:, c], right[:, c])['stable_distance'] for c in range(left.shape[1])])


def trajectory_distances(left, right):
    rows=[[pure_distance_details(np.asarray(left[j,:,c]),np.asarray(right[j,:,c])) for c in range(left.shape[2])] for j in range(224)]
    return {key:np.array([[row[key] for row in record] for record in rows]) for key in rows[0][0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args()
    runtime = require_remote_cpu(args.output)
    controls_sha = verified_controls(args.output/'known_answers.json', args.spec)
    result = {'controls_sha256': controls_sha, 'runtime': runtime, 'channels': {}, 'comparisons': {}}
    for channel in (1, 5):
        folder = args.output/f'k{channel}'
        prep = json.loads((folder/'preparation.json').read_text())
        free = json.loads((folder/'free_projector.json').read_text())
        validate_roster(prep['labels'], packet_labels(channel,'E'))
        data = {}
        completions = {}
        result['channels'][str(channel)] = {'free_projector': free, 'preparation': prep, 'modes': {}}
        for mode in ('E', 'I16', 'I32'):
            meta = json.loads((folder/(mode+'_readout_complete.json')).read_text())
            path = folder/(mode+'_observables.npz')
            if digest(path) != meta['observables_sha256']:
                raise RuntimeError('observable binding mismatch')
            with np.load(path) as f:
                d = {k: f[k] for k in f.files}
            complete=json.loads((folder/(mode+'_complete.json')).read_text())
            completions[mode]=complete
            validate_observable_schema(d,complete['columns'])
            if meta['source_snapshot'] != source_snapshot(args.spec) or meta['controls_sha256'] != controls_sha:
                raise RuntimeError('readout controlled-source mismatch')
            for name in ('preparation','root_quality','admission','free_projector'):
                filename={'preparation':'preparation.json','root_quality':'roots_and_quality.json','admission':mode+'_admission.json','free_projector':'free_projector.json'}[name]
                if digest(folder/filename) != meta[name+'_sha256']:
                    raise RuntimeError('readout artifact binding mismatch')
            for name in ('free_eigenvalues','free_eigenvectors'):
                if digest(folder/(name+'.npy')) != meta[name+'_sha256']:
                    raise RuntimeError('free basis binding mismatch')
            state=np.load(folder/(mode+'_states.npy'),mmap_mode='r')
            validate_completed_states(complete,state.shape,packet_labels(channel,mode))
            if digest(folder/(mode+'_states.npy')) != complete['state_sha256'] or meta['saved_state_sha256'] != complete['state_sha256']:
                raise RuntimeError('saved state identity mismatch')
            del state
            comparison_association(d,d,complete['labels'],complete['labels'])
            admissions=json.loads((folder/(mode+'_admission.json')).read_text())
            if [row['record'] for row in admissions] != list(range(97,321)):
                raise RuntimeError('admission record join mismatch')
            data[mode] = d
            records = d['record_indices']
            if digest(folder/(mode+'_ledger.json')) != complete['ledger_sha256']:
                raise RuntimeError('ledger identity mismatch')
            ledger = json.loads((folder/(mode+'_ledger.json')).read_text())
            if [row['record'] for row in ledger] != list(range(97,320)):
                raise RuntimeError('ledger clock join mismatch')
            if any(len(row['before_norm']) != complete['columns'] for row in ledger):
                raise RuntimeError('ledger column join mismatch')
            if digest(folder/(mode+'_exponentials.json')) != complete['exponentials_sha256']:
                raise RuntimeError('exponential metadata identity mismatch')
            exponentials=json.loads((folder/(mode+'_exponentials.json')).read_text())
            expected_substeps=1 if mode=='E' else 16 if mode=='I16' else 32
            if [(row['record'],row['substep']) for row in exponentials] != [(j,s) for j in range(97,320) for s in range(expected_substeps)]:
                raise RuntimeError('exponential complete step/substep coverage mismatch')
            columns = []
            for column in range(d['norm'].shape[1]):
                endpoint = {key: value[-1, ..., column] for key, value in d.items() if value.ndim >= 2 and value.shape[0] == 224 and value.shape[-1] == d['norm'].shape[1]}
                maxima = {key: extreme(d[key][:, column], records) for key in ('energy_H', 'energy_0', 'homogeneous_mean_excess', 'negative_band_weight', 'high_share', 'negative_cosine_share', 'opposed_current_share', 'wall_64', 'packet_row_max_excess', 'packet_cauchy_max_excess', 'radial_mean_band_excess', 'radial_second_band_excess', 'H_mean_band_excess', 'H_second_band_excess')}
                norm = extreme(abs(d['norm'][:, column]-1), records)
                frozen_defects = np.array([r['frozen_energy_defect'][column] for r in ledger])
                work_defects = np.array([r['work_defect'][column] for r in ledger])
                energy_scale = np.maximum(1., 2*d['H_construction_bound'][:, column])
                energy_allowance = 1e-9*np.maximum(energy_scale[:-1], energy_scale[1:])
                imaginary_bad = np.zeros(224, dtype=bool)
                for key, value in d.items():
                    if key.endswith('_raw_imag'):
                        imaginary_bad |= ~np.isfinite(value[:,column]) | (abs(value[:, column])/d['norm'][:, column] > 1e-10*energy_scale)
                ledger_imaginary_bad=np.zeros(223,dtype=bool)
                for key in ('frozen_before_raw_imag','frozen_after_raw_imag','next_raw_imag','work_raw_imag'):
                    residues=np.array([row[key][column] for row in ledger])
                    norms=np.array([row['before_norm' if key=='frozen_before_raw_imag' else 'after_norm'][column] for row in ledger])
                    ledger_imaginary_bad |= ~np.isfinite(residues) | (abs(residues)/norms > 1e-10*np.maximum(energy_scale[:-1],energy_scale[1:]))
                substep_bad=[{'record':row['record'],'substep':row['substep']} for row in exponentials if not np.isfinite(row['after_norm'][column]) or abs(row['after_norm'][column]-1)>1e-9]
                failures = {'substep_norm':substep_bad, 'ledger_imaginary':records[:-1][ledger_imaginary_bad].tolist(),
                    'readout_admission': [row['record'] for row in admissions if not row['admitted_columns'][column]],
                    'imaginary_expectations': records[imaginary_bad].tolist(),
                    'packet_row_bound': records[d['packet_row_max_adjusted_excess'][:, column] > 0].tolist(),
                    'packet_cauchy_bound': records[d['packet_cauchy_max_adjusted_excess'][:, column] > 0].tolist(),
                    'norm': records[abs(d['norm'][:, column]-1) > 1e-9].tolist(),
                    'radial_mean': records[d['radial_mean_band_excess'][:, column] > free['edge_allowance']+1e-10*max(1., free['edge'])].tolist(),
                    'radial_second': records[d['radial_second_band_excess'][:, column] > (free['edge']+free['edge_allowance'])**2-free['edge']**2+1e-10*max(1., free['edge']**2)].tolist(),
                    'H_mean': records[d['H_mean_band_excess'][:, column] > 1e-10*np.maximum(1., d['H_construction_bound'][:, column])].tolist(),
                    'H_finer_mean': records[d['H_finer_mean_excess'][:, column] > 1e-10*np.maximum(1., d['H_finer_bound'][:, column])].tolist(),
                    'H_finer_second': records[d['H_finer_second_excess'][:, column] > 1e-10*np.maximum(1.,d['H_finer_bound'][:,column]**2)].tolist(),
                    'H_second': records[d['H_second_band_excess'][:, column] > 1e-10*np.maximum(1., d['H_construction_bound'][:, column]**2)].tolist(),
                    'variance_H': records[~np.isfinite(d['variance_H'][:, column])].tolist(),
                    'variance_0': records[~np.isfinite(d['variance_0'][:, column])].tolist(),
                    'position_variance': records[~np.isfinite(d['position_variance'][:,column])].tolist(),
                    'E_frozen': records[:-1][abs(frozen_defects) > energy_allowance].tolist() if mode == 'E' else None,
                    'E_work': records[:-1][abs(work_defects) > energy_allowance].tolist() if mode == 'E' else None}
                columns.append({'label': prep['labels'][column], 'endpoint': endpoint, 'maxima': maxima,
                    'norm_drift': norm, 'maximum_substep_norm_drift':max(abs(row['after_norm'][column]-1) for row in exponentials), 'failures': failures,
                    'frozen_energy_max_defect': float(np.max(abs(frozen_defects))) if mode == 'E' else None,
                    'work_max_defect': float(np.max(abs(work_defects))) if mode == 'E' else None,
                    'velocity_lapse_reversals': reversals(d['velocity_lapse'][:, column], records),
                    'velocity_total_reversals': reversals(d['velocity_total'][:, column], records),
                    'velocity_flow_reversals': reversals(d['velocity_flow'][:, column], records),
                    'current_Q_reversals': reversals(d['current_Q'][:, column], records),
                    'centroid_increment_reversals': reversals(np.diff(d['centroid'][:, column])/8, records[:-1], intervals=True)})
            result['channels'][str(channel)]['modes'][mode] = {'columns': columns, 'readout': meta,
                'observables_sha256': digest(path), 'norm_admitted': all(not c['failures']['norm'] and not c['failures']['substep_norm'] for c in columns),
                'trajectory_admitted': [all(v is None or len(v)==0 for v in c['failures'].values()) for c in columns],
                'missing_geometry_is_separate': True}
        for leftmode, rightmode, tag in [('I16', 'I32', 'cadence'), ('E', 'I16', 'rule')]:
            leftstate = np.load(folder/(leftmode+'_states.npy'), mmap_mode='r')
            rightstate = np.load(folder/(rightmode+'_states.npy'), mmap_mode='r')
            count = comparison_association(data[leftmode],data[rightmode],completions[leftmode]['labels'],completions[rightmode]['labels'],primary_only=tag=='cadence')
            distance_details=trajectory_distances(leftstate[:,:,:count],rightstate[:,:,:count])
            distances=distance_details['stable_distance']
            differences = {key: data[rightmode][key][..., :count].astype(float)-data[leftmode][key][..., :count].astype(float)
                for key in data[leftmode] if data[leftmode][key].ndim >= 2 and data[leftmode][key].shape[0] == 224 and data[leftmode][key].shape[-1] == leftstate.shape[2] and data[rightmode][key].shape[-1] == rightstate.shape[2]}
            np.savez(folder/(tag+'_comparison.npz'), pure_state_distance=distances, **{'distance_'+k:v for k,v in distance_details.items()}, **differences)
            result['comparisons'][f'k{channel}_{tag}'] = {'per_packet_max_distance': [extreme(distances[:, c], np.arange(97,321)) for c in range(count)],
                'endpoint_distance': distances[-1], 'per_packet_differences': {key: located_differences(value,np.arange(97,321)) for key,value in differences.items()},
                'sensitivity_only': True}
        if channel == 1:
            for mode in ('E', 'I16'):
                state = np.load(folder/(mode+'_states.npy'), mmap_mode='r')
                distance_details=trajectory_distances(state[:,:,:16],state[:,:,16:])
                distance=distance_details['stable_distance']
                differences = {key: value[..., 16:].astype(float)-value[..., :16].astype(float) for key, value in data[mode].items()
                    if value.ndim >= 2 and value.shape[0] == 224 and value.shape[-1] == 32}
                np.savez(folder/(mode+'_width_comparison.npz'), pure_state_distance=distance, **{'distance_'+k:v for k,v in distance_details.items()}, **differences)
                result['comparisons'][mode+'_width'] = {'endpoint_distance': distance[-1], 'sensitivity_only': True,
                    'per_packet_differences': {key: located_differences(value,np.arange(97,321)) for key,value in differences.items()}}
    (args.output/'summary.json').write_text(json.dumps(clean(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps({'summary_saved': True}))

if __name__ == '__main__':
    main()
