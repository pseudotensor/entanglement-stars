"""All-record conditional radial source comparison; no packet propagation."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from packet_core_v1 import row_constant, validate_history, verified_controls, source_snapshot
from run_PACKET_secondroute_v1 import BASE, PINNED, digest, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args()
    runtime = require_remote_cpu(args.output)
    controls_sha = verified_controls(args.output/'known_answers.json', args.spec)
    if any(digest(BASE+s) != h for s, h in PINNED.items()):
        raise RuntimeError('source binding changed')
    raw = json.loads(Path(BASE+'.json').read_text())
    if raw['args']['kappa_sea'] != 10:
        raise RuntimeError('source channel roster not ten')
    with np.load(BASE+'_fields.npz', allow_pickle=False) as archive:
        f = {key: archive[key] for key in ('T', 'R', 'N_applied', 'v_applied', 'eps_tot', 'J_tot', 'eps_pairs', 'J_pairs', 'g', 'ell', 'tau')}
    validate_history(raw['records'], f)
    radius = f['R']
    flow = f['v_applied']
    rho = f['eps_tot']/(4*np.pi*radius**2)
    current = f['J_tot']/(4*np.pi*radius**2)
    signed = rho+flow*current
    chain = np.zeros_like(signed)
    for k in range(1, 11):
        for j in range(321):
            bound, _, _ = row_constant(radius, k, flow[j])
            chain[j] += 16*k*bound/(4*np.pi*radius**2)
    cubic = 4*np.sqrt(.2**2+1.5+flow**2/8)
    arrays = {'R': radius, 'T': f['T'], 'W': flow, 'rho': rho, 'J': current, 'source_signed': signed,
        'source_absolute': abs(signed), 'radial_bound': chain, 'radial_ratio': abs(signed)/chain,
        'radial_excess': abs(signed)-chain, 'cubic_comparator': cubic,
        'cubic_comparison_ratio': abs(signed)/cubic, 'cubic_comparison_excess': abs(signed)-cubic}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('nonfinite derived stored-source profiles')
    np.savez(args.output/'stored_source_profiles.npz', **arrays)
    rows = []
    for j in range(321):
        row = {'record': j, 'T': float(f['T'][j]), 'T_over_rs': float(f['T'][j]/64), 'packet_window': j >= 97}
        for key in ('radial_ratio', 'radial_excess', 'cubic_comparison_ratio', 'cubic_comparison_excess'):
            maximum = float(np.max(arrays[key][j]))
            ties = np.flatnonzero(arrays[key][j] == maximum)
            row[key] = {'maximum': maximum, 'indices': ties.tolist(), 'radii': radius[ties].tolist()}
        row['radial_numerical_exceedance_count'] = int(np.count_nonzero(arrays['radial_excess'][j] > 1e-10*np.maximum(1., chain[j])))
        rows.append(row)
    write(args.output/'stored_source_summary.json', {'controls_sha256': controls_sha, 'source_snapshot':source_snapshot(args.spec), 'runtime': runtime, 'records': rows,
        'source_bindings': {BASE+s: h for s, h in PINNED.items()},
        'radial_status': 'conditional on source allocation/reference and all-record contraction premises',
        'native_covariance_verification': 'unavailable', 'kernel_averaged_verification': 'unavailable',
        'cubic_status': 'cross-representation arithmetic comparator; no native theorem validation or violation',
        'weights': [16*k for k in range(1, 11)], 'channel_range': list(range(1, 11)),
        'normalization': 'shell per-bin energy divided by 4*pi*R^2*a, a=1'})
    print(json.dumps({'records': 321, 'source_readout_complete': True}))

if __name__ == '__main__':
    main()
