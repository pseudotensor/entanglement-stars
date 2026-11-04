"""Read field/record metadata for the assigned curvature calculation, without curvature arithmetic."""
import argparse
import json
from pathlib import Path
import numpy as np

SOURCES = {
    'prod_r11': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11',
    'B2': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096',
    'P64_s4': '<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40',
    'P64_s2': '<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d',
    'P64_s8': '<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40',
    'P64_s4_tau4': '<internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg',
    'P64_s4_tau2': '<internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'scope': 'Input metadata only; no curvature, source-bound or cross-route result computed.', 'sources': {}}
    for name, base in SOURCES.items():
        record_path, field_path = Path(base+'.json'), Path(base+'_fields.npz')
        row = {'record_path': str(record_path), 'field_path': str(field_path),
               'records_exist': record_path.exists(), 'fields_exist': field_path.exists()}
        if record_path.exists():
            raw = json.loads(record_path.read_text())
            row['args'] = raw.get('args')
            row['head'] = raw.get('head')
            row['script_metadata'] = raw.get('script')
            row['records'] = [{k: r[k] for k in ('T', 'T_over_rs', 'converged', 'unconverged',
                             'indicator', 'record_label', 'regime', 'label', 'solver',
                             'halvings', 'nsub_max') if k in r} for r in raw['records']]
        if field_path.exists():
            with np.load(field_path, allow_pickle=True) as fields:
                row['arrays'] = {k: {'shape': list(fields[k].shape), 'dtype': str(fields[k].dtype)}
                                 for k in fields.files}
                row['scalars'] = {k: fields[k].item() for k in ('g', 'ell', 'tau', 'solver', 'label') if k in fields}
                row['radial_grid'] = fields['R'].tolist()
                row['times'] = fields['T'].tolist()
                row['node_grid_first'] = np.asarray(fields['r_nodes'][0], dtype=float).tolist() if 'r_nodes' in fields else None
                row['node_grid_last'] = np.asarray(fields['r_nodes'][-1], dtype=float).tolist() if 'r_nodes' in fields else None
        result['sources'][name] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {'records_exist': v['records_exist'], 'fields_exist': v['fields_exist'],
                         'field_arrays': v.get('arrays'), 'args': v.get('args'),
                         'first_record': v.get('records', [None])[0]}
                      for k, v in result['sources'].items()}, indent=2))


if __name__ == '__main__':
    main()
