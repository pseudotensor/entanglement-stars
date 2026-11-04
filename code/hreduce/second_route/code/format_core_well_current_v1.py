#!/usr/bin/env python3
"""Format saved current JSON only; never load a state or repeat a contraction."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'results/second_route/core_well/current_v1'


def main():
    result = json.loads((FOLDER/'core_well_current_v1.json').read_text())
    if result['status'] != 'completed':
        raise RuntimeError('Retain incomplete evidence; no completed-result table')
    fields = ['row_kind', 'channel', 'cut', 'inside_count', 'matches_applied_mask',
        'j_lower', 'j_upper', 'per_T_over_rs_lower', 'per_T_over_rs_upper']
    rows = []
    for cut, metadata in result['cuts'].items():
        base = {'cut': cut, 'inside_count': metadata['inside_count'],
            'matches_applied_mask': json.dumps(metadata['matches_interpolated_applied_mask'])}
        for channel, data in result['channels'].items():
            value = data['cuts'][cut]['current_exact']
            rows.append(dict(base, row_kind='channel', channel=channel,
                j_lower=value['decimal_lower'], j_upper=value['decimal_upper']))
        value = result['totals'][cut]['j_exact']
        other = result['totals'][cut]['current_per_T_over_rs_exact']
        rows.append(dict(base, row_kind='total', channel='all',
            j_lower=value['decimal_lower'], j_upper=value['decimal_upper'],
            per_T_over_rs_lower=other['decimal_lower'],
            per_T_over_rs_upper=other['decimal_upper']))
    with (FOLDER/'current_table.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t',
            restval='not_applicable', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'source_status': result['status'], 'rows': len(rows),
        'output': str(FOLDER/'current_table.tsv')}))


if __name__ == '__main__':
    main()
