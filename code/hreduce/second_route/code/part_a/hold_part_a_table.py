#!/usr/bin/env python3
"""Load four accepted Part A JSONs only; no spectral-array arithmetic."""
import json
import os
from pathlib import Path

from hold_part_a_io import (BASE, CASES, METHOD, environment, identity, product,
                           same_content, unchanged)


# Fix all four pins together after both actual judgments, before this loader runs.
# This table module is not a bound dependency of the raw/assessment producers.
ACCEPTED_CHECKS = {'c15_k1': '954dfa65db289aed48f78dd7a44bc940d11e46333ce8c28ec50422c7a3d74a4a', 'c15_k2': '7f98dbacf37746eb341fe35bd0962069be7e343a33449439da4e5c1fb84cb34d', 'c30_k1': '60375e738801f3e134a6a216f8e195f000bdb7005fee7ec4523433af51916a72', 'c30_k2': '491b34b23b54dfb1fe3aff9860041ba0df2018f96dd69ff6bc50a0471327c7c5'}


def run():
    environment()
    if any(value is None for value in ACCEPTED_CHECKS.values()):
        raise RuntimeError('All four actual Part A judgments must first be pinned and reviewed')
    root = BASE/'holds'/'part_a'
    paths = [root/'part_a_table.json',root/'part_a_table.tsv']
    if any(p.exists() or p.with_name(p.name+'.partial').exists() for p in paths):
        raise FileExistsError('A table product or partial already exists')
    inputs = {'code':identity(__file__),'helper':identity(Path(__file__).with_name('hold_part_a_io.py'))}
    rows = []
    for case in CASES:
        saved = identity(root/case/'split_checks.json')
        if saved['sha256'] != ACCEPTED_CHECKS[case]:
            raise ValueError('An actual accepted Part A result differs')
        inputs[case] = saved
        result = json.loads(Path(saved['path']).read_text())
        if ((result['method'],result['case'],result['completed'],result['physical_rows_released'],result['cut_radius'])
                != (METHOD,case,True,False,1536.)):
            raise ValueError('The table requires exactly the four completed unreleased native cases')
        hold,channel = case.split('_')
        if (result['input']['hold_rs'],result['input']['kappa']) != (float(hold[1:]),int(channel[1:])):
            raise ValueError('The accepted result has another hold/channel')
        arrays = identity(result['arrays']['path'])
        if not same_content(arrays,result['arrays']):
            raise ValueError('The accompanying small atomic arrays differ')
        inputs[case+'_arrays'] = arrays
        rows.append({name:result[name] for name in (
            'case','input','cut_radius','weights_are_metadata_only','counts','energy_groups','fractions',
            'threshold_allocation','ambiguous_energy_signed','ambiguous_energy_absolute','clusters',
            'anchors','density','arithmetic_assumptions','statuses','arrays')})
    unchanged(inputs)
    fields = ('case','cut_a','localized_modes','extended_modes','ambiguous_modes',
              'E_total','E_localized','E_extended','fraction_localized','fraction_extended',
              'threshold_status','nontrivial_clusters','construction_status','physical_rows_released','first_route_split')
    lines = ['# Compact saved-basis display; full intervals and scopes: part_a_table.json. Kappa3 incomplete; no drain certificate.',
             '\t'.join(fields)]
    for row in rows:
        values = [row['case'],str(row['cut_radius']),str(row['counts']['nominal_localized']),
            str(row['counts']['nominal_extended']),str(row['counts']['threshold_ambiguous'])]
        values += [row['energy_groups'][kind]['real']['declared_6e'] for kind in ('total','localized','extended')]
        values += [row['fractions'][kind]['declared_6e'] if row['fractions'][kind] is not None else 'unresolved'
                   for kind in ('localized','extended')]
        values += [row['statuses']['threshold_classification'],str(row['clusters']['nontrivial_count']),
                   'unresolved source/transform and exact-h localization','false',row['statuses']['first_route_split']]
        values = [value.replace('\t',' ').replace('\n',' ') for value in values]
        lines.append('\t'.join(values))
    temporary = paths[1].with_name(paths[1].name+'.partial')
    with temporary.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    unchanged(inputs)
    os.link(temporary,paths[1])
    temporary.unlink()
    product(paths[0],dict(method=METHOD+'-four-case-table',completed=True,physical_rows_released=False,
        inputs=inputs,rows=rows,table=identity(paths[1]),
        scope='four saved-basis per-channel rows only; no weighted sum, physical gate or kappa3 completion'))
    unchanged(inputs)


if __name__ == '__main__':
    run()
