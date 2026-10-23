"""Render saved Bprime comparison JSON into lean Markdown, with no numeric rerun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bprime_runtime import admit, identity, require_identity, verify_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--comparison-sha256', required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    runtime = admit(args.outdir, __file__, 'Saved-only Bprime lean Markdown renderer', 1)
    if any((args.outdir/name).exists() for name in ('table.md', 'render.json')):
        raise ValueError('Rendered outputs already exist; use a fresh directory')
    source = identity(__file__)
    binding = require_identity(args.comparison, args.comparison_sha256)
    data = json.loads(args.comparison.read_text())
    for key in ('input_summary_binding', 'production_summary_binding', 'input_artifact_binding', 'production_artifact_binding', 'artifact'):
        item = data[key]
        require_identity(item['path'], item['sha256'])
    lines = ['Fixed sixteen source spectral directions; independent fine-step losses and full128-coordinate residuals.', '',
             'T/L = '+format(data['T_over_L'], '.17g')+'. Overall fixed gates: '+('PASS' if data['all_passed'] else 'FAIL')+'.', '',
             '| Source rank (zero based) | Source p | Own loss | Own minus source | Initial norm squared | Full residual column 2-norm |',
             '|---|---|---|---|---|---|']
    for row in data['rows']:
        lines.append('| '+str(row['source_rank_zero_based'])+' | '+' | '.join(format(row[key], '.12g') for key in
                     ('source_p', 'own_delivered_weight', 'own_minus_source', 'initial_norm_squared', 'full_eigenpair_residual_column_norm2'))+' |')
    lines += ['', 'Fine block residual and convergence:', '', '| Saved check | Measured | Threshold | Status |', '|---|---|---|---|']
    for row in data['checks']:
        lines.append('| '+row['name']+' | '+format(row['measured'], '.12g')+' | '+format(row['threshold'], '.12g')+' | '+('PASS' if row['passed'] else 'FAIL')+' |')
    metrics = data['metrics']['fine']
    lines += ['', 'beta = '+format(metrics['beta'], '.12g')+', selected-block term = '+format(metrics['within_selected_block_norm2'], '.12g')+
              ', off-span term = '+format(metrics['off_span_norm2'], '.12g')+'.', '',
              'Full basis max |Q-I| = '+format(data['full_basis_Q_minus_I_maximum_entry'], '.12g')+
              '; benchmark max |Q_W-I| = '+format(data['benchmark_Q_minus_I_maximum_entry'], '.12g')+'.', '',
              'Source full-spectrum context (source only): '+json.dumps(data['source_full_spectrum_context'], sort_keys=True)+'.', '',
              '| Source T/L | Gram pairing max entry | Ordered spectrum max difference |', '|---|---|---|']
    for row in data['source_symmetry']:
        lines.append('| '+' | '.join(format(row[key], '.12g') for key in ('T_over_L', 'Gram_relation_maximum_entry', 'spectrum_maximum_difference'))+' |')
    lines += ['', data['scope'], '', 'Saved comparison: '+str(args.comparison)+'; SHA256 '+binding['sha256']+'.', '']
    output = args.outdir/'table.md'
    output.write_text('\n'.join(lines))
    require_identity(source['path'], source['sha256'])
    require_identity(binding['path'], binding['sha256'])
    verify_threads(runtime, 'after_rendering')
    record = {'runtime': runtime, 'source_binding': source, 'comparison_binding': binding, 'rendered_Markdown': identity(output)}
    (args.outdir/'render.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
