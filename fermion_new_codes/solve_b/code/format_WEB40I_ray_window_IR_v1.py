#!/usr/bin/env python3
"""Render saved infrared rows and aggregates; never run a solver."""
import argparse
import json
from pathlib import Path
from format_WEB40I_ray_window_v1 import number, table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--execution",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve saved report")
    d=json.loads(args.input.read_text());e=json.loads(args.execution.read_text())
    lines=["# X1 missing-infrared readback: "+d["phase"],"",
        "Formatter-only extraction of saved fields. Finite k<=8, q in [1e-8,1e-3]. "
        "Endpoint and mesh differences are empirical controls, not certified transmission errors or angular-tail bounds.","",
        f"Producer `{d['source_sha256']}`. Actual outer exit {e['actual_outer_returncode']}. "
        f"Whole outer wall {number(e['outer_wall_seconds'])} s; started {e['started_utc']}; finished {e['finished_utc']}.",
        f"Complete: {d['complete']}. Reason: {d['reason']}.",
        "Resources: `"+json.dumps(d["resources"],sort_keys=True)+"`.",
        "Prior charge: `"+json.dumps(d["charged_prior"],sort_keys=True)+"`.","",
        "## Raw quadrature rows","",
        "Large current-cancellation defects are retained literally. Neither a large defect alone nor a small one certifies Gamma accuracy.",""]
    rows=[]
    for rb in d['readbacks']:
        for r in rb['rows']:
            rows.append([rb['settings']['name'],r['k'],r['quadrature_index'],r['q'],r['weight_dq'],
                         r['gamma'],r['flux_defect'],r['relative_matching_residual'],r['endpoint_column_condition']])
    table(lines,['setting','k','node','q','dq weight','Gamma','Gamma+R-1','matching','column condition'],rows)
    lines+=['## Infrared cumulative content','','S, N and rs E are the saved duration-weighted panel sums only; high-frequency content is separate.','']
    rows=[]
    for rb in d['readbacks']:
        for a in rb.get('aggregates',[]):
            for c in a['cumulative']:
                rows.append([rb['settings']['name'],a['criterion'],a['convention'],c['k_cap'],c['complete'],*c['S_N_rsE']])
    table(lines,['setting','criterion','endpoint','k cap','complete','IR S','IR N','IR rs E'],rows)
    lines+=['## Half-inner endpoint diagnostics','','The refined endpoint/tolerance setting is unchanged except inner z=5e-5.','']
    table(lines,['k','q','Gamma','Gamma minus base','relative change','Gamma+R-1','matching','column condition'],
          [[r[k] for k in ('k','q','gamma','gamma_minus_base','gamma_relative_change','flux_defect','relative_matching_residual','endpoint_column_condition')]
           for r in d['inner_diagnostics']])
    lines+=['All raw currents, endpoint amplitudes, solver call counts and failed/active attempt state remain in the JSON. '
            'No automatic continuation, solved-history radiation or independent numerical route is claimed.','']
    args.output.write_text('\n'.join(lines))


if __name__=='__main__':
    main()
