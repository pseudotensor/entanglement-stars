"""Reduce only own saved evolved-cut outputs; no first-route comparison."""
import argparse
import json
import math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=require_remote_cpu(args.output)
    raw=json.loads((args.output/'results.json').read_text())
    groups={}
    for row in raw['rows']: groups.setdefault(row['clock_rs'],[]).append(row)
    result=[]
    for clock,channels in sorted(groups.items()):
        channels.sort(key=lambda r:r['kappa'])
        complete=[r['kappa'] for r in channels]==list(range(1,11))
        for shift in (-1,0,1):
            terms=[]
            for row in channels:
                c=next(c for c in row['cuts'] if c['site_shift']==shift)
                term={'kappa':row['kappa'],'weight':row['weight'],'source_dtype':row['source_dtype'],
                      'raw_global_valid':row['raw_global']['valid'],'dust_rank':row['dust_rank'],
                      'occupation_min':row['occupation_min'] if row['dust_rank'] else None,
                      'occupation_max':row['occupation_max'] if row['dust_rank'] else None}
                for label in ('raw','orthonormal_control'):
                    val=c[label]['physical_exterior_entropy']
                    term[label]=row['weight']*val if val is not None else None
                for label in ('held','flat'):
                    val=c[label]['entropy']
                    term[label]=row['weight']*val if val is not None else None
                for label in ('raw','orthonormal_control'):
                    term[label+'_excess']=term[label]-term['held'] if term[label] is not None and term['held'] is not None else None
                term['held_minus_flat']=term['held']-term['flat'] if term['held'] is not None and term['flat'] is not None else None
                terms.append(term)
            totals={}
            total_names=('raw','orthonormal_control','held','flat','raw_excess','orthonormal_control_excess','held_minus_flat')
            for name in total_names:
                available=complete and all(t[name] is not None for t in terms)
                totals[name]=math.fsum(t[name] for t in terms) if available else None
            prefixes=[]
            for last_kappa in range(1,11):
                subset=[t for t in terms if t['kappa']<=last_kappa]
                prefix_complete=[t['kappa'] for t in subset]==list(range(1,last_kappa+1))
                prefix_totals={name:math.fsum(t[name] for t in subset)
                    if prefix_complete and all(t[name] is not None for t in subset) else None
                    for name in total_names}
                prefixes.append({'through_kappa':last_kappa,'channels_complete':prefix_complete,
                    'scope':'Cumulative finite-channel prefix only; no omitted-channel bound.',
                    'totals':prefix_totals})
            area=channels[0]['area_over_a2']; eta_area=channels[0]['eta_area']
            signed=totals['held_minus_flat']
            result.append({'clock_rs':clock,'site_shift':shift,'channels_complete':complete,
                           'cut_label':channels[0]['cut_label'],'radius_over_a':channels[0]['radius_over_a'],
                           'area_over_a2':area,'eta_area':eta_area,
                           'shift_area_convention':'same primary interpolated-root or charge-fallback area for all three partitions',
                           'terms':terms,'cumulative_prefixes':prefixes,'totals':totals,
                           'per_area':{key:val/area if val is not None else None for key,val in totals.items()},
                           'over_eta_area':{key:val/eta_area if val is not None else None for key,val in totals.items()},
                           'r_held_10':abs(signed)/area if signed is not None else None,
                           'r_held_10_triangle':math.fsum(abs(t['held_minus_flat']) for t in terms)/area if signed is not None else None,
                           'r_cont_10':None,'r_ref_3D':None})
    scope=('Finite-channel declared-array and explicit orthonormal-column diagnostics. T12 uses a charge fallback, not a trapping root. '
           'Continued and three-dimensional references, their remainder, and physical-history raw-key identification remain unavailable. '
           'Any eta normalization uses the supplied massless comparator, not a massive coefficient.')
    output={'scope':scope,
            'runtime':runtime,'rows':result,'unavailable_slices':raw.get('unavailable_slices',[])}
    (args.output/'tables.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Own evolved-cut numerical tables','',scope,'',
           'Fixed primary area is retained for both shifted partitions. All ten channels are required for each full sum; cumulative prefixes in tables.json carry no omitted-channel bound.','',
           '| T/r0 | cut | shift | radius/a | raw total | control total | held | flat | raw excess | control excess | r held,10 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    def fmt(x): return 'unavailable' if x is None else f'{x:.10g}'
    for r in result:
        t=r['totals']
        lines.append('| '+' | '.join([fmt(r['clock_rs']),r['cut_label'],str(r['site_shift']),fmt(r['radius_over_a'])]+
                     [fmt(t[k]) for k in ('raw','orthonormal_control','held','flat','raw_excess','orthonormal_control_excess')]+[fmt(r['r_held_10'])])+' |')
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'saved':str(args.output/'tables.json'),'rows':len(result)}))


if __name__=='__main__': main()
