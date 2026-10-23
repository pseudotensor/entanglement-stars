"""Direct historical bilinear identities; no least-squares solve or new model."""
import argparse
import json
import os
from pathlib import Path
import sys
import mpmath as mp

NAMES=['P64_s4','P64_s8','S4_128_s8','S4_128_s16']
KEYS=['T_f','R_t_over_rs','R_t_over_2gE']


def coefficients(values,L):
    A,B,C,D=values
    return [21*A-14*B-24*C+18*D,(3*A-2*B-4*C+3*D)/L,
            (-7*A+7*B+6*C-6*D)/L,(-A+B+C-D)/(L*L)]


def scaled(a,b):return abs(a-b)/max(mp.mpf(1),abs(a),abs(b))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
    if os.environ.get('QI2_REMOTE_CPU')!='1' or sys.executable!='<internal>/python':
        raise RuntimeError('Use the prescribed remote qi2 CPU interpreter')
    if not str(out).startswith('<internal>/solve_b_bulk'):raise RuntimeError('bulk output path required')
    source=json.loads((out/'scalar_tables.json').read_text())
    cached=json.loads((out/'record_diagnostics.json').read_text())['historical']['old_width_fit']
    bindings=[]
    for name in NAMES:
        member=source[name];row=member['formation']['row'];s=member['scalars']
        bindings.append(dict(member=name,source=member['source'],source_index=row['source_index'],
                             T=row['T'],rs=s['rs'],g=s['g'],E=row['E'],R_t=row['R_t'],sigma=s['sigma']))
    result=dict(scope='Historical four-row diagnostic direct identities only; no general fit rerun, field propagation, model selection or physical-law inference.',
                members=NAMES,raw_bindings=bindings,precisions={},known_answers={})
    native={}
    for dps in [80,160]:
        mp.mp.dps=dps;L=mp.log(2);variables=[(mp.log(1/mp.mpf(b['rs'])),mp.log(b['sigma'])) for b in bindings]
        controls=[]
        for basis in range(4):
            ys=[[mp.mpf(1),z,h,z*h][basis] for z,h in variables];cs=coefficients(ys,L)
            expected=[mp.mpf(int(i==basis)) for i in range(4)]
            recovered=[sum(c*v for c,v in zip(cs,[1,z,h,z*h])) for z,h in variables]
            errors=[scaled(a,b) for a,b in zip(cs,expected)]+[scaled(a,b) for a,b in zip(recovered,ys)]
            maximum=max(errors);passed=maximum<mp.mpf('1e-60')
            if not passed:raise RuntimeError('basis known answer failed')
            controls.append(dict(basis=['1','z','h','zh'][basis],maximum_scaled=mp.nstr(maximum,dps),passed=passed))
        result['known_answers'][str(dps)]=controls;block={};native[str(dps)]={}
        for key in KEYS:
            ys=[]
            for binding in bindings:
                T,rs,g,E,R=(mp.mpf(binding[k]) for k in ['T','rs','g','E','R_t'])
                ys.append(T/rs if key=='T_f' else R/rs if key=='R_t_over_rs' else R/(2*g*E))
            cs=coefficients(ys,L);fitted=[sum(c*v for c,v in zip(cs,[1,z,h,z*h])) for z,h in variables]
            residuals=[a-b for a,b in zip(ys,fitted)]
            predictions={'P64_s2':2*ys[0]-ys[1],'S4_128_s4':2*ys[2]-ys[3]}
            old=cached['ordinates'][key];old_differences={}
            for field,new in [('coefficients',cs),('fitted',fitted),('residuals',residuals)]:
                old_differences[field]=[mp.nstr(scaled(a,mp.mpf(b)),dps) for a,b in zip(new,old[field])]
            old_max=max(mp.mpf(v) for values in old_differences.values() for v in values)
            if old_max>=mp.mpf('1e-60'):raise RuntimeError('cached old80 solve differs from direct identity')
            if key=='T_f':
                exact=[mp.mpf(63)/4,mp.mpf(13)/(16*L),mp.mpf(17)/(8*L),1/(4*L*L)]
                clock_error=max(scaled(a,b) for a,b in zip(cs,exact))
                if clock_error>=mp.mpf('1e-60'):raise RuntimeError('historical exact clock identity failed')
            else:clock_error=None
            native[str(dps)][key]=dict(coefficients=cs,predictions=predictions)
            block[key]=dict(ordinates=[mp.nstr(v,dps) for v in ys],coefficients=[mp.nstr(v,dps) for v in cs],
                            fitted=[mp.nstr(v,dps) for v in fitted],computed_residuals=[mp.nstr(v,dps) for v in residuals],
                            predictions={k:mp.nstr(v,dps) for k,v in predictions.items()},analytic_training_residuals=[0,0,0,0],
                            analytic_rms=0,residual_dof=0,dof_rms=None,
                            cached_old80_scaled_differences=old_differences,cached_old80_maximum_scaled=mp.nstr(old_max,dps),
                            exact_clock_identity_maximum_scaled=None if clock_error is None else mp.nstr(clock_error,dps))
        result['precisions'][str(dps)]=block
    mp.mp.dps=180;cross={}
    for key in KEYS:
        low,high=native['80'][key],native['160'][key]
        comparisons=[scaled(a,b) for a,b in zip(low['coefficients'],high['coefficients'])]
        comparisons.extend(scaled(low['predictions'][n],high['predictions'][n]) for n in low['predictions'])
        maximum=max(comparisons)
        if maximum>=mp.mpf('1e-60'):raise RuntimeError('historical direct80/160 convergence failed')
        cross[key]=dict(maximum_scaled=mp.nstr(maximum,80),passed=True)
    result['cross_precision']=cross
    result['precision_scope']='Observed 80/160 convergence and direct identity controls, not rigorous interval enclosures. Raw cached80 differences retained; no foreign fit bitwise equality asserted.'
    (out/'historical_width_direct_control.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(ordinates=3,precisions=2,basis_controls=8,clock_identity=True,old80_comparisons=True,cross_precision=True)))


if __name__=='__main__':main()
