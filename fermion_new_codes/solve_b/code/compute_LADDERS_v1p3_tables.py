"""Independent sections 1--4 scalar tables from raw inputs and own masks."""
import argparse
import json
import math
from pathlib import Path


PRIMITIVES = ['T', 'T_over_rs', 'E', 'E_over_E0', 'R_t', 'R_t_over_2gE',
              'R_c', 'min_N_minus_absv', 'trapping_zeros', 'tangencies',
              'surface_jets', 'kappa_T', 'r_in']


def ratios(E, E0, root, rs, g, sigma, centroid):
    return dict(E_over_E0=E/E0, R_t_over_rs=None if root is None else root/rs,
                R_t_over_2gE=None if root is None else root/(2*g*E),
                gate=None if root is None else sigma/root,
                R_c_over_rs=None if centroid is None else centroid/rs)


def argmin_rows(rows, key):
    eligible = [r for r in rows if r.get(key) is not None]
    if not eligible:
        return None
    value = min(r[key] for r in eligible)
    tied = [r for r in eligible if r[key] == value]
    return dict(value=value, T_over_rs=tied[0]['T_over_rs'],
                source_index=tied[0]['source_index'], ties=[r['source_index'] for r in tied])


def episodes(rows):
    answer, active = [], []
    def close(following):
        answer.append(dict(start=active[0]['T_over_rs'], last=active[-1]['T_over_rs'],
                           count=len(active), source_indices=[r['source_index'] for r in active],
                           first_R_t=active[0]['R_t'], last_R_t=active[-1]['R_t'],
                           minimum=argmin_rows(active, 'F'),
                           certified_minimum=argmin_rows([r for r in active if r['certified']], 'F'),
                           noncounting=sum(r['counts'] is not True for r in active),
                           uncertified=sum(r['certified'] is not True for r in active),
                           closes_at=None if following is None else following['T_over_rs'],
                           closing_source_index=None if following is None else following['source_index']))
    for row in rows:
        if row['zeros']:
            active.append(row)
        elif active:
            close(row)
            active = []
    if active:
        close(None)
    return answer


def match_clock(rows, physical_T):
    allowance = 16 * math.ulp(max(1., abs(physical_T)))
    matches = [r for r in rows if abs(r['T']-physical_T) <= allowance]
    if len(matches) > 1:
        raise ValueError('ambiguous clock match')
    return dict(target_T=physical_T, tolerance_T=allowance,
                row=matches[0] if matches else None)


def controls():
    actual = ratios(10, 12, 3, 8, .25, 1, 2)
    expected = dict(E_over_E0=5/6, R_t_over_rs=3/8, R_t_over_2gE=3/5,
                    gate=1/3, R_c_over_rs=1/4)
    tests = [dict(name='distinct ratio denominators', actual=actual, expected=expected,
                  passed=actual == expected)]
    rows = [dict(source_index=i, T=i, T_over_rs=i, zeros=z, R_t=max(z) if z else None,
                 F=-i, certified=i < 2, counts=i != 2)
            for i, z in enumerate([[], [1,3], [2,4], [], [2,5], [3,6]])]
    found = episodes(rows)
    tests.append(dict(name='episode closure and certified minimum', actual=found,
                      passed=(len(found)==2 and found[0]['start']==1 and found[0]['last']==2
                              and found[0]['count']==2 and found[0]['closes_at']==3
                              and found[0]['minimum']['value']==-2
                              and found[0]['certified_minimum']['value']==-1
                              and found[1]['closes_at'] is None)))
    tests.append(dict(name='exact match and unavailable clock',
                      passed=match_clock(rows,2)['row']['source_index']==2
                      and match_clock(rows,2.5)['row'] is None))
    linear = 0-1*(1-0)/(-1-1)
    tests.append(dict(name='linear zero', actual=linear, expected=.5, passed=linear==.5))
    if not all(t['passed'] for t in tests):
        raise RuntimeError('scalar-table controls failed')
    return tests


def member_readout(path, masks):
    raw = json.loads(Path(path).read_text())
    args, head = raw['args'], raw['head']
    rs, mass, g, sigma = (args[k] for k in ['rs','m','g','ell'])
    E0, NB, R0 = head['E0'], head['N_B'], head['R0']
    if masks['clock_contract']['errors']:
        raise ValueError('unadmitted clock contract')
    rows = []
    for mask in masks['rows']:
        index = mask['source_index']
        rec = raw['records'][index]
        zeros = rec.get('trapping_zeros')
        if not isinstance(zeros,list) or not all(isinstance(z,(int,float)) and math.isfinite(z) for z in zeros):
            raise ValueError('malformed zeros')
        root = rec.get('R_t')
        row = dict(source_index=index,T=rec['T'],T_over_rs=rec['T']/rs,
                   primitives={k:rec.get(k) for k in PRIMITIVES},
                   F=rec.get('min_N_minus_absv'), R_t=root, R_c=rec.get('R_c'),
                   E=rec['E'], zeros=zeros, tangencies=rec.get('tangencies'),
                   **{k:mask[k] for k in ['kept','confirming','counts','certified','kept_prefix','banded']})
        row.update(ratios(rec['E'], E0, root, rs, g, sigma, rec.get('R_c')))
        jets = rec.get('surface_jets')
        derivative = jets.get('d_N_plus_v') if isinstance(jets,dict) else None
        row['outer_slope'] = derivative[0] if derivative else None
        row['internal_checks'] = dict(
            outer_root_difference=None if not zeros or root is None else root-max(zeros),
            stored_charge_ratio_difference=None if rec.get('R_t_over_2gE') is None or root is None
                else rec['R_t_over_2gE']-row['R_t_over_2gE'],
            stored_energy_ratio_difference=None if rec.get('E_over_E0') is None
                else rec['E_over_E0']-row['E_over_E0'],
            slope_kappa_difference=None if row['outer_slope'] is None or rec.get('kappa_T') is None
                else row['outer_slope']-rec['kappa_T'])
        rows.append(row)
    formed = next((i for i,r in enumerate(rows) if r['zeros']),None)
    formation = None
    if formed is not None:
        row = rows[formed]
        prev = rows[formed-1] if formed else None
        bracket = None
        if prev is not None:
            lo, hi, Flo, Fhi = prev['T_over_rs'],row['T_over_rs'],prev['F'],row['F']
            linear = lo-Flo*(hi-lo)/(Fhi-Flo) if Fhi != Flo and Flo*Fhi <= 0 else None
            bracket = dict(previous_index=prev['source_index'], previous_clock=lo,
                           previous_F=Flo, width=hi-lo, linear_zero=linear,
                           quadratic_status='definition_unresolved: source stencil unspecified')
        formation = dict(row=row,bracket=bracket,
                         offsets={str(offset):match_clock(rows,row['T']+rs*offset)
                                  for offset in [0,.5,1,2,4]})
    pF = (9*math.pi*NB/32)**(1/3)/(mass*R0)
    scalars = dict(m=mass,rs=rs,g=g,sigma=sigma,tau=args['tau'],tau_over_rs=args['tau']/rs,
                   sigma_over_rs=sigma/rs,sites=args['nsites'],NB=NB,
                   nominal_count_ratio=NB*mass**3/1.087,E0=E0,charge0=2*g*E0,
                   rest_energy_ratio=E0/(NB*mass),charge_ratio=E0*g**1.5*mass**2/.135858,
                   f_kin=head['f_kin'],pF_over_m_isotropic=pF,isotropic_x=pF*pF,
                   isotropic_coldness=pF*pF/5,own_x=head['f_kin']/.3,
                   own_pF_over_m=math.sqrt(head['f_kin']/.3),
                   onset_isotropic=1.222*NB**(1/3),
                   onset_isotropic_over_rs=1.222*NB**(1/3)/rs,
                   exclusion_floor_literal=.49*NB**(1/3),
                   NB_preparation_identity_difference=NB-rs/(2*g*mass),
                   R0=R0,R0_argument_difference=R0-args['R0_over_rs']*rs)
    minimum = argmin_rows([r for r in rows if r['certified']], 'R_c')
    if minimum:
        minimum['over_rs'] = minimum['value']/rs
        minimum['at_last_certified'] = minimum['T_over_rs']==masks['summary']['last_certified']
    return dict(source=path,scalars=scalars,mask_summary=masks['summary'],rows=rows,
                formation=formation,episodes=episodes(rows),centroid_minimum=minimum,
                matched={str(t):match_clock(rows,rs*t) for t in range(0,41,2)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    masks=json.loads((out/'residual_series.json').read_text())
    tests=controls()
    (out/'scalar_known_answers.json').write_text(json.dumps(tests,indent=2)+'\n')
    result={}
    (out/'scalar_records').mkdir(exist_ok=True)
    for name,path in json.loads(Path(args.inputs).read_text()).items():
        member=member_readout(path,masks[name])
        records=member.pop('rows')
        record_path=out/'scalar_records'/(name+'.json')
        record_path.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        member['record_table']=str(record_path)
        result[name]=member
        (out/'scalar_tables.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(member=name,formed=result[name]['formation'] is not None)),flush=True)


if __name__=='__main__':
    main()
