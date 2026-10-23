"""Lossless printed-cell inventory and independent scalar-cell comparisons."""
import argparse
from decimal import Decimal, getcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import mpmath as mp

getcontext().prec=200
NUMBER = re.compile(r'(?<![A-Za-z_0-9.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_0-9]|\.\d)')


def numeric(target, own):
    if own is None:
        return dict(target=target, own=None, status='input_unavailable')
    value=Decimal(target)
    unit=Decimal(1).scaleb(value.as_tuple().exponent)
    # Ordinary scalar operation allowance; high-precision controls separately
    # adjudicate transcendental and near-boundary entries before final delivery.
    if isinstance(own,dict):
        allowance=Decimal(own['error']); actual=Decimal(own['value']); display=own['value']
    else:
        allowance=Decimal.from_float(32*math.ulp(float(own))) if not isinstance(own,int) else Decimal(0)
        actual=Decimal(own) if isinstance(own,int) else Decimal.from_float(own); display=own
    lo,hi=value-unit/2,value+unit/2
    if isinstance(own,dict) and 'rational' in own:
        rational=Fraction(int(own['rational'][0]),int(own['rational'][1]))
        lower,upper=Fraction(lo),Fraction(hi)
        tie=rational==lower or rational==upper
        agrees=(lower<rational<upper) or (tie and int(value/unit)%2==0)
        return dict(target=target,own=display,exact_rational=own['rational'],numerical_allowance='0',
                    rounding_interval=[str(lo),str(hi)],rounding_tie=tie,tie_policy='nearest, ties to even',
                    boundary_overlap=False,status='agrees' if agrees else 'arithmetic_mismatch')
    exact_integer=isinstance(own,int)
    contained=actual-allowance>lo and actual+allowance<hi
    disjoint=actual+allowance<lo or actual-allowance>hi
    status=('agrees' if actual==value else 'arithmetic_mismatch') if exact_integer else (
        'agrees' if contained else 'arithmetic_mismatch' if disjoint else 'definition_unresolved')
    return dict(target=target,own=display,rounding_interval=[str(lo),str(hi)],
                numerical_allowance=str(allowance),boundary_overlap=not exact_integer and not contained and not disjoint,
                status=status)


def exact(value):
    fraction=value if isinstance(value,Fraction) else Fraction(value)
    return dict(value=str(Decimal(fraction.numerator)/Decimal(fraction.denominator)),error='0',
                rational=[str(fraction.numerator),str(fraction.denominator)])


def row_value(member,row,key):
    s=member['scalars'];F=Fraction;root=row['R_t'];rs=F(s['rs'])
    if key=='T_over_rs':return exact(F(row['T'])/rs)
    if key=='E_over_E0':return exact(F(row['E'])/F(s['E0']))
    if key=='R_t_over_rs':return None if root is None else exact(F(root)/rs)
    if key=='R_t_over_2gE':return None if root is None else exact(F(root)/(2*F(s['g'])*F(row['E'])))
    if key=='gate':return None if root is None else exact(F(s['sigma'])/F(root))
    if key=='R_c_over_rs':return None if row['R_c'] is None else exact(F(row['R_c'])/rs)
    return None if row.get(key) is None else exact(row[key])


def scalar_precision(member):
    s=member['scalars'];F=Fraction;m,g,rs,NB,E0=(F(s[k]) for k in ['m','g','rs','NB','E0'])
    values={k:exact(s[k]) for k in ['m','rs','sigma','tau','NB','E0','f_kin']}
    values.update(tau_over_rs=exact(F(s['tau'])/rs),sigma_over_rs=exact(F(s['sigma'])/rs),
                  nominal_count_ratio=exact(NB*m**3/F('1.087')),charge0=exact(2*g*E0),
                  rest_energy_ratio=exact(E0/(NB*m)),own_x=exact(F(s['f_kin'])/F('0.3')))
    if g==F(1,4):values['charge_ratio']=exact(E0*m*m/F('0.135858')/8)
    precision={}
    for dps in [80,160]:
        mp.mp.dps=dps
        nb=mp.mpf(s['NB']);r=mp.mpf(s['rs']);mass=mp.mpf(s['m']);R0=mp.mpf(s['R0'])
        pf=(9*mp.pi*nb/32)**(mp.mpf(1)/3)/(mass*R0)
        calculated=dict(pF_over_m_isotropic=pf,isotropic_coldness=pf*pf/5,
                        onset_isotropic=mp.mpf('1.222')*nb**(mp.mpf(1)/3),
                        onset_isotropic_over_rs=mp.mpf('1.222')*nb**(mp.mpf(1)/3)/r,
                        exclusion_floor_literal=mp.mpf('0.49')*nb**(mp.mpf(1)/3),
                        own_pF_over_m=mp.sqrt(mp.mpf(s['f_kin'])/mp.mpf('0.3')))
        precision[str(dps)]={k:mp.nstr(v,dps) for k,v in calculated.items()}
    for key,value in precision['160'].items():values[key]=precise(precision['80'][key],value)
    values['sites']=s['sites']
    return values,precision


def clock_value(member,value):
    if value is None:return None
    rows=[r for r in member['_records'].values() if r['T_over_rs']==value]
    if len(rows)!=1:raise ValueError('summary clock lacks unique own raw binding')
    return row_value(member,rows[0],'T_over_rs')


def number_list(cell, values):
    tokens=NUMBER.findall(cell)
    if len(tokens)!=len(values):
        return [dict(status='pending', reason='compound token count differs',
                     tokens=tokens, own=values)]
    return [numeric(token,value) for token,value in zip(tokens,values)]


def labels(cell,row):
    actual=dict(noncounting='(n.c.)' in cell, uncertified='(unc.)' in cell, banded='†' in cell)
    expected=dict(noncounting=not row['counts'],
                  uncertified=bool(row['counts'] and not row['certified']),banded=row['banded'])
    return dict(target=actual,own=expected,
                status='agrees' if actual==expected else 'label_or_membership_mismatch')


def missing(cell):
    return dict(target=cell,own=None,status='agrees' if cell.strip()=='-' or cell.startswith('- (') else 'label_or_membership_mismatch')


def scalar_cell(section,column,cell,member):
    s=member['scalars']; summary=member['mask_summary']; formation=member['formation']
    v=lambda key:member['precision_scalars'].get(key,s[key])
    if section=='1':
        direct={2:'m',3:'rs',4:'sigma',6:'sites',7:'NB',8:'nominal_count_ratio',9:'E0',
                10:'charge0',11:'rest_energy_ratio',12:'charge_ratio',13:'f_kin',
                14:'pF_over_m_isotropic',15:'isotropic_coldness',18:'exclusion_floor_literal'}
        if column in direct:
            return number_list(cell,[v(direct[column])])
        if column==5:
            return number_list(cell,[v('tau'),v('tau_over_rs')])
        if column==17:
            return number_list(cell,[v('onset_isotropic'),v('onset_isotropic_over_rs')])
        if column==19:
            return number_list(cell,[summary['records'],clock_value(member,summary['first']),clock_value(member,summary['last']),len(summary['dropped_without_E'])])
        if column in [20,21]:
            val=summary['last_certified' if column==20 else 'last_kept_prefix']
            if val is None:
                return [dict(target=cell,own=None,status='agrees' if cell=='none' or cell.startswith('none (') else 'label_or_membership_mismatch')]
            return number_list(cell,[clock_value(member,val)])
        if column==22:
            failure=summary['first_failure']
            if failure is None:
                return [dict(target=cell,own=None,status='agrees' if cell=='none' else 'label_or_membership_mismatch')]
            result=number_list(cell.split('(')[0],[clock_value(member,failure['T_over_rs'])])
            reported=[reason for reason in ['kept','confirming'] if reason+' residual above' in cell]
            result.append(dict(target=reported,own=failure['reasons'],status='agrees' if reported==failure['reasons'] else 'label_or_membership_mismatch'))
            return result
        if column==23:
            args=member['_args'];values=[clock_value(member,summary['last'])]
            if cell.startswith('stopped at'):values.append(exact(args['T_over_rs']))
            if 'certified to' in cell and summary['last_certified'] is not None:values.append(clock_value(member,summary['last_certified']))
            result=mixed(cell,values)
            if 'no trapping zero' in cell:
                result.append(dict(target='no trapping zero',own=formation is None,
                    status='agrees' if formation is None else 'label_or_membership_mismatch'))
            return result
        if column==1:
            checks=[]
            patterns=[(r'\(([\d.]+),\s*([\d.]+)\)',[v('m'),v('rs')]),
                      (r'sigma/a\s+([\d.]+)',[v('sigma')]),(r'tau\s+([\d.]+)',[v('tau')]),
                      (r'release at\s+([\d.]+)\s+r_s',[exact(Fraction(s['R0'])/Fraction(s['rs']))]),
                      (r'width point \(([\d.]+),',[v('rs')]),(r'point at\s+([\d.]+)',[v('rs')]),
                      (r'to\s+([\d.]+)\)',[clock_value(member,summary['last'])])]
            for pattern,values in patterns:
                for match in re.finditer(pattern,cell):
                    checks.extend(numeric(token,value) for token,value in zip(match.groups(),values))
            checks.append(dict(target=cell,status='source_dependent_primitive',
                reason='Declared role and provenance: manual occurrence map section 1; explicit parameter occurrences checked separately.'))
            return checks
    if section=='3':
        minimum=member['centroid_minimum']
        if column in [1,2,3]:
            if minimum is None:
                return [missing(cell)]
            r=member['_records'][minimum['source_index']]
            return number_list(cell,[row_value(member,r,{1:'R_c',2:'T_over_rs',3:'R_c_over_rs'}[column])])
        if column in [5,6]:
            return number_list(cell,[v('onset_isotropic' if column==5 else 'exclusion_floor_literal')])
        if column==7 and minimum is None:
            return [dict(target=cell,own=None,status='agrees' if cell=='no' else 'label_or_membership_mismatch',reason='empty certified prefix has no attained minimum')]
        if column==7 and minimum is not None:
            expected=minimum['at_last_certified']; reported=cell.startswith('yes')
            return [dict(target=cell,own=expected,status='agrees' if expected==reported else 'label_or_membership_mismatch')]
    if section=='2':
        if formation is None:
            if column==1:
                result=number_list(cell,[clock_value(member,summary['last'])])
                result.append(dict(target=cell,own='none to last',status='agrees' if cell.startswith('none to ') else 'label_or_membership_mismatch'))
                return result
            if column>=2:
                return [missing(cell)]
        else:
            row=formation['row']; bracket=formation['bracket']
            direct={1:'T_over_rs',4:'R_t',5:'R_t_over_rs',6:'R_t_over_2gE',7:'E_over_E0',10:'gate'}
            if column in direct:
                return number_list(cell,[row_value(member,row,direct[column])])
            if column==2 and bracket:
                previous=member['_records'][bracket['previous_index']]
                return number_list(cell,[row_value(member,previous,'T_over_rs'),exact(previous['F']),
                    exact((Fraction(row['T'])-Fraction(previous['T']))/Fraction(s['rs']))])
            if column==3:
                tokens=NUMBER.findall(cell)
                if tokens and bracket:
                    previous=member['_records'][bracket['previous_index']]
                    lo,hi=Fraction(previous['T'])/Fraction(s['rs']),Fraction(row['T'])/Fraction(s['rs'])
                    f0,f1=Fraction(previous['F']),Fraction(row['F'])
                    linear=exact(lo-f0*(hi-lo)/(f1-f0))
                else:linear=None
                return ([numeric(tokens[0],linear)] if tokens and bracket else [])+[
                    dict(target=tokens[1] if len(tokens)>1 else cell,own=None,
                         status='definition_unresolved',reason='quadratic source stencil unspecified')]
            if column==8:
                return number_list(cell,[exact(z) for z in row['zeros']])
            if column==9:
                result=number_list(cell,[row_value(member,row,'outer_slope')])
                result.append(dict(target=cell.split(';')[0],own=not bool(row['tangencies']),
                                   status='agrees' if cell.startswith('yes;') and not row['tangencies'] else 'label_or_membership_mismatch'))
                return result
            if column==11:
                pieces=[piece.strip() for piece in cell.split(';')]
                result=number_list(pieces[0],[exact(row['kept'])])
                if row['confirming'] is None:
                    result.append(missing(pieces[1]))
                else:
                    result.extend(number_list(pieces[1],[exact(row['confirming'])]))
                return result
            if column==12:
                reported=not cell.startswith('uncertified')
                return [dict(target=cell,own=row['certified'],status='agrees' if reported==row['certified'] else 'label_or_membership_mismatch')]
            if column in [13,14]:
                episode=member['episodes'][0]
                if column==13:
                    is_open=episode['closes_at'] is None
                    u=summary['last'] if is_open else episode['closes_at']
                    result=number_list(cell,[clock_value(member,u)])
                    result.append(dict(target=cell.startswith('open to'),own=is_open,
                        status='agrees' if cell.startswith('open to')==is_open else 'label_or_membership_mismatch'))
                    return result
                minimum=episode['minimum'];values=[exact(minimum['value']),clock_value(member,minimum['T_over_rs'])]
                if 'over certified records' in cell:
                    subset=episode['certified_minimum']
                    values.extend([None if subset is None else exact(subset['value']),
                                   None if subset is None else clock_value(member,subset['T_over_rs'])])
                    values.extend([episode['noncounting'],episode['count'],episode['uncertified']])
                return mixed(cell,values)
    return [dict(status='pending',reason='remaining categorical or compound binding')]


def precise(low,high):
    if high is None: return None
    if isinstance(high,int): return high
    return dict(value=str(high),error=str(abs(Decimal(str(high))-Decimal(str(low)))*32+Decimal('1e-150')))


def mixed(cell,values):
    token_pattern=NUMBER.pattern+r'|(?<![A-Za-z0-9_])-(?![A-Za-z0-9_])'
    tokens=re.findall(token_pattern,cell)
    if len(tokens)!=len(values):
        return [dict(status='pending',reason='mixed compound token count differs',tokens=tokens,own=values)]
    result=[]
    for token,value in zip(tokens,values):
        if token=='-': result.append(missing(token) if value is None else dict(status='arithmetic_mismatch',target=token,own=value))
        else: result.append(numeric(token,value))
    return result


def extra_cell(section,subsection,column,cell,cells,member,extras,fit_name):
    low,high=extras['controls']['80'],extras['controls']['160']
    name=cells[0]
    if section=='10' and member is not None:
        rows=[row for row in extras['source_audit']['rows'] if row['report']==extras['document'] and row['member']==name]
        if len(rows)!=1:return [dict(status='definition_unresolved',reason='source row lacks unique metadata binding')]
        own=rows[0]
        value={0:name,1:own['source'],2:own['actual']['bytes'],3:own['actual']['md5'],4:own['actual']['script']}[column]
        return ([numeric(cell,value)] if column==2 else
                [dict(target=cell,own=value,status='agrees' if cell==value else 'label_or_membership_mismatch')])
    if column==0 and member is not None:
        return [dict(target=cell,own=name,source=member['source'],status='agrees',reason='exact member/input identity')]
    if section=='7' and column==1 and name in high['catalogs']:
        catalog=high['catalogs'][name]
        return [dict(target=cell,own=catalog,status='input_unavailable' if catalog.get('status')=='input_unavailable'
                     else 'source_dependent_primitive',reason='supplied preparation provenance; own arithmetic does not rerun preparation')]
    if section=='5' and member is not None:
        s=member['scalars'];p=member['precision_scalars'];formation=member['formation']
        scaling=extras['document'].startswith('SCALING')
        schema=({'5.1':{1:'rs',3:'sigma_over_rs'},'5.4':{1:'m',2:'release'}} if scaling
                else {'5.3':{1:'m',2:'pF_over_m_isotropic',3:'own_pF_over_m'}})
        if column in schema.get(subsection,{}):
            key=schema[subsection][column]
            return number_list(cell,[exact(Fraction(s['R0'])/Fraction(s['rs'])) if key=='release' else p[key]])
        first_col=4 if (scaling and subsection=='5.1') or (not scaling and subsection=='5.3') else 3 if scaling and subsection=='5.4' else None
        if first_col is not None:
            if scaling and subsection=='5.1' and column==2:return number_list(cell,[p['tau'],p['tau_over_rs']])
            if column==first_col:
                if formation is None:return number_list(cell,[clock_value(member,member['mask_summary']['last'])])
                result=number_list(cell,[row_value(member,formation['row'],'T_over_rs')])
                if scaling:
                    result.append(dict(target=cell,own=formation['row']['certified'],status='agrees' if
                        ('uncertified' in cell)==(not formation['row']['certified']) else 'label_or_membership_mismatch'))
                return result
            if column>first_col:
                offset=column-first_col
                if scaling and subsection=='5.4':
                    keys={1:'R_t',2:'R_t_over_rs',3:'R_t_over_2gE',4:'gate',5:'end'}
                else:keys={1:'step',2:'R_t',3:'R_t_over_rs',4:'R_t_over_2gE',5:'gate',6:'end' if scaling else 'class'}
                key=keys.get(offset)
                if key=='end':return number_list(cell,[clock_value(member,member['mask_summary']['last_certified'])])
                if formation is None:return [missing(cell)]
                if key=='class':return [dict(target=cell,own=formation['row']['certified'],status='agrees' if cell=='certified' and formation['row']['certified'] else 'label_or_membership_mismatch')]
                if key=='step':return number_list(cell,[exact(Fraction(s['tau'])/Fraction(s['rs']))])
                if key:return number_list(cell,[row_value(member,formation['row'],key)])
        if not scaling and subsection=='5.1' and name in ['M3b_s4','P64_s4','M3c_s4_nk'] and column in [1,2,3]:
            ordinate={1:'T_f',2:'R_t_over_rs',3:'R_t_over_2gE'}[column]
            hi=extras['descriptive']['160']['fits']['mass_isotropic'];lo=extras['descriptive']['80']['fits']['mass_isotropic']
            index=hi['members'].index(name)
            return number_list(cell,[precise(lo['ordinates'][ordinate]['residuals'][index],hi['ordinates'][ordinate]['residuals'][index])])
        if not scaling and subsection=='5.1' and column==4:
            names={'M3b_s4':['M3b_s8'],'P64_s4':['P64_s2','P64_s8','P64_s4_tau4','P64_s4_tau2'],
                   'M3c_s4_nk':['M3c_s4_plain']}.get(name)
            if names:
                values=[]
                for control in names:
                    a=high['pairs'][name+'__'+control]['formation'];b=low['pairs'][name+'__'+control]['formation']
                    for field in ['T_over_rs','R_t_over_rs']:
                        left=extras['members'][name];right=extras['members'][control]
                        lv=row_value(left,left['formation']['row'],field)['rational']
                        rv=row_value(right,right['formation']['row'],field)['rational']
                        values.append(exact(Fraction(int(rv[0]),int(rv[1]))-Fraction(int(lv[0]),int(lv[1]))))
                return number_list(cell,values)
    if section=='5' and subsection=='5.3' and name in ['64','128'] and len(cells)==7:
        a,b=extras['members'][cells[1]],extras['members'][cells[2]]
        if column==0:return number_list(cell,[exact(a['scalars']['rs'])])
        if column in [1,2]:return [dict(target=cell,own=cells[column],status='agrees',reason='prescribed ordered width pair')]
        if column in [3,4,5]:
            key={3:'T_over_rs',4:'R_t_over_rs',5:'R_t_over_2gE'}[column]
            x=row_value(a,a['formation']['row'],key)['rational'];y=row_value(b,b['formation']['row'],key)['rational']
            value=Fraction(int(y[0]),int(y[1]))-Fraction(int(x[0]),int(x[1]))
            return number_list(cell,[exact(value)])
        if column==6:
            flag=a['formation']['row']['certified'] and b['formation']['row']['certified']
            return [dict(target=cell,own=flag,status='agrees' if (cell=='yes')==flag else 'label_or_membership_mismatch')]
    if subsection=='6.1' and name in extras['diagnostics']['members']:
        diagnostic=extras['diagnostics']['members'][name];stat=diagnostic['stats']
        if column==3:return number_list(cell,[stat['records'],len(diagnostic['dropped'])])
        if column==1:
            a=diagnostic['args'];newton=a.get('mix')=='newton'
            values=([a['nk_maxiter'],a['nk_krylov'],exact(a['end_above'])] if newton else [a['maxit']])
            metadata=diagnostic['script'] in cell and ('newton' in cell if newton else 'plain' in cell)
            if newton:metadata=metadata and ('set '+str(a.get('nk_set'))) in cell
            return number_list(cell,values)+[dict(target=cell,own=dict(script=diagnostic['script'],mix=a.get('mix'),set=a.get('nk_set')),
                status='agrees' if metadata else 'label_or_membership_mismatch')]
        if column in [4,5,6,7]:
            return number_list(cell,[{4:stat['stored_true'],5:stat['counting'],6:stat['stored_false'],7:len(stat['kept_above'])}[column]])
        if column in [9,10]:
            peak=stat['kept_maximum' if column==9 else 'prefix_kept_maximum']
            if peak is None:return [missing(cell)]
            values=[exact(peak['value'])]
            if column==9:values.append(clock_value(member,peak['T_over_rs']))
            return number_list(cell,values)
        if column in [12,13]:
            value=member['mask_summary']['last_kept_prefix' if column==12 else 'last_certified']
            return ([dict(target=cell,own=None,status='agrees' if cell.startswith('none') else 'label_or_membership_mismatch')]
                    if value is None else number_list(cell,[clock_value(member,value)]))
        if column==11:
            inside,outside=stat['banded_inside'],stat['banded_outside']
            values=[clock_value(member,t) for t in inside+outside[:6]]
            if len(outside)>6:values.append(len(outside)-6)
            return (number_list(cell,values) if values else
                    [dict(target=cell,own=[[],[]],status='agrees' if cell=='none (none)' else 'label_or_membership_mismatch')])
        if column==2:
            values=[];names=[]
            for engine in diagnostic['engines']:
                values.extend([clock_value(member,engine['first']),clock_value(member,engine['last']),engine['count']]);names.append(engine['engine'])
            return number_list(cell,values)+[dict(target=cell,own=names,
                status='agrees' if all(engine in cell for engine in names if engine is not None) else 'label_or_membership_mismatch')]
        if column==8:
            parts=cell.split(';');above=stat['confirming_above'];values=[len(above)]
            shown=above[:8]
            values.extend(clock_value(member,t) for t in shown)
            if len(above)>8:values.append(len(above)-8)
            maximum=stat['confirming_maximum']
            values.append(None if maximum is None else exact(maximum['value']))
            if '(of ' in cell:values.extend([stat['plain_kept_false'],stat['plain_following']])
            elif maximum is not None:values.append(clock_value(member,maximum['T_over_rs']))
            return mixed(cell,values)
    if section in ['1','3','7'] and name in high['catalogs']:
        catalog=high['catalogs'][name]; old=low['catalogs'][name]
        if catalog.get('status')=='input_unavailable':
            if (section=='1' and column==16) or (section=='3' and column==4) or (section=='7' and column>0):
                return [dict(target=cell,own=None,status='input_unavailable',reason='no supplied catalog')]
        else:
            if section=='1' and column==16:
                return number_list(cell,[catalog['onset'],exact(Fraction(catalog['onset'])/Fraction(member['scalars']['rs']))])
            if section=='3' and column==4:
                return number_list(cell,[catalog['onset']])
            if section=='7':
                if column==2:
                    values=[]
                    for key,value in catalog['columns'].items(): values.extend([int(key),value])
                    return number_list(cell,values)
                if column in [3,4,5]:
                    return number_list(cell,[catalog[{3:'top',4:'max_columns',5:'onset'}[column]]])
                if column==6:
                    comparisons=catalog['head_comparisons']
                    if not comparisons:return [missing(cell)]
                    return number_list(cell,[precise(old['head_comparisons'][q]['relative'],comparisons[q]['relative'])
                                             for q in ['N_B','E0','f_kin']])
    if subsection=='6.3' and name+'__'+cells[1] in high['pairs']:
        pair=high['pairs'][name+'__'+cells[1]]; previous=low['pairs'][name+'__'+cells[1]]
        if column==1:return [dict(target=cell,own=cells[1],status='agrees',reason='control pair belongs to fixed specified union')]
        if column==2:
            a=extras['members'][name]['scalars'];b=extras['members'][cells[1]]['scalars']
            expected=('preparation' if a['R0']!=b['R0'] else 'control_sigma' if a['sigma']!=b['sigma'] else
                      'control_step' if a['tau']!=b['tau'] else 'comparator' if cells[1]=='P64_s4_c19' else 'control_solver')
            return [dict(target=cell,own=expected,status='agrees' if cell.startswith(expected) else 'label_or_membership_mismatch')]
        if column==3:
            if pair['formation'] is None:return [missing(cell)]
            expected=' / '.join('certified' if flag else 'uncertified' for flag in pair['formation']['certified'])
            return [dict(target=cell,own=expected,status='agrees' if cell==expected else 'label_or_membership_mismatch')]
        if column in range(4,9):
            key={4:'clock',5:'radius',6:'relative',7:'normalized_radius',8:'root_fraction'}[column]
            if pair['formation'] is None:return [missing(cell)]
            a=extras['members'][name];b=extras['members'][cells[1]]
            ar,br=a['formation']['row'],b['formation']['row']
            def fraction_of(value):return Fraction(int(value['rational'][0]),int(value['rational'][1]))
            field={'clock':'T_over_rs','radius':'R_t','normalized_radius':'R_t_over_rs','root_fraction':'R_t_over_2gE'}.get(key,'R_t')
            difference=fraction_of(row_value(b,br,field))-fraction_of(row_value(a,ar,field))
            if key=='relative':difference/=Fraction(ar['R_t'])
            return number_list(cell,[exact(difference)])
        if column==9:
            a,b=pair['primary'],previous['primary']
            return mixed(cell,[a['count'],precise(b['first'],a['first']),precise(b['last'],a['last'])])
        if column in [10,11,12]:
            key={10:'E',11:'F',12:'centroid'}[column]
            a,b=pair['primary']['maxima'][key],previous['primary']['maxima'][key]
            return [missing(cell)] if a is None else number_list(cell,[precise(b['value'],a['value'])])
        if column in [13,14]:
            which='primary' if column==13 else 'diagnostic'
            a,b=pair[which],previous[which]; values=[]
            if column==14:values.append(a['count'])
            for key in ['R_t','relative']:
                x,y=a['maxima'][key],b['maxima'][key]
                values.append(None if x is None else precise(y['value'],x['value']))
                if key=='relative':values.append(None if x is None else precise(y['T_over_rs'],x['T_over_rs']))
            if column==14:values.append(precise(b['relative_at_16'],a['relative_at_16']))
            values.append(a['both_zero_count'])
            return mixed(cell,values)
    if section=='5' and subsection=='5.3' and name in ['certified','every_formation_record'] and column==0:
        return [dict(target=cell,own=name,status='agrees',reason='prescribed fit membership subset')]
    if fit_name and name in ['T_f','R_t_over_rs','R_t_over_2gE']:
        hi=extras['descriptive']['160']['fits'][fit_name]['ordinates'][name]
        lo=extras['descriptive']['80']['fits'][fit_name]['ordinates'][name]
        def p(key,index=None):
            return precise(lo[key] if index is None else lo[key][index],hi[key] if index is None else hi[key][index])
        if column==0:return [dict(target=cell,own=name,status='agrees',reason='prescribed ordinate label')]
        if column==1:
            if cell.startswith('not printed'):
                return number_list(cell,[hi['dof']])
            result=number_list(cell,[p('coefficients',i) for i in range(len(hi['coefficients']))])
            if fit_name=='width_certified' and name=='T_f' and len(result)==4:
                result[3].update(status='roundoff_zero',reason='independent exact width-clock identity has c3=0')
            return result
        if column==2:
            values=[]
            for i in range(len(hi['fitted'])):
                values.extend([precise(lo['ordinate'][i][0],hi['ordinate'][i][0]),p('fitted',i),p('residuals',i)])
            return number_list(cell,values)
        if column==3:
            if hi['dof']==0:
                return [dict(target=cell,own=dict(rms=hi['rms'],dof_rms=None,dof=0),status='roundoff_zero',
                             reason='analytic full-rank interpolation; zero residual to roundoff, residual-dof scatter undefined')]
            return number_list(cell,[p('rms'),p('dof_rms')])
        if column==4:
            if hi['dof']==0:
                return number_list(cell,[p('max_residual'),0])
            return number_list(cell,[p('max_residual')])
    if section=='5' and name in ['certified','every_formation_record']:
        hi=extras['descriptive']['160']['rank_tables'][name]
        lo=extras['descriptive']['80']['rank_tables'][name]
        if column in [2,3,4]:return number_list(cell,[hi[{2:'rows',3:'rank',4:'rows_minus_rank'}[column]]])
        if column==5:return number_list(cell,[precise(a,b) for a,b in zip(lo['singular_values'],hi['singular_values'])])
        if column==1:
            values=[]
            for member_name in hi['members']:
                s=extras['members'][member_name]['scalars'];values.extend([exact(s[k]) for k in ['rs','sigma','tau']])
            return number_list(cell,values)
    return None


def prose_calculations(line,members,extras):
    """Bind prose quantities to named own rows, never to a number pool."""
    match=re.match(r'- ([A-Za-z0-9_]+): (\d+): ',line)
    if match and match[1] in members:
        member=members[match[1]];values=[];checks=[]
        clauses=re.split(r'; (?=\d+: )',line.split(': ',1)[1])
        if len(clauses)!=len(member['episodes']):return [dict(status='label_or_membership_mismatch',reason='episode count differs')]
        for index,(clause,episode) in enumerate(zip(clauses,member['episodes']),1):
            first=member['_records'][episode['source_indices'][0]];last=member['_records'][episode['source_indices'][-1]]
            minimum=member['_records'][episode['minimum']['source_index']]
            values.extend([index,row_value(member,first,'T_over_rs'),row_value(member,last,'T_over_rs'),episode['count'],
                           row_value(member,first,'R_t'),row_value(member,last,'R_t'),row_value(member,minimum,'F'),row_value(member,minimum,'T_over_rs')])
            if episode['closes_at'] is not None:values.append(clock_value(member,episode['closes_at']))
            if episode['noncounting']:values.extend([episode['noncounting'],episode['count']])
            if episode['uncertified']:values.append(episode['uncertified'])
            is_open='open at the last record' in clause
            checks.append(dict(target=is_open,own=episode['closes_at'] is None,status='agrees' if is_open==(episode['closes_at'] is None) else 'label_or_membership_mismatch',episode=index))
            if '(unc.)' in clause:
                checks.append(dict(target='outside prefix',own=minimum['certified'],status='agrees' if not minimum['certified'] else 'label_or_membership_mismatch'))
        return number_list(line,values)+checks
    match=re.match(r'- ([A-Za-z0-9_]+): (\d+) records do not count:',line)
    if match and match[1] in members:
        member=members[match[1]];rows=[r for r in member['_records'].values() if not r['counts']]
        values=[len(rows)];checks=[]
        clauses=line.split('records do not count: ',1)[1].split('); ')
        if len(clauses)!=len(rows):return [dict(status='label_or_membership_mismatch',reason='failing-row clause count differs')]
        for clause,row in zip(clauses,rows):
            values.append(row_value(member,row,'T_over_rs'))
            reasons=[reason for reason in ['kept','confirming'] if reason+' residual above' in clause]
            values.extend(exact(Fraction(1,1000)) for _ in reasons)
            values.append(row_value(member,row,'kept'))
            if ', confirming ' in clause:values.append(row_value(member,row,'confirming'))
            else:checks.append(dict(target='confirming omitted',own=row['confirming'],status='agrees' if row['confirming'] is None else 'label_or_membership_mismatch'))
            expected=[key for key in ['kept','confirming'] if row[key] is not None and row[key]>0.001]
            checks.append(dict(target=reasons,own=expected,source_index=row['source_index'],status='agrees' if reasons==expected else 'label_or_membership_mismatch'))
        return mixed(line[2:] if line.startswith('- ') else line,values)+checks
    if line.startswith('Over the 0.5 r_s after the first saved record'):
        pairs=re.findall(r'([A-Za-z0-9_]+) ([-.\d]+) to ([-.\d]+)',line)
        values=[];differences=[]
        for name,_,_ in pairs:
            if name not in members:continue
            member=members[name];first=member['formation']['row'];later=member['formation']['offsets']['0.5']['row']
            a=row_value(member,first,'R_t_over_2gE');b=row_value(member,later,'R_t_over_2gE')
            av=Fraction(int(a['rational'][0]),int(a['rational'][1]));bv=Fraction(int(b['rational'][0]),int(b['rational'][1]))
            differences.append(bv-av);values.extend([a,b])
        # 2GE's leading digit is part of an identifier excluded by NUMBER.
        return number_list(line,[exact(Fraction(1,2)),exact(min(differences)),exact(max(differences))]+values)
    return None


def fit_prose(line,line_number,members,extras):
    scaling=extras['document'].startswith('SCALING')
    ranges=([(297,306,'size_certified'),(307,321,'size_labelled'),(340,353,'width_certified'),(354,366,'width_labelled')]
            if scaling else [(258,278,'mass_isotropic'),(286,298,'mass_own')])
    fit_name=next((name for lo,hi,name in ranges if lo<=line_number<=hi),None)
    if fit_name:
        a=extras['descriptive']['160']['fits'][fit_name];b=extras['descriptive']['80']['fits'][fit_name]
        if line.startswith('- model:'):
            sample=next(iter(a['ordinates'].values()))
            values=([len(sample['coefficients']),1,3] if fit_name.startswith('size_') else
                    [len(sample['coefficients']),2] if fit_name=='mass_isotropic' else
                    [len(sample['coefficients']),exact(Fraction('0.3'))] if fit_name=='mass_own' else [])
            return number_list(line,values)+[dict(status='adopted_definition',reason='Chosen diagnostic design checked against frozen contract and saved own design; no new law or fitted model selected.')]
        if line.startswith('- members:'):

            member_names=line.split(': ',1)[1].split(' (')[0].split(', ')
            sample=next(iter(a['ordinates'].values()))
            return number_list(line,[len(a['members']),len(sample['coefficients']),sample['dof']])+[
                dict(target=member_names,own=a['members'],status='agrees' if member_names==a['members'] else 'label_or_membership_mismatch')]
        if line.startswith('- x per member:'):
            return number_list(line,[precise(y[0],x[0]) for x,y in zip(a['variables'],b['variables'])])
        if line.startswith('- (z, h) per member:'):
            return number_list(line,[precise(y[j],x[j]) for x,y in zip(a['variables'],b['variables']) for j in [0,1]])
        match=re.match(r"- ([A-Za-z0-9_]+): the fit's value",line)
        if match:
            member=members[match[1]];values=[]
            for key in ['T_f','R_t_over_rs','R_t_over_2gE']:
                values.append(precise(b['ordinates'][key]['predictions'][match[1]],a['ordinates'][key]['predictions'][match[1]]))
            if 'holds no trapping zero' in line:
                values.append(clock_value(member,member['mask_summary']['last']))
            elif "member's record" in line:
                values.extend(row_value(member,member['formation']['row'],key) for key in ['T_over_rs','R_t_over_rs','R_t_over_2gE'])
            return number_list(line,values)
    if (scaling and line_number in [323,324,325]) or (not scaling and line_number in [280,281,282]):
        key=line.strip().split(':')[0][2:];family='size' if scaling else 'mass_isotropic'
        a=extras['descriptive']['160']['powers'][family][key];b=extras['descriptive']['80']['powers'][family][key]
        return number_list(line,[precise(b[k],a[k]) for k in ['p','Y0','A']])
    if scaling and line_number in [340,354]:
        name='width_certified' if line_number==340 else 'width_labelled'
        a=extras['descriptive']['160']['fits'][name];sample=next(iter(a['ordinates'].values()))
        values=[len(a['members']),len(sample['coefficients']),sample['dof']] if line_number==340 else [len(a['members'])]
        return number_list(line,values)
    if scaling and line_number in [307,312,359]:
        member=members['S4_128_s4'];values=([exact(member['scalars'][k]) for k in ['m','rs','sigma']] if line_number==307 else [])
        return number_list(line,values)+[dict(target='uncertified formation',own=member['formation']['row']['certified'],status='agrees' if not member['formation']['row']['certified'] else 'label_or_membership_mismatch')]
    if scaling and line_number==286:
        names=['S4_16_s4','RS32_s4','P64_s4','S4_128_s4'];formed=sum(members[n]['formation'] is not None and members[n]['formation']['row']['certified'] for n in names)
        return number_list(line,[formed,len(names)])
    if (scaling and line_number==322) or (not scaling and line_number==279):
        name='size_labelled' if scaling else 'mass_isotropic';a=extras['descriptive']['160']['fits'][name]
        reported=line.split(': ',1)[1].split(';')[0].split(', ')
        return number_list(line,[len(a['members'])])+[dict(target=reported,own=a['members'],status='agrees' if reported==a['members'] else 'label_or_membership_mismatch')]
    return None


def member_prose(line,members):
    match=re.match(r'- ([A-Za-z0-9_]+) \(',line)
    if not match or match[1] not in members:return None
    member=members[match[1]];s=member['scalars'];summary=member['mask_summary'];formation=member['formation']
    claims=[];covered=set()
    def bind(pattern,values):
        for match in re.finditer(pattern,line):
            expected=values(match) if callable(values) else values
            if len(match.groups())!=len(expected):raise ValueError('bad prose occurrence binding')
            for group,value in enumerate(expected,1):
                span=match.span(group)
                if span in covered:continue
                claims.append(dict(**numeric(match[group],value),span=list(span),binding=pattern));covered.add(span)
    bind(r'\((\d*\.?\d+), (\d+)\)',[exact(s['m']),exact(s['rs'])])
    bind(r'\) s(\d+) t(\d+)',[exact(s['sigma']),exact(s['tau'])])
    bind(r'kappa_sea (\d+)',[member['_args'].get('kappa_sea')]);bind(r'(\d+) sites',[s['sites']]);bind(r'tau (\d+)',[exact(s['tau'])]);bind(r'N_B (\d+)',[exact(s['NB'])])
    bind(r'sigma/a (\d+)',[exact(s['sigma'])]);bind(r'point at (\d+)',[exact(s['rs'])])
    bind(r'width point \((\d+),',[exact(s['rs'])]);bind(r'release at (\d+) r_s',[exact(Fraction(s['R0'])/Fraction(s['rs']))])
    bind(r'finished to (\d+)',[clock_value(member,summary['last'])]);bind(r'to (\d*\.?\d+)\)',[clock_value(member,summary['last'])])
    bind(r'records to (\d*\.?\d+)',[clock_value(member,summary['last'])])
    bind(r'last converged record (\d*\.?\d+)',[clock_value(member,summary['last_certified'])])
    if formation:
        row=formation['row']
        bind(r'first saved zero (\d*\.?\d+)',[row_value(member,row,'T_over_rs')])
        bind(r'at R_t (\d*\.?\d+) a \((\d*\.?\d+) r_s, root fraction (\d*\.?\d+), gate (\d*\.?\d+)\)',[row_value(member,row,k) for k in ['R_t','R_t_over_rs','R_t_over_2gE','gate']])
    if summary['first_failure']:
        failure=summary['first_failure']
        bind(r'first non-counting record (\d*\.?\d+)',[clock_value(member,failure['T_over_rs'])])
    bind(r'above (1e-3)',[exact(Fraction(1,1000))])
    # This phrase is known to mean kept-only in the source, and needs its
    # explicit wording correction in the semantic map.
    kept_fail=next((r for r in member['_records'].values() if r['kept']>.001),None)
    if kept_fail:bind(r'first record above 1e-3 \((\d*\.?\d+)\)',[row_value(member,kept_fail,'T_over_rs')])
    # Single-field metadata and quoted comparator labels are source claims.
    bind(r'the (1)-field production',[1])
    unbound=[dict(token=m[0],span=list(m.span())) for m in NUMBER.finditer(line) if m.span() not in covered]
    claims.append(dict(status='pending' if unbound else 'source_dependent_primitive',unbound_numeric_occurrences=unbound,
                       reason='Role/provenance and interpretation use manual occurrence map; numeric spans independently bound where present.'))
    return claims


def historical_prose(line,members,extras):
    history=extras['diagnostics']['historical']['old_report']['members']
    diagnostics=extras['diagnostics']['members']
    if line.startswith('- Certified extents that move (T/r_s):'):
        body=line.split(': ',1)[1];values=[];checks=[]
        for clause in body.split('; '):
            for name in sorted(members,key=len,reverse=True):
                if clause.startswith(name+' '):
                    old=history[name]['mask_summary']['last_certified'];new=members[name]['mask_summary']['last_certified']
                    if ' to ' in clause:
                        if old is not None:values.append(clock_value(members[name],old))
                        values.append(clock_value(members[name],new))
                    break
        # The final sentence has unchanged truncated endpoints in parentheses.
        if 'Unchanged and short of the last record:' in body:
            tail=body.split('Unchanged and short of the last record:',1)[1]
            for name,_ in re.findall(r'([A-Za-z0-9_]+) \(([^)]+)\)',tail):
                old=history[name]['mask_summary']['last_certified'];new=members[name]['mask_summary']['last_certified']
                
                if new is not None:values.append(clock_value(members[name],new))
                checks.append(dict(target='unchanged',own=old==new,status='agrees' if old==new else 'label_or_membership_mismatch'))
        return number_list(line,values)+checks
    match=re.match(r'  - ([A-Za-z0-9_]+): (counting records|first non-counting record)',line)
    if not match or match[1] not in members:return None
    name=match[1];member=members[name];old=history[name];s=member['scalars'];values=[];checks=[]
    for clause in re.split(r'; (?![^()]*\))',line.split(': ',1)[1]):
        if clause.startswith('counting records'):
            values.extend([old['mask_summary']['counting'],member['mask_summary']['counting']])
        elif clause.startswith('non-counting records'):
            values.extend([old['mask_summary']['records']-old['mask_summary']['counting'],member['mask_summary']['records']-member['mask_summary']['counting']])
        elif clause.startswith('first non-counting record'):
            failure=member['mask_summary']['first_failure'];values.append(clock_value(member,failure['T_over_rs']))
            reasons=[r for r in ['kept','confirming'] if r+' residual above' in clause]
            values.extend(exact(Fraction(1,1000)) for _ in reasons)
            checks.append(dict(target=reasons,own=failure['reasons'],status='agrees' if reasons==failure['reasons'] else 'label_or_membership_mismatch'))
        elif clause.startswith('min R_c over the certified records'):
            a=old['centroid_minimum'];b=member['centroid_minimum']
            if a:values.extend([exact(a['value']),clock_value(member,a['T_over_rs'])])
            if b:values.extend([exact(b['value']),clock_value(member,b['T_over_rs']),exact(Fraction(b['value'])/Fraction(s['rs']))])
        elif clause.startswith('episode '):
            index=int(clause.split()[1]);a=old['episodes'][index-1]['certified_minimum'];b=member['episodes'][index-1]['certified_minimum']
            values.append(index)
            if a:values.append(exact(a['value']))
            if b:values.extend([exact(b['value']),clock_value(member,b['T_over_rs'])])
        else:
            # Flags on matched clocks and offsets bind to full old/new rows.
            numbers=NUMBER.findall(clause)
            if 'matched clocks' in clause:
                times=[int(t) for t in numbers]
                for t in times:
                    row=next(r for r in member['_records'].values() if r['T_over_rs']==t)
                    oldmask=diagnostics[name]['old_report'];oldrow=dict(counts=oldmask['counts'][row['source_index']],certified=oldmask['prefix'][row['source_index']])
                    values.append(row_value(member,row,'T_over_rs'))
                    checks.append(dict(target='uncertified removed',clock=t,own=[oldrow['certified'],row['certified']],status='agrees' if not oldrow['certified'] and row['certified'] else 'label_or_membership_mismatch'))
            elif 'T_f +' in clause:
                for text in numbers:
                    offset=str(float(text)).removesuffix('.0');row=member['formation']['offsets'][offset]['row']
                    oldformation=old['formation'];oldrow=oldformation['offsets'][offset]['row']
                    values.append(exact(Fraction(row['T'])/Fraction(s['rs'])-Fraction(member['formation']['row']['T'])/Fraction(s['rs'])))
                    checks.append(dict(target='uncertified removed',offset=offset,own=[oldrow['certified'],row['certified']],status='agrees' if not oldrow['certified'] and row['certified'] else 'label_or_membership_mismatch'))
            else:
                for text in numbers:
                    t=int(text);row=next(r for r in member['_records'].values() if r['T_over_rs']==t)
                    oldmask=diagnostics[name]['old_report'];oldrow=dict(counts=oldmask['counts'][row['source_index']],certified=oldmask['prefix'][row['source_index']])
                    values.append(row_value(member,row,'T_over_rs'))
                    good=(not oldrow['counts'] and row['counts'])
                    checks.append(dict(target=clause,own=[oldrow['counts'],row['counts'],oldrow['certified'],row['certified']],status='agrees' if good else 'label_or_membership_mismatch'))
    return number_list(line,values)+checks


def control_change_prose(line,members,extras):
    match=re.match(r'  - ([A-Za-z0-9_]+) against ([A-Za-z0-9_]+):',line)
    if not match:return None
    pairs=extras['diagnostics']['historical']['old_report']['control_pairs']
    key=match[1]+'__'+match[2]
    if key not in pairs:key=match[2]+'__'+match[1]
    if key not in pairs:return [dict(status='input_unavailable',reason='ordered prescribed pair absent')]
    left,right=key.split('__');a,b=members[left],members[right]
    old=pairs[key];new=extras['controls']['160']['pairs'][key]
    def fraction(value):return Fraction(int(value['rational'][0]),int(value['rational'][1]))
    def peak_value(block,field):
        peak=block['primary']['maxima'][field]
        if peak is None:return None
        i,j=peak['indices'][0];ar,br=a['_records'][i],b['_records'][j]
        if Fraction(ar['T'])!=Fraction(br['T']):raise ValueError('peak rows do not share physical clock')
        if field=='E':value=abs(fraction(row_value(b,br,'E_over_E0'))-fraction(row_value(a,ar,'E_over_E0')))
        elif field=='centroid':value=abs(Fraction(br['R_c'])-Fraction(ar['R_c']))/Fraction(a['scalars']['rs'])
        elif field=='F':value=abs(Fraction(br['F'])-Fraction(ar['F']))
        else:
            value=abs(Fraction(br['R_t'])-Fraction(ar['R_t']))
            if field=='relative':value/=Fraction(ar['R_t'])
        return exact(value)
    values=[];checks=[]
    for clause in line.split(': ',1)[1].split('; '):
        if clause.startswith('common certified clocks'):values.extend([old['primary']['count'],new['primary']['count']])
        elif clause.startswith('last common clock'):
            for block in [old,new]:
                t=block['primary']['last'];values.append(None if t is None else clock_value(a,float(t)))
        elif clause.startswith('common zero records'):values.extend([old['primary']['both_zero_count'],new['primary']['both_zero_count']])
        elif clause.startswith('raw common clocks'):values.extend([old['diagnostic']['count'],new['diagnostic']['count']])
        elif clause.startswith('max '):
            field=('E' if '|d E/E0|' in clause else 'F' if '|d min(' in clause else 'centroid' if '|d R_c|' in clause else 'relative' if '|d R_t|/R_t' in clause else 'R_t')
            values.extend([peak_value(old,field),peak_value(new,field)])
        elif clause.startswith('at '):
            for block in [old,new]:
                peak=block['primary']['maxima']['relative'];values.append(None if peak is None else clock_value(a,float(peak['T_over_rs'])))
        elif clause.startswith('classes '):
            reported=clause.removeprefix('classes ').split(' to ')
            actual=[]
            for block in [old,new]:actual.append(' / '.join('certified' if flag else '30.3 only' for flag in block['formation']['certified']))
            checks.append(dict(target=reported,own=actual,ordered_pair=key,status='agrees' if reported==actual else 'label_or_membership_mismatch'))
        else:return [dict(status='pending',reason='historical control clause not bound',text=clause)]
    text=line.strip().removeprefix('- ').replace('30.3 only','kept-only').replace('N - |v|','N_minus_abs_v')
    text=re.sub(r'\bnone\b','-',text)
    return mixed(text,values)+checks


def governing_prose(line,line_number,members,extras):
    document=extras['document']
    if not document.startswith('LADDER_certification_list'):return None
    if line.startswith('The test also makes P64_s4_tau4'):
        tau=members['P64_s4_tau4'];row=next(r for r in tau['_records'].values() if r['T_over_rs']==13.4375)
        values=[row_value(tau,row,'T_over_rs'),row_value(tau,row,'confirming'),clock_value(tau,tau['mask_summary']['last_certified'])]
        raw_bindings=[]
        for name in ['S4_R2_m0p2','M3a_s2']:
            member=members[name];failure=member['mask_summary']['first_failure'];index=failure['source_index']
            raw=json.loads(Path(member['source']).read_text())['records'][index]
            if raw['T']!=failure['T']:raise ValueError('raw following-entry clock binding differs')
            kept_index=failure['kept_index'];following=raw['iteration_history'][kept_index+1]
            values.extend([clock_value(member,failure['T_over_rs']),kept_index,exact(failure['kept'])])
            values.extend(exact(v) for v in following[:3]) if name=='S4_R2_m0p2' else values.append(exact(following[1]))
            raw_bindings.append(dict(member=name,source_index=index,kept_index=kept_index,following_index=kept_index+1))
        return number_list(line,values)+[dict(status='source_dependent_primitive',raw_bindings=raw_bindings,reason='Saved iteration components checked directly; dE does not enter the gate.')]
    if line.startswith("Consequence for the ruling's live text:"):
        member=members['P64_s4_tau2'];old=extras['diagnostics']['members']['P64_s4_tau2']['intermediate_rule']['end']
        return number_list(line,[exact(member['scalars']['tau']),clock_value(member,old),clock_value(member,member['mask_summary']['last_certified'])])
    if line.startswith('With these named, P64_s2 and RS32_s2'):
        ends=[members[n]['mask_summary']['last_certified'] for n in ['P64_s2','RS32_s2']]
        if ends[0]!=ends[1]:raise ValueError('shared extent statement lacks identical own endpoints')
        return number_list(line,[clock_value(members['P64_s2'],ends[0])])
    return None


# Exact whole-line dispositions authored in manual occurrence map section 8.
# They are not numerical passes and never consume a mixed numeric claim.
MANUAL_DISPOSITIONS={}
def add_manual_lines(document,lines,status,reference):
    for line in lines:MANUAL_DISPOSITIONS[(document,line)]=(status,reference)

add_manual_lines('S',[5,36,62,90,134,160,186,212,238,264,385,387,411,420,468,478,484,43,118],
                 'heading_definition','8.1 and 8.2')
add_manual_lines('M',[5,28,50,73,109,132,158,184,210,307,309,328,334,350,369,378,384,36,96],
                 'heading_definition','8.1 and 8.2')
add_manual_lines('S',[158,266,66,92,136,162,188,214,240,288,378,301,314,346,361,333,369,389,422,444,486,331],
                 'adopted_definition','8.1')
add_manual_lines('M',[130,236,238,54,75,111,134,160,186,212,301,262,268,290,311,336,352,386,284],
                 'adopted_definition','8.1')
add_manual_lines('L',[5],'adopted_definition','8.1')
add_manual_lines('S',[282],'required_scope_correction','8.1; universal fit wording correction')
add_manual_lines('M',[250],'required_scope_correction','8.1; universal fit wording correction')
add_manual_lines('S',[295,327],'conditional_analytic_definition','8.1; separate comparator assumptions retained')
add_manual_lines('S',[22],'heading_definition','8.2; control-table introducer and internal locator')
add_manual_lines('M',[19],'heading_definition','8.2; control-table introducer and internal locator')
add_manual_lines('S',[512,513],'metadata_and_source_process','8.2; file identities and execution attribution; payloads separately pending')
add_manual_lines('M',[407,408],'metadata_and_source_process','8.2; file identities and execution attribution; payloads separately pending')
add_manual_lines('S',[480,482,508,509,510,514,515],'metadata_and_source_process','8.2')
add_manual_lines('M',[380,382,403,404,405,409,410],'metadata_and_source_process','8.2')
add_manual_lines('S',[481],'partially_defined_format','8.2; twenty defined arrays exact, five auxiliary definitions unresolved')
add_manual_lines('M',[381],'partially_defined_format','8.2; twenty defined arrays exact, five auxiliary definitions unresolved')
add_manual_lines('S',[516],'required_documentary_correction','8.2; copied section locator does not locate the numbers')
add_manual_lines('M',[411],'metadata_and_source_process','8.2; W4 locator')
add_manual_lines('S',[476],'input_unavailable_in_named_collection','8.2')
add_manual_lines('M',[374,376],'input_unavailable_in_named_collection','8.2')
add_manual_lines('L',[18],'metadata_and_source_process','8.2')
add_manual_lines('M',[3],'source_process','8.2; program/version and stored-files attribution')
add_manual_lines('F',[1,3,4],'source_process','8.2; complete supplied provider usage record retained')
add_manual_lines('F',[6,12],'adopted_definition','8.2')
add_manual_lines('S',[367],'adopted_definition','8.3; stipulated design and column normalization, not a measured dependence')


def manual_disposition(entry):
    document=entry['document']
    prefix=('S' if document.startswith('SCALING') else 'M' if document.startswith('MASS') else
            'L' if document.startswith('LADDER_certification_list') else 'F' if document.startswith('ReviewA_cert_rule') else None)
    key=(prefix,entry['line'])
    if key not in MANUAL_DISPOSITIONS:return
    if entry.get('status')!='pending':return
    status,reference=MANUAL_DISPOSITIONS[key]
    entry['status']=status
    entry['manual_disposition_reference']='LADDERS_v1p3_manual_occurrence_map_v1.md section '+reference
    entry['scope']='Manual interpretation/definition/provenance disposition; not an arithmetic agreement or a physical-law clearance.'


def metadata_projection(line,line_number,members,extras):
    scaling=extras['document'].startswith('SCALING');mass=extras['document'].startswith('MASS')
    if not (scaling or mass):return None
    s=lambda n,k:exact(members[n]['scalars'][k])
    text=re.sub(r'^#+\s+(?:\d+(?:\.\d+)?\s+)?','',line)
    if line_number==1:
        if scaling:
            names=['S4_16_s4','RS32_s4','P64_s4','S4_128_s4']
            values=[s('P64_s4','m'),s('P64_s4','sigma')]+[s(n,'rs') for n in names]+[s('P64_s4','rs'),s('S4_128_s4','rs'),exact(Fraction(members['S4_R2_m0p2']['scalars']['R0'])/Fraction(members['S4_R2_m0p2']['scalars']['rs']))]
        else:
            names=['M3b_s4','P64_s4','M3a_s4','M3c_s4_nk','M3d_s4']
            values=[s('P64_s4','sigma'),s('P64_s4','tau')]+[s(n,'m') for n in names[:3]]+[s('P64_s4','rs'),s('M3c_s4_nk','m'),s('M3c_s4_nk','rs'),s('M3d_s4','m'),s('M3d_s4','rs')]
        checks=[members[n]['scalars']['sigma']==4 for n in names]
        return number_list(text,values)+[dict(status='agrees' if all(checks) else 'label_or_membership_mismatch',members=names,quantity='common sigma/a')]
    if scaling and line_number==284:return number_list(text,[s('P64_s4','sigma')])
    if scaling and line_number==376:
        return number_list(text,[exact(Fraction(members[n]['scalars']['R0'])/Fraction(members[n]['scalars']['rs'])) for n in ['S4_R2_m0p2','P64_s4']])
    if scaling and line_number==442:return number_list(text,[s('RS32_s4','rs')])
    if mass and line_number==254:
        names=['M3b_s4','P64_s4','M3c_s4_nk'];count=sum(members[n]['formation'] is not None and members[n]['formation']['row']['certified'] for n in names)
        return number_list(text,[count,2])
    if mass and line_number==299:
        names=['M3b_s4','P64_s4','M3a_s4'];count=sum(members[n]['formation'] is not None and members[n]['formation']['row']['certified'] for n in names)
        return number_list(text,[s('P64_s4','rs'),count])
    if scaling and line_number==3:
        projection=line.split('The size member ',1)[1]
        return number_list(projection,[s('RS32_s4','rs'),1,s('RS32_s4','tau'),s('RS32_s4','sites')])+[dict(status='source_process',reason='Producer versions, stored-only composition and execution attribution are documentary metadata; raw member projection checked separately.')]
    return None


def integration_prose(line,line_number,members,extras):
    if not extras['document'].startswith('LADDER_certification_list'):return None
    diagnostics=extras['diagnostics']['members']
    if line.startswith('Still open from the ruling'):
        fragment=line.split('what the producer',1)[1].split(', and the provenance',1)[0]
        firsts=[diagnostics[n]['residual_invN']['above_summary']['first'] for n in ['P64_s2','RS32_s2']]
        values=[exact(Fraction(1,1000))]+[clock_value(members[n],t) for n,t in zip(['P64_s2','RS32_s2'],firsts)]+[exact(Fraction(1,10000))]
        good=all(members[n]['_records'][r['source_index']]['kept']<.0001 and
                 members[n]['_records'][r['source_index']]['confirming'] is not None and
                 members[n]['_records'][r['source_index']]['confirming']<.0001
                 for n in ['P64_s2','RS32_s2'] for r in diagnostics[n]['residual_invN']['above'])
        return number_list(fragment,values)+[dict(status='agrees' if good else 'label_or_membership_mismatch',
            claim='kept and applicable confirming below 1e-4 on the flagged integration-residual rows'),
            dict(status='source_process',reason='Open-item wording and quoted obsolete clock16 clause are documentary history; manual map sections7/8, no present prefix assertion.')]
    if not line.startswith('1. **residual_invN**'):return None
    results=[]
    pattern=r'Its (\d+) large values on P64_s2 \(([-.\d]+) to ([-.\d]+), up to ([-.\d]+)\) and (\d+) on RS32_s2 \(([-.\d]+) to ([-.\d]+), up to ([-.\d]+)\)'
    match=re.search(pattern,line)
    if match is None:raise ValueError('integration-diagnostic prose pattern changed')
    values=[];rowsets=[]
    for name in ['P64_s2','RS32_s2']:
        block=diagnostics[name]['residual_invN'];summary=block['above_summary'];member=members[name]
        values.extend([summary['count'],clock_value(member,summary['first']),clock_value(member,summary['last']),exact(block['maximum']['value'])])
        rowsets.append((name,block))
    results.extend(number_list(match[0],values))
    for name,target in [('P64_s2','0.464'),('RS32_s2','3.08')]:
        actual=diagnostics[name]['residual_invN']['maximum']['value']
        results.append(dict(target=target,own=actual,strict_upper_bound_holds=Fraction(actual)<=Fraction(target),
                            status='required_scope_correction' if Fraction(actual)>Fraction(target) else 'agrees',
                            reason='Nearest descriptive rounding is checked separately; qualify the rounded maximum as approximate rather than treating it as an exact upper bound.'))
    ratios=[(n,b['above_summary']['max_ratio_row']) for n,b in rowsets]
    winner=max(ratios,key=lambda pair:Decimal(pair[1]['precision']['160']['ratio']))
    bound=re.search(r'change by up to (\d+) times',line)
    if bound is None:raise ValueError('integration-diagnostic factor claim changed')
    own=winner[1]['precision']['160']['ratio'];passed=Decimal(own)<=Decimal(bound[1])
    results.append(dict(target=bound[1],own=own,comparison='maximum coarse/fine ratio <= printed upper factor',
                        member=winner[0],source_index=winner[1]['source_index'],clock=winner[1]['T_over_rs'],
                        status='agrees' if passed else 'arithmetic_mismatch'))
    gauge=all(any(interval[0]==0 for interval in r['gauge']) and
              r['W_negative']['min_W_over_max']<0 and r['W_negative']['inside_gauge_intervals']
              for _,block in rowsets for r in block['above'])
    before=all(r['source_index']<members[n]['formation']['row']['source_index'] for n,block in rowsets for r in block['above'])
    results.extend([dict(claim='central gauge interval and negative W on all flagged rows',own=gauge,status='agrees' if gauge else 'label_or_membership_mismatch'),
                    dict(claim='all flagged rows strictly before first saved formation',own=before,status='agrees' if before else 'label_or_membership_mismatch')])
    match=re.search(r'P64_s2 at or below ([\deE.+-]+) from (\d+) to (\d+)',line)
    if match is None:raise ValueError('postformation integration bound changed')
    own_rows=[r for r in diagnostics['P64_s2']['residual_invN']['rows'] if 12<=r['T_over_rs']<=40]
    maximum=max(own_rows,key=lambda r:r['value'])
    results.append(dict(target=match[1],own=maximum['value'],source_index=maximum['source_index'],
                        comparison='inclusive saved-clock range 12 <= T/rs <= 40; maximum <= printed bound',
                        status='agrees' if Fraction(maximum['value'])<=Fraction(match[1]) else 'arithmetic_mismatch'))
    results.extend(number_list(match[2]+' to '+match[3],[clock_value(members['P64_s2'],12),clock_value(members['P64_s2'],40)]))
    results.append(dict(status='definition_and_source_provenance',reason='Constraint-diagnostic formula, finite-difference step and foreign program/line-number attributions are supplied non-code definitions, not independently verified integrator logic. The explicit unverified J/v mechanism is preserved; manual map sections7/8.'))
    return results


def certification_change_prose(line,line_number,members,extras):
    diagnostics=extras['diagnostics']['members']
    changed=[n for n,m in members.items() if diagnostics[n]['intermediate_rule']['end']!=m['mask_summary']['last_certified']]
    if extras['document'].startswith('LADDER_certification_list') and line_number==1:
        return [dict(target='four changed members',own=len(changed),members=changed,status='agrees' if len(changed)==4 else 'arithmetic_mismatch'),
                dict(status='heading_definition',reason='Version labels are identifiers; count compares every intermediate/current prefix.')]
    if line.startswith('- That test shortens extents of LADDER_certification_list'):
        pattern=r'([A-Za-z0-9_]+) (none|[-.\d]+) to (none|[-.\d]+) \(record ([-.\d]+): kept ([-.\deE]+), following entry ([-.\deE]+)\)'
        matches=list(re.finditer(pattern,line));values=[];named=[];checks=[]
        for match in matches:
            name=match[1];named.append(name);member=members[name];failure=member['mask_summary']['first_failure']
            old=diagnostics[name]['intermediate_rule']['end'];new=member['mask_summary']['last_certified']
            for text,value in [(match[2],old),(match[3],new)]:
                if text=='none':checks.append(dict(target='none',own=value,status='agrees' if value is None else 'label_or_membership_mismatch'))
                else:values.append(clock_value(member,value))
            values.extend([clock_value(member,failure['T_over_rs']),exact(failure['kept']),exact(failure['confirming'])])
        values.extend([len(members)-len(changed),len(members)])
        checks.append(dict(target='no other member',named_changes=named,own_global_changes=changed,
                           status='required_scope_correction' if set(named)!=set(changed) else 'agrees',
                           reason='Name the report-member restriction or enumerate all four global changes; manual map sections2/8.'))
        return number_list(line,values)+checks
    if extras['document'].startswith('LADDER_certification_list') and line_number==3:
        fragment=line.split('It shortens',1)[1]
        return number_list(fragment,[len(members)-len(changed)])+[
            dict(target=4,own=len(changed),status='agrees' if len(changed)==4 else 'arithmetic_mismatch',members=changed),
            dict(status='metadata_and_source_process',reason='Date/version, program basename and digest prefixes use source_metadata_audit; operative rule is defined in manual map sections7/8.')]
    return None


def formation_band_prose(line,members,extras):
    document=extras['document']
    if not document.startswith(('SCALING','MASS')):return None
    names=[row['member'] for row in extras['source_audit']['rows'] if row['report']==document]
    old=extras['diagnostics']['historical']['old_report']['members']
    if line.startswith('- Formations that become certified:'):
        named=[m[1] for m in re.finditer(r'([A-Za-z0-9_]+) at ([-.\d]+)',line) if m[1] in members]
        changed=[n for n in names if members[n]['formation'] and members[n]['formation']['row']['certified'] and not old[n]['formation']['row']['certified']]
        lost=[n for n in names if members[n]['formation'] and not members[n]['formation']['row']['certified'] and old[n]['formation']['row']['certified']]
        values=[row_value(members[n],members[n]['formation']['row'],'T_over_rs') for n in named]
        if "S4_128_s4's at" in line:values.append(row_value(members['S4_128_s4'],members['S4_128_s4']['formation']['row'],'T_over_rs'))
        checks=[dict(target=named,own=changed,status='agrees' if set(named)==set(changed) else 'label_or_membership_mismatch'),
                dict(target='no formation loses certification',own=lost,status='agrees' if not lost else 'label_or_membership_mismatch')]
        if 'every formation is certified' in line:
            bad=[n for n in names if members[n]['formation'] and not members[n]['formation']['row']['certified']]
            checks.append(dict(target='every supplied document formation certified',own=bad,status='agrees' if not bad else 'label_or_membership_mismatch'))
        else:
            checks.append(dict(target='S4_128_s4 stays uncertified',own=members['S4_128_s4']['formation']['row']['certified'],status='agrees' if not members['S4_128_s4']['formation']['row']['certified'] else 'label_or_membership_mismatch'))
        return number_list(line.replace('30.3 only','kept-only'),values)+checks
    if not line.startswith('- Banded records inside the certified extents,'):return None
    pattern=r'([A-Za-z0-9_]+) ([-.\d]+) \(kept ([-.\deE]+), confirming ([-.\deE]+)\)'
    values=[];named=[]
    for match in re.finditer(pattern,line):
        name=match[1];member=members[name];row=next(r for r in member['_records'].values() if r['T_over_rs']==float(match[2]))
        named.append((name,row['source_index']));values.extend(row_value(member,row,k) for k in ['T_over_rs','kept','confirming'])
    actual=[(n,r['source_index']) for n in names for r in members[n]['_records'].values() if r['banded'] and r['certified']]
    outside=[(n,r['source_index']) for n in names for r in members[n]['_records'].values() if r['banded'] and not r['certified']]
    values.append(len(outside))
    # The last section6.1 locator is an identifier, not a measured decimal.
    text=line.replace('listed in 6.1','listed in the convergence table')
    formed_bands=[n for n in names if members[n]['formation'] and members[n]['formation']['row']['banded']]
    return number_list(text,values)+[
        dict(target=named,own=actual,status='agrees' if set(named)==set(actual) else 'label_or_membership_mismatch'),
        dict(target='none at first formation',own=formed_bands,status='agrees' if not formed_bands else 'label_or_membership_mismatch')]


def miscellaneous_prose(line,line_number,members,extras):
    scaling=extras['document'].startswith('SCALING');mass=extras['document'].startswith('MASS')
    if scaling and line_number==116:
        member=members['S4_128_s4'];stat=extras['diagnostics']['members']['S4_128_s4']['stats'];peak=stat['kept_maximum']
        values=[row_value(member,member['formation']['row'],'T_over_rs'),member['mask_summary']['records']-member['mask_summary']['counting'],exact(Fraction(1,1000)),exact(peak['value']),clock_value(member,peak['T_over_rs'])]
        return number_list(line,values)+[dict(status='required_scope_correction',claim='twenty-one fail either required residual; not twenty-one kept failures',kept_failures=len(stat['kept_above']),manual_reference='manual map sections3/8')]
    if scaling and line_number==88:
        s=lambda n,k:members[n]['scalars'][k]
        names=['S4_16_s4','RS32_s4','P64_s4','S4_128_s4'];ratio=lambda n,k:exact(Fraction(s(n,k))/Fraction(s(n,'rs')))
        values=[ratio(names[0],'tau'),exact(s(names[0],'rs')),exact(s(names[2],'rs')),ratio(names[1],'tau'),exact(s(names[1],'rs')),exact(s(names[3],'rs')),exact(s('P64_s4','sigma'))]
        values.extend(ratio(n,'sigma') for n in names)
        values.extend([exact(s('P64_s4','rs')),exact(s('RS32_s4','rs')),exact(s('RS32_s4','tau')),exact(s('RS32_s4_tau1','tau'))])
        fragment=line.split('; the (32, 4, 8)',1)[0]
        absent=[n for n,m in members.items() if m['scalars']['m']==.2 and m['scalars']['rs']==32 and m['scalars']['sigma']==4 and m['scalars']['tau']==8]
        ratios_agree=Fraction(s(names[0],'tau'))/Fraction(s(names[0],'rs'))==Fraction(s(names[2],'tau'))/Fraction(s(names[2],'rs')) and Fraction(s(names[1],'tau'))/Fraction(s(names[1],'rs'))==Fraction(s(names[3],'tau'))/Fraction(s(names[3],'rs'))
        return number_list(fragment,values)+[dict(status='agrees' if ratios_agree else 'label_or_membership_mismatch',claim='paired equal step ratios'),
            dict(target=[32,4,8],own_members=absent,status='required_scope_correction' if not absent else 'agrees',reason='Missing tuple must not stand for the actual size32 tau2/tau1 pair; manual map sections3/8.')]
    if scaling and line_number==466:
        a=extras['controls']['160']['catalogs']['RS32_s4']['head_comparisons'];b=extras['controls']['80']['catalogs']['RS32_s4']['head_comparisons'];m=members['RS32_s4']
        values=[exact(m['scalars']['rs']),m['_args']['kappa_sea']]+[precise(b[k]['relative'],a[k]['relative']) for k in ['N_B','E0','f_kin']]
        return number_list(line,values)+[dict(status='source_process',reason='Prepared/stored primitive arithmetic checked; execution of the named preparation is supplied attribution, not an independent rerun.')]
    if mass and line_number==128:
        member=members['M3d_s4'];rows=[r for r in member['_records'].values() if r['certified']]
        minimum=member['centroid_minimum'];outward=all(Fraction(b['R_c'])>Fraction(a['R_c']) for a,b in zip(rows,rows[1:]))
        return number_list(line,[clock_value(member,minimum['T_over_rs']),len(rows)])+[
            dict(claim='all consecutive centroid differences positive',own=outward,status='agrees' if outward else 'label_or_membership_mismatch'),
            dict(status='required_scope_correction',reason='Every other comparison is restricted to the MASS set with nonempty certified prefixes; manual map sections4/8.')]
    if mass and line_number==252:
        main=['M3b_s4','P64_s4','M3a_s4','M3c_s4_nk','M3d_s4'];formed=[n for n in main if members[n]['formation'] and members[n]['formation']['row']['certified']]
        return number_list(line,[clock_value(members[n],members[n]['mask_summary']['last']) for n in ['M3a_s4','M3d_s4']])+[
            dict(target=['M3b_s4','P64_s4','M3c_s4_nk'],own=formed,status='agrees' if formed==['M3b_s4','P64_s4','M3c_s4_nk'] else 'label_or_membership_mismatch')]
    if mass and line_number==274:
        fit=extras['descriptive']['160']['fits']['mass_isotropic']['ordinates']['T_f']['predictions'];results=[]
        for name in ['M3a_s4','M3d_s4']:
            member=members[name];t=Decimal(fit[name]);start=Decimal(member['mask_summary']['first']);end=Decimal(member['mask_summary']['last'])
            results.append(dict(member=name,fit_clock=str(t),first=str(start),last=str(end),
                relation='within saved clock range with no sampled root, below its censor' if start<=t<=end else 'beyond saved endpoint' if t>end else 'before saved interval',
                status='agrees' if member['formation'] is None and t>=start else 'label_or_membership_mismatch',
                scope='Placement of prescribed diagnostic prediction; no observed formation or law.'))
        results.append(dict(status='required_scope_correction',reason='State M3a prediction within saved no-sampled-root range below censor and M3d prediction beyond endpoint explicitly; ambiguous below-or-beyond-records wording is not a uniform range statement.'))
        return results
    return None


def end_state_prose(line,line_number,members,extras):
    scaling=extras['document'].startswith('SCALING');mass=extras['document'].startswith('MASS')
    if not (scaling or mass):return None
    s=lambda n,k:exact(members[n]['scalars'][k])
    row_at=lambda n,t:next(r for r in members[n]['_records'].values() if r['T_over_rs']==t)
    clock=lambda n,t:clock_value(members[n],t)
    end=lambda n:clock(n,members[n]['mask_summary']['last'])
    prefix=lambda n:clock(n,members[n]['mask_summary']['last_certified'])
    field=lambda n,r,k:row_value(members[n],r,k)
    checks=[];values=None
    if scaling and line_number==470:
        n='S4_16_s4';a='S4_16_s2';b='S4_16_s4_tau1'
        values=[s(n,'m'),s(n,'rs'),end(n),s(a,'sigma'),end(a),s(b,'tau'),end(b)]
        good=all(members[x]['formation'] is None for x in [n,a,b])
        checks.append(dict(claim='no sampled trapping root on all three histories',own=good,status='agrees' if good else 'label_or_membership_mismatch'))
    elif scaling and line_number==471:
        n='S4_128_s4';m=members[n];failure=m['mask_summary']['first_failure'];stat=extras['diagnostics']['members'][n]['stats']
        values=[s(n,'m'),s(n,'rs'),s(n,'sigma'),prefix(n),clock(n,failure['T_over_rs']),exact(Fraction(1,1000)),exact(failure['kept']),exact(failure['confirming']),
                exact(Fraction(1,1000)),clock(n,stat['kept_above'][0]),end(n),field(n,m['formation']['row'],'T_over_rs'),field(n,m['formation']['row'],'R_t')]
        for n in ['S4_128_s4_nk_v36m','S4_128_s4_nk_v36q']:
            f=members[n]['mask_summary']['first_failure'];values.extend([end(n),prefix(n),clock(n,f['T_over_rs'])])
            values.extend(exact(Fraction(1,1000)) for _ in f['reasons']);values.extend([exact(f['kept']),exact(f['confirming'])])
        checks.append(dict(status='required_scope_correction',reason='Two partial Newton outputs exist; no landed run must mean no completed requested-range Newton history; manual map sections3/8.'))
    elif scaling and line_number==472:
        n='P64_s2';text=line.split('; the plain rung',1)[0]
        values=[s(n,'sigma'),s(n,'rs')]
        result=number_list(text,values);good=members[n]['mask_summary']['last_certified']==members[n]['mask_summary']['last']
        return result+[dict(claim='certified prefix reaches last saved row',own=good,status='agrees' if good else 'label_or_membership_mismatch'),
                       dict(status='source_process',reason='Composite Newton/seam provenance is supplied; the alternative plain file is absent from the named collection, not a demonstrated exhaustive box-wide absence. Legacy rung/digest labels are locators.')]
    elif scaling and line_number==473:
        n='RS32_s2';row=row_at(n,0);values=[field(n,row,'T_over_rs'),field(n,row,'kept'),field(n,row,'confirming'),s(n,'rs')]
        text=line.replace('(6.3)','(control table)');good=row['banded'] and all(members[x]['mask_summary']['last_certified']==members[x]['mask_summary']['last'] for x in ['RS32_s2','RS32_s4'])
        return number_list(text,values)+[dict(claim='T0 band and full two-member certified prefixes',own=good,status='agrees' if good else 'label_or_membership_mismatch')]
    elif scaling and line_number==474:
        n='S4_R2_m0p2';f=members[n]['mask_summary']['first_failure'];values=[s(n,'m'),s(n,'rs'),exact(Fraction(members[n]['scalars']['R0'])/Fraction(members[n]['scalars']['rs'])),prefix(n),clock(n,f['T_over_rs']),exact(Fraction(1,1000)),exact(f['kept']),exact(f['confirming']),field(n,members[n]['formation']['row'],'T_over_rs')]
        checks.append(dict(claim='first formation certified',own=members[n]['formation']['row']['certified'],status='agrees' if members[n]['formation']['row']['certified'] else 'label_or_membership_mismatch'))
    elif scaling and line_number==475:
        n='S4_128_s8';match=re.search(r'T/r_s (\d+) slice',line)
        values=[clock(n,30),end(n)];text=match[0]+'; '+line.split('records to ',1)[1]
        return number_list(text,values)+[dict(status='source_process',reason='Raw physical clock30 exists and final endpoint is checked; queue/resume timestamp, machine and parent-resume attribution remain documentary metadata, not independently timed execution.')]
    elif mass and line_number==371:
        n='M3a_s4';a='M3a_s2';b='M3a_s4_tau4';first_kept=next(r for r in members[a]['_records'].values() if r['kept']>.001)
        main=['M3b_s4','P64_s4','M3a_s4'];formed=sum(members[x]['formation'] is not None and members[x]['formation']['row']['certified'] for x in main)
        values=[s(n,'m'),s(n,'rs'),end(n),s(a,'sigma'),exact(Fraction(1,1000)),field(a,first_kept,'T_over_rs'),s(b,'tau'),end(b),s(n,'rs'),formed]
        good=all(members[x]['formation'] is None for x in [n,a,b]) and not row_at(a,0)['counts']
        checks.extend([dict(claim='no sampled root and sigma2 first record fails',own=good,status='agrees' if good else 'label_or_membership_mismatch'),
                       dict(status='required_scope_correction',reason='At1.5 is the first kept residual failure; first either-test failure is zero; manual map sections4/8.')])
    elif mass and line_number==372:
        n='M3c_s4_nk';row=row_at(n,22);last=row_at(n,members[n]['mask_summary']['last']);formation=members[n]['formation']['row']
        values=[s(n,'m'),s(n,'rs'),prefix(n),field(n,row,'T_over_rs'),field(n,row,'kept'),field(n,row,'confirming'),end(n),field(n,last,'kept'),end(n),field(n,formation,'T_over_rs')]
        good=row['banded'] and row['counts'] and not last['counts'] and formation['certified']
        checks.append(dict(claim='banded local row, failed endpoint and certified first formation',own=good,status='agrees' if good else 'label_or_membership_mismatch'))
    elif mass and line_number==373:
        n='M3d_s4';rows=[r for r in members[n]['_residual_records'].values() if not r['converged']];kept=max(r['kept'] for r in rows);following=max(r['confirming'] for r in rows if r['confirming'] is not None)
        values=[s(n,'m'),s(n,'rs'),end(n),prefix(n),len(rows),exact(kept),exact(following),members[n]['_args']['kappa_sea'],s(n,'sites')]
        checks.extend([dict(target='1.69e-6',own=kept,strict_upper_bound_holds=Fraction(kept)<=Fraction('1.69e-6'),status='required_scope_correction' if Fraction(kept)>Fraction('1.69e-6') else 'agrees',reason='Rounded descriptive maximum needs approximately or an outward displayed bound.'),
                       dict(target='1.82e-5',own=following,status='agrees' if Fraction(following)<=Fraction('1.82e-5') else 'arithmetic_mismatch',comparison='following maximum <= printed bound'),
                       dict(status='input_unavailable_in_named_collection',reason='No supplied M3d catalog; no isotropic preparation or foreign preparation rerun is inferred.')])
    elif mass and line_number==375:
        n='P64_s4_tau2';f=members[n]['mask_summary']['first_failure'];a='P64_s4_tau4';row=row_at(a,13.4375)
        values=[s(n,'tau'),s(n,'m'),s(n,'rs'),prefix(n),clock(n,f['T_over_rs']),exact(Fraction(1,1000)),exact(f['kept']),exact(f['confirming']),s(a,'tau'),field(a,row,'T_over_rs'),field(a,row,'kept'),field(a,row,'confirming'),s(n,'tau')]
        good=row['banded'] and row['counts'] and members[a]['mask_summary']['last_certified']==members[a]['mask_summary']['last']
        checks.append(dict(claim='tau4 confirming-band row and complete certified prefix',own=good,status='agrees' if good else 'label_or_membership_mismatch'))
    if values is None:return None
    return number_list(line,values)+checks


def acceptance_prose(line,line_number,members,extras):
    scaling=extras['document'].startswith('SCALING');mass=extras['document'].startswith('MASS')
    if not ((scaling and line_number in [33,440]) or (mass and line_number in [25,348])):return None
    matches=list(re.finditer(r'([A-Za-z0-9_]+) against ([A-Za-z0-9_]+) \(([^)]*)\):',line))
    if not matches:raise ValueError('acceptance prose pair locators changed')
    controls=extras['controls']['160']['pairs'];values=[];checks=[]
    if scaling and line_number==33:values=[2,exact(members['RS32_s4']['scalars']['rs'])]
    elif scaling:values=[2,exact(Fraction(1,2)),10]
    elif line_number==25:values=[exact(members['P64_s2']['scalars']['sigma']),exact(members['P64_s2']['scalars']['rs'])]
    else:
        for n in ['M3b_s4','M3c_s4_nk','M3c_s4_nk','M3a_s4','P64_s4']:
            values.extend(exact(members[n]['scalars'][k]) for k in ['m','rs'])
        for n in ['P64_s4_tau4','P64_s4_tau2']:
            values.extend([clock_value(members[n],members[n]['mask_summary']['last_certified']),exact(members[n]['scalars']['tau'])])
        values.extend([exact(members['P64_s2']['scalars']['sigma']),clock_value(members['P64_s2'],members['P64_s2']['mask_summary']['last_certified'])])
        line=line.replace('(6.1)','(convergence table)')
        matches=list(re.finditer(r'([A-Za-z0-9_]+) against ([A-Za-z0-9_]+) \(([^)]*)\):',line))
    def fraction(value):return Fraction(int(value['rational'][0]),int(value['rational'][1]))
    for index,match in enumerate(matches):
        key=match[1]+'__'+match[2]
        if key not in controls:key=match[2]+'__'+match[1]
        block=controls[key];left,right=key.split('__');a,b=members[left],members[right]
        ar,br=a['formation']['row'],b['formation']['row']
        dt=abs(fraction(row_value(b,br,'T_over_rs'))-fraction(row_value(a,ar,'T_over_rs')))
        dr=abs(Fraction(br['R_t'])-Fraction(ar['R_t']))/Fraction(ar['R_t'])
        peak=block['primary']['maxima']['relative'];i,j=peak['indices'][0];pa,pb=a['_records'][i],b['_records'][j]
        if pa['T']!=pb['T']:raise ValueError('common-radius peak physical clocks differ')
        maximum=abs(Fraction(pb['R_t'])-Fraction(pa['R_t']))/Fraction(pa['R_t'])
        values.extend([exact(dt),exact(100*dr),block['primary']['both_zero_count'],exact(100*maximum),row_value(a,pa,'T_over_rs')])
        clause=line[match.end():matches[index+1].start() if index+1<len(matches) else len(line)]
        tail=clause.split(';',1)[1] if ';' in clause else ''
        pattern=r'(clock spread exceeds) ([-.\d]+) r_s|(radius spread at formation exceeds|radius difference over the common records exceeds|radius difference over the common records exceeding) ([-.\d]+) percent'
        for threshold in re.finditer(pattern,tail):
            values.append(exact(Fraction(1,2)) if threshold[1] else 10)
        classes=' / '.join('certified' if r['certified'] else 'uncertified' for r in [ar,br])
        checks.append(dict(target=match[3],own=classes,ordered_pair=key,status='agrees' if match[3]==classes else 'label_or_membership_mismatch',
                            formation_clock_pass=dt<=Fraction(1,2),formation_radius_pass=dr<=Fraction(1,10),common_radius_pass=maximum<=Fraction(1,10)))
        if 'within both criteria at formation' in tail:
            checks.append(dict(claim='formation criteria pass',own=[dt<=Fraction(1,2),dr<=Fraction(1,10)],status='agrees' if dt<=Fraction(1,2) and dr<=Fraction(1,10) else 'label_or_membership_mismatch'))
        if 'control fails' in tail:
            checks.append(dict(claim='declared full comparison fails',own=not(dt<=Fraction(1,2) and dr<=Fraction(1,10) and maximum<=Fraction(1,10)),status='agrees' if not(dt<=Fraction(1,2) and dr<=Fraction(1,10) and maximum<=Fraction(1,10)) else 'label_or_membership_mismatch'))
    if mass and line_number==348:
        checks.append(dict(status='input_unavailable_in_named_collection',reason='Named M3b/M3c step controls and M3c width control are not supplied; plain comparator is a separate solver comparison. M3a supplied controls have no sampled root; inventory in manual map sections4/8.'))
    return number_list(line,values)+checks


def mass_dependence_prose(line,line_number,members,extras):
    if not extras['document'].startswith('MASS'):return None
    s=lambda n,k:exact(members[n]['scalars'][k])
    f=lambda n,k:row_value(members[n],members[n]['formation']['row'],k)
    frac=lambda v:Fraction(int(v['rational'][0]),int(v['rational'][1]))
    if line_number==71:
        names=[r['member'] for r in extras['source_audit']['rows'] if r['report']==extras['document']]
        xs=[frac(members[n]['precision_scalars']['own_x']) for n in names]
        a=extras['controls']['160']['pairs']['P64_s4__P64_s4_c19'];b=extras['controls']['80']['pairs']['P64_s4__P64_s4_c19']
        maximum=a['primary']['maxima']['E'];old=b['primary']['maxima']['E']
        values=[2,5,2,exact(Fraction('0.3')),exact(min(xs)),exact(max(xs)),clock_value(members['P64_s4'],0),precise(old['value'],maximum['value']),a['primary']['count']]
        fixed=['M3a_s4','P64_s4','M3b_s4'];ordered=[frac(members[n]['precision_scalars']['own_x']) for n in fixed]
        monotone=all(y>=x for x,y in zip(ordered,ordered[1:])) or all(y<=x for x,y in zip(ordered,ordered[1:]))
        return number_list(line.replace('(6.3)','(control table)'),values)+[
            dict(claim='own coldness nonmonotone on fixed-size64 masses0.1,0.2,0.4',members=fixed,own=not monotone,status='agrees' if not monotone else 'label_or_membership_mismatch'),
            dict(target='1.2e-8',own=maximum['value'],strict_uniform_bound_holds=Decimal(maximum['value'])<=Decimal('1.2e-8'),status='required_scope_correction' if Decimal(maximum['value'])>Decimal('1.2e-8') else 'agrees',reason='Nearest rounded maximum agrees separately; say maximum difference approximately1.2e-8 or give an outward bound.'),
            dict(status='adopted_definition',reason='Squared moment/fifth and literal0.3 are stipulated comparator definitions; preparation/execution remains supplied provenance.')]
    if line_number==94:
        n='M3b_s8';member=members[n];row=member['formation']['row'];episode=member['episodes'][0]
        separation=Fraction(row['zeros'][-1])-Fraction(row['zeros'][0])
        values=[row_value(member,row,'F'),exact(episode['minimum']['value']),episode['count'],exact(separation),
                f('M3b_s4','R_t_over_2gE'),s('M3b_s4','m'),s('M3b_s4','rs'),f('M3b_s4','gate'),
                f('P64_s4','R_t_over_2gE'),s('P64_s4','m'),s('P64_s4','rs'),f('M3c_s4_nk','R_t_over_2gE'),s('M3c_s4_nk','m'),s('M3c_s4_nk','rs')]
        return number_list(line,values)+[dict(status='required_scope_correction',reason='Three first-root fractions do not identify matter-core versus vacuum-boundary branches without radial matter profiles; grazing is a descriptive shallow sampled episode, not a tangency proof.')]
    if line_number==256:
        a=members['M3b_s4'];b=members['M3b_s8'];ar=a['formation']['row'];br=b['formation']['row']
        dt=abs(frac(row_value(b,br,'T_over_rs'))-frac(row_value(a,ar,'T_over_rs')))
        relative=abs(Fraction(br['R_t'])-Fraction(ar['R_t']))/Fraction(ar['R_t'])
        book=extras['precision_bookkeeping'];clock=precise(book['comparators']['80']['production_nominal_clock'],book['comparators']['160']['production_nominal_clock']);w4=book['W4_primitives']
        fit=extras['descriptive']['160']['fits']['mass_isotropic'];sample=next(iter(fit['ordinates'].values()))
        values=[2,-8,3,4,3,len(fit['members']),len(sample['coefficients']),sample['dof'],1,5,clock,
                exact(Fraction(members['P64_s4']['scalars']['R0'])/Fraction(members['P64_s4']['scalars']['rs'])),exact(w4['fold']['T_over_rs']),exact(w4['first_record']['T_over_rs']),
                s('M3b_s4','m'),s('M3b_s4','rs'),exact(dt),exact(100*relative),s('M3b_s8','sigma')]
        return number_list(line,values)+[
            dict(status='conditional_analytic_definition',reason='Powers -8/3 and4/3 follow only at fixed declared g/release rule; slope1/5 is a clock comparator assumption, not every ordinate or a physical-law fit.'),
            dict(status='required_scope_correction',reason='W4 names its own W2 profile and charge; the primitive head differs from P64. Do not identify the supplied W4 comparator with the identical P64 prepared state.',W4_E0=w4['E0'],P64_E0=members['P64_s4']['scalars']['E0']),
            dict(claim='M3b width comparison outside formation criteria',own=[dt>Fraction(1,2),relative>Fraction(1,10)],status='agrees' if dt>Fraction(1,2) or relative>Fraction(1,10) else 'label_or_membership_mismatch'),
            dict(status='source_dependent_primitive',reason='W4 fold/first-record values are saved event primitives; this route does not independently integrate the W4 event.')]
    return None


def payload_clauses(line,line_number,members,extras):
    document=extras['document'];prefix='S' if document.startswith('SCALING') else 'M' if document.startswith('MASS') else None
    key=(prefix,line_number)
    if key not in {('S',7),('M',7),('S',12),('M',12),('M',24),('S',32),('S',34),('M',26),('S',511),('M',406)}:return None
    binding=extras['payload_attribution'];reference='payload_attribution_summary.json'
    if any(binding[k] for k in ['unmapped_bindings','unclassified_paths','stale_target_bindings','blocking_stage_bindings']):
        return [dict(status='pending',reason='Complete typed payload integration has unresolved mapping blockers.',own_source=reference)]
    common=dict(own_source=reference,manual_reference='LADDERS_v1p3_manual_occurrence_map_v1.md section8.7')
    if line_number==7:
        no_physical_change=not binding['changed_physical_record_scalar_nodes'] and all(row['node_kind']=='container' for row in binding['changed_raw_preparation_binding_nodes'])
        return [dict(status='adopted_definition',reason='Inspected full production rule, prefix ancestry and confirmer applicability are adopted definitions, not a residual-to-solution error theorem.',**common),
                dict(status='represented_input_identity' if no_physical_change else 'label_or_membership_mismatch',reason='Complete versioned bindings show unchanged represented physical record scalars and preparation leaves, separately from changed selection and fit membership.',**common),
                dict(status='required_scope_correction',reason='Every converged record must say every converged plain record; Newton retains its explicit confirming test. Underlying preparation/design/first-root inputs stay fixed while certified fit-row selection changes.',**common),
                dict(status='source_process',reason='Foreign producer execution/read-as-before attribution remains supplied-source process, not an own rerun.',**common)]
    if line_number==12:
        return [dict(status='bounded_payload_coverage',reason='All typed old/current nodes have exact stage pointers or explicit unresolved/input/format/process dispositions; parent shape checks never replace child bindings.',**common),
                dict(status='required_scope_correction',reason='Use main changes plus complete full-readout attribution, or explicitly include every corresponding episode/count/flag change. Even displayed episode counts have changes beyond the bullets; every other member value is unchanged is too broad. New confirmer/band bookkeeping, changed gate wording and process fields are additional export changes.',**common)]
    if key==('M',24):
        names=['mass_formed_certified','mass_diag_iso','mass_diag_own','mass_power3_iso']
        exact_identity=all(binding['named_unchanged_fit_subtrees'][name] for name in names)
        return [dict(status='agrees' if exact_identity else 'label_or_membership_mismatch',claim='All complete named mass fit trees are present and unchanged; their own arithmetic bindings remain separate.',members=names,**common),
                dict(status='adopted_definition',reason='Section5.1 through5.3 are documentary locators for these named mass diagnostics.',**common)]
    if key==('S',32):
        fit_identity=all(binding['named_unchanged_fit_subtrees'].values())
        design_identity=all(binding['old_labelled_five_current_primary_design_identity'].values())
        release_identity=all(check['present_in_both'] and check['exact_typed_identity'] for release in binding['release_preparation_and_firstroot_checks'].values() for check in release['checks'].values())
        membership=binding['S32_named_membership_changes']
        return [dict(status='agrees' if fit_identity and design_identity and release_identity and binding['removed_old_labelled_five_key'] else 'label_or_membership_mismatch',claim='Seven unchanged named fit trees, old-labelled/current-primary ordered five-row design and ordinate identity, removed duplicate key, and both release preparations/first-root primitives.',**common),
                dict(status='agrees' if membership['status']=='agrees' else 'label_or_membership_mismatch',claim='Exact printed five-member order; P64_s2 leaves the every-record uncertified list; width-two-to-four becomes both-certified; certified rank rows add exactly RS32_s2 and P64_s2.',own_membership=membership,**common),
                dict(status='required_scope_correction',reason='No release pair changes is supported only for preparation and physical first-root primitives. Release-control spreads and prefix/local sample populations change and remain separately bound.',**common)]
    if key in [('S',34),('M',26)]:
        diff=extras['payload_diff'];changes=diff['changed']
        bands=[row for row in changes if row['path'].endswith('/banded') and row.get('old')=='<absent>']
        checks=number_list(line.split('Every changed value',1)[1],[len(changes),len(bands)])
        checks.extend([dict(status='agrees' if diff['n_changed']==len(changes) and len({row['path'] for row in changes})==len(changes) else 'label_or_membership_mismatch',claim='Declared diff path count and duplicate-free path summary.',**common),
                       dict(status='required_scope_correction',reason='Diff is a path summary with string previews/list summaries, not every complete old/new value. Complete values are the separately bound old/current JSONs; typed-node counts are a different inventory.',**common),
                       dict(status='source_process',reason='Program/version attribution and earlier issued text history remain source process.',**common)])
        return checks
    represented=extras['historical_arrays'];metadata=extras['source_metadata']
    physical_ok=represented['summary']=={'exact_identity':405} and not represented['unavailable']
    raw_ok=all(all(row['matches'].values()) for row in extras['source_audit']['rows'])
    metadata_ok=all(row.get('exists') and row.get('printed_md5_agrees',True) for row in metadata['rows'])
    return [dict(status='represented_input_identity' if physical_ok and raw_ok and metadata_ok else 'label_or_membership_mismatch',reason='Verified source identities and exact fifteen-field old/current/own represented-array identity; separate full payload bindings do not imply equality of unexported profiles or parent histories.',**common),
            dict(status='metadata_and_source_process',reason='The named program versions, helper and rerun attribution are supplied process. Opaque source bytes were identified; no foreign implementation was decoded, imported or executed.',**common)]


def audit(report,members,extras):
    extras['document']=report.name
    entries=[]; section=''; subsection=''; headers=None; previous=None; table=0
    fit_counts={};fit_name=None
    lines=report.read_text().splitlines()
    for line_number,line in enumerate(lines,1):
        if line.startswith('## '):
            section=line[3:].split()[0]; subsection=''; headers=None
        if line.startswith('### '):
            subsection=line[4:].split()[0]; headers=None;fit_name=None
        if not line.startswith('|'):
            if line.strip():
                entry=dict(document=report.name,line=line_number,kind='heading' if line.startswith('#') else 'prose',
                           section=section,subsection=subsection,text=line,status='pending')
                calculations=None if (report.name.startswith('SCALING') and line_number==32) else payload_clauses(line,line_number,members,extras)
                if calculations is None:calculations=prose_calculations(line,members,extras)
                if calculations is None:calculations=fit_prose(line,line_number,members,extras)
                if calculations is None:calculations=member_prose(line,members)
                if calculations is None:calculations=historical_prose(line,members,extras)
                if calculations is None:calculations=control_change_prose(line,members,extras)
                if calculations is None:calculations=governing_prose(line,line_number,members,extras)
                if calculations is None:calculations=metadata_projection(line,line_number,members,extras)
                if calculations is None:calculations=integration_prose(line,line_number,members,extras)
                if calculations is None:calculations=certification_change_prose(line,line_number,members,extras)
                if calculations is None:calculations=formation_band_prose(line,members,extras)
                if calculations is None:calculations=miscellaneous_prose(line,line_number,members,extras)
                if calculations is None:calculations=end_state_prose(line,line_number,members,extras)
                if calculations is None:calculations=acceptance_prose(line,line_number,members,extras)
                if calculations is None:calculations=mass_dependence_prose(line,line_number,members,extras)
                if calculations is None:calculations=historical_width_binding(line,line_number,members,extras)
                if calculations is not None:
                    entry.pop('status');entry['calculations']=calculations
                entries.append(entry)
            previous=None
            continue
        protected=line.replace('|v|','\u241fv\u241f')
        cells=[cell.strip().replace('\u241f','|') for cell in protected.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?',cell) for cell in cells):
            table+=1; headers=previous; continue
        if headers is None:
            previous=cells
            entries.append(dict(document=report.name,line=line_number,kind='table_header',
                                section=section,subsection=subsection,text=line,status='pending'))
            continue
        if line_number<len(lines) and re.fullmatch(r'\|[\s|:\-]+\|',lines[line_number]):
            previous=cells
            entries.append(dict(document=report.name,line=line_number,kind='table_header',
                                section=section,subsection=subsection,text=line,status='pending'))
            continue
        if cells[0]=='T_f' and section=='5':
            ordinal=fit_counts.get(subsection,0);fit_counts[subsection]=ordinal+1
            if report.name.startswith('SCALING'):
                fit_name=({'5.2':['size_certified','size_labelled'],'5.3':['width_certified','width_labelled']}
                          .get(subsection,[None,None]))[ordinal] if ordinal<2 else None
            else:fit_name={'5.1':'mass_isotropic','5.2':'mass_own'}.get(subsection)
        for column,cell in enumerate(cells):
            entry=dict(document=report.name,line=line_number,section=section,subsection=subsection,
                       table=table,row=cells[0],column=column,text=cell,kind='table_cell')
            entry['calculation_key']=f'{section}/{subsection}/{cells[0]}/{column}'
            if cells[0] in members:
                entry['raw_source']=members[cells[0]]['source']
                entry['formation_source_index']=(members[cells[0]]['formation'] or {}).get('row',{}).get('source_index')
            extra=extra_cell(section,subsection,column,cell,cells,members.get(cells[0]),extras,fit_name)
            if report.name.startswith('LADDER_certification_list') and cells[0] in members:
                member=members[cells[0]];failure=member['mask_summary']['first_failure']
                if column==0:
                    entry['calculations']=[dict(target=cell,own=cells[0],status='agrees')]
                elif column==1:
                    val=extras['diagnostics']['members'][cells[0]]['intermediate_rule']['end']
                    entry['calculations']=number_list(cell,[clock_value(member,val)])
                elif column==2:
                    val=member['mask_summary']['last_certified']
                    entry['calculations']=([dict(target=cell,own=None,status='agrees' if cell=='none' else 'label_or_membership_mismatch')]
                                           if val is None else number_list(cell,[clock_value(member,val)]))
                elif column==3:
                    values=[clock_value(member,failure['T_over_rs']),exact(failure['kept']),exact(failure['confirming'])]
                    if 'formation' in cell:values.append(row_value(member,member['formation']['row'],'T_over_rs'))
                    entry['calculations']=number_list(cell,values)
                else:entry['calculations']=[dict(status='pending')]
            elif extra is not None:
                entry['calculations']=extra
            elif cells[0] in members and section in ['1','2','3']:
                entry['calculations']=scalar_cell(section,column,cell,members[cells[0]])
            elif subsection in ['4a','4b','4c','4d'] and column==0:
                bindings=[(n,m['matched'][str(int(cells[0]))]['row']) for n,m in members.items() if str(int(cells[0])) in m.get('matched',{}) and n in headers]
                entry['calculations']=number_list(cell,[exact(int(cells[0]))])
                entry['clock_row_bindings']=[dict(member=n,source_index=None if r is None else r['source_index']) for n,r in bindings]
            elif subsection in ['4a','4b','4c','4d'] and column>0 and column<len(headers) and headers[column] in members:
                member=members[headers[column]]
                row=member['matched'][str(int(cells[0]))]['row']
                key={'4a':'F','4b':'E_over_E0','4c':'R_c_over_rs','4d':'R_t_over_rs'}[subsection]
                if row is None:
                    entry['calculations']=[missing(cell)]
                elif row[key] is None:
                    entry['calculations']=[dict(target=cell,own=None,status='agrees' if cell.startswith('none') else 'label_or_membership_mismatch'),labels(cell,row)]
                else:
                    entry['calculations']=number_list(cell,[row_value(member,row,key)])+[labels(cell,row)]
            elif subsection=='4e' and cells[0] in members and 1<=column<=5:
                offsets=['0','0.5','1','2','4'];formation=members[cells[0]]['formation']
                row=None if formation is None else formation['offsets'][offsets[column-1]]['row']
                entry['calculations']=([missing(cell)] if row is None else
                    mixed(cell.replace('R_t/r_s none','R_t/r_s -'),
                          [row_value(members[cells[0]],row,key) for key in ['R_t_over_rs','R_t_over_2gE','E_over_E0','F','gate']])+[labels(cell,row)])
            else:
                entry['calculations']=[dict(status='pending',reason='binding not yet implemented')]
            entries.append(entry)
        previous=cells
    for entry in entries:manual_disposition(entry)
    return entries


def historical_width_binding(line,line_number,members,extras):
    if extras['document'].startswith('ReviewA_cert_rule') and line_number in [8,10,14,16]:
        diagnostics=extras['diagnostics']['members']
        rows=lambda n:list(members[n]['_residual_records'].values())
        clock=lambda n,t:clock_value(members[n],t)
        s=lambda n,k:exact(members[n]['scalars'][k])
        if line_number==16:
            values=[s('P64_s4_tau4','tau'),clock('P64_s4_tau4',members['P64_s4_tau4']['mask_summary']['last_certified']),s('P64_s4_tau2','tau'),clock('P64_s4_tau2',members['P64_s4_tau2']['mask_summary']['last_certified']),s('P64_s2','sigma'),clock('P64_s2',members['P64_s2']['mask_summary']['last_certified']),s('P64_s2','sigma'),clock('P64_s2',16),s('RS32_s2','rs'),s('RS32_s2','sigma'),2]
            return number_list(line,values)+[dict(status='required_scope_correction',reason='The old12.78 endpoint is superseded by12.75; the size32 failure concerns the declared control criteria, and the quoted seam/todo closure is documentary, not a mass-member result.')]
        formations=[]
        for name,m in members.items():
            if m['formation'] is None:continue
            i=m['formation']['row']['source_index'];r=m['_residual_records'][i]
            formations.append(dict(member=name,source_index=i,T_over_rs=r['T_over_rs'],contract=r['contract'],converged=r['converged'],kept=r['kept'],confirming=r['confirming'],banded=r['banded']))
        if line_number==14:
            good=not any(r['banded'] for r in formations)
            return [dict(status='adopted_definition',reason='10^-3 admission/prefix and10^-4-to10^-3 band constants are stipulated, not measured quantities.'),dict(status='agrees' if good else 'label_or_membership_mismatch',claim='No actual first-root record is banded',formations=formations),dict(status='conditional_analytic_definition',reason='2rs epsilon requires actual field-error and regular exterior slope; tenth-spacing is an approximate size-specific example, not a residual-to-error proof.')]
        if line_number==10:
            bands={}
            for name in ['S4_R2_m0p2','S4_R2_m0p1','S4_128_s4','M3a_s2']:
                bands[name]=[r['T_over_rs'] for r,prefix in zip(rows(name),diagnostics[name]['intermediate_rule']['prefix']) if prefix and r['contract'].startswith('plain') and r['kept']>.0001 and r['kept']<=.001]
            expected={'S4_R2_m0p2':[6.125,6.5],'S4_R2_m0p1':None,'S4_128_s4':None,'M3a_s2':None}
            band_good=bands['S4_R2_m0p2']==expected['S4_R2_m0p2'] and len(bands['S4_R2_m0p1'])==2 and len(bands['S4_128_s4'])==2 and len(bands['M3a_s2'])==8
            boundary_check=dict(status='agrees' if all(t>32 for t in bands['S4_R2_m0p1']) else 'required_scope_correction',
                claim='Two intermediate S4_R2_m0p1 band clocks strictly past32',clocks=bands['S4_R2_m0p1'],
                reason='Count agrees separately; actual clocks32 and32.125 require at or after32, or their exact clocks.')
            formation_good=all((r['confirming'] is not None and r['confirming']<.0001) if r['contract'].startswith('Newton') else (r['converged'] and r['kept']<.000001) for r in formations)
            counterexamples=[]
            for name,t in [('P64_s4_tau2',12.78125),('M3a_s2',0)]:
                r=next(r for r in rows(name) if r['T_over_rs']==t)
                counterexamples.append(dict(member=name,source_index=r['source_index'],T_over_rs=t,kept=r['kept'],confirming=r['confirming'],passed=r['kept']<.0001 and r['confirming'] is not None and r['confirming']>.001))
            return [boundary_check,dict(status='definition_unresolved',reason='Six quoted Newton pairs and approximate factor4 have no supplied pair locators; no pair sample is selected after inspection.'),dict(status='agrees' if formation_good else 'label_or_membership_mismatch',claim='Complete first-root set meets the named solver-specific strict local conditions',formations=formations),dict(status='agrees' if band_good else 'label_or_membership_mismatch',claim='Named intermediate plain-kept band counts and clocks',bands=bands),dict(status='required_scope_correction',reason='Only a banded plain row can shorten is false; operative test includes every unconverged plain row.',counterexamples=counterexamples),dict(status='adopted_definition',reason='Following/integration thresholds1e-3 and flagged-row bound1e-4 are specified; two bookkeeping items is a heading count.'),dict(status='source_process',reason='Quoted seam16, solver-detail printing and missing-following-entry handling retain the separate documentary/declared-rule scope.')]+[dict(status='agrees' if diagnostics[n]['residual_invN']['above_summary']['first']==t and all(members[n]['_residual_records'][r['source_index']]['kept']<.0001 and members[n]['_residual_records'][r['source_index']]['confirming'] is not None and members[n]['_residual_records'][r['source_index']]['confirming']<.0001 for r in diagnostics[n]['residual_invN']['above']) else 'label_or_membership_mismatch',member=n,claim='First integration residual above1e-3 clock and kept/applicable confirming below1e-4 on every flagged row',first_clock=clock(n,diagnostics[n]['residual_invN']['above_summary']['first'])) for n,t in [('P64_s2',.5),('RS32_s2',.125)]]
        # F8: bind measured primitives and retain stipulated/control premises separately.
        part_a=line.split('(a) ',1)[1].split('(b) ',1)[0]
        part_a=re.sub(r'(?<=\d)a\b',' a',part_a)
        part_b=line.split('(b) ',1)[1].split('(c) ',1)[0]
        part_c=line.split('(c) ',1)[1]
        false=[r for r in rows('M3d_s4') if not r['converged']]
        first=next(r for r in rows('P64_s2') if r['T_over_rs']==0)
        cadence=exact(Fraction(members['P64_s4']['scalars']['tau'])/Fraction(members['P64_s4']['scalars']['rs']))
        values=[cadence,exact(Fraction(1,2)),10,2,exact(Fraction(128,1000)),exact(Fraction(1,1000)),s('P64_s4','rs'),None,exact(Fraction(1,1000000)),exact(first['confirming']),clock('P64_s2',0),exact(max(r['kept'] for r in false)),s('P64_s2','sigma'),len(rows('P64_s2')),len(rows('RS32_s2'))]
        checks=number_list(part_a,values)+number_list(part_b,[exact(Fraction(1,1000000)),4,len(members)])
        kept=[n for n in members if diagnostics[n]['stats']['kept_above']]
        newton=[n for n in members if any(r['contract'].startswith('Newton') for r in rows(n))]
        a=members['RS32_s4'];b=members['RS32_s2'];ar=a['formation']['row'];br=b['formation']['row']
        dt=abs(Fraction(br['T'])/Fraction(b['scalars']['rs'])-Fraction(ar['T'])/Fraction(a['scalars']['rs']))
        pair=extras['controls']['160']['pairs']['RS32_s4__RS32_s2'];peak=pair['primary']['maxima']['relative']
        peak80=extras['controls']['80']['pairs']['RS32_s4__RS32_s2']['primary']['maxima']['relative']
        percentage=precise(str(Decimal(peak80['value'])*100),str(Decimal(peak['value'])*100))
        checks.extend(number_list(part_c,[s('RS32_s2','rs'),s('RS32_s2','sigma'),exact(dt),percentage,s('P64_s2','sigma'),s('P64_s4','sigma')]))
        checks.extend([dict(status='agrees' if len(kept)==7 and len(newton)==5 and len(false)==5 else 'label_or_membership_mismatch',claim='Seven kept-crossing histories, five Newton histories and five M3d stored-false rows',kept_crossing_members=kept,newton_members=newton,false_indices=[r['source_index'] for r in false]),dict(status='agrees' if all(diagnostics[n]['old_report']['end'] is None for n in ['P64_s2','RS32_s2']) else 'label_or_membership_mismatch',claim='Old strict prefixes empty for both named full histories'),dict(status='conditional_analytic_definition',reason='2rs epsilon and.128-to.13 arithmetic are conditional on actual field error/regular slope; proposed.002 clock shift lacks its depth-rate premise.'),dict(status='required_scope_correction',reason='P64 width2/4 passes at formation only; its common-clock radius criterion fails.'),dict(status='source_process',reason='Three-lane agreement is attribution; stop target and criteria are declared, not field-error bounds.')])
        return checks
    scaling=extras['document'].startswith('SCALING');mass=extras['document'].startswith('MASS')
    if (scaling and line_number==60) or (mass and line_number==48):
        formations=[]
        for name,m in members.items():
            if m['formation'] is None:continue
            r=m['formation']['row'];good=len(r['zeros'])==2 and not r['tangencies'] and r['outer_slope'] is not None and r['outer_slope']>0
            formations.append(dict(member=name,source_index=r['source_index'],zeros=r['zeros'],tangencies=r['tangencies'],outer_slope=r['outer_slope'],passed=good))
        prep={n:m['_args'].get('prep_ell') for n,m in members.items()}
        known={n:v for n,v in prep.items() if v is not None}
        return [dict(status='adopted_definition',reason='Thresholds, band endpoints, kept-only/confirmed prefixes, normalization constants/powers and root/ratio definitions are fixed by manual map8.4; formula/identifier digits are not measured values.'),
                dict(status='agrees' if all(r['passed'] for r in formations) else 'label_or_membership_mismatch',claim='Every actual first-root record has two zeros, no saved tangencies and positive outer slope',formations=formations),
                dict(status='agrees' if all(m['scalars']['g']==.25 for m in members.values()) and all(v==4 for v in known.values()) else 'label_or_membership_mismatch',claim='Raw coupling1/4 and available preparation widths4',available_preparation=known),
                dict(status='source_dependent_primitive',reason='Omitted preparation arguments stay absent; record-zero head ratios retain each own raw normalization.',missing_preparation=[n for n,v in prep.items() if v is None],head_ratios={n:row_value(m,m['_records'][0],'E_over_E0') for n,m in members.items()}),
                dict(status='required_scope_correction',reason='Residual-to-root error requires an actual field perturbation and regular slope; tenth-spacing is not uniform over sizes. Only primary certified fits exclude uncertified formations.')]
    if (scaling and line_number==64) or (mass and line_number==52):
        release={n:exact(Fraction(m['scalars']['R0'])/Fraction(m['scalars']['rs'])) for n,m in members.items()}
        radius_two=[n for n,m in members.items() if Fraction(m['scalars']['R0'])/Fraction(m['scalars']['rs'])==2]
        catalog=extras['controls']['160']['catalogs']['P64_s4']
        good=all(m['scalars']['NB_preparation_identity_difference']==0 and m['scalars']['g']==.25 for m in members.values())
        return [dict(status='adopted_definition',reason='Sea lower endpoint1, count formula, release/preparation/clock multipliers and literal isotropic/exclusion coefficients and cube-root powers are stipulated definitions in manual map8.4; their table formula arithmetic is separately bound.'),
                dict(status='agrees' if good and len(radius_two)==2 else 'label_or_membership_mismatch',claim='All member count identities and coupling; exactly two radius-two release members',radius_two_members=radius_two,release_factors=release,kappa_sea={n:m['_args'].get('kappa_sea') for n,m in members.items()}),
                numeric('20',int(catalog['onset'])),dict(status='source_dependent_primitive',claim='Production onset is four times catalog maximum columns',max_columns=catalog['max_columns'],onset=catalog['onset'],source=catalog['source_path']),
                dict(status='input_unavailable',reason='The separate T0 valley census supporting12-to20 is absent; unsplit column arithmetic cannot validate it.'),
                dict(status='source_process',reason='Filling/projection/coordinate-rest preparation are documentary declarations; four absent old preparation arguments are not filled by defaults.')]
    if extras['document'].startswith('LADDER_certification_list') and line_number==21:
        member=members['P64_s2'];rows=list(member['_residual_records'].values())
        text=line.split('whose ',1)[1].split('; every record converged.',1)[0]
        text=re.sub(r'T00(?:16|17)\.00','named_checkpoint',text)
        values=[len(rows)]+[clock_value(member,t) for t in [0,16,0,16.125,17,17.125,40]]
        flags=[r['converged'] for r in rows]
        checks=number_list(text,values)
        checks.append(dict(claim='every saved convergence flag true',own=all(flags),status='agrees' if all(flags) else 'label_or_membership_mismatch'))
        checks.append(dict(status='input_unavailable',reason='Three parent files are absent from named inputs; saved composite clocks and raw engine labels do not prove record-by-record parent identity.',engines=extras['diagnostics']['members']['P64_s2']['engines']))
        checks.append(dict(status='metadata_and_source_process',reason='Named composite/hash identity is bound by source_metadata_audit; three-run attribution, filename checkpoint digits and quoted seam-removal commit history retain the accepted documentary disposition.'))
        return checks
    if extras['document'].startswith('SCALING') and line_number==329:
        names=extras['descriptive']['160']['fits']['width_labelled']['members']
        certified=extras['descriptive']['160']['fits']['width_certified']['members']
        s=lambda n,k:exact(members[n]['scalars'][k])
        pair=lambda n:[s(n,'rs'),s(n,'sigma')]
        values=[len(names),s('P64_s4','rs')]+[s(n,'sigma') for n in ['P64_s2','P64_s4','P64_s8']]
        values.extend([s('S4_128_s4','rs')]+[s(n,'sigma') for n in ['S4_128_s4','S4_128_s8','S4_128_s16']])
        for n in certified:values.extend(pair(n))
        n='S4_128_s4';m=members[n];values.extend(pair(n));values.extend([row_value(m,m['formation']['row'],'T_over_rs'),clock_value(m,m['mask_summary']['last_certified']),exact(Fraction(1,1000))])
        failure=next(r for r in m['_residual_records'].values() if r['kept']>.001)
        values.append(clock_value(m,failure['T_over_rs']))
        for n in ['P64_s4','S4_128_s4']:
            values.extend([exact(Fraction(members[n]['scalars']['tau'])/Fraction(members[n]['scalars']['rs'])),s(n,'rs')])
        values.append(s('P64_s4','rs'));a=members['P64_s4'];b=members['P64_s4_tau4']
        for key in ['T_over_rs','R_t_over_rs']:
            av=row_value(a,a['formation']['row'],key);bv=row_value(b,b['formation']['row'],key)
            values.append(exact(Fraction(int(bv['rational'][0]),int(bv['rational'][1]))-Fraction(int(av['rational'][0]),int(av['rational'][1]))))
        values.append(len({a['scalars']['rs'],b['scalars']['rs']}));values.extend(pair('S4_128_s8'));values.append(clock_value(members['S4_128_s8'],members['S4_128_s8']['mask_summary']['last']))
        return number_list(line,values)+[dict(status='required_scope_correction',reason='Per-halving changes belong to the named P64 tau8-to4 pair at one tested size; no general step law follows.'),dict(status='source_process',reason='Finished/final producer attribution is source provenance; saved endpoint is checked independently.')]
    if extras['document'].startswith('SCALING') and line_number==374:
        historical=extras['precision_bookkeeping']['historical_null']['160']
        current=extras['descriptive']['160']['rank_tables']['certified']
        text=line.replace('(z - 1)(2/3 - 7h/24 + j)','(declared polynomial null)')
        checks=number_list(text,[historical['rank'],len(historical['triples']),current['rank'],len(current['singular_values']),current['rows'],current['rows_minus_rank']])
        checks.append(dict(status='adopted_definition',reason='Displayed polynomial coefficients belong to the independently proved hypothetical null identity in the accepted manual map; this is distinct from the actual old eight-row full-rank set.'))
        return checks
    if not extras['document'].startswith('SCALING') or line_number!=32:return None
    old=extras['historical_width']['precisions'];new=extras['descriptive']
    values=[len(extras['historical_width']['members']),len(new['160']['fits']['width_certified']['members']),0,
            new['160']['fits']['width_certified']['ordinates']['T_f']['dof']]
    values.extend(precise(a,b) for a,b in zip(old['80']['T_f']['coefficients'],old['160']['T_f']['coefficients']))
    for key in ['coefficients','residuals']:
        values.extend(precise(a,b) for a,b in zip(new['80']['fits']['width_certified']['ordinates']['T_f'][key],new['160']['fits']['width_certified']['ordinates']['T_f'][key]))
    rank=new['160']['rank_tables']['certified']
    values.extend([2,2,4,exact(members['P64_s4']['scalars']['rs']),8,rank['rows'],8,rank['rank'],0,rank['rows_minus_rank']])
    text=line.replace('(5.3)','(width section)').replace('(5.1, 5.2)','(size sections)').replace('(5.4)','(release section)').replace('version 1.2','previous version')
    checks=number_list(text,values)
    if len(checks)==len(values):checks[11].update(status='roundoff_zero',reason='Accepted independent exact five-row width-clock identity has c3=0; supplied tiny nonzero retained as roundoff.')
    checks.extend(payload_clauses(line,line_number,members,extras))
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args(); out=Path(args.output)
    fixtures=[('positive below rounding boundary','1.00',1.0049,'agrees'),
              ('positive above rounding boundary','1.00',1.0051,'arithmetic_mismatch'),
              ('negative below magnitude boundary','-1.00',-1.0049,'agrees'),
              ('negative above magnitude boundary','-1.00',-1.0051,'arithmetic_mismatch'),
              ('scientific notation','6.4e-4',.000637,'agrees'),
              ('integer count exact','1',1,'agrees'),
              ('integer count mismatch','1',2,'arithmetic_mismatch'),
              ('exact positive tie to even down','1.00',exact(Fraction('1.005')),'agrees'),
              ('exact positive tie to even up','1.02',exact(Fraction('1.015')),'agrees'),
              ('exact negative tie to even','-1.00',exact(Fraction('-1.005')),'agrees'),
              ('odd tie rejected','1.01',exact(Fraction('1.005')),'arithmetic_mismatch'),
              ('signed zero','-0.0000',exact(0),'agrees'),
              ('percentage already scaled','50.0',exact(Fraction(1,2)*100),'agrees'),
              ('unavailable value','1.0',None,'input_unavailable')]
    fixtures.append(('uncertainty touches rounding endpoint','1.0',dict(value='1.04',error='0.01'),'definition_unresolved'))
    controls=[]
    for name,target,own,expected in fixtures:
        actual=numeric(target,own)
        controls.append(dict(name=name,actual=actual,expected_status=expected,passed=actual['status']==expected))
    controls.append(dict(name='identifiers and field constants excluded',
                         passed=NUMBER.findall('P64_s4 R_t/(2gE) 0.125')==['0.125']))
    controls.append(dict(name='sentence-ending decimal punctuation retained',
                         passed=NUMBER.findall('clock 32.125. next 40. end')==['32.125','40.']))
    controls.append(dict(name='missing scalar differs from formula subtraction',
                         passed=[item['status'] for item in mixed('1.0; -',[exact(1),None])]==['agrees','agrees']))
    if not all(item['passed'] for item in controls):
        raise RuntimeError('rounding controls failed')
    (out/'target_known_answers.json').write_text(json.dumps(controls,indent=2)+'\n')
    if args.controls_only:
        print(json.dumps(dict(controls=len(controls),passed=True)))
        return
    members=json.loads((out/'scalar_tables.json').read_text())
    precision_checks={}
    for name,member in members.items():
        member['_records']={row['source_index']:row for row in json.loads(Path(member['record_table']).read_text())}
        member['precision_scalars'],precision_checks[name]=scalar_precision(member)
    (out/'scalar_precision_checks.json').write_text(json.dumps(precision_checks,indent=2)+'\n')
    extras=dict(controls=json.loads((out/'controls_catalogs.json').read_text()),
                descriptive=json.loads((out/'descriptive_arithmetic.json').read_text()),
                diagnostics=json.loads((out/'record_diagnostics.json').read_text()),
                source_audit=json.loads((out/'source_audit.json').read_text()),
                precision_bookkeeping=json.loads((out/'precision_bookkeeping.json').read_text()),
                historical_width=json.loads((out/'historical_width_direct_control.json').read_text()))
    extras.update(payload_attribution=json.loads((out/'payload_attribution_summary.json').read_text()),
                  payload_diff=json.loads((Path(args.reports[0]).parent/'ladders_diff_v1_3.json').read_text()),
                  historical_arrays=json.loads((out/'historical_physical_arrays.json').read_text()),
                  source_metadata=json.loads((out/'source_metadata_audit.json').read_text()))
    residuals=json.loads((out/'residual_series.json').read_text())
    for name,member in members.items():member['_residual_records']={row['source_index']:row for row in residuals[name]['rows']}
    extras['members']=members
    for name,member in members.items():
        member['_args']=extras['diagnostics']['members'][name]['args']
    entries=[]
    for name in args.reports:
        entries.extend(audit(Path(name),members,extras))
    (out/'target_coverage.json').write_text(json.dumps(entries,indent=2,allow_nan=False)+'\n')
    counts={}
    for entry in entries:
        for item in entry.get('calculations',[entry]):
            status=item['status']; counts[status]=counts.get(status,0)+1
    (out/'target_coverage_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':
    main()
