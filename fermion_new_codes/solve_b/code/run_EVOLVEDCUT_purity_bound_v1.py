"""Outward evaluation of the frozen saved-complement operation-model bound."""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from cpu_runtime import require_remote_cpu
from decimal_interval_v1 import Interval as I, binary_entropy_increasing, positive, PRECISION

MATCHES={12:(0,58),15:(1,51),20:(1,38),28:(1,37),35:(1,37),40:(1,36)}
MISSING_HYPOTHESES=[
    'Original binary64 BLAS/dot/norm and scaling errors satisfy the frozen operation inequalities, without unbounded overflow or underflow.',
    'Original sqrt and log/log1p errors satisfy the stated elementary/correct-rounding model.',
    'Original radial-operator formation is covered by the fixed 32-rounding per-entry model.',
    'The first-route 1.85e-7 allowance is supplied, not independently certified by this lane.'
]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if isinstance(value,I):return value.as_dict()
    if isinstance(value,Decimal):return str(value)
    if isinstance(value,Fraction):return {'numerator':value.numerator,'denominator':value.denominator,'interval':I(value).as_dict()}
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value


def write(path,value):Path(path).write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')


def exact_binary_inputs(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('nonfinite saved scalar')
        n,d=value.as_integer_ratio();return {'numerator':n,'denominator':d}
    if isinstance(value,dict):return {k:exact_binary_inputs(v) for k,v in value.items()}
    if isinstance(value,list):return [exact_binary_inputs(v) for v in value]
    return value


class Model:
    def __init__(self,u=Fraction(1,2**53)):self.u=I(u);self.s=self.gamma(32)
    def gamma(self,j):
        ju=j*self.u;return ju/positive(1-ju,'gamma denominator')
    def g(self,k):return self.gamma(8*k+16)
    def c(self,entries):return 1/(positive(1-self.gamma(8*entries+32),'norm denominator')*positive(1-self.u,'subtraction denominator'))
    def defect(self,n,m,measured):
        delta=(self.c(m*m)*I(measured)+self.g(n)*m)/positive(1-self.g(n)*I(m).sqrt(),'Gram closure denominator')
        if delta.lo<0 or delta.hi>=1:raise ValueError('enclosed Gram defect not below one')
        return delta
    def singular(self,delta,m):return ((1+delta).sqrt(),positive(1-delta,'singular-value lower square').sqrt(),(m+I(m).sqrt()*delta).sqrt())
    def small(self,d,a2,af,f,entropy,check):
        dv=self.defect(d,d,check['eigenvector_orthogonality_frobenius'])
        xi=I(check['eigvalsh_eigh_max_difference'])/positive(1-self.u,'spectrum-difference denominator')
        ell=I(max(abs(entropy['minimum']),abs(entropy['maximum'])))+xi
        vf=(d+I(d).sqrt()*dv).sqrt()
        r=self.c(d*d)*I(check['eigenpair_frobenius_residual'])+self.g(d)*af*vf+self.s*ell*vf
        tv=dv/(1+positive(1-dv,'small eigenbasis lower square').sqrt())
        beta=r+(a2+ell)*tv
        argument=((f+beta)/I(d).sqrt()+xi).cap('.5')
        zeta=self.gamma(d+8)*d*I(2).ln()
        upper=d*binary_entropy_increasing(argument)+zeta
        return {'delta_V':dv,'xi':xi,'ell':ell,'r':r,'t_V':tv,'beta_F':beta,'modulus_argument':argument,
                'zeta_d':zeta,'conditional_model_upper':upper,'certified_from_saved_record_upper':None}
    def channel(self,row):
        n,p,d=row['source_row_dimension'],row['polar_column_count'],row['dimension'];orth=row['orthogonality']
        dt=self.defect(n,p,orth['joint_defect_frobenius'])
        dg=self.defect(p,p,orth['joint_eigenvector_orthogonality_frobenius'])
        dq=self.defect(n,p,orth['orthonormal_control_frobenius'])
        tf=(p+I(p).sqrt()*dt).sqrt();v2,_,vf=self.singular(dg,p)
        emin=positive(I(orth['joint_eigenvalue_min']),'source Gram eigenvalue minimum')
        dp=1/(positive(1-self.u,'sqrt denominator')*emin.sqrt())
        e1=self.g(p)*tf*vf;bf=tf*v2+e1
        e2=self.s*bf*dp;cf=bf*dp+e2;e3=self.g(p)*cf*vf
        eq=e1*dp*v2+e2*v2+e3
        phi_c=self.g(p)*d*(1+dq);a2_c=1+dq+phi_c;af_c=I(d).sqrt()*(1+dq)+phi_c
        range_c=I(2).sqrt()*eq/positive(1-dq,'Q lower square').sqrt()
        f_c=phi_c+dq+range_c
        state={'delta_T':dt,'delta_G':dg,'delta_Q':dq,'d_plus':dp,'e1':e1,'e2':e2,'e3':e3,'E_Q':eq,
               'phi_C':phi_c,'a_C_2':a2_c,'a_C_F':af_c,'source_range_bridge':range_c,'f_C':f_c}
        h=row['held_operator'];hm=row['held_spectrum_metadata'];ph=h['negative_count']
        dh=self.defect(n,n,h['eigenvector_orthogonality_frobenius']);v2,vm,vf=self.singular(dh,n)
        ell=I(hm['maximum_absolute_eigenvalue']);gap=positive(I(hm['minimum_absolute_eigenvalue']),'held gap')
        r0=self.c(n*n)*I(h['eigenpair_frobenius_residual'])
        bh=(v2*ell+r0+self.s*ell*vf)/positive(vm-self.g(6)*I(6).sqrt()*vf,'held residual closure denominator')
        rh=r0+self.g(6)*I(6).sqrt()*bh*vf+self.s*ell*vf
        chi=self.s/positive(1-self.s,'operator formation denominator')*I(n).sqrt()*bh
        rhstar=rh+chi*v2;eh=rhstar/vm
        if eh.hi>=gap.lo:raise ValueError('held gap closure failed')
        range_h=I(2).sqrt()*eh/positive(2*gap-eh,'held projector separation')
        phi_h=self.g(ph)*d*(1+dh);a2_h=1+dh+phi_h;af_h=I(d).sqrt()*(1+dh)+phi_h
        f_h=phi_h+dh+range_h
        held={'delta_H':dh,'ell_H':ell,'g_H':gap,'r0':r0,'b_H':bh,'r_H':rh,'chi_H':chi,'r_H_star':rhstar,
              'e_H':eh,'held_projector_bridge':range_h,'phi_H':phi_h,'a_H_2':a2_h,'a_H_F':af_h,'f_H':f_h}
        shift=row['site_shift'];checks=row['small_block_checks']
        state_small=self.small(d,a2_c,af_c,f_c,row['polar_interior'],checks[f'orthonormal_control_interior_shift{shift}'])
        held_small=self.small(d,a2_h,af_h,f_h,row['held'],checks[f'held_shift{shift}'])
        total=state_small['conditional_model_upper']+held_small['conditional_model_upper']
        complement=Fraction.from_float(row['polar_physical_interior_entropy']);held_reading=Fraction.from_float(row['held']['entropy'])
        return {'state_formation':state,'held_formation':held,'state_small':state_small,'held_small':held_small,
                'saved_complement_reading_exact':complement,'saved_held_reading_exact':held_reading,
                'held_subtracted_reading_exact':complement-held_reading,
                'conditional_model_upper_interval':total,'conditional_model_upper':str(total.hi),
                'weighted_conditional_model_upper_interval':row['weight']*total,
                'weighted_conditional_model_upper':str((row['weight']*total).hi),
                'certified_from_saved_record_upper':None,'availability':'available_under_stated_model'}


def validate_rows(saved):
    rows=saved['rows']
    if digest(saved['source'])!=saved['source_sha256'] or digest(saved['first_table_source'])!=saved['first_table_sha256']:raise ValueError('frozen original source hash changed')
    if len(rows)!=60 or len({(r['clock_rs'],r['kappa']) for r in rows})!=60:raise ValueError('expected sixty unique matched records')
    for clock,(shift,sites) in MATCHES.items():
        group=[r for r in rows if r['clock_rs']==clock]
        if sorted(r['kappa'] for r in group)!=list(range(1,11)) or sum(r['weight'] for r in group)!=880:raise ValueError('channel/multiplicity contract')
        for r in group:
            if r['site_shift']!=shift or r['interior_sites']!=sites or r['dimension']!=2*sites or r['weight']!=16*r['kappa']:raise ValueError('matched partition contract')
            if r['source_row_dimension']!=4096 or r['polar_column_count']!=2048+r['dust_rank'] or r['polar_exterior']['dimension']!=r['polar_column_count']:raise ValueError('source rank contract')
            if not r['binary_occupations_confirmed'] or r['ideal_polar_global_entropy']!=0:raise ValueError('ideal purity contract')
            if not r['orthogonality']['polar_accepted'] or not r['polar_interior']['valid'] or not r['held']['valid']:raise ValueError('saved gate contract')
            if r['held_operator']['zero_sign_unresolved'] or not r['held_operator']['eigenvector_isometry_accepted']:raise ValueError('held basis gate')
            hm=r['held_spectrum_metadata'];h=r['held_operator']
            if digest(hm['source'])!=hm['sha256']:raise ValueError('saved held spectrum changed')
            if hm['dimension']!=4096 or hm['dtype']!='float64' or hm['negative_count']!=2048 or h['negative_count']!=2048 or hm['minimum_absolute_eigenvalue']!=h['nearest_zero']:raise ValueError('full held spectrum binding')
            for item in r['small_block_checks'].values():
                if item['dimension']!=2*sites:raise ValueError('small dimension')
                for key in ('eigenpair_frobenius_residual','eigenvector_orthogonality_frobenius','eigvalsh_eigh_max_difference'):
                    if not isinstance(item[key],(int,float)) or not math.isfinite(item[key]) or item[key]<0:raise ValueError('invalid small residual')
            for dictionary,keys in [(r['orthogonality'],['joint_defect_frobenius','joint_eigenvector_orthogonality_frobenius','orthonormal_control_frobenius']),
                                    (h,['eigenpair_frobenius_residual','eigenvector_orthogonality_frobenius'])]:
                if any(not math.isfinite(dictionary[k]) or dictionary[k]<0 for k in keys):raise ValueError('invalid norm diagnostic')
    first=next(r for r in rows if r['clock_rs']==12 and r['kappa']==1)
    c=first['small_block_checks']['orthonormal_control_interior_shift0']
    expected=[116,2053,2.3138386761991154,0.9760720296720067,1.6603208258650978e-14,2.9993561552753807e-13,3.5107008446166876e-15,1.001625193035389e-11]
    actual=[first['dimension'],first['polar_column_count'],first['polar_physical_interior_entropy'],first['held']['entropy'],c['eigenpair_frobenius_residual'],c['eigenvector_orthogonality_frobenius'],c['eigvalsh_eigh_max_difference'],first['orthogonality']['orthonormal_control_frobenius']]
    if actual!=expected:raise ValueError('frozen T12 channel-one known input binding failed')
    return rows


def comparison(reading,upper,printed):
    center=Fraction(printed);distance=abs(I(reading-center));display=I('0.0000000005')
    lower=max(Decimal(0),(distance-display).lo);higher=(distance+display).hi
    gap=I(lower,higher);allowance=I('0.000000185')+I(upper.hi)
    verdict='pass' if gap.hi<=allowance.lo else 'fail' if gap.lo>allowance.hi else 'unresolved_display_precision'
    return {'printed_first_route_center':printed,'printed_display_half_width':'5e-10','absolute_gap_including_display':gap,
            'supplied_first_route_allowance':'1.85e-7','conditional_comparison_allowance':allowance,
            'conditional_model_and_supplied_allowance_verdict':verdict,'unconditional_certified_verdict':None}


def controls():
    tests={}
    tests['rational_one_third_enclosed']=(I(1)/3).contains_fraction(Fraction(1,3))
    tests['signed_interval_product']=(I(-2,3)*I(-5,7)).lo==Decimal(-15) and (I(-2,3)*I(-5,7)).hi==Decimal(21)
    root=I(2).sqrt();tests['sqrt_exact_rational_enclosure']=Fraction(root.lo)**2<=2<=Fraction(root.hi)**2
    tests['perfect_square_exact']=I(4).sqrt().lo==I(4).sqrt().hi==2
    partial=sum((Fraction(2, (2*k+1)*3**(2*k+1)) for k in range(110)),Fraction())
    tail=Fraction(2, (2*110+1)*3**(2*110+1))*Fraction(9,8)
    log=I(2).ln();tests['log_two_independent_rational_series']=Fraction(log.lo)<=partial and Fraction(log.hi)>=partial+tail
    tests['binary64_exact_rational_input']=I(.1).contains_fraction(Fraction.from_float(.1))
    def rejects(fn):
        try:fn();return False
        except ValueError:return True
    tests['division_zero_rejected']=rejects(lambda:I(1)/I(-1,1))
    tests['negative_sqrt_rejected']=rejects(lambda:I(-1).sqrt())
    tests['failed_Gram_defect_unavailable']=rejects(lambda:Model().defect(2,2,1.))
    zero=Model(u=Fraction(0));entropy={'minimum':0.,'maximum':1.};check={'eigenvector_orthogonality_frobenius':0.,'eigenpair_frobenius_residual':0.,'eigvalsh_eigh_max_difference':0.}
    small=zero.small(2,I(1),I(2).sqrt(),I(0),entropy,check)
    fixture={'source_row_dimension':4,'polar_column_count':2,'dimension':2,'site_shift':0,'weight':16,
             'orthogonality':{'joint_defect_frobenius':0.,'joint_eigenvector_orthogonality_frobenius':0.,'orthonormal_control_frobenius':0.,'joint_eigenvalue_min':1.},
             'held_operator':{'negative_count':2,'eigenvector_orthogonality_frobenius':0.,'eigenpair_frobenius_residual':0.},
             'held_spectrum_metadata':{'maximum_absolute_eigenvalue':1.,'minimum_absolute_eigenvalue':.5},
             'polar_interior':entropy,'held':{**entropy,'entropy':0.},'polar_physical_interior_entropy':0.,
             'small_block_checks':{'orthonormal_control_interior_shift0':check,'held_shift0':check}}
    whole=zero.channel(fixture)
    tests['formal_exact_arithmetic_zero_bound']=whole['conditional_model_upper_interval'].lo==whole['conditional_model_upper_interval'].hi==0
    failed_gap={**fixture,'held_spectrum_metadata':{'maximum_absolute_eigenvalue':1.,'minimum_absolute_eigenvalue':0.}}
    tests['failed_held_gap_unavailable']=rejects(lambda:Model().channel(failed_gap))
    tests['modulus_saturation']=binary_entropy_increasing(I('.9').cap('.5')).lo==log.lo and binary_entropy_increasing(I('.9').cap('.5')).hi==log.hi
    tests['display_interval_not_exact_first_number']=comparison(Fraction(1),I(0),'1.000000000')['absolute_gap_including_display'].hi==Decimal('5e-10')
    tests['original_hypotheses_not_certified']=small['certified_from_saved_record_upper'] is None
    if not all(tests.values()):raise RuntimeError('outward saved-bound manufactured control failed: '+repr(tests))
    return tests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--controls-only',action='store_true')
    args=p.parse_args();runtime=require_remote_cpu(args.output)
    files=[Path(__file__),Path(__file__).with_name('decimal_interval_v1.py'),Path(__file__).with_name('cpu_runtime.py'),args.spec]
    hashes={str(path.name):digest(path) for path in files}
    if args.controls_only:
        validated=validate_rows(json.loads(args.input.read_text()))
        answer={'input_metadata_sha256':digest(args.input),'known_input_bindings_passed':len(validated)==60,'runtime':runtime,'source_spec_hashes':hashes,'precision_decimal_digits':PRECISION,'tests':controls(),'all_passed':True}
        write(args.output/'known_answers.json',answer);print(json.dumps({'all_passed':True,'controls':len(answer['tests'])}),flush=True);return
    prior=json.loads((args.output/'known_answers.json').read_text())
    if not prior['all_passed'] or prior['source_spec_hashes']!=hashes or prior['input_metadata_sha256']!=digest(args.input):raise ValueError('passing exact-source manufactured controls required')
    saved=json.loads(args.input.read_text());source_rows=validate_rows(saved);model=Model();rows=[]
    for source in source_rows:
        result={'clock_rs':source['clock_rs'],'kappa':source['kappa'],'weight':source['weight'],'site_shift':source['site_shift'],
                'interior_sites':source['interior_sites'],'dimension':source['dimension'],'polar_column_count':source['polar_column_count'],
                'source_dtype':source['source_dtype'],'input_metadata':source,'exact_binary64_inputs':exact_binary_inputs(source)}
        try:result.update(model.channel(source))
        except ValueError as error:result.update(availability='unavailable',reason=str(error),conditional_model_upper=None,certified_from_saved_record_upper=None)
        rows.append(result)
    summaries=[]
    first={row['clock_rs']:row for row in saved['first_route_printed_rows']}
    if set(first)!=set(MATCHES):raise ValueError('six unique printed centers required')
    for clock in MATCHES:
        group=[r for r in rows if r['clock_rs']==clock];item={'clock_rs':clock,'channel_count':len(group),'weight_sum':sum(r['weight'] for r in group)}
        if any(r['availability']!='available_under_stated_model' for r in group):
            item.update(availability='unavailable',conditional_model_upper=None,certified_from_saved_record_upper=None);summaries.append(item);continue
        complement=sum((r['weight']*r['saved_complement_reading_exact'] for r in group),Fraction())
        held=sum((r['weight']*r['saved_held_reading_exact'] for r in group),Fraction());reading=complement-held
        uc=sum((r['weight']*r['state_small']['conditional_model_upper'] for r in group),I(0))
        uh=sum((r['weight']*r['held_small']['conditional_model_upper'] for r in group),I(0));upper=uc+uh
        item.update(availability='available_under_stated_model',weighted_complement_exact=complement,weighted_held_exact=held,
                    weighted_held_subtracted_exact=reading,weighted_state_upper=uc,weighted_held_upper=uh,
                    conditional_model_upper_interval=upper,conditional_model_upper=str(upper.hi),certified_from_saved_record_upper=None,
                    comparison=comparison(reading,upper,first[clock]['printed_excess_decimal']))
        summaries.append(item)
    write(args.output/'results.json',{'scope':'Frozen saved-data conditional operation-model upper bounds; no eigensolver, covariance reconstruction or entropy-trace rerun.',
          'runtime':runtime,'python_version':sys.version,'precision_decimal_digits':PRECISION,'source_spec_hashes':hashes,
          'input_metadata_source':str(args.input),'input_metadata_sha256':digest(args.input),'original_model_hypotheses_unverified':MISSING_HYPOTHESES,
          'outward_arithmetic_documentation':'https://docs.python.org/3/library/decimal.html','rows':rows,'summaries':summaries})
    print(json.dumps({'channel_rows':len(rows),'summaries':len(summaries),'available_channels':sum(r['availability']=='available_under_stated_model' for r in rows)}),flush=True)


if __name__=='__main__':main()
