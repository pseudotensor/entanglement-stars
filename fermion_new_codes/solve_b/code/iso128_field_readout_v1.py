"""Independent dense root/minimum and saved-data helpers for ISO128."""
from __future__ import annotations
import math
from decimal import Decimal,localcontext
import mpmath
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import cumulative_simpson
from scipy.special import erfc
from iso128_rebuild_core_v1 import gaussian_kernels
from iso128_field_core_v1 import DomainUnavailable, real_vector


def extrema(values, radius):
    if np.iscomplexobj(values) or np.iscomplexobj(radius):raise ValueError('Real extrema inputs required')
    values=np.asarray(values,dtype=float);radius=np.asarray(radius,dtype=float)
    if values.shape != radius.shape or values.ndim != 1:
        raise ValueError('Extrema values/radius shape mismatch')
    finite=np.isfinite(values)&np.isfinite(radius)
    if not np.any(finite):
        return {'available':False,'value':None,'radii':[],'missing_count':len(values)}
    value=float(np.min(values[finite]));ties=finite&(values==value)
    return {'available':True,'value':value,'radii':radius[ties].tolist(),
            'missing_count':int(np.sum(~finite))}


def uniform_grid(end,step):
    if not all(math.isfinite(x) and x>0 for x in (end,step)):
        raise ValueError('Positive finite grid specification required')
    intervals=round(end/step)
    if not math.isclose(intervals*step,end,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Grid endpoint must be included exactly')
    return np.arange(intervals+1,dtype=float)*step


def vacuum_readout(grid):
    grid=real_vector(grid,'vacuum search grid')
    if grid[0]!=0 or np.any(np.diff(grid)<=0):raise ValueError('Vacuum domain from centre required')
    tied={'available':True,'value':1.,'radii':grid.tolist(),'missing_count':0,
          'whole_domain_tied':True,'tied_interval':[0.,float(grid[-1])]}
    return {'status':'no_root','resolved':True,'outer':None,'inner':None,'roots':[],
            'minimum':tied,'N_minimum':tied,'sampled_N_minimum':tied,'sampled_minimum':tied,
            'unresolved':[],'grid':grid.tolist(),'analytic_branch':'identically_zero_sources'}


def _bracketed(fun,left,right):
    return float(brentq(fun,left,right,xtol=1e-10,rtol=1e-13))


def dense_readout(observable, grid, lower=0., upper=None, extra_candidates=()):
    """Input callable supplies f,g,df,dg,N,dN with explicit representation.

    g is the root surrogate (U/r-1 in the production IVP). Minima always
    evaluate f itself. Every unresolved candidate survives in the output.
    """
    grid=real_vector(grid,'search grid')
    upper=float(grid[-1] if upper is None else upper)
    if np.any(np.diff(grid)<=0) or grid[0] != lower or grid[-1] != upper:
        raise ValueError('Unique ascending search grid with both endpoints required')
    cache={}
    def at(r):
        r=float(r)
        if r not in cache:
            row=observable(r)
            if set(row) != {'f','g','df','dg','N','dN'} or not all(math.isfinite(float(x)) for x in row.values()):
                raise DomainUnavailable('Nonfinite/incomplete readout function')
            if row['N']<=0:
                raise DomainUnavailable('Nonpositive readout lapse')
            cache[r]={k:float(v) for k,v in row.items()}
        return cache[r]
    samples=[at(r) for r in grid]
    stationary={k:[] for k in ('df','dg','dN')};unresolved=[]
    for key in stationary:
        values=np.array([x[key] for x in samples])
        for i,(left,right) in enumerate(zip(grid[:-1],grid[1:])):
            a,b=values[i:i+2]
            if (a<0<b) or (b<0<a):
                try:
                    r=_bracketed(lambda r:at(r)[key],left,right)
                    stationary[key].append({'r':r,'bracket':[float(left),float(right)],'residual':at(r)[key]})
                except (ValueError,DomainUnavailable,FloatingPointError,OverflowError) as exc:
                    unresolved.append({'kind':'stationary_refinement_failed','quantity':key,'bracket':[float(left),float(right)],'reason':str(exc)})
            if a==0:
                stationary[key].append({'r':float(left),'bracket':[float(left),float(left)],'residual':0.})
            if a==b==0 and key in ('df','dg'):
                unresolved.append({'kind':'plateau','quantity':key,'interval':[float(left),float(right)]})
        if values[-1]==0:
            stationary[key].append({'r':float(grid[-1]),'bracket':[float(grid[-1]),float(grid[-1])],'residual':0.})
    # Critical points of both functions split intervals before the crossing
    # search, so same-sign endpoints alone cannot hide the tested pair.
    splits=np.unique(np.concatenate([grid,np.array([x['r'] for key in ('df','dg') for x in stationary[key]]),
                                    np.asarray(extra_candidates,dtype=float)]))
    if np.any(splits<lower) or np.any(splits>upper):
        raise ValueError('Extra candidate outside readout domain')
    roots=[];exact=[]
    for left,right in zip(splits[:-1],splits[1:]):
        a,b=at(left)['g'],at(right)['g']
        if (a<0<b) or (b<0<a):
            try:
                root=_bracketed(lambda r:at(r)['g'],left,right)
                roots.append({'r':root,'bracket':[float(left),float(right)],'signs':[int(np.sign(a)),int(np.sign(b))],
                    'direction':'negative_to_positive' if at(left)['f']<0 else 'positive_to_negative',
                    'f_residual':at(root)['f'],'g_residual':at(root)['g'],'resolved':abs(at(root)['f'])<=1e-9})
            except (ValueError,DomainUnavailable,FloatingPointError,OverflowError) as exc:
                unresolved.append({'kind':'root_refinement_failed','bracket':[float(left),float(right)],'reason':str(exc)})
        if a==0 and left>0:exact.append(float(left))
    if at(splits[-1])['g']==0 and splits[-1]>0:exact.append(float(splits[-1]))
    for r in exact:
        if any(r==x['r'] for x in roots):continue
        index=int(np.searchsorted(splits,r));left=float(splits[max(0,index-1)]);right=float(splits[min(len(splits)-1,index+1)])
        a,b=at(left)['f'],at(right)['f']
        if left==r or right==r:
            direction='boundary_zero';resolved=False
        elif (a<0<b) or (b<0<a):
            direction='negative_to_positive' if a<0 else 'positive_to_negative';resolved=True
        else:
            direction='tangent_unresolved';resolved=False
        roots.append({'r':r,'bracket':[left,right],'signs':[int(np.sign(a)),int(np.sign(b))],
                      'direction':direction,'f_residual':at(r)['f'],'g_residual':at(r)['g'],'resolved':resolved and abs(at(r)['f'])<=1e-9})
    roots.sort(key=lambda x:x['r'])
    for a,b in zip(roots[:-1],roots[1:]):
        if b['r']-a['r']<1e-7:
            unresolved.append({'kind':'close_pair','radii':[a['r'],b['r']]})
    for item in stationary['df']:
        if abs(at(item['r'])['f'])<=1e-10:
            unresolved.append({'kind':'near_tangent_unresolved','r':item['r'],'f':at(item['r'])['f']})
    candidates=[{'r':float(lower),'kind':'endpoint'},{'r':upper,'kind':'endpoint'}]
    candidates += [{'r':x['r'],'kind':'stationary','bracket':x['bracket']} for x in stationary['df']]
    candidates += [{'r':float(r),'kind':'declared_breakpoint'} for r in extra_candidates]
    sampled_f=np.array([x['f'] for x in samples])
    for i in range(1,len(grid)-1):
        if sampled_f[i]<=sampled_f[i-1] and sampled_f[i]<=sampled_f[i+1] and (sampled_f[i]<sampled_f[i-1] or sampled_f[i]<sampled_f[i+1]):
            candidates.append({'r':float(grid[i]),'kind':'sampled_local_minimum'})
            try:
                fit=minimize_scalar(lambda r:at(r)['f'],bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-11})
                if not fit.success:raise DomainUnavailable(str(fit.message))
                candidates.append({'r':float(fit.x),'kind':'bounded_local_minimum','success':True})
            except (ValueError,DomainUnavailable,FloatingPointError,OverflowError) as exc:
                unresolved.append({'kind':'minimum_refinement_failed','r':float(grid[i]),'bracket':[float(grid[i-1]),float(grid[i+1])],'reason':str(exc)})
    for x in candidates:x['value']=at(x['r'])['f']
    minimum=extrema([x['value'] for x in candidates],[x['r'] for x in candidates])
    if minimum['value']>float(np.min(sampled_f))+1e-12:
        unresolved.append({'kind':'candidate_minimum_above_sampled'})
    n_candidates=np.unique([lower,upper]+list(extra_candidates)+[x['r'] for x in stationary['dN']])
    nminimum=extrema([at(r)['N'] for r in n_candidates],n_candidates)
    sampled_n=extrema([x['N'] for x in samples],grid)
    if nminimum['value']>sampled_n['value']+1e-12:unresolved.append({'kind':'candidate_lapse_minimum_above_sampled'})
    outer=inner=None
    for i,root in enumerate(roots):
        outside=splits[splits>=root['bracket'][1]]
        outside=outside[outside>root['r']]
        root['outside_positive_on_inspected_support']=bool(len(outside) and all(at(r)['f']>0 for r in outside))
        if root['resolved'] and root['direction']=='negative_to_positive' and root['outside_positive_on_inspected_support']:
            outer=root
            inner=roots[i-1] if i>0 and roots[i-1]['resolved'] and roots[i-1]['direction']=='positive_to_negative' else None
    unresolved.extend({'kind':'unresolved_root','r':x['r'],'direction':x['direction']} for x in roots if not x['resolved'])
    if outer is None and any(x['resolved'] and x['direction']=='negative_to_positive' for x in roots):
        unresolved.append({'kind':'outer_positive_support_not_established'})
    status='pair' if inner is not None else 'outer_only' if outer is not None else 'no_root' if not roots else 'unresolved'
    if at(upper)['f']<0:status='boundary_truncated'
    return {'status':status,'resolved':not unresolved,'outer':outer,'inner':inner,'roots':roots,'minimum':minimum,
            'N_minimum':nminimum,'sampled_N_minimum':sampled_n,'sampled_minimum':extrema(sampled_f,grid),
            'declared_breakpoints':list(extra_candidates),
            'candidates':candidates,'stationary':stationary,'unresolved':unresolved,
            'grid':grid.tolist(),'samples':samples,'evaluation_cache':[dict(r=r,**row) for r,row in sorted(cache.items())]}


def cumulative_w(radius,N,rho,g):
    if np.iscomplexobj(g) or not math.isfinite(g) or g<=0:raise ValueError('Positive finite real coupling required')
    radius=real_vector(radius,'W radius');N=real_vector(N,'W lapse');rho=real_vector(rho,'W density')
    if radius[0] != 0 or np.any(np.diff(radius)<=0) or radius.shape != N.shape or rho.shape != radius.shape or np.any(N<=0):
        raise ValueError('W quadrature requires associated positive-lapse samples from centre')
    # SciPy cumulative_simpson: quadratic integration of each half-panel;
    # its final odd endpoint uses the three-point backward panel. Controls
    # compare every endpoint with explicit quadratic moments.
    integrand=8*np.pi*g*radius**2*N**2*rho
    result=cumulative_simpson(integrand,x=radius,initial=0)
    if not np.all(np.isfinite(integrand)) or not np.all(np.isfinite(result)):raise DomainUnavailable('Nonfinite derived W quadrature')
    return integrand,result


def tail0(L,R,ell):
    if any(np.iscomplexobj(x) for x in (L,R,ell)):raise ValueError('Real tail inputs required')
    L,R=np.broadcast_arrays(np.asarray(L,dtype=float),np.asarray(R,dtype=float))
    if np.any(~np.isfinite(L)) or np.any(~np.isfinite(R)) or np.any(L<0) or np.any(R<=0) or not math.isfinite(ell) or ell<=0:
        raise ValueError('Finite nonnegative lower limit, positive shell/width required')
    a=(L-R)/(math.sqrt(2)*ell);b=(L+R)/(math.sqrt(2)*ell)
    # Difference of exponentials via expm1 avoids loss at the centre.
    difference=np.exp(-a*a)*(-np.expm1(-2*L*R/ell**2))
    return .5*(erfc(a)+erfc(b))+ell/(R*math.sqrt(2*np.pi))*difference


def lower_mass0(L,R,ell):
    """Stable small lower Gaussian-shell mass, analytic erfc at 80 digits.

    This avoids subtracting a near-unit upper mass in binary64. The erfc
    difference is tiny itself on the 8/12-width bands, and remains positive.
    This diagnostic is not an outward bound on arbitrary exterior fields.
    """
    if any(np.iscomplexobj(x) or not math.isfinite(x) for x in (L,R,ell)) or L<0 or R<=0 or ell<=0:
        raise ValueError('Valid real Gaussian mass domain required')
    if L==0:return 0.
    with mpmath.workdps(80):
        l,r,s=map(mpmath.mpf,(float(L),float(R),float(ell)))
        a=(r-l)/(mpmath.sqrt(2)*s);b=(r+l)/(mpmath.sqrt(2)*s)
        result=(mpmath.erfc(a)-mpmath.erfc(b))/2-s/(r*mpmath.sqrt(2*mpmath.pi))*(mpmath.exp(-a*a)-mpmath.exp(-b*b))
        if result<0 or result>1:raise DomainUnavailable('Analytic lower mass precision unresolved')
        return float(result)


def backaverage(field, shells, ell, step, band):
    if getattr(field,'field_role',None)!='own_normalized_field':raise ValueError('Explicit normalized own field role required')
    if any(np.iscomplexobj(x) or not math.isfinite(x) or x<=0 for x in (ell,step,band)):raise ValueError('Positive finite real averaging recipe required')
    shells=real_vector(shells,'applied shell radius')
    if np.any(shells<=0) or np.any(np.diff(shells)<=0):raise ValueError('Unique positive applied shells required')
    rows=[]
    for R in shells:
        lower=max(0.,R-band*ell);upper=R+band*ell
        count=round((upper-lower)/step)
        if not math.isclose(count*step,upper-lower,rel_tol=0,abs_tol=1e-12):raise ValueError('Applied endpoint missing')
        r=lower+np.arange(count+1)*step
        N,v=field.applied_exterior(r)
        if np.iscomplexobj(N) or np.iscomplexobj(v) or np.shape(N)!=r.shape or np.shape(v)!=r.shape or not np.all(np.isfinite(N)) or not np.all(np.isfinite(v)) or np.any(N<=0) or np.any(v>0):raise DomainUnavailable('Invalid declared applied field values')
        k0,k1=gaussian_kernels(r,R,ell);measure=4*np.pi*r*r
        scal=np.trapezoid(measure*k0*N,x=r);vec=np.trapezoid(measure*k1*v,x=r)
        row=[scal,vec,np.trapezoid(measure*k0,x=r),np.trapezoid(measure*k1,x=r),
                     lower_mass0(lower,R,ell),float(tail0(upper,R,ell))]
        if not np.all(np.isfinite(row)):raise DomainUnavailable('Nonfinite derived backaverage')
        rows.append(row)
    return {'R':shells,'values':np.asarray(rows),'columns':['Nbar','vbar','scalar_weight','vector_weight','lower_omitted_scalar_mass','upper_omitted_scalar_mass'],
            'step':step,'band':band,'renormalized':False,'exterior':'declared_analytic_vacuum'}


def empirical_band(primary, variants, floor):
    if not math.isfinite(floor) or floor<=0:raise ValueError('Positive finite reporting guard required')
    if primary is None or not math.isfinite(primary) or set(variants)!={'H','C','O','mesh'} or any(x is None or not math.isfinite(x) for x in variants.values()):
        return {'status':'precision_unresolved','band':None,'spread':None}
    differences={key:float(value-primary) for key,value in variants.items()}
    spread=max(abs(x) for x in differences.values());half=max(4*spread,floor)
    if not all(math.isfinite(x) for x in [spread,half,primary-half,primary+half]):return {'status':'precision_unresolved','band':None,'spread':None,'reason':'derived_nonfinite'}
    return {'status':'empirical','band':[primary-half,primary+half],'spread':spread,'differences':differences,'half_width':half}


def printed_precision(band, displayed, decimals):
    if band is None:return 'precision_unresolved'
    if len(band)!=2 or not all(math.isfinite(x) for x in band) or band[0]>band[1]:raise ValueError('Ordered finite empirical band required')
    if not isinstance(displayed,str) or not isinstance(decimals,int) or decimals<0:raise ValueError('Literal displayed decimal token and nonnegative decimal count required')
    center=Decimal(displayed)
    if not center.is_finite():raise ValueError('Finite display token required')
    with localcontext() as ctx:
        ctx.prec=max(80,len(displayed)+decimals+10)
        half=Decimal(5).scaleb(-decimals-1);lower=center-half;upper=center+half
    left,right=map(Decimal.from_float,map(float,band))
    if left>lower and right<upper:return 'agrees_at_printed_precision'
    if right<lower or left>upper:return 'mismatch_at_named_convention'
    return 'precision_unresolved'


def compare_applied(own,stored,own_radius,stored_radius,own_clock,expected_clock,role):
    if role not in ('N_applied','v_applied'):raise ValueError('Declared applied field role required')
    radius=real_vector(own_radius,'own applied radius');reference_radius=real_vector(stored_radius,'stored applied radius')
    aa=real_vector(own,'own averaged field');bb=real_vector(stored,'stored applied field')
    if radius.shape!=reference_radius.shape or not np.array_equal(radius,reference_radius) or np.any(np.diff(radius)<=0) or radius[0]<=0:
        raise ValueError('Exact applied radial association required')
    if aa.shape!=radius.shape or bb.shape!=radius.shape:raise ValueError('Applied fields/radii shape mismatch')
    if np.iscomplexobj(own_clock) or not math.isfinite(own_clock) or own_clock!=expected_clock:
        raise ValueError('Exact physical clock association required')
    difference=aa-bb
    if not np.all(np.isfinite(difference)):raise DomainUnavailable('Derived applied difference nonfinite')
    maximum=float(np.max(np.abs(difference)));scale=float(np.max(np.abs(bb)))
    ratio=maximum/scale if scale else 0. if maximum==0 else None
    if ratio is not None and not math.isfinite(ratio):raise DomainUnavailable('Derived applied normalized difference nonfinite')
    return {'difference':difference,'max_abs':maximum,'stored_scale':scale,
            'max_abs_over_stored_max':ratio,'normalized_available':ratio is not None,
            'zero_scale_status':'nonzero_scale' if scale else 'both_zero' if maximum==0 else 'nonzero_difference_zero_scale',
            'radii_at_max':radius[np.abs(difference)==maximum].tolist(),'orientation':'own_minus_stored','role':role}


def geometry_signature(readout):
    roots=readout['roots'];positions=[x['r'] for x in roots]
    def index(root):
        if root is None:return None
        matches=[i for i,x in enumerate(roots) if x==root]
        if len(matches)!=1:raise ValueError('Selected root lacks unique inventory association')
        return matches[0]
    grid=readout['grid'];classes=[]
    for r in readout['minimum']['radii']:
        kind='centre' if r==0 else 'outer_boundary' if r==grid[-1] else 'declared_breakpoint' if r in readout.get('declared_breakpoints',[]) else 'interior'
        classes.append(kind)
    return {'status':readout['status'],'root_count':len(roots),'directions':[x['direction'] for x in roots],
            'selected_inner_index':index(readout['inner']),'selected_outer_index':index(readout['outer']),
            'minimum_location_classes':sorted(set(classes)),'whole_domain_tie':readout['minimum'].get('whole_domain_tied',False)}


def geometry_consistency(variants,expected):
    if set(variants)!={'P','H','C','O'}:return {'resolved':False,'reason':'missing_variant','signatures':{}}
    signatures={};reason=[];expected_mismatches=[]
    for variant,row in variants.items():
        if not row.get('available') or set(row.get('dense_readouts',{}))!={'16','32'}:
            reason.append(variant+': incomplete');continue
        for mesh,readout in row['dense_readouts'].items():
            name=variant+mesh
            if not readout['resolved']:reason.append(name+': unresolved geometry')
            if readout['status']!=expected:expected_mismatches.append(name)
            signatures[name]=geometry_signature(readout)
    if len(signatures)!=8 or any(x!=signatures.get('P16') for x in signatures.values()):reason.append('root inventory, pocket association or minimum location class differs')
    return {'resolved':not reason,'reason':reason,'signatures':signatures,
            'expected_geometry_met':not expected_mismatches and len(signatures)==8,
            'expected_geometry_mismatches':expected_mismatches}


def defect_summary(difference,radius,collar,charge_denominator):
    if np.iscomplexobj(difference) or np.iscomplexobj(radius):raise ValueError('Real defect data required')
    d=np.asarray(difference,dtype=float);r=real_vector(radius,'defect radius')
    if d.shape!=r.shape or np.any(np.diff(r)<=0) or not math.isfinite(charge_denominator) or charge_denominator<=0:
        raise ValueError('Associated defect grid/positive charge denominator required')
    finite=np.isfinite(d);indexes=np.flatnonzero(r==collar)
    maximum=float(np.max(np.abs(d[finite]))) if np.any(finite) else None
    tied=finite&(np.abs(d)==maximum) if maximum is not None else np.zeros_like(finite)
    c=float(d[indexes[0]]) if len(indexes)==1 and finite[indexes[0]] else None
    charge=c/charge_denominator if c is not None else None
    if charge is not None and not math.isfinite(charge):raise DomainUnavailable('Derived charge defect nonfinite')
    return {'available':bool(np.any(finite)),'missing_count':int(np.sum(~finite)),
            'max_absolute':maximum,'all_max_radii':r[tied].tolist(),'signed_values_at_all_max':d[tied].tolist(),
            'signed_collar_defect':c,'charge_discrepancy_at_collar':charge,'orientation':'first_minus_second'}


def additional_collar_metadata(U_alt,y_primary,y_alt,primary_min,g):
    values=(U_alt,y_primary,y_alt,primary_min,g)
    if any(np.iscomplexobj(x) or not math.isfinite(x) for x in values) or g<=0:
        raise DomainUnavailable('Finite real collar sensitivity inputs required')
    try:
        E=U_alt/(2*g);delta=y_primary-y_alt
        if not math.isfinite(delta):raise DomainUnavailable('Derived collar gauge difference nonfinite')
        factor=math.exp(delta);minimum=primary_min*factor
    except (OverflowError,ZeroDivisionError) as exc:raise DomainUnavailable('Derived collar sensitivity unavailable') from exc
    if not all(math.isfinite(x) for x in (E,factor,minimum)) or factor<=0:raise DomainUnavailable('Derived collar sensitivity nonfinite')
    return {'collar':4128.,'E':E,'normalization_factor':factor,'fmin_common_0_4096_domain':minimum}
