"""One missing-only110 generation with separate-complex tensor-Gauss bounds.

No111 evaluation, accepted-leaf reevaluation, adaptive second generation or retry.
"""
from fractions import Fraction as F
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

BASE=Path('<internal>/eta_massive_scalar_v1')
OLD=BASE/'environment_continuation'
ROOT=OLD/'separate_images_v1'
SOURCE=OLD/'fixed_mass_sharp_repair_result.json'
AUDIT=OLD/'saved_110_width_attribution.json'
ENGINE=BASE/'fixed_mass_interval_sharp_repair.py'
PINS={SOURCE:'0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e',
      AUDIT:'10470fc19620d3b3259176769f2caca1836f9fe634f675d475fe83c3ad9758ac',
      ENGINE:'d962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ends(x):
    return tuple(F(int(x[k]['numerator']),int(x[k]['denominator'])) for k in ('lower_exact','upper_exact'))


def number(x):return F(int(x['numerator']),int(x['denominator']))


def pack(x):
    if isinstance(x,F):return dict(numerator=str(x.numerator),denominator=str(x.denominator))
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x


def main():
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory required')
    target=ROOT/'separate_images_result.json'
    if target.exists() or target.with_suffix('.json.tmp').exists():raise FileExistsError('Never retry or overwrite')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[8] if len(cpus)>8 else cpus[0]])
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(110,120))
    def deadline(signum,frame):raise TimeoutError('One-generation allocation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(140)
    start=time.monotonic()
    report=dict(status='incomplete',y='1/25',purpose='SecIII G5/I6a missing110 only',
        science='T8 separate-complex radius4; T9 pre-node width floor',
        resources=dict(wall_seconds=150,internal_seconds=140,cpu_seconds=120,soft_cpu_seconds=110,
            address_space_bytes=2*2**30,threads=1,nice=os.getpriority(os.PRIO_PROCESS,0)),
        source_pins={str(k):v for k,v in PINS.items()},parents=[],
        counts=dict(evaluations=0,refused_images=0,refusals_by_reason={}),guard_samples={},tail_radius_maxima={},
        attempted_child_count=0,completed_parent_modulus_count=0,completed_parent_node_count=0,replaced_parent_ids=[],results={})
    engine=None;frontier=None;fixed=None;image_context=None
    def persist():
        report['elapsed_seconds']=time.monotonic()-start
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['producer_sha256']=sha(Path(__file__))
        if frontier is not None:
            report['frontier_real']=frontier[:2];report['frontier_imaginary']=frontier[2:]
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(pack(report),indent=2,allow_nan=False)+'\n')
        temporary.replace(target)
    last_checkpoint=start
    def checkpoint(completed,force=False):
        nonlocal last_checkpoint
        if force or completed%16==0 or time.monotonic()-last_checkpoint>=10:
            persist();last_checkpoint=time.monotonic()
    try:
        for path,expected in PINS.items():
            if sha(path)!=expected:raise ValueError('Changed input: '+str(path))
        source=json.loads(SOURCE.read_text());audit=json.loads(AUDIT.read_text())
        if source['status']!='incomplete' or source['y']!='1/25' or audit['status']!='pass':
            raise ValueError('Unexpected saved source/audit state')
        leaves=source['active_panel']['leaves'];parents=[x for x in leaves if x['method']=='nonnegative_real_bound']
        fixed=[x for x in leaves if x['method']=='tensor_gauss']
        if len(parents)!=443 or len(fixed)!=3798:raise ValueError('Wrong missing-only leaf set')
        frontier=[sum((ends(x[key])[j] for x in leaves),F(0)) for key in ('real','imaginary') for j in (0,1)]
        if tuple(frontier[:2])!=ends(source['active_panel']['real']) or tuple(frontier[2:])!=ends(source['active_panel']['imaginary']):
            raise ValueError('Saved exact frontier sum changed')
        report['unchanged_gauss_source_ids']=[x['id'] for x in fixed]
        G=sum((ends(x['real'])[1]-ends(x['real'])[0] for x in fixed),F(0))
        WF=sum((ends(x['real'])[1]-ends(x['real'])[0] for x in parents),F(0))
        omission=ends(source['omitted_integral_bounds']['eta110_nonnegative'])[1]
        B_run=min(F(1,10000),F(1,5000)-omission);B_print=F(8,1000000)-omission
        if min(B_run-G,B_print-G)<=0:raise ValueError('No positive missing-width budget')
        report.update(fixed_gauss_width=G,old_fallback_width=WF,omitted_upper=omission,
            retained_run_budget=B_run,retained_print_budget=B_print,
            preserved111='Source certificate referenced unchanged; no111 call exists in this producer')
        import flint
        from flint import arb,acb,ctx
        ctx.prec,ctx.threads=192,1
        spec=importlib.util.spec_from_file_location('owned_fixed_mass_integrand',ENGINE)
        engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        engine.arb,engine.acb=arb,acb;engine.REPORT=report;engine.START=start;engine.constants()
        def saved_ball(record):
            lo,hi=ends(record)
            return engine.point(lo).union(engine.point(hi))
        engine.NODES=[(saved_ball(x['node']),saved_ball(x['weight'])) for x in source['gauss_nodes_and_weights']]
        if len(engine.NODES)!=8 or not all(x>-1 and x<1 and w>0 for x,w in engine.NODES):
            raise ValueError('Saved positive Gauss-rule inputs invalid')
        moments=[]
        for degree in range(16):
            value=sum((w*x**degree for x,w in engine.NODES),arb(0))
            expected=engine.point(F(2,degree+1) if degree%2==0 else F(0))
            if not value.contains(expected):raise ValueError('Saved-rule polynomial known answer failed')
            moments.append(dict(degree=degree,quadrature=engine.interval(value),expected=str(F(2,degree+1) if degree%2==0 else F(0))))
        tensor_controls=[]
        for area in (F(1),F(6)):
            raw_constant=sum((wx*wy for x,wx in engine.NODES for y,wy in engine.NODES),arb(0))*engine.point(area/4)
            raw14=sum((wx*wy*x**14*y**14 for x,wx in engine.NODES for y,wy in engine.NODES),arb(0))*engine.point(area/4)
            raw16=sum((wx*wy*x**16 for x,wx in engine.NODES for y,wy in engine.NODES),arb(0))*engine.point(area/4)
            exact_deficit=area*F(2**16*math.factorial(8)**4,17*math.factorial(16)**2)
            deficit=engine.point(area/F(17))-raw16
            allowance=(arb(8)/3*arb(4)**-16*engine.point(area)*(arb(2)**40+1)).upper()
            if (not raw_constant.contains(engine.point(area)) or not raw14.contains(engine.point(area/F(225)))
                    or not deficit.contains(engine.point(exact_deficit)) or not deficit.lower()>0 or not deficit.upper()<=allowance):
                raise ArithmeticError('Tensor polynomial known answer failed')
            tensor_controls.append(dict(area=area,constant_raw=engine.interval(raw_constant),u14v14_raw=engine.interval(raw14),
                u16_raw=engine.interval(raw16),u16_deficit=engine.interval(deficit),u16_exact_deficit=exact_deficit,
                u16_Mz=str(2**40),u16_Mt='1',T8_allowance=engine.interval(allowance)))
        report['known_answers']=dict(saved_rule_moments=moments,tensor_polynomials=tensor_controls,
            source_width_attribution='Pinned exact cover/control passed')
        report['guard_margin_scope']='Logged callback positive-real and denominator checks; every unlogged manual branch inequality remains unchanged in the pinned kernel'
        def commit_frontier(entry,cover,sums):
            nonlocal frontier
            previous=entry['complete_child_sum'] if entry['frontier_committed'] else [*ends(entry['old_real']),*ends(entry['old_imaginary'])]
            if sums[1]-sums[0]>=previous[1]-previous[0]:return
            replacement=[a-b+c for a,b,c in zip(frontier,previous,sums)]
            old_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGXCPU})
            try:
                frontier=replacement
                if not entry['frontier_committed']:report['replaced_parent_ids'].append(entry['source_parent_id'])
                entry['frontier_committed']=True
                entry['children']=cover;entry['complete_child_sum']=sums
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        original_positive=engine.positive_real;original_denominator=engine.denominator;original_evaluate=engine.evaluate
        def positive(z,stage):
            if image_context is not None:
                image_context['last_guard']=dict(stage=stage,real=engine.interval(z.real),imaginary=engine.interval(z.imag)) if z.is_finite() else dict(stage=stage,nonfinite=True)
            value=original_positive(z,stage)
            if image_context is not None:
                margin=engine.rational(value.real.lower());old=image_context['positive_real_margins'].get(stage)
                if old is None or margin<old:image_context['positive_real_margins'][stage]=margin
            return value
        def denominator(z):
            value=original_denominator(z)
            if image_context is not None:
                image_context['nonzero_denominator_checks']+=1
                margin=engine.rational(abs(value).lower())
                old=image_context.get('least_denominator_absolute_lower')
                if old is None or margin<old:image_context['least_denominator_absolute_lower']=margin
            return value
        def evaluate(*args,**kwargs):
            if report['counts']['evaluations']>=116952:raise RuntimeError('New evaluation cap reached')
            return original_evaluate(*args,**kwargs)
        engine.positive_real,engine.denominator,engine.evaluate=positive,denominator,evaluate
        parents.sort(key=lambda x:(-(ends(x['real'])[1]-ends(x['real'])[0]),x['id']))
        panel=dict(direction='110');next_id=source['counts']['created_cells']
        for parent in parents:
            a,b,c,d=map(F,parent['bounds_z_t']);mz,mt=(a+b)/2,(c+d)/2
            bounds=((a,mz,c,mt),(a,mz,mt,d),(mz,b,c,mt),(mz,b,mt,d))
            if sum(((v-u)*(t-s) for u,v,s,t in bounds),F(0))!=(b-a)*(d-c):raise ValueError('Child cover area failure')
            entry=dict(source_parent_id=parent['id'],parent_budget_advisory=(B_print-G)*(ends(parent['real'])[1]-ends(parent['real'])[0])/WF,
                old_bounds=parent['bounds_z_t'],old_real=parent['real'],old_imaginary=parent['imaginary'],children=[],frontier_committed=False)
            report['parents'].append(entry)
            for child_bounds in bounds:
                if report['attempted_child_count']>=1772:raise RuntimeError('One-generation cell cap reached')
                next_id+=1;report['attempted_child_count']+=1
                a0,b0,c0,d0=child_bounds;cz,ct,hz,ht=(a0+b0)/2,(c0+d0)/2,(b0-a0)/2,(d0-c0)/2
                value,radius=engine.real_bound(child_bounds)
                child=dict(id=next_id,bounds_z_t=[str(x) for x in child_bounds],
                    depths=[x+1 for x in parent['depths']],method='nonnegative_real_bound',
                    real=engine.interval(value.real),imaginary=engine.interval(value.imag),
                    real_bound_radius=engine.interval(radius),images={},node_evaluations=0)
                entry['children'].append(child)
                for kind,z,t in (
                    ('separate_z_image',acb(arb(engine.point(cz),4*engine.point(hz)),arb(0,4*engine.point(hz))),acb(arb(engine.point(ct),engine.point(ht)))),
                    ('separate_t_image',acb(arb(engine.point(cz),engine.point(hz))),acb(arb(engine.point(ct),4*engine.point(ht)),arb(0,4*engine.point(ht))))):
                    image_context=dict(positive_real_margins={},nonzero_denominator_checks=0)
                    child['images'][kind]=image_context
                    try:
                        M=engine.evaluate(z,t,panel,kind,modulus=True).upper()
                        image_context.update(status='pass',M=engine.interval(M))
                    except engine.NonanalyticImage as error:
                        image_context.update(status='refused',guard_reason=error.reason,image=error.image,input_image=error.input_image)
                    finally:image_context=None
                if all(x['status']=='pass' for x in child['images'].values()):
                    Mz,Mt=[saved_ball(child['images'][k]['M']) for k in ('separate_z_image','separate_t_image')]
                    E=(arb(8)/3*arb(4)**-16*engine.point((b0-a0)*(d0-c0))*(Mz+Mt)).upper()
                    child['quadrature_error']=engine.interval(E)
                    q=2*engine.rational(E);r=ends(child['real'])[1]-ends(child['real'])[0]
                    child['minimum_possible_width']=min(q,r);child['node_pruned_no_width_gain']=q>=r
                else:
                    child['minimum_possible_width']=ends(child['real'])[1]-ends(child['real'])[0]
                    child['node_pruned_no_width_gain']=True
            sums=[sum((ends(x[key])[j] for x in entry['children']),F(0)) for key in ('real','imaginary') for j in (0,1)]
            entry['complete_child_sum']=sums
            commit_frontier(entry,entry['children'],sums)
            report['completed_parent_modulus_count']+=1
            checkpoint(report['completed_parent_modulus_count'])
        floor=G+sum((min(ends(x['old_real'])[1]-ends(x['old_real'])[0],sum((y['minimum_possible_width'] for y in x['children']),F(0))) for x in report['parents']),F(0))
        report['pre_node_floor']=floor
        report['pre_node_print_target_impossible']=floor>B_print
        report['pre_node_run_target_impossible']=floor>B_run
        persist()
        if floor<=B_run:
            for entry in report['parents']:
                working=copy.deepcopy(entry['children'])
                entry['node_attempt_children']=working
                for child in working:
                    if child['node_pruned_no_width_gain']:continue
                    a,b,c,d=map(F,child['bounds_z_t']);cz,ct,hz,ht=(a+b)/2,(c+d)/2,(b-a)/2,(d-c)/2
                    value=acb(0)
                    try:
                        for x,wx in engine.NODES:
                            for y,wy in engine.NODES:
                                value+=wx*wy*engine.evaluate(acb(engine.point(cz)+engine.point(hz)*x),acb(engine.point(ct)+engine.point(ht)*y),panel,'gauss_node')
                                child['node_evaluations']+=1
                        E=saved_ball(child['quadrature_error'])
                        value=engine.point(hz)*engine.point(ht)*value+acb(arb(0,E),arb(0,E))
                        if not value.is_finite() or not value.imag.contains(0):raise engine.NonanalyticImage('nonreal_cell_integral',value)
                        lo,hi=engine.rational(value.real.lower()),engine.rational(value.real.upper())
                        q=2*number(child['quadrature_error']['upper_exact'])
                        if hi-lo<q:raise ArithmeticError('Actual Gauss width below saved Cauchy allowance')
                        child['gauss_real'],child['gauss_imaginary']=engine.interval(value.real),engine.interval(value.imag)
                        if hi-lo<ends(child['real'])[1]-ends(child['real'])[0]:
                            child.update(method='tensor_gauss_separate_images',real=child['gauss_real'],imaginary=child['gauss_imaginary'])
                    except engine.NonanalyticImage as error:
                        child['node_refusal']=dict(reason=error.reason,image=error.image,input_image=error.input_image)
                sums=[sum((ends(x[key])[j] for x in working),F(0)) for key in ('real','imaginary') for j in (0,1)]
                entry['node_child_sum']=sums
                commit_frontier(entry,working,sums)
                report['completed_parent_node_count']+=1
                checkpoint(report['completed_parent_node_count'])
        if not frontier[2]<=0<=frontier[3]:raise ArithmeticError('Nonreal frontier')
        width=frontier[1]-frontier[0]
        if width<floor:raise ArithmeticError('Frontier contradicts fixed-allowance lower width floor')
        if width<=B_run:
            report['results']['eta110_y1_25']=engine.rational_interval(frontier[0],frontier[1]+omission)
            report['status']='printed_target_met' if width<=B_print else 'original_target_only'
        else:
            report['status']='incomplete_one_generation_floor' if floor>B_run else 'incomplete_one_generation_width'
        report['retained_width']=width
    except Exception as error:
        report['status']='incomplete'
        report['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(120,120));persist()
    print(json.dumps(dict(status=report['status'],counts=report['counts'],attempted_children=report['attempted_child_count'])))
    if report['status'] not in ('printed_target_met','original_target_only'):raise SystemExit(1)


if __name__=='__main__':main()
