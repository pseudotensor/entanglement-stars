"""Exact attribution of the completed saved T9 floor; no integrand imports."""
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

OLD = Path('<internal>/environment_continuation')
ROOT = OLD/'separate_images_v1'
SOURCE = ROOT/'separate_images_result.json'
PRIOR = OLD/'fixed_mass_sharp_repair_result.json'
AUDIT = OLD/'saved_110_width_attribution.json'
PINS = {SOURCE: '276090a9343faa76b34710c78f3a2d56ebe71357b42e1e96fd6c794d2972d09f',
        PRIOR: '0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e',
        AUDIT: '10470fc19620d3b3259176769f2caca1836f9fe634f675d475fe83c3ad9758ac'}
PRODUCER_SHA = 'bdc538b97df321ae42e92579fdad94142974322e36a9c8e56169c8523ed650b7'
EXPECTED_F = F(770678303515242634977384944594654624568596871032262939522121219985821,
               13803492693581127574869511724554050904902217944340773110325048447598592)
LIMIT = 8*2**20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(x):
    return F(int(x['numerator']), int(x['denominator']))


def ends(x):
    lo, hi = (rational(x[k]) for k in ('lower_exact', 'upper_exact'))
    require(lo <= hi, 'Reversed endpoints')
    return lo, hi


def point(x):
    lo, hi = ends(x)
    require(lo == hi and lo >= 0, 'Nonnegative point bound required')
    return lo


def vector(record):
    return [*ends(record['real']), *ends(record['imaginary'])]


def width(record):
    lo, hi = ends(record['real'])
    return hi-lo


def pack(x):
    if isinstance(x, F):
        return str(x)
    if isinstance(x, dict):
        return {str(k): pack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [pack(v) for v in x]
    return x


def fixtures():
    require(min(F(1,4), 4*F(1,8)) == F(1,4), 'Active-cap fixture')
    require(4*F(1,8)-F(1,4) == F(1,4), 'Cap discount fixture')
    require(min(F(1,2), F(2,5)+F(3,10)) == F(1,2), 'Second cap fixture')
    require(min(F(1,2), F(3,10)) == F(3,10), 'Removal fixture one')
    require(min(F(1,2), F(2,5)) == F(2,5), 'Removal fixture two')
    require(min(F(2), F(2,5)+F(3,10)) == F(7,10), 'Inactive-cap fixture')
    children = [('refused:p', 'refused:mass')]*4
    require(len(children) == 4 and sum(len(c) for c in children) == 8,
            'Overlapping image-refusal fixture')
    rp, ap, wp = F(1,8), F(0), F(1,4)
    require(min(wp, rp+ap) == F(1,8) and min(wp, ap) == 0
            and min(wp, rp) == F(1,8), 'Class-removal fixture')
    require(sum((min(F(1), F(0)) for _ in range(4)), F(0)) == 0,
            'All-zero floor fixture')


def main():
    require(Path(__file__).resolve().parent == ROOT, 'Named box directory required')
    target = ROOT/'saved_t9_floor_attribution.json'
    temporary = target.with_suffix('.json.tmp')
    if target.exists() or temporary.exists():
        raise FileExistsError('Existing attribution is never overwritten or retried')
    with target.open('x') as stream:
        stream.write('{"status":"running","scientific_evaluations":0}\n')
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [allowed[8] if len(allowed)>8 else allowed[0]])
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (2*2**30, 2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (55, 60))
    def deadline(signum, frame):
        raise TimeoutError('Read-only attribution allocation deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGXCPU, deadline)
    signal.alarm(85)
    start = time.monotonic()
    report = dict(status='incomplete', scientific_evaluations=0,
                  scope='Exact saved T9 floor and cover attribution only',
                  image_order=['separate_z_image','separate_t_image'],
                  source_pins={str(p): h for p, h in PINS.items()},
                  parents=[], children=[])
    try:
        fixtures()
        data = {}
        for path, expected in PINS.items():
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during read')
            require(hashlib.sha256(raw).hexdigest() == expected, 'Input hash changed')
            data[path] = json.loads(raw)
        source, prior, audit = (data[p] for p in (SOURCE, PRIOR, AUDIT))
        require(source['status'] == 'incomplete_one_generation_floor' and source['y'] == '1/25',
                'Unexpected generation state')
        require(source['producer_sha256'] == PRODUCER_SHA and audit['status'] == 'pass',
                'Producer/audit identity mismatch')
        require(source['source_pins'][str(PRIOR)] == PINS[PRIOR]
                and source['source_pins'][str(AUDIT)] == PINS[AUDIT], 'Generation inputs changed')
        require(prior['status'] == 'incomplete' and prior['y'] == '1/25', 'Old source state')
        old = prior['active_panel']
        old_leaves = {p['id']: p for p in old['leaves']}
        require(len(old_leaves) == 4241, 'Old IDs duplicated')
        fallback = {k:v for k,v in old_leaves.items() if v['method'] == 'nonnegative_real_bound'}
        fixed = {k:v for k,v in old_leaves.items() if v['method'] == 'tensor_gauss'}
        require(len(fallback) == 443 and len(fixed) == 3798, 'Unexpected old leaf partition')
        require(len(source['unchanged_gauss_source_ids']) == 3798
                and set(source['unchanged_gauss_source_ids']) == set(fixed), 'Fixed Gauss IDs changed')
        parents = source['parents']
        parent_ids = [p['source_parent_id'] for p in parents]
        require(len(parents) == 443 and set(parent_ids) == set(fallback), 'Parent IDs changed')
        require(source['completed_parent_modulus_count'] == 443
                and source['attempted_child_count'] == 1772
                and source['completed_parent_node_count'] == 0, 'Incomplete generation counts')
        require(len(source['replaced_parent_ids']) == 443
                and set(source['replaced_parent_ids']) == set(fallback), 'Replacement IDs changed')
        G = sum((width(p) for p in fixed.values()), F(0))
        WF = sum((width(p) for p in fallback.values()), F(0))
        omission = ends(prior['omitted_integral_bounds']['eta110_nonnegative'])[1]
        budgets = dict(run=min(F(1,10000), F(1,5000)-omission),
                       printed=F(8,1000000)-omission)
        require(G == rational(source['fixed_gauss_width'])
                and WF == rational(source['old_fallback_width'])
                and omission == rational(source['omitted_upper']), 'Saved aggregate inputs differ')
        require(budgets['run'] == rational(source['retained_run_budget'])
                and budgets['printed'] == rational(source['retained_print_budget']), 'Budget mismatch')
        old_totals = [sum((vector(p)[j] for p in old_leaves.values()), F(0)) for j in range(4)]
        require(old_totals == [*ends(old['real']), *ends(old['imaginary'])], 'Old four totals differ')
        require(old_totals == [rational(x) for x in audit['real_endpoints']+audit['imaginary_endpoints']],
                'Passed audit four totals differ')
        require(G == rational(audit['categories']['tensor_gauss']['width'])
                and WF == rational(audit['categories']['nonnegative_real_bound']['width'])
                and budgets['run'] == rational(audit['retained_budget_run'])
                and budgets['printed'] == rational(audit['retained_budget_print'])
                and ends(prior['omitted_integral_bounds']['eta110_nonnegative']) == tuple(map(rational,audit['omitted_interval'])),
                'Passed audit width/omission/budget mismatch')
        frontier = old_totals[:]
        signatures = defaultdict(lambda: dict(count=0, uncapped_minimum_sum=F(0)))
        histogram = Counter()
        child_ids = set()
        evaluations = 0
        for parent in parents:
            identity = parent['source_parent_id']
            original = fallback[identity]
            require(parent['old_bounds'] == original['bounds_z_t']
                    and ends(parent['old_real']) == ends(original['real'])
                    and ends(parent['old_imaginary']) == ends(original['imaginary']), 'Parent source changed')
            require(parent['frontier_committed'] is True, 'Uncommitted parent')
            a,b,c,d = map(F, parent['old_bounds'])
            mz,mt = (a+b)/2, (c+d)/2
            expected = {(a,mz,c,mt),(a,mz,mt,d),(mz,b,c,mt),(mz,b,mt,d)}
            children = parent['children']
            require(len(children) == 4 and {tuple(map(F,x['bounds_z_t'])) for x in children} == expected,
                    'Midpoint cover changed')
            sums = [sum((vector(ch)[j] for ch in children), F(0)) for j in range(4)]
            require(sums == [rational(x) for x in parent['complete_child_sum']], 'Child four sums differ')
            wp = width(original)
            require(sums[1]-sums[0] < wp, 'Fallback replacement not strictly narrower')
            frontier = [v-o+n for v,o,n in zip(frontier, vector(original), sums)]
            rp = ap = F(0)
            child_rows = []
            for child in children:
                cid = child['id']
                require(type(cid) is int and cid not in child_ids and cid not in old_leaves, 'Child ID collision')
                child_ids.add(cid)
                require(child['method'] == 'nonnegative_real_bound' and child['node_evaluations'] == 0,
                        'Unexpected child method/node work')
                lo,hi,ilo,ihi = vector(child)
                require(ilo <= 0 <= ihi, 'Nonreal child')
                r = hi-lo
                bounds = tuple(map(F,child['bounds_z_t']))
                x0,x1,t0,t1 = bounds
                require(x0<x1 and t0<t1, 'Degenerate child rectangle')
                images = child['images']
                require(set(images) == {'separate_z_image','separate_t_image'}, 'Missing image')
                signature=[]; moduli=[]; refused=False
                for name in ('separate_z_image','separate_t_image'):
                    image = images[name]; evaluations += 1
                    if image['status'] == 'pass':
                        moduli.append(point(image['M'])); signature.append('pass')
                    else:
                        require(image['status'] == 'refused' and isinstance(image['guard_reason'],str), 'Unknown image state')
                        reason=image['guard_reason']; histogram[reason]+=1
                        signature.append('refused:'+reason); refused=True
                q = None
                if refused:
                    lam=r; rp+=lam; prune=True; limiting='fallback_refused'
                else:
                    e=point(child['quadrature_error'])
                    require(e >= F(8,3)*F(4)**-16*(x1-x0)*(t1-t0)*sum(moduli,F(0)),
                            'Saved separate-image allowance too small')
                    q=2*e; lam=min(r,q); ap+=lam; prune=q>=r
                    limiting='allowance' if q<r else 'fallback' if r<q else 'tie'
                require(lam == rational(child['minimum_possible_width'])
                        and child['node_pruned_no_width_gain'] is prune, 'Child minimum/prune mismatch')
                key='|'.join(signature)+'|'+limiting
                signatures[key]['count']+=1; signatures[key]['uncapped_minimum_sum']+=lam
                child_rows.append(dict(id=cid,parent_id=identity,bounds=bounds,
                    half_widths=((x1-x0)/2,(t1-t0)/2),fallback_width=r,
                    twice_allowance=q,minimum=lam,images=signature,limiting=limiting,
                    necessary_change={k:G+min(wp,lam)>B for k,B in budgets.items()}))
            sp=rp+ap; cp=min(wp,sp)
            report['parents'].append(dict(id=identity,bounds=(a,b,c,d),old_width=wp,
                refused_minima=rp,passed_minima=ap,uncapped=sp,contribution=cp,
                cap_discount=sp-cp,fallback_children_width=sums[1]-sums[0],
                child_ids=[ch['id'] for ch in sorted(child_rows,key=lambda x:(-x['minimum'],x['id']))]))
            report['children'].extend(sorted(child_rows,key=lambda x:(-x['minimum'],x['id'])))
        require(len(child_ids)==1772 and evaluations==3544, 'Incomplete child/image coverage')
        require(histogram == Counter(nonpositive_real_p=740,nonpositive_real_mass_squared=20,epsilon_real_guard=6),
                'Image refusal histogram differs')
        require(source['counts']['evaluations']==evaluations and source['counts']['refused_images']==sum(histogram.values())
                and source['counts']['refusals_by_reason']==dict(histogram), 'Source count mismatch')
        require(frontier == [rational(x) for x in source['frontier_real']+source['frontier_imaginary']],
                'New four frontier totals differ')
        require(frontier[2]<=0<=frontier[3] and frontier[1]-frontier[0]==rational(source['retained_width']),
                'Retained width differs')
        floor=G+sum((p['contribution'] for p in report['parents']),F(0))
        require(floor==rational(source['pre_node_floor'])==EXPECTED_F, 'Stored T10 floor differs')
        require(all((floor>B)==source['pre_node_'+('run' if k=='run' else 'print')+'_target_impossible']
                    for k,B in budgets.items()), 'Strict target comparisons differ')
        totals={k:sum((p[k] for p in report['parents']),F(0))
                for k in ('refused_minima','passed_minima','cap_discount')}
        require(floor==G+totals['refused_minima']+totals['passed_minima']-totals['cap_discount'],
                'Attribution identity failed')
        require(totals['cap_discount']==0, 'Expected inactive parent caps not verified')
        class_floors=dict(refused_to_zero=G+sum((min(p['old_width'],p['passed_minima']) for p in report['parents']),F(0)),
                         passed_to_zero=G+sum((min(p['old_width'],p['refused_minima']) for p in report['parents']),F(0)))
        child_by_id={ch['id']:ch for ch in report['children']}
        for reason in ('nonpositive_real_p','nonpositive_real_mass_squared','epsilon_real_guard'):
            removed={ch['id'] for ch in report['children'] if 'refused:'+reason in ch['images']}
            class_floors[reason+'_children_to_zero']=G+sum((
                min(p['old_width'],sum((child_by_id[cid]['minimum'] for cid in p['child_ids']
                                      if cid not in removed),F(0)))
                for p in report['parents']),F(0))
        report['parents'].sort(key=lambda p:(-p['contribution'],p['id']))
        report.update(status='pass',controls='Exact fixtures, pinned sources, midpoint covers, four sums and T10 passed',
            fixed_gauss_width=G,old_fallback_width=WF,omission=omission,budgets=budgets,
            old_frontier=old_totals,new_frontier=frontier,retained_width=frontier[1]-frontier[0],
            floor=floor,attribution_totals=totals,all_parent_caps_inactive=all(p['cap_discount']==0 for p in report['parents']),
            joint_signatures=dict(signatures),refused_image_histogram=dict(histogram),
            idealized_class_floors=class_floors,
            idealized_class_target_obstruction={name:{k:v>B for k,B in budgets.items()} for name,v in class_floors.items()},
            counts=dict(parents=443,children=1772,image_records=3544,refused_images=sum(histogram.values()),node_evaluations=0))
    except Exception as error:
        report.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.alarm(0)
        resource.setrlimit(resource.RLIMIT_CPU,(60,60))
        report.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        encoded=(json.dumps(pack(report),separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(encoded)>LIMIT:
            report=dict(status='incomplete',error='Encoded diagnostic exceeds 8 MiB',
                encoded_bytes=len(encoded),scientific_evaluations=0,source_pins=report['source_pins'],
                checker_sha256=report['checker_sha256'])
            encoded=(json.dumps(report,separators=(',',':'))+'\n').encode()
        with temporary.open('xb') as stream:
            stream.write(encoded)
        temporary.replace(target)
    if report['status']!='pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
