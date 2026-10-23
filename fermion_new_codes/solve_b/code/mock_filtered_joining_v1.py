"""UNRUN exact #145 source/protection algebra; no propagation entry point.

Both joining routes retain the complete short/extended N2 sectors. Raw maps,
contexts, validation, manufactured controls and a bounded runner must still be
completed/reviewed before any numerical use of this source candidate.
"""
from fractions import Fraction as F
import run_mock_vertex_admission_v1 as e
import mock_filtered_vertex_v1 as flat
import mock_filtered_clockwell_v1 as clock


def canonical_terms(rows,modes):
    """Coalesce normal-ordered physical monomials, charging both CAR parities."""
    result={}
    for coefficient,ops in rows:
        coefficient=e.cc(coefficient);ops=tuple(ops)
        if len(ops) not in (2,4):raise ValueError('quadratic/quartic monomial required')
        half=len(ops)//2
        if any(type(j) is not int or not 0<=j<modes or type(d) is not bool for j,d in ops):
            raise ValueError('canonical mode/creation labels')
        if tuple(d for j,d in ops)!=(True,)*half+(False,)*half:raise ValueError('normal-ordered monomial required')
        creators=tuple(j for j,d in ops[:half]);annihilators=tuple(j for j,d in ops[half:])
        if len(set(creators))!=half or len(set(annihilators))!=half:continue
        inversions=sum(a>b for i,a in enumerate(creators) for b in creators[i+1:])
        inversions+=sum(a<b for i,a in enumerate(annihilators) for b in annihilators[i+1:])
        key=tuple((j,True) for j in sorted(creators))+tuple((j,False) for j in sorted(annihilators,reverse=True))
        value=result.get(key,e.ZERO)+((-1)**inversions)*coefficient
        if value:result[key]=value
        else:result.pop(key,None)
    return result


def target_terms(basis,U,mass=F(3,5)):
    flat.validate_U(U)
    rows=list(clock.free_terms(basis,mass))
    for j,k,value,ops in clock.weighted_rows(basis,clock.expanded_forward(basis)):
        rows.append((U*value,ops));rows.append((U*value.conj(),flat.adjoint_ops(ops)))
    return canonical_terms(rows,basis.modes)


def _columns(operators,basis):
    annihilators=tuple(j for j,creation in operators if not creation)
    if any(j>=basis.modes for j in annihilators):return ()
    required=sum(1<<j for j in annihilators)
    return tuple(i for i,mask in enumerate(basis.states) if mask&required==required)


def monomial_matrix(basis,terms):
    index={mask:i for i,mask in enumerate(basis.states)};matrix={}
    for operators,value in terms.items():
        for column in _columns(operators,basis):
            final,sign=flat.car(basis.states[column],operators,basis.modes)
            if sign:
                if final not in index:raise ValueError('monomial leaves declared full sector')
                e.add_entry(matrix,(index[final],column),value*sign)
    return matrix


def _subtract(a,b):
    result=dict(a)
    for key,value in b.items():
        z=result.get(key,e.ZERO)-value
        if z:result[key]=z
        else:result.pop(key,None)
    return result


def _gram(matrix,rows):
    """Exact rectangular A^dag A, retaining all same-row interference."""
    groups={}
    for (i,j),value in matrix.items():
        if not 0<=i<rows:raise ValueError('rectangular row locator')
        groups.setdefault(i,[]).append((j,value))
    result={}
    for entries in groups.values():
        for i,a in entries:
            for j,b in entries:e.add_entry(result,(i,j),a.conj()*b)
    return result


def tested_joining(short,U):
    if short.role!='weighted' or short.n not in (2,3):raise ValueError('joining source only for n2/n3')
    flat.validate_U(U);long=clock.StaticBasis.make(short.n+4,'joining_reference')
    small=target_terms(short,U);large=target_terms(long,U)
    inside={ops:z for ops,z in large.items() if all(mode<short.modes for mode,creation in ops)}
    internal_difference=_subtract(inside,small)
    grouped={};inventory=[]
    lower_bases={0:(0,),1:tuple(1<<j for j in range(short.modes))}
    lower_indices={number:{mask:i for i,mask in enumerate(states)} for number,states in lower_bases.items()}
    for operators,value in large.items():
        outside=tuple((mode,creation) for mode,creation in operators if mode>=short.modes)
        if not outside:continue
        survives=all(creation for mode,creation in outside)
        inventory.append(dict(operators=operators,coefficient=value,outside=outside,vacuum_survives=survives))
        if not survives:continue
        P=tuple(sorted(mode for mode,creation in outside))
        if len(P) not in (1,2) or any(mode>=long.modes for mode in P):raise ValueError('joining support/pattern')
        number=2-len(P);O=grouped.setdefault(P,{})
        for column in _columns(operators,short):
            final,sign=flat.car(short.states[column],operators,long.modes)
            if not sign:continue
            actual=tuple(j for j in range(short.modes,long.modes) if final>>j&1)
            if actual!=P:raise ValueError('outside canonical factorization')
            inner=final&((1<<short.modes)-1)
            if inner not in lower_indices[number]:raise ValueError('complete lower-number target basis')
            e.add_entry(O,(lower_indices[number][inner],column),value*sign)
    grouped={P:matrix for P,matrix in grouped.items() if matrix}
    long_index={mask:i for i,mask in enumerate(long.states)};S={};Q={}
    for P,O in grouped.items():
        number=2-len(P);outside=sum(1<<j for j in P)
        for (row,column),value in O.items():
            final=lower_bases[number][row]|outside
            e.add_entry(S,(long_index[final],column),value)
        Q=e.matrix_add(Q,_gram(O,len(lower_bases[number])))
    return dict(short=short,long=long,U=U,short_terms=small,long_terms=large,
        internal_difference=internal_difference,patterns=grouped,pattern_target_bases=lower_bases,
        S=S,Q=Q,outside_inventory=inventory,empty_extension=True,added_sites=4,
        profile_short=clock.profile(short),profile_long=clock.profile(long))


def independent_joining(short,U):
    """Complete independently assembled sparse H_long J - J H_short."""
    if short.role!='weighted' or short.n not in (2,3):raise ValueError('reference short size/role')
    flat.validate_U(U);long=clock.StaticBasis.make(short.n+4,'joining_reference')
    Ws=clock.independent_vertex(short);Wl=clock.independent_vertex(long)
    Hs=e.matrix_add(clock.free_independent(short,F(3,5)),Ws['matrix'],U)
    Hl=e.matrix_add(clock.free_independent(long,F(3,5)),Wl['matrix'],U)
    index={mask:i for i,mask in enumerate(long.states)}
    embedding=tuple(index[mask] for mask in short.states);inverse={j:i for i,j in enumerate(embedding)}
    S={}
    for (i,j),value in Hl.items():
        if j in inverse:e.add_entry(S,(i,inverse[j]),value)
    for (i,j),value in Hs.items():e.add_entry(S,(embedding[i],j),-value)
    Q=_gram(S,long.dimension)
    return dict(short=short,long=long,U=U,H_short=Hs,H_long=Hl,embedding=embedding,S=S,Q=Q,
        profile_short=clock.profile(short),profile_long=clock.profile(long),empty_extension=True,added_sites=4)


def protected_terms(terms,collected,modes):
    groups=[];seen=set()
    for block in collected:
        if any(type(j) is not int or not 0<=j<modes for j in block):raise ValueError('protected physical modes')
        group=set(block)
        if not group or len(group)!=len(block) or group&seen:raise ValueError('protected partition prefix')
        seen|=group;groups.append(group)
    kept={};removed={}
    for operators,value in terms.items():
        support={j for j,creation in operators}
        crossing=any(support&group and not support<=group for group in groups)
        (removed if crossing else kept)[operators]=value
    return dict(kept=kept,removed=removed,collected=tuple(tuple(block) for block in collected))


def projected_source(tested,reference,added):
    if type(added) is not int or added not in (1,2,3):raise ValueError('fixed shorter extension projection')
    short=tested['short'];long=tested['long'];limit=8*(short.n+added)
    index={mask:i for i,mask in enumerate(long.states)};S={};selected={}
    lower=tested['pattern_target_bases']
    for P,O in tested['patterns'].items():
        if any(j>=limit for j in P):continue
        selected[P]=O;outside=sum(1<<j for j in P)
        for (i,j),z in O.items():e.add_entry(S,(index[lower[2-len(P)][i]|outside],j),z)
    allowed=tuple(i for i,mask in enumerate(long.states) if mask>>limit==0)
    allowed_set=set(allowed);independent={key:z for key,z in reference['S'].items() if key[0] in allowed_set}
    return dict(added_sites=added,patterns=selected,retained_full_row_indices=allowed,
        retained_full_row_masks=tuple(long.states[i] for i in allowed),S=S,independent_S=independent,
        Q=_gram(S,long.dimension),independent_Q=_gram(independent,long.dimension))


def source_support_inventory(tested):
    rows=[]
    for term in tested['outside_inventory']:
        ops=term['operators'];old_ann=tuple(j//8 for j,d in ops if not d and j<tested['short'].modes)
        created=tuple(j//8 for j,d in ops if d)
        survives=term['vacuum_survives']
        good=(not survives) or (bool(old_ann) and max(created)<=max(old_ann)+4)
        rows.append(dict(operators=ops,vacuum_survives=survives,old_annihilation_sites=old_ann,
            creation_sites=created,range_four_verified=good))
    return dict(rows=rows,all_range_four=all(row['range_four_verified'] for row in rows),
        fourth_added_site_present=any(term['vacuum_survives'] and any(j//8==tested['short'].n+3 for j,d in term['outside'] if d) for term in tested['outside_inventory']))
