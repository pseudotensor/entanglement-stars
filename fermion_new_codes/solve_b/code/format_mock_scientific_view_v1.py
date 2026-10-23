#!/usr/bin/env python3
"""Literal JSON presentation only; never imports or evaluates scientific code.

Implements only MOCK_output_scientific_view_spec_v1.md. All originals stay
on the box; the source-bound view requires them for exact reconstruction.
"""
import hashlib
import json
import argparse
import copy
import re
from pathlib import Path

MARKER='_bulk_literal_reference'
DUPLICATE='_same_as_displayed'
SPEC_SHA='14effe0b00e0d032a165a660167f189572c686384d08b6e9fcc9140446f4211e'


def canonical(value):
    return json.dumps(value,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON object key')
        result[key]=value
    return result


def parse(line):return json.loads(line,object_pairs_hook=unique_object)


def pointer(path):
    return ''.join('/'+str(item).replace('~','~0').replace('/','~1') for item in path)


def dereference(record,text):
    if text=='':return record
    if not text.startswith('/'):raise ValueError('invalid JSON pointer')
    for part in text[1:].split('/'):
        if re.search(r'~(?![01])',part):raise ValueError('invalid JSON pointer escape')
        part=part.replace('~1','/').replace('~0','~')
        if isinstance(record,list) and re.fullmatch(r'0|[1-9][0-9]*',part) is None:raise ValueError('invalid JSON array index')
        record=record[int(part)] if isinstance(record,list) else record[part]
    return record


def shape(value):
    if not isinstance(value,list):return None
    dims=[len(value)];level=value
    while level and all(isinstance(x,list) for x in level):
        lengths={len(x) for x in level}
        if len(lengths)!=1:break
        length=next(iter(lengths));dims.append(length)
        if not length:break
        level=[y for x in level for y in x]
    return dims


def ancestor(record,path):
    for item in path:record=record[item]
    return record


def selection(family,record,path,value):
    if not path:return None
    key=path[-1];parent=ancestor(record,path[:-1]);kind=record.get('kind');large=isinstance(value,(list,dict)) and len(value)>8
    if family=='baseline' and large:
        if key=='phase_component_errors' and isinstance(value,list) and 'budget' in path[:-1]:return '3.1.1 per-eigenmode component array'
        if key=='row_keys' and isinstance(parent,dict) and {'dimension','modes','register'}<=set(parent) and ('density' in parent or 'matrix' in parent):return '3.1.2 reduced-density row map'
        protected=isinstance(parent,dict) and parent.get('kind')=='protected' and {'plan','operator','groups','panels'}<=set(parent)
        copied_model=isinstance(parent,dict) and {'metadata','actual_H','terms'}<=set(parent)
        if key=='terms' and (protected or copied_model or path==('data','terms') and kind in ('assembled_model_before_audit','model')):return '3.1.3 ordered canonical monomials'
        matrices=('observed','independent_actual','intended','represented_input_lift','assembly_difference','mass_difference','total_target_difference','pair_W')
        if isinstance(value,dict) and (kind=='independent_matrix_before_audit' and len(path)==2 and path[0]=='data' and key in matrices or kind=='model' and path==('data','mass_difference')):return '3.1.4 exact matrix components'
    if family=='static' and large and path[0]=='data':
        relative=path[1:]
        if kind=='actual_terms' and relative in (('rows',),('basis',),('pairs',)):return '3.2 static terms/basis components'
        if kind in ('tested_vertex','Hamiltonian') and relative in (('basis',),('pairs',)):return '3.2 static canonical basis components'
        if kind=='exact_cached_actions' and (relative in (('input',),('output',)) or len(relative)==2 and relative[0]=='centres'):return '3.2 complete cached action vector'
        if kind=='ordinary_inputs' and (relative in (('wedgeH',),('wedgeL',)) or len(relative)==2 and relative[0]=='legs' and type(relative[1]) is int):return '3.2 ordinary input components'
        if kind=='tangent_inputs' and (relative==('raw_wedge_derivative',) or len(relative)==2 and relative[0] in ('heavy','light','heavy_derivative','light_derivative') and type(relative[1]) is int):return '3.2 full tangent input components'
    if family!='baseline' or not isinstance(value,str) or len(value)<=96 or re.fullmatch(r'[+-]?[0-9]+(?:/[1-9][0-9]*)?',value) is None:return None
    if key in ('numerator','norm_squared','normalized_exact') and len(path)>=2 and path[-2] in ('target_energy','target_start','target_end'):return '4 exact energy intermediate referenced'
    if key=='raw_norm_squared' and isinstance(parent,dict) and {'energy_start','energy_end'}<=set(parent):return '4 exact action-norm intermediate referenced'
    if key=='norm_exact' and kind=='norm_energy' and path==('data','norm_exact'):return '4 exact diagnostic norm intermediate referenced'
    if key=='exact_squared' and len(path)>=2 and path[-2] in ('C_HL','cross_norm'):return '4 exact cross-norm intermediate referenced'
    if key in ('re','im') and len(path)>=3 and path[-2]=='exact' and any(x in ('z','site_z') for x in path[:-2]):return '4 exact witness intermediate referenced'
    data=record.get('data',{})
    if kind=='check' and isinstance(data,dict) and str(data.get('name','')).startswith('pure_cross_norm_identity_') and path in (('data','detail','C_squared'),('data','detail','product')):return '4 exact cross-identity intermediate referenced'
    if key=='_squared' and 'computed_distance' in path[:-1] and isinstance(parent,dict) and {'lower','upper','norm_a_squared','norm_b_squared'}<=set(parent):return '4 exact normalized distance intermediate referenced'
    return None


def reference(source,source_sha,line,record,path,value,rule):
    ref=dict(source_report=str(source),source_sha256=source_sha,line=line,kind=record['kind'],pointer=pointer(path),
        subtree_sha256=digest(value),type=type(value).__name__,count=len(value),shape=shape(value),rule=rule,scientific_role=rule)
    if isinstance(value,str):
        parts=value.lstrip('+-').split('/')
        ref['rational_lexical']=dict(sign='-' if value.startswith('-') else '+',numerator_characters=len(parts[0]),denominator_characters=len(parts[1]) if len(parts)==2 else 0,
            display='exact intermediate referenced')
    return ref


def transform(record,source,source_sha,line,family,inventory,displayed):
    def visit(value,path=()):
        if isinstance(value,dict) and (MARKER in value or DUPLICATE in value):raise ValueError('reserved reference marker in original report')
        rule=selection(family,record,path,value)
        if rule:
            ref=reference(source,source_sha,line,record,path,value,rule);inventory.append(ref)
            return {MARKER:ref}
        eligible=isinstance(value,(dict,list)) and path not in ((),('data',))
        encoded=canonical(value) if eligible else b''
        identity=hashlib.sha256(encoded).hexdigest() if len(encoded)>=512 else None
        if identity:
            for oldbytes,oldline,oldpath in displayed.get(identity,()):
                if oldline<line and oldbytes==encoded:
                    ref=reference(source,source_sha,line,record,path,value,'5 identical contextual semantic copy')
                    ref.update(displayed_line=oldline,displayed_pointer=pointer(oldpath),displayed_subtree_sha256=identity)
                    inventory.append(ref);return {DUPLICATE:ref}
        if isinstance(value,dict):result={k:visit(v,path+(k,)) for k,v in value.items()}
        elif isinstance(value,list):result=[visit(v,path+(j,)) for j,v in enumerate(value)]
        else:result=value
        if identity and not any(oldbytes==encoded for oldbytes,oldline,oldpath in displayed.get(identity,())):
            displayed.setdefault(identity,[]).append((encoded,line,path))
        return result
    return visit(record)


def reconstruct(view,original_rows,view_rows,source,source_sha,line,family,inventory):
    def original(n):return parse(original_rows[n-1])
    def visit(value,at_line,path):
        if isinstance(value,dict) and (MARKER in value or DUPLICATE in value):
            marker=MARKER if MARKER in value else DUPLICATE
            if set(value)!={marker}:raise ValueError('ambiguous reference object')
            ref=value[marker]
            if (ref['source_report']!=str(source) or ref['source_sha256']!=source_sha or ref['line']!=at_line):raise ValueError('reference source/line mismatch')
            if ref['pointer']!=pointer(path):raise ValueError('reference pointer differs from candidate location')
            record=original(at_line);raw=dereference(record,ref['pointer'])
            wanted=selection(family,record,tuple_pointer(record,ref['pointer']),raw) if marker==MARKER else '5 identical contextual semantic copy'
            expected=reference(source,source_sha,at_line,record,tuple_pointer(record,ref['pointer']),raw,wanted)
            if marker==DUPLICATE:
                oldline=ref['displayed_line'];oldpointer=ref['displayed_pointer']
                if type(oldline) is not int or not 1<=oldline<at_line:raise ValueError('forward/cyclic displayed link')
                if not isinstance(raw,(dict,list)) or len(canonical(raw))<512:raise ValueError('invalid contextual copy')
                if tuple_pointer(record,ref['pointer']) in ((),('data',)):raise ValueError('record root copy prohibited')
                previous=dereference(original(oldline),oldpointer)
                if canonical(raw)!=canonical(previous):raise ValueError('contextual copy original bytes differ')
                shown=dereference(view_rows[oldline-1]['record'],oldpointer)
                if isinstance(shown,dict) and (MARKER in shown or DUPLICATE in shown):raise ValueError('first contextual display was itself a reference')
                expanded=visit(shown,oldline,tuple_pointer(original(oldline),oldpointer))
                if canonical(expanded)!=canonical(raw):raise ValueError('prior contextual display mismatch')
                expected.update(displayed_line=oldline,displayed_pointer=oldpointer,displayed_subtree_sha256=digest(raw))
            if wanted is None or expected!=ref:raise ValueError('reference rule/type/count/shape/hash mismatch')
            inventory.append(ref)
            return raw
        if isinstance(value,dict):return {k:visit(v,at_line,path+(k,)) for k,v in value.items()}
        if isinstance(value,list):return [visit(v,at_line,path+(j,)) for j,v in enumerate(value)]
        return value
    return visit(view,line,())


def tuple_pointer(record,text):
    if text=='':return ()
    if not text.startswith('/'):raise ValueError('invalid JSON pointer')
    result=[]
    for part in text[1:].split('/'):
        if re.search(r'~(?![01])',part):raise ValueError('invalid JSON pointer escape')
        part=part.replace('~1','/').replace('~0','~')
        if isinstance(record,list) and re.fullmatch(r'0|[1-9][0-9]*',part) is None:raise ValueError('invalid JSON array index')
        key=int(part) if isinstance(record,list) else part
        result.append(key);record=record[key]
    return tuple(result)


def wrap(item):
    text=canonical(item).decode();chunks=[text[j:j+178] for j in range(0,len(text),178)]
    return (chunks[0]+'\n'+''.join('+ '+chunk+'\n' for chunk in chunks[1:])).encode()


def unwrap(data):
    objects=[];current=''
    for line in data.decode().splitlines():
        if line.startswith('+ '):current+=line[2:]
        else:
            if current:objects.append(parse(current))
            current=line
    if current:objects.append(parse(current))
    return objects


def candidate(source,original_rows,source_sha,spec,spec_sha,family):
    inventory=[];displayed={};view_rows=[]
    for number,line in enumerate(original_rows,1):
        record=parse(line)
        view=transform(record,source,source_sha,number,family,inventory,displayed)
        view_rows.append(dict(source_line=number,record=view))
    header=dict(kind='scientific_view_header',source_report=str(source),source_sha256=source_sha,
        formatter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_bytes=sum(map(len,original_rows)),source_lines=len(original_rows),specification=str(spec),specification_sha256=spec_sha,
        family=family,continuation='A line prefixed + space continues the preceding JSON line; join before decoding.',
        scope='All original records and permitted semantic copies retained; only specified bulk components and baseline exact intermediates are source references. No numerical evaluation.',
        reference_inventory='Every inline reference object is one complete inventory entry. Original reports are required for reconstruction.')
    by_rule={}
    for ref in inventory:by_rule[ref['rule']]=by_rule.get(ref['rule'],0)+1
    footer=dict(kind='scientific_view_verification',records=len(view_rows),references=len(inventory),references_by_rule=by_rule,
        reference_inventory_sha256=digest(inventory),inventory_location='All inline reference objects, in original traversal order, form the complete inventory.',status='exact_original_byte_reconstruction_PASS',numerical_evaluation=False)
    data=b''.join(wrap(row) for row in [header,*view_rows,footer])
    decoded=unwrap(data)
    if decoded[0]!=header or decoded[-1]!=footer or len(decoded)!=len(original_rows)+2:raise ValueError('candidate framing failed')
    parsed_rows=decoded[1:-1];checked_inventory=[]
    for number,row in enumerate(parsed_rows,1):
        if set(row)!={'source_line','record'} or row['source_line']!=number:raise ValueError('row order/shape mismatch')
        per_row=[]
        restored=reconstruct(row['record'],original_rows,parsed_rows,source,source_sha,number,family,per_row)
        # Nested backward-display verification can revisit earlier reference entries;
        # the direct inventory is collected from the current candidate tree below.
        if canonical(restored)+b'\n'!=original_rows[number-1]:raise ValueError('exact original byte reconstruction failed')
        def direct(value):
            if isinstance(value,dict) and (MARKER in value or DUPLICATE in value):
                checked_inventory.append(value[MARKER if MARKER in value else DUPLICATE]);return
            if isinstance(value,dict):
                for child in value.values():direct(child)
            elif isinstance(value,list):
                for child in value:direct(child)
        direct(row['record'])
    if checked_inventory!=inventory:raise ValueError('reference inventory reconstruction failed')
    return data,footer


def build(source,output,spec,family):
    source=Path(source).resolve();output=Path(output).resolve();spec=Path(spec).resolve()
    if not source.is_relative_to(Path('<internal>/solve_b_bulk')) or output.parent!=source.parent:raise ValueError('box-only source and same-directory view')
    if output.exists():raise FileExistsError('never overwrite any view')
    formatter_bytes=Path(__file__).read_bytes()
    spec_bytes=spec.read_bytes();spec_sha=hashlib.sha256(spec_bytes).hexdigest()
    if spec_sha!=SPEC_SHA:raise ValueError('frozen scientific presentation specification')
    original=source.read_bytes();source_sha=hashlib.sha256(original).hexdigest()
    data,footer=candidate(source,original.splitlines(keepends=True),source_sha,spec,spec_sha,family)
    if source.read_bytes()!=original or spec.read_bytes()!=spec_bytes or Path(__file__).read_bytes()!=formatter_bytes:raise ValueError('original source/specification/formatter changed')
    with output.open('xb') as f:f.write(data)
    return dict(source_sha256=source_sha,output_sha256=hashlib.sha256(data).hexdigest(),output_bytes=len(data),**{k:v for k,v in footer.items() if k!='reference_inventory'})


def self_test(directory,spec):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(Path('<internal>/solve_b_bulk')):raise ValueError('box-only fixtures')
    directory.mkdir(exist_ok=False)
    long='-'+'1'*120+'/7';vector=[{'re':'0','im':'1'} for _ in range(12)]
    samples=[dict(index=j,clock=str(j)+'/8',upper=0.25,n_last_exact=long) for j in range(20)]
    duplicate=dict(context='complete first contextual payload',entries=['literal-'+str(j)+'x'*80 for j in range(10)])
    baseline=[dict(kind='norm_energy',data=dict(target_energy=dict(numerator=long,norm_squared=long,normalized_exact=long,lower=-1.,upper=1.,state_error=0.),norm_exact=long,norm_drift=0.)),
        dict(kind='purification',data=dict(integrands=samples,numerator=long,payload=duplicate)),
        dict(kind='check',data=dict(name='literal fixture check',**{'pass':True},reason='preserved',detail=dict(payload=duplicate))),
        dict(kind='action',data=dict(budget=dict(phase_component_errors=[0.1]*12,total=0.2,represented_time=1.)))]
    static=[dict(kind='ordinary_inputs',data=dict(legs=[vector,vector],wedgeH=vector,wedgeL=vector,centre_envelopes={str(j):long.replace('/7','/9') for j in range(12)},QH=long)),
        dict(kind='ordinary_contractions',data=dict(raw=long,expected=long,centres={str(j):long for j in range(12)})),
        dict(kind='tangent_contractions',data=dict(target=long,centres={str(j):long for j in range(12)}))]
    results=[];loaded={}
    def check(name,condition):
        results.append(dict(name=name,**{'pass':bool(condition)}))
        if not condition:raise AssertionError(name)
    for family,records in (('baseline',baseline),('static',static)):
        source=directory/(family+'_fixture_source.md');output=directory/(family+'_fixture_view.md')
        with source.open('xb') as f:f.write(b''.join(canonical(row)+b'\n' for row in records))
        result=build(source,output,spec,family);decoded=unwrap(output.read_bytes());rows=decoded[1:-1]
        check(family+'_byte_reconstruction',result['status']=='exact_original_byte_reconstruction_PASS')
        loaded[family]=(source,source.read_bytes().splitlines(keepends=True),rows,result['source_sha256'])
    br=loaded['baseline'][2];sr=loaded['static'][2]
    check('long_proof_samples_literal',br[1]['record']['data']['integrands']==samples)
    check('generic_numerator_literal',br[1]['record']['data']['numerator']==long)
    check('eligible_energy_reference',MARKER in br[0]['record']['data']['target_energy']['numerator'])
    check('static_first_envelopes_literal',sr[0]['record']['data']['centre_envelopes']==static[0]['data']['centre_envelopes'])
    check('static_centre_values_literal',sr[1]['record']['data']['centres']==static[1]['data']['centres'])
    check('static_long_targets_literal',sr[1]['record']['data']['raw']==long and sr[2]['record']['data']['target']==long)
    check('bulk_vector_reference',MARKER in sr[0]['record']['data']['legs'][0])
    check('outer_leg_indices_retained',len(sr[0]['record']['data']['legs'])==2)
    check('contextual_backward_link',DUPLICATE in br[2]['record']['data']['detail']['payload'])
    def reject(name,changed,line,family):
        source,original,rows,sha=loaded[family]
        try:reconstruct(changed,original,rows,source,sha,line,family,[])
        except (ValueError,KeyError,IndexError,TypeError):check(name,True)
        else:check(name,False)
    broken=copy.deepcopy(br[0]['record']);broken['data']['target_energy']['numerator'][MARKER]['pointer']='<internal>/norm_exact'
    reject('wrong_equal_value_pointer_rejected',broken,1,'baseline')
    broken=copy.deepcopy(sr[0]['record']);broken['data']['legs'][0][MARKER]['subtree_sha256']='0'*64
    reject('wrong_subtree_hash_rejected',broken,1,'static')
    broken=copy.deepcopy(br[2]['record']);broken['data']['detail']['payload'][DUPLICATE]['displayed_line']=3
    reject('forward_link_rejected',broken,3,'baseline')
    report=dict(scope='Artificial literal JSON formatting fixtures only; no scientific evaluation.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=results,all_pass=all(x['pass'] for x in results))
    with (directory/'fixture_report.md').open('xb') as f:f.write(wrap(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',nargs='?');parser.add_argument('output',nargs='?');parser.add_argument('--spec',required=True)
    parser.add_argument('--family',choices=('baseline','static'));parser.add_argument('--self-test')
    args=parser.parse_args()
    if args.self_test:
        if args.source or args.output or args.family:raise ValueError('fixtures are a separate operation')
        result=self_test(args.self_test,args.spec)
    else:
        if not args.source or not args.output or not args.family:raise ValueError('source/output/family required')
        result=build(args.source,args.output,args.spec,args.family)
    print(json.dumps(result,separators=(',',':')))


if __name__=='__main__':main()
