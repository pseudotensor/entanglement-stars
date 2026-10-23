#!/usr/bin/env python3
"""Second lossless literal view: shared schemas, scalars and check membership.

No science imports/evaluation. K defines ordered object keys; L a literal
subtree; G a check's bit/detail and full ordered names; R a non-check row.
O restores exact row order using check-group runs and non-check row counts.
Candidate bytes must reconstruct the complete original bytes before writing.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json


def enc(obj):return json.dumps(obj,separators=(',',':'),allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('output');args=p.parse_args()
    source=Path(args.input);out=Path(args.output)
    if out.exists():raise FileExistsError('never overwrite any preceding view')
    raw=source.read_bytes();rows=[json.loads(x) for x in raw.decode().splitlines()]
    if raw!=(''.join(enc(x)+'\n' for x in rows)).encode():raise ValueError('not complete canonical newline input')
    if not rows or rows[-1]['kind']!='termination':raise ValueError('unfinished input')
    counts=Counter();objects={};schemas={}
    def scan(obj):
        if isinstance(obj,dict):
            if '@' in obj or '!' in obj:raise ValueError('reserved literal encoding key in original')
            keys=tuple(obj)
            if keys not in schemas:schemas[keys]=len(schemas)
            for child in obj.values():scan(child)
        elif isinstance(obj,list):
            for child in obj:scan(child)
        elif not isinstance(obj,str):return
        key=enc(obj)
        if len(key)>=48:counts[key]+=1;objects[key]=obj
    for row in rows:scan(row)
    keys=sorted((k for k,n in counts.items() if n>=2 and (n-1)*(len(k)-10)>len(k)+24),key=lambda k:(len(k),k))
    ids={key:i for i,key in enumerate(keys)}
    def pack(obj,skip=None):
        if not isinstance(obj,(dict,list,str)):return obj
        key=enc(obj)
        if key!=skip and key in ids:return {'@':ids[key]}
        if isinstance(obj,dict):return {'!':[schemas[tuple(obj)]]+[pack(v) for v in obj.values()]}
        if isinstance(obj,list):return [pack(v) for v in obj]
        return obj
    groups=[];group_ids={};other=[];order=[]
    for row in rows:
        if list(row)==['kind','data'] and row['kind']=='check' and list(row['data'])==['name','pass','detail']:
            d=row['data'];key=enc([d['pass'],d['detail']])
            if key not in group_ids:group_ids[key]=len(groups);groups.append([d['pass'],d['detail'],[]])
            gid=group_ids[key];groups[gid][2].append(d['name'])
            if not order or order[-1][0]!='C':order.append(['C',[]])
            runs=order[-1][1]
            if runs and runs[-1][0]==gid:runs[-1][1]+=1
            else:runs.append([gid,1])
        else:
            other.append(row)
            if order and order[-1][0]=='R':order[-1][1]+=1
            else:order.append(['R',1])
    header={'original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),'original_rows':len(rows),
        'schemas':len(schemas),'literals':len(keys),'check_groups':len(groups),'checks':sum(len(g[2]) for g in groups),
        'rule':'+ continues the preceding literal line; join before decoding. K ordered keys; L literal; G [pass,detail,complete ordered names]; R row; O original row order. @ literal reference; ! schema and ordered values. No value is omitted.'}
    view=['# Lossless complete controls view v2\n','H '+enc(header)+'\n']
    for key,i in schemas.items():view.append('K '+enc([i,list(key)])+'\n')
    for i,key in enumerate(keys):view.append('L '+enc([i,pack(objects[key],key)])+'\n')
    for i,(bit,detail,names) in enumerate(groups):view.append('G '+enc([i,bit,pack(detail),names])+'\n')
    for row in other:view.append('R '+enc(pack(row))+'\n')
    for entry in order:view.append('O '+enc(entry)+'\n')
    # Bound display width without changing one byte of a logical JSON line.
    wrapped=view[:2]
    for line in view[2:]:
        body=line[:-1]
        for start in range(0,len(body),180):wrapped.append(('' if start==0 else '+ ')+body[start:start+180]+'\n')
    candidate=''.join(wrapped).encode();ks={};ls={};gs={};rs=[];os=[]
    logical=[]
    for line in candidate.decode().splitlines()[2:]:
        if line.startswith('+ '):
            if not logical:raise ValueError('orphan continuation')
            logical[-1]+=line[2:]
        else:logical.append(line)
    # Independently parse the exact candidate which will be written.
    for line in logical:
        tag,payload=line.split(' ',1);obj=json.loads(payload)
        if tag=='K':ks[obj[0]]=obj[1]
        elif tag=='L':ls[obj[0]]=obj[1]
        elif tag=='G':gs[obj[0]]=obj[1:]
        elif tag=='R':rs.append(obj)
        elif tag=='O':os.append(obj)
        else:raise ValueError('unknown candidate tag')
    def expand(obj):
        if isinstance(obj,dict):
            if list(obj)==['@']:return expand(ls[obj['@']])
            if list(obj)==['!']:
                sid,*values=obj['!']
                if len(ks[sid])!=len(values):raise ValueError('schema arity')
                return dict(zip(ks[sid],[expand(v) for v in values]))
            raise ValueError('unknown encoded dictionary')
        if isinstance(obj,list):return [expand(v) for v in obj]
        return obj
    recovered=[];used=Counter();rindex=0
    for tag,payload in os:
        if tag=='R':
            for _ in range(payload):recovered.append(expand(rs[rindex]));rindex+=1
        elif tag=='C':
            for gid,count in payload:
                bit,detail,names=gs[gid]
                for _ in range(count):
                    name=names[used[gid]];used[gid]+=1
                    recovered.append({'kind':'check','data':{'name':name,'pass':bit,'detail':expand(detail)}})
        else:raise ValueError('unknown order tag')
    if rindex!=len(rs) or any(used[gid]!=len(data[2]) for gid,data in gs.items()):raise ValueError('unconsumed membership')
    if (''.join(enc(row)+'\n' for row in recovered)).encode()!=raw:raise ValueError('original byte reconstruction failed')
    with out.open('xb') as stream:stream.write(candidate)
    print(enc({'status':'literal_byte_reconstruction_PASS',**header,'output_bytes':len(candidate),
        'output_sha256':hashlib.sha256(candidate).hexdigest()}))

if __name__=='__main__':main()
