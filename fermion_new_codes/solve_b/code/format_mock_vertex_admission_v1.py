#!/usr/bin/env python3
"""Lossless text-only factorization of an existing written report; no numerics."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import defaultdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report');parser.add_argument('output')
    args=parser.parse_args()
    source=Path(args.report);target=Path(args.output)
    if target.exists():raise FileExistsError('Never overwrite a formatted report')
    raw=source.read_bytes();lines=raw.decode().splitlines(keepends=True)
    labels=[];checkgroups={};groups=[];probe_rows=[];tangent_rows=[];values=[];value_index={};events=[]
    for line in lines:
        if line.startswith('PROBE '):labels.append(line[6:line.index(' {')])
    label_index={v:i for i,v in enumerate(labels)}
    suffixes=[];suffix_index={}
    def value_id(v):
        encoded=json.dumps(v,separators=(',',':'))
        if encoded not in value_index:
            value_index[encoded]=len(values);values.append(encoded)
        return value_index[encoded]
    def parse_table(payload,rows):
        obj=json.loads(payload);keys=tuple(obj)
        row={'keys':keys,'values':[value_id(obj[k]) for k in keys]}
        rows.append(row);return len(rows)-1
    for line in lines:
        if line.startswith('CHECK '):
            body=line[6:].rstrip('\n')
            verdict='PASS' if ' PASS' in body else 'FAIL'
            name,sep,payload=body.partition(' '+verdict)
            payload=payload[1:] if payload.startswith(' ') else payload
            key=(verdict,payload)
            if key not in checkgroups:
                checkgroups[key]=len(groups);groups.append({'verdict':verdict,'payload':payload,'members':[]})
            gid=checkgroups[key];member=name
            matches=[v for v in labels if name.startswith(v+'_')]
            if matches:
                label=max(matches,key=len);suffix=name[len(label):]
                if suffix not in suffix_index:suffix_index[suffix]=len(suffixes);suffixes.append(suffix)
                member=('P',label_index[label],suffix_index[suffix])
            groups[gid]['members'].append(member);events.append(('C',gid,len(groups[gid]['members'])-1))
        elif line.startswith('PROBE '):
            marker=line.index(' {');label=line[6:marker];payload=line[marker+1:].rstrip('\n')
            idx=parse_table(payload,probe_rows);events.append(('P',label,idx))
        elif line.startswith('TANGENT '):
            idx=parse_table(line[8:].rstrip('\n'),tangent_rows);events.append(('T',idx))
        else:events.append(('L',line))
    def restore_name(member):
        return labels[member[1]]+suffixes[member[2]] if isinstance(member,tuple) else member
    def restore_row(row):
        return '{'+','.join(json.dumps(k)+':'+values[v] for k,v in zip(row['keys'],row['values']))+'}'
    restored=[]
    for event in events:
        if event[0]=='L':restored.append(event[1])
        elif event[0]=='C':
            g=groups[event[1]];restored.append('CHECK '+restore_name(g['members'][event[2]])+' '+g['verdict']+(' '+g['payload'] if g['payload'] else '')+'\n')
        elif event[0]=='P':restored.append('PROBE '+event[1]+' '+restore_row(probe_rows[event[2]])+'\n')
        elif event[0]=='T':restored.append('TANGENT '+restore_row(tangent_rows[event[1]])+'\n')
    reconstructed=''.join(restored).encode()
    if reconstructed!=raw:raise AssertionError('Lossless exact original-byte reconstruction failed')
    # This on-box raw factor record retains exact global interleaving as well.
    factorpath=target.with_suffix('.factors.json')
    with factorpath.open('x') as f:json.dump({'labels':labels,'suffixes':suffixes,'values':values,'groups':groups,'probes':probe_rows,'tangents':tangent_rows,'events':events},f,separators=(',',':'))
    def ranges(ids):
        result=[]
        for v in ids:
            if result and v==result[-1][-1]+1:result[-1].append(v)
            else:result.append([v])
        return ','.join(str(r[0]) if len(r)==1 else str(r[0])+'..'+str(r[-1]) for r in result)
    with target.open('x') as f:
        def out(s):f.write(s+'\n')
        out('# Complete lossless written-view factorization')
        out('Original SHA256 '+hashlib.sha256(raw).hexdigest())
        out('Exact original-byte reconstruction PASS; '+str(len(lines))+' original lines; '+str(sum(len(g['members']) for g in groups))+' checks; '+str(len(probe_rows))+' probes; '+str(len(tangent_rows))+' tangents.')
        out('No scientific evaluation. Raw records and exact global-order factor file remain on box. Check group memberships preserve multiplicity; P indices refer to probe names; S indices append exact check suffixes. A..B means every consecutive index inclusive, in order.')
        out('## Probe names')
        for i,v in enumerate(labels):out('P'+str(i)+' '+v)
        out('## Check suffixes')
        for i,v in enumerate(suffixes):out('S'+str(i)+' '+v)
        out('## Complete check payload groups and all members')
        for i,g in enumerate(groups):
            out('G'+str(i)+' '+g['verdict']+' '+g['payload'])
            plain=[];by_suffix=defaultdict(list)
            for member in g['members']:
                if isinstance(member,tuple):by_suffix[member[2]].append(member[1])
                else:plain.append(member)
            if plain:out(' names '+json.dumps(plain,separators=(',',':')))
            for sid,ids in by_suffix.items():out(' members S'+str(sid)+' P['+ranges(ids)+']')
        out('## Field-value dictionary (V indices)')
        for i,v in enumerate(values):out('V'+str(i)+' '+v)
        for kind,rows in [('PROBE',probe_rows),('TANGENT',tangent_rows)]:
            out('## Complete '+kind+' rows as field-value indices')
            previous=None
            for i,row in enumerate(rows):
                if row['keys']!=previous:out('Fields '+json.dumps(row['keys'],separators=(',',':')));previous=row['keys']
                out(kind+str(i)+' '+','.join(str(v) for v in row['values']))
        out('## All remaining original lines, in original relative order')
        for event in events:
            if event[0]=='L':out(json.dumps(event[1]))
        out('END complete lossless written view; original-byte reconstruction verified.')
    print(json.dumps({'view':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'lines':len(target.read_text().splitlines()),'original_sha256':hashlib.sha256(raw).hexdigest(),'reconstruction_exact':True}))

if __name__=='__main__':main()
