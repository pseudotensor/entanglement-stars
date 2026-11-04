"""Format-only inventory of specified residual inputs; no field calculation."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import zipfile

ROOT = Path('<internal>/input_inventory')
INPUTS = [('M3b\\_s4 (321; 40)', '<internal>/collapse_lad_M3b_s4_m0p4_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('P64\\_s4 (321; 40)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json'), ('M3a\\_s4 (321; 40)', '<internal>/collapse_lad_M3a_s4_m0p1_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('M3c\\_s4\\_nk (267; 22.1667)', '<internal>/collapse_lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j.json'), ('M3d\\_s4 (21; 0.4908)', '<internal>/collapse_lad_M3d_s4_m0p05_rs326_n3912_R04_k12_tau8_ell4_prep4_A_R2_avg_plain_v36e.json'), ('M3b\\_s8 (321; 40)', '<internal>/collapse_lad_M3b_s8_m0p4_rs64_n2048_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json'), ('P64\\_s2 composite (321; 40)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json'), ('P64\\_s8 (321; 40)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json'), ('P64\\_s4\\_tau4 (641; 40)', '<internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg.json'), ('P64\\_s4\\_tau2 (689; 21.5)', '<internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg.json'), ('P64\\_s4\\_c19 (321; 40)', '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json'), ('M3a\\_s2 (321; 40)', '<internal>/collapse_lad_M3a_s2_m0p1_rs64_n2048_R04_k10_tau8_ell2_prep4_A_R2_avg_plain_v36d.json'), ('M3a\\_s4\\_tau4 (257; 16)', '<internal>/collapse_lad_M3a_s4_tau4_m0p1_rs64_n2048_R04_k10_tau4_ell4_prep4_A_R2_avg_plain_v36d.json'), ('M3c\\_s4\\_plain (266; 22.0833)', '<internal>/collapse_lad_M3c_s4_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_16\\_s4 (321; 40)', '<internal>/collapse_lad_S4_16_s4_m0p2_rs16_n512_R04_k10_tau2_ell4_prep4_A_R2_avg_plain_v36d.json'), ('RS32\\_s4 (641; 40)', '<internal>/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg.json'), ('S4\\_128\\_s4 (306; 19.0625)', '<internal>/collapse_lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_16\\_s2 (321; 40)', '<internal>/collapse_lad_S4_16_s2_m0p2_rs16_n512_R04_k10_tau2_ell2_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_16\\_s4\\_tau1 (257; 16)', '<internal>/collapse_lad_S4_16_s4_tau1_m0p2_rs16_n512_R04_k10_tau1_ell4_prep4_A_R2_avg_plain_v36d.json'), ('RS32\\_s2 (641; 40)', '<internal>/collapse_c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0b.json'), ('RS32\\_s4\\_tau1 (1281; 40)', '<internal>/collapse_c25_rs32_A_R2_tau1_ell4_avg.json'), ('S4\\_128\\_s4\\_nk\\_v36m (48; 2.9375)', '<internal>/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m.json'), ('S4\\_128\\_s4\\_nk\\_v36q (46; 2.8125)', '<internal>/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json'), ('S4\\_128\\_s8 (641; 40)', '<internal>/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_128\\_s16 (641; 40)', '<internal>/collapse_lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_R2\\_m0p2 (321; 40)', '<internal>/collapse_lad_S4_R2_m0p2_m0p2_rs64_n2048_R02_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('S4\\_R2\\_m0p1 (321; 40)', '<internal>/collapse_lad_S4_R2_m0p1_m0p1_rs64_n2048_R02_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json'), ('COLLAPSE plain rung 2 (321; 40)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_plain_v36d_from0.json'), ('COLLAPSE rung 4 first (146; 18.125)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json'), ('COLLAPSE rung 8 first (146; 18.125)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0.json'), ('COLLAPSE bounce 4 (129; 16)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_bounce.json'), ('COLLAPSE bounce 8 (129; 16)', '<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_bounce.json')]

def identity(path):
    x=path.stat()
    return dict(path=str(path),resolved_path=str(path.resolve()),size=x.st_size,
                mtime_ns=x.st_mtime_ns,device=x.st_dev,inode=x.st_ino)

def literal(value):
    if isinstance(value,float) and not math.isfinite(value):
        return dict(nonfinite_literal=repr(value))
    if isinstance(value,dict):
        return {k:literal(v) for k,v in value.items()}
    if isinstance(value,list):
        return [literal(v) for v in value]
    return value

def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Box directory required')
    out=ROOT/'residual_input_inventory_v1_1.json'
    if out.exists():
        raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    if os.getpriority(os.PRIO_PROCESS,0)<10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    import numpy as np
    result=dict(purpose='Task5 input-format inventory only',scientific_evaluations=0,inputs=[])
    for label,filename in INPUTS:
        path=Path(filename);row=dict(label=label,path=filename,exists=path.exists())
        if path.exists():
            initial=identity(path);raw=path.read_bytes();data=json.loads(raw)
            if identity(path)!=initial:
                raise RuntimeError('Input changed')
            records=data.get('records',[])
            row.update(identity=initial,sha256=hashlib.sha256(raw).hexdigest(),
                       root_keys=list(data),args=data.get('args'),head=data.get('head'),
                       record_count=len(records),
                       first_record=records[0] if records else None,
                       last_record_keys=list(records[-1]) if records else [])
        fields=path.with_name(path.stem+'_fields.npz')
        row['fields']=dict(path=str(fields),exists=fields.exists())
        if fields.exists():
            initial=identity(fields);members=[]
            with zipfile.ZipFile(fields) as z:
                for item in z.infolist():
                    with z.open(item) as stream:
                        version=np.lib.format.read_magic(stream)
                        if version==(1,0):
                            shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                        elif version==(2,0):
                            shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError('Unrecognized NPY header')
                    members.append(dict(name=item.filename,shape=shape,dtype=str(dtype),
                        hasobject=dtype.hasobject,fortran_order=order,size=item.file_size,
                        compressed_size=item.compress_size,crc32=item.CRC))
            if identity(fields)!=initial:
                raise RuntimeError('Fields changed')
            row['fields'].update(identity=initial,members=members)
        result['inputs'].append(row)
    result['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with out.open('x') as stream:
        json.dump(literal(result),stream,indent=2,allow_nan=False)
        stream.write('\n')

if __name__=='__main__':
    main()
