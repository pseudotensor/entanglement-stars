"""Own raw-column source reconstruction; never imports a foreign implementation."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import zipfile
import numpy as np

from cpu_runtime import require_remote_cpu
from iso128_rebuild_core_v1 import (file_identity, source_file_bindings,
    validate_slice_metadata, validate_run_metadata, raw_channel_sources, difference_details)
from iso128_rebuild_core_v1 import validate_inventory, weighted_sources, require_source_cpu, validate_saved_arrays


def archive_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate NPZ member names')
        for name in names:
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, order, dtype = np.lib.format._read_array_header(stream, version)
            result[name] = {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': bool(order),
                            'uncompressed_bytes': archive.getinfo(name).file_size}
    return result


def require_controls(path, bindings):
    controls = json.loads(Path(path).read_text())
    if controls['bindings'] != bindings or not controls['passed'] or not controls['checks']:
        raise ValueError('Manufactured controls do not bind the current source files')
    if controls['count'] != len(controls['checks']) or not all(row['passed'] for row in controls['checks']):
        raise ValueError('Incomplete/failed manufactured controls')
    return file_identity(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--small-metadata',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--clock',choices=['T0005.00','T0011.50'],required=True);args=p.parse_args()
    runtime=require_source_cpu(args.outdir)
    bindings=source_file_bindings(args.spec);control_identity=require_controls(args.controls,bindings)
    if any(args.outdir.iterdir()):raise ValueError('Use a fresh owned output directory; no implicit overwrite/replay')
    inventory=json.loads(args.inventory.read_text());small=json.loads(args.small_metadata.read_text())
    validate_inventory(inventory)
    entry=inventory['archives'][args.clock];path=Path(entry['identity']['path'])
    actual=file_identity(path)
    if actual != entry['identity']:raise ValueError('Slice bytes differ from pinned inventory')
    headers=archive_headers(path)
    if headers != entry['headers']:raise ValueError('Slice header schema differs from pinned inventory')
    run_path=Path(inventory['run_identity']['path']);run_identity=file_identity(run_path)
    if run_identity != inventory['run_identity']:raise ValueError('Run JSON identity changed')
    run=json.loads(run_path.read_text());E0=validate_run_metadata(run['args'],run['head'])
    expected_time={'T0005.00':640.,'T0011.50':1472.}[args.clock]
    timespec={'T0005.00':5.,'T0011.50':11.5}[args.clock]
    with np.load(path,allow_pickle=False) as archive:
        values={key:archive[key] for key in ['T','rs','R','kappas','weights','dtype']+['f_'+str(k) for k in range(1,11)]}
        association=validate_slice_metadata(values,headers,expected_time)
        for key,value in values.items():
            if not np.array_equal(value,small['slices'][args.clock]['values'][key]):
                raise ValueError('Small metadata differs from actual slice: '+key)
        radius=values['R'];aggregates={schedule:np.zeros((4096,2),complex) for schedule in ['s64','s127']}
        pair_aggregates={schedule:np.zeros((4096,2),complex) for schedule in aggregates}
        collected={schedule:{} for schedule in aggregates}
        collected_pairs={schedule:{} for schedule in aggregates}
        total_scale=np.zeros((4096,2));pair_scale=np.zeros_like(total_scale)
        summaries=[];products=[]
        for k,weight in zip(values['kappas'],values['weights']):
            k=int(k);sea=archive['sea_'+str(k)];reference=archive['R2_'+str(k)];dust=archive['dust_'+str(k)]
            f=values['f_'+str(k)]
            results={name:raw_channel_sources(sea,reference,dust,f,radius,k,block=block,reverse_blocks=reverse)
                     for name,block,reverse in [('s64',64,False),('s127',127,True)]}
            arrays={'R':radius,'f':f,'weight':np.array(weight)};summary={'kappa':k,'weight':float(weight),'schedules':{},'order':{}}
            primary=results['s64'];scale_pair=np.abs(primary['parts']['sea']['local_complex'].real)+np.abs(primary['parts']['reference']['local_complex'].real)
            scale_total=scale_pair+np.abs(primary['parts']['dust']['local_complex'].real)
            pair_scale+=weight*scale_pair;total_scale+=weight*scale_total
            for name,result in results.items():
                collected[name][k]={'total_complex':result['total_complex']}
                collected_pairs[name][k]={'total_complex':result['pairs_complex']}
                sums={}
                for role,data in result['parts'].items():
                    arrays[f'{name}_{role}_local']=data['local_complex']
                    arrays[f'{name}_{role}_weighted']=weight*data['local_complex']
                    arrays[f'{name}_{role}_norms_squared']=data['column_norms_squared']
                    sums[role]={'global_real':data['global_complex'].real.tolist(),'global_imaginary':data['global_complex'].imag.tolist(),
                                'column_count':len(data['column_norms_squared']),
                                'max_column_norm_squared_defect':float(np.max(np.abs(data['column_norms_squared']-1),initial=0)) if len(data['column_norms_squared']) else None}
                for role in ['pairs','total']:
                    arrays[f'{name}_{role}_local']=result[role+'_complex']
                    arrays[f'{name}_{role}_weighted']=weight*result[role+'_complex']
                summary['schedules'][name]=sums
            for role in ['sea','reference','dust','pairs','total']:
                left=primary['parts'][role]['local_complex'] if role in primary['parts'] else primary[role+'_complex']
                other=results['s127'];right=other['parts'][role]['local_complex'] if role in other['parts'] else other[role+'_complex']
                scale=scale_total if role in ['dust','total'] else scale_pair
                for j,observable in enumerate(['eps','J']):
                    item,details=difference_details(left[:,j].real,right[:,j].real,scale[:,j],radius)
                    summary['order'][role+'_'+observable]=item
                    for field,a in details.items():arrays[f'order_{role}_{observable}_{field}']=a
            validate_saved_arrays(arrays)
            output=args.outdir/f'k{k}_sources.npz';np.savez(output,**arrays);products.append(file_identity(output))
            summaries.append(summary)
            print(json.dumps({'clock':args.clock,'kappa':k,'source_saved':str(output)}),flush=True)
            del sea,reference,dust,results,arrays,primary
        for name in aggregates:
            aggregates[name]=weighted_sources(collected[name],values['kappas'],values['weights'])['total_complex']
            pair_aggregates[name]=weighted_sources(collected_pairs[name],values['kappas'],values['weights'])['total_complex']
    combined={'R':radius,'T':np.array(expected_time),'rs':np.array(128.),'E0':np.array(E0),
              'scale_pair':pair_scale,'scale_total':total_scale}
    order={}
    for name in aggregates:
        if not np.all(np.isfinite(aggregates[name])) or not np.all(np.isfinite(pair_aggregates[name])):
            raise ValueError('Nonfinite weighted aggregate')
        combined[name+'_total']=aggregates[name];combined[name+'_pairs']=pair_aggregates[name]
    for role,items,scale in [('total',aggregates,total_scale),('pairs',pair_aggregates,pair_scale)]:
        for j,observable in enumerate(['eps','J']):
            summary,details=difference_details(items['s64'][:,j].real,items['s127'][:,j].real,scale[:,j],radius)
            order[role+'_'+observable]=summary
            for key,value in details.items():combined[f'order_{role}_{observable}_{key}']=value
    validate_saved_arrays(combined)
    out=args.outdir/'combined_sources.npz';np.savez(out,**combined);products.append(file_identity(out))
    result={'available':True,'scope':'raw stored-column shell sources only; no field solve or original-source comparison',
            'T':expected_time,'T_over_rs':timespec,'runtime':runtime,'numpy':np.__version__,'python':platform.python_version(),
            'spec_source_bindings':bindings,'controls':control_identity,'slice':actual,'run':run_identity,
            'inventory':file_identity(args.inventory),'small_metadata':file_identity(args.small_metadata),
            'association':association,'schedules':{'s64':{'block':64,'order':'ascending'},'s127':{'block':127,'order':'descending'}},
            'channels':summaries,'order':order,'products':products,
            'order_concern':any(x['order_concern'] for x in order.values()) or any(x['order_concern'] for ch in summaries for x in ch['order'].values())}
    (args.outdir/'source_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':str(args.outdir/'source_complete.json'),'order_concern':result['order_concern']}))


if __name__=='__main__':main()
