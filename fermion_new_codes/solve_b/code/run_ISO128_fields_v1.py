"""Own ISO128 field production from frozen own shell sources only.

Original fields and printed answers are deliberately absent from this producer.
No physical execution is admitted before the same-pair manufactured review.
"""
from __future__ import annotations
import argparse,json,pickle,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline,PPoly
from iso128_rebuild_core_v1 import file_identity, require_source_cpu, source_file_bindings
from iso128_field_core_v1 import GaussianSources,SolvedField,DomainUnavailable
from iso128_field_readout_v1 import dense_readout,vacuum_readout,uniform_grid,cumulative_w,backaverage,extrema,empirical_band,geometry_consistency,defect_summary,additional_collar_metadata
from control_ISO128_fields_v1 import field_bindings


def source_association(complete, combined_identity, values, clock, source_bindings):
    expected={'T0005.00':(640.,5.),'T0011.50':(1472.,11.5)}
    if clock not in expected:raise ValueError('Unknown slice clock')
    T,t=expected[clock]
    if (not complete.get('available') or complete.get('T')!=T or complete.get('T_over_rs')!=t
            or complete['spec_source_bindings']!=source_bindings):
        raise ValueError('Own source completion clock/bindings unavailable')
    matches=[x for x in complete['products'] if Path(x['path']).resolve()==Path(combined_identity['path']).resolve()]
    if matches!=[combined_identity]:raise ValueError('Loaded source is not its unique frozen product')
    for key,target in [('T',T),('rs',128.),('E0',235.17137214931623)]:
        value=np.asarray(values[key])
        if value.shape!=() or np.iscomplexobj(value) or float(value)!=target:
            raise ValueError('Source scalar association failure: '+key)
    if np.iscomplexobj(values['R']) or not np.array_equal(values['R'],np.arange(1.,4097.)):
        raise ValueError('Exact unshifted shell grid required')
    for key in ('s64_total','s127_total'):
        a=values[key]
        if a.shape!=(4096,2) or a.dtype!=np.dtype('complex128') or not np.all(np.isfinite(a)):
            raise ValueError('Full finite raw complex source totals required')


def require_controls(path,bindings):
    x=json.loads(Path(path).read_text())
    if (x['bindings']!=bindings or not x['passed'] or x['count']!=len(x['checks'])
            or not x['checks'] or not all(y['passed'] for y in x['checks'])):
        raise ValueError('Current complete manufactured field controls required')
    return file_identity(path)


def parity_spline(field):
    nodes=uniform_grid(field.collar,field.ell);values=field.fields(nodes)
    reflected=np.r_[-nodes[:0:-1],nodes]
    N=make_interp_spline(reflected,np.r_[values['N'][:0:-1],values['N']],k=5,bc_type='not-a-knot')
    v=make_interp_spline(reflected,np.r_[-values['v'][:0:-1],values['v']],k=5,bc_type='not-a-knot')
    def observable(r):
        if not 0<=r<=field.collar:raise DomainUnavailable('Parity spline extrapolation forbidden')
        n,flow,np_,vp=map(float,(N(r),v(r),N(r,nu=1),v(r,nu=1)))
        if r==0.:flow=0. # Explicit odd-parity centre limit.
        if n<=0 or flow>0:raise DomainUnavailable('Parity spline lapse/flow domain failure')
        f=n-abs(flow);fp=np_-abs(vp) if r==0 else np_-np.sign(flow)*vp
        # Production roots use U/r-1. Here g=(v/N)^2-1 is the same
        # root surrogate on the explicitly distinct spline representation.
        q=flow/n;qp=vp/n-flow*np_/(n*n)
        return dict(f=f,g=q*q-1,df=fp,dg=2*q*qp,N=n,dN=np_)
    cusps=[float(r) for r in PPoly.from_spline(v).roots(extrapolate=False) if np.isfinite(r) and 0<r<field.collar]
    return nodes,values,N,v,observable,cusps


def sample_solution(field,radius):
    f=field.fields(radius);sources=field.source.sample(radius)
    rhs=np.zeros((len(radius),2))
    positive=radius>0
    if not field.vacuum:
        for i in np.flatnonzero(positive):
            r=radius[i]
            if r<field.r0:
                A,B,C,D=(field.coefficients[k] for k in ('A','B','C','D'))
                rhs[i]=[3*A*r*r+5*B*r**4,2*C*r+4*D*r**3]
            else:rhs[i]=field.rhs(r,[f['U'][i],f['y'][i]])
    return {'r':radius,**f,'rho':sources[:,0],'J':sources[:,1],'Uprime':rhs[:,0],'yprime':rhs[:,1],
            'Walg':f['N']**2*f['U'],'f':f['N']-np.abs(f['v'])}


def write_source_evaluations(source,path):
    # These are all direct source queries, including trial, readout and
    # sampled diagnostic queries. Preserve them independently of plots.
    with Path(path).open('w') as stream:
        for row in source.evaluations:stream.write(json.dumps(row,allow_nan=False)+'\n')


def scalar_readouts(summary):
    return {'Rt':summary['outer']['r'] if summary['outer'] else None,
            'rin':summary['inner']['r'] if summary['inner'] else None,
            'fmin':summary['minimum']['value']}


def main():
    p=argparse.ArgumentParser()
    for key in ('outdir','sources','source-spec','field-spec','controls'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--clock',choices=['T0005.00','T0011.50'],required=True);a=p.parse_args()
    runtime=require_source_cpu(a.outdir)
    if any(a.outdir.iterdir()):raise ValueError('Fresh field output directory required')
    bindings=field_bindings(a.field_spec);controls=require_controls(a.controls,bindings)
    complete_path=a.sources/'source_complete.json';complete=json.loads(complete_path.read_text())
    for product in complete['products']:
        if file_identity(product['path'])!=product:raise ValueError('Frozen own source product changed')
    combined_path=a.sources/'combined_sources.npz';combined_identity=file_identity(combined_path)
    with np.load(combined_path,allow_pickle=False) as archive:
        values={key:archive[key] for key in ('R','T','rs','E0','s64_total','s127_total')}
    source_association(complete,combined_identity,values,a.clock,source_file_bindings(a.source_spec))
    E0=float(values['E0']);results={};products=[]
    for variant in ('P','H','C','O'):
        directory=a.outdir/variant;directory.mkdir()
        shell=values['s127_total' if variant=='O' else 's64_total']
        source=GaussianSources(values['R'],shell[:,0].real,shell[:,1].real,4.)
        try:field=SolvedField(source,.25,4096.,4144.,4.,variant).integrate()
        except DomainUnavailable as exc:
            results[variant]={'available':False,'status':'regular_branch_unavailable','reason':str(exc),
                              'centre_sources':source.centre,'centre_omissions':source.centre_omissions}
            evaluation_path=directory/'source_evaluations.jsonl';write_source_evaluations(source,evaluation_path);products.append(file_identity(evaluation_path))
            summary_path=directory/'field_summary.json';summary_path.write_text(json.dumps(results[variant],indent=2,allow_nan=False)+'\n');products.append(file_identity(summary_path))
            continue
        meta={'available':field.status=='complete','status':field.status,'failure':field.failure,
              'coefficients':field.coefficients,'centre_sources':source.centre,'centre_omissions':source.centre_omissions,'r0':field.r0,'rtol':field.rtol,
              'max_step':field.max_step,'atol':getattr(field,'atol',None),'correction_sizes':getattr(field,'correction_sizes',None),
              'nfev':field.nfev,'accepted_steps':field.steps,'rejected_domain_trials':field.trials}
        # Persist accepted steps and dense state even if a later diagnostic
        # fails. These are our own trusted objects, never external input.
        dense_path=directory/'own_dense.pkl';dense_path.write_bytes(pickle.dumps(field,protocol=5));products.append(file_identity(dense_path))
        if field.status=='complete':
            try:
                all_arrays={}
                for name,r in [('uniform32',uniform_grid(4144.,4/32)),('nodes',uniform_grid(4096.,4.)),
                               ('sites',values['R']),('steps',np.array([x['r'] for x in field.steps]) if field.steps else np.array([0.,4144.]))]:
                    sampled=sample_solution(field,r)
                    for key,val in sampled.items():all_arrays[name+'_'+key]=val
                readouts={}
                for mesh in (16,32):
                    grid=np.unique(np.r_[uniform_grid(4096.,4/mesh),field.r0,[x['r'] for x in field.steps if x['r']<=4096.]])
                    readouts[str(mesh)]=vacuum_readout(grid) if field.vacuum else dense_readout(field.observables,grid)
                meta['dense_readouts']=readouts
                meta['node_minimum']=extrema(all_arrays['nodes_f'],all_arrays['nodes_r'])
                meta['site_minimum']=extrema(all_arrays['sites_f'],all_arrays['sites_r'])
                charge=float(field.state(4096.)[0]);meta['E']=charge/.5;meta['E_over_E0']=meta['E']/E0
                if not np.isfinite(meta['E']) or not np.isfinite(meta['E_over_E0']):
                    meta['E']=None;meta['E_over_E0']=None
                    raise DomainUnavailable('Nonfinite derived collar charge')
                meta['collar_and_tail']={str(r):{key:float(value) for key,value in field.fields(r).items()} for r in (4096.,4128.,4144.)}
                alternate_state=field.state(4128.)
                meta['additional_collar']=additional_collar_metadata(float(alternate_state[0]),field.y_collar,float(alternate_state[1]),readouts['32']['minimum']['value'],.25)
                factor=meta['additional_collar']['normalization_factor']
                def alternate_observable(r):
                    row=field.observables(r)
                    return {key:value*factor if key in ('f','df','N','dN') else value for key,value in row.items()}
                alternate_grid=np.unique(np.r_[uniform_grid(4128.,4/32),field.r0,[x['r'] for x in field.steps if x['r']<=4128.]])
                meta['additional_collar']['extended_domain_readout']=vacuum_readout(alternate_grid) if field.vacuum else dense_readout(alternate_observable,alternate_grid)
                meta['additional_collar']['extended_minimum_domain']=[0.,4128.]
                common=uniform_grid(4096.,4/32);primary=field.fields(common);alternate=field.fields(common,normalization=4128.)
                all_arrays['additional_collar_r']=common
                for key in ('N','v'):
                    all_arrays['additional_collar_'+key]=alternate[key]
                    all_arrays['additional_collar_'+key+'_minus_primary']=alternate[key]-primary[key]
                tailr=4096.+uniform_grid(48.,4/32);tail=field.fields(tailr);vacN,vacv=field.applied_exterior(tailr)
                all_arrays['tail_r']=tailr
                for key,vac in [('N',vacN),('v',vacv)]:
                    all_arrays['tail_Gaussian_'+key]=tail[key];all_arrays['tail_vacuum_'+key]=vac
                    all_arrays['tail_'+key+'_Gaussian_minus_vacuum']=tail[key]-vac
                    meta.setdefault('Gaussian_vs_vacuum_tail_defects',{})[key]=defect_summary(tail[key]-vac,tailr,4096.,1.)
                meta['additional_collar']['comparison_domain']=[0.,4096.]
                meta['Gaussian_vs_vacuum_tail_domain']=[4096.,4144.]
                meta['exterior_extension_root']=charge if charge>4096. else None
                for mesh in (16,32):
                    r=uniform_grid(4144.,4/mesh);data=sample_solution(field,r)
                    integrand,W=cumulative_w(r,data['N'],data['rho'],.25)
                    for key,val in dict(r=r,integrand=integrand,Wint=W,Walg=data['Walg'],difference=W-data['Walg']).items():all_arrays['W'+str(mesh)+'_'+key]=val
                meta['W_checks']={}
                for mesh in (16,32):
                    delta=all_arrays['W'+str(mesh)+'_difference'];r=all_arrays['W'+str(mesh)+'_r'];at_collar=np.flatnonzero(r==4096.)
                    if len(at_collar)!=1:raise DomainUnavailable('W collar association unavailable')
                    meta['W_checks'][str(mesh)]=defect_summary(delta,r,4096.,.5)
                fine=all_arrays['W32_Wint'][::2];coarse=all_arrays['W16_Wint'];difference=fine-coarse
                if not np.array_equal(all_arrays['W32_r'][::2],all_arrays['W16_r']):raise DomainUnavailable('W refinement lacks exact common grid')
                all_arrays['W_refinement_r']=all_arrays['W16_r'];all_arrays['W_refinement_fine_minus_coarse']=difference
                meta['W_checks']['refinement']=defect_summary(difference,all_arrays['W16_r'],4096.,.5)
                average_meta={}
                for label,step,band in [('primary',4/16,8),('refined',4/32,8),('extended',4/32,12)]:
                    avg=backaverage(field,values['R'],4.,step,band)
                    all_arrays['average_'+label]=avg['values'];average_meta[label]={k:v for k,v in avg.items() if k not in ('R','values')}
                all_arrays['average_step_difference']=all_arrays['average_refined']-all_arrays['average_primary']
                all_arrays['average_band_difference']=all_arrays['average_extended']-all_arrays['average_refined']
                meta['averages']=average_meta
                meta['applied_minimum']=extrema(all_arrays['average_primary'][:,0]-np.abs(all_arrays['average_primary'][:,1]),values['R'])
                if variant=='P':
                    nodes,nodevalues,Ns,vs,observable,cusps=parity_spline(field)
                    grid=np.unique(np.r_[uniform_grid(4096.,4/32),cusps])
                    meta['parity_spline_readout']=dense_readout(observable,grid,extra_candidates=cusps)
                    meta['parity_spline']={'degree':5,'boundary':'not-a-knot','N_t':Ns.t.tolist(),'N_c':Ns.c.tolist(),
                        'v_t':vs.t.tolist(),'v_c':vs.c.tolist(),'domain':[0.,4096.],'extrapolation':False,'v_zero_cusps':cusps}
                for key,value in all_arrays.items():
                    if not np.all(np.isfinite(value)):raise DomainUnavailable('Nonfinite saved field diagnostic '+key)
                arrays_path=directory/'field_arrays.npz';np.savez(arrays_path,**all_arrays);products.append(file_identity(arrays_path))
            except (DomainUnavailable,ValueError,FloatingPointError,OverflowError) as exc:
                meta['available']=False;meta['diagnostic_status']='unresolved';meta['diagnostic_reason']=str(exc)
                partial=directory/'partial_field_arrays.npz'
                np.savez(partial,**all_arrays);products.append(file_identity(partial))
        evaluation_path=directory/'source_evaluations.jsonl';write_source_evaluations(source,evaluation_path);products.append(file_identity(evaluation_path))
        summary_path=directory/'field_summary.json';summary_path.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n');products.append(file_identity(summary_path))
        results[variant]=meta
        print(json.dumps({'variant':variant,'available':meta['available'],'status':meta['status'],'summary':str(summary_path)}),flush=True)
    bands={}
    if results.get('P',{}).get('available'):
        primary=scalar_readouts(results['P']['dense_readouts']['16']);primary['E_over_E0']=results['P']['E_over_E0']
        for quantity,value in primary.items():
            other={}
            for v in ('H','C','O'):
                row=results.get(v,{})
                other[v]=(row['E_over_E0'] if quantity=='E_over_E0' else scalar_readouts(row['dense_readouts']['16'])[quantity]) if row.get('available') and (quantity=='E_over_E0' or all(x['resolved'] for x in row['dense_readouts'].values())) else None
            other['mesh']=results['P']['E_over_E0'] if quantity=='E_over_E0' else scalar_readouts(results['P']['dense_readouts']['32'])[quantity] if all(x['resolved'] for x in results['P']['dense_readouts'].values()) else None
            bands[quantity]=empirical_band(value,other,1e-10 if quantity in ('Rt','rin') else 1e-11)
    expected_geometry='no_root' if a.clock=='T0005.00' else 'pair'
    geometry_check=geometry_consistency(results,expected_geometry);geometry_consistent=geometry_check['resolved']
    if not geometry_consistent:
        for quantity in ('Rt','rin','fmin'):
            if quantity in bands:bands[quantity]={'status':'precision_unresolved','band':None,'spread':None,'diagnostic_unmatched_band':bands[quantity],'reason':'geometry_association_unresolved'}
    minband=bands.get('fmin',{}).get('band')
    geometry_admitted=geometry_consistent and geometry_check['expected_geometry_met'] and minband is not None and (minband[0]>0 if expected_geometry=='no_root' else minband[1]<0)
    result={'scope':'independent direct-Gaussian IVP on frozen own raw shell sources; historical field representation matching unresolved',
            'runtime':runtime,'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),
            'clock':a.clock,'T':complete['T'],'T_over_rs':complete['T_over_rs'],'E0':E0,'field_bindings':bindings,
            'source_complete':file_identity(complete_path),'combined_source':combined_identity,'controls':controls,
            'available':all(results.get(v,{}).get('available',False) for v in ('P','H','C','O')),
            'variants':{v:{k:row.get(k) for k in ('available','status','reason','diagnostic_reason','E','E_over_E0')} for v,row in results.items()},
            'geometry_consistent':geometry_consistent,'geometry_check':geometry_check,'geometry_admitted_under_declared_recipe':geometry_admitted,
            'historical_definition_match':'unresolved: realized support/collar/dense representation not supplied',
            'empirical_bands':bands,'products':products}
    (a.outdir/'field_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':str(a.outdir/'field_complete.json'),'available':result['available']}))


if __name__=='__main__':main()
