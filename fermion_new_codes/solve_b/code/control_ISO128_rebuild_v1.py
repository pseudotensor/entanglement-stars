"""Manufactured-only checks of the production ISO128 source/kernel helpers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from cpu_runtime import require_remote_cpu
from iso128_rebuild_core_v1 import (bare_actions, column_sources, raw_channel_sources,
                                    weighted_sources, gaussian_kernels,
                                    gaussian_centre_coefficients, validate_slice_metadata,
                                    difference_details, source_comparison_index, validate_run_metadata)
from iso128_rebuild_core_v1 import (source_file_bindings, validate_inventory, require_source_cpu,
    validate_one_thread, validate_saved_arrays, validate_bound_comparison_source)


def packed(x):
    a = np.asarray(x)
    def safe(values):
        result=values.astype(object);result[~np.isfinite(values)]=None
        return result.tolist()
    return {'real': safe(a.real), 'imag': safe(a.imag)} if np.iscomplexobj(a) else safe(a)


class Checks:
    def __init__(self):
        self.rows = []

    def close(self, name, actual, expected, tol=5e-13, **extra):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.rows.append(dict(name=name, actual=packed(actual), expected=packed(expected),
                                  passed=False, shape_mismatch=True))
            return
        if not np.all(np.isfinite(expected)):
            self.rows.append(dict(name=name, actual=packed(actual), expected=packed(expected),
                                  passed=False, finite_expected=False))
            return
        if not np.all(np.isfinite(actual)):
            self.rows.append(dict(name=name, actual=packed(actual), expected=packed(expected),
                                  passed=False, finite_actual=False))
            return
        with np.errstate(over='ignore',invalid='ignore'):
            difference = (actual != expected).astype(float) if actual.dtype == np.dtype('bool') else np.abs(actual - expected)
        if not np.all(np.isfinite(difference)):
            self.rows.append(dict(name=name,actual=packed(actual),expected=packed(expected),
                                  passed=False, finite_difference=False))
            return
        error = float(np.max(difference, initial=0))
        limit = tol * max(1.0, float(np.max(np.abs(expected), initial=0)))
        self.rows.append(dict(name=name, actual=packed(actual), expected=packed(expected),
                              error=error, limit=limit,
                              passed=bool(np.all(np.isfinite(actual)) and error <= limit), **extra))

    def reject(self, name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError) as error:
            self.rows.append(dict(name=name, passed=True, rejection=type(error).__name__, message=str(error)))
        else:
            self.rows.append(dict(name=name, passed=False, rejection=None))


def dense_operators(r, k, m=.2, a=1.):
    s1 = np.array([[0, 1], [1, 0]], complex)
    s2 = np.array([[0, -1j], [1j, 0]], complex)
    s3 = np.diag([1., -1.])
    h = np.zeros((2 * len(r), 2 * len(r)), complex)
    q = np.zeros_like(h)
    for n, radius in enumerate(r):
        h[2*n:2*n+2, 2*n:2*n+2] = m*s3 + k/radius*s2
        if n + 1 < len(r):
            h[2*n:2*n+2, 2*n+2:2*n+4] = -1j*s1/(2*a)
            h[2*n+2:2*n+4, 2*n:2*n+2] = 1j*s1/(2*a)
        if n + 2 < len(r):
            q[2*n:2*n+2, 2*n+4:2*n+6] = -1j*np.eye(2)/(4*a)
            q[2*n+4:2*n+6, 2*n:2*n+2] = 1j*np.eye(2)/(4*a)
    return h, q


def source_checks(c):
    for component, sign in [(0, 1), (1, -1)]:
        x = np.zeros((2, 1), complex); x[component, 0] = 1
        result = column_sources(x, [2.], [1.], 3.)['local_complex'].real
        c.close('one-site mass ' + str(component), result, [[sign*.2, 0]])
    for sign in [-1, 1]:
        x = np.array([[1], [sign*1j]], complex)/np.sqrt(2)
        c.close('one-site angular ' + str(sign), column_sources(x, [2.], [1.], 3.)['local_complex'].real,
                [[sign*1.5, 0]])
        x = np.zeros((4, 1), complex); x[0, 0]=1/np.sqrt(2); x[3, 0]=sign*1j/np.sqrt(2)
        c.close('one-step allocation ' + str(sign), column_sources(x, [1.,2.], [1.], 1)['local_complex'].real,
                [[.1+sign*.25, 0], [-.1+sign*.25, 0]])
        x = np.zeros((6, 1), complex); x[0, 0]=1/np.sqrt(2); x[4, 0]=sign*1j/np.sqrt(2)
        c.close('two-step allocation ' + str(sign), column_sources(x, [1.,2.,3.], [1.], 1)['local_complex'].real,
                [[.1, sign*.125], [0,0], [.1, sign*.125]])
    rng=np.random.default_rng(130)
    for size in [1,2,3,5]:
        r=np.arange(1.,size+1); h,q=dense_operators(r,5)
        x=(rng.normal(size=(2*size,131))+1j*rng.normal(size=(2*size,131))).astype('complex128')
        f=np.linspace(0,1,131); covariance=(x*f)@x.conj().T
        expected=np.zeros((size,2))
        for n in range(size):
            pi=np.zeros_like(h); pi[2*n,2*n]=1; pi[2*n+1,2*n+1]=1
            for j,a in enumerate([h,q]): expected[n,j]=np.trace(covariance@(pi@a+a@pi)/2).real
        for block,reverse in [(64,False),(127,True)]:
            y=column_sources(x,r,f,5,block=block,reverse_blocks=reverse)
            c.close(f'dense endpoint n{size} b{block}',y['local_complex'].real,expected)
            c.close(f'global Hermitian n{size} b{block}',y['global_complex'].imag,[0.,0.])
        ah,aq=bare_actions(x,r,5)
        c.close(f'action h n{size}',ah,h@x); c.close(f'action Q n{size}',aq,q@x)
        c.close(f'Hermiticity h n{size}',h,h.conj().T)
        c.close(f'Hermiticity Q n{size}',q,q.conj().T)
        nfield=np.linspace(.5,1.3,size); vfield=np.linspace(-.7,.2,size)
        nd=np.diag(np.repeat(nfield,2)); vd=np.diag(np.repeat(vfield,2))
        clocked=(nd@h+h@nd+vd@q+q@vd)/2
        production=column_sources(x,r,f,5)['local_complex'].real
        c.close(f'variable clocked trace n{size}', np.sum(nfield*production[:,0]+vfield*production[:,1]),
                np.trace(covariance@clocked).real)
        permutation=rng.permutation(131)
        c.close(f'column permutation n{size}',column_sources(x[:,permutation],r,f[permutation],5)['local_complex'],
                column_sources(x,r,f,5)['local_complex'])
        c.close(f'global phase n{size}',column_sources(x*np.exp(.37j),r,f,5)['local_complex'],
                column_sources(x,r,f,5)['local_complex'])
    r=np.arange(1.,6.); sea=np.eye(10,dtype=complex)[:,:2]; dust=np.zeros((10,1),complex);dust[0,0]=2
    result=raw_channel_sources(sea,sea,dust,[.6],r,1)
    c.close('identical sea reference',result['pairs_complex'],np.zeros((5,2)))
    c.close('raw norm and fraction',result['energy'],[.48,0,0,0,0])
    c.close('raw squared norm retained',result['parts']['dust']['column_norms_squared'],[4.])
    empty=np.zeros((10,0),complex)
    negative_sea=sea[:,:1].copy()
    result=raw_channel_sources(negative_sea,2*negative_sea,empty,[],r,1)
    c.close('empty dust retained',result['parts']['dust']['local_complex'],np.zeros((5,2)))
    c.close('negative pair retained',result['energy'],[-.6,0,0,0,0])
    c.close('negative pair genuinely nonzero',bool(np.min(result['energy'])<0),True)
    c.close('empty dust pair total equality',result['total_complex'],result['pairs_complex'])
    channels={k:raw_channel_sources(sea,sea,dust,[.6],r,k) for k in [1,5]}
    c.close('weighted composition',weighted_sources(channels,[1,5],[16,80])['energy'],96*channels[1]['energy'])
    c.reject('duplicate weights',lambda:weighted_sources(channels,[1,1],[16,16]))
    c.reject('wrong multiplicity',lambda:weighted_sources(channels,[1,5],[16,16]))
    for name,value in [('negative',[-.1]),('above one',[1.1]),('nonfinite',[np.nan]),('length',[]),('complex',[.6j])]:
        c.reject('occupation guard '+name,lambda value=value:column_sources(dust,r,value,1))
    c.reject('complex radius',lambda:column_sources(dust,r.astype(complex),[.6],1))
    c.reject('bad row count',lambda:column_sources(dust[:-1],r,[.6],1))
    c.reject('unsupported dtype',lambda:column_sources(dust.astype('complex64'),r,[.6],1))
    bad=dust.copy();bad[0,0]=np.inf
    c.reject('nonfinite column',lambda:column_sources(bad,r,[.6],1))
    c.reject('empty invalid operator',lambda:column_sources(empty,r,[],np.nan))
    nfield=np.array([1.,2.,3.,4.]);vfield=np.array([-.5,.25,-.75,.125]);r4=np.arange(1.,5.)
    h,q=dense_operators(r4,1);nd=np.diag(np.repeat(nfield,2));vd=np.diag(np.repeat(vfield,2))
    clocked=(nd@h+h@nd+vd@q+q@vd)/2
    for role,second_row,expected in [('nearest',3,.65),('two-step',4,.24375)]:
        x=np.zeros((8,1),complex);x[0,0]=1/np.sqrt(2);x[second_row,0]=1j/np.sqrt(2)
        production=column_sources(x,r4,[1.],1)['local_complex'].real
        c.close('hand variable field '+role,np.sum(nfield*production[:,0]+vfield*production[:,1]),expected)
        c.close('hand clocked dense '+role,(x.conj().T@clocked@x).real.item(),expected)
        wrong=np.sum((x.conj()*(clocked@x)).real,axis=1).reshape(4,2).sum(axis=1)
        c.close('clocked-source mutation '+role,
                abs(np.sum(nfield*wrong+vfield*production[:,1])-expected)>1e-5,True)
    angular=np.zeros((10,1),complex);angular[:2,0]=[1/np.sqrt(2),1j/np.sqrt(2)]
    channels={k:raw_channel_sources(sea,sea,angular,[.6],r,k) for k in [1,5]}
    combined=weighted_sources(channels,[1,5],[16,80])['energy']
    c.close('angular weighted distinct channels',combined,[249.6,0,0,0,0])
    swapped=16*channels[5]['energy']+80*channels[1]['energy']
    c.close('channel-swap mutation distinguished',abs(combined[0]-swapped[0])>1.,True)


def metadata_checks(c):
    counts=[6,5,4,3,3,2,1,1,0,0]
    meta={'T':np.array(640.),'rs':np.array(128.),'R':np.arange(1.,4097.),
          'kappas':np.arange(1,11),'weights':np.arange(1,11)*16.,'dtype':np.array('complex128')}
    headers={}
    for key in ['T','rs','R','kappas','weights','dtype']:
        headers[key+'.npy']={'shape':list(meta[key].shape),'dtype':str(meta[key].dtype),'fortran_order':False}
    for k,count in enumerate(counts,1):
        meta['f_'+str(k)]=np.ones(count)
        headers['f_'+str(k)+'.npy']={'shape':[count],'dtype':'float64','fortran_order':False}
        for role,cols in [('sea',4096),('R2',4096),('dust',count)]:
            headers[f'{role}_{k}.npy']={'shape':[8192,cols],'dtype':'complex128','fortran_order':False}
    meta['f_5'][-1]=.6
    c.close('full roster dust count',validate_slice_metadata(meta,headers,640.)['weighted_dust_count'],1280.)
    for name,key,value in [('clock','T',641.),('scalar shape','T',[640.]),('rs','rs',64.),
                            ('shifted radius','R',meta['R']+.5),('complex radius','R',meta['R'].astype(complex)),
                            ('reversed radius','R',meta['R'][::-1]),('duplicate channel','kappas',np.ones(10,int)),
                            ('weight','weights',np.ones(10)),('fraction','f_5',[1.,1.,1.])]:
        altered=copy.deepcopy(meta);altered[key]=value
        c.reject('metadata '+name,lambda altered=altered:validate_slice_metadata(altered,headers,640.))
    altered=copy.deepcopy(headers);del altered['R2_10.npy']
    c.reject('metadata missing channel',lambda:validate_slice_metadata(meta,altered,640.))
    altered2=copy.deepcopy(headers);altered2['sea_1.npy']['shape']=[4096,8192]
    c.reject('metadata transposed columns',lambda:validate_slice_metadata(meta,altered2,640.))
    altered3=copy.deepcopy(headers);altered3['R.npy']['dtype']='int64'
    c.reject('metadata scalar vector dtype',lambda:validate_slice_metadata(meta,altered3,640.))
    config={'m':.2,'g':.25,'rs':128.,'ell':4,'nsites':4096,'half_offset':0,'reference':'R2'}
    c.close('charge-normalizer guard valid',validate_run_metadata(config,{'E0':235.17137214931623}),235.17137214931623)
    for bad in [0.,-1.,np.nan,[235.17137214931623]]:
        c.reject('charge-normalizer bad '+str(bad),lambda bad=bad:validate_run_metadata(config,{'E0':bad}))
    fields={k:np.zeros((3,4)) for k in ['eps_tot','J_tot','eps_pairs','J_pairs']};r=np.arange(1.,5.)
    record={'T':640.,'T_over_rs':5.}
    c.close('comparison unique exact clock',source_comparison_index(r,r,[0.,640.,1472.],record,fields),1)
    c.reject('comparison shifted clocks',lambda:source_comparison_index(r,r,[0.,641.,1473.],record,fields))
    c.reject('comparison shifted radius',lambda:source_comparison_index(r,r+.1,[0.,640.,1472.],record,fields))
    missing=fields.copy();del missing['J_pairs']
    c.reject('comparison missing field',lambda:source_comparison_index(r,r,[0.,640.,1472.],record,missing))
    partial=fields.copy();partial['eps_tot']=np.zeros((3,3))
    c.reject('comparison partial field',lambda:source_comparison_index(r,r,[0.,640.,1472.],record,partial))
    summary,arrays=difference_details([0.,1.,2.,3.],[0.,1.,1.,1.],[0.,2.,0.,4.],r)
    c.close('comparison zero exact status',summary['zero_scale_exact_indices'],[0])
    c.close('comparison zero nonzero status',summary['zero_scale_nonzero_indices'],[2])
    c.close('comparison normalized valid',arrays['normalized'][[0,1,3]],[0.,0.,.5])
    c.close('comparison unavailable ratio',np.isnan(arrays['normalized'][2]),True)
    c.reject('comparison finite subtraction overflow',lambda:difference_details([1e308],[-1e308],[1.],[1.]))
    c.reject('comparison finite ratio overflow',lambda:difference_details([1e308],[0.],[1e-308],[1.]))
    env={name:'1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']}
    validate_one_thread({'threads':1},env)
    c.close('runtime one-thread positive',True,True)
    c.reject('runtime resident-count guard',lambda:validate_one_thread({'threads':2},env))
    bad_env=env.copy();bad_env['MKL_NUM_THREADS']='8'
    c.reject('runtime backend cap guard',lambda:validate_one_thread({'threads':1},bad_env))
    identity={'path':'<internal>/combined_sources.npz','bytes':1,'sha256':'fixture'}
    complete={'products':[identity.copy()],'T':640.,'T_over_rs':5.}
    values={'T':np.array(640.),'rs':np.array(128.),'E0':np.array(235.17137214931623),'R':np.arange(1.,4097.)}
    validate_bound_comparison_source(complete,identity,values)
    c.close('bound combined positive association',True,True)
    unbound=copy.deepcopy(complete);unbound['products'][0]['path']='<internal>/other.npz'
    c.reject('unbound combined product',lambda:validate_bound_comparison_source(unbound,identity,values))
    for key,bad in [('T',641.),('rs',64.),('E0',235.)]:
        wrong=copy.deepcopy(values);wrong[key]=np.array(bad)
        c.reject('combined scalar '+key,lambda wrong=wrong:validate_bound_comparison_source(complete,identity,wrong))
    saved={'fixture_normalized':np.array([0.,np.nan]),'fixture_normalized_available':np.array([True,False]),'raw':np.array([1.])}
    validate_saved_arrays(saved);c.close('saved unavailable normalized branch',True,True)
    bad_saved=copy.deepcopy(saved);bad_saved['raw'][0]=np.inf
    c.reject('nonfinite raw saved source',lambda:validate_saved_arrays(bad_saved))
    bad_saved2=copy.deepcopy(saved);bad_saved2['fixture_normalized'][0]=np.inf
    c.reject('nonfinite available normalized source',lambda:validate_saved_arrays(bad_saved2))
    recorder=Checks();recorder.close('shape mutation',[1.],[1.,1.])
    c.close('check recorder shape guard',recorder.rows[0]['passed'],False)
    recorder=Checks();recorder.close('nonfinite expected',1.,np.nan)
    c.close('check recorder finite expected guard',recorder.rows[0]['passed'],False)


def gaussian_checks(c):
    ell=4.;normal=(2*np.pi*ell**2)**(-1.5)
    for r,R in [(0.,1.),(.0001,2.),(2.,4.),(12.,10.),(100.,100.),(-2.,4.)]:
        expected=[]
        for power in [0,1]:
            integral,error=quad(lambda mu:mu**power*np.exp(-(r*r+R*R-2*r*R*mu)/(2*ell**2))/2,
                                -1,1,epsabs=2e-14,epsrel=2e-13,limit=200)
            expected.append(integral)
        k0,k1=gaussian_kernels(r,R,ell)
        c.close(f'angular Gaussian r{r} R{R}',np.array([k0,k1])/normal,expected,tol=1e-10)
    for R in [0.,1.,4.,20.]:
        end=R+12*ell
        tail=float(erfc(12/np.sqrt(2))+np.sqrt(2/np.pi)*12*np.exp(-72))
        vector_tail=R*tail+ell*np.sqrt(2/np.pi)*(12**2+2)*np.exp(-72)
        scalar,error=quad(lambda r:4*np.pi*r*r*gaussian_kernels(r,R,ell)[0],0,end,
                          points=[R] if R else None,epsabs=2e-13,epsrel=2e-13)
        vector,errorv=quad(lambda r:4*np.pi*r**3*gaussian_kernels(r,R,ell)[1],0,end,
                           points=[R] if R else None,epsabs=2e-12,epsrel=2e-13)
        c.close(f'constant adjoint R{R}',scalar,1.,tol=1e-10,analytic_tail_allowance=tail,quadrature_report=error)
        c.close(f'linear vector adjoint R{R}',vector,R,tol=1e-10,analytic_tail_allowance=vector_tail,quadrature_report=errorv)
    R=np.array([1.,4.,12.]);e=np.array([.1,-.2,.3]);j=np.array([.03,.04,-.02])
    coeff=gaussian_centre_coefficients(e,j,R,ell)
    coeff1=gaussian_centre_coefficients([1.],[1.],[ell],ell);base=normal*np.exp(-.5)
    c.close('direct centre coefficients',np.array([coeff1['rho0'],ell**2*coeff1['rho2'],
               ell*coeff1['J1'],ell**3*coeff1['J3']])/base,[1.,-1/3,1/3,-2/15],tol=1e-10)
    c.reject('complex centre sources',lambda:gaussian_centre_coefficients([1j],[1.],[ell],ell))
    c.close('centre K1',gaussian_kernels(0.,R,ell)[1],[0.,0.,0.],tol=1e-10)
    for r in [1e-3,-1e-3]:
        k0,k1=gaussian_kernels(r,R,ell)
        c.close('centre rho series '+str(r),np.sum(e*k0),coeff['rho0']+coeff['rho2']*r*r,tol=1e-10)
        c.close('centre J series '+str(r),np.sum(j*k1),coeff['J1']*r+coeff['J3']*r**3,tol=1e-10)
    actual=gaussian_kernels(0.,4.,ell)[0]
    altered=gaussian_kernels(0.,4.,np.sqrt(ell**2+1/12))[0]
    c.close('variance mutation distinguished',abs(actual-altered)>1e-10,True)
    c.close('volume mutation distinguished',abs(actual*4*np.pi*4**2-actual)>1e-10,True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True);args=p.parse_args()
    runtime=require_source_cpu(args.output.parent)
    if args.output.exists():raise ValueError('Refusing implicit control replay/overwrite')
    c=Checks();source_checks(c);metadata_checks(c);gaussian_checks(c)
    bindings=source_file_bindings(args.spec)
    result={'scope':'manufactured inputs only','runtime':runtime,'bindings':bindings,
            'checks':c.rows,'passed':all(r['passed'] for r in c.rows),'count':len(c.rows)}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'count':len(c.rows),'passed':result['passed'],'output':str(args.output)}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
