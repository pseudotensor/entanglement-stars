"""Independent saved-column contractions; EVOLVEDCUT specification, no propagation."""
import argparse
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--times',default='12,15,16,20,24,25,28,30,35,40')
    parser.add_argument('--channels',default='1,2,3,4,5,6,7,8,9,10')
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args()
    runtime=require_remote_cpu(args.output)
    import numpy as np
    from scipy.linalg import eig_banded, eigh, eigvalsh
    from scipy.sparse import coo_matrix
    eps=np.finfo(float).eps
    s1=np.array([[0,1],[1,0]],complex)
    s2=np.array([[0,-1j],[1j,0]],complex)
    s3=np.diag([1.,-1.])

    def encode_scalar(value):
        if isinstance(value,np.generic): return value.item()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    def write(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=encode_scalar)+'\n')
    spectral_checks={}
    def spectrum(a,key=None,measured=False):
        ev=eigvalsh(a,check_finite=False,driver='evr')
        if key is not None:
            check={'dimension':len(ev),'eigenpair_frobenius_residual':None,
                   'eigenvector_orthogonality_frobenius':None,
                   'representative_control_id':'T12_k1' if current_dtype=='complex128' else 'T15_k1'}
            if measured or len(a)<=256:
                e,u=eigh(a,check_finite=False,driver='evr')
                residual=a@u-u*e[None,:]
                check.update({'eigenpair_frobenius_residual':float(np.linalg.norm(residual)),
                              'eigenpair_max_column_residual':float(np.max(np.linalg.norm(residual,axis=0),initial=0)),
                              'eigenvector_orthogonality_frobenius':float(np.linalg.norm(u.conj().T@u-np.eye(len(e)))),
                              'eigvalsh_eigh_max_difference':float(np.max(np.abs(ev-e),initial=0))})
            spectral_checks[key]=check
            write(destination/'spectral_checks.json',spectral_checks)
        return ev
    def norms_hermitian(a):
        ev=spectrum(a)
        return {'operator':float(np.max(np.abs(ev),initial=0)),'frobenius':float(np.linalg.norm(a))}
    def entropy(ev):
        ev=np.asarray(ev)
        scale=max(1.,float(np.max(np.abs(ev),initial=0)))
        tol=64*len(ev)*eps*scale
        excursion=np.maximum(-ev,0)+np.maximum(ev-1,0)
        rejected=bool(np.max(excursion,initial=0)>tol)
        borderline=bool(abs(float(ev.min(initial=1))+tol)<=tol/2 or
                        abs(float(ev.max(initial=0))-(1+tol))<=tol/2)
        invalid=rejected or borderline
        x=np.clip(ev,0,1)
        mid=x[(x>0)&(x<1)]
        return {'entropy':None if invalid else float(np.sum(-mid*np.log(mid)-(1-mid)*np.log1p(-mid))),
                'dimension':len(ev),'minimum':float(ev.min(initial=1)),'maximum':float(ev.max(initial=0)),
                'rounding_tolerance':tol,'outside_count':int(np.count_nonzero(excursion)),
                'clipping_displacement':float(np.sum(excursion)),'valid':not invalid,
                'covariance_gate':'unresolved' if borderline else 'rejected' if rejected else 'accepted',
                'borderline_policy':'leave unresolved; no beta adjudication performed'}

    def operator(r,n,v,k,m=.2,periodic=False):
        size=len(r); rows=[]; cols=[]; values=[]
        def block(i,j,b):
            for al in range(2):
                for be in range(2):
                    if b[al,be]!=0:
                        rows.append(2*i+al); cols.append(2*j+be); values.append(b[al,be])
        for j in range(size): block(j,j,n[j]*(m*s3+k/r[j]*s2))
        for step,pauli,den,field in ((1,s1,4j,n),(2,np.eye(2),8j,v)):
            for j in range(size if periodic else size-step):
                other=(j+step)%size
                b=(field[j]+field[other])/den*pauli
                block(j,other,b); block(other,j,b.conj().T)
        h=coo_matrix((values,(rows,cols)),shape=(2*size,2*size)).tocsr()
        if periodic: return h
        band=np.zeros((5,2*size),complex)
        for off in range(5):
            diag=h.diagonal(-off)
            band[off,:len(diag)]=diag
        reconstructed=coo_matrix(h.shape,dtype=complex).tocsr()
        from scipy.sparse import diags
        for off in range(5):
            d=band[off,:2*size-off]
            reconstructed+=diags(d,-off,shape=h.shape)
            if off: reconstructed+=diags(d.conj(),off,shape=h.shape)
        difference=h-reconstructed
        defect=float(np.max(np.abs(difference.data),initial=0))
        if defect>1e-14: raise RuntimeError('Banded block reconstruction failed')
        return h,band,defect

    def diagonalize(r,n,v,k):
        h,band,defect=operator(r,n,v,k)
        e,u=eig_banded(band,lower=True,check_finite=False,overwrite_a_band=True)
        residual=h@u-u*e[None,:]
        maxcol=float(np.max(np.linalg.norm(residual,axis=0)))
        orth=float(np.linalg.norm(u.conj().T@u-np.eye(len(e))))
        error=float(np.linalg.norm(residual))
        negative=e<0
        report={'negative_count':int(negative.sum()),'nearest_zero':float(np.min(np.abs(e))),
                'zero_sign_unresolved':bool(np.min(np.abs(e))<=error),
                'eigenpair_max_column_residual':maxcol,'eigenpair_frobenius_residual':error,
                'eigenvector_orthogonality_frobenius':orth,'eigenvector_orthogonality_tolerance':64*len(e)*eps,
                'eigenvector_isometry_accepted':bool(orth<=64*len(e)*eps),'band_reconstruction_max':defect}
        return e,u[:,negative],report

    # Known answers are evaluated before any production slice is opened.
    r=np.arange(1.,9.)
    h,band,_=operator(r,np.full(8,1.3),np.zeros(8),0)
    actual=eig_banded(band,lower=True,eigvals_only=True)
    expected=np.sort(np.r_[-1.3*np.sqrt(.2**2+np.cos(np.pi*r/9)**2),1.3*np.sqrt(.2**2+np.cos(np.pi*r/9)**2)])
    open_error=float(np.max(np.abs(actual-expected)))
    r=np.arange(1.,17.)
    periodic=operator(r,np.full(16,.9),np.full(16,-.4),0,periodic=True).toarray()
    p=2*np.pi*np.arange(16)/16
    expected=np.sort(np.r_[-.4*np.sin(2*p)/2-.9*np.sqrt(.2**2+np.sin(p)**2),
                           -.4*np.sin(2*p)/2+.9*np.sqrt(.2**2+np.sin(p)**2)])
    periodic_error=float(np.max(np.abs(spectrum(periodic)-expected)))
    flat_h,flat_band,_=operator(r,np.ones(16),np.zeros(16),3)
    e,u=eigh(flat_h.toarray())
    _,u2=eigh(1.7*flat_h.toarray())
    pp=u[:,e<0]@u[:,e<0].conj().T
    p2=u2[:,:int((e<0).sum())]@u2[:,:int((e<0).sum())].conj().T
    scale_error=float(np.linalg.norm(pp-p2))
    vacuum_a=entropy(spectrum(pp[:8,:8]))
    vacuum_b=entropy(spectrum(u[:8,e<0]@u[:8,e<0].conj().T))
    vacuum_excess=vacuum_a['entropy']-vacuum_b['entropy']
    controls={'open_constant_clock_max_error':open_error,'periodic_constant_fields_max_error':periodic_error,
              'flat_mass_gap':float(np.min(np.abs(e))),'clock_rescaling_projector_frobenius':scale_error,
              'vacuum_self_excess':vacuum_excess,'vacuum_cut_entropy':vacuum_a['entropy'],
              'two_mode_log2_error':abs(entropy(np.array([.5]))['entropy']-np.log(2))}
    write(args.output/'serialization_control.json',{'numpy_boolean':np.bool_(True),
          'numpy_float':np.float64(1),'numpy_integer':np.int64(1)})
    write(args.output/'operator_controls.json',controls)
    if max(open_error,periodic_error,scale_error,abs(vacuum_excess))>1e-10 or np.min(np.abs(e))<.2-1e-12:
        raise RuntimeError('Operator known-answer failure')
    print(json.dumps({'controls':'passed',**controls}),flush=True)
    if args.controls_only: return

    # Isolate primitive metadata; never read saved entropy or first-route comparisons.
    source=json.loads(args.metadata.read_text())
    for row in source['rows']:
        if row['history']=='prod' and (row['m']!=.2 or row['g']!=.25 or row['sigma_over_a']!=4
                                      or row['tau']!=8 or row['rs']!=64 or row['nsites']!=2048):
            raise ValueError('Production metadata differs from the specified parent')
    records={float(row['T_over_rs']):{'E':float(row['record']['E']),
             'source_slice':row['slice']} for row in source['rows'] if row['history']=='prod'}
    flat_cache={}
    rows=[]
    unavailable_slices=[]
    for clock in map(float,args.times.split(',')):
        path=args.inputs/f'orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T{clock:07.2f}.npz'
        with np.load(path,allow_pickle=False) as data:
            fields=[np.asarray(data[key]) for key in ('R','N_applied','v_applied')]
            if any(np.iscomplexobj(field) and np.any(field.imag!=0) for field in fields):
                raise ValueError('Applied fields/grid are not real')
            r,n,v=[np.asarray(field.real,float) for field in fields]
            rs=float(data['rs']); actual_clock=float(data['T'])/rs
            if actual_clock!=clock or rs!=64 or not np.array_equal(r,np.arange(1,2049)):
                raise ValueError('Stored grid/clock mismatch')
            if not np.isfinite(n).all() or not np.isfinite(v).all() or np.any(n<=0):
                raise ValueError('Invalid applied fields')
            f=n-np.abs(v)
            roots=[]; ambiguous=[]
            for j in range(len(r)-1):
                if f[j]<0 and f[j+1]>0:
                    roots.append({'radius':float(r[j]-f[j]*(r[j+1]-r[j])/(f[j+1]-f[j])),
                                  'bracket':[int(j),int(j+1)],'values':[float(f[j]),float(f[j+1])]})
                if f[j]==0:
                    if j>0 and f[j-1]<0 and f[j+1]>0:
                        roots.append({'radius':float(r[j]),'bracket':[j-1,j+1],'values':[float(f[j-1]),float(f[j+1])]})
                    else: ambiguous.append(j)
            if f[-1]==0: ambiguous.append(len(f)-1)
            selected=roots[-1]['radius'] if roots else None
            geometry_reason=None
            if any(selected is None or r[j]>=selected for j in ambiguous):
                geometry_reason='ambiguous zero at or beyond the selected outer root'
            if selected is None and clock!=12: geometry_reason='no applied root; fallback not specified for this slice'
            if f[-1]<0: geometry_reason='outer trapped interval reaches the wall without a resolved boundary'
            if selected is not None and np.any(v[roots[-1]['bracket']]>0):
                geometry_reason='selected outer-root neighborhood has outward flow'
            if geometry_reason:
                failure={'clock_rs':clock,'source':str(path),'roots':roots,'ambiguous_zero_indices':ambiguous,
                         'available':False,'reason':geometry_reason}
                unavailable_slices.append(failure)
                write(args.output/f'selection_T{clock:g}.json',failure)
                print(json.dumps(failure),flush=True)
                continue
            radius=selected if selected is not None else 2*.25*records[clock]['E']
            count=int(np.count_nonzero(r<=radius)); area=float(4*np.pi*radius**2)
            selection={'clock_rs':clock,'stored_T':float(data['T']),'source':str(path),'roots':roots,
                       'ambiguous_zero_indices':ambiguous,'applied_root_present':selected is not None,
                       'cut_label':'applied_root' if selected is not None else 'charge_fallback',
                       'radius_over_a':radius,'area_over_a2':area,'eta_input':.9192,'eta_area':.9192*area,
                       'E_a':records[clock]['E'],'g':.25,'interior_site_count':count,
                       'partition_bond_midpoint':float((r[count-1]+r[count])/2)}
            write(args.output/f'selection_T{clock:g}.json',selection)
            np.savez_compressed(args.output/f'fields_T{clock:g}.npz',R=r,N_applied=n,v_applied=v)
            for k in map(int,args.channels.split(',')):
                destination=args.output/f'T{clock:g}_k{k}'
                if (destination/'result.json').exists():
                    rows.append(json.loads((destination/'result.json').read_text())); continue
                destination.mkdir(exist_ok=True)
                ks=np.asarray(data['kappas']); weights=np.asarray(data['weights'])
                if float(weights[np.flatnonzero(ks==k)[0]])!=16*k: raise ValueError('Multiplicity mismatch')
                sea=np.asarray(data[f'sea_{k}'],complex); dust=np.asarray(data[f'dust_{k}'],complex)
                occ=np.asarray(data[f'f_{k}'],float)
                dtype=str(data[f'sea_{k}'].dtype)
                if str(data['dtype'].item())!=dtype:
                    raise ValueError('Declared column storage dtype mismatch')
                current_dtype=dtype
                spectral_checks={}
                representative=k==1 and clock in (12,15)
                if sea.shape[0]!=4096 or dust.shape!=(4096,len(occ)) or np.any(occ<0) or np.any(occ>1):
                    raise ValueError('Column/occupation contract failure')
                if not all(np.isfinite(x).all() for x in (sea,dust,occ)): raise ValueError('Nonfinite columns')
                tc=np.column_stack((sea,dust)); gram=tc.conj().T@tc
                ge,gu=eigh(gram,check_finite=False,driver='evr')
                gram_eigenpair_residual=float(np.linalg.norm(gram@gu-gu*ge[None,:]))
                gram_eigenvector_orthogonality=float(np.linalg.norm(gu.conj().T@gu-np.eye(len(ge))))
                np.savez_compressed(destination/'gram_diagnostics.npz',joint_gram_eigenvalues=ge,occupations=occ)
                gram_scale=max(1.,float(np.max(np.abs(ge))))
                gram_tolerance=64*max(tc.shape)*eps*gram_scale
                gram_defect_frobenius=float(np.linalg.norm(gram-np.eye(len(ge))))
                resolved_gram=bool(1-gram_defect_frobenius-gram_tolerance>0 and ge.min()>gram_tolerance)
                q=(tc@gu/np.sqrt(ge)[None,:])@gu.conj().T if ge.min()>0 else None
                q_residual=float(np.linalg.norm(q.conj().T@q-np.eye(len(ge)))) if q is not None else None
                q_tolerance=64*max(tc.shape)*eps
                polar_accepted=resolved_gram and q_residual is not None and q_residual<=q_tolerance
                occupation=np.r_[np.ones(sea.shape[1]),occ]
                z=tc*np.sqrt(occupation)[None,:]
                zq=q*np.sqrt(occupation)[None,:] if q is not None else None
                gs=gram[:sea.shape[1],:sea.shape[1]]-np.eye(sea.shape[1])
                gd=gram[sea.shape[1]:,sea.shape[1]:]-np.eye(len(occ))
                cross=gram[:sea.shape[1],sea.shape[1]:]
                orth={'sea':norms_hermitian(gs),'dust':norms_hermitian(gd),
                      'cross_frobenius':float(np.linalg.norm(cross)),
                      'cross_operator':float(np.linalg.norm(cross,2)) if cross.size else 0.,
                      'joint_eigenvalue_min':float(ge.min()),'joint_eigenvalue_max':float(ge.max()),
                      'joint_eigenpair_frobenius_residual':gram_eigenpair_residual,
                      'joint_eigenvector_orthogonality_frobenius':gram_eigenvector_orthogonality,
                      'joint_defect_frobenius':gram_defect_frobenius,'gram_tolerance':gram_tolerance,
                      'resolved_positive_gram':resolved_gram,'polar_accepted':polar_accepted,
                      'polar_residual_tolerance':q_tolerance,
                      'joint_defect_operator':float(np.max(np.abs(ge-1))),
                      'orthonormal_control_frobenius':q_residual,
                      'covariance_distance_upper':float(np.max(np.abs(np.sqrt(ge)-1))*(np.sqrt(ge.max())+1)) if ge.min()>0 else None}
                write(destination/'orthogonality.json',orth)
                raw_global=spectrum(z.conj().T@z,'raw_global',representative)
                raw_global_report=entropy(raw_global)
                np.save(destination/'raw_global_spectrum.npy',raw_global)
                control_global=spectrum(zq.conj().T@zq,'polar_global',representative) if zq is not None else None
                control_global_report=entropy(control_global) if control_global is not None else {'valid':False,'entropy':None}
                if control_global is not None: np.save(destination/'polar_global_spectrum.npy',control_global)
                write(destination/'global_covariance.json',{'raw':raw_global_report,'polar_control':control_global_report})
                e,held,held_info=diagonalize(r,n,v,k)
                write(destination/'held_operator.json',held_info)
                np.save(destination/'held_operator_eigenvalues.npy',e)
                if k not in flat_cache:
                    basis_path=args.output/f'flat_k{k}_basis.npy'
                    info_path=args.output/f'flat_k{k}_operator.json'
                    if basis_path.exists() and info_path.exists():
                        flat=np.load(basis_path,mmap_mode='r',allow_pickle=False)
                        flat_info=json.loads(info_path.read_text())
                    else:
                        fe,flat,flat_info=diagonalize(r,np.ones(len(r)),np.zeros(len(r)),k)
                        np.save(args.output/f'flat_k{k}_eigenvalues.npy',fe)
                        np.save(basis_path,flat)
                        write(info_path,flat_info)
                    flat_cache[k]=(flat,flat_info)
                flat,flat_info=flat_cache[k]
                entropies=[]
                for shift in (-1,0,1):
                    split=2*(count+shift)
                    row={'site_shift':shift,'interior_sites':count+shift,
                         'partition_bond_midpoint':float((r[count+shift-1]+r[count+shift])/2),
                         'normalization_area_over_a2':area,'normalization_eta_area':.9192*area,
                         'normalization_radius_is_primary':True}
                    for label,cols in (('raw',z),('orthonormal_control',zq)):
                        if cols is None:
                            row[label]={'exterior':{'entropy':None,'valid':False},'interior':{'entropy':None,'valid':False},
                                        'physical_exterior_entropy':None,'physical_interior_entropy':None,
                                        'reason':'polar control unresolved; no rank truncation'}
                            continue
                        exterior=spectrum(cols[split:].conj().T@cols[split:],f'{label}_exterior_shift{shift}',representative and shift==0)
                        interior=spectrum(cols[:split]@cols[:split].conj().T,f'{label}_interior_shift{shift}')
                        np.savez_compressed(destination/f'{label}_cut{count+shift}.npz',exterior=exterior,interior=interior)
                        row[label]={'exterior':entropy(exterior),'interior':entropy(interior)}
                        row[label]['global_raw_physical']=raw_global_report['valid'] if label=='raw' else None
                        accepted=raw_global_report['valid'] if label=='raw' else polar_accepted and control_global_report['valid']
                        row[label]['physical_exterior_entropy']=row[label]['exterior']['entropy'] if accepted else None
                        row[label]['physical_interior_entropy']=row[label]['interior']['entropy'] if accepted else None
                    for label,cols,info in (('held',held,held_info),('flat',flat,flat_info)):
                        ev=spectrum(cols[:split]@cols[:split].conj().T,f'{label}_shift{shift}')
                        np.save(destination/f'{label}_cut{count+shift}.npy',ev)
                        row[label]=entropy(ev)
                        if info['zero_sign_unresolved'] or not info['eigenvector_isometry_accepted']:
                            row[label]['entropy']=None;row[label]['valid']=False
                    entropies.append(row)
                    write(destination/'cuts.json',entropies)
                identity=None
                if k==1 and clock in (12,15):
                    a=z[:2*count]@z[:2*count].conj().T
                    kk=z[:2*count]@z[2*count:].conj().T
                    source_rows=z[:2*count]@z.conj().T
                    impurity=a-source_rows@source_rows.conj().T
                    residual=a-a@a-kk@kk.conj().T-impurity
                    identity={'residual':norms_hermitian(residual),
                              'relative_frobenius':float(np.linalg.norm(residual)/max(1.,np.linalg.norm(a-a@a))),
                              'impurity':norms_hermitian(impurity)}
                    np.savez_compressed(destination/'identity.npz',A=a,KK=kk@kk.conj().T,impurity=impurity,residual=residual)
                result={**selection,'kappa':k,'weight':16*k,'source_dtype':dtype,'occupation_min':float(occ.min()) if len(occ) else None,
                        'occupation_max':float(occ.max()) if len(occ) else None,'dust_rank':len(occ),'orthogonality':orth,
                        'raw_global':raw_global_report,'polar_global':control_global_report,
                        'ideal_polar_global_entropy':entropy(occupation)['entropy'] if polar_accepted else None,
                        'held_operator':held_info,'flat_operator':flat_info,'identity':identity,'cuts':entropies,
                        'spectral_checks':spectral_checks}
                write(destination/'result.json',result); rows.append(result)
                write(args.output/'results.json',{'runtime':runtime,'rows':rows,'unavailable_slices':unavailable_slices})
                print(json.dumps({'completed':str(destination),'raw_physical':result['raw_global']['valid']}),flush=True)
    write(args.output/'results.json',{'runtime':runtime,'rows':rows,'unavailable_slices':unavailable_slices})


if __name__=='__main__': main()
