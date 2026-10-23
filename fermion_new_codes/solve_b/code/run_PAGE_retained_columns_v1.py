"""One reviewed retained full-column control; no propagation or first-route readouts."""
import argparse,json,math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cut-structure',type=Path,required=True)
    a=p.parse_args();runtime=require_remote_cpu(a.output);a.output.mkdir(exist_ok=True,parents=True)
    import numpy as np
    import scipy.linalg as la
    import torch
    from flint import arb,ctx
    ctx.prec=128;ctx.threads=1
    underflow_budget=arb(2)**-1074*(16384*8192)**2
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def save(name,obj):
        (a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def up(value):return float(np.nextafter(float(value.upper()),math.inf))
    def gamma(n):
        u=arb(2)**-53
        return n*u/(1-n*u)
    def upper_sum(values):
        flat=np.asarray(values,dtype=np.float64).ravel()
        return up(arb(float(np.sum(flat,dtype=np.float64)))/(1-gamma(flat.size)))
    def norm_f_upper(matrix):
        # abs(z)^2 is evaluated as two real squares plus their sum.
        ss=arb(upper_sum(matrix.real**2+matrix.imag**2))/(1-gamma(3))+underflow_budget
        return up(ss.sqrt())
    def row_bound(matrix):
        # No reliance on a library complex absolute-value rounding bound.
        nonnegative=np.abs(matrix.real)+np.abs(matrix.imag)
        maximum=float(np.max(np.sum(nonnegative,axis=1)))
        return up(arb(maximum)/(1-gamma(matrix.shape[1]+1))+underflow_budget)
    def assembly_error(block,gram):
        inner=block.shape[0] if gram else block.shape[1]
        return up(2*gamma(2*inner+2)*arb(norm_f_upper(block))**2+underflow_budget)
    def matrix_info(matrix):
        return {'frobenius':float(la.norm(matrix,'fro')),'max_abs_entry':float(np.max(np.abs(matrix))),
                'max_abs_row_sum':float(np.max(np.sum(np.abs(matrix),axis=1))),
                'unsymmetrized_hermiticity_frobenius':float(la.norm(matrix-matrix.conj().T,'fro'))}
    def entropy(values):
        clipped=np.clip(values,0,1);positive=clipped[(clipped>0)&(clipped<1)]
        total=float(np.sum(-positive*np.log(positive)-(1-positive)*np.log1p(-positive)))
        displacement=float(np.sum(np.abs(values-clipped)));d=len(values)
        t=min(displacement/d,.5) if d else 0.
        allowance=d*(-t*math.log(t)-(1-t)*math.log1p(-t)) if t>0 else 0.
        excursion=float(max(0.,-np.min(values),np.max(values)-1)) if d else 0.
        return {'evaluated_clipped_entropy':total if excursion<=1e-8 else None,'n_eigenvalues':d,'minimum':float(np.min(values)) if d else None,
                'maximum':float(np.max(values)) if d else None,'outside_count':int(np.count_nonzero((values<0)|(values>1))),
                'maximum_excursion':excursion,'endpoint_tolerance':1e-8,'within_endpoint_tolerance':excursion<=1e-8,
                'clipping_displacement':displacement,'clipping_continuity_allowance':allowance,
                'continuity_scope':'Evaluation diagnostic only; raw covariance and unbounded eigensolver error are not certified.'}
    def eigen_checks(H,values,Q,eta_A):
        M=H@Q;L=Q*values;J=Q.conj().T@Q
        R=M-L;O=J-np.eye(len(Q));u=arb(2)**-53
        delta_M=2*gamma(2*len(Q)+2)*norm_f_upper(H)*norm_f_upper(Q)+underflow_budget
        delta_L=u*norm_f_upper(Q)*float(np.max(np.abs(values)))+underflow_budget
        r=up(arb(norm_f_upper(R))+delta_M+delta_L+u*(arb(norm_f_upper(M))+norm_f_upper(L))+underflow_budget)
        delta_J=2*gamma(2*len(Q)+2)*arb(norm_f_upper(Q))**2+underflow_budget
        e=up(arb(norm_f_upper(O))+delta_J+u*(arb(norm_f_upper(J))+arb(len(Q)).sqrt())+underflow_budget)
        eig_bound=None
        if e<1:
            dq=arb(e)/(1+(1-arb(e)).sqrt())
            eig_bound=up(arb(eta_A)+r+(arb(norm_f_upper(H))+float(np.max(np.abs(values))))*dq)
        checks={'residual_frobenius':float(la.norm(R,'fro')),
            'orthogonality_frobenius':float(la.norm(O,'fro')),'residual_upper_bound':r,
            'orthogonality_upper_bound':e,'full_sorted_eigenvalue_error_bound':eig_bound}
        return checks,R,O
    def identity(Z,inside):
        Zi=Z[:inside];Zo=Z[inside:];G=Z.conj().T@Z
        A=Zi@Zi.conj().T;K=Zi@Zo.conj().T;B=Zi@(np.eye(G.shape[0])-G)@Zi.conj().T
        terms={'A_minus_A2':A-A@A,'KKdagger':K@K.conj().T,'impurity':B}
        residual=terms['A_minus_A2']-terms['KKdagger']-B
        scale=max(1.,*(la.norm(x,'fro') for x in terms.values()))
        return {'terms':{k:matrix_info(v) for k,v in terms.items()},'residual_frobenius':float(la.norm(residual,'fro')),
                'residual_spectral_norm_diagnostic':float(la.norm(residual,2)),
                'relative_residual_frobenius':float(la.norm(residual,'fro')/scale)}
    controls={}
    for f in (1.,.37):
        Z=np.sqrt(f/2)*np.ones((2,1),complex)
        H=Z[:1]@Z[:1].conj().T;ev,Q=la.eigh(H)
        checks,_,_=eigen_checks(H,ev,Q,assembly_error(Z[:1],False))
        controls[str(f)]={'full_eigenvector_checks':checks,'identity':identity(Z,1),'row_entropy':entropy(ev),
                         'gram_entropy':entropy(la.eigvalsh(Z[:1].conj().T@Z[:1]))}
        if controls[str(f)]['identity']['residual_frobenius']>1e-13:raise RuntimeError('Known identity failed')
        expected=entropy(np.array([f/2]))['evaluated_clipped_entropy']
        if abs(controls[str(f)]['row_entropy']['evaluated_clipped_entropy']-expected)>1e-13:raise RuntimeError('Known entropy failed')
    controls['empty']=entropy(np.zeros(2));controls['log2']=math.log(2)
    P=np.array([[.5,.5],[.5,.5]],complex)
    controls['independent_reference_cut']=entropy(la.eigvalsh(P[:1,:1]))
    controls['self_vacuum_excess']=controls['1.0']['row_entropy']['evaluated_clipped_entropy']-controls['independent_reference_cut']['evaluated_clipped_entropy']
    if controls['empty']['evaluated_clipped_entropy']!=0:raise RuntimeError('Empty-state entropy failed')
    if abs(controls['1.0']['row_entropy']['evaluated_clipped_entropy']-math.log(2))>1e-13:raise RuntimeError('log2 known answer failed')
    if abs(controls['self_vacuum_excess'])>1e-13:raise RuntimeError('Self-vacuum known answer failed')
    save('known_answers.json',controls)
    allowed=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
             (np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.dtype,np.ndarray]
    allowed.extend(type(np.dtype(n)) for n in ('float64','float32','complex128','complex64','int64','int32','uint8','bool'))
    with torch.serialization.safe_globals(allowed):data=torch.load(a.input,map_location='cpu',weights_only=True,mmap=True)
    Z=data['Phi'].numpy()
    if Z.shape!=(16384,8192) or Z.dtype!=np.complex128:raise RuntimeError('Only the specified full-column control is authorized')
    if not np.isfinite(Z).all():raise RuntimeError('Nonfinite columns')
    metadata={k:data[k] for k in ('k','t','t_over_rs','rs_nominal','nsites','n2','n_occ','layout','what','record_json','fields_npz') if k in data}
    expected={'kappa':18,'k':96,'t':768,'t_over_rs':12,'rs_nominal':64,'nsites':8192,'n2':16384,'n_occ':8192}
    if any(data.get(key)!=value for key,value in expected.items()):raise RuntimeError('Metadata differs from specified control')
    cut_structure=json.loads(a.cut_structure.read_text())
    if cut_structure['kappa']!=18 or cut_structure['clock']!={'k':96,'t':768.,'t_over_rs':12.}:raise RuntimeError('Cut metadata clock mismatch')
    parent=cut_structure['parent']
    if parent['record']!=data['record_json'] or parent['fields']!=data['fields_npz'] or parent['fields_key']!='applied' or parent['nsites']!=8192:
        raise RuntimeError('Cut metadata parent mismatch')
    for cut,inside in ((2,256),(30,3840)):
        c=cut_structure['cuts'][str(cut)]
        if c['R_abs']!=64*cut or c['n_modes_Xc']!=inside or c['n_modes_X']!=len(Z)-inside or c['n_sites_Xc']!=inside//2:
            raise RuntimeError('Geometric cut dimensions mismatch')
    save('cut_structure.json',cut_structure)
    save('input.json',{'source':str(a.input),'payload_key':'Phi','shape':list(Z.shape),'dtype':str(Z.dtype),'metadata':metadata,'runtime':runtime})
    G=Z.conj().T@Z;np.save(a.output/'global_gram.npy',G)
    skew=float(la.norm(G-G.conj().T,'fro'));H=(G+G.conj().T)/2
    defect=H.copy();defect.flat[::len(G)+1]-=1
    info=matrix_info(defect);zfn=norm_f_upper(Z)
    info.update({'trace_real':float(np.trace(G).real),'trace_imag':float(np.trace(G).imag),
        'accumulation_dtype':'complex128','column_frobenius_upper':zfn,
        'product_operator_roundoff_allowance':assembly_error(Z,True),
        'symmetrization_allowance':up(arb(2)**-53*norm_f_upper(G)),
        'unsymmetrized_gram_hermiticity_frobenius':skew,
        'global_operator_defect_bound':None,'physical_covariance_admission':'unresolved pending conservative product and spectral bounds',
        'declared_occupied_projector_admission':False})
    subtraction_allowance=up(arb(2)**-53*(arb(norm_f_upper(H))+arb(len(G)).sqrt())+underflow_budget)
    eta=arb(info['product_operator_roundoff_allowance'])+info['symmetrization_allowance']+underflow_budget
    info['global_operator_defect_bound']=up(arb(min(norm_f_upper(defect),row_bound(defect)))+subtraction_allowance+eta)
    info['declared_occupied_projector_admission']=info['global_operator_defect_bound']<=1e-8
    info['isometry_status']='consistent to tolerance' if info['declared_occupied_projector_admission'] else 'unresolved'
    info['impurity_upper_bound']=up(arb(info['global_operator_defect_bound'])*(1+arb(info['global_operator_defect_bound'])))
    off=H.copy();off.flat[::len(G)+1]=0
    info['exact_gram_upper_bound']=up(arb(float(np.max(H.diagonal().real)))+row_bound(off)+eta)
    info['physical_covariance_admission']='certified' if info['exact_gram_upper_bound']<=1 else 'unresolved'
    info['underflow_allowance']=up(underflow_budget)
    # The deliberately coarse underflow allowance dominates all product/reduction counts here.
    save('global_covariance.json',info);del defect,H,off
    # Products A, K and impurity are independent; no identity term is obtained by subtraction.
    Zi=Z[:256];Zo=Z[256:]
    A=Zi@Zi.conj().T;K=Zi@Zo.conj().T
    impurity=Zi@(np.eye(len(G))-G)@Zi.conj().T;KK=K@K.conj().T;lhs=A-A@A
    residual=lhs-KK-impurity
    np.savez(a.output/'stored_identity.npz',A=A,K=K,impurity=impurity,KKdagger=KK,residual=residual)
    scale=max(1.,*(la.norm(x,'fro') for x in (lhs,KK,impurity)))
    save('identity.json',{'interior_rows':256,'cut':'R<=128a','terms':{k:matrix_info(v) for k,v in
        {'A_minus_A2':lhs,'KKdagger':KK,'impurity':impurity}.items()},'residual_frobenius':float(la.norm(residual,'fro')),
        'residual_spectral_norm_diagnostic':float(la.norm(residual,2)),
        'relative_residual_frobenius':float(la.norm(residual,'fro')/scale)})
    del K,KK,impurity,lhs,residual,Zo,Zi
    rows=[]
    for cut,inside in ((2,256),(30,3840)):
        row={'cut_over_rs':cut,'interior_coordinates':inside,'exterior_coordinates':len(Z)-inside,
             'cut_rule':'I={R<=r_cut}, X={R>r_cut}','channel':18,'t_over_rs':12,
             'S_state':None,'S_reference':None,'excess':None,'weight':288,
             'physical_scope':'Diagnostic control; physical entropy is not released. See recorded global and spectral admission statuses.'}
        for side,block in (('interior',Z[:inside]),('exterior',Z[inside:])):
            gram=block.shape[0]>block.shape[1]
            M=block.conj().T@block if gram else block@block.conj().T
            prefix=f'cut{cut}_{side}';np.save(a.output/(prefix+'_matrix.npy'),M)
            diag=matrix_info(M);eta_A=up(arb(assembly_error(block,gram))+arb(2)**-53*norm_f_upper(M)+underflow_budget)
            H=(M+M.conj().T)/2;del M
            if cut==2 and side=='interior':
                values,Q=la.eigh(H,check_finite=False,driver='evr')
                diag['full_eigenvector_checks'],R,O=eigen_checks(H,values,Q,eta_A)
                np.savez(a.output/(prefix+'_eigenvectors_checks.npz'),Q=Q,residual=R,orthogonality=O)
                del Q,R,O
            else:
                values=la.eigvalsh(H,overwrite_a=True,check_finite=False,driver='evr')
            del H
            np.save(a.output/(prefix+'_eigenvalues.npy'),values)
            diag.update(entropy(values));diag.update({'representation':'occupied Gram' if gram else 'row covariance',
                'assembly_error_upper_bound':eta_A,'eigensolver':'scipy.linalg.eigh/eigvalsh driver evr','eigensolver_full_spectrum_residual_bound':diag.get('full_eigenvector_checks',{}).get('full_sorted_eigenvalue_error_bound'),
                'eigensolver_scope':'Full eigenvector checks with assembly/product allowances.' if cut==2 and side=='interior' else 'Diagnostic eigenvalues only; no full-spectrum residual enclosure.'})
            row[side]=diag;save(prefix+'.json',diag)
            print(json.dumps({'completed':prefix,'diagnostic_entropy':diag['evaluated_clipped_entropy']}),flush=True)
        sx=row['exterior']['evaluated_clipped_entropy'];si=row['interior']['evaluated_clipped_entropy']
        row['diagnostic_side_gap']=sx-si if sx is not None and si is not None else None
        rows.append(row);save('rows.json',rows)
    save('result.json',{'scope':'One full-coordinate retained control only; no first-route comparison.','global_covariance':info,'rows':rows})


if __name__=='__main__':main()
