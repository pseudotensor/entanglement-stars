"""Independent finite-difference implementation of the frozen curvature specification.

No input histories or foreign numerical implementations are imported here.
"""
import math
import numpy as np


def ordered_grid(x, minimum=3):
    x = np.asarray(x)
    if x.ndim != 1 or len(x) < minimum or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError('malformed coordinate grid')
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('nonfinite, duplicate or unordered coordinate')
    return x


def weights_at(nodes, at, derivative):
    nodes = np.asarray(nodes, dtype=float)
    if len(np.unique(nodes)) != len(nodes) or not np.all(np.isfinite(nodes)):
        raise ValueError('invalid interpolation abscissae')
    scale = float(np.max(np.abs(nodes-at)))
    if not scale > 0 or derivative >= len(nodes):
        raise ValueError('insufficient interpolation stencil')
    z = (nodes-at)/scale
    matrix = np.stack([z**k for k in range(len(nodes))])
    rhs = np.zeros(len(nodes)); rhs[derivative] = math.factorial(derivative)
    return np.linalg.solve(matrix, rhs)/scale**derivative


def derivative_table(x, derivative, width=3):
    x = ordered_grid(x, max(width, 4 if derivative == 2 and width == 3 else width))
    indices, weights, centered, widths = [], [], [], []
    maximum_width = max(width, 4 if derivative == 2 and width == 3 else width)
    for i in range(len(x)):
        actual = 4 if derivative == 2 and width == 3 and i in (0, len(x)-1) else width
        start = min(max(i-actual//2, 0), len(x)-actual)
        ids = np.arange(start, start+actual)
        w = weights_at(x[ids], x[i], derivative)
        pad = maximum_width-actual
        indices.append(np.pad(ids, (0, pad), constant_values=ids[-1]))
        weights.append(np.pad(w, (0, pad)))
        centered.append(actual % 2 == 1 and i == ids[actual//2])
        widths.append(actual)
    return {'indices': np.asarray(indices), 'weights': np.asarray(weights),
            'abscissae': x[np.asarray(indices)], 'at': x,
            'centered': np.asarray(centered), 'widths': np.asarray(widths),
            'derivative': np.asarray(derivative), 'nominal_width': np.asarray(width)}


def differentiate(values, table, axis):
    values = np.moveaxis(np.asarray(values, dtype=float), axis, -1)
    selected = np.take(values, table['indices'], axis=-1)
    # Ignore explicitly padded zero-weight terms, including possible NaNs there.
    products = np.where(table['weights'] != 0,
                        selected*table['weights'], 0.)
    answer = np.sum(products, axis=-1)
    return np.moveaxis(answer, -1, axis)


def prefix(local):
    state = True; output = []
    for value in local:
        if state is False or value is False:
            state = False
        elif state is None or value is None:
            state = None
        output.append(state)
    return output


def maximum(values, radii, selected=None):
    values, radii = np.asarray(values), np.asarray(radii)
    if selected is None:
        selected = np.ones(len(values), dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    valid = selected & np.isfinite(values)
    if not np.any(valid):
        return {'value': None, 'indices': [], 'radii': [], 'partial': bool(np.any(selected)),
                'available_points': 0, 'requested_points': int(np.count_nonzero(selected))}
    peak = float(np.max(values[valid]))
    ids = np.flatnonzero(valid & (values == peak))
    return {'value': peak, 'indices': ids.tolist(), 'radii': radii[ids].tolist(),
            'partial': bool(np.any(selected & ~np.isfinite(values))),
            'available_points': int(np.count_nonzero(valid)),
            'requested_points': int(np.count_nonzero(selected))}


def nominal_gate(values):
    values=np.asarray(values)
    return np.isfinite(values)&(values<=.1)


def linear_sphere_samples(r,n,v,sites):
    r,sites=ordered_grid(r),ordered_grid(sites)
    if sites[0]<=0 or sites[0]<r[0] or sites[-1]>r[-1]:
        raise ValueError('site sampling outside stored metric domain')
    right=np.searchsorted(r,sites,side='left');left=np.maximum(0,right-1)
    equal=r[right]==sites;left[equal]=right[equal]
    fraction=np.zeros_like(sites)
    distinct=left!=right;fraction[distinct]=(sites[distinct]-r[left[distinct]])/(r[right[distinct]]-r[left[distinct]])
    nn=(1-fraction)*n[:,left]+fraction*n[:,right]
    vv=(1-fraction)*v[:,left]+fraction*v[:,right]
    support=(np.isfinite(n[:,left])&np.isfinite(n[:,right])&(n[:,left]>0)&(n[:,right]>0)
             &np.isfinite(v[:,left])&np.isfinite(v[:,right]))
    valid=support&np.isfinite(nn)&np.isfinite(vv)&(nn>0)
    c=np.divide(vv,sites*nn,out=np.full_like(nn,np.nan),where=valid)
    return dict(R=sites,N=nn,v=vv,V=c*c,primitive_support_valid=support,valid=valid,
                left_node_indices=left,right_node_indices=right,right_fraction=fraction)


def full_frame_domain_support(domain,tables):
    first=np.all(np.take(domain,tables['radial_first']['indices'],axis=1),axis=-1)
    second=np.all(np.take(domain,tables['radial_second']['indices'],axis=1),axis=-1)
    time_first=np.all(np.take(first,tables['time_first']['indices'],axis=0),axis=1)
    return second&time_first


def tensors_from_components(x, y, z, u, v):
    tensor = np.zeros((4, 4, 4, 4))
    for a,b,c,d,value in [(0,1,0,1,x),(0,2,0,2,y),(0,3,0,3,y),
                         (0,2,1,2,z),(0,3,1,3,z),
                         (1,2,1,2,u),(1,3,1,3,u),(2,3,2,3,v)]:
        for aa,bb,cc,dd,sign in [(a,b,c,d,1),(b,a,c,d,-1),
                                 (a,b,d,c,-1),(b,a,d,c,1),
                                 (c,d,a,b,1),(d,c,a,b,-1),
                                 (c,d,b,a,-1),(d,c,b,a,1)]:
            tensor[aa,bb,cc,dd] = sign*value
    return tensor


def support_masks(tables, primitive_valid, local_strain_pass, pre_prefix,
                  solver_prefix, time_indices, original_seams, center_index=None,
                  wall_index=None):
    """Declared dependency unions for components and nested derivative operations.

    original_seams entries are False/True/None, and time_indices bind each
    variant row to its original record. Unknown support never passes admission.
    """
    def reduce_samples(array, table, axis):
        moved=np.moveaxis(np.asarray(array,dtype=bool),axis,-1)
        reduced=np.all(np.take(moved,table['indices'],axis=-1),axis=-1)
        return np.moveaxis(reduced,-1,axis)
    def dependencies(array, kind):
        if kind=='direct': return array.copy()
        spatial=reduce_samples(array,tables['radial_first'],1)
        if kind=='space1':return spatial
        if kind=='Y':return spatial & reduce_samples(array,tables['time_first'],0)
        second=reduce_samples(array,tables['radial_second'],1)
        return second & reduce_samples(spatial,tables['time_first'],0)
    nt,nr=primitive_valid.shape
    time_ids=tables['time_first']['indices']
    time_pre=np.all(np.asarray(pre_prefix,dtype=object)[time_ids]==True,axis=1)
    time_solver=np.all(np.asarray(solver_prefix,dtype=object)[time_ids]==True,axis=1)
    pp_raw=np.asarray(pre_prefix,dtype=object);sp_raw=np.asarray(solver_prefix,dtype=object)
    time_pre_unknown=np.any(pp_raw[time_ids]==None,axis=1)&~np.any(pp_raw[time_ids]==False,axis=1)
    time_solver_unknown=np.any(sp_raw[time_ids]==None,axis=1)&~np.any(sp_raw[time_ids]==False,axis=1)
    seams=[];unknown=[];touches_boundary=[]
    known_boundaries=[i for i,value in enumerate(original_seams) if value is True]
    for ids in time_ids:
        original=np.asarray(time_indices)[ids]
        segment=original_seams[int(original.min()):int(original.max())]
        seams.append(any(value is True for value in segment))
        unknown.append(any(value is None for value in segment))
        touches_boundary.append(any(original.min()<=k<=original.max() for k in known_boundaries))
    seam=np.asarray(seams); unresolved=np.asarray(unknown)
    output={}
    for kind in ['direct','space1','Y','frame']:
        if kind in ('direct','space1'):
            pp=np.asarray(pre_prefix,dtype=object)==True
            sp=np.asarray(solver_prefix,dtype=object)==True
            pu=pp_raw==None;su=sp_raw==None
            known_seam=np.zeros(nt,dtype=bool); unknown_seam=known_seam.copy()
        else:
            pp,sp,known_seam,unknown_seam=time_pre,time_solver,seam,unresolved
            pu,su=time_pre_unknown,time_solver_unknown
        touches=np.zeros(nr,dtype=bool);touches_wall=touches.copy()
        for target, index in [(touches,center_index),(touches_wall,wall_index)]:
            if index is None:continue
            if kind=='direct':target[index]=True
            else:
                target[:]=np.any(tables['radial_first']['indices']==index,axis=1)
                if kind=='frame':target[:]|=np.any(tables['radial_second']['indices']==index,axis=1)
        valid=dependencies(primitive_valid,kind)
        gate=dependencies(local_strain_pass,kind)
        prefix_ok=pp[:,None]&sp[:,None]
        no_seam=~(known_seam|unknown_seam)
        output[kind+'_support_valid']=valid
        output[kind+'_support_positive_strain_pass']=gate
        output[kind+'_support_pre_prefix']=np.broadcast_to(pp[:,None],(nt,nr)).copy()
        output[kind+'_support_solver_prefix']=np.broadcast_to(sp[:,None],(nt,nr)).copy()
        output[kind+'_support_pre_prefix_unresolved']=np.broadcast_to(pu[:,None],(nt,nr)).copy()
        output[kind+'_support_solver_prefix_unresolved']=np.broadcast_to(su[:,None],(nt,nr)).copy()
        output[kind+'_support_center_touch']=np.broadcast_to(touches,(nt,nr)).copy()
        output[kind+'_support_wall_touch']=np.broadcast_to(touches_wall,(nt,nr)).copy()
        numerical_endpoints=np.zeros(nr,dtype=bool)
        if kind=='direct':numerical_endpoints[[0,-1]]=True
        else:
            numerical_endpoints=np.any((tables['radial_first']['indices']==0)|(tables['radial_first']['indices']==nr-1),axis=1)
            if kind=='frame':numerical_endpoints|=np.any((tables['radial_second']['indices']==0)|(tables['radial_second']['indices']==nr-1),axis=1)
        output[kind+'_support_numerical_grid_endpoint_touch']=np.broadcast_to(numerical_endpoints,(nt,nr)).copy()
        output[kind+'_support_known_seam']=np.broadcast_to(known_seam[:,None],(nt,nr)).copy()
        output[kind+'_support_unknown_seam']=np.broadcast_to(unknown_seam[:,None],(nt,nr)).copy()
        if kind in ('direct','space1'):
            touch=np.asarray([int(i) in known_boundaries for i in time_indices])
        else:touch=np.asarray(touches_boundary)
        output[kind+'_support_seam_span_touch']=np.broadcast_to(touch[:,None],(nt,nr)).copy()
        output[kind+'_support_seam_touch_only']=np.broadcast_to((touch&~known_seam)[:,None],(nt,nr)).copy()
        output[kind+'_all_support_restricted']=valid&gate&prefix_ok&no_seam[:,None]&~touches[None,:]
    return output


def center_proxy(t, r, n, v, positive_nodes, time_width=3):
    if r[0] != 0 or len(r) <= positive_nodes:
        return None
    ids = np.arange(1, positive_nodes+1)
    n0 = n[:,0]
    valid = (np.all(np.isfinite(n[:,:positive_nodes+1])&(n[:,:positive_nodes+1]>0)
                    &np.isfinite(v[:,:positive_nodes+1]),axis=1)&(v[:,0]==0))
    intercept = weights_at(r[ids]**2, 0., 0)
    slope = weights_at(r[:positive_nodes+1]**2, 0., 1)
    v1 = (v[:,ids]/r[ids])@intercept
    n2 = n[:,:positive_nodes+1]@slope
    safe_n0 = np.where(valid, n0, np.nan)
    k = v1/safe_n0
    time_table=derivative_table(t,1,time_width)
    kt = differentiate(k,time_table,0)
    x = 2*n2/safe_n0-kt/safe_n0-k*k
    return {'N0': n0, 'stored_v0': v[:,0], 'parity_valid': valid,
            'v1': v1, 'N2': n2, 'k': k, 'k_t': kt,
            'positive_node_indices':np.broadcast_to(ids,(len(t),len(ids))).copy(),
            'v1_interpolation_weights':np.broadcast_to(intercept,(len(t),len(ids))).copy(),
            'N2_interpolation_indices':np.broadcast_to(np.arange(positive_nodes+1),(len(t),positive_nodes+1)).copy(),
            'N2_interpolation_weights':np.broadcast_to(slope,(len(t),positive_nodes+1)).copy(),
            'time_stencil_indices':time_table['indices'],'time_stencil_weights':time_table['weights'],
            'full_time_support_valid':np.all(valid[time_table['indices']],axis=1)&np.isfinite(x),
            'X': x, 'Y': x.copy(), 'Z': np.where(np.isfinite(x),0.,np.nan),
            'U': k*k, 'V': k*k,
            'frame_norm': np.sqrt(12*(x*x+k**4)),
            'kretschmann': 12*(x*x+k**4)}


def frame_from_jets(n, v, r, nr, nrr, vr, vrr, bt, ct):
    """Apply the specified frame identities to supplied numerical or analytic jets."""
    n,v,r,nr,nrr,vr,vrr,bt,ct = np.broadcast_arrays(n,v,r,nr,nrr,vr,vrr,bt,ct)
    a,b = nr/n,vr/n
    c = np.divide(v,r*n,out=np.full_like(n,np.nan,dtype=float),where=r>0)
    br = vrr/n-a*b
    cr = np.divide(b-c,r,out=np.full_like(n,np.nan,dtype=float),where=r>0)-a*c
    d0b,d0c = (bt+v*br)/n,(ct+v*cr)/n
    x = nrr/n-d0b-b*b
    y = np.divide(a,r,out=np.full_like(n,np.nan,dtype=float),where=r>0)-d0c-c*c
    z,u,q = a*c,b*c,c*c
    squared = 4*(x*x+2*y*y+2*u*u+q*q+4*z*z)
    invariant = 4*(x*x+2*y*y+2*u*u+q*q-4*z*z)
    return dict(A=a,B=b,C=c,B_r=br,C_r=cr,D0_B=d0b,D0_C=d0c,
                X=x,Y=y,Z=z,U=u,V=q,frame_norm=np.sqrt(squared),kretschmann=invariant)


def calculate(t, r, n, v, sigma, radial_width=3, time_width=3, product_checks=True):
    t, r = ordered_grid(t, time_width), ordered_grid(r, max(radial_width, 4))
    n, v = np.asarray(n, dtype=float), np.asarray(v, dtype=float)
    if n.shape != (len(t),len(r)) or v.shape != n.shape:
        raise ValueError('field/grid shape mismatch')
    domain = np.isfinite(n) & (n > 0) & np.isfinite(v)
    safe_n = np.where(domain, n, np.nan)
    safe_v = np.where(domain, v, np.nan)
    dr = derivative_table(r, 1, radial_width)
    drr = derivative_table(r, 2, radial_width)
    dt = derivative_table(t, 1, time_width)
    nr, nrr = differentiate(safe_n, dr, 1), differentiate(safe_n, drr, 1)
    vr, vrr = differentiate(safe_v, dr, 1), differentiate(safe_v, drr, 1)
    a, b = nr/safe_n, vr/safe_n
    w = safe_v/safe_n
    positive = r > 0
    c = np.full_like(n, np.nan); c[:,positive] = w[:,positive]/r[positive]
    bt, ct = differentiate(b, dt, 0), differentiate(c, dt, 0)
    frame = frame_from_jets(safe_n,safe_v,r,nr,nrr,vr,vrr,bt,ct)
    gk, gh, ga = sigma*np.maximum(np.abs(b),np.abs(c)), sigma*np.abs(vr), sigma*np.abs(a)
    result = dict(T=t,R=r,N=n,v=v,N_r=nr,N_rr=nrr,v_r=vr,v_rr=vrr,w=w,
                  B_t=bt,C_t=ct,**frame,
                  field_domain_valid=domain,positive_radius=positive,
                  gK=gk,gH=gh,gA=ga,
                  gK_pass=nominal_gate(gk),gH_pass=nominal_gate(gh),gA_pass=nominal_gate(ga),
                  radial_centered=dr['centered']&drr['centered'],time_centered=dt['centered'])
    if product_checks:
        formed_br = differentiate(b,dr,1)
        formed_cr = np.full_like(c,np.nan)
        positive_dr=derivative_table(r[positive],1,radial_width)
        formed_cr[:,positive] = differentiate(c[:,positive],positive_dr,1)
        wr = differentiate(w,dr,1)
        nt, vt, vrt = differentiate(safe_n,dt,0), differentiate(safe_v,dt,0), differentiate(vr,dt,0)
        wt = differentiate(w,dt,0)
        alternate_y = np.full_like(n,np.nan)
        alternate_y[:,positive] = a[:,positive]/r[positive]-(wt[:,positive]+safe_v[:,positive]*wr[:,positive])/(r[positive]*safe_n[:,positive])
        result.update(B_r_product_discrepancy=formed_br-frame['B_r'],
                      C_r_product_discrepancy=formed_cr-frame['C_r'],
                      B_t_product_discrepancy=bt-(vrt/safe_n-vr*nt/safe_n**2),
                      Y_product_discrepancy=frame['Y']-alternate_y,N_t=nt,v_t=vt,v_rt=vrt,w_t=wt,w_r=wr)
    tables = {'radial_first':dr,'radial_second':drr,'time_first':dt}
    if product_checks:
        tables['formed_C_positive_radial_first']={**positive_dr,'positive_radius_indices':np.flatnonzero(positive)}
    centers = {str(k): center_proxy(t,r,n,v,k,time_width) for k in (2,3)} if r[0] == 0 else {}
    return result, tables, centers
