import json, os, resource, time
from pathlib import Path
from fractions import Fraction
import numpy as np
started=time.perf_counter()
root=Path('<internal>/results')
p=root/'x_nonlinear_gaussian_continued.json'
r=json.loads(p.read_text())
lines=[]
def emit(label,value):
    lines.append(label+'\n'+json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
lines.append('X existing-data assessment. No source/field evaluation, rejected-state reconstruction, eigensolve, evolution, or model rerun. Ordinary binary64 point diagnostics; no enclosure. Originals remain unchanged.')
lines.append('Execution distinction: original PID 2878166 predates per-corrector/checkpoint reporting and retained no NPZ. Continued PID 2992391 executes the repaired reporting path and fresh deterministic flat reconstruction, charging all prior work and repair downtime. The eight local Parent.assemble lines 284-291 computing r1/transpose identity/beta were added after that launch and were NOT executed by either model run. The retained-array metrics below are this separate data-only readout. Rejected trial matrices and earlier overwritten accepted full-state arrays are unavailable; no metric is retroactively credited to them.')
emit('Final status/failure/resources/caps',{'status':r['status'],'failure':r.get('failure'),'resources':r['resources'],'caps':r['caps'],'cap_overshoot_seconds':r['resources']['elapsed_seconds']-r['caps']['seconds'],'runtime':r['runtime']})
emit('False downstream flags',{k:v for k,v in r.items() if isinstance(v,bool)})
ops={v['requested_field_evaluation']:v for v in r['field_work_records']}
for b in r['boxes']:
    L=b['L']; prep=b.get('preparation',{}); attempts=prep.get('corrector_attempts',[])
    accepted=[v for v in attempts if v['status']=='accepted']
    emit('L='+str(L)+' flat controls and phase resources',{'flat_controls':b.get('flat_controls'),'feasibility':b.get('feasibility')})
    emit('L='+str(L)+' complete preparation attempts and dispositions',{'status':b['status'],'preparation_status':prep.get('status'),'attempts':attempts,'reached_loads':prep.get('reached_loads',[]),'last_valid_load_exact':prep.get('last_valid_load_exact'),'stop':prep.get('stop'),'box_failure':b.get('failure')})
    for a in accepted:
        emit('L='+str(L)+' accepted label '+a['candidate_load']+' stored field record',{'attempt':a,'field_record':ops.get(a['field_counter_after']),'full_state_currently_retained':a['candidate_load']==prep.get('last_valid_load_exact'),'earlier_full_psi_J_and_charge_unavailable_if_false':a['candidate_load']!=prep.get('last_valid_load_exact')})
    if L==5 and attempts:
        st=Fraction(attempts[-1]['load_step'])
        emit('L=5 terminal step-floor disposition',{'attempted_increment_exact':str(st),'attempted_increment_equals_floor':st==Fraction(1,65536),'resulting_halved_increment_exact':str(st/2),'resulting_halved_increment_below_floor':st/2<Fraction(1,65536),'evolution_at_intermediate_load':False})
    state_path=Path(prep['last_valid_native_state_file'])
    last_id=accepted[-1]['field_counter_after'] if accepted else next(v['requested_field_evaluation'] for v in r['field_work_records'] if v['phase']=='L='+str(L)+' flat algebraic control')
    rec=ops[last_id]; sigma=rec['sigma_min']; sigma0=b['flat_controls']['sigma_min_J0']
    with np.load(state_path,allow_pickle=False) as d:
        I=d['interior']; B=d['collar']; psi=d['psi']; N=d['N']; C=d['clock']; J=d['J']; coords=d['coordinates']
        excit=np.stack([d['sea_clock_'+str(i)] for i in range(1,5)])
        collar=float(C[B].sum()); functional=float(N@C); positive=float(np.sum(excit/psi[None,:])); interior=float(np.sum(C[I]/psi[I])); lapse_int=float(N[I]@C[I])
        r1=J.sum(axis=0); identity=J[I,:].T@(N[I]-1)+r1; beta=float(np.linalg.norm(r1))/sigma
        report={'state_file':str(state_path),'state_load_exact':prep['last_valid_load_exact'],'basis_sites':len(coords),'interior_sites':len(I),'J_shape':list(J.shape),'matched_field_evaluation':last_id,'stored_field_sigma_min':sigma,'stored_flat_sigma_min':sigma0,'sigma_ratio':sigma/sigma0,'clock_residual_inf':float(np.max(np.abs(C[I]))),'ruler_residual_inf':float(np.max(np.abs(J.T@N))),'interior_psi_min':float(psi[I].min()),'interior_psi_max':float(psi[I].max()),'interior_N_min':float(N[I].min()),'interior_N_max':float(N[I].max()),'interior_N_min_site':int(I[np.argmin(N[I])]),'interior_N_max_site':int(I[np.argmax(N[I])]),'lower_lapse_guard_sites':I[N[I]<=.125].tolist(),'upper_lapse_guard_sites':I[N[I]>=8].tolist(),'marker_trace_real':float(np.trace(d['marker']).real),'collar_charge':collar,'full_functional':functional,'positive_source_readout':positive,'collar_minus_positive':collar-positive,'minus_interior_clock_over_psi':-interior,'positive_identity_error':collar-positive+interior,'full_minus_collar':functional-collar,'interior_N_clock':lapse_int,'functional_identity_error':functional-collar-lapse_int,'existing_J_r1_norm_2':float(np.linalg.norm(r1)),'existing_J_transpose_identity_residual_2':float(np.linalg.norm(identity)),'existing_J_beta_numerical':beta,'beta_margin_to_seven_eighths':.875-beta,'lapse_deviation_inf':float(np.max(np.abs(N[I]-1))),'metric_scope':'separate retained-array ordinary floating-point readout, not a directed inverse/arithmetic certificate','full_covariances_retested_or_evolved':False}
        emit('L='+str(L)+' actual retained-state readout',report)
    if attempts:
        latest=attempts[-1]
        field_rows=[v for v in r['field_work_records'] if latest['field_counter_before']<v['requested_field_evaluation']<=latest['field_counter_after']]
        emit('L='+str(L)+' terminal attempt evaluated trial records; not stationary roots',field_rows)
lines.append('Optional flat-start affine-linearity test NOT evaluated: the required load-specific full initial source/J matrices at multiple loads were not retained. Rejected matrices were not reconstructed. The L=7 NPZ is the actual last-valid flat state, not a successful nonzero-load preparation. No prescribed load was evolved and no time ladder/endpoint/response/collapse gate is credited.')
emit('Reader resources',{'PID':os.getpid(),'nice':os.nice(0),'own_threads':len(os.listdir('/proc/self/task')),'elapsed_seconds':time.perf_counter()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_limit_seconds':60,'virtual_memory_limit_KiB':1048576,'new_field_source_evaluations':0})
text='\n\n'.join(lines)+'\n'
out=root/'x_nonlinear_gaussian_existing_data.txt'
with out.open('x') as f: f.write(text)
print(text)
print('READOUT_FILE '+str(out))
