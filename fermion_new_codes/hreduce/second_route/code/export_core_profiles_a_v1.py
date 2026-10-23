"""Saved-column presentation only: no state, Gram, FFT or eigensystem work."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

BASE=Path('<internal>/profiles_v1')
OLD=BASE/'memory_continuation'
PHASE=BASE.parent/'profiles_missing_v2'/'relocated_inputs'
ROOT=BASE.parent/'profiles_a_tables_v1'
PINS={OLD/'core_well_profiles_v1.json':'93024674443168e17c357cc7917ab3f789330ceaff19ecbc0d8f021a16927e9c',
      PHASE/'core_well_profiles_v1.json':'859e538c30e0f92622efee0c8009b6c2b749edc372b5a14a2deb0d26280b5246'}
np=None


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x


def extrema(R,value,mask):
    use=mask&np.isfinite(value);ids=np.flatnonzero(use)
    result=dict(mask_rows=int(mask.sum()),finite_rows=len(ids),undefined_rows=int((mask&~np.isfinite(value)).sum()),
        mask_radius_span=[float(R[mask].min()),float(R[mask].max())] if mask.any() else None)
    if not len(ids):return dict(**result,minimum=None,maximum=None)
    lo=ids[np.argmin(value[ids])];hi=ids[np.argmax(value[ids])]
    result.update(minimum=dict(value=value[lo],site=int(lo)+1,R_over_a=R[lo]),
                  maximum=dict(value=value[hi],site=int(hi)+1,R_over_a=R[hi]))
    return result


def main():
    global np
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory required')
    if any(ROOT.glob('core_profiles_A_*')):raise FileExistsError('Existing presentation requires assessment')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[key]='1'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[cpus[9] if len(cpus)>9 else cpus[0]])
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(45,50))
    def deadline(signum,frame):raise TimeoutError('Saved-column presentation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(55)
    import numpy as numpy
    np=numpy
    start=time.monotonic()
    report=dict(status='incomplete',scope='Own saved finite-radial columns and conditional scalar comparisons only',
        scientific_contractions=0,producer_sha256=digest(Path(__file__)),input_products=[],slices={},
        resources=dict(wall_seconds=60,internal_seconds=55,cpu_seconds=50,address_space_bytes=2*2**30,threads=1),
        comparison_scopes=dict(continuum_pi_over_24='Conditional sixteen-state isotropic disjoint-valley ball',
            native_four='Comparison only; no matched cubic reconstruction',
            flat_positive_two='Comparison only; requires homogeneous free positive-band complement',
            radial_1760='Two radial components and all ten declared weights; finite polar qualification'),
        ties='Smallest radius',undefined='undefined in TSV; null in JSON; never zero')
    radial=ROOT/'core_profiles_A_radial_v1.tsv';shell=ROOT/'core_profiles_A_shell_v1.tsv'
    try:
        reports={}
        for path,expected in PINS.items():
            if digest(path)!=expected:raise ValueError('Changed report input')
            reports[path]=json.loads(path.read_text())
        old,phase=reports[OLD/'core_well_profiles_v1.json'],reports[PHASE/'core_well_profiles_v1.json']
        if phase['status']!='completed_phase_A_only':raise ValueError('Unaccepted phase state')
        entries={Path(x['path']).name:x for x in old['saved_arrays']}
        entries.update({Path(x['path']).name:x for x in phase['saved_arrays']})
        radial_count=shell_count=0
        with radial.open('x',newline='') as rf,shell.open('x',newline='') as sf:
            rw=sw=None
            def cell(value):
                if isinstance(value,(bool,np.bool_)):return int(value)
                return format(float(value),'.17g') if math.isfinite(float(value)) else 'undefined'
            for clock in (12,15,16):
                products={}
                for stem in ('raw','matched','mass_premises','momentum'):
                    entry=entries[f'{stem}_T{clock:02d}.npz'];path=Path(entry['path'])
                    if path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
                        raise ValueError('Changed saved product: '+str(path))
                    with np.load(path,allow_pickle=False) as z:products[stem]={k:z[k] for k in z.files}
                    report['input_products'].append(entry)
                raw,matched,mass,momentum=[products[k] for k in ('raw','matched','mass_premises','momentum')]
                R=raw['R'];weights=raw['weights'];profile=phase['slices'][str(clock)]['matched']
                if len(R)!=2048 or not np.array_equal(R,np.arange(1,2049)) or weights.sum()!=880:
                    raise ValueError('Known grid/weight bookkeeping failed')
                volume=4*np.pi*R*R;n=matched['number_density'];q=matched['q']
                ratios=dict(n_over_conditional_pi_24=24*n/np.pi,n_over_native_four_comparator=n/4,
                    n_over_flat_positive_two_comparator=n/2,q_over_radial_capacity_1760=q/1760)
                if not np.array_equal(ratios['n_over_flat_positive_two_comparator'],2*ratios['n_over_native_four_comparator']):
                    raise ValueError('Power-two comparator identity failed')
                identity_error=max(float(np.max(np.abs(ratios['n_over_conditional_pi_24']-(96/np.pi)*ratios['n_over_native_four_comparator']))),
                    float(np.max(np.abs(ratios['q_over_radial_capacity_1760']-(volume/1760)*n))))
                columns=dict(T_over_rs=np.full(2048,clock),R_over_a=R)
                for label in ('dense','applied'):
                    for key in ('N','v','w','M','source','I','I_collar'):
                        columns[label+'_'+key]=raw[label+'_'+key] if label+'_'+key in raw else raw[key+'_'+label]
                mapping=dict(joint_q='q',number_density_a3='number_density',bare_energy_shell='bare_energy',
                    bare_energy_density_a4='bare_energy_density',dust_J_shell='source_J',dust_J_density_a4='source_J_density',
                    enclosed_joint_count='cumulative_count',bare_energy_per_dust_a='bare_energy_per_dust',
                    energy_per_dust_valid='ratio_valid',dense_fit_mask='dense_fit_mask',applied_fit_mask='applied_fit_mask')
                for output,key in mapping.items():columns[output]=matched[key]
                for key in ('eps_tot','eps_pairs','J_tot','J_pairs'):
                    columns['historical_'+key+'_shell']=raw[key]
                    columns['historical_'+key+'_density']=raw[key]/volume
                for label in ('dense','applied'):
                    columns[label+'_M_over_mu_enclosed_count']=mass[label+'_M_over_m_enclosed_count']
                    edge=np.append(matched[label+'_edge_log_slope'],np.nan)
                    columns[label+'_edge_log_slope_to_next_site']=edge
                    columns[label+'_edge_slope_defined']=np.isfinite(edge)
                columns.update(ratios)
                if rw is None:
                    rw=csv.writer(rf,delimiter='\t');rw.writerow(columns)
                for i in range(2048):rw.writerow([cell(x[i]) for x in columns.values()])
                radial_count+=2048
                shell_columns=dict(T_over_rs=np.full(16,clock),shell_index=np.arange(1,17),
                    first_site=128*np.arange(16)+1,last_site=128*np.arange(1,17),
                    shell_number=momentum['shell_number'],shell_high_number=momentum['shell_high_number'],
                    shell_high_share=momentum['shell_high_share'])
                if sw is None:
                    sw=csv.writer(sf,delimiter='\t');sw.writerow(shell_columns)
                for i in range(16):sw.writerow([cell(x[i]) for x in shell_columns.values()])
                shell_count+=16
                masks=dict(full=np.ones(2048,dtype=bool),R_le_R50=R<=profile['R50'])
                density={}
                for mask_name,mask in masks.items():
                    density[mask_name]={key:extrema(R,value,mask) for key,value in dict(
                        joint_q=q,number_density_a3=n,bare_energy_density_a4=matched['bare_energy_density'],**ratios).items()}
                premises={}
                for label in ('dense','applied'):
                    ratio=mass[label+'_M_over_m_enclosed_count']
                    for mask_name,mask in dict(full_positive_enclosed_count=matched['cumulative_count']>0,
                            saved_primary_fit=matched[label+'_fit_mask']).items():
                        premises[label+'_'+mask_name]=dict(range=extrema(R,ratio,mask),
                            absolute_departure_from_one=extrema(R,np.abs(ratio-1),mask))
                packing=[]
                for alpha,key in ((.5,'R50'),(.9,'R90')):
                    enclosed=alpha*profile['count'];comparison=float(np.cbrt(18*enclosed/np.pi**2))
                    packing.append(dict(alpha=alpha,saved_R_over_a=profile[key],interpolated_quantile_count=enclosed,
                        conditional_pi_4_radius_over_a=comparison,R_over_conditional_radius=profile[key]/comparison))
                report['slices'][clock]=dict(density_extrema=density,mass_premise_ranges=premises,packing=packing,
                    integer_site_half_window_count=profile['primary_window_count'],capacity_identity_max_discrepancy=identity_error,
                    saved_known_count=profile['count'],saved_momentum_summary=profile['momentum_summary'],
                    saved_polar_channels=phase['slices'][str(clock)]['matched_channels'])
        if radial_count!=6144 or shell_count!=48:raise ValueError('Known output row counts failed')
        report.update(status='completed_saved_column_presentation',radial_rows=radial_count,shell_rows=shell_count,
            outputs=[dict(path=str(p),size=p.stat().st_size,sha256=digest(p)) for p in (radial,shell)])
    except Exception as error:report['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(50,50))
        report['elapsed_seconds']=time.monotonic()-start
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with (ROOT/'core_profiles_A_extrema_v1.json').open('x') as stream:
            json.dump(clean(report),stream,indent=2,allow_nan=False);stream.write('\n')
    if report['status']!='completed_saved_column_presentation':raise SystemExit(1)


if __name__=='__main__':main()
