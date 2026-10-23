"""Copy three saved Killing allocations and partition them into fixed shells."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

SOURCE = Path('<internal>/spectra_missing_v2')
ROOT = SOURCE.parent/'profiles_b_tables_v1'
PINS = {12: '3225d0307585a457d131802b370e4a9bec62b65378a93caadb4c55668efca57a',
        15: '2f1f5e32184276860094d07d14f8c8c3a9ffeb5d502ab57a676836ef3e19a2f7',
        16: '7794caf01bdbc47d5ca3caffd1f6ba348665702ef9b979dc8101c52b15486399'}
SHAPES = dict(count_bins=(6,), energy_bins=(6,), site_bins=(6,2048),
              energy_site_bins=(6,2048), edges=(5,))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    s = path.stat()
    return dict(size=s.st_size, inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns)


def text_cell(x):
    return format(float(x), '.17g')


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Named box output directory required')
    if any(ROOT.glob('core_profiles_B_*')):
        raise FileExistsError('Existing presentation or temporary output requires assessment')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [allowed[9] if len(allowed)>9 else allowed[0]])
    if os.getpriority(os.PRIO_PROCESS,0) < 10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(45,50))
    def deadline(signum,frame):
        raise TimeoutError('Saved Killing table allocation deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.signal(signal.SIGXCPU,deadline)
    signal.alarm(55)
    start = time.monotonic()
    radial = ROOT/'core_profiles_B_radial_v1.tsv'
    shell = ROOT/'core_profiles_B_shell_v1.tsv'
    radial_tmp = radial.with_suffix('.tsv.tmp')
    shell_tmp = shell.with_suffix('.tsv.tmp')
    summary = ROOT/'core_profiles_B_summary_v1.json'
    report = dict(status='incomplete', scope='Three saved weighted Killing products only',
        state_contractions=0, FFTs=0, diagonalizations=0, multiplicity_reapplied=False,
        signed_local_allocation=True, edge_membership_certified=False,
        producer_sha256=digest(Path(__file__)), input_products=[], slices={},
        resources=dict(wall_seconds=60,internal_seconds=55,cpu_seconds=50,
                       soft_cpu_seconds=45,address_space_bytes=2*2**30,threads=1),
        bin_labels=['(-inf,-b]','(-b,-mu]','(-mu,0]','(0,mu]','(mu,b]','(b,inf)'],
        units=dict(clock='T/r_s',radius='R/a',number='count',energy='aE',mode_bin='aE_mode'),
        limitations=dict(T15='Historical state stored as complex64; contractions do not restore missing precision',
            independent_pointwise_q='Accepted producer control inherited, not repeated',
            independent_native_energy='Accepted producer h-action control inherited, not repeated',
            history_error='No new bound',signed_values='Retained without clipping or probability/escape interpretation'))
    try:
        import numpy as np
        def extrema(values):
            lo=int(np.argmin(values)); hi=int(np.argmax(values))
            return dict(minimum=dict(value=float(values[lo]),index=lo+1),
                        maximum=dict(value=float(values[hi]),index=hi+1),
                        negative_entries=int(np.count_nonzero(values<0)))
        def all_bin_extrema(values):
            # Site/shell first, then bin, fixes both tie priorities explicitly.
            flat=values.T.reshape(-1)
            lo=int(np.argmin(flat)); hi=int(np.argmax(flat))
            return dict(minimum=dict(value=float(flat[lo]),index=lo//6+1,bin_index=lo%6),
                        maximum=dict(value=float(flat[hi]),index=hi//6+1,bin_index=hi%6),
                        negative_entries=int(np.count_nonzero(values<0)))
        radial_rows = shell_rows = 0
        observed = {}
        with radial_tmp.open('x',newline='') as rf, shell_tmp.open('x',newline='') as sf:
            rw=csv.writer(rf,delimiter='\t'); sw=csv.writer(sf,delimiter='\t')
            rw.writerow(('T_over_rs','site_index','R_over_a','bin_index',
                         'number_allocation','a_energy_allocation'))
            sw.writerow(('T_over_rs','shell_index','site_first','site_last',
                         'R_first_over_a','R_last_over_a','bin_index',
                         'number_allocation','a_energy_allocation'))
            for clock in (12,15,16):
                path=SOURCE/f'killing_sum_T{clock:02d}.npz'
                before=identity(path); raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=PINS[clock] or identity(path)!=before:
                    raise ValueError('Changed saved Killing product')
                observed[path]=before
                with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
                    if set(archive.files)!=set(SHAPES):
                        raise ValueError('Changed saved schema')
                    arrays={key:archive[key] for key in SHAPES}
                for key,value in arrays.items():
                    if value.shape!=SHAPES[key] or value.dtype!=np.dtype('float64') or not np.isfinite(value).all():
                        raise ValueError('Shape, dtype or finite-value check failed: '+key)
                edges=arrays['edges']; b=math.sqrt(1+0.2**2)
                if not np.array_equal(edges,np.array([-b,-0.2,0,0.2,b],dtype=np.float64)) or not np.all(np.diff(edges)>0):
                    raise ValueError('Changed bin edges or ordering')
                if np.any(arrays['count_bins']<0):
                    raise ValueError('Negative global count in the accepted input')
                report['input_products'].append(dict(path=str(path),sha256=PINS[clock],**before,
                    schema={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in arrays.items()}))
                number=arrays['site_bins']; energy=arrays['energy_site_bins']
                number_shell=np.empty((6,16),dtype=np.float64)
                energy_shell=np.empty((6,16),dtype=np.float64)
                for bin_index in range(6):
                    for s in range(16):
                        sl=slice(128*s,128*(s+1))
                        number_shell[bin_index,s]=math.fsum(float(x) for x in number[bin_index,sl])
                        energy_shell[bin_index,s]=math.fsum(float(x) for x in energy[bin_index,sl])
                for j in range(2048):
                    for i in range(6):
                        rw.writerow((clock,j+1,j+1,i,text_cell(number[i,j]),text_cell(energy[i,j])))
                        radial_rows+=1
                for s in range(16):
                    first,last=128*s+1,128*(s+1)
                    for i in range(6):
                        sw.writerow((clock,s+1,first,last,first,last,i,
                                     text_cell(number_shell[i,s]),text_cell(energy_shell[i,s])))
                        shell_rows+=1
                bins=[]
                for i in range(6):
                    nsites=math.fsum(float(x) for x in number[i]); nshells=math.fsum(float(x) for x in number_shell[i])
                    esites=math.fsum(float(x) for x in energy[i]); eshells=math.fsum(float(x) for x in energy_shell[i])
                    bins.append(dict(bin_index=i,global_number=float(arrays['count_bins'][i]),
                        global_a_energy=float(arrays['energy_bins'][i]),
                        residuals=dict(number_site_minus_global=nsites-float(arrays['count_bins'][i]),
                            number_shell_minus_global=nshells-float(arrays['count_bins'][i]),
                            number_shell_minus_site=nshells-nsites,
                            energy_site_minus_global=esites-float(arrays['energy_bins'][i]),
                            energy_shell_minus_global=eshells-float(arrays['energy_bins'][i]),
                            energy_shell_minus_site=eshells-esites),
                        site_number=extrema(number[i]),site_a_energy=extrema(energy[i]),
                        shell_number=extrema(number_shell[i]),shell_a_energy=extrema(energy_shell[i])))
                ordering=[]
                for s in range(16):
                    ordering.append(dict(shell_index=s+1,
                        number_site_first_minus_bin_first=math.fsum(math.fsum(float(number[i,j]) for i in range(6))
                            for j in range(128*s,128*(s+1)))-math.fsum(float(x) for x in number_shell[:,s]),
                        energy_site_first_minus_bin_first=math.fsum(math.fsum(float(energy[i,j]) for i in range(6))
                            for j in range(128*s,128*(s+1)))-math.fsum(float(x) for x in energy_shell[:,s])))
                report['slices'][clock]=dict(edges=edges.tolist(),global_count_minus_640=math.fsum(float(x) for x in arrays['count_bins'])-640,
                    bins=bins,shell_summation_order_residuals=ordering,
                    all_bins_extrema=dict(site_number=all_bin_extrema(number),site_a_energy=all_bin_extrema(energy),
                        shell_number=all_bin_extrema(number_shell),shell_a_energy=all_bin_extrema(energy_shell)))
        if radial_rows!=36864 or shell_rows!=288:
            raise ValueError('Output coverage mismatch')
        for path,before in observed.items():
            clock=int(path.stem[-2:])
            if identity(path)!=before or digest(path)!=PINS[clock]:
                raise ValueError('Saved input changed before output completion')
        radial_tmp.replace(radial); shell_tmp.replace(shell)
        report.update(status='completed_saved_Killing_presentation',radial_rows=radial_rows,shell_rows=shell_rows,
            outputs=[dict(path=str(path),size=path.stat().st_size,sha256=digest(path)) for path in (radial,shell)])
    except Exception as error:
        report.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.alarm(0); resource.setrlimit(resource.RLIMIT_CPU,(50,50))
        report.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary=summary.with_suffix('.json.tmp')
        with temporary.open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
        temporary.replace(summary)
    if report['status']!='completed_saved_Killing_presentation':
        raise SystemExit(1)


if __name__=='__main__':
    main()
