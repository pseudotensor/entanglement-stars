"""Core Phase B: thirteen missing spectra, retained products reused verbatim."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

BASE = Path('<internal>/profiles_v1')
OLD = BASE/'memory_continuation'
PHASE_A = BASE.parent/'profiles_missing_v2'/'relocated_inputs'
ROOT = BASE.parent/'spectra_missing_v2'
PINS = {
    BASE/'core_well_profiles_v1.py': '8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54',
    OLD/'core_well_profiles_v1.json': '93024674443168e17c357cc7917ab3f789330ceaff19ecbc0d8f021a16927e9c',
    PHASE_A/'core_well_profiles_v1.json': '859e538c30e0f92622efee0c8009b6c2b749edc372b5a14a2deb0d26280b5246',
    PHASE_A/'execution.json': 'ee44fae3ad74649e4879011d214322a84409808ef446d8ff0a352dc435d9d4d8',
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in named box directory')
    if any(ROOT.glob('*.npz*')) or any(ROOT.glob('core_well_profiles_v1.json*')):
        raise FileExistsError('Existing Phase B requires assessment, never retry')
    for path,expected in PINS.items():
        if digest(path)!=expected:
            raise ValueError('Changed predecessor: '+str(path))
    prior=json.loads((OLD/'core_well_profiles_v1.json').read_text())
    phase_a=json.loads((PHASE_A/'core_well_profiles_v1.json').read_text())
    execution=json.loads((PHASE_A/'execution.json').read_text())
    if (phase_a['status']!='completed_phase_A_only' or execution['exit_code']!=0
            or phase_a['remaining_joint_pairs'] or phase_a['remaining_summary_clocks']):
        raise ValueError('Profile summaries must be complete before spectra')
    pending=[[t,k] for t in (16,12,15) for k in range(1,6)
             if (t,k) not in ((16,1),(16,2))]
    if phase_a['remaining_spectral_pairs']!=pending:
        raise ValueError('Unexpected missing spectra')
    entries={Path(x['path']).name:x for x in prior['saved_arrays']}
    entries.update({Path(x['path']).name:x for x in phase_a['saved_arrays']})
    inputs=[f'raw_T{t:02d}.npz' for t in (16,12,15)]
    inputs += [f'joint_T{t:02d}_k{k:02d}.npz' for t in (16,12,15) for k in range(1,6)]
    inputs += [f'spectral_T16_k{k:02d}.npz' for k in (1,2)]
    for name in inputs:
        item=entries[name];path=Path(item['path'])
        if path.stat().st_size!=item['size'] or digest(path)!=item['sha256']:
            raise ValueError('Changed saved product: '+name)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:')))*1024
    if (available<64*2**30 or shutil.disk_usage('/').free<100_000_000_000
            or shutil.disk_usage('/data').free<8*2**30):
        raise RuntimeError('Required resource reserve unavailable')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[key]='4'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    cpus=sorted(os.sched_getaffinity(0))[:4]
    if len(cpus)!=4:
        raise RuntimeError('Four CPUs unavailable')
    os.sched_setaffinity(0,cpus)
    if os.getpriority(os.PRIO_PROCESS,0)<10:
        os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(28800,28800))
    def deadline(signum,frame):
        raise TimeoutError('Phase B checkpoint deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(7180)
    import numpy as np
    import scipy
    from scipy.linalg import eig_banded
    spec=importlib.util.spec_from_file_location('own_core_spectral_base',BASE/'core_well_profiles_v1.py')
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    engine.ROOT,engine.START,engine.OWNED=ROOT,time.monotonic(),True
    engine.np,engine.eig_banded=np,eig_banded
    engine.REPORT=copy.deepcopy(phase_a)
    report=engine.REPORT
    report['predecessor_phase_A']=dict(path=str(PHASE_A/'core_well_profiles_v1.json'),
        execution=execution,phases=report.pop('phases'),resources=report.pop('resources'))
    report.update(status='incomplete_phase_B',purpose='Task3 thirteen missing spectra and Killing sums',
        phases=[],saved_arrays=[],box_only_arrays=[],new_spectral_actions=[],
        remaining_killing_sum_clocks=[16,12,15])
    report['reused_spectral_products']=[entries[f'spectral_T16_k{k:02d}.npz'] for k in (1,2)]
    report['resources']=dict(wall_seconds=7200,internal_deadline_seconds=7180,
        cpu_seconds=28800,address_space_bytes=4*2**30,threads=4)
    report['implementation']=dict(python=sys.version.split()[0],numpy=np.__version__,
        scipy=scipy.__version__,producer_sha256=digest(Path(__file__)),
        cpu_affinity=cpus,nice=os.getpriority(os.PRIO_PROCESS,0))
    report['slices']={int(k):v for k,v in report['slices'].items()}
    for row in report['slices'].values():
        for key in ('channels','matched_channels'):
            row[key]={int(k):v for k,v in row[key].items()}
    try:
        engine.phase('Phase B starting; all saved products verified')
        for clock in (16,12,15):
            with np.load(Path(entries[f'raw_T{clock:02d}.npz']['path']),allow_pickle=False) as z:
                data={k:z[k] for k in z.files}
            total_count=np.zeros(6);total_energy=np.zeros(6)
            total_sites=np.zeros((6,2048));total_site_energy=np.zeros_like(total_sites)
            for k in range(1,6):
                if (clock,k) in ((16,1),(16,2)):
                    entry=entries[f'spectral_T16_k{k:02d}.npz']
                    info=prior['slices']['16']['matched_channels'][str(k)]['spectral']
                    engine.phase(f'T16 k{k} existing spectrum reused verbatim')
                else:
                    with np.load(Path(entries[f'joint_T{clock:02d}_k{k:02d}.npz']['path']),allow_pickle=False) as z:
                        Z,f=z['dust'],z['f']
                    info=engine.spectral(clock,k,Z,f,data)
                    del Z,f
                    entry=report['saved_arrays'][-1]
                    report['slices'][clock]['matched_channels'][k]['spectral']=info
                    report['remaining_spectral_pairs'].remove([clock,k])
                    report['new_spectral_actions'].append([clock,k])
                    engine.save()
                weight=data['weights'][k-1]
                total_count+=weight*np.asarray(info['count_bins'])
                total_energy+=weight*np.asarray(info['energy_bins'])
                with np.load(Path(entry['path']),allow_pickle=False) as z:
                    total_sites+=weight*z['site_bins']
                    total_site_energy+=weight*z['energy_site_bins']
            engine.arrays(f'killing_sum_T{clock:02d}',count_bins=total_count,energy_bins=total_energy,
                site_bins=total_sites,energy_site_bins=total_site_energy,
                edges=np.array([-np.sqrt(1.04),-.2,0,.2,np.sqrt(1.04)]))
            report['slices'][clock]['killing_sum']=dict(count_bins=total_count,energy_bins=total_energy,
                signed_local_allocation=True,edge_membership_certified=False)
            report['remaining_killing_sum_clocks'].remove(clock)
            engine.phase(f'T{clock} weighted Killing sums saved')
            del data
        if len(report['new_spectral_actions'])!=13:
            raise ValueError('Unexpected missing action count')
        report['status']='completed_phase_B_only'
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);engine.save()
    if report['status']!='completed_phase_B_only':
        raise SystemExit(1)


if __name__=='__main__':
    main()
