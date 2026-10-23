"""Environment-only continuation of the approved, unfinished fixed-mass check.

Serves insert I6a and item 7's light-cut ratio. No scientific producer edit.
Run only inside the external 299-second GNU timeout on the named box CPU.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


START = time.monotonic()
BASE = Path('<internal>/ETA_massive_environment_continuation_v1')
RUN = BASE / 'run'
INPUT = BASE / 'input'
PRODUCER = BASE / 'code/fixed_mass_interval_sharp_repair.py'
DEPENDENCY = Path('<internal>/python_flint_0_9_0')
DEPENDENCY_SHA = 'ab973b61f29d812c0a6af918f052edfd7ae5bdb39a9e77023a70bee5b4e1088e'
EXPECTED = {
    'producer': 'd962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870',
    'stable_result.json': '71c5df353b17ecde07d69872c611ce8c998fa419243d5151f71ca1713213bf52',
    'fixed_mass_sharp_repair_result.json': '53ad011b7e46d3c6128c0cd130e109696fcaf22b383ec80ec1373a2f187f5ed4',
    'fixed_mass_sharp_repair_exit.json': '613fcaff816e4a9dc8e9372eec8ac0255053427175766972acc5afdad11687e3',
}
RECEIPT = RUN / 'execution.json'


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save(record):
    tmp = RECEIPT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(RECEIPT)


def dependency_identity():
    files = sorted(p for p in DEPENDENCY.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(DEPENDENCY).as_posix().encode() + b'\0'
                      + identity(path)['sha256'].encode() + b'\n')
    metadata = (DEPENDENCY / 'python_flint-0.9.0.dist-info/METADATA').read_text()
    if '\nName: python-flint\n' not in '\n' + metadata or '\nVersion: 0.9.0\n' not in '\n' + metadata:
        raise ValueError('The existing dependency is not pinned python-flint 0.9.0')
    if digest.hexdigest() != DEPENDENCY_SHA:
        raise ValueError('The pinned dependency tree changed')
    return dict(path=str(DEPENDENCY), version='0.9.0', file_count=len(files),
                bytes=sum(p.stat().st_size for p in files), sha256=digest.hexdigest())


def main():
    if Path(__file__).resolve().parent != BASE / 'code':
        raise RuntimeError('Wrong continuation launcher location')
    parent = Path('/proc', str(os.getppid()), 'cmdline').read_bytes().split(b'\0')
    if not (Path(os.fsdecode(parent[0])).name == 'timeout'
            and b'--signal=KILL' in parent and b'299s' in parent):
        raise RuntimeError('The external 299-second SIGKILL timeout is required')
    if RUN.exists():
        raise FileExistsError('A prior continuation exists; never repeat it')
    paths = {'producer': PRODUCER, **{name: INPUT / name for name in EXPECTED if name != 'producer'}}
    inputs = {name: identity(path) for name, path in paths.items()}
    if any(inputs[name]['sha256'] != value for name, value in EXPECTED.items()):
        raise ValueError('Approved producer, axial input or startup-failure record changed')
    failed = json.loads(paths['fixed_mass_sharp_repair_result.json'].read_text())
    prior = json.loads(paths['fixed_mass_sharp_repair_exit.json'].read_text())
    if (failed['error'] != "ModuleNotFoundError: No module named 'flint'"
            or any(failed['counts'][k] != 0 for k in ('evaluations', 'created_cells', 'refused_images'))
            or failed['results'] or failed['attempts111'] or failed['accepted_subcharts111']
            or prior['exit_code'] != 1 or prior['producer_sha256'] != EXPECTED['producer']
            or prior['axial_input_sha256'] != EXPECTED['stable_result.json']):
        raise ValueError('The bound prior record is not the zero-science startup failure')
    debit = prior['elapsed_seconds']
    if not (0 <= debit < 1 and 299 + debit < 300):
        raise RuntimeError('The original aggregate time budget is unavailable')
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100 * 2**30:
        raise RuntimeError('The memory/root-disk reserve is unavailable')
    if shutil.disk_usage('/data').free < 2**30:
        raise RuntimeError('Insufficient result storage')
    dependency = dependency_identity()
    cpu = min(os.sched_getaffinity(0))
    settings = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(DEPENDENCY))
    environment = os.environ.copy()
    environment.update(settings)
    RUN.mkdir()
    alias = RUN / PRODUCER.name
    alias.symlink_to(PRODUCER)
    (RUN / 'stable_result.json').symlink_to(INPUT / 'stable_result.json')
    # Reserve 1.5 seconds within the external cap for the final receipt.
    child_seconds = 297.5 - (time.monotonic() - START)
    if child_seconds <= 295:
        raise RuntimeError('Insufficient unchanged internal-save window remains')
    command = ['timeout', '--signal=KILL', f'{child_seconds:.6f}s', 'nice', '-n', '10',
               'taskset', '-c', str(cpu), '<internal>/python', str(alias)]
    record = dict(status='running', purpose='insert I6a; item7 fixed-mass light-cut ratio',
        inputs=inputs, launcher=identity(Path(__file__)), dependency=dependency,
        command=command, wrapper_pid=os.getpid(), started_unix=time.time(), selected_cpu=cpu,
        prior_launch_debit_seconds=debit, aggregate_ceiling_seconds=300,
        external_hard_seconds=299, producer_specification_seconds=300,
        child_hard_seconds=child_seconds, internal_save_seconds=295,
        address_space_bytes=805306368, cpu_threads=1, nice=10,
        environment_settings=settings, available_memory_bytes=available,
        root_free_bytes=shutil.disk_usage('/').free)
    save(record)
    with (RUN / 'stdout.log').open('xb') as out, (RUN / 'stderr.log').open('xb') as err:
        child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err)
        record['child_pid'] = child.pid
        save(record)
        record['exit_code'] = child.wait()
    record.update(status='finished', elapsed_seconds=time.monotonic() - START,
                  maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    record['aggregate_elapsed_seconds'] = debit + record['elapsed_seconds']
    for name in ('stdout.log', 'stderr.log', 'fixed_mass_sharp_repair_result.json'):
        path = RUN / name
        if path.exists():
            record[name] = identity(path)
    save(record)
    print(json.dumps({key: record[key] for key in ('status', 'exit_code', 'elapsed_seconds',
          'aggregate_elapsed_seconds', 'maximum_child_rss_kib')}))


if __name__ == '__main__':
    main()
