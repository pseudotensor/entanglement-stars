"""Restore only the pinned dependency path for the reviewed fixed-mass job."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

PARENT = Path('<internal>/eta_massive_scalar_v1')
ROOT = PARENT / 'environment_continuation'
DEPENDENCY = '<internal>/python_flint_0_9_0'
PRODUCER_NAME = 'fixed_mass_interval_sharp_repair.py'
RESULT_NAME = 'fixed_mass_sharp_repair_result.json'
PINS = {
    PRODUCER_NAME: 'd962ea9b594c2495284069eb2814ba6d6263786c374bd1b51475d15b29f3e870',
    'stable_result.json': '71c5df353b17ecde07d69872c611ce8c998fa419243d5151f71ca1713213bf52',
    RESULT_NAME: '53ad011b7e46d3c6128c0cd130e109696fcaf22b383ec80ec1373a2f187f5ed4',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the box continuation directory')
    for name, digest in PINS.items():
        if sha(PARENT / name) != digest:
            raise ValueError('Changed original input: ' + name)
    prior = json.loads((PARENT / RESULT_NAME).read_text())
    previous_exit_path = PARENT / 'fixed_mass_sharp_repair_exit.json'
    previous_exit = json.loads(previous_exit_path.read_text())
    if (prior['error'] != "ModuleNotFoundError: No module named 'flint'"
            or prior['counts']['evaluations'] != 0
            or prior['counts']['created_cells'] != 0
            or prior['results'] or prior['accepted_subcharts111']
            or previous_exit['status'] != 'finished'
            or previous_exit['exit_code'] != 1
            or previous_exit['elapsed_seconds'] != 0.06138305366039276):
        raise ValueError('The original failure is not the reviewed startup failure')
    if not (Path(DEPENDENCY) / 'flint' / '__init__.py').is_file():
        raise FileNotFoundError('Existing pinned dependency directory is missing')
    for name in (RESULT_NAME, 'execution.json', 'stdout.log', 'stderr.log'):
        if (ROOT / name).exists() or (ROOT / name).with_suffix('.json.tmp').exists():
            raise FileExistsError('Existing continuation requires assessment, never retry')
    for name in (PRODUCER_NAME, 'stable_result.json'):
        destination = ROOT / name
        if destination.is_symlink():
            if destination.resolve() != PARENT / name:
                raise ValueError('Unexpected input symlink')
        elif destination.exists():
            raise FileExistsError('Input path is not the prescribed symlink')
        else:
            destination.symlink_to(PARENT / name)
    available = int(next(line.split()[1] for line in
        Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))) * 1024
    if available < 64 * 2**30 or shutil.disk_usage('/').free < 100_000_000_000:
        raise RuntimeError('Memory or root reserve unavailable')
    if shutil.disk_usage('/data').free < 2**30:
        raise RuntimeError('Data reserve unavailable')
    cpu = min(os.sched_getaffinity(0))
    env = os.environ.copy()
    settings = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'BLIS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=DEPENDENCY)
    env.update(settings)
    command = ['timeout', '--foreground', '-s', 'KILL', '299s', 'nice', '-n', '10',
        'taskset', '-c', str(cpu), '<internal>/python',
        str(ROOT / PRODUCER_NAME)]
    record = dict(status='running', purpose='I6a fixed-mass unchanged-code environment continuation',
        prior_exit_path=str(previous_exit_path), prior_exit_sha256=sha(previous_exit_path),
        prior_elapsed_seconds=previous_exit['elapsed_seconds'], pins=PINS,
        wrapper_sha256=sha(Path(__file__)), wrapper_pid=os.getpid(), selected_cpu=cpu,
        command=command, environment_settings=settings,
        wall_seconds=299, original_aggregate_wall_seconds=300,
        producer_internal_save_seconds=295, producer_address_space_bytes=805306368)
    target = ROOT / 'execution.json'
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
    def save():
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(target)
    started = time.monotonic()
    try:
        with (ROOT / 'stdout.log').open('xb') as out, (ROOT / 'stderr.log').open('xb') as err:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, cwd=ROOT)
            record['child_pid'] = child.pid
            save()
            record['exit_code'] = child.wait()
        record['status'] = 'finished'
    except Exception as error:
        record.update(status='failed_or_uncertain', error=type(error).__name__ + ': ' + str(error))
    record['elapsed_seconds'] = time.monotonic() - started
    record['aggregate_elapsed_seconds'] = record['prior_elapsed_seconds'] + record['elapsed_seconds']
    record['maximum_child_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    for name in (RESULT_NAME, 'stdout.log', 'stderr.log'):
        path = ROOT / name
        if path.exists():
            record[name] = dict(size=path.stat().st_size, sha256=sha(path))
    record['original_inputs_unchanged'] = all(sha(PARENT / name) == digest for name, digest in PINS.items())
    save()
    if record.get('exit_code') != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
