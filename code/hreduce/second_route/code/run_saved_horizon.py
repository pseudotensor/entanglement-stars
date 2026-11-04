#!/usr/bin/env python3
"""Execute one inherited HORIZON stage, preserving its inputs and attempt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time


BASE = Path('<internal>/second_route_bulk')
CODE = BASE / 'code'
OUTPUT = BASE / 'horizon' / 'hreduce_v1'
FIXTURE = BASE / 'checks' / 'held_readouts' / 'fixture_arrays.npz'
PINS = {
    'hold_spectral_checks.py': 'cac12901c4a0f6815f0b22d2dbadc578ed9f1ad747692a1f871afe85e0155fbe',
    'hold_exact.py': '4b58467b3a76f14894885220b8cb247c964be0a086478958e856308a42c60598',
    'horizon_fixtures.py': '80860368aecaf6c53d66ddcd04680ec27a22d9f65a8142142b1468397d832475',
    'horizon_exact.py': '5411e253cb6bad08a27eee2d5afa97a347bffdf71e42766e91554bc82db95f38',
    'horizon_bins.py': '030ab15b9c3ff30a0193557c9f8932c06b4f7eecab0e7ba578d61231f364a711',
    'hold_error_bounds.py': '12bacc16d463d8906da8010acffc59497d6dfab810b2804bd23ef37e45ba58c8',
    'hold_endpoint_control.py': '25d43bd4bb0342fb02d61b5579073683177079351efdf263f30f21def08bc421',
    'hold_precision_checks.py': '1f6f58a5a87958ba62f9de8a8797ab282864c71d24700eab5dcfc032c6d5b7da',
    'hreduce_horizon_json.py': 'b14df91339323133a59e455974761ad0c8c3a199d82cbaadf3efd93329d45b8b',
    'hreduce_horizon_stage.py': '503cafe6b4ac49d05298327272799e1742459ccb7517ffa485697749a07fd484',
    'recover_saved_horizon_fixture.py': '2147e5bd543d47fbf3f35da2116e018c55cacd2d0293340262c339665835b960',
}


def persist(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (64 * 2**30, 64 * 2**30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fixture', 'fixture-recovery', 'spectrum', 'checks'), required=True)
    args = parser.parse_args()
    if shutil.disk_usage('/').free < 100 * 10**9:
        raise RuntimeError('Root-disk reserve below 100 GB')
    if shutil.disk_usage(BASE).free < 20 * 2**30:
        raise RuntimeError('Insufficient bulk space')
    available = next(int(line.split()[1]) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    if available - 64 * 2**30 < 64 * 2**30:
        raise RuntimeError('Dense step would violate the memory reserve')
    for name, pin in PINS.items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != pin:
            raise RuntimeError('An inherited reviewed module changed: ' + name)
    fixture_out, exterior = OUTPUT / 'fixture', OUTPUT / 'exterior'
    if args.stage == 'fixture':
        if fixture_out.exists():
            raise FileExistsError('Never repeat an existing fixture stage')
        if not FIXTURE.is_file():
            raise FileNotFoundError('The existing fixture eigensystem is required')
        program = [str(CODE / 'horizon_fixtures.py'), '--fixture', str(FIXTURE),
                   '--out', str(fixture_out)]
        wall = '300s'
    elif args.stage == 'fixture-recovery':
        if not (fixture_out / 'horizon_fixture_arrays.npz').is_file():
            raise FileNotFoundError('Recovery requires the surviving original arrays')
        if (fixture_out / 'horizon_fixture_checks.json').exists():
            raise FileExistsError('The fixture report already exists')
        program = [str(CODE / 'recover_saved_horizon_fixture.py')]
        wall = '300s'
    else:
        accepted_fixture = fixture_out / 'horizon_fixture_checks.json'
        if not accepted_fixture.is_file() or not json.loads(accepted_fixture.read_text())['passed']:
            raise RuntimeError('The required fixture is unavailable or failed')
        if args.stage == 'spectrum':
            if exterior.exists():
                raise FileExistsError('Existing exterior output requires assessment, never another solve')
        elif not (exterior / 'spectrum.npz').is_file() or (exterior / 'bin_checks.json').exists():
            raise RuntimeError('Checks require the saved spectrum and no completed checks')
        program = [str(CODE / 'hreduce_horizon_stage.py'), args.stage, '--out', str(exterior)]
        wall = '4h'
    records = OUTPUT / 'execution'
    records.mkdir(parents=True, exist_ok=True)
    result_path = records / (args.stage + '_exit.json')
    if result_path.exists() or result_path.with_suffix('.tmp').exists():
        raise FileExistsError('An existing attempt cannot silently be repeated')
    env = os.environ.copy()
    env.update({key: '16' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    command = ['timeout', '-s', 'TERM', '-k', '5s', wall, 'nice', '-n', '10',
               '<internal>/python', *program]
    result = dict(stage=args.stage, status='running', command=command,
        cpu_only=True, maximum_numerical_threads=16, memory_limit_bytes=64 * 2**30,
        inherited_code_pins=PINS, started_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    persist(result_path, result)
    before = time.monotonic()
    child = subprocess.run(command, env=env, capture_output=True,
                           preexec_fn=limits, check=False)
    result.update(status='completed' if child.returncode == 0 else 'failed',
        returncode=child.returncode, elapsed_seconds=time.monotonic() - before,
        maximum_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        stdout=child.stdout.decode(errors='replace'), stderr=child.stderr.decode(errors='replace'))
    persist(result_path, result)
    print(json.dumps({key: result[key] for key in ('stage', 'status', 'returncode',
        'elapsed_seconds', 'maximum_child_rss_kib')}))
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
