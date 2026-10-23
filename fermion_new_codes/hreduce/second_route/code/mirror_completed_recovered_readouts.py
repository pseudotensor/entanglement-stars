#!/usr/bin/env python3
"""Copy one existing own readout snapshot; never launch or repeat a producer."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


REMOTE = '<internal>/hreduce_readouts_v1'
REPAIR_REMOTE = '<internal>/hreduce_readouts_repair_v1'
RESULTS = Path(__file__).resolve().parents[2] / 'results'
LOCAL = RESULTS / 'second_route/holds/k3_source/recovered_readouts_v1'
REPAIR_LOCAL = RESULTS / 'second_route/holds/k3_source/recovered_readouts_repair_v1'
LEGACY_STAGES = ('union_setup', 'union_probe_0', 'union_probe_1', 'union_probe_2',
                 'inside_setup', 'inside_probe_0', 'inside_probe_1')
LIMIT = 20_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(path):
    if (not any(path.startswith(root) for root in (REMOTE, REPAIR_REMOTE))
            or '..' in PurePosixPath(path).parts
            or any(char in path for char in (' ', '\n', "'", '"'))):
        raise ValueError('Only named own readout products may be copied')
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=10', 'compute-host', 'cat', path])


def save_changed(path, data):
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('c15', 'c30'), required=True)
    parser.add_argument('--stage', choices=('prepare', 'diagnostics'), required=True)
    parser.add_argument('--repair', action='store_true',
        help='Copy only this lane\'s explicit repair run; do not duplicate imported legacy products')
    args = parser.parse_args()
    if args.repair and args.case == 'c15' and args.stage == 'prepare':
        raise ValueError('C15 preparation already has its original branch copy')
    product_root = REPAIR_REMOTE if args.repair and args.stage == 'diagnostics' else REMOTE
    run_root = REPAIR_REMOTE if args.repair else REMOTE
    local_root = REPAIR_LOCAL if args.repair else LOCAL
    prefix = product_root + args.case + '/' + args.stage + '/'
    destination = local_root / args.case / args.stage
    journal_path = prefix + args.stage + '_progress.json'
    journal_bytes = fetch(journal_path)
    journal = json.loads(journal_bytes)
    large = RESULTS / 'LARGE_FILES.tsv'
    lines = large.read_text().splitlines()
    known = {line.split('\t')[0]: line.split('\t') for line in lines[1:]}
    copied, retained, legacy_reused = [], [], []

    def keep_on_box(path, size, sha, reason, mutable=False):
        entry = [path, str(size), sha, 'compute-host:' + path, reason]
        if path in known:
            previous = known[path]
            if not mutable and previous[1:3] != entry[1:3]:
                raise RuntimeError('A completed large product changed: ' + path)
            if mutable:
                lines[lines.index('\t'.join(previous))] = '\t'.join(entry)
        else:
            lines.append('\t'.join(entry))
        known[path] = entry
        retained.append(PurePosixPath(path).name)

    if args.repair and args.case == 'c15' and tuple(journal.get('legacy_stage_names', ())) != LEGACY_STAGES:
        raise ValueError('The repair journal lacks its exact seven imported stage names')
    for stage_name, stage in journal['stages'].items():
        if stage['status'] != 'completed':
            continue
        for item in stage['outputs'].values():
            path, size, sha = item['path'], item['size'], item['sha256']
            if str(PurePosixPath(path).parent) + '/' != prefix:
                if (args.repair and args.case == 'c15' and args.stage == 'diagnostics'
                        and stage_name in LEGACY_STAGES
                        and str(PurePosixPath(path).parent) + '/' == REMOTE + 'c15/diagnostics/'):
                    # Their old branch copies and LARGE_FILES entries already
                    # exist. Preserve the journal association without duplicates.
                    legacy_reused.append(PurePosixPath(path).name)
                    continue
                raise ValueError('A stage output is outside its own case directory')
            name = PurePosixPath(path).name
            target = destination / name
            raw_probe = (args.stage == 'diagnostics' and name.endswith('.npz')
                and name.startswith(('union_probe_', 'inside_probe_')))
            raw_preparation = (args.stage == 'prepare' and name.endswith('.npz')
                and (name.startswith('window') or (name.startswith('cut')
                    and name.endswith(('_chi.npz', '_energy.npz', '_native_current.npz')))))
            if size > LIMIT or raw_probe or raw_preparation:
                reason = ('raw probe projection matrices; box only even below 20 MB; JSON checks mirrored'
                    if raw_probe else
                    'raw preparation vectors/matrices; box only even below 20 MB; JSON controls mirrored'
                    if raw_preparation else
                    'required recovered readout raw array; box only; small controls mirrored')
                keep_on_box(path, size, sha, reason)
                if target.exists():
                    if target.stat().st_size != size or digest(target.read_bytes()) != sha:
                        raise RuntimeError('Local raw product differs; refusing removal: ' + str(target))
                    target.unlink()
                continue
            if (target.exists() and target.stat().st_size == size
                    and digest(target.read_bytes()) == sha):
                continue
            data = fetch(path)
            if len(data) != size or digest(data) != sha:
                raise RuntimeError('Copied small product differs: ' + path)
            save_changed(target, data)
            copied.append(target.name)
    if len(journal_bytes) > LIMIT:
        keep_on_box(journal_path, len(journal_bytes), digest(journal_bytes),
            'oversize readout journal; exact current snapshot stays on box', mutable=True)
    else:
        save_changed(destination / PurePosixPath(journal_path).name, journal_bytes)
    record_path = run_root + 'execution/' + args.case + '_' + args.stage + '_exit.json'
    record_bytes = fetch(record_path)
    if len(record_bytes) <= 1_000_000:
        save_changed(local_root / 'execution' / PurePosixPath(record_path).name, record_bytes)
    else:
        keep_on_box(record_path, len(record_bytes), digest(record_bytes),
            'execution output exceeds short-final-record limit', mutable=True)
    save_changed(large, ('\n'.join(lines) + '\n').encode())
    print(json.dumps({'case': args.case, 'stage': args.stage, 'explicit_repair': args.repair,
        'completed': journal['completed'],
        'stage_statuses': {k: v['status'] for k, v in journal['stages'].items()},
        'small_copied_verified': copied, 'large_box_only': retained,
        'legacy_products_not_duplicated': legacy_reused,
        'journal_bytes': len(journal_bytes)}))


if __name__ == '__main__':
    main()
