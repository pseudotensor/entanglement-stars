"""Extract saved non-leaf records; never import or rerun the producer."""
import hashlib
import json
from pathlib import Path


BASE = Path('<internal>/ETA_massive_environment_continuation_v1')
RUN = BASE / 'run'


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def omit_leaf_payload(value, raw_path):
    if isinstance(value, list):
        return [omit_leaf_payload(item, raw_path) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'leaves':
                if not isinstance(item, list):
                    raise TypeError('Unexpected raw-leaf format')
                result['raw_leaves_on_box'] = dict(count=len(item), source=str(raw_path))
            else:
                result[key] = omit_leaf_payload(item, raw_path)
        return result
    return value


def main():
    raw = RUN / 'fixed_mass_sharp_repair_result.json'
    summary = RUN / 'summary.json'
    large = RUN / 'LARGE_FILES.tsv'
    if summary.exists() or large.exists():
        raise FileExistsError('The final extraction already exists')
    receipt = json.loads((RUN / 'execution.json').read_text())
    if receipt['status'] != 'finished':
        raise RuntimeError('The producer is not terminal')
    raw_id = identity(raw)
    if receipt.get(raw.name) != raw_id:
        raise ValueError('The terminal raw identity changed')
    payload = dict(raw_result=raw_id,
                   extraction='All non-leaf fields retained; complete raw leaf covers remain on the box.',
                   producer=omit_leaf_payload(json.loads(raw.read_text()), raw))
    data = json.dumps(payload, indent=2) + '\n'
    if len(data.encode()) > 20_000_000:
        raise RuntimeError('The non-leaf summary is too large to copy')
    with summary.open('x') as stream:
        stream.write(data)
    rows = ['path\tsize_bytes\tsha256_or_file_count\tbox_location']
    for name in (raw.name, 'stdout.log', 'stderr.log'):
        path = RUN / name
        info = identity(path)
        rows.append(f"run/{name}\t{info['bytes']}\t{info['sha256']}\tcompute-host:{path}")
    dependency = receipt['dependency']
    rows.append(f"existing pinned dependency\t{dependency['bytes']}\t{dependency['file_count']} files; tree sha256 {dependency['sha256']}\tcompute-host:{dependency['path']}")
    with large.open('x') as stream:
        stream.write('\n'.join(rows) + '\n')
    print(json.dumps({'summary': identity(summary), 'exclusions': identity(large)}))


if __name__ == '__main__':
    main()
