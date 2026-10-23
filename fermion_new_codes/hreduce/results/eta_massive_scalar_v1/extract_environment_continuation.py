"""Extract saved scalar data; never evaluate an integrand or rerun quadrature."""
import hashlib
import json
from pathlib import Path

ROOT = Path('<internal>/environment_continuation')
SOURCE = ROOT / 'fixed_mass_sharp_repair_result.json'
EXPECTED = '0f28c5a62b55b59f6ad3eba202fa04c563b077da4bd0a1a74df96f20a69fcd3e'


def main():
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError('Terminal source changed')
    record = json.loads(data)
    if record['status'] != 'incomplete':
        raise ValueError('Unexpected terminal status')
    panel = record.get('active_panel', {})
    leaves = panel.pop('leaves', [])
    panel['leaves_retained_only_in_source'] = len(leaves)
    record['source'] = dict(path=str(SOURCE), size=len(data), sha256=EXPECTED,
        extraction='All fields verbatim except active_panel.leaves; complete cover retained in source')
    target = ROOT / 'fixed_mass_scalar_result.json'
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'error', 'results', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
