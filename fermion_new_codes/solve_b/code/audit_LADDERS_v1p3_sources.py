"""Audit printed source-table entries without importing any producer code."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    known = hashlib.md5(b'abc').hexdigest() == '900150983cd24fb0d6963f7d28e17f72'
    if not known:
        raise RuntimeError('MD5 known answer failed')
    rows = []
    for report_name in args.reports:
        report = Path(report_name)
        for line_number, line in enumerate(report.read_text().splitlines(), 1):
            cells = [cell.strip() for cell in line.split('|')]
            if len(cells) != 7 or not cells[2].startswith('/') or not cells[2].endswith('.json'):
                continue
            member, source, byte_target, digest_target, script_target = cells[1:6]
            path = Path(source)
            row = dict(report=report.name, line=line_number, member=member, source=source,
                       printed=dict(bytes=int(byte_target), md5=digest_target, script=script_target))
            try:
                raw = json.loads(path.read_text())
                actual = dict(bytes=path.stat().st_size, md5=digest(path), script=raw.get('script'))
                row.update(actual=actual, matches={key: actual[key] == row['printed'][key]
                                                   for key in actual})
            except (OSError, ValueError) as error:
                row['unavailable'] = str(error)
            rows.append(row)
    if not rows:
        raise RuntimeError('No source table rows found')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Printed raw-run source-table metadata only; no solver code read or run.',
                  known_answers=dict(md5_abc=known), rows=rows)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), unavailable=sum('unavailable' in row for row in rows),
                          mismatches=sum(not all(row.get('matches', {}).values()) for row in rows))))


if __name__ == '__main__':
    main()
