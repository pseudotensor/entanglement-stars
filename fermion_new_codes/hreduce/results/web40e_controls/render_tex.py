"""Render the requested lane fragments on the authorized box, retaining a lean result."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import json
from pathlib import Path
import re
import subprocess
import sys

request = json.load(sys.stdin)
root = Path(request['remote_root'])
root.mkdir(parents=True, exist_ok=True)
results = []
for item in request['sources']:
    name = item['name']
    assert re.fullmatch(r'[A-Za-z0-9_]+', name)
    folder = root / name
    folder.mkdir(exist_ok=True)
    (folder / 'input.tex').write_text(item['text'])
    wrapper = r''' + '"""' + r'''
\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,bm,mathtools}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\pagestyle{empty}
\begin{document}
\input{input.tex}
\end{document}
''' + '"""' + r'''
    (folder / 'render.tex').write_text(wrapper)
    proc = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
         '-file-line-error', 'render.tex'], cwd=folder,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=90,
    )
    (folder / 'compile.stdout.txt').write_text(proc.stdout)
    log = (folder / 'render.log').read_text() if (folder / 'render.log').exists() else proc.stdout
    problems = [line for line in log.splitlines()
                if 'Overfull ' in line or 'Undefined control sequence' in line
                or ('undefined' in line.lower() and ('reference' in line.lower()
                                                    or 'citation' in line.lower()))]
    pages = None
    if proc.returncode == 0:
        info = subprocess.check_output(['pdfinfo', str(folder / 'render.pdf')], text=True)
        match = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
        pages = int(match.group(1)) if match else None
    files = [p for p in folder.iterdir() if p.is_file()]
    results.append({
        'source': item['path'], 'remote_folder': str(folder),
        'passed': proc.returncode == 0 and not problems,
        'returncode': proc.returncode, 'pages': pages, 'problems': problems,
        'remote_file_count': len(files),
        'remote_bytes': sum(p.stat().st_size for p in files),
        'source_bytes': len(item['text'].encode()),
    })
print(json.dumps({'purpose': request['purpose'], 'results': results}, indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--purpose', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', default='<internal>/hreduce_web40e_20261016')
    parser.add_argument('sources', type=Path, nargs='+')
    args = parser.parse_args()
    sources = [{'name': p.stem, 'path': str(p), 'text': p.read_text()} for p in args.sources]
    request = {'purpose': args.purpose, 'remote_root': args.remote_root, 'sources': sources}
    remote_command = shlex.join([
        'env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'nice', '-n', '10', '<internal>/python', '-c', REMOTE,
    ])
    run = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
         'compute-host', remote_command], input=json.dumps(request),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    result = json.loads(run.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not all(item['passed'] for item in result['results']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
