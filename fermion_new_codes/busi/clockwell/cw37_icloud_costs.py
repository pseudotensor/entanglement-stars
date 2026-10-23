"""cw37: icloud's MEASURED n = 2, 3 costs (#138 sec. 8 report input), from the saved termination rows and their
per-phase records (read-only). Per job: wall, CPU, peak RSS; per clock: summed phase CPU of phases whose name
carries that clock; per category: summed phase CPU by readout family. Also the readout-job time files.
Output: <internal>/cw37_icloud_costs.json"""
import json, re, os, collections
ROOT = '<internal>/solve_b_bulk'
OUT = '<internal>/cw37_icloud_costs.json'
CATS = [('coherence', r'coherence|dephased'), ('entropy', r'entropy'), ('register', r'register\b|register_\(|first_'),
        ('actions', r'action'), ('wall', r'wall'), ('protected_collection', r'protected|certified|collection|purification|shared'),
        ('assembly_and_static', r'assembl|model|matrix|monomial|literal|source|eigenbasis|exact|independent'), ('expm_check', r'expm')]
res = {'jobs': []}
for top, arm in (('MOCK_unitary_lead_run_v1', 'on-site'), ('MOCK_filtered_physical_v1', 'filtered (endpoint-copy halo)')):
    for name in sorted(os.listdir(f'{ROOT}/{top}')):
        p = f'{ROOT}/{top}/{name}/written_report.md'
        if not (name.startswith('even_') or name.startswith('odd_')) or not os.path.isfile(p):
            continue
        term = None
        for line in open(p):
            if line.startswith('{"kind":"termination"'):
                term = json.loads(line)['data']
        ph = term.get('phases') or []
        per_clock = collections.defaultdict(float); per_cat = collections.defaultdict(float); other = 0.0
        for x in ph:
            nm = x['phase']; c = x.get('cpu') or 0.0
            m = re.search(r'[_(]([0-9]+\.[0-9]+)', nm)
            if m:
                per_clock[float(m.group(1))] += c
            for cat, pat in CATS:
                if re.search(pat, nm):
                    per_cat[cat] += c; break
            else:
                other += c
        per_cat['other'] = other
        res['jobs'].append(dict(arm=arm, job=name, wall_s=term.get('wall'), cpu_s=term.get('cpu'), peak_rss_kib=term.get('rss_kib'),
                                threads=term.get('threads'), phases=len(ph), per_clock_cpu_s=dict(sorted(per_clock.items())),
                                per_category_cpu_s=dict(per_cat), serialization_cpu_s=term.get('serialization_cpu')))
        print(arm, name, term.get('wall'), term.get('cpu'), term.get('rss_kib'), flush=True)
for top in ('MOCK_comparison_readouts_v1', 'MOCK_filtered_comparison_readouts_v1'):
    for fn in sorted(os.listdir(f'{ROOT}/{top}')):
        if fn.endswith('_time.txt'):
            t = open(f'{ROOT}/{top}/{fn}').read()
            g = lambda pat: (re.search(pat, t).group(1) if re.search(pat, t) else None)
            res.setdefault('readout_jobs', []).append(dict(dir=top, job=fn[:-9], user_s=g(r'User time \(seconds\): ([0-9.]+)'),
                                                           wall=g(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), rss_kib=g(r'Maximum resident set size \(kbytes\): ([0-9]+)')))
json.dump(res, open(OUT, 'w'), indent=1)
