"""cw59: write MOCK_n7_pilot_v1.md.wip (#497 STEP 2) from cw58 (measured) and the n = 5 v1 per-stage costs (for the one
stage the pilot does not exercise: the per-clock readouts). Every number is emitted here. No state readout appears.
Output: <internal>/MOCK_n7_pilot_v1.md.wip"""
import json, re, hashlib
P7 = '<internal>/n7pilot'; R5 = '<internal>/n5run'; B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def f(x, s=5):
    return f'{x:.{s}g}'

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

p = json.load(open(f'{P7}/cw58_pilot.json')); st = {s['stage']: s for s in p['stages']}; t = tparse(f'{P7}/cw58_time.txt')
D7 = p['dim']; D5 = 780
n5 = {fo: json.load(open(f'{R5}/{fo}_g1/result.json')) for fo in ('onsite', 'filtered')}
readout5 = {fo: [c for c in n5[fo]['cost'] if c['stage'] == 'ride_along_register_and_Cq'][0]['cpu_s'] + [c for c in n5[fo]['cost'] if c['stage'] == 'actions_admission_gated_readouts'][0]['cpu_s'] for fo in n5}
scale = (D7 / D5) ** 3
L = ['# MOCK n = 7 costing and admission pilot v1 (#497 STEP 2)', '',
     'WORKING FILE (.wip); not renamed. Author: busi (analyst engineer), busi inbox #497. A costing pilot only: NO state readout was computed for printing (no f_L, N_L, N_H, p_LL or any other); the full n = 7 test is NOT authorized and was not run. Builder: busi/code/clockwell/cw32_mockmodel.py (reproduces the saved n = 3 states; see MOCK_n5_plan_v2.md). Filtered form: endpoint-copy halo N_-1 = N_0 = 1/2, N_7 = N_6 = 1 (declared convention).', '',
     '## 1. Dimensions', '',
     f"Active dimension (complete N = 2 sector, 56 modes): {D7}; registered R x fermion {2 * D7}; joining-source extension n + 4 = 11 sites: " + ', '.join(f"{s['ext_dim']} (S shape {s['S_shape'][0]} x {s['S_shape'][1]})" for k, s in st.items() if k.startswith('joining_source')) + '. nnz: ' + ', '.join(f"{k.replace('assembly_', '')} {s['nnz']}" for k, s in st.items() if k.startswith('assembly')) + '.', '',
     '## 2. Measured stages (cap: ulimit -t 1000, ulimit -v 8000000; nice 10; 4 BLAS threads; box CPU)', '',
     '| stage | CPU s | wall s | peak RSS KiB (process so far) | certified residual ||H V - V Lambda||_F (ball, 128 bit) |', '|---|---|---|---|---|']
for s in p['stages']:
    L.append(f"| {s['stage']} | {f(s['cpu_s'])} | {f(s['wall_s'])} | {s['peak_rss_kib']} | {f(s['residual_frob_upper'], 3) if 'residual_frob_upper' in s else ''} |")
res_ok = all(s['residual_frob_upper'] < 1e-10 for s in p['stages'] if 'residual_frob_upper' in s)
mid = {fo: st[f'midpoints_256_{fo}']['per_midpoint_cpu_s'] for fo in ('onsite', 'filtered')}
L += ['', f"Process total: {t['user']} s user + {t['sys']} s sys, wall {t['wall']}, peak RSS {t['rss']} KiB, exit {t['exit']} (under the 1000 CPU s hard cap). Midpoint throughput (propagate B in the eigenbasis + one joining-source product): on-site {f(mid['onsite'] * 1e3, 4)} ms, filtered {f(mid['filtered'] * 1e3, 4)} ms CPU per midpoint. Residuals certify (all below 1e-10, the scale at which n = 5 certified): {res_ok}.", '',
     '## 3. Projection of a full n = 7 run (not authorized; for the decision only)', '',
     f"Assumptions, stated for the reader to change: six arms (two forms x g = 0, 1, 2); clocks to t = 7; Bwall at P = 1024 panels per unit (7168 midpoints per arm); protected segment (one more diagonalization and residual); per-clock readouts priced from the n = 5 measured readout stages scaled by (D7/D5)^3 = {f(scale, 4)} (the only extrapolated stage); plus the g = 0 yardstick midpoints at the same count; a second-read recertification of equal cost.", '']
arm = {}
for fo in ('onsite', 'filtered'):
    base = st[f'assembly_H1_{fo}']['cpu_s'] + st[f'diagonalization_H1_{fo}']['cpu_s'] + st[f'certified_residual_H1_{fo}_ball']['cpu_s'] + st[f'joining_source_n_plus_4_{fo}']['cpu_s']
    arm[fo] = base + 7168 * mid[fo] + (st[f'diagonalization_H1_{fo}']['cpu_s'] + st[f'certified_residual_H1_{fo}_ball']['cpu_s']) + 7 / 5 * readout5[fo] * scale
yard = sum(7168 * 3 * mid[fo] for fo in mid)
total = 3 * arm['onsite'] + 3 * arm['filtered'] + yard
L += ['| item | projected CPU s |', '|---|---|', f"| one on-site arm | {f(arm['onsite'], 4)} |", f"| one filtered arm | {f(arm['filtered'], 4)} |",
      f"| yardstick (both forms, 3 integrands per midpoint) | {f(yard, 4)} |", f"| six arms + yardstick | {f(total, 4)} |", f"| with an equal second-read recertification | {f(2 * total, 4)} |", '',
      f"Admission rule (reviewer A mock5x): ADMIT if the residual certifies and the projected full run (both forms, P pass, second-read recertification) <= 1e4 CPU s. Residual certifies: {res_ok}; projected {f(2 * total, 4)} CPU s <= 1e4: {2 * total <= 1e4}; reading: {'ADMIT (costing only; the test itself still needs the lead and owner rule)' if res_ok and 2 * total <= 1e4 else 'NOT ADMITTED: report and stop'}. Peak memory projection: the pilot peak {t['rss']} KiB came from the 128-bit ball residual of a 1540 x 1540 product; it stays under the 8 GB cap.", '',
      '## 4. Files (box <internal>/n7pilot, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
for p_ in (f'{P7}/cw58_pilot.json', f'{P7}/cw58_time.txt', f'{P7}/cw58_stdout.txt', f'{B}/code/cw32_mockmodel.py', f'{B}/code/cw58_n7_pilot.py', f'{B}/code/cw59_write_n7_pilot.py'):
    L.append(f'| `{p_}` | `{md5(p_)}` |')
L.append('')
out = f'{P7}/MOCK_n7_pilot_v1.md.wip'
open(out, 'w').write('\n'.join(L))
print('wrote', out, md5(out), 'projection', 2 * total, 'residual ok', res_ok)
