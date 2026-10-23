"""cw65: write MOCK_n7_v1.md.wip (#500) from the two cw64 arm results. Every number is emitted here; interval endpoints are
rounded outward. Output: <internal>/MOCK_n7_v1.md.wip"""
import json, os, re, hashlib
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
R = '<internal>/n7run'; B = '<internal>/clockwell'
RULE = '<internal>/MOCK_n7_rule_agreed_v1.md'  # box copy of deliveries/lead_gaps/MOCK_n7_rule_agreed_v1.md

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def out(x, s, mode):
    if x == 0:
        return '0'
    d = Decimal(repr(float(x))); e = d.adjusted() - s + 1
    q = d.quantize(Decimal(1).scaleb(e), rounding=mode)
    return format(q.normalize(), 'g') if abs(q.adjusted()) < 6 else format(q, 'E')

up = lambda x, s=6: 'n/a' if x is None else out(x, s, ROUND_CEILING)
lo = lambda x, s=6: 'n/a' if x is None else out(x, s, ROUND_FLOOR)
I = lambda v, s=10: 'undefined' if v is None else f'[{lo(v[0], s)}, {up(v[1], s)}]'

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

forms = ('onsite', 'filtered')
r = {fo: json.load(open(f'{R}/{fo}/result.json')) for fo in forms}
tm = {fo: tparse(f'{R}/{fo}_time.txt') for fo in forms}
lab = lambda fo: 'on-site' if fo == 'onsite' else 'filtered (endpoint-copy halo: N_-1 = N_0 = 1/2, N_7 = N_6 = 1)'
cpu = {fo: float(tm[fo]['user']) + float(tm[fo]['sys']) for fo in forms}
spent = sum(cpu.values())
L = ['# MOCK n = 7 transport-only test v1 (#500; owner-approved)', '',
     f"WORKING FILE (.wip); an independent second read follows; not renamed. Author: busi (analyst engineer). Run ONE SHOT per form, exactly per deliveries/lead_gaps/MOCK_n7_rule_agreed_v1.md (box copy `{RULE}`, md5 `{md5(RULE)}`), with busi's own builder (busi/code/clockwell/cw32_mockmodel.py) and busi/code/clockwell/cw64_n7_run.py, one process per form. A PASS or FAIL here is a transport statement at n = 7 only; no release, purification, recovery, conversion time or Page time follows. No paper text.", '',
     '## 1. Verdicts (t* = 4, fixed)', '',
     '| form | f_L(4) at g = 1 | f_L1(4) (n = 7 free-phase yardstick) | I = f_L(4) - f_L1(4) | 1/2 (printed alongside, unjudged) | all prerequisites | verdict |', '|---|---|---|---|---|---|---|']
for fo in forms:
    x = r[fo]
    L.append(f"| {lab(fo)} | {I(x.get('f_L_t4'))} | {I(x.get('f_L1_t4'))} | {I(x.get('I'))} | 1/2 | {all(x['prerequisites'].values()) if 'prerequisites' in x else False} | **{x.get('verdict')}**{(' (' + x['void_reason'] + ')') if x.get('void_reason') else ''} |")
L += ['', '## 2. Prerequisites (ball-certified on the actual states before any division)', '',
      '| form | linearity max abs(H2 - 2 H1 + H0) | residual H0 / H1 (ball, <= 1e-10) | lower ||VB||^2 | lower N_L/||psi||^2 >= 0.05 at t = 1..4 (g = 1) | the same on psi1 | wall g = 1 at t = 4: E / Dunkl-Williams | wall psi1 at t = 4 (Dunkl-Williams, construction C) | l - E > 0 (g = 1 / psi1) |', '|---|---|---|---|---|---|---|---|---|']
for fo in forms:
    x = r[fo]
    if 'wall_psi1' not in x:
        L.append(f"| {lab(fo)} | run stopped before the walls: {x.get('void_reason')} | | | | | | | |"); continue
    wg = x['wall_g1']; wp = x['wall_psi1']
    L.append(f"| {lab(fo)} | {up(x['linearity'], 3)} | {up(x['residual_H0'], 3)} / {up(x['residual_H1'], 3)} | {lo(x['VB_norm2_lower'], 8)} | {x['prerequisites']['N_L_g1_all_clocks']} | {x['prerequisites']['N_L_psi1_all_clocks']} | {up(wg['E']['4'], 6)} / {up(wg['DW_t4'], 6)} | {up(wp['DW']['4'], 6)} | {lo(wg['l_minus_E_t4'], 6)} / {lo(wp['l_minus_E']['4'], 6)} |")
L += ['', 'Per clock (g = 1 pure B; psi1):', '', '| form | t | N_L/||psi||^2 (g = 1) | f_L (g = 1) | p_LL (printed, unjudged) | N_L/||psi1||^2 | f_L1 | psi1 wall (Dunkl-Williams) |', '|---|---|---|---|---|---|---|---|']
for fo in forms:
    x = r[fo]
    if 'psi1' not in x:
        continue
    for a, b in zip(x['g1'], x['psi1']):
        dwp = x.get('wall_psi1', {}).get('DW', {}).get(str(int(a['t'])))
        L.append(f"| {lab(fo)} | {a['t']:g} | {I(a['N_L'])} | {I(a.get('f_L'))} | {I(a['p_LL'])} | {I(b['N_L_over_norm2'])} | {I(b['f_L1'])} | {up(dwp, 6)} |")
L += ['', 'N_L for the g = 1 state is the light number per unit norm (N2 the only state divisor); f_L is formed only after its N_L lower bound is certified positive. psi1 lies in the light-pair sector (weight outside it: ' + ', '.join(f"{lab(fo)} {up(max(b['weight_outside_LL'] for b in r[fo]['psi1']), 3)}" for fo in forms if 'psi1' in r[fo]) + '), so N_L/||psi1||^2 = 2; no heavy fraction is formed for either state, so no heavy normalizer is used.', '',
      '## 3. Wall setup (P set at run start, before any midpoint)', '',
      '| form | ||T_ext||_2 HH -> LL (float / certified upper, 11 sites) | T certificate | C_join (g = 1) | L (g = 1) | La / Lc (psi1) | P (g = 1) | quadrature share (g = 1) | P (psi1) | quadrature share (psi1) |', '|---|---|---|---|---|---|---|---|---|---|']
for fo in forms:
    w = r[fo].get('wall_setup')
    if w:
        L.append(f"| {lab(fo)} | {up(w['T_norm_float'], 10)} / {up(w['T_norm_certified_upper'], 10)} | {w['T_psd_certificate']} | {up(w['C_join_g1'], 6)} | {up(w['L_g1'], 6)} | {up(w['La'], 6)} / {up(w['Lc'], 6)} | {w['P_g1']} | {up(w['quadrature_share_g1'], 4)} | {w['P_psi1']} | {up(w['quadrature_share_psi1'], 4)} |")
L += ['', 'Sector restriction of the block norm: the zero-order wall error is a heavy-pair vector, so only the heavy-pair to light-pair block of V_ext acts on it; the other blocks of V_ext from the heavy-pair sector are zero (' + ', '.join(f"{lab(fo)} {up(r[fo]['wall_setup']['T_other_blocks_from_HH'], 3)}" for fo in forms if 'wall_setup' in r[fo]) + '). For the g = 1 state both the frozen bound E and its Dunkl-Williams form are printed; the verdict uses the Dunkl-Williams value, the larger of the two (the rule\'s literal normalization).', '',
      '## 4. CPU spent and retained output', '',
      '| form | CPU s (user + sys) | wall | peak RSS KiB | exit | budget projection before midpoints (CPU s) | allowance given |', '|---|---|---|---|---|---|---|']
for fo in forms:
    b = r[fo].get('budget', {})
    L.append(f"| {lab(fo)} | {up(cpu[fo], 6)} | {tm[fo]['wall']} | {tm[fo]['rss']} | {tm[fo]['exit']} | {up(b.get('projected_midpoint_cpu_s'), 5)} | {r[fo]['allowance_cpu_s']} |")
L += ['', f"busi CPU spent on this test: {up(spent, 6)} s of its 5000 s share; remaining for the independent second read under the 1e4 total cap: {lo(1e4 - spent, 6)} s. Caps: nice 10, ulimit -v 8000000, ulimit -t per process (on-site 2500, filtered 3600), BLAS threads 4 (on-site) and 2 (filtered, during the B' gate package), box CPU only.", '',
      'Per stage (CPU s):', '', '| form | ' + ' | '.join(c['stage'] for c in r['onsite']['cost']) + ' |', '|---|' + '---|' * len(r['onsite']['cost'])]
for fo in forms:
    L.append(f"| {lab(fo)} | " + ' | '.join(up(c['cpu_s'], 4) for c in r[fo]['cost']) + ' |')
ret = {fo: {fn: os.path.getsize(f'{R}/{fo}/{fn}') for fn in os.listdir(f'{R}/{fo}')} for fo in forms}
L += ['', 'Retained (small, by design: /data is nearly full): ' + '; '.join(f"{lab(fo)}: " + ', '.join(f"{fn} {sz} bytes" for fn, sz in sorted(ret[fo].items())) for fo in forms) + '. No Hamiltonian, eigenbasis or joining source is stored; the second reader rebuilds them.', '',
      '## 5. For the second reader', '',
      f"Budget: busi spent {up(spent, 6)} CPU s; the remaining cap for the independent read is {lo(1e4 - spent, 6)} CPU s (1e4 total), 8 GB. Inputs: none from icloud. Model: busi/code/clockwell/cw32_mockmodel.py at n = 7 (modes 8x + 4A + c; well (1/2, 1, ..., 1); m_H = binary64(3/5); K exact integers; filtered: endpoint-copy halo N_-1 = 1/2, N_7 = 1; basis by occupation bitmask). Steps: (1) H0, H1, H2 (g = 0, 1, 2) and linearity; (2) eigendecompose H0, H1, certify residuals <= 1e-10; (3) g = 1: psi(t) = exp(-i H1 t) B, t = 1..4; N_L/||psi||^2, f_L = <N_L on sites 1-6>/<N_L>; (4) psi1(t) = -i int_0^t U0(t-s) V U0(s) B ds (V = H1 - H0), f_L1 = light particles on sites 1-6 / 2; (5) walls at t = 4 on the 11-site empty extension: g = 1 S-source Duhamel (64-panel-per-unit midpoint rule refined to P as printed), first-order bound C on psi1 (combined source, certified HH -> LL block norm in the V_ext e0 term), both normalized by 2E/(2l - E); (6) I = f_L(4) - f_L1(4), PASS if inside [-0.1, 0.1], FAIL (signed) if disjoint, VOID otherwise or on any prerequisite. The state vectors in states_small.npz (psi(t), psi1(t), t = 1..4; md5 below) allow a direct check of the readouts without propagation. Any verdict-relevant DISAGREE gives VOID (rule).", '',
      '## 6. Files (box <internal>/n7run, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
for p_ in [f'{R}/{fo}/{fn}' for fo in forms for fn in sorted(ret[fo])] + [f'{R}/{fo}_time.txt' for fo in forms] + [RULE] + [f'{B}/code/{x}' for x in ('cw32_mockmodel.py', 'cw45_n5_run.py', 'cw55_yardstick.py', 'cw62_tnorm_dw.py', 'cw64_n7_run.py', 'cw65_write_n7.py')]:
    L.append(f'| `{p_}` | `{md5(p_)}` |')
L.append('')
p = f'{R}/MOCK_n7_v1.md.wip'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p), 'spent', spent)
