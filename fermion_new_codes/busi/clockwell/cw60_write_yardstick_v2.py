"""cw60 (v2 of cw57: adds sec. 7 citing the independent second read and quoting its wall-bound observation verbatim, marked
as not a verdict change; writes the final name MOCK_n5_yardstick_v1.md). cw57: write MOCK_n5_yardstick_v1.md.wip (#497 STEP 1) from cw55 (verdict run) and cw56 (identical run with the E1 parts
recorded). Every number is emitted here; interval endpoints rounded outward.
Output: <internal>/MOCK_n5_yardstick_v1.md.wip"""
import json, re, hashlib
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
Y = '<internal>/yardstick_v1'; R = '<internal>/n5run'; B = '<internal>/clockwell'

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

def I(v, s=10):
    return f'[{out(v[0], s, ROUND_FLOOR)}, {out(v[1], s, ROUND_CEILING)}]'

def up(x, s=6):
    return out(x, s, ROUND_CEILING)

def lo(x, s=6):
    return out(x, s, ROUND_FLOOR)

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

SR = '<internal>/MOCK_n5_yardstick_secondread_v1.md'
_lines = open(SR).read().split('\n')
q0 = [i for i, x in enumerate(_lines) if x.startswith('**Looseness.**')][0]
q1 = [i for i, x in enumerate(_lines) if x.startswith('Rigor of the tighter variants')][0] + 1
quote = [x for i, x in enumerate(_lines[q0:q1]) if not _lines[q0 + i].startswith('**Other parts of the bound.**') and not _lines[q0 + i].startswith('Own parts of Bwall1(3)')]
a = json.load(open(f'{Y}/cw55_yardstick.json')); b = json.load(open(f'{Y}/cw56_yardstick.json'))
lab = lambda fo: 'on-site' if fo == 'onsite' else 'filtered (endpoint-copy halo)'
same = all(a['forms'][fo]['Delta_interval'] == b['forms'][fo]['Delta_interval'] and [c['Bwall1'] for c in a['forms'][fo]['clocks']] == [c['Bwall1'] for c in b['forms'][fo]['clocks']] for fo in a['forms'])
L = ['# MOCK n = 5 free-phase yardstick v1 (#497 STEP 1)', '',
     'Final file after the independent second read (sec. 7). Author: busi (analyst engineer), busi inbox #497 after ReviewA_mock5x_v1.md and ReviewB_mock5x_v1.md (joint). No new diagonalization: the saved n = 5 g = 0, 1, 2 Hamiltonians, the saved g = 0 eigenbases and the saved joining sources S of each form (md5 checked against the v1 result.json files). Filtered rows use the endpoint-copy halo N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention). No paper text.', '',
     '## 1. Verdicts', '',
     '| form | f_L(3) at g = 1 (saved v1) | f_L1(3) (yardstick) | I = f_L(3) - f_L1(3) | all prerequisites | verdict |', '|---|---|---|---|---|---|']
for fo, x in a['forms'].items():
    L.append(f"| {lab(fo)} | {I(x['f_L_g1_t3'])} | {I(x['clocks'][2]['f_L1'])} | {I(x['Delta_interval'])} | {x['prerequisites_all']} | **{x['verdict']}** |")
failed = {fo: [k for k, c in [('wall controls t = 3 on psi1 (Bwall1(3) <= 0.1)', x['clocks'][2]['Bwall1'] <= 0.1), ('||V B|| > 0', x['VB_norm2_lower'] > 0), ('lower N_L[psi1]/||psi1||^2 >= 0.05 at t = 1, 2, 3', all(c['N_L_over_norm2'][0] >= 0.05 for c in x['clocks'])), ('linearity H2 - H1 = H1 - H0', x['linearity_recheck_max_abs_H2_minus_2H1_plus_H0'] < 1e-12)] if not c] for fo, x in a['forms'].items()}
L += ['', 'Failed prerequisites (computed): ' + '; '.join(f"{lab(fo)}: {', '.join(v) if v else 'none'}" for fo, v in failed.items()) + '. Under the rule stated in #497, a prerequisite failure gives VOID; I is printed but not judged.', '',
      'Reading of I without judgement (not a verdict): ' + '; '.join(f"{lab(fo)}: I misses [-0.1, 0.1] on the {'negative' if x['Delta_interval'][1] < -0.1 else ('positive' if x['Delta_interval'][0] > 0.1 else 'straddling')} side" for fo, x in a['forms'].items()) + '. The VOID comes from the wall prerequisite on psi1 (sec. 3), not from I.', '',
      '## 2. Prerequisites and normalizers (ball-certified before any division)', '',
      '| form | max abs(H2 - 2 H1 + H0) (linearity recheck) | V maps HH outside LL (max abs) | H0 species mixing (max abs) | lower ||V B||^2 |', '|---|---|---|---|---|']
for fo, x in a['forms'].items():
    L.append(f"| {lab(fo)} | {up(x['linearity_recheck_max_abs_H2_minus_2H1_plus_H0'], 3)} | {up(x['structure']['V_from_HH_outside_LL'], 3)} | {up(x['structure']['H0_mixing'], 3)} | {lo(x['VB_norm2_lower'], 10)} |")
L += ['', '| form | t | ||psi1||^2 (ball) | lower ||psi1|| (after eps1) | eps1 | N_L[psi1] / ||psi1||^2 | f_L1 | Bwall1 | prerequisites at t |', '|---|---|---|---|---|---|---|---|---|']
for fo, x in a['forms'].items():
    for c in x['clocks']:
        L.append(f"| {lab(fo)} | {c['t']:g} | {I(c['psi1_norm2_ball'])} | {lo(c['psi1_norm_lower'], 10)} | {up(c['eps1'], 3)} | {I(c['N_L_over_norm2'])} | {I(c['f_L1'])} | {up(c['Bwall1'], 6)} | {c['prerequisites_ok']} |")
L += ['', 'psi1 lies entirely in the LL sector (V maps heavy pairs only to light pairs; H0 preserves species; both checked above), so N_L[psi1] / ||psi1||^2 = 2 and N_H[psi1] = 0: no heavy fraction is formed for psi1 and no heavy denominator is used. At the quadrature midpoints the integrands are unnormalized, so no normalizer is formed there.', '',
      'Saved g = 1 pure-B N_L and N_H (v1, reprinted):', '', '| form | t | N_L | N_H |', '|---|---|---|---|']
for fo, x in a['forms'].items():
    for c in x['saved_g1_NL_NH']:
        L.append(f"| {lab(fo)} | {c['t']:g} | {I(c['N_L'])} | {I(c['N_H'])} |")
L += ['', '## 3. Wall control of psi1 (S-source Duhamel bound, first order in g)', '',
      'e1 = psi1_ext - J psi1 obeys i e1\' = H0_ext e1 + S0 psi1 + V_ext e0 + S_V psi0 (S0 = the saved g = 0 joining source, S_V = S1 - S0, e0 the g = 0 wall error), so ||e1(t)|| <= E1(t) = (sampled S0 psi1 term) + (sampled S_V psi0 term) + ||V_ext|| int E0 + (quadrature term), with every operator norm a Schur bound; Bwall1(t) = 2 E1(t) / lower ||psi1(t)||. P = the smallest power of two (from 64) with the quadrature part of Bwall1(3) <= 0.01.', '',
      '| form | P | Lipschitz La, Lc | ||V||, ||H0||, C0, ||S_V||, ||V_ext|| (Schur) | Bwall1(1) / (2) / (3) | parts of Bwall1(3): S0 psi1 / S_V psi0 / V_ext E0 / quadrature |', '|---|---|---|---|---|---|']
for fo, x in a['forms'].items():
    p3 = b['forms'][fo]['clocks'][2]['Bwall1_parts']
    L.append(f"| {lab(fo)} | {x['P']} | {up(x['La'], 5)}, {up(x['Lc'], 5)} | {up(x['V_norm'], 5)}, {up(x['H0_norm'], 5)}, {up(x['C0'], 5)}, {up(x['S_V_norm'], 5)}, {up(x['V_ext_norm'], 5)} | " + ' / '.join(up(c['Bwall1'], 5) for c in x['clocks']) + f" | {up(p3['a'], 4)} / {up(p3['c'], 4)} / {up(p3['b'], 4)} / {up(p3['q'], 4)} |")
L += ['', f"The parts come from cw56, an identical rerun that records them (same Delta intervals and Bwall1 values as cw55: {same}). In both forms the V_ext E0 term (the g = 0 wall error carried through the converter, bounded by ||V_ext|| times the time integral of the zero-order Duhamel bound) is the largest part; the bound construction for psi1 is busi's (#497 names the S-source wall bound but not its first-order form), and a tighter first-order bound is possible but is not attempted here (one shot).", '',
      '## 4. Certified error terms', '', '| form | ||H0 V0 - V0 W||_F (ball) | ||V0 V0^dag - I||_F (ball) | ||V0 c - B|| (ball) |', '|---|---|---|---|']
for fo, x in a['forms'].items():
    L.append(f"| {lab(fo)} | {up(x['R0_frob'], 3)} | {up(x['G_frob'], 3)} | {up(x['d0'], 3)} |")
t55 = tparse(f'{Y}/cw55_time.txt'); t56 = tparse(f'{Y}/cw56_time.txt')
L += ['', '## 5. Cost (projection was about 60 CPU s)', '', '| item | CPU s | midpoints | wall | peak RSS KiB |', '|---|---|---|---|---|']
for fo, x in a['forms'].items():
    L.append(f"| {lab(fo)} setup (cw55) | {up(x['setup_cpu_s'], 4)} | | | |")
    L.append(f"| {lab(fo)} midpoints (cw55) | {up(x['midpoint_cpu_s'], 4)} | {x['midpoints']} | {up(x['wall_s'], 4)} s (form total) | |")
L += [f"| cw55 process | {t55['user']} user + {t55['sys']} sys | | {t55['wall']} | {t55['rss']} |", f"| cw56 process (parts rerun) | {t56['user']} user + {t56['sys']} sys | | {t56['wall']} | {t56['rss']} |", '',
      'Over the projection because the P rule required P = 2048 (on-site) and 4096 (filtered) for the first-order Lipschitz constants; caps held (ulimit -v 8000000, ulimit -t 3600, nice 10, 4 threads, box CPU).', '',
      '## 6. For the second reader', '',
      'Inputs (busi box, md5 in sec. 7): n5run/<form>_g0/hamiltonian.npz, eigenbasis.npz, joining_source_S.npz; n5run/<form>_g1/hamiltonian.npz, joining_source_S.npz, result.json (f_L(3), N_L, N_H); n5run/<form>_g2/hamiltonian.npz. Steps: (i) V = H1 - H0; check H2 - H1 = H1 - H0; (ii) psi1(t) = -i V0 [ e^{-iWt} ( (Vt o K(t)) c ) ], Vt = V0^dag V V0, c = V0^dag B, K_mk(t) = (e^{i(w_m - w_k)t} - 1)/(i(w_m - w_k)) (= t if equal); (iii) f_L1(t) = (number of light particles on sites 1-4)/2 averaged over |psi1|^2 / ||psi1||^2 (psi1 is LL only); (iv) Bwall1 as in sec. 3 with E0 the g = 0 Duhamel prefix (f0 = ||S0 psi0||, L0 from v1 g = 0 result.json); (v) verdict: PASS if I within [-0.1, 0.1], FAIL if I misses it, VOID if a prerequisite fails (Bwall1(3) <= 0.1, ||VB|| > 0, N_L fraction >= 0.05, linearity). Criterion: the integers and verdicts of secs. 1-3; an independent tighter wall bound for psi1 would be a new construction, not a check of this one.', '',
      '## 7. Independent second read', '',
      f"Second read: `{SR}` (md5 `{md5(SR)}`, copy of <internal>/MOCK_n5_yardstick_secondread_v1.md), coordinator-run, different agent: all 9 items AGREE, no defects; both forms VOID by the rule as executed. Its judgment of the first-order wall bound is VALID but needlessly loose in the normalization step. Its observation is quoted below verbatim (lines {q0 + 1} to {q1} of that file). It is NOT a verdict change: the run is one shot, and whether to adopt the tighter bound is for the lead and the models.", ''] + ['> ' + x if x.strip() else '>' for x in quote] + ['',
      '## 8. Files (box <internal>/yardstick_v1, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
fl = [f'{Y}/cw55_yardstick.json', f'{Y}/cw56_yardstick.json'] + [f'{R}/{fo}_g{g}/{fn}' for fo in ('onsite', 'filtered') for g, fns in (('0', ('hamiltonian.npz', 'eigenbasis.npz', 'joining_source_S.npz', 'result.json')), ('1', ('hamiltonian.npz', 'joining_source_S.npz', 'result.json')), ('2', ('hamiltonian.npz',))) for fn in fns]
fl += [f'{B}/code/{x}' for x in ('cw32_mockmodel.py', 'cw55_yardstick.py', 'cw56_yardstick_parts.py', 'cw57_write_yardstick.py', 'cw60_write_yardstick_v2.py')] + [SR]
for p_ in fl:
    L.append(f'| `{p_}` | `{md5(p_)}` |')
L.append('')
p = f'{Y}/MOCK_n5_yardstick_v1.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p))
