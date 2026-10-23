"""cw54 (v2 of cw53: second-read repairs on the N2 bound and eps(t) wording, gate margins, second-read section; writes
the final name MOCK_n5_v1.1.md). cw53: write MOCK_n5_v1.1.md.wip (#495) from cw50 (wall recertification), cw51 (bright registered input), cw52 (Pi sector).
v1 (MOCK_n5_v1.md) is kept. Every number is emitted here; interval endpoints are rounded outward.
Output: <internal>/MOCK_n5_v1.1.md.wip"""
import json, os, re, hashlib
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
R = '<internal>/n5run'; B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def f(x, s=6):
    return 'n/a' if x is None else f'{x:.{s}g}'

def out(x, s, mode):
    if x == 0:
        return '0'
    d = Decimal(repr(float(x))); e = d.adjusted() - s + 1
    q = d.quantize(Decimal(1).scaleb(e), rounding=mode)
    return format(q.normalize(), 'g') if abs(q.adjusted()) < 6 else format(q, 'E')

def I(v, s=10):
    return 'undefined' if v is None else f'[{out(v[0], s, ROUND_FLOOR)}, {out(v[1], s, ROUND_CEILING)}]'

def up(x, s=8):
    return out(x, s, ROUND_CEILING)

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

rc = json.load(open(f'{R}/recert_v1/cw50_recert.json')); br = json.load(open(f'{R}/bright_v1/cw51_bright.json')); pi = json.load(open(f'{R}/pi_v1/cw52_pi.json'))
lab = lambda fo: 'on-site' if fo == 'onsite' else 'filtered (endpoint-copy halo)'
HALO = 'endpoint-copy halo: N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention, not a measurement)'
v1 = f'{R}/MOCK_n5_v1.md'
L = ['# MOCK n = 5 v1.1: one-shot wall recertification, bright registered input, Pi-sector check (#495)', '',
     f'Final file after the independent second read (sec. 6). Author: busi (analyst engineer), busi inbox #495, after ReviewA_mock4_v1.md and ReviewB_mock4_v1.md (both accept n = 5). v1 is kept unchanged: `{v1}` md5 `{md5(v1)}`. Filtered rows carry the {HALO}. No paper text.', '',
     '## 1. One-shot recertification of the two g = 1 pure-B wall bounds', '',
     f"Rule stated in advance (#495): P = the smallest power of two with quadrature term 5 L / (4 P) <= 0.01 for both g = 1 arms. Computed: P = {rc['P_rule']}; quadrature term at t = 5: " + '; '.join(f"{lab(fo)} {up(v, 6)}" for fo, v in rc['quadrature_term_at_5'].items()) + ". Inputs: the saved hamiltonian.npz, eigenbasis.npz and joining_source_S.npz of each g = 1 arm (md5 checked against the v1 result.json: " + ', '.join(f"{lab(fo)} {a['saved_md5_verified']}" for fo, a in rc['arms'].items()) + "); every midpoint contribution recomputed at each P (no subtraction of the old remainder); C_join, L, clocks, thresholds and gate text unchanged; one shot, no further refinement.", '',
     'Certification note: each midpoint uses f_mid = ||S phi|| (1 + 1e-12) / sqrt(lower N2) + 1e-13 + C_join eps_n with lower N2 = fl(sum |phi_i|^2)(1 - gamma) (a priori floating bound of a sum of nonnegative terms; v1 used ball arithmetic for this one quantity). The correct count for a complex state of dimension D = 780 is m = 2D = 1560 real squared terms, so the bound is gamma_{2D} = 2D u / (1 - 2D u); the code used gamma_D with D = 780, and the final multiplication by (1 - gamma) adds one more rounding; both differences (below 1e-13 relative) are covered by the (1 + 1e-12) inflation of the numerator (the second read\'s own fully stated bound inflates by at most 3.8e-13 and held at 80 exact midpoints per arm). No gate margin depends on this bound: ' + '; '.join(f"{lab(fo)}: |Bwall(3) - 0.1| = {f(abs(a['ladder'][str(rc['P_rule'])]['Bwall']['3'] - 0.1), 3)}, |Bwall(4) - 0.1| = {f(abs(a['ladder'][str(rc['P_rule'])]['Bwall']['4'] - 0.1), 3)}" for fo, a in rc['arms'].items()) + f" (smallest {f(min(min(abs(a['ladder'][str(rc['P_rule'])]['Bwall'][t] - 0.1) for t in ('3', '4')) for a in rc['arms'].values()), 3)}), while the N2 question moves Bwall by less than 1e-13. eps(t) is recomputed from the saved eigenbasis with the eigen-residual certified in ball arithmetic: " + '; '.join(f"{lab(fo)} ||H V - V Lambda||_F <= {up(a['residual_frob_upper'], 3)}, ||V c - psi0|| <= {up(a['d0'], 3)}" for fo, a in rc['arms'].items()) + "; it does not include a ball-arithmetic bound on the Hamiltonian assembly (the binary64 assembly error of H itself enters only through the saved e_H, which is the exact parameter deltas plus an a priori 1e-13 ||H||_F allowance); that error is about 1e-11 and harmless at these margins.", '',
     '| form | P | sampled-integrand sum at t = 1 / 2 / 3 / 4 / 5 | quadrature term t L / (4P) at t = 3 / 5 | Bwall at t = 1 / 2 / 3 / 4 / 5 |', '|---|---|---|---|---|']
for fo, a in rc['arms'].items():
    for P, x in a['ladder'].items():
        L.append(f"| {lab(fo)} | {P} | " + ' / '.join(up(x['sampled_sum'][str(t)], 6) for t in range(1, 6)) + f" | {up(x['quadrature_term']['3'], 4)} / {up(x['quadrature_term']['5'], 4)} | " + ' / '.join(up(x['Bwall'][str(t)], 6) for t in range(1, 6)) + ' |')
L += ['', '| form | original verdict (P = 64, v1) | original t* | recertified t* (P = ' + str(rc['P_rule']) + ') | recertified verdict | f_L(t*) | f_H(t*) | p_LL(t* - 1) | <N_L>(t*) | <N_H>(t*) |', '|---|---|---|---|---|---|---|---|---|---|']
for fo, a in rc['arms'].items():
    g = a['gate_recertified']; o = a['gate_original_P64']
    L.append(f"| {lab(fo)} | {o['verdict']} ({o.get('reason', '')}) | {o.get('t_star')} | {g.get('t_star')} | **{g['verdict']}** | {I(g.get('f_L'))} | {I(g.get('f_H'))} | {I(g.get('p_LL_prev'))} | {I(g.get('N_L'))} | {I(g.get('N_H'))} |")
fails = all(a['gate_recertified']['verdict'] == 'FAIL' for a in rc['arms'].values())
L += ['', ("Both recertified g = 1 gates read FAIL under the frozen rule: t* = 3, the prerequisites hold with lower bounds (lower p_LL(2) >= 0.1, lower <N_L>(3) >= 0.05, <N_H>(3) certified positive) and upper f_L(3) < 1/2. Meaning: for the pure bright input B at n = 5, the converted light had not mostly left the core (sites 1-4 hold less than half of <N_L>) by t = 3, the last wall-controlled clock; nothing follows about later clocks or larger chains, and no conversion time, Page time, turnover or maximum. The original P = 64 VOIDs are preserved above.") if fails else 'See the table for each recertified verdict.', '',
      '## 2. Bright registered input (ungated; no decoder SDP)', '',
      "Psi_bright = (|0_R>|01>_H + |1_R> B')/sqrt 2, B' = (|03>_H + |12>_H)/sqrt 2 at site 0, propagated with the saved eigenbases (md5 checked). Admission per clock: N2 certified positive (ball arithmetic) with |N2 - 1| within (1 + 2 eps)^2 - 1; Gram distance of the two branches from I_2 within 2 (2 sqrt 2 eps + 2 eps^2). D_RK for light-only and complete whole-lead output (allowance +-3 eps_n); the (B, D) dark-code control values from v1 beside them.", '',
      '| form | g | t | N2 enclosure | admitted | Gram distance | Gram admitted | D_RK light (bright) | D_RK complete (bright) | allowance | D_RK light (B, D control) | D_RK complete (B, D control) |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
for k, info in br.items():
    if k == 'cpu_s_total':
        continue
    for r in info['rows']:
        L.append(f"| {lab(info['form'])} | {info['g']} | {f(r['t'])} | {I(r['N2'], 16)} | {r['admitted']} | {up(r['gram_distance'], 3)} | {r['gram_admitted']} | {f(r['D_light'], 10)} | {f(r['D_all'], 10)} | {up(r['allowance'], 3)} | {f(r['dark_control_D_light'], 10)} | {f(r['dark_control_D_all'], 10)} |")
alladm = all(r['admitted'] and r['gram_admitted'] for k, i in br.items() if k != 'cpu_s_total' for r in i['rows'])
L += ['', f"All bright-input states admitted (N2 and Gram): {alladm}. These are ungated ride-along readouts; D_RK > 1e-4 means not decoupled to the tolerance; the optimal-decoder audit for Psi_bright is a separately approved follow-up, not run here.", '',
      '## 3. Pi-sector check of the extra dark directions (printed readout)', '',
      "Pi = (-1)^x beta per mode; Pi_total in {0, +-2} for a two-particle state. W = the exactly dark heavy N = 2 subspace (largest H_H-invariant subspace of ker T at g = 1, as in plan v2 / cw42). Expected (reviewer A mock4): dim(W intersect {Pi_total = +-2}) = 60 on-site and 20 filtered at n = 5, 18 and 6 at n = 3.", '',
      '| source | max abs [Pi, h] (one-body) | max abs [Pi_total, H0] | max abs [Pi_total, H] at g = 1 | dim W | dim(W intersect Pi_total = +-2) | expected | dim(W intersect Pi_total = 0) | dim(W intersect protected sector) | protected dim |', '|---|---|---|---|---|---|---|---|---|---|']
exp = {('onsite', 5): 60, ('filtered', 5): 20, ('onsite', 3): 18, ('filtered', 3): 6}
for x in pi:
    L.append(f"| {x['label']} | {f(x['max_comm_Pi_h'], 3)} | {f(x['max_comm_Pitotal_H0'], 3)} | {f(x['max_comm_Pitotal_H_g1'], 4)} | {x['dim_W']} | {x['dim_W_Pi_pm2']} | {exp[(x['form'], x['n'])]} ({'match' if x['dim_W_Pi_pm2'] == exp[(x['form'], x['n'])] else 'DIFFERENT'}) | {x['dim_W_Pi_0']} | {x['dim_W_cap_protected']} | {x['protected_dim']} |")
acc = []
for n in (3, 5):
    on = [x for x in pi if x['form'] == 'onsite' and x['n'] == n and x['source'] == 'busi builder'][0]
    fi = [x for x in pi if x['form'] == 'filtered' and x['n'] == n and x['source'] == 'busi builder'][0]
    acc.append(f"n = {n}: on-site extras dim W - protected = {on['dim_W'] - on['protected_dim']}; on-site dim(W cap Pi = +-2) - filtered dim(W cap Pi = +-2) = {on['dim_W_Pi_pm2'] - fi['dim_W_Pi_pm2']} (the filtered W is exactly the protected sector: {fi['dim_W'] == fi['dim_W_cap_protected'] == fi['protected_dim']}); both on-site Pi_total = 0 and filtered Pi_total = 0 parts equal: {on['dim_W_Pi_0'] == fi['dim_W_Pi_0']}")
okacc = all(('True' in a_.split('equal:')[-1]) and (a_.split('extras dim W - protected = ')[1].split(';')[0] == a_.split('filtered dim(W cap Pi = +-2) = ')[1].split(' ')[0]) for a_ in acc)
L += ['', 'Accounting (computed): ' + '; '.join(acc) + '. ' + ('So the on-site extra dark directions all lie in Pi_total = +-2 (equal extra count and equal Pi_total = 0 parts), consistent' if okacc else 'The accounting does NOT close as expected; see the numbers. Compare')
      + " with reviewer A's account (pairs with Pi_total = +-2 in the flavour-triplet part never share a site, so the on-site vertex never sees them). Pi_total commutes with H0 but not with the full H (commutator columns above); it is a statement about the free dynamics and the dark subspace, N = 2 only.", '']
# costs
t50 = tparse(f'{R}/recert_v1/cw50_time.txt'); t51 = tparse(f'{R}/bright_v1/cw51_time.txt'); t52 = tparse(f'{R}/pi_v1/cw52_time.txt')
L += ['## 4. Measured cost (projections were about 1.1e3 CPU s for the recertification and about 73 CPU s for the bright input)', '',
      '| item | CPU s | wall | peak RSS KiB |', '|---|---|---|---|']
for fo, a in rc['arms'].items():
    L.append(f"| recertification setup, {lab(fo)} (load, md5, ball residual, eps terms) | {f(a['setup_cpu_s'], 4)} | {f(a['setup_wall_s'], 4)} s | |")
    for P, x in a['ladder'].items():
        L.append(f"| midpoints, {lab(fo)}, P = {P} ({5 * int(P)} midpoints) | {f(x['cpu_s'], 4)} | {f(x['wall_s'], 4)} s | |")
L.append(f"| recertification process total (cw50) | {t50['user']} user + {t50['sys']} sys | {t50['wall']} | {t50['rss']} |")
for k, info in br.items():
    if k != 'cpu_s_total':
        L.append(f"| bright input, {lab(info['form'])}, g = {info['g']} | {f(info['cpu_s'], 4)} | {f(info['wall_s'], 4)} s | |")
L += [f"| bright input process total (cw51) | {t51['user']} user + {t51['sys']} sys | {t51['wall']} | {t51['rss']} |",
      f"| Pi-sector check (cw52) | {t52['user']} user + {t52['sys']} sys | {t52['wall']} | {t52['rss']} |", '',
      'Caps held: ulimit -v 8000000, nice 10, 4 BLAS threads, box CPU only, every process far under 2 CPU-h. The recertification is far below its projection because the midpoint N2 bound uses the a priori floating bound (sec. 1) instead of a per-midpoint ball-arithmetic sum.', '',
      '## 5. For the second reader', '',
      f"Inputs (busi box files, md5 in sec. 7): the v1 arm directories n5run/<form>_g<G>/ (hamiltonian.npz, eigenbasis.npz, joining_source_S.npz, result.json). (1) Recertification, each g = 1 form: phi(t) = V exp(-i w t) V^dagger B (B = |01>_H at site 0, basis order by occupation bitmask, see busi/code/clockwell/cw32_mockmodel.py); for P in (64, 256, 1024): sampled sum(t) = sum over midpoints (k + 1/2)/P < t of (1/P)(||S phi|| / ||phi|| + C_join eps_n); Bwall(t) = sampled sum + t L / (4 P), with C_join and L from result.json (wall). P rule: smallest power of two with 5 L / (4 P) <= 0.01. Gate: t* = last t in 1..5 with Bwall <= 0.1 at P = {rc['P_rule']}; then the frozen gate of plan v2 sec. 5 with the pure-B readouts of v1 result.json (clocks). Criterion: same t* and verdicts; the error terms (eps_n ~ 1e-11) cannot move a verdict here. (2) Bright input: rows (B/sqrt 2, B'/sqrt 2), propagate each with the saved eigenbasis; N2, Gram matrix 2 <r_a, r_b> / N2 against I_2; D_RK by the graded partial trace (sign -1 when the traced mode precedes the kept mode), K = complement of the output (light modes of sites 1-4, or all modes of sites 1-4). Criterion: values within the printed allowance; states md5 in sec. 6. (3) Pi: pi(m) = (-1)^(m div 8) beta[m mod 4], beta = (1, 1, -1, -1); W as in cw42 (subspace recursion with absolute rank threshold 1e-10 x operator scale); dim(W cap S) = number of eigenvalues of Z^dagger P_S Z above 1 - 1e-9. Criterion: the integers of sec. 3.", '',
      '## 6. Independent second read and repairs', '',
      f"Second read: `<internal>/MOCK_n5_v11_secondread_v1.md` (md5 `{md5('<internal>/MOCK_n5_v11_secondread_v1.md')}`, copy of <internal>/MOCK_n5_v11_secondread_v1.md), coordinator-run, different agent: all 8 items AGREE; both g = 1 gates FAIL. Repairs, regenerated by cw54_write_n5_v1_1_v2.py (no hand edits): (1) the N2 lower bound counts 2D real squared terms (gamma_{{2D}}), the extra rounding is covered by the numerator inflation, and the gate margins are printed to show that no verdict depends on it (sec. 1); (2) eps(t) is stated to certify the eigen-residual in ball arithmetic but not the Hamiltonian assembly error (about 1e-11, harmless at these margins) (sec. 1).", '',
      '## 7. Files (box <internal>/n5run, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
files = [f'{R}/recert_v1/cw50_recert.json', f'{R}/bright_v1/cw51_bright.json', f'{R}/pi_v1/cw52_pi.json']
files += [info['states_file'] for k, info in br.items() if k != 'cpu_s_total']
for fo in ('onsite', 'filtered'):
    for fn in ('hamiltonian.npz', 'eigenbasis.npz', 'joining_source_S.npz', 'result.json'):
        files.append(f'{R}/{fo}_g1/{fn}')
files += [f'{B}/code/{x}' for x in ('cw32_mockmodel.py', 'cw42_dark_subspace_v2.py', 'cw50_n5_wall_recert.py', 'cw51_bright_input.py', 'cw52_pi_sector.py', 'cw53_write_n5_v1_1.py', 'cw54_write_n5_v1_1_v2.py')]
for p_ in files:
    L.append(f'| `{p_}` | `{md5(p_)}` |')
L.append('')
p = f'{R}/MOCK_n5_v1.1.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p))
