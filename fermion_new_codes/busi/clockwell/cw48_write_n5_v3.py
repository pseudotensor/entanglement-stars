"""cw48 (v3: repairs after the independent second read: OUTWARD rounding of every printed interval endpoint, nominal-clock
statement, scoped dark-code wording, full-H symmetry wording, second-read section; writes the final name MOCK_n5_v1.md).
cw47 was v2 of cw46: adds the computed split of each g = 1 Bwall into its discretization term and its sampled-integrand
term, so the reader sees what drives t*): write MOCK_n5_v1.md.wip from the six cw45 arm results (n5run/<form>_g<G>/result.json), their time files,
the plan v2 and the cw42 dark-subspace readout. Every number is emitted here.
Output: <internal>/MOCK_n5_v1.md.wip"""
import json, os, re, hashlib
R = '<internal>/n5run'; P5 = '<internal>/n5prep'; B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def f(x, s=6):
    return 'n/a' if x is None else f'{x:.{s}g}'

from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

def out(x, s, mode):
    """x rounded to s significant digits in the given direction (outward rounding of interval endpoints)"""
    if x == 0:
        return '0'
    d = Decimal(repr(float(x)))
    e = d.adjusted() - s + 1
    q = d.quantize(Decimal(1).scaleb(e), rounding=mode)
    return format(q.normalize(), 'g') if abs(q.adjusted()) < 6 else format(q, 'E')

def I(v, s=10):
    return 'undefined' if v is None else f'[{out(v[0], s, ROUND_FLOOR)}, {out(v[1], s, ROUND_CEILING)}]'

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)'), ('exit', r'Exit status: ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    return o

HALO = 'endpoint-copy halo: N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention, not a measurement)'
arms = [('onsite', '0'), ('onsite', '1'), ('onsite', '2'), ('filtered', '0'), ('filtered', '1'), ('filtered', '2')]
res = {a: json.load(open(f'{R}/{a[0]}_g{a[1]}/result.json')) for a in arms}
lab = lambda fo: 'on-site' if fo == 'onsite' else 'filtered (endpoint-copy halo)'
plan = f'{P5}/MOCK_n5_plan_v2.md'
L = ['# MOCK n = 5 v1: the n = 5 no-flow mock, both converter forms, g = 0, 1, 2 (#492)', '',
     f'Final file after the independent second read (sec. 11). Author: busi (analyst engineer). Run exactly as frozen in `{plan}` (md5 `{md5(plan)}`), with busi\'s own pipeline (busi/code/clockwell/cw45_n5_run.py, one process per arm; builder cw32 reproduces the 12 saved n = 3 states, plan sec. 1). Filtered rows carry the {HALO}.', '',
     'Gated readouts are for the normalized PURE input B = |01>_H at site 0; registered readouts are separate and ungated. No rate, conversion time, Page time, turnover or maximum is printed or implied.', '',
     'Clocks: t = 1, 2, 3, 4, 5; t = 4 and t = 5 are NOMINAL ballistic arrival clocks (site 4 and the wall at the lead maximal group velocity 1; a scheduling choice, not a strict arrival time). The certified Bwall, not those clocks, controls every wall claim.', '',
     '## 1. Gate (g = 1, per form)', '',
     '| form | t* (last clock with certified Bwall <= 0.1) | Bwall(t*) | verdict | reason |', '|---|---|---|---|---|']
for fo in ('onsite', 'filtered'):
    r = res[(fo, '1')]; g = r['gate']; ts = g.get('t_star')
    L.append(f"| {lab(fo)} | {f(ts)} | {f(r['wall']['prefix'].get(str(int(ts))) if ts else None, 8)} | **{g['verdict']}** | {g.get('reason', g.get('meaning', ''))} |")
vv = {fo: res[(fo, '1')]['gate'] for fo in ('onsite', 'filtered')}
allvoid = all(v['verdict'] == 'VOID' for v in vv.values())
L += ['', ("Both g = 1 gates are VOID (" + '; '.join(f"{lab(fo)}: {v.get('reason')}" for fo, v in vv.items()) + "): the certified wall bound exceeds 0.1 before the third sampled clock, so the wall-controlled window is too short for the gate; under the frozen rule nothing is concluded about transport, and the f_L, f_H values below are readouts only.") if allvoid else "See the table for each verdict and its reason.", '',
      'What drives t* (computed): Bwall(t) = (sampled-integrand sum) + t L / (4 x 64), the second term coming from the frozen 64 panels per unit. ' + '; '.join(f"{lab(fo)}, g = 1: at t = 3, Bwall = {f(res[(fo, '1')]['wall']['prefix']['3'], 6)} of which the panel term 3 L / 256 = {f(3 * res[(fo, '1')]['wall']['L'] / 256, 6)}" for fo in ('onsite', 'filtered')) + '. The panel count is frozen by the plan (inherited from the n = 3 protocol); a finer panel count would be a protocol change for the lead and both models, not made here.', '',
      '## 2. Bwall (pure B; Duhamel state-distance bound to the empty 4-site extension; not a probability)', '',
      '| form | g | C_join | H_bound (Schur) | L | Bwall(1) | Bwall(2) | Bwall(3) | Bwall(4) | Bwall(5) |', '|---|---|---|---|---|---|---|---|---|---|']
for a in arms:
    w = res[a]['wall']
    L.append(f"| {lab(a[0])} | {a[1]} | {f(w['C_join'], 8)} | {f(w['H_bound'], 8)} | {f(w['L'], 8)} | " + ' | '.join(f(w['prefix'][str(k)], 6) for k in range(1, 6)) + ' |')
L += ['', '## 3. Pure-input readouts (certified intervals; gated only at g = 1, and VOID there)', '',
      '| form | g | t | N2 enclosure | eps (state) | admitted | p_LL | <N_L> | <N_H> | f_L | f_H |', '|---|---|---|---|---|---|---|---|---|---|---|']
for a in arms:
    for c in res[a]['clocks']:
        L.append(f"| {lab(a[0])} | {a[1]} | {f(c['t'])} | {I(c['N2'], 16)} | {f(c['eps'], 3)} | {c['admitted']} | {I(c['p_LL'])} | {I(c['N_L'])} | {I(c['N_H'])} | {I(c['f_L'])} | {I(c['f_H'])} |")
L += ['', 'f_L = <N_L on sites 1-4> / <N_L>, f_H = <N_H on sites 1-4> / <N_H>, by interval division after certifying the divisor positive; "undefined" where the divisor is not certified positive (g = 0: no light).', '',
      '## 4. LL-band readout (static)', '', '| form | g | LL dimension | LL spectrum [min, max] | <B|H|B> | inside |', '|---|---|---|---|---|---|']
for a in arms:
    b = res[a]['LL_band']
    L.append(f"| {lab(a[0])} | {a[1]} | {b['dim']} | [{f(b['min'], 10)}, {f(b['max'], 10)}] | {f(b['heavy_input_energy'], 10)} | {b['inside']} |")
L += ['', '## 5. Ride-along, ungated: register decoupling and protected Cq at site 4', '',
      '| form | g | t | D_RK light-only whole lead | D_RK complete whole lead | allowance (+-) |', '|---|---|---|---|---|---|']
for a in arms:
    for x in res[a]['register_ungated']:
        L.append(f"| {lab(a[0])} | {a[1]} | {f(x['t'])} | {f(x['D_light'], 10)} | {f(x['D_all'], 10)} | {f(x['allowance'], 3)} |")
L += ['', 'Light-only D_RK and any light-only decoder are uninformative for this two-particle, light-vacuum (B, D) code under these Hamiltonians, at every size (MOCK_recovery_audit_v1.2 secs. 9, 11; no many-body claim); they are printed only as ride-along readouts.', '',
      '| form | g | Cq = S(sites 3-4) - S(site 4) on the protected state at t2 = 5 (B1 = site 4 protected from t1 = 4) | interval | protected residual | eps (protected) |', '|---|---|---|---|---|---|']
for a in arms:
    q = res[a]['Cq_ungated']
    L.append(f"| {lab(a[0])} | {a[1]} | {f(q['value'], 10)} | {I(q['interval'])} | {f(q['protected_residual'], 3)} | {f(q['eps_protected'], 3)} |")
dsub = json.load(open(f'{P5}/cw42_dark_subspace.json'))
L += ['', '## 6. Static dark-subspace readout (reviewer A A, from the plan; cw42)', '',
      'The last column is a check on the FULL H at g = 1 (converter included), computed on the assembled matrix, so the R symmetry of these Hamiltonians is shown, not assumed; the theorem itself needs only R for H0 and for the |Q| = 2 heavy factor.', '',
      '| source | dim W | n(2n-1) | heavy dimension | norm P_W B | norm P_W B\' | norm P_W D | max abs [Gamma(r), H] |', '|---|---|---|---|---|---|---|---|']
for x in dsub:
    L.append(f"| {x['label']} | {x['dim_W']} | {x['protected_dim']} | {x['heavy_dim']} | {f(x['P_W_B'], 3)} | {f(x['P_W_Bprime'], 3)} | {f(x['P_W_D'], 12)} | {f(x['comm_R_H_max'], 3)} |")
L += ['', '## 7. Assembly, certification and admission checks', '',
      '| form | g | dimension | nnz | max abs(route A - route B) | Hermiticity defect | e_H | eigen residual norm (upper) | all 5 pure states admitted | all 5 registered Gram admitted |', '|---|---|---|---|---|---|---|---|---|---|']
for a in arms:
    r = res[a]; s_ = r['assembly']
    L.append(f"| {lab(a[0])} | {a[1]} | {s_['dim']} | {s_['nnz']} | {f(s_['max_abs_routeA_minus_routeB'], 3)} | {f(s_['hermitian_defect'], 3)} | {f(s_['e_H'], 3)} | {f(r['eig']['residual_frob_upper'], 3)} | {all(c['admitted'] for c in r['clocks'])} | {all(c['registered_admitted'] for c in r['clocks'])} |")
L += ['', '## 8. Measured cost and storage (the plan\'s 179 CPU s was a projection)', '',
      '| form | g | CPU s (process) | user + sys s (time) | wall | peak RSS KiB | exit | bytes retained |', '|---|---|---|---|---|---|---|---|']
tc = 0.0; tb = 0
for a in arms:
    t = tparse(f'{R}/{a[0]}_g{a[1]}_time.txt'); r = res[a]; by = sum(r['bytes'].values()) + os.path.getsize(f'{R}/{a[0]}_g{a[1]}/result.json')
    tc += r['cpu_s_total']; tb += by
    L.append(f"| {lab(a[0])} | {a[1]} | {f(r['cpu_s_total'], 5)} | {t['user']} + {t['sys']} | {t['wall']} | {t['rss']} | {t['exit']} | {by} |")
L += ['', f"Total over six arms: {f(tc, 5)} CPU s (process, BLAS 4 threads), {tb} bytes retained. Caps held: ulimit -v 8000000, nice 10, 4 threads, box CPU only; no process near 2 CPU-h.", '',
      'Per-stage CPU (s):', '', '| form | g | ' + ' | '.join(c['stage'] for c in res[arms[0]]['cost']) + ' |', '|---|---|' + '---|' * len(res[arms[0]]['cost'])]
for a in arms:
    L.append(f"| {lab(a[0])} | {a[1]} | " + ' | '.join(f(c['cpu_s'], 4) for c in res[a]['cost']) + ' |')
L += ['', '## 9. For the second reader', '',
      'Inputs: no icloud file is read by the run. The model is fully specified in the plan (sec. 3) and in busi/code/clockwell/cw32_mockmodel.py (docstring: modes 8x + 4A + c; h[x,x] = m_A N_x beta, h[x,x+1] = -i (N_x + N_{x+1}) alpha / 4, alpha = sigma_x x sigma_x, beta = diag(1,1,-1,-1), m_H = binary64(3/5); converter K (exact integers, MOCK_recovery_audit_v1.2 sec. 11); filtered fields L[b,j,x] = delta_{xj}/2 + (-1)^b (delta_{x,j-1} + delta_{x,j+1})/4, F = sum_{j,k=-1..5} G_{j-k} (N_j + N_k)/2 O_jk, N_-1 = 1/2, N_5 = 1; basis |m1 m2> = c^dag_m1 c^dag_m2 |0>, m1 < m2, ordered by occupation bitmask). Saved per arm (md5 below): H, eigenbasis, pure and registered states at t = 1..5, joining source S, protected Hamiltonian, eigenbasis and state, LL spectrum, result.json.', '',
      'Gate re-derivation per form at g = 1: (i) rebuild H (or load hamiltonian.npz) and propagate B = |01>_H to the 320 midpoints (k + 1/2)/64, k = 0..319; (ii) f_mid = ||S phi|| / ||phi|| + C_join eps_n with S from joining_source_S.npz (or rebuilt: S = H_ext J - J H, 4 added empty sites, N = 1, halo copy for the filtered form), C_join = sqrt(max row sum of |S^dagger S|); (iii) Bwall(t) = sum over panels up to t of (f_mid/64 + L/(4 x 64^2)), L = C_join x max row sum of |H|; (iv) t* = last clock in {1,...,5} with Bwall <= 0.1; VOID if t* < 3. Criterion: the verdicts of sec. 1. The certified error terms (eps, e_H, residuals) are listed in sec. 7 and in result.json; they change Bwall only in the ~1e-10 place.', '',
      'Other readouts: p_LL, <N_L>, <N_H>, f_L, f_H from the normalized pure states (sec. 3 definitions); D_RK by the graded partial trace (sign -1 when the traced mode precedes the kept mode) with K = complement of the output; Cq from the protected state (monomials meeting site 4 and its complement deleted from t = 4 to 5).', '',
      '## 11. Independent second read and repairs', '',
      f"Second read: `<internal>/MOCK_n5_secondread_v1.md` (md5 `{md5('<internal>/MOCK_n5_secondread_v1.md')}`, copy of <internal>/MOCK_n5_secondread_v1.md), coordinator-run, different agent: all 8 items AGREE (dim W also confirmed by an exact rank modulo two primes; both g = 1 gates VOID). Repairs, regenerated by cw48_write_n5_v3.py (no hand edits): (1) every printed interval endpoint is rounded outward (lower endpoint down, upper up) instead of to nearest; (2) the nominal-ballistic-clock statement (top of file); (3) the dark-code sentence of sec. 5 scoped to this two-particle, light-vacuum code under these Hamiltonians; (4) sec. 6 states that the [Gamma(r), H] column is a full-H check.", '',
      "The second reader's observation, recomputed here: the sampled-integrand part of Bwall(3) alone is " + '; '.join(f"{lab(fo)}: {f(res[(fo, '1')]['wall']['prefix']['3'] - 3 * res[(fo, '1')]['wall']['L'] / 256, 4)} (below 0.1: {res[(fo, '1')]['wall']['prefix']['3'] - 3 * res[(fo, '1')]['wall']['L'] / 256 < 0.1})" for fo in ('onsite', 'filtered')) + "; so the frozen 64-panel term is what pushes Bwall(3) above 0.1 in both forms.", '',
      '## 10. Files (box <internal>/n5run, MD5SUMS there)', '', '| file | md5 |', '|---|---|']
for a in arms:
    for fn, m in res[a]['files'].items():
        L.append(f"| `{R}/{a[0]}_g{a[1]}/{fn}` | `{m}` |")
    L.append(f"| `{R}/{a[0]}_g{a[1]}/result.json` | `{md5(f'{R}/{a[0]}_g{a[1]}/result.json')}` |")
for fn in ('cw32_mockmodel.py', 'cw45_n5_run.py', 'cw46_write_n5.py', 'cw47_write_n5_v2.py', 'cw48_write_n5_v3.py'):
    L.append(f'| `{B}/code/{fn}` | `{md5(f"{B}/code/{fn}")}` |')
L.append('')
p = f'{R}/MOCK_n5_v1.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p))
