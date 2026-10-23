"""cw38: write MOCK_n5_plan_v1.md (#491 B): regeneration gate result (cw33), the #138 sec. 8 cost report with measured
numbers (cw37 icloud jobs, the recovery-audit runs, the second read's own run, cw36 n = 5 pilot), the frozen collection
plan, every definition and divisor, and the run GATE verbatim. Every number is emitted here from saved JSON/text.
The lead dispersion's maximal group velocity is computed here (grid) for the free-front arrival clocks.
Output: <internal>/MOCK_n5_plan_v1.md"""
import json, re, os, hashlib, math
import numpy as np
P5 = '<internal>/n5prep'; RC = '<internal>/recovery'; B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def f(x, s=6):
    return 'n/a' if x is None else (x if isinstance(x, str) else f'{x:.{s}g}')

def tparse(p):
    t = open(p).read(); o = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)')):
        m = re.search(pat, t); o[k] = m.group(1) if m else None
    if o['user'] is None:
        m = re.search(r'([0-9.]+) s wall ([0-9.]+) s user ([0-9]+) KiB', t)
        if m:
            o.update(wall=m.group(1) + ' s', user=m.group(2), rss=m.group(3))
    return o

reg = json.load(open(f'{P5}/cw33_regeneration.json')); pil = json.load(open(f'{P5}/cw36_pilot.json')); ic = json.load(open(f'{P5}/cw37_icloud_costs.json'))
# lead dispersion: h(k) = alpha sin k + beta m (N = 1, a = 1); bands +-sqrt(sin^2 k + m^2); group velocity d/dk
k = np.linspace(0, 2 * np.pi, 200001)
vmax = {}
for name, m in (('light (m = 0)', 0.0), ('heavy (m = 3/5)', 0.6)):
    e = np.sqrt(np.sin(k) ** 2 + m ** 2)
    vmax[name] = float(np.max(np.abs(np.gradient(e, k))))
v = max(vmax.values()); t_site4 = 4 / v; t_wall = 5 / v
clocks = sorted(set([1.0, 2.0, 3.0, 4.0, round(t_site4, 6), round(t_wall, 6)]))

L = ['# MOCK n = 5 plan v1 (#491 B): regeneration gate, #138 sec. 8 cost report, frozen collection plan, run gate', '',
     'PREPARED, NOT RUN. Author: busi (analyst engineer), busi inbox #491, after ReviewA_mock2_v1.md (3) and ReviewB_mock2_v1.md (3). The n = 5 interacting run starts only after the lead posts the joint OK. Every number below is emitted by busi/code/clockwell/cw38_write_plan.py from the saved products named in sec. 7.', '',
     '## 1. Regeneration gate (reviewer A (iii)), RUN as part of the preparation', '',
     "busi's own builder (cw32_mockmodel.py; no icloud code imported) assembles the n = 3 Hamiltonian of both forms and propagates the frozen preparations by its own eigendecomposition; cw33 compares with icloud's saved records. Criteria: same basis order; ||psi_busi - psi_saved|| <= eps_saved (the saved action's normalized-state error radius, the addendum tolerance); p_LL (pure input), D_RK (registered, light-only and complete whole-lead output) and the protected conditional entropy Cq (t1 = 1 to t2 = 2, protection deletes every monomial meeting site 2 and its complement) inside the saved intervals.", '',
     '| form | g | max abs(H_busi - H_saved) | t | state distance | eps_saved | state ok | p_LL busi | p_LL in saved interval | D_RK light busi (in) | D_RK all busi (in) |', '|---|---|---|---|---|---|---|---|---|---|---|']
for c in reg['cohorts']:
    form = 'on-site' if c['arm'] == 'baseline' else 'filtered (endpoint-copy halo)'
    for x in c['clocks']:
        L.append(f"| {form} | {c['g']} | {f(c['max_abs_dH'], 3)} | {f(x['t'])} | {f(x['state_distance'], 3)} | {f(x['eps_saved'], 3)} | {x['state_ok']} | {f(x['pLL_busi'], 12)} | {x['pLL_inside']} | {f(x['D_RK']['light']['busi'], 12)} ({x['D_RK']['light']['inside']}) | {f(x['D_RK']['all']['busi'], 12)} ({x['D_RK']['all']['inside']}) |")
L += ['', '| form | g | Cq busi | Cq saved interval | inside | cohort reproduced |', '|---|---|---|---|---|---|']
for c in reg['cohorts']:
    form = 'on-site' if c['arm'] == 'baseline' else 'filtered (endpoint-copy halo)'
    L.append(f"| {form} | {c['g']} | {f(c['Cq_busi'], 12)} | [{f(c['Cq_saved'][0], 12)}, {f(c['Cq_saved'][1], 12)}] | {c['Cq_inside']} | {c['admitted']} |")
L += ['', f"**Regeneration gate: {'PASS: all 12 saved n = 3 states and their p_LL, D_RK and Cq reproduced' if not reg['stop'] else 'FAIL: STOP'}** (cw33: CPU {f(reg['cpu_s'], 4)} s, peak RSS {reg['peak_rss_kib']} KiB).", '',
      '## 2. #138 sec. 8 cost report (measured numbers only)', '',
      "### 2.1 icloud's recorded n = 2, 3 jobs (termination rows; one thread each)", '',
      '| arm | job | wall s | CPU s | peak RSS KiB | phases | CPU by family (s): coherence / entropy / register / actions / wall / protected+collection / assembly+static |', '|---|---|---|---|---|---|---|']
for j in ic['jobs']:
    c = j['per_category_cpu_s']
    L.append(f"| {j['arm']} | {j['job']} | {f(j['wall_s'])} | {f(j['cpu_s'])} | {j['peak_rss_kib']} | {j['phases']} | {f(c.get('coherence', 0), 4)} / {f(c.get('entropy', 0), 4)} / {f(c.get('register', 0), 4)} / {f(c.get('actions', 0), 4)} / {f(c.get('wall', 0), 4)} / {f(c.get('protected_collection', 0), 4)} / {f(c.get('assembly_and_static', 0), 4)} |")
L += ['', 'Per-clock CPU (summed phase CPU of the phases carrying that clock) for the two n = 3, g = 1 jobs:', '', '| clock | on-site CPU s | filtered CPU s |', '|---|---|---|']
a = [j for j in ic['jobs'] if j['job'] == 'even_n3_g1_v1' and j['arm'] == 'on-site'][0]['per_clock_cpu_s']
b = [j for j in ic['jobs'] if j['job'] == 'even_n3_g1_v1' and j['arm'] != 'on-site'][0]['per_clock_cpu_s']
for clk in sorted(set(list(a) + list(b)), key=float):
    if float(clk) <= 4.0 and abs(float(clk) * 4 - round(float(clk) * 4)) < 1e-9:
        L.append(f"| {f(float(clk))} | {f(a.get(clk, 0), 4)} | {f(b.get(clk, 0), 4)} |")
L += ['', 'icloud readout jobs (time files): ' + '; '.join(f"{x['dir'].replace('MOCK_', '').replace('_v1', '')}/{x['job']}: {x['wall']} wall, {x['user_s']} s user, {x['rss_kib']} KiB" for x in ic.get('readout_jobs', []) if 'n3' in x['job']) + '.', '']
# audit costs
au = []
p19 = sum(float(m.group(1)) for m in re.finditer(r'cpu ([0-9.]+)s', open(f'{RC}/cw19_stdout.txt').read()))
t21 = tparse(f'{RC}/cw21_time.txt'); t24 = tparse(f'{RC}/cw24_time.txt'); t27 = tparse(f'{RC}/cw27_time.txt')
sr = open('<internal>/MOCK_recovery_secondread_v1.md').read()
msr = re.search(r'CPU ([0-9.]+) s \(process\), wall ([0-9.]+) s, peak RSS ([0-9]+) KiB', sr)
L += ['### 2.2 Recovery audit (#490) runs, itemised', '', '| run | CPU s | wall | peak RSS KiB | note |', '|---|---|---|---|---|',
      f"| cw19 (full-space SDP, v1) | {f(p19, 6)} | | not recorded | stopped by busi after 5 of 24 solves |",
      f"| cw21 (full-space SDP, v2) | {t21['user']} user + {t21['sys']} sys | {t21['wall']} | {t21['rss']} | stopped by busi: 62 GB RSS on the first g = 1 diagnostic block, approaching the 2 CPU-h cap |",
      f"| cw24 (support-compressed, v3) | {t24['user']} user + {t24['sys']} sys | {t24['wall']} | {t24['rss']} | superseded (NaN in a ball square) |",
      f"| cw27 (v4, reported) | {t27['user']} user + {t27['sys']} sys | {t27['wall']} | {t27['rss']} | final |",
      f"| second read (its own report) | {msr.group(1) if msr else 'n/a'} | {msr.group(2) + ' s' if msr else 'n/a'} | {msr.group(3) if msr else 'n/a'} | independent route |", '',
      '### 2.3 n = 5 dimensions (busi builder) and nnz', '']
d = pil['dims']
L += ['| quantity | value |', '|---|---|'] + [f'| {k} | {v} |' for k, v in d.items()]
for form, A in pil['arms'].items():
    L.append(f"| nnz of H, {form}, g = 1 | {A['nnz_H']} |")
    L.append(f"| converter monomials, {form} | {A['quartic_monomials']} |")
L += ['', 'All dense matrices stay under the 2048 cap except the n + 4 = 9 joining-source extension (used only as rectangular columns on the embedded short basis, never diagonalised; its two-particle dimension is listed).', '',
      "### 2.4 Measured pilot (cw36: one g = 1 arm per form, busi's pipeline, no physical value printed or kept)", '',
      '| form | stage | CPU s | wall s | peak RSS KiB (process so far) |', '|---|---|---|---|---|']
tot = {}
for form, A in pil['arms'].items():
    for s_ in A['stages']:
        L.append(f"| {form} | {s_['stage']} | {f(s_['cpu_s'], 4)} | {f(s_['wall_s'], 4)} | {s_['peak_rss_kib']} |")
    tot[form] = {s_['stage']: s_['cpu_s'] for s_ in A['stages']}
proj = {}
for form, st_ in tot.items():
    proj[form] = st_['assembly'] + st_['eigendecomposition'] + st_['certified_residual_ball_128bit'] + st_['actions_5_clocks_pure_and_registered'] \
        + st_['extension_assembly_n_plus_4'] + st_['joining_source_and_320_midpoints'] + len(clocks) * st_['readouts_one_clock'] \
        + st_['protected_assembly_and_eigendecomposition'] + st_['LL_band_spectrum']
six = 3 * proj['onsite'] + 3 * proj['filtered']
L += ['', f"### 2.5 Projection (from the measured pilot; readouts priced at {len(clocks)} clocks)", '',
      f"Per arm: on-site {f(proj['onsite'], 4)} CPU s, filtered {f(proj['filtered'], 4)} CPU s. Six arms (two forms x g = 0, 1, 2): {f(six, 4)} CPU s. Peak RSS measured in the pilot: {pil['peak_rss_kib']} KiB. This projection prices busi's certified pipeline as piloted (ball-arithmetic residual certificate; readouts with continuity allowances); it is a measured-stage sum, not a promise, and the measured totals are appended after the run.", '',
      '### 2.6 Threads and hard caps', '',
      'One process per arm. nice -n 10; ulimit -v 8000000 (8 GB); OMP/OPENBLAS/MKL threads 4 (at most 8); box CPU compute-host only, no GPU; each process stops at 2 CPU-hours (timeout and a CPU-time check between stages); on any cap the partial outputs are kept and the run is reported, not retried. The full-space SDP route that reached 62 GB is not used.', '',
      '## 3. Collection plan (frozen before any evolution)', '',
      'Model: the n = 3 programme at n = 5 sites (modes 8x + 4A + c, 40 modes; complete N = 2 sector, dimension listed in 2.3; registered R x fermion dimension twice that); a = 1; m_H = 3/5 (represented binary64), m_L = 0; no flow.',
      'Well: N = (1/2, 1, 1, 1, 1). Converter forms: on-site N_x W_x; filtered with the endpoint-copy halo N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention, not a measurement), same G_0, G_1, G_2, stencil and allocation as at n = 3. Couplings g = 0, 1, 2 (g = 1 gated; g = 0 null and g = 2 sensitivity ungated). Source site 0 = core; lead = sites 1-4.',
      'Code: the n = 3 code unchanged: pure input B = |01>_H at site 0; registered (|0_R>B + |1_R>D)/sqrt 2, D = (|03>_H - |12>_H)/sqrt 2.',
      f"Clocks: t = 1, 2, 3, 4 plus the free-front arrival clocks from the lead's maximal group velocity v_max = max over k and bands of |d/dk sqrt(sin^2 k + m^2)| (N = 1, a = 1): " + ', '.join(f'{k_}: {f(v_, 8)}' for k_, v_ in vmax.items()) + f"; v_max = {f(v, 8)}. Arrival at site 4 (4 bonds from the source): t = 4 / v_max = {f(t_site4, 8)}; at the wall (first added site, 5 bonds): t = 5 / v_max = {f(t_wall, 8)}. (The slower core bond only delays the true front; these are lower bounds of the free-front arrival.) Sampled clocks: {', '.join(f(c_) for c_ in clocks)}.",
      f"Collection (ride-along, ungated): B1 = site 4 collected at t1 = {f(t_site4)}, B2 = site 3 collected at t2 = {f(t_wall)}; protected segment [t1, t2] deletes every monomial meeting a collected block and its complement (monomial granularity, filtered quartics included).", '',
      '## 4. Definitions and divisors', '',
      '- N2(t) = ||psi_hat(t)||^2 of the computed raw state: the only trace divisor of every state readout (rho = psi psi^dagger / N2). Certified positive lower bound and |N2 - 1| within (1 + eps)^2 - 1 printed before any division; else the state is not admitted.',
      '- State error eps(t): with H V = V Lambda + Rres (computed eigendecomposition), c = V^dagger psi0, phi(t) = V exp(-i Lambda t) c: ||phi(t) - psi(t)|| <= ||V c - psi0|| + t ||Rres||_F ||c|| + t e_H, all norms in ball arithmetic; e_H = ||H_binary64 - H_target||_F with the exact rational target (m = 3/5, exact G). Normalized-state error <= 2 eps / sqrt(lower N2).',
      '- p_LL(t) = <P_LL> / N2, P_LL the projector on states with both particles light; interval +- 2 eps_n.',
      '- <N_L>(t), <N_H>(t): light / heavy number expectations (divided by N2); f_L(t) = <N_L on sites 1-4> / <N_L>(t), f_H(t) = <N_H on sites 1-4> / <N_H>(t). Divisors <N_L>, <N_H>: certified lower bound > 0 printed before division (at g = 0, <N_L> = 0 and f_L prints "undefined"); numerator and divisor intervals each +- 2 * 2 eps_n (||N|| <= 2), the quotient by interval division.',
      f"- Bwall(t): Duhamel bound of the distance between the short-chain state and the state in the uniform empty extension by 4 added sites (all added clocks 1; filtered: halo copy at both ends of the extended chain). S = H_ext J - J H_short (J = embedding; exact joining source, rectangular); f(s) = ||S psi_hat(s)|| / sqrt(N2) + C_join eps_n(s), C_join^2 = R(S^dagger S) (absolute-row-sum Schur bound); L = C_join R(H_short); Bwall(t) = sum over 64 equal panels per unit on [0, t] of (d f_mid + L d^2 / 4), d = 1/64. Divisor: sqrt(N2) only. A state-distance bound, not a probability.",
      '- LL-band readout (static): the spectrum of P_LL H P_LL (LL sector, dimension listed in 2.3) printed with its minimum and maximum, beside the heavy input energy <B|H|B> and whether it lies inside [min, max]; no divisor.',
      '- t* = the last sampled clock with certified Bwall(t*) <= 0.1 (upper end).',
      '- Ride-along, ungated: Cq at site 4 (protected conditional entropy S(A2; t2) - S(A1; t2), A1 = B1, A2 = B1 + B2, on the protected state; divisor N2) and D_RK (register, light-only and complete whole-lead output; no divisor beyond N2). Entropy intervals by the Audenaert continuity bound on the allowed subsystem dimension; D_RK +- 3 eps_n.',
      "- Not added: the light-only decoder audit (MOCK_recovery_audit_v1.1 sec. 9: by a symmetry of the vertex the dark codeword never makes light, and F* = 1/4 + (1 - p0)/4 <= 1/2 for every light-only output at every n, so it would FAIL identically).", '',
      '## 5. GATE (verbatim from #491)', '',
      'Per form at g = 1, t* = last clock with certified Bwall <= 0.1; VOID if t* < 3 or p_LL(t*-1) < 0.1 or <N_L>(t*) < 0.05; PASS if lower f_L(t*) >= 1/2 and lower f_L > upper f_H; FAIL if upper f_L < 1/2 with p_LL(t*-1) >= 0.1; else UNRESOLVED; certify every actual-state norm and divisor positive before division; g = 2 and g = 0 ungated.', '',
      f"Clock reading of 't* - 1': the sampled clock preceding t* in the list {', '.join(f(c_) for c_ in clocks)}.", '',
      '## 6. Run procedure after the joint OK', '',
      'Six processes (form x g), each: assemble and compare with an independent second assembly route; admit H; eigendecompose and certify the residual; propagate the pure and registered states to every sampled clock and to the 64-per-unit Bwall midpoints; admission of every state (N2, Gram); readouts of sec. 4; protected segment for Cq; save every state, matrix, eigenbasis and readout with md5 under <internal>/n5run; a second agent reads before anything is filed. Odd (N = 1) control: one common run per form, ungated.', '',
      '## 7. Files (box <internal>/n5prep, MD5SUMS there; code busi/code/clockwell/)', '', '| file | md5 |', '|---|---|']
for pth in (f'{P5}/cw33_regeneration.json', f'{P5}/cw36_pilot.json', f'{P5}/cw37_icloud_costs.json', f'{P5}/cw34_dark.json', f'{P5}/cw35_dark_symmetry.json',
            f'{B}/code/cw32_mockmodel.py', f'{B}/code/cw33_regeneration_gate.py', f'{B}/code/cw34_dark_codeword.py', f'{B}/code/cw35_dark_symmetry.py',
            f'{B}/code/cw36_n5_pilot.py', f'{B}/code/cw37_icloud_costs.py', f'{B}/code/cw38_write_plan.py'):
    L.append(f'| `{pth}` | `{md5(pth)}` |')
L.append('')
p = f'{P5}/MOCK_n5_plan_v1.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p), 'six-arm projection CPU s', six, 'clocks', clocks)
