"""cw29 (v3 of cw23, for the cw27 run): write deliveries draft MOCK_recovery_audit_v1.md.wip from cw18 (admission), cw22 (known answers),
cw21 (certified SDP), cw06 (fixed receiver F_e, busi) and cw05 (saved D_RK). Every number is emitted here.
Output: <internal>/MOCK_recovery_audit_v1.md.wip"""
import json, hashlib, re, os
R = '<internal>/recovery'
B = '<internal>/clockwell'

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

adm = json.load(open(f'{R}/cw18_admission.json')); sdp = json.load(open(f'{R}/cw27_sdp.json'))
st = json.load(open(f'{R}/cw28_selftest_v4.json')); ind = json.load(open(f'{B}/cw06_independent.json')); ext = json.load(open(f'{B}/cw05_extract.json'))
HALO = 'endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement)'

def f(x, s=10):
    return 'n/a' if x is None else f'{x:.{s}g}'

def timeinfo(p):
    if not os.path.exists(p):
        return {}
    t = open(p).read(); out = {}
    for k, pat in (('user', r'User time \(seconds\): ([0-9.]+)'), ('sys', r'System time \(seconds\): ([0-9.]+)'),
                   ('wall', r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'), ('rss', r'Maximum resident set size \(kbytes\): ([0-9]+)')):
        m = re.search(pat, t); out[k] = m.group(1) if m else None
    return out

def cw19_partial_cpu():
    tot = 0.0; n = 0
    for line in open(f'{R}/cw19_stdout.txt'):
        m = re.search(r'cpu ([0-9.]+)s', line)
        if m:
            tot += float(m.group(1)); n += 1
    return tot, n

L = ['# MOCK recovery audit v1: channel-optimal decoders on the saved n = 3 mock states (#490)', '',
     'WORKING FILE (.wip). Not a final name; an independent second read follows. Author: busi (analyst engineer), busi inbox #490, following deliveries/lead_gaps/ReviewA_mock1x_v1.md item B1 and ReviewB_mock1x_v1.md item B. Inputs are icloud saved registered states, read in place and read-only; nothing was propagated.', '',
     f'Frozen scope: n = 3; g = 1 and the g = 0 control; clocks t = 1, 2, 4; both converter forms: on-site baseline and filtered control with the {HALO}. The registered state is (|0_R>|B> + |1_R>|D>)/sqrt 2 evolved, R an ordinary reference qubit. Output (i) whole-lead light modes (sites 1, 2; light components; the receiver); output (ii) all light modes including the core (transport diagnostic, not an expanded receiver). Every O sector (vacuum, one and two particles) is kept; no postselection, no branch renormalization. Null value of the entanglement fidelity: 1/4.', '',
     '## 1. Admission (before any division or optimization)', '',
     'Squared norm N2 of each saved raw state, exact (binary64 entries are dyadic rationals), with certified decimal enclosure; |N2 - 1| against the addendum normalized-state tolerance (1 + eps)^2 - 1 (eps = saved state error of that action) and against the run spec norm-drift threshold 2e-12; the only trace divisor used downstream is N2 (rho_RO = C C^dagger / N2; no block or branch divisor); the code Gram matrix G_rs = 2 <v_r, v_s> / N2 of the two evolved codeword branches and its certified Frobenius distance from I_2 against 2 (2 sqrt 2 eps + 2 eps^2).', '',
     '| arm | g | t | raw array (md5) | N2 certified enclosure | N2 > 0 | abs(N2 - 1) | addendum tolerance | within | within 2e-12 | Gram distance from I_2 (upper) | Gram tolerance | admitted |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for s in adm['states']:
    arm = 'filtered (endpoint-copy halo)' if s['arm'] == 'filtered' else 'on-site'
    L.append(f"| {arm} | {s['g']} | {f(s['t'])} | `{os.path.basename(s['array'])}` (`{s['array_md5']}`) | [{s['N2_lower'][:24]}, {s['N2_upper'][:24]}] | {s['N2_positive']} | {f(s['abs_N2_minus_1'], 4)} | {f(s['tolerance_addendum'], 4)} | {s['within_addendum']} | {s['within_runspec_2e_12']} | {f(s['gram']['frobenius_distance_upper'], 4)} | {f(s['gram']['tolerance'], 4)} | {s['admitted']} |")
L += ['', f"Admission result: {'every state admitted; optimization proceeded' if not adm['stop'] else 'FAILED: STOP (no optimization)'}.", '',
      '## 2. Known-answer tests of the SDP and certification (run first)', '',
      '| test | certified lower | certified upper | known value | pass |', '|---|---|---|---|---|']
for k, v in st.items():
    L.append(f"| {k} | {f(v['LB'], 12)} | {f(v['UB'], 12)} | {f(v['target'])} | {v['ok']} |")
L += ['', '## 3. Certified optimal entanglement fidelity per state', '',
      f'Gate on the physical-target bounds (saved-state bounds widened by 2 eps): lower > 1/2 PASS (a one-shot transfer witness only, not recovery, not return exclusion); upper <= 1/2 FAIL; otherwise UNRESOLVED. Rows for the filtered control carry the {HALO}.', '',
      '| arm | g | t | output | certified F* (saved state) [lower, upper] | physical target [lower, upper] | numerical primal / dual | gate | blocks k: O-weight | fixed receiver F_e (busi, sec. 4 of the clock-well file) | saved D_RK (same output, (ii): n/a) |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
gates = {}
for s in sdp['states']:
    arm = 'filtered (endpoint-copy halo)' if s['arm'] == 'filtered' else 'on-site'
    key = f"{s['arm']}/even_n3_g{s['g']}_v1"
    fe = [x for x in ind['cohorts'][key]['receiver'] if x['clock'] == s['t'] and x['choice'] == 'light' and not x['first']]
    P = ext['physical'][key]
    dr = [x for x in P['register'] if x['clock'] == s['t'] and x['role'] == 'light' and not x['first_collected']]
    for oname, o in s['outputs'].items():
        lab = 'i whole-lead light (receiver)' if oname.startswith('i_') else 'ii all light incl. core (diagnostic)'
        bw = '; '.join(f"{k}: {f(b['weight'], 6)}" for k, b in o['blocks'].items())
        fer = f"[{f(fe[0]['F_lower'])}, {f(fe[0]['F_upper'])}]" if (fe and oname.startswith('i_')) else 'n/a'
        drr = f"[{f(dr[0]['D']['lower'])}, {f(dr[0]['D']['upper'])}]" if (dr and oname.startswith('i_')) else 'n/a'
        L.append(f"| {arm} | {s['g']} | {f(s['t'])} | {lab} | [{f(o['F_lower_saved_state'], 12)}, {f(o['F_upper_saved_state'], 12)}] | [{f(o['F_lower_physical'], 12)}, {f(o['F_upper_physical'], 12)}] | {f(o['numerical_primal'], 12)} / {f(o['numerical_dual'], 12)} | {o['gate']} | {bw} | {fer} | {drr} |")
        gates[(s['arm'], s['g'], s['t'], oname)] = o
L += ['', '## 4. Per-state summary', '']
for (arm, g, t, oname), o in gates.items():
    if not oname.startswith('i_'):
        continue
    o2 = gates[(arm, g, t, 'ii_all_light_incl_core')]
    L.append(f"- {'filtered (endpoint-copy halo)' if arm == 'filtered' else 'on-site'}, g = {g}, t = {f(t)}: receiver F* in [{f(o['F_lower_physical'], 8)}, {f(o['F_upper_physical'], 8)}] -> {o['gate']}; transport diagnostic F* in [{f(o2['F_lower_physical'], 8)}, {f(o2['F_upper_physical'], 8)}] -> {o2['gate']}.")
allfail = all(o['gate'] == 'FAIL' for (a, g, t, on), o in gates.items() if on.startswith('i_'))
maxub = max(o['F_upper_physical'] for (a, g, t, on), o in gates.items() if on.startswith('i_'))
maxub2 = max(o['F_upper_physical'] for (a, g, t, on), o in gates.items() if not on.startswith('i_'))
L += ['', f"Overall: the receiver gate is {'FAIL at every sampled clock, both converter forms and both couplings' if allfail else 'not FAIL everywhere (see the table)'}; the largest certified physical upper bound over all receiver states is {f(maxub, 10)} and over all diagnostic states {f(maxub2, 10)}. In the reviewer A B1 wording, a FAIL at every clock licenses 'for every decoder' in place of 'declared receiver' in the recovery finding at this size (n = 3); it says nothing about larger chains.", '']
# costs
tt = timeinfo(f'{R}/cw27_time.txt'); t24 = timeinfo(f'{R}/cw24_time.txt'); t21 = timeinfo(f'{R}/cw21_time.txt'); p19, n19 = cw19_partial_cpu()
L += ['## 5. Cost (all busi runs for this audit, including the two stopped versions)', '',
      '| item | CPU s | wall | peak RSS KiB |', '|---|---|---|---|',
      f"| cw18 admission | {timeinfo(f'{R}/cw18_time.txt').get('user', 'see cw18_time.txt')} | | |",
      f"| cw19 (v1; full-space SDP; stopped by busi after {n19} of 24 solves to speed up assembly) | {f(p19, 6)} (sum of per-solve process CPU) | | not recorded |",
      f"| cw21 (v2; same full-space SDP; stopped by busi on its first g = 1 transport-diagnostic block because of memory and the 2 CPU-hour limit) | {t21.get('user')} user + {t21.get('sys')} sys | {t21.get('wall')} | {t21.get('rss')} |",
      f"| cw24 (v3; SDP on supp(rho_O), certified on the full space; superseded: its ball-arithmetic squares returned NaN on every rank-one block, so those blocks were reported UNCERTIFIED) | {t24.get('user')} user + {t24.get('sys')} sys | {t24.get('wall')} | {t24.get('rss')} |",
      f"| cw27 (v4 = cw24 with x*x squares; the reported results) | {f(sdp.get('cpu_s_total'), 6)} (process CPU) / {tt.get('user')} user + {tt.get('sys')} sys | {tt.get('wall')} | {tt.get('rss')} |",
      f"| total CPU of all versions | {f(p19 + sum(float(x.get(k) or 0) for x in (t21, t24, tt) for k in ('user', 'sys')), 6)} | | |", '',
      'The solves that cw19 and cw21 completed, and every cw24 block that cw24 certified, agree with cw27 (all stdout files are kept).',
      '']
L += ['Per-solve CPU (process time, all threads):', '', '| arm | g | t | output | CPU s | wall s |', '|---|---|---|---|---|---|']
for (arm, g, t, oname), o in gates.items():
    L.append(f"| {arm} | {g} | {f(t)} | {oname} | {f(o['cpu_s'], 5)} | {f(o['wall_s'], 5)} |")
L += ['', 'Box CPU only (compute-host), nice 10, BLAS and Rayon threads 4, no GPU. Packages: Clarabel and python-flint installed in a private venv <internal>/venv_cw (system site packages inherited; the shared qi2 environment untouched).', '',
      '## 6. For the second reader', '',
      'Inputs (icloud saved outputs on compute-host, read-only): for each state the registered primary action raw vector (npz, shape 2 x 276, R first) listed with its md5 in sec. 1, and the basis row keys of the same physical job report (row_keys of any `coherent_conversion` row: (register index, occupation bitmask), modes 0-23, mode m = 8x + 4A + c, A = 1 light). Reports: ' + '; '.join(f"`{s['report']}` md5 `{s['report_md5']}`" for s in {x['cohort']: x for x in adm['states']}.values()) + '.', '',
      'Exact formulation (cw27). For output mode set O (i: {12,13,14,15,20,21,22,23}; ii: {4,...,7} added), write each basis state |m1 m2> (m1 < m2) as |a>_O |t>_K with sign -1 when m1 is outside O and m2 inside (graded reordering, O first). C[(r, a), t] = sign * psi[r, (m1, m2)]; rho_RO = C C^dagger / N2. rho_RO is block diagonal in k = |a| in {0, 1, 2}. For each block: dual SDP  min Tr sigma_k  s.t.  I_2 x sigma_k - rho_k >= 0 (sigma_k Hermitian d_k x d_k); primal SDP  max Tr[rho_k Z_k]  s.t.  Z_k >= 0, Tr_R Z_k = I. F* = (1/2) sum_k (optimum). (Koenig-Renner-Schaffner: this is the optimal entanglement fidelity over all CPTP decoders O -> qubit; block decomposition is exact because rho is block diagonal and pinching preserves dual feasibility and trace.) Null check: rho = I/2 x rho_O gives 1/4. Numerical solve: each block is compressed to Q = orthonormal basis of supp(rho_O^(k)) (left singular vectors of [C_0 | C_1] with singular value > 1e-13 of the largest); this is exact (compressing sigma and Z to the support preserves feasibility and the objective); the solution is lifted back (sigma -> Q sigma Q^dagger, Z -> (I x Q) Z (I x Q)^dagger) and certified on the full space.', '',
      f"Certification. Upper: sigma' = sigma_num + delta I; I x sigma' - rho = B B^dagger + eta I + E with dyadic B, eta = {sdp['eta']}; ball arithmetic (python-flint acb, 256 bits) shows ||E||_F < eta, hence I x sigma' - rho >= 0 exactly, and UB = (1/2) sum_k Tr sigma'_k. Lower: Z_k = A A^dagger with dyadic A (exactly PSD); s_k = certified upper bound of lambda_max(Tr_R Z_k) (s I - T = B B^dagger + eta I + E with ||E||_F < eta in ball arithmetic; Gershgorin as fallback); Z_k / s_k has Tr_R <= I (completion to a channel adds a nonnegative term); LB = (1/2) sum_k ||A^dagger C_k||_F^2 / (N2 s_k), evaluated in ball arithmetic. Physical-target bounds widen by 2 eps (normalized-vector distance bound; F is a POVM-element expectation). A second reader can re-derive F* with any SDP solver from the inputs above; the gate needs only the comparison with 1/2.", '',
      '## 7. Files (box <internal>/recovery, MD5SUMS there; code busi/code/clockwell/)', '', '| file | md5 |', '|---|---|']
for fn in ('cw18_admission.json', 'cw28_selftest_v4.json', 'cw27_sdp.json', 'cw19_stdout.txt', 'cw21_stdout.txt'):
    L.append(f'| `{R}/{fn}` | `{md5(f"{R}/{fn}")}` |')
for fn in ('cw18_recovery_admission.py', 'cw19_recovery_sdp.py', 'cw20_sdp_selftest.py', 'cw21_recovery_sdp_v2.py', 'cw22_sdp_selftest_v2.py', 'cw23_write_audit.py', 'cw24_recovery_sdp_v3.py', 'cw25_sdp_selftest_v3.py', 'cw26_write_audit_v2.py', 'cw27_recovery_sdp_v4.py', 'cw28_sdp_selftest_v4.py', 'cw29_write_audit_v3.py'):
    L.append(f'| `{B}/code/{fn}` | `{md5(f"{B}/code/{fn}")}` |')
L.append('')
p = f'{R}/MOCK_recovery_audit_v1.md.wip'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p))
