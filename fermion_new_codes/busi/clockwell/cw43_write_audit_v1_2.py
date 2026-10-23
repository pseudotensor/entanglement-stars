"""cw43: writes MOCK_recovery_audit_v1.2.md (#492): v1.1 regenerated plus sec. 11 (reviewer A gaps a, b, c, d, f and
reviewer B's wording points). v1.1 (cw39) is kept. cw39 wrote MOCK_recovery_audit_v1.1.md (#491 A): the v1 content regenerated, the one 'exactly' wording fixed,
plus sec. 9 (v1.1 additions: final v1 md5 and cw27_sdp.json identity, g = 0 array identity across forms, the
dark-codeword theorem with its checks). v1 (cw31) is kept unchanged. cw31 was
(v4 was cw30, for the cw27 run; adds the script-computed agreement of the superseded runs and the cw18 time parse): write deliveries draft MOCK_recovery_audit_v1.md.wip from cw18 (admission), cw22 (known answers),
cw21 (certified SDP), cw06 (fixed receiver F_e, busi) and cw05 (saved D_RK). Every number is emitted here.
Output: <internal>/MOCK_recovery_audit_v1.md"""
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

L = ['# MOCK recovery audit v1.2: channel-optimal decoders on the saved n = 3 mock states (#490; v1.1 per #491; v1.2 per #492)', '',
     'Version 1.2 (sec. 9 lists the v1.1 changes, sec. 11 the v1.2 changes; v1 and v1.1 are kept). Final delivery after the coordinator-run independent second read (sec. 7). Author: busi (analyst engineer), busi inbox #490, following deliveries/lead_gaps/ReviewA_mock1x_v1.md item B1 and ReviewB_mock1x_v1.md item B. Inputs are icloud saved registered states, read in place and read-only; nothing was propagated.', '',
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
vac = [o['blocks']['0'] for (a, g, t, on), o in gates.items() if on.startswith('i_')]
vac_txt = f"between [{f(min(b['cert_LB'] for b in vac), 12)}, {f(max(b['cert_UB'] for b in vac), 12)}] (certified block bounds; within 1e-15 of 1/4 in the second read; exactly 1/4 by the theorem of sec. 9)"
allfail = all(o['gate'] == 'FAIL' for (a, g, t, on), o in gates.items() if on.startswith('i_'))
maxub = max(o['F_upper_physical'] for (a, g, t, on), o in gates.items() if on.startswith('i_'))
maxub2 = max(o['F_upper_physical'] for (a, g, t, on), o in gates.items() if not on.startswith('i_'))
L += ['', f"Overall: the receiver gate is {'FAIL at every sampled clock, both converter forms and both couplings' if allfail else 'not FAIL everywhere (see the table)'}; the largest certified physical upper bound over all receiver states is {f(maxub, 10)} and over all diagnostic states {f(maxub2, 10)}. In the reviewer A B1 wording, a FAIL at every clock licenses 'for every decoder' in place of 'declared receiver' in the recovery finding, with this scope only: decoders acting on the whole-lead LIGHT modes, at the sampled clocks t = 1, 2, 4, at n = 3. It is not a statement about decoders acting on the full heavy-plus-light lead output (not tested by this audit), about other clocks, or about larger chains. The empty-output (vacuum, k = 0) sector contributes " + vac_txt + " to F* in every receiver state, which is why the FAILs are clear-cut.", '']
# costs
cmp_rows = []
for ver in ('cw19', 'cw21', 'cw24'):
    pth = f'{R}/{ver}_sdp.json'
    if not os.path.exists(pth):
        continue
    old_ = json.load(open(pth)); ncmp = 0; dmax = 0.0
    for so in old_['states']:
        sn = [x for x in sdp['states'] if x['arm'] == so['arm'] and x['g'] == so['g'] and x['t'] == so['t']][0]
        for on, oo in so['outputs'].items():
            for kk, bo in oo['blocks'].items():
                if bo.get('psd_certified'):
                    bn = sn['outputs'][on]['blocks'][kk]
                    dmax = max(dmax, abs(bo['num_dual_half'] - bn['num_dual_half']), abs(bo['num_primal_half'] - bn['num_primal_half'])); ncmp += 1
    cmp_rows.append(f'{ver}: {ncmp} certified blocks, max |numerical optimum difference| {f(dmax, 3)}')
agree_txt = 'Agreement of the superseded runs with cw27 on every block they certified (numerical primal and dual optima): ' + '; '.join(cmp_rows) + '. All stdout files are kept.'
tt = timeinfo(f'{R}/cw27_time.txt'); t24 = timeinfo(f'{R}/cw24_time.txt'); t21 = timeinfo(f'{R}/cw21_time.txt'); p19, n19 = cw19_partial_cpu()
L += ['## 5. Cost (all busi runs for this audit, including the two stopped versions)', '',
      '| item | CPU s | wall | peak RSS KiB |', '|---|---|---|---|',
      f"| cw18 admission | {re.search(r'([0-9.]+) s user', open(f'{R}/cw18_time.txt').read()).group(1)} user | {re.search(r'([0-9.]+) s wall', open(f'{R}/cw18_time.txt').read()).group(1)} s | {re.search(r'([0-9]+) KiB', open(f'{R}/cw18_time.txt').read()).group(1)} |",
      f"| cw19 (v1; full-space SDP; stopped by busi after {n19} of 24 solves to speed up assembly) | {f(p19, 6)} (sum of per-solve process CPU) | | not recorded |",
      f"| cw21 (v2; same full-space SDP; stopped by busi on its first g = 1 transport-diagnostic block because of memory and the 2 CPU-hour limit) | {t21.get('user')} user + {t21.get('sys')} sys | {t21.get('wall')} | {t21.get('rss')} |",
      f"| cw24 (v3; SDP on supp(rho_O), certified on the full space; superseded: its ball-arithmetic squares returned NaN on every rank-one block, so those blocks were reported UNCERTIFIED) | {t24.get('user')} user + {t24.get('sys')} sys | {t24.get('wall')} | {t24.get('rss')} |",
      f"| cw27 (v4 = cw24 with x*x squares; the reported results) | {f(sdp.get('cpu_s_total'), 6)} (process CPU) / {tt.get('user')} user + {tt.get('sys')} sys | {tt.get('wall')} | {tt.get('rss')} |",
      f"| total CPU of all versions | {f(p19 + sum(float(x.get(k) or 0) for x in (t21, t24, tt) for k in ('user', 'sys')), 6)} | | |", '',
      agree_txt,
      '']
L += ['Per-solve CPU (process time, all threads):', '', '| arm | g | t | output | CPU s | wall s |', '|---|---|---|---|---|---|']
for (arm, g, t, oname), o in gates.items():
    L.append(f"| {arm} | {g} | {f(t)} | {oname} | {f(o['cpu_s'], 5)} | {f(o['wall_s'], 5)} |")
L += ['', 'Box CPU only (compute-host), nice 10, BLAS and Rayon threads 4, no GPU. Packages: Clarabel and python-flint installed in a private venv <internal>/venv_cw (system site packages inherited; the shared qi2 environment untouched).', '',
      '## 6. For the second reader', '',
      'Inputs (icloud saved outputs on compute-host, read-only): for each state the registered primary action raw vector (npz, shape 2 x 276, R first) listed with its md5 in sec. 1, and the basis row keys of the same physical job report (row_keys of any `coherent_conversion` row: (register index, occupation bitmask), modes 0-23, mode m = 8x + 4A + c, A = 1 light). Reports: ' + '; '.join(f"`{s['report']}` md5 `{s['report_md5']}`" for s in {x['cohort']: x for x in adm['states']}.values()) + '.', '',
      'Exact formulation (cw27). For output mode set O (i: {12,13,14,15,20,21,22,23}; ii: {4,...,7} added), write each basis state |m1 m2> (m1 < m2) as |a>_O |t>_K with sign -1 when m1 is outside O and m2 inside (graded reordering, O first). C[(r, a), t] = sign * psi[r, (m1, m2)]; rho_RO = C C^dagger / N2. rho_RO is block diagonal in k = |a| in {0, 1, 2}. For each block: dual SDP  min Tr sigma_k  s.t.  I_2 x sigma_k - rho_k >= 0 (sigma_k Hermitian d_k x d_k); primal SDP  max Tr[rho_k Z_k]  s.t.  Z_k >= 0, Tr_R Z_k = I. F* = (1/2) sum_k (optimum). (Koenig-Renner-Schaffner: this is the optimal entanglement fidelity over all CPTP decoders O -> qubit; block decomposition is exact because rho is block diagonal and pinching preserves dual feasibility and trace.) Null check: rho = I/2 x rho_O gives 1/4. Numerical solve: each block is compressed to Q = orthonormal basis of supp(rho_O^(k)) (left singular vectors of [C_0 | C_1] with singular value > 1e-13 of the largest); compression to the exact support would preserve feasibility and the objective, but the numerical support uses a 1e-13 singular-value cutoff, so the compressed solve is only a numerical candidate; the reported bounds do not rely on the compression because the solution is lifted back (sigma -> Q sigma Q^dagger, Z -> (I x Q) Z (I x Q)^dagger) and certified on the full space (the upper certificate absorbs any discarded component through delta, the lower certificate is exactly PSD on the full space).', '',
      f"Certification. Upper: sigma' = sigma_num + delta I; I x sigma' - rho = B B^dagger + eta I + E with dyadic B, eta = {sdp['eta']}; ball arithmetic (python-flint acb, 256 bits) shows ||E||_F < eta, hence I x sigma' - rho >= 0 exactly, and UB = (1/2) sum_k Tr sigma'_k. Lower: Z_k = A A^dagger with dyadic A (exactly PSD); s_k = certified upper bound of lambda_max(Tr_R Z_k) (s I - T = B B^dagger + eta I + E with ||E||_F < eta in ball arithmetic; Gershgorin as fallback); Z_k / s_k has Tr_R <= I (completion to a channel adds a nonnegative term); LB = (1/2) sum_k ||A^dagger C_k||_F^2 / (N2 s_k), evaluated in ball arithmetic. Physical-target bounds widen by 2 eps (normalized-vector distance bound; F is a POVM-element expectation). A second reader can re-derive F* with any SDP solver from the inputs above; the gate needs only the comparison with 1/2.", '',
      '## 7. Independent second read and repairs', '',
      f"Second read: `<internal>/MOCK_recovery_secondread_v1.md` (md5 `{md5('<internal>/MOCK_recovery_secondread_v1.md')}`, copy of <internal>/MOCK_recovery_secondread_v1.md), coordinator-run, different agent: all 12 states AGREE; its own certified bounds lie inside these. Repairs, regenerated by cw31_write_audit_v5.py (no hand edits): (1) the 'for every decoder' sentence in sec. 4 now states its scope (whole-lead light modes only, t = 1, 2, 4, n = 3; not decoders on the full heavy-plus-light lead output) and cites the empty-output sector contribution; (2) sec. 6 no longer calls the support compression exact (1e-13 cutoff) and states why the bounds stand (re-certified on the full space).", '',
      '## 8. Files (box <internal>/recovery, MD5SUMS there; code busi/code/clockwell/)', '', '| file | md5 |', '|---|---|']
for fn in ('cw18_admission.json', 'cw28_selftest_v4.json', 'cw27_sdp.json', 'cw19_stdout.txt', 'cw21_stdout.txt'):
    L.append(f'| `{R}/{fn}` | `{md5(f"{R}/{fn}")}` |')
for fn in ('cw18_recovery_admission.py', 'cw19_recovery_sdp.py', 'cw20_sdp_selftest.py', 'cw21_recovery_sdp_v2.py', 'cw22_sdp_selftest_v2.py', 'cw23_write_audit.py', 'cw24_recovery_sdp_v3.py', 'cw25_sdp_selftest_v3.py', 'cw26_write_audit_v2.py', 'cw27_recovery_sdp_v4.py', 'cw28_sdp_selftest_v4.py', 'cw29_write_audit_v3.py', 'cw30_write_audit_v4.py', 'cw31_write_audit_v5.py'):
    L.append(f'| `{B}/code/{fn}` | `{md5(f"{B}/code/{fn}")}` |')
L.append(f'| `{B}/code/cw39_write_audit_v1_1.py` | `{md5(f"{B}/code/cw39_write_audit_v1_1.py")}` |')
# ---------------- sec. 9: v1.1 additions
v1 = f'{R}/MOCK_recovery_audit_v1.md'; wip = f'{R}/MOCK_recovery_audit_v1.md.wip'
wip_txt = open(wip).read(); m27 = re.search(r'cw27_sdp\.json` \| `([0-9a-f]{32})`', wip_txt)
now27 = md5(f'{R}/cw27_sdp.json')
import numpy as _np
g0 = []
for t in (1.0, 2.0, 4.0):
    a = [x for x in adm['states'] if x['g'] == '0' and x['t'] == t and x['arm'] == 'baseline'][0]
    b = [x for x in adm['states'] if x['g'] == '0' and x['t'] == t and x['arm'] == 'filtered'][0]
    za = _np.load(a['array']); zb = _np.load(b['array'])
    da = float(_np.abs(za[za.files[0]] - zb[zb.files[0]]).max())
    g0.append((t, a['array_md5'], b['array_md5'], da))
ds = json.load(open('<internal>/cw35_dark_symmetry.json'))
dk = json.load(open('<internal>/cw34_dark.json'))
maxleakD = max(r['results']['D']['max_rel_leak'] for r in dk)
ns = sorted(set(r['n'] for r in dk))
maxcomm = max(max(c['comm_Q_H0'], c['comm_R_H0'], c['H_offblock_Qmod4']) for c in ds['checks'])
ncf = len(ds['closed_form_vs_cw27'])
L += ['## 9. v1.1 additions (#491; both models accepted v1)', '',
      f"(a) Final v1: `{v1}` md5 `{md5(v1)}` (kept unchanged). cw27_sdp.json md5 now `{now27}`; md5 recorded in the .wip the second reader checked `{m27.group(1) if m27 else 'not found'}`: {'unchanged' if m27 and m27.group(1) == now27 else 'DIFFERENT'}. Only the sec. 4 and sec. 6 wording was regenerated between the .wip and v1.", '',
      "(b) The g = 0 arrays coincide across the two converter forms (expected: at g = 0 the halo multiplies a zero interaction): " + '; '.join(f"t = {f(t)}: on-site md5 `{ma}`, filtered md5 `{mb}`, {'byte-identical' if ma == mb else 'max |difference| ' + f(dd, 3)}" for t, ma, mb, dd in g0) + '.', '',
      "(c) Why the empty-output sector gives 1/4: a symmetry of the vertex, and its consequence at every n. Let q = diag(+1, -1, -1, +1) on the four components of every mode (sigma_z x sigma_z) and r = 1 x sigma_x (c0 <-> c1, c2 <-> c3), extended to Q = dGamma(q) and the unitary R = Gamma(r) on Fock space (both species). "
      "(1) beta = sigma_z x 1 and alpha = sigma_x x sigma_x commute with q and with r, and both act site by site, so dGamma(h) commutes with Q and R for every clock profile N, mass and size n. "
      "(2) Every converter monomial (on-site; or filtered, any stencil L, kernel G and halo weight, since the fields d_{b,j} mix sites but not components) moves a component pair Q to a pair P with K_PQ nonzero; K is nonzero only where q(P) = +-q(Q), so H conserves Q mod 4. "
      "(3) The heavy pairs with |q| = 2 are 03 (q = +2) and 12 (q = -2), and K's columns for 03 and 12 are equal; so on any heavy state with |Q| = 2 the heavy factor of every converter monomial is proportional to H_03 + H_12, which is R-even (r maps 03 to 12). The dark codeword D = (|03> - |12>)/sqrt 2 has |Q| = 2 and RD = -D; H0 preserves both, so the H0-evolved D stays R-odd with |Q| = 2, every converter heavy factor annihilates it (its image is a multiple of the R-even vacuum with an R-odd coefficient, i.e. zero), V exp(-iH0 t) D = 0 for all t, and exp(-iHt) D = exp(-iH0 t) D: D never acquires light, at every clock, n, profile, mass, coupling and form. "
      "(4) B = |01>_H has Q = 0, so the B and D branches lie in different Q mod 4 sectors for all t; their components with an empty light output O are orthogonal (Q is additive and the empty O carries Q = 0). Hence for any output O made of light modes only (the receiver (i) or the diagnostic (ii)), with p0 the probability that the B branch has no particle in O: rho_RO's empty-O block is (1/2) diag(p0, 1) (so lambda_max(G0) = 1 exactly, G0 the Gram matrix of the two branches' empty-O components), the k >= 1 blocks are (1/2)|0><0| x X_k (only the B branch), and F* = (1/2)[lambda_max of the empty block + (1/2)(1 - p0)] = 1/4 + (1 - p0)/4 <= 1/2. "
      "So the light-only FAIL (upper <= 1/2) holds at every n, every clock, both forms and every g for this code: it is a property of the code (one codeword is dark by a symmetry of the vertex), not of n = 3; the n = 5 light-only receiver is therefore not informative and is not added (reviewer A mock2 (3)).", '',
      f"Script-emitted checks (cw34, cw35, busi's own builder cw32 which reproduces the saved n = 3 Hamiltonians and states, see MOCK_n5_plan_v1.md): K column 03 = column 12: {ds['K_col03_equals_col12']}; K charge rule q(P) = +-q(Q): {ds['K_charge_rule']}; max over n in {{2, 3, 5, 7}} and both forms of |[Q, H0]|, |[R, H0]| and of the H entries between Q = 0 and Q = 2 mod 4: {f(maxcomm, 3)}; Krylov test (n = {', '.join(str(x) for x in ns)}; both forms; clock-well and random profiles; two masses): max ||V q_j|| / ||V|| over the Krylov basis of D = {f(maxleakD, 3)}; the closed form 1/4 + (1 - p0)/4 lies inside all {ncf} certified intervals of sec. 3: {ds['closed_form_all_inside']}.", '',
      "(d) reviewer B's sign repair: the phrase '(-1)^(k(2-k)) = +1' occurs in the second read's sec. 1 (not in this file); (-1)^(k(2-k)) is +1 at k = 0, 2 and -1 at k = 1, and a uniform sign on a block cancels in rho_k = C_k C_k^dagger, so no number changes. The one 'exactly' that rested on a 1e-15 display (sec. 4) now cites the theorem in (c). ", '',
      '## 10. Files added in v1.1', '', '| file | md5 |', '|---|---|']
for pth in ('<internal>/cw32_mockmodel.py', '<internal>/cw34_dark_codeword.py', '<internal>/cw35_dark_symmetry.py',
            '<internal>/cw34_dark.json', '<internal>/cw35_dark_symmetry.json'):
    L.append(f'| `{pth}` | `{md5(pth)}` |')
L.append(f'| `{B}/code/cw43_write_audit_v1_2.py` | `{md5(f"{B}/code/cw43_write_audit_v1_2.py")}` |')
L.append(f'| `<internal>/cw42_dark_subspace_v2.py` | `{md5("<internal>/cw42_dark_subspace_v2.py")}` |')
L.append(f'| `<internal>/cw42_dark_subspace.json` | `{md5("<internal>/cw42_dark_subspace.json")}` |')
# ---------------- sec. 11: v1.2 additions (exact integer arithmetic on K)
PAIR = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
Kint = [(4, 0, 0, 0, 0, -4), (0, -8, 0, 0, 0, 0), (0, 0, -4, -4, 0, 0), (0, 0, -4, -4, 0, 0), (0, 0, 0, 0, -8, 0), (-4, 0, 0, 0, 0, 4)]
import importlib.util as _iu
_spec = _iu.spec_from_file_location('cw32', '<internal>/cw32_mockmodel.py'); _m = _iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
same_as_builder = all(int(_m.K[i][j]) == Kint[i][j] and float(Kint[i][j]) == _m.K[i][j] for i in range(6) for j in range(6))
qcomp = (1, -1, -1, 1)
qpair = [qcomp[a] + qcomp[b] for a, b in PAIR]
col_eq = all(Kint[i][2] == Kint[i][3] for i in range(6))
charge_rule = all(Kint[i][j] == 0 or qpair[i] in (qpair[j], -qpair[j]) for i in range(6) for j in range(6))
nnzK = sum(1 for i in range(6) for j in range(6) if Kint[i][j])
dsub = json.load(open('<internal>/cw42_dark_subspace.json'))
L += ['', '## 11. v1.2 additions (#492; both models confirmed the theorem of sec. 9(c))', '',
      '(a) The converter matrix K, exact integers (rows: light creation pair P, columns: heavy annihilation pair Q, both in the antisymmetric component-pair order ' + ', '.join(f'{a}{b}' for a, b in PAIR) + '; the same integers as the frozen adapter and as busi\'s builder: ' + str(same_as_builder) + '):', '',
      '| P \\ Q | ' + ' | '.join(f'{a}{b}' for a, b in PAIR) + ' |', '|---|' + '---|' * 6]
for i, (a, b) in enumerate(PAIR):
    L.append(f'| {a}{b} | ' + ' | '.join(str(Kint[i][j]) for j in range(6)) + ' |')
L += ['', f"Pair charges q(P) = q(a) + q(b) with q = (+1, -1, -1, +1): " + ', '.join(f'{a}{b}: {qpair[i]:+d}' for i, (a, b) in enumerate(PAIR)) + f". Exact integer arithmetic (Python int, no floating point): column 03 equals column 12 entry by entry: {col_eq}; every nonzero K_PQ ({nnzK} of 36) has q(P) = +-q(Q): {charge_rule}. These two facts are exact properties of the integer matrix; the phrase 'symmetry of the vertex' in sec. 9 is a conclusion drawn from this matrix, not from a separately stated principle.", '',
      "(b) Hypotheses of the 'every n' statement: (H1) the one-body generator is h = sum over site operators tensored with 1, beta = sigma_z x 1 and alpha = sigma_x x sigma_x only (any real clock profile, mass and size); (H2) the converter is built from this K with component-blind spatial fields (each field d_{b,j} mixes sites with scalar stencil weights, never components) and scalar kernel and halo weights. Both hold for the on-site and filtered forms as implemented; the zero commutators and Krylov leakage at n = 2, 3, 5, 7 (sec. 9) are evidence for the implementation, and the algebra under (H1)-(H2) is the proof.", '',
      "(c) Scope of R: the proof uses only that R commutes with H0 and that the heavy annihilation factor on the |Q| = 2 sector is R-even. R is not claimed to be a symmetry of the full H. (Measured, not claimed: max |[Gamma(r), H]| at g = 1 is " + ', '.join(f"{x['label']}: {x['comm_R_H_max']:.3g}" for x in dsub) + '.)', '',
      "(d) N = 2 only. Nothing is claimed for many-body states: for example c0^dag c1^dag c3^dag |0> at one site is not annihilated by H_03 + H_12 (h_3 h_0 removes c0 and c3 and leaves -c1^dag|0>), so no many-body dark remnant follows from this theorem.", '',
      "(f) The closed form F* = 1/4 + (1 - p0)/4 is an identity for the exact state; the statement that it lies inside all 24 certified intervals of sec. 3 is what covers the computed states.", '',
      "Wording (reviewer B): '03 and 12' denote the converter's antisymmetric component-pair columns (same-centre field pairs), not every spatially separated two-particle basis state. 'Every n' means this two-particle, initially light-vacuum code under the stated Hamiltonians; it is not a general many-body or arbitrary-intervention no-go. F* = 1/2 (attained only at p0 = 0) would be the perfect transmission of a classical bit, not qubit recovery. p0 is conditional on the normalized B branch; the registered state's empty-output (vacuum) weight is (1 + p0)/2.", '',
      "Dark-subspace readout (reviewer A A, cw42; W = largest H_H-invariant subspace of ker T, T the heavy-to-light converter block at g = 1): " + '; '.join(f"{x['label']}: dim W = {x['dim_W']} (n(2n-1) = {x['protected_dim']}, heavy sector {x['heavy_dim']}), ||P_W B|| = {x['P_W_B']:.2g}, ||P_W B'|| = {x['P_W_Bprime']:.2g}, ||P_W D|| = {x['P_W_D']:.12f}" for x in dsub) + ". For the filtered form W is exactly the symmetry-protected sector; for the on-site form W is larger than it (additional exactly dark directions), with B and B' outside W in both forms.", '']
p = f'{R}/MOCK_recovery_audit_v1.2.md'
open(p, 'w').write('\n'.join(L))
print('wrote', p, md5(p))
