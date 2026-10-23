"""cw17 (v2 of cw14, after the independent second read MOCK_clockwell_secondread_v1.md): write busi's two clock-well reads (complete files, WORKING names .wip) from the saved JSON products
cw05 (icloud saved values), cw06 (busi independent reductions), cw09 (static halo), cw11 (baseline V_X probes),
cw12 (return-probability re-derivation). Every number in the two files is emitted here from those JSONs.
Outputs (box): <internal>/MOCK_filtered_clockwell_v1.md.wip
               <internal>/MOCK_unitary_lead_baseline_v1.md.wip
"""
import json, hashlib, math, os

B = '<internal>/clockwell'
SR = f'{B}/inputs/MOCK_clockwell_secondread_v1.md'
J = {k: f'{B}/{k}.json' for k in ('cw05_extract', 'cw06_independent', 'cw09_static_halo', 'cw11_vertex_probes', 'cw12_return_check', 'cw16_intake_audit')}

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

ext = json.load(open(J['cw05_extract'])); ind = json.load(open(J['cw06_independent']))
aud = json.load(open(J['cw16_intake_audit'])); st = json.load(open(J['cw09_static_halo'])); vp = json.load(open(J['cw11_vertex_probes'])); rc = json.load(open(J['cw12_return_check']))
JMD5 = {k: md5(p) for k, p in J.items()}
MAN = f'{B}/INPUT_MD5SUMS.txt'; MANMD5 = md5(MAN)

HALO = 'endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement; no lapse sweep; n = 7 static only)'
HALO_SHORT = '[endpoint-copy halo; N_-1 = 1/2, N_n = 1]'
DECL = 'endpoint-copy halo; N_-1=N_0=1/2; N_n=N_(n-1)=1; declared convention, not a measurement'
GS = [('0', '0'), ('0p5', '1/2'), ('1', '1'), ('2', '2')]

def f(x, s=6):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return str(x)
    return f'{x:.{s}g}'

def I(lo, hi, s=10):
    return f'[{f(lo, s)}, {f(hi, s)}]'

def iv(d, s=10):
    return I(d['lower'], d['upper'], s)

def key(arm, n, gs):
    return f'{arm}/even_n{n}_g{gs}_v1'

def conv_summary(P):
    conv = [c for c in P['conversion'] if not c['dephased']]
    stt = {s['clock']: s for s in P['conversion_status']}
    ncc = sum(1 for s in P['conversion_status'] if s['coherent_conversion'])
    mx = max(conv, key=lambda c: c['p_LL']['value'])
    at4 = [c for c in conv if c['clock'] == 4.0][0]
    return conv, ncc, len(P['conversion_status']), mx, at4, stt

def wall_at(P, t):
    w = P.get('wall') or {}
    for c, b in zip(w.get('prefix_clocks') or [], w.get('bounds') or []):
        if abs(float(c) - t) < 1e-12:
            return b
    return None

def reg(P, clk, role, first=False):
    r = [x for x in P['register'] if x['clock'] == clk and x['role'] == role and bool(x['first_collected']) == first]
    return r[0] if r else None

def fe(arm, n, gs, clk, ch, first=False):
    """F_e interval: icloud's saved interval if its readout job exists, else busi's reduction (labelled)."""
    k = key(arm, n, gs)
    Q = ext['readouts'].get(k)
    if Q and not Q.get('run_by'):
        r = [x for x in Q['receiver'] if x['clock'] == clk and x['choice'] == ch and bool(x['first']) == first]
        if r:
            return I(*r[0]['fidelity_interval']), 'icloud'
    R = ind['cohorts'][k]
    r = [x for x in R['receiver'] if x['clock'] == clk and x['choice'] == ch and x['first'] == first]
    return (I(r[0]['F_lower'], r[0]['F_upper']), 'busi') if r else ('n/a', '')

def purity_row(arm, n, gs, clk, cut):
    k = key(arm, n, gs); R = ind['cohorts'][k]; P = ext['physical'][k]
    r = [x for x in R['cuts'] if x['clock'] == clk and x['cut'] == cut]
    c = [x for x in P['conversion'] if x['clock'] == clk and not x['dephased']]
    if not r or not c:
        return 'n/a'
    eps = min(1.0, c[0]['state_error'] or 0.0); p = r[0]['purity']
    lo = max(p - 4*eps, 1.0/r[0]['dim_support']); hi = min(1.0, p + 4*eps)
    return I(lo, hi, 8)

def cohort_rows(arm, ns):
    rows = []
    for n in ns:
        for gs, gl in GS:
            k = key(arm, n, gs); P = ext['physical'][k]; R = ind['cohorts'][k]; t = P['termination']
            devs = [max(x['dev_pLL'] for x in R['conversion']), max(x['dev_C'] for x in R['conversion']),
                    max(x['dev_S'] for x in R['cuts'] if x['dev_S'] is not None),
                    max(x['dev_D'] for x in R['register'] if x['dev_D'] is not None)]
            pu = R['purification']
            devs.append(max(pu['dev_finite'], pu['dev_ext']) if pu else 0.0)
            ok = (t['status'] == 'complete' and t['all_pass'] and t['sources_unchanged'] and not P['checks_failed']
                  and P['checks_total'] == t['checks'] and max(devs) <= 1e-12)
            rows.append((n, gl, k, P, t, devs, ok))
    return rows

def header_audit(P):
    tot = bad = 0
    for lst in (P['conversion'], P['conversion_status']):
        for x in lst:
            tot += 1; bad += (x.get('header') != DECL)
    for kk in ('wall',):
        if P.get(kk):
            tot += 1; bad += (P[kk].get('header') != DECL)
    if P.get('purification'):
        tot += 1; bad += (P['purification'].get('header') != DECL)
    if P.get('purification_status'):
        tot += 1; bad += (P['purification_status'].get('header') != DECL)
    if P.get('B2_at_t1'):
        tot += 1; bad += (P['B2_at_t1'].get('header') != DECL)
    m = (P.get('wall') or {}).get('context_model') or {}
    tot += 1; bad += (m.get('halo_values') != ['1/2', '1'])
    return tot, bad

# ---------------------------------------------------------------- shared table builders
def t_conversion(arms, ns, title, halo):
    L = []
    if halo:
        L += [f'**{title}** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '']
    else:
        L += [f'**{title}**', '']
    L += ['| arm | n | g | clocks with coherent conversion (of 17) | max sampled p_LL (clock) | p_LL interval at max | C_HL interval at max | p_LL interval at t = 4 | C_HL interval at t = 4 |',
          '|---|---|---|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            for arm in arms:
                P = ext['physical'][key(arm, n, gs)]
                conv, ncc, ntot, mx, at4, _ = conv_summary(P)
                lab = 'filtered ' + HALO_SHORT if arm == 'filtered' else 'on-site baseline'
                L.append(f"| {lab} | {n} | {gl} | {ncc} of {ntot} | {f(mx['p_LL']['value'])} ({f(mx['clock'])}) | {iv(mx['p_LL'])} | {iv(mx['C_HL'])} | {iv(at4['p_LL'])} | {iv(at4['C_HL'])} |")
    return L

def t_populations(arm, ns, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    b2r = ext['physical'][key(arm, 3, '1')]['B2_at_t1']; b2clk = b2r['clock']
    b2ok = all(abs(ext['physical'][key(arm, 3, gs)]['B2_at_t1']['site_' + sp]['value'] - [c for c in ext['physical'][key(arm, 3, gs)]['conversion'] if c['clock'] == ext['physical'][key(arm, 3, gs)]['B2_at_t1']['clock'] and not c['dephased']][0]['per_site']['1_' + sp]['value']) == 0 for gs, _ in GS for sp in ('H', 'L', 'total'))
    L += [f"Site-1 column: the saved record's populations are the t = {f(b2clk)} (first collection clock t1) per-site populations ({'verified identical to the per-site t = ' + f(b2clk) + ' conversion rows for every n = 3 cohort' if b2ok else 'NOT identical to the per-site rows: see records'}); its preceding_primary_clock field ({f(b2r['preceding_primary_clock'])}) is not the clock of these values.", '']
    L += [f'| n | g | core H at t = 4 | core L | lead H | lead L | state error (each) | n = 3: site 1 (later-collected B2) H / L / total at t = {f(b2clk)}, error |',
          '|---|---|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            P = ext['physical'][key(arm, n, gs)]
            at4 = [c for c in P['conversion'] if c['clock'] == 4.0 and not c['dephased']][0]
            s = at4['sites']; b2 = P.get('B2_at_t1')
            bb = (f"{f(b2['site_H']['value'])} / {f(b2['site_L']['value'])} / {f(b2['site_total']['value'])}, {f(b2['site_total']['error'], 3)}"
                  if b2 else 'not applicable (one lead site)')
            L.append(f"| {n} | {gl} | {f(s['core_H']['value'])} | {f(s['core_L']['value'])} | {f(s['lead_H']['value'])} | {f(s['lead_L']['value'])} | {f(s['core_H']['error'], 3)} | {bb} |")
    return L

def t_purification(arms, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['Rows: n = 3 only (n = 2 has one lead site: purification test not applicable by protocol).', '',
          '| arm | g | primary c2 = S(A2;2) - S(A1;2) | primary D2 = S(A2;2) - S(A1;1) | protected conditional Cq | protected Duhamel bound B_first (t = 1) | B_second (t = 2) | Bwall(1) | Bwall(2) | U_cond finite | U_cum finite | U_cond extended | U_cum extended | icloud status (finite / extended) | busi recomputed: exclusion established? |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for gs, gl in GS:
        for arm in arms:
            k = key(arm, 3, gs); P = ext['physical'][k]; pu = P['purification']; R = ind['cohorts'][k]['purification']
            lab = 'filtered' if arm == 'filtered' else 'on-site'
            est = 'no' if not (R['bounded_return_exclusion_finite'] or R['bounded_return_exclusion_ext']) else 'YES'
            L.append(f"| {lab} | {gl} | {iv(pu['primary_conditional'], 8)} | {iv(pu['primary_cumulative'], 8)} | {iv(pu['protected_conditional'], 8)} | {f(R['B1'])} | {f(R['B2'])} | {f(R['Bwall1'])} | {f(R['Bwall2'])} | {f(R['U_cond_finite'])} | {f(R['U_cum_finite'])} | {f(R['U_cond_ext'])} | {f(R['U_cum_ext'])} | {pu['finite_chain']['status']} / {pu['empty_uniform_extension']['status']} | {est} |")
    L += ['', ('**Table ' + title.split('.')[0].replace('**', '').replace('Table ', '') + 'b. Primary-versus-protected endpoint distance, proof bound and switch work** ' + (HALO_SHORT if halo else '')).rstrip(), '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['Computed primary-versus-protected endpoint distance and its proof bound (consistency of the return comparison), and the recorded switch work:', '',
          '| arm | g | computed normalized-state distance interval | B-plus-endpoint-error bound | consistency | switch work interval |', '|---|---|---|---|---|---|']
    for gs, gl in GS:
        for arm in arms:
            pu = ext['physical'][key(arm, 3, gs)]['purification']; cd = pu['computed_distance']; sw = pu.get('switch_work') or {}
            L.append(f"| {'filtered' if arm == 'filtered' else 'on-site'} | {gl} | {I(cd.get('lower'), cd.get('upper'))} | {f(pu['computed_distance_bound'])} | {pu['consistency']} | {I(sw.get('lower'), sw.get('upper')) if sw else 'not recorded'} |")
    return L

def t_wall(arms, ns, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['Pure-input Duhamel distance bound to the declared empty four-site (filtered) / uniform (on-site) extension; a state-distance bound, not a return probability.', '',
          '| arm | n | g | Bwall(1) | Bwall(2) | Bwall(3) | Bwall(4) |', '|---|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            for arm in arms:
                P = ext['physical'][key(arm, n, gs)]
                L.append(f"| {'filtered' if arm == 'filtered' else 'on-site'} | {n} | {gl} | " + ' | '.join(f(wall_at(P, t)) for t in (1.0, 2.0, 3.0, 4.0)) + ' |')
    return L

def t_return(arms, ns, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['Lüders two-time history (addendum sec. 3): escape test at t_e = 1 (all particles outside site 0), then some particle at site 0 at t. Joint probabilities, no postselection.', '',
          '| arm | n | g | p_escape interval | joint p_return at t = 2 | at t = 3 | at t = 4 | max joint p_return over the 13 clocks (clock) | busi re-derivation inside all icloud intervals |',
          '|---|---|---|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            for arm in arms:
                k = key(arm, n, gs); Q = ext['readouts'].get(k)
                lab = 'filtered' if arm == 'filtered' else 'on-site'
                if not Q or not Q['return']:
                    L.append(f'| {lab} | {n} | {gl} | MISSING (readout job never run) | MISSING | MISSING | MISSING | MISSING | not run (see sec. 8) |')
                    continue
                if Q.get('run_by'):
                    lab += ' (busi-run, icloud adapter)'
                rr = {x['readout_clock']: x for x in Q['return']}
                mxr = max(Q['return'], key=lambda x: x['joint_return']['interval'][1])
                ins = rc.get(k, {}).get('all_inside')
                L.append(f"| {lab} | {n} | {gl} | {I(*rr[1.0]['escape']['interval'])} | {I(*rr[2.0]['joint_return']['interval'])} | {I(*rr[3.0]['joint_return']['interval'])} | {I(*rr[4.0]['joint_return']['interval'])} | {I(*mxr['joint_return']['interval'])} ({f(mxr['readout_clock'])}) | {f(ins)} |")
    return L

def t_recovery(arm, ns, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['D_RK = 1/2 ||rho_RK - rho_R x rho_K||_1 with K = all fermion modes outside the output O (decoupled to tolerance only if upper(D) <= 1e-4); F_e = entanglement fidelity of the fixed local-pair CPTP receiver (addendum sec. 2; 1/4 is the failure-output value). Source column: icloud = icloud saved readout; busi = busi reduction of the saved registered vector (for filtered n = 3, g = 0, 1/2, 2, whose readout jobs icloud never ran; busi later ran the icloud adapter for them and the adapter own F_e equals the busi value to the deviation in Table F6).', '',
          '| n | g | clock | output | D_RK interval | status | F_e interval | F_e source |', '|---|---|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            P = ext['physical'][key(arm, n, gs)]
            items = [(1.0, 'light', False), (1.0, 'all', False)]
            if n == 3:
                items += [(1.0, 'light', True), (1.0, 'all', True)]
            items += [(2.0, 'light', False), (2.0, 'all', False), (4.0, 'light', False), (4.0, 'all', False)]
            for clk, ch, first in items:
                r = reg(P, clk, ch, first)
                F, src = fe(arm, n, gs, clk, ch, first)
                out = ('first block (site 2) ' if first else 'whole lead ') + ('light only' if ch == 'light' else 'heavy + light')
                L.append(f"| {n} | {gl} | {f(clk)} | {out} | {iv(r['D'])} | {r['status']} | {F} | {src} |")
    return L

def t_purity(arm, ns, title, halo):
    L = [f'**{title}** {HALO_SHORT}' if halo else f'**{title}**', '']
    if halo:
        L += [f'Halo rule: {HALO}.', '']
    L += ['Fixed-cut purity Tr(rho_A^2) of the pure input, busi graded reduction of the saved raw vectors, interval = value +- 4 min(1, eps) intersected with [1/d, 1] (addendum sec. 1); busi values agree with icloud saved purities to the deviation in the cohort table where icloud computed them.', '',
          '| n | g | cut | t = 1 | t = 2 | t = 4 |', '|---|---|---|---|---|---|']
    for n in ns:
        for gs, gl in GS:
            cuts = ['sites0', 'lead'] + (['sites2'] if n == 3 else [])
            for cut in cuts:
                lab = {'sites0': 'core (site 0)', 'lead': 'whole lead', 'sites2': 'B1 (site 2)'}[cut]
                L.append(f"| {n} | {gl} | {lab} | " + ' | '.join(purity_row(arm, n, gs, t, cut) for t in (1.0, 2.0, 4.0)) + ' |')
    return L

def second_reader_inputs(k):
    P = ext['physical'][k]
    return f"`{P['report']}` (md5 `{P['report_md5']}`) and the npz arrays it names in the same directory (md5s in `{MAN}`, md5 `{MANMD5}`)"

# ---------------------------------------------------------------- filtered document
def filtered_doc():
    L = []
    L += ['# MOCK filtered clock well v1: endpoint-copy halo, filtered control vs on-site baseline (busi takeover of icloud #145)', '',
          'Final delivery after the coordinator-run independent second read (sec. 10). Author: busi (analyst engineer), lane busi, under lead order busi inbox #489 (icloud out of usage until 29 Oct).', '',
          f'**Halo rule, printed as required by #145: {HALO}.** Every clock-well table below repeats it; every result is labelled endpoint-copy halo. It is a declared convention, not a measurement, and no second halo value was propagated.', '',
          'Scope: fixed no-flow lattice mock, n = 2 and 3 propagated, n = 7 static only; g = 0, 1/2, 1, 2 with g = 1 the baseline strength. No rate, flow, current, chronology, source, Page time or manuscript text is printed or implied. Nothing here was propagated by busi: all states, eigensystems, entropies and proofs are icloud saved outputs on the box; busi only read them, re-derived the scalar readouts by its own reductions of the saved arrays, and re-evaluated the protocol inequalities.', '']
    # sec 1
    L += ['## 1. What was pending, and what busi did', '',
          'From icloud STATUS (#145 line) and its sole-reviewer record (sections up to 81): all eight filtered physical cohorts were complete and reaped; the four size-two physical judgments were closed by the original pair; size-three zero coupling had author closure only; the size-three half and double coupling physical assessments had no closure; the size-two half and double coupling new-readout (receiver / purity / return-history) assessments were pending; the size-three new-readout and input-audit jobs for g = 0, 1/2, 2 were never run by icloud (only g = 1 exists); under the coordinator ruling busi ran them with the unchanged icloud adapter, with a busi intake audit (sec. 8, Tables F6 and F6b). icloud delivered no clock-well file and no on-site baseline read (deliveries/solve_b has no MOCK_unitary_lead_run_v1.md); the baseline read is delivered separately as MOCK_unitary_lead_baseline_v1.md.wip.', '',
          'busi did, on the box CPU only (nice 10, no GPU; thread counts: cw06 and cw12 ran with OMP/OPENBLAS/MKL = 4 BLAS threads, every other busi script single-threaded Python, and the three adapter intake and readout jobs of sec. 8 with all six thread variables = 1), reading icloud outputs in place: (a) extraction of every saved scalar readout (cw05); (b) an independent reduction of the saved density matrices and raw state vectors: populations, C_HL, fixed-cut entropies and purities, D_RK and the fixed-receiver F_e, by busi\'s own graded fermionic partial trace (cw06); (c) re-evaluation of the bounded-return-exclusion inequalities from the saved scalars with the Audenaert function (cw06); (d) exact rational recomputation of the #145 static halo quantities and comparison with the saved static records (cw09); (e) re-derivation of the saved two-time return probabilities from the saved Hamiltonian and saved t = 1 state, for the cohorts where icloud computed them (cw12).', '']
    # sec 2 static
    be = st['busi_exact']; recs = st['records']
    L += ['## 2. Static #145 reporting (from the saved static jobs; n = 7 static only)', '',
          f'**Table F1. Halo quantities, exact, U = 1** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| quantity | busi exact (from G0, G1, G2) | decimal | saved static record equal? |', '|---|---|---|---|',
          f"| sum G_r over r = -2..2 | {be['sum_G']['exact']} | {f(be['sum_G']['float'])} | kernel admitted by icloud |",
          f"| site-0 <L01|V_N|H01>, endpoint-copy halo, (19G0+20G1+3G2)/64 | {be['site0_01_endpoint_copy']['exact']} | {f(be['site0_01_endpoint_copy']['float'], 12)} | n = 2: {f(recs['2']['site0_01_equal_busi'])}; n = 3: {f(recs['3']['site0_01_equal_busi'])}; n = 7: {f(recs['7']['site0_01_equal_busi'])} |",
          f"| same entry, flat (no clock), (9G0+8G1+G2)/16 | {be['site0_01_flat']['exact']} | {f(be['site0_01_flat']['float'], 12)} | saved flat target check passes |",
          f"| same entry with N_-1 = 1 (rejected wrong-convention control), (5G0+6G1+G2)/16 | {be['site0_01_Nminus1_eq_1_rejected']['exact']} | {f(be['site0_01_Nminus1_eq_1_rejected']['float'], 12)} | control only |",
          f"| same entry with the halo centre removed (rejected), (9G0+6G1)/32 | {be['site0_01_halo_removed_rejected']['exact']} | {f(be['site0_01_halo_removed_rejected']['float'], 12)} | control only |",
          f"| dH/dN_-1 site-0 01 entry, (G0+4G1+G2)/32 | {be['dH_dNminus1_site0_01']['exact']} | {f(be['dH_dNminus1_site0_01']['float'], 12)} | n = 2/3/7: {f(recs['2']['same_site01_equal_busi'])}/{f(recs['3']['same_site01_equal_busi'])}/{f(recs['7']['same_site01_equal_busi'])} |",
          f"| exact Schur (l1 row-sum) bound of ||dH/dN_-1|| = R(W_-1), n = 2, (2G0+9G1+6G2)/32 | {be['schur_R_Wminus1_n2']['exact']} | {f(be['schur_R_Wminus1_n2']['float'], 12)} | {f(recs['2']['schur_equal_busi'])} (tied rows: {recs['2']['schur_ties']}) |",
          f"| same, n = 3 and n = 7, (2G0+9G1+9G2)/32 | {be['schur_R_Wminus1_n3_n7']['exact']} | {f(be['schur_R_Wminus1_n3_n7']['float'], 12)} | n = 3: {f(recs['3']['schur_equal_busi'])}, n = 7: {f(recs['7']['schur_equal_busi'])} (tied rows: {recs['3']['schur_ties']}, {recs['7']['schur_ties']}) |",
          '', 'General U multiplies the Schur bound by |U|. It measures sensitivity to the declared convention, not measured halo robustness.', '',
          f'**Table F2. Centre -1 allocated rows beside their flat values** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| n | centre -1 entries | ratio allocated N_-1 W_-1 / flat W_-1 (count) | distinct |flat| -> |allocated| values (count) |', '|---|---|---|---|']
    for n in ('2', '3', '7'):
        R = recs[n]
        vals = '; '.join(f"{v['flat_float']:.6g} -> {v['allocated_float']:.6g} ({v['count']})" for v in R['flat_vs_allocated_abs_values'])
        L.append(f"| {n} | {R['centre_minus1_entries']} | " + ', '.join(f'{a} ({b})' for a, b in R['allocated_over_flat_ratios'].items()) + f' | {vals} |')
    L += ['', f'**Table F3. Joining (wall) source constants and static admission** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| n | U | C_join^2 = R(Q) | H_bound = R(H_short) | L^2 | static checks (all pass) | static wall / CPU s, RSS KiB |', '|---|---|---|---|---|---|---|']
    for n in ('2', '3', '7'):
        R = recs[n]; t = R['termination']
        if R['joining']:
            for U, j in R['joining'].items():
                L.append(f"| {n} | {U} | {j['Cjoin_squared']} | {j['Hbound']} | {j['L_squared'] if len(j['L_squared']) < 60 else j['L_squared'][:57] + '...'} | {t['checks']} ({f(t['all_pass'])}) | {f(t['wall'], 5)} / {f(t['cpu'], 5)}, {t['rss_kib']} |")
        else:
            L.append(f"| {n} | n/a (static only, no joining reference at n = 7) | | | | {t['checks']} ({f(t['all_pass'])}) | {f(t['wall'], 5)} / {f(t['cpu'], 5)}, {t['rss_kib']} |")
    n7 = [x for x in recs['7']['target_checks'] if ('full_support' in x['name'] or 'threshold' in x['name']) and x['name'].startswith(('n7_collinear_1_(1, 1)_(1, 1)', 'n7_valley_0000', 'n7_valley_0101', 'full_support_spin', 'weighted_tangent_(0, 1)', 'weighted_tangent_(1, 1)'))]
    n7 = [x for x in n7 if x['name'].endswith(('full_support_target', 'threshold_zero', 'intervalley_zero', 'spin_average', 'full_support_derivative'))]
    L += ['', f'**Table F4. n = 7 static interior (full-support site 3) targets, the comparison targets for n = 2, 3** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| saved check | value | target | pass |', '|---|---|---|---|']
    for x in n7:
        L.append(f"| {x['name']} | {x['actual']} | {x['expected']} | {f(x['pass'])} |")
    # sec 3 cohort assessment
    rows = cohort_rows('filtered', (2, 3))
    L += ['', '## 3. Cohort assessments (all eight filtered physical cohorts; the pending ones are n = 3, g = 0, 1/2, 2)', '',
          f'**Table F5. Filtered physical cohorts: completion, controls and busi re-derivation** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          'Deviation columns: max |busi - icloud saved| over all clocks for p_LL, C_HL (from the saved density arrays), fixed-cut entropies (busi reduction of the saved raw vectors), D_RK (busi reduction of the saved registered vectors), and the four return-exclusion upper bounds (busi Audenaert re-evaluation). Accepted means: termination complete, every saved predicate true, sources unchanged, predicate count equal to the termination count, and every deviation <= 1e-12.', '',
          '| n | g | report md5 | predicates (all true) | sources unchanged | job wall / CPU s, RSS KiB | header audit (records with halo header / mismatches) | dev p_LL | dev C_HL | dev S | dev D_RK | dev U | busi verdict |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, gl, k, P, t, devs, ok in rows:
        ht, hb = header_audit(P)
        L.append(f"| {n} | {gl} | `{P['report_md5']}` | {P['checks_total']} ({f(t['all_pass'])}) | {f(t['sources_unchanged'])} | {f(t['wall'], 6)} / {f(t['cpu'], 6)}, {t['rss_kib']} | {ht} / {hb} | " + ' | '.join(f(d, 2) for d in devs) + f" | {'accepted' if ok else 'NOT accepted'} |")
    # g=0 identity control
    a = ind['cohorts'][key('filtered', 3, '0')]; b = ind['cohorts'][key('baseline', 3, '0')]
    a2 = ind['cohorts'][key('filtered', 2, '0')]; b2 = ind['cohorts'][key('baseline', 2, '0')]
    def maxdiff(x, y, fld, lst):
        m = 0.0
        for u in x[lst]:
            v = [w for w in y[lst] if w['clock'] == u['clock'] and all(u.get(kk) == w.get(kk) for kk in ('cut', 'choice', 'first') if kk in u)]
            if v:
                m = max(m, abs(u[fld] - v[0][fld]))
        return m
    L += ['', f"Zero-coupling identity control (at g = 0 the halo multiplies a zero interaction, so the filtered and on-site g = 0 trajectories must coincide; only the wall source differs): max |filtered - on-site| at n = 2 / n = 3: p_LL {f(maxdiff(a2, b2, 'pLL', 'conversion'), 2)} / {f(maxdiff(a, b, 'pLL', 'conversion'), 2)}; p_HH {f(maxdiff(a2, b2, 'pHH', 'conversion'), 2)} / {f(maxdiff(a, b, 'pHH', 'conversion'), 2)}; cut entropies {f(maxdiff(a2, b2, 'S', 'cuts'), 2)} / {f(maxdiff(a, b, 'S', 'cuts'), 2)}; D_RK {f(maxdiff(a2, b2, 'D', 'register'), 2)} / {f(maxdiff(a, b, 'D', 'register'), 2)}; F_e {f(maxdiff(a2, b2, 'F_e', 'receiver'), 2)} / {f(maxdiff(a, b, 'F_e', 'receiver'), 2)}.", '']
    # new-readout assessments n2 g1/2, g2 (and all existing)
    L += [f'**Table F6. Filtered new readouts (receiver, purity, two-time history): icloud saved jobs vs busi re-derivation** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| n | g | run by | readout report md5 | termination all true | receivers / purities / return clocks | max |busi F_e - adapter| | max |busi purity - adapter| | busi return re-derivation inside adapter intervals (trajectory check at t = 1) | busi verdict |',
          '|---|---|---|---|---|---|---|---|---|---|']
    cmp = ind['compare_with_icloud_readouts']
    for n in (2, 3):
        for gs, gl in GS:
            k = key('filtered', n, gs); Q = ext['readouts'].get(k)
            if not Q:
                L.append(f'| {n} | {gl} | MISSING: icloud never ran this readout job | | | | | | not assessable (see sec. 8) |')
                continue
            c = cmp[k]; r = rc[k]; t = Q['termination']
            ok = t['all_pass'] and c['max_dev_F'] <= 1e-12 and c['max_dev_purity'] <= 1e-12 and r['all_inside']
            L.append(f"| {n} | {gl} | {'busi (icloud adapter unchanged, cw15)' if Q.get('run_by') else 'icloud'} | `{Q['report_md5']}` | {f(t['all_pass'])} | {len(Q['receiver'])} / {len(Q['purity'])} / {len(Q['return'])} | {f(c['max_dev_F'], 2)} | {f(c['max_dev_purity'], 2)} | {f(r['all_inside'])} ({f(r['trajectory_check_t1'], 2)}) | {'accepted' if ok else 'NOT accepted'} |")
    L += ['', f'**Table F6b. busi retained-input intake audits for the three busi-run readout jobs (adapter stage intake, then busi assessment file bound by the readout gate)** {HALO_SHORT}', '', f'Halo rule: {HALO}.', '',
          '| n | g | intake journal sha256 | intake report sha256 (identical bytes to the journal for intake jobs, as in icloud n = 3, g = 1) | intake binding | busi assessment file sha256 | checks passed | intake wall / CPU s, RSS KiB | verdict |', '|---|---|---|---|---|---|---|---|---|']
    for gs, gl in (('0', '0'), ('0p5', '1/2'), ('2', '2')):
        a_ = aud[gs]
        L.append(f"| 3 | {gl} | `{a_['journal_sha256']}` | `{a_['report_sha256']}` | `{a_['binding']}` | `{a_['review_sha256']}` | {sum(a_['checks'].values())} of {len(a_['checks'])} | {f(a_['wall'], 4)} / {f(a_['cpu'], 4)}, {a_['rss_kib']} | {'accepted' if a_['ok'] else 'NOT accepted'} |")
    # sec 4 conversion
    L += ['', '## 4. Conversion (coherent conversion: positive lower bounds on both p_LL and C_HL = ||P_HH rho P_LL||_F)', '']
    L += t_conversion(['filtered', 'baseline'], (2, 3), 'Table F7. Coherent conversion, filtered control beside the on-site baseline (same states, clocks, U)', True)
    L += ['']
    L += t_populations('filtered', (2, 3), 'Table F8. Filtered populations at t = 4 and the later-collected block before collection', True)
    # sec 5 return exclusion
    L += ['', '## 5. Return exclusion', '']
    L += t_purification(['filtered', 'baseline'], 'Table F9. Bounded-return-exclusion purification (n = 3, collection B1 = site 2 at t = 1, B2 = site 1 at t = 2)', True)
    L += ['']
    L += t_wall(['filtered', 'baseline'], (2, 3), 'Table F10. Wall (extension) bound', True)
    L += ['']
    L += t_return(['filtered', 'baseline'], (2, 3), 'Table F11. Two-time return probability', True)
    # sec 6 recovery
    L += ['', '## 6. Recovery and purification readouts', '']
    L += t_recovery('filtered', (2, 3), 'Table F12. Filtered register decoupling and fixed-receiver fidelity', True)
    L += ['']
    L += t_purity('filtered', (2, 3), 'Table F13. Filtered fixed-cut purities', True)
    # sec 7 judgments
    L += ['', '## 7. Judgments', '']
    jl = []
    for n, gl, k, P, t, devs, ok in rows:
        if n == 3 and gl in ('0', '1/2', '2'):
            jl.append((f'J-F-phys-n3-g{gl}', f"Filtered physical cohort n = 3, g = {gl} {HALO_SHORT}: {'ACCEPTED' if ok else 'NOT ACCEPTED'} (completion, {P['checks_total']} saved predicates true, sources unchanged, busi re-derivation max deviation {f(max(devs), 2)}).", k))
    for gs, gl in (('0p5', '1/2'), ('2', '2')):
        k = key('filtered', 2, gs); c = cmp[k]; r = rc[k]; t = ext['readouts'][k]['termination']
        ok = t['all_pass'] and c['max_dev_F'] <= 1e-12 and c['max_dev_purity'] <= 1e-12 and r['all_inside']
        jl.append((f'J-F-read-n2-g{gl}', f"Filtered new readouts n = 2, g = {gl} {HALO_SHORT}: {'ACCEPTED' if ok else 'NOT ACCEPTED'} (receiver F_e and purities re-derived to {f(max(c['max_dev_F'], c['max_dev_purity']), 2)}; every joint return and escape probability re-derived inside icloud's interval).", k))
    for gs, gl in (('0', '0'), ('0p5', '1/2'), ('2', '2')):
        k = key('filtered', 3, gs); c = cmp[k]; r = rc[k]; Q = ext['readouts'][k]; t = Q['termination']; a_ = aud[gs]
        ok = a_['ok'] and t['all_pass'] and t['status'] == 'complete' and c['max_dev_F'] <= 1e-12 and c['max_dev_purity'] <= 1e-12 and r['all_inside']
        jl.append((f'J-F-read-n3-g{gl}', f"Filtered new readouts n = 3, g = {gl} {HALO_SHORT}, run by busi with icloud's unchanged adapter: {'ACCEPTED' if ok else 'NOT ACCEPTED'} (intake audit {sum(a_['checks'].values())} of {len(a_['checks'])} checks; readout termination complete and all true; adapter F_e and purities equal busi's reduction to {f(max(c['max_dev_F'], c['max_dev_purity']), 2)}; every joint return and escape probability re-derived by busi inside the adapter's interval).", k))
    # conversion judgment
    cc = []
    for n in (2, 3):
        for gs, gl in GS:
            P = ext['physical'][key('filtered', n, gs)]; _, ncc, ntot, mx, _, _ = conv_summary(P)
            Pb = ext['physical'][key('baseline', n, gs)]; _, nccb, _, mxb, _, _ = conv_summary(Pb)
            cc.append((n, gl, ncc, ntot, mx['p_LL']['value'], mxb['p_LL']['value'], nccb))
    pos = [x for x in cc if x[1] != '0']; zer = [x for x in cc if x[1] == '0']
    allpos = all(x[2] == x[3] - 1 for x in pos); allzero = all(x[2] == 0 for x in zer)
    ratios = ', '.join(f'n = {x[0]}, g = {x[1]}: {f(x[4], 4)} vs {f(x[5], 4)}' for x in pos)
    above = []
    for x in pos:
        gs_ = [a for a, b in GS if b == x[1]][0]
        cf = {c['clock']: c for c in ext['physical'][key('filtered', x[0], gs_)]['conversion'] if not c['dephased']}
        cb = {c['clock']: c for c in ext['physical'][key('baseline', x[0], gs_)]['conversion'] if not c['dephased']}
        hi = [t for t in sorted(cf) if cf[t]['p_LL']['lower'] > cb[t]['p_LL']['upper']]
        lo = [t for t in sorted(cf) if cf[t]['p_LL']['upper'] < cb[t]['p_LL']['lower']]
        big = max(hi, key=lambda t: cf[t]['p_LL']['value'] - cb[t]['p_LL']['value']) if hi else None
        above.append((x[0], x[1], len(hi), len(lo), len(cf), big, cf, cb))
    abtxt = '; '.join(f"n = {a[0]}, g = {a[1]}: filtered above on-site at {a[2]} of {a[4]} clocks, below at {a[3]}" + (f" (largest excess at t = {f(a[5])}: {f(a[6][a[5]]['p_LL']['value'], 4)} against {f(a[7][a[5]]['p_LL']['value'], 4)})" if a[5] is not None else '') for a in above)
    nmin = min(a[2] for a in above); nmax = max(a[2] for a in above)
    jl.append(('J-F-conversion', f"Conversion {HALO_SHORT}: for every g > 0 cohort coherent conversion is resolved at {'every sampled clock except t = 0' if allpos else 'the counts in Table F7'}; at g = 0 {'no clock resolves conversion (zero control)' if allzero else 'see Table F7'}. The MAXIMUM sampled p_LL of the filtered control is below the on-site maximum at every (n, g > 0): filtered vs on-site {ratios}. Clock by clock this is not an ordering: the filtered p_LL exceeds the on-site value (disjoint intervals) at {nmin} to {nmax} of the 17 clocks per cohort: {abtxt}. Population plus coherence, sampled clocks only; no rate.", None))
    exc = [ind['cohorts'][key(a_, 3, gs)]['purification'] for a_ in ('filtered', 'baseline') for gs, _ in GS]
    none_est = not any(p['bounded_return_exclusion_finite'] or p['bounded_return_exclusion_ext'] for p in exc)
    minU = min(min(p['U_cond_finite'], p['U_cum_finite'], p['U_cond_ext'], p['U_cum_ext']) for p in exc)
    c2pos = all(ext['physical'][key(a_, 3, gs)]['purification']['primary_conditional']['lower'] > 0 and ext['physical'][key(a_, 3, gs)]['purification']['primary_cumulative']['lower'] > 0 for a_ in ('filtered', 'baseline') for gs, _ in GS)
    retpos = [(k, max(x['joint_return']['interval'][0] for x in Q['return'])) for k, Q in ext['readouts'].items() if k.startswith('filtered') and Q['return']]
    jl.append(('J-F-return-exclusion', f"Return exclusion {HALO_SHORT}: bounded-return-exclusion purification is {'NOT established in any n = 3 cohort, filtered or on-site, finite chain or empty extension' if none_est else 'established somewhere: see Table F9'} (smallest of the four upper bounds over all n = 3 cohorts {f(minU, 6)}; the criterion needs both bounds < 0); status return_or_precision_unresolved. The primary entropy increments c2 and D2 are {'strictly positive in every n = 3 cohort (no cut-entropy decrease at collection)' if c2pos else 'not all positive: see Table F9'}. n = 2: not applicable (one lead site). Two-time history over all eight filtered cohorts: the smallest lower endpoint of the per-cohort maximum joint return probability is {f(min(x[1] for x in retpos), 4)}, so returns to the source after an escape do occur with nonzero probability (Table F11; J-F-return-history).", None))
    hist = []
    for n in (2, 3):
        for gs, gl in GS:
            Q = ext['readouts'][key('filtered', n, gs)]
            esc = [x for x in Q['return'] if x['readout_clock'] == 1.0][0]['escape']['interval']
            mxr = max(Q['return'], key=lambda x: x['joint_return']['interval'][1])
            after = [x for x in Q['return'] if x['readout_clock'] > 1.0]
            pos_after = all(x['joint_return']['interval'][0] > 0 for x in after)
            cond = [x['conditional_interval'] for x in after if x['conditional_interval']]
            cmax = max(cond, key=lambda v: v[1]) if cond else None
            hist.append((n, gl, esc, mxr, pos_after, cmax, set(x['conditional_status'] for x in Q['return'])))
    allpos_after = all(h[4] for h in hist)
    jl.append(('J-F-return-history', f"Two-time (Lüders) return history {HALO_SHORT}, escape test t_e = 1, all eight filtered cohorts now present: " + '; '.join(f"n = {h[0]}, g = {h[1]}: p_escape {I(*h[2], 6)}, max joint p_return {I(*h[3]['joint_return']['interval'], 6)} at t = {f(h[3]['readout_clock'])}, max conditional {I(*h[5], 6) if h[5] else 'n/a'}" for h in hist) + f". Joint return lower endpoint strictly positive at {'every' if allpos_after else 'not every'} readout clock after t_e in every cohort: escapes are followed by returns to the source with nonzero probability, so no zero-return statement is supported. These are endpoint joint probabilities of a declared two-time measurement, not first-return distributions or rates.", None))
    allnd = all(r['status'] == 'not_decoupled_to_1e-4' for kk, P in ext['physical'].items() if kk.startswith('filtered') for r in P['register'])
    fmax = max(x['F_upper'] for kk, R in ind['cohorts'].items() if kk.startswith('filtered') for x in R['receiver'])
    jl.append(('J-F-recovery', f"Recovery and purification {HALO_SHORT}: {'every' if allnd else 'not every'} registered readout (both sizes, all g, clocks 0, 1, 2, 4, whole-lead light-only and complete output, and the n = 3 first block) is not_decoupled_to_1e-4; the largest F_e upper endpoint of the fixed local-pair receiver over all these readouts is {f(fmax, 9)}, maximum {f(fmax, 8)} (1/4 = failure value, 1 = perfect), so this receiver does not recover the code; this rejects only the declared receiver, not every recovery map. Purities in Table F13.", None))
    for i, (tag, txt, _) in enumerate(jl, 1):
        L.append(f'{i}. **{tag}.** {txt}')
    # sec 8 missing
    q3 = ext['readouts'][key('filtered', 3, '1')]['time']
    tb = open(f'{B}/icloud_tree_before.md5').read().split()[0]; ta = open(f'{B}/icloud_tree_after.md5').read().split()[0]
    L += ['', '## 8. Jobs busi ran after the coordinator ruling, and what was not run', '',
          f"1. Filtered n = 3, g = 0, 1/2, 2 retained-input audits and new-readout jobs (fixed receiver, pure-cut purity, Lüders two-time return history), never run by icloud, were run by busi with icloud's adapter UNCHANGED (source <internal>/run_mock_filtered_comparison_readouts_v1.py, md5 `{md5('<internal>/cw15_run_mock_filtered_comparison_readouts_v1_COPY.py')}`; byte-identical copy busi/code/clockwell/cw15_run_mock_filtered_comparison_readouts_v1_COPY.py), icloud's accepted adapter-control gate, and busi's own intake assessment (cw16, Table F6b) as the review the readout gate binds. The only change requested by the coordinator, the output path, could not be made in the code: the adapter's sha-bound runtime guard (cpu_runtime.py) and its output gate require outputs under <internal>/MOCK_filtered_comparison_readouts_v1, and editing either changes a frozen source sha so the adapter refuses to run. busi therefore ran it in a private user+mount namespace (busi/code/clockwell/cw15_launch_overlay.sh) where that directory is an overlay with icloud's directory as the read-only lower layer and <internal>/upper as the upper layer: every byte written lands in the busi directory, and paths recorded inside these journals read <internal>/NAME while the files physically are <internal>/NAME. icloud's two trees (MOCK_filtered_comparison_readouts_v1 and MOCK_filtered_physical_v1; every path, size and mtime) hash to `{tb}` before and `{ta}` after the jobs ({'unchanged' if tb == ta else 'CHANGED'}). Each adapter job (intake and readouts) ran with all six thread variables set to 1 (OMP, MKL, OPENBLAS, NUMEXPR, VECLIB, BLIS), one resident thread recorded in its termination row, nice 10, box CPU, no GPU; busi's own re-derivations (cw06, cw12) used 4 BLAS threads. Costs: " + '; '.join(f"g = {gl}: intake {f(aud[gs]['wall'], 4)} s, readouts {ext['readouts'][key('filtered', 3, gs)]['time']['wall']} wall ({ext['readouts'][key('filtered', 3, gs)]['time']['user_s']} s user, RSS {ext['readouts'][key('filtered', 3, gs)]['time']['rss_kib']} KiB)" for gs, gl in (('0', '0'), ('0p5', '1/2'), ('2', '2'))) + ". F_e and purity for these three cohorts keep the 'busi' label in Tables F12 and F13 (busi post-processing of the saved vectors); the adapter's own values agree (Table F6).",
          '2. n = 7 propagation: disabled by #145 (needs the #138 sec. 8 report and a declared collection plan). Not attempted.', '',
          'No other pending item needed a computation outside the saved outputs.', '']
    # sec 9 second reader
    L += ['## 9. For the second reader', '',
          f'All inputs are icloud box files under <internal>/solve_b_bulk on compute-host, read in place (read-only). The md5 of every file busi read is in `{MAN}` (md5 `{MANMD5}`). Each judgment below can be re-derived without busi scripts.', '']
    for i, (tag, txt, k) in enumerate(jl, 1):
        if tag.startswith('J-F-phys'):
            P = ext['physical'][k]
            L.append(f"{i}. **{tag}.** Inputs: {second_reader_inputs(k)}. Quantities: (a) the `termination` row: status complete, all_pass true, sources_unchanged true, checks equal to the number of `check` rows, every `check` row pass true; (b) for every `coherent_conversion` row with dephased false: load its `density.matrix.array` (276 x 276), label each basis row by its `row_keys` entry (register index, occupation bitmask; mode m has species (m mod 8) div 4, 0 = heavy), and recompute p_LL = trace over LL rows, C_HL = Frobenius norm of the HH x LL block; (c) for each primary `entropy` row: reduce the `_snapshot.raw_vector` by the graded partial trace (sign -1 when the traced mode precedes the kept mode) onto the row's `modes` and compute the von Neumann entropy; (d) for each `register` row: take the registered primary `action` raw vector (shape 2 x 276, R first) at that clock and recompute D_RK with K = complement of `output`; (e) from the `purification` row recompute U_cond = upper(protected_conditional) + F_137(min(1,B_second)) + F_37(min(1,B_second)), U_cum = upper(protected_conditional) + F_137(min(1,B_second)) + F_37(min(1,B_first)) (B_first, B_second, dimensions [137, 37] are fields of the purification row; h the natural-log binary entropy), with F_d(T) = T ln(d-1) + h(T) for T < 1 - 1/d and ln d otherwise; then again with B_first + Bwall_first and B_second + Bwall_second (extension); compare with finite_chain and empty_uniform_extension. Criterion: (a) all true and |(b)-(e) - saved| <= 1e-12. Also check the declared halo header string on the conversion, status, wall, purification and B2 rows and wall_complete.context.model.halo_values = ['1/2', '1'].")
        elif tag.startswith('J-F-read'):
            Q = ext['readouts'][k]
            extra = ''
            if Q.get('run_by'):
                gs_ = k.split('_g')[1].split('_v1')[0]; a_ = aud[gs_]
                extra = f" Also the busi-run intake `<internal>/written_report.md` (md5 `{md5(f'<internal>/written_report.md')}`, journal md5 `{md5(f'<internal>/records.ndjson')}`) and busi's assessment `{a_['review']}` (md5 `{md5(a_['review'])}`): check its termination row and the cw16 criteria listed in that file (sources identical to icloud's n = 3, g = 1 intake, 34 selected / 256 excluded Actions, parent journal sha256 unchanged)."
            L.append(f"{i}. **{tag}.** Inputs: `{Q['report']}` (md5 `{Q['report_md5']}`) and the physical job {second_reader_inputs(k)}.{extra} Quantities: F_e = sum_b |<Phi|v_b>|^2 + (1 - sum_b ||v_b||^2)/4 with v_b[r,0] = C[r,(m0,m1)], v_b[r,1] = (C[r,(m0,m3)] - C[r,(m1,m2)])/sqrt 2 for every included (species, lead site) branch, C the normalized registered vector, Phi = (|00>+|11>)/sqrt 2; purity Tr rho_A^2 of the pure input on the four/two fixed cuts; p_escape = ||Q_E psi(1)||^2 and p_return(t) = ||P_C exp(-iH(t-1)) Q_E psi(1)||^2 with H the first `numerical_action` operator array and P_C = basis states with a particle on site 0. Criterion: termination all true; |F_e - centre of icloud's fidelity_computed_interval| <= 1e-12 and |purity - centre of icloud unclipped interval| <= 1e-12 for every row; each recomputed probability inside icloud's interval (1e-12 slack).")
        elif tag == 'J-F-conversion':
            L.append(f"{i}. **{tag}.** Inputs: the eight filtered and eight on-site physical reports (paths and md5 in Table F5 here and Table B1 of the baseline file). Quantity: per clock, the `conversion_status` row's coherent_conversion flag, equivalently lower(p_LL) > 0 and lower(C_HL) > 0 in the `coherent_conversion` row; and max over the 17 clocks of p_LL value. Criterion: count of true flags per cohort; comparison of the two maxima per (n, g).")
        elif tag == 'J-F-return-exclusion':
            L.append(f"{i}. **{tag}.** Inputs: the `purification` row of each n = 3 physical report (filtered and on-site), the `wall_complete` row (bounds at prefix clocks 1 and 2), and the `return_probability` rows of the readout reports in <internal>/MOCK_filtered_comparison_readouts_v1 and MOCK_comparison_readouts_v1/ (md5s in Table F6 and the manifest). Quantity: the four upper bounds of item J-F-phys (e). Criterion: exclusion established only if both U_cond and U_cum < 0 (finite chain), and separately both with B + Bwall (extension); c2, D2 lower endpoints > 0; joint p_return lower endpoint > 0 at some t > 1.")
        elif tag == 'J-F-return-history':
            L.append(f"{i}. **{tag}.** Inputs: the `return_probability` rows of the eight filtered readout reports (icloud: <internal>/even_n2_g*_v1 and even_n3_g1_v1; busi-run: <internal>/even_n3_g{{0,0p5,2}}_v1; report md5s in Table F6) and the physical jobs (Table F5). Quantity: p_escape = ||Q_E psi(1)||^2 with psi(1) the saved primary pure vector at t = 1 and Q_E the projector on basis states with no particle on site 0 (modes 0-7); p_return(t) = ||P_C exp(-iH(t-1)) Q_E psi(1)||^2, t = 1, 5/4, ..., 4, H the first `numerical_action` operator array (check: exp(-iH) on the saved t = 0 vector reproduces the saved t = 1 vector); conditional = p_return / p_escape. Criterion: each recomputed value inside the saved interval (1e-12 slack); report whether every joint lower endpoint after t = 1 is > 0.")
        elif tag == 'J-F-recovery':
            L.append(f"{i}. **{tag}.** Inputs: the `register` rows of the eight filtered physical reports and the registered raw vectors (as in J-F-phys (d)). Quantity: D_RK interval and status per row; F_e by the formula in J-F-read on the registered vectors at clocks 0, 1, 2, 4 for light-only / complete whole-lead output and (n = 3, t = 1) site-2 output, interval +- min(1, eps) with eps the action's metadata.error. Criterion: not decoupled iff upper(D_RK) > 1e-4; report max F_e upper endpoint.")
    L += ['', '## 10. Independent second read and repairs', '',
          f"Second read: `{SR}` (md5 `{md5(SR)}`, copy of <internal>/MOCK_clockwell_secondread_v1.md), coordinator-run, different agent: every judgment AGREES; no number differs by more than its stated tolerance. Repairs made by regenerating with cw17_write_reads_v2.py (no hand edits): (1) J-F-conversion now states the ordering of the MAXIMUM sampled p_LL and the clock-by-clock counts where the filtered p_LL exceeds the on-site value; (2) J-F-recovery and J-B-recovery state the largest F_e upper endpoint to full digits instead of 'never exceeds'; (3) Tables F8 and B4 label the site-1 (B2) populations with their actual clock t1 = 1 (they were mislabelled t = 0.75); (4) the distance / switch-work sub-table under Table F9 (now Table F9b) carries the endpoint-copy halo header with both halo values; minor: thread counts per stage in secs. 1 and 8, and the adapter-agreement note in Table F12.", '']
    # sec 10 files
    L += ['', '## 11. Files (box <internal>/clockwell, MD5SUMS there) and code (busi/code/clockwell/)', '',
          '| product | md5 |', '|---|---|']
    for k_, p in J.items():
        L.append(f'| `{p}` | `{JMD5[k_]}` |')
    L.append(f'| `{MAN}` (input manifest) | `{MANMD5}` |')
    for gs in ('0', '0p5', '2'):
        for nm in (f'intake_even_n3_g{gs}_v1', f'even_n3_g{gs}_v1'):
            for fl in ('written_report.md', 'records.ndjson'):
                pp = f'<internal>/{fl}'
                L.append(f'| `{pp}` (busi-run adapter output) | `{md5(pp)}` |')
        L.append(f"| `{aud[gs]['review']}` (busi intake assessment) | `{md5(aud[gs]['review'])}` |")
    for pp in ('<internal>/cw15_launch_overlay.sh', '<internal>/cw15_run_mock_filtered_comparison_readouts_v1_COPY.py', '<internal>/cw16_intake_audit.py'):
        L.append(f'| `{pp}` | `{md5(pp)}` |')
    L += ['', 'Scripts (each a new file; final runs on the box with nice 10, timings in the *_time.txt files next to the products): cw01_inventory_kinds.py, cw02_schema_peek.py, cw03_keys.py, cw04_kinds_one.py (schema reads only); cw05_extract.py; cw06_independent.py; cw08_static_peek.py, cw10_check_names.py (schema reads only); cw09_static_halo.py; cw11_vertex_probes.py; cw12_return_check.py; cw13_input_manifest.py; cw14_write_reads.py (v1 writer), cw17_write_reads_v2.py (this file\'s writer); cw15_launch_overlay.sh with cw15_run_mock_filtered_comparison_readouts_v1_COPY.py (icloud adapter, byte-identical); cw16_intake_audit.py.', '']
    return '\n'.join(L) + '\n'

# ---------------------------------------------------------------- baseline document
def baseline_doc():
    L = []
    L += ['# MOCK unitary lead v1: on-site baseline read (#138 baseline; busi takeover of icloud)', '',
          'Final delivery after the coordinator-run independent second read (sec. 8). Author: busi (analyst engineer), lane busi, busi inbox #489. icloud had closed every on-site baseline scientific judgment with its original pair but delivered no baseline file; this file is that read, from icloud saved outputs only.', '',
          'Model: the gate\'s literal on-site converter U sum_x N_x W_x with N = (1/2, 1, ..., 1), no halo (the halo rule applies only to the filtered control), n = 2, 3, g = 0, 1/2, 1, 2 (g = 1 baseline), full N = 2 sector (and the N = 1 odd control), fixed no-flow background. No rate, chronology, source or flow is printed.', '']
    # cohort table
    rows = cohort_rows('baseline', (2, 3))
    L += ['## 1. Cohorts', '', '**Table B1. On-site cohorts: completion, controls and busi re-derivation** (deviation columns as in the filtered file, Table F5)', '',
          '| n | g | report md5 | predicates (all true) | sources unchanged | job wall / CPU s, RSS KiB | dev p_LL | dev C_HL | dev S | dev D_RK | dev U | busi verdict |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, gl, k, P, t, devs, ok in rows:
        L.append(f"| {n} | {gl} | `{P['report_md5']}` | {P['checks_total']} ({f(t['all_pass'])}) | {f(t['sources_unchanged'])} | {f(t['wall'], 6)} / {f(t['cpu'], 6)}, {t['rss_kib']} | " + ' | '.join(f(d, 2) for d in devs) + f" | {'accepted' if ok else 'NOT accepted'} |")
    for nm in ('odd_n2_v1', 'odd_n3_v1'):
        P = ext['physical']['baseline/' + nm]; t = P['termination']
        L.append(f"| odd {nm[5]} | all g (one common odd H) | `{P['report_md5']}` | {P['checks_total']} ({f(t['all_pass'])}) | {f(t['sources_unchanged'])} | {f(t['wall'], 6)} / {f(t['cpu'], 6)}, {t['rss_kib']} | | | | | | {'accepted' if t['all_pass'] and t['status'] == 'complete' else 'NOT accepted'} |")
    cmp = ind['compare_with_icloud_readouts']
    L += ['', '**Table B2. On-site new readouts (receiver, purity, two-time history)**', '',
          '| cohort | readout report md5 | termination all true | max |busi F_e - icloud| | max |busi purity - icloud| | busi return re-derivation inside icloud intervals |', '|---|---|---|---|---|---|']
    for k, Q in ext['readouts'].items():
        if not k.startswith('baseline'):
            continue
        c = cmp.get(k); r = rc.get(k)
        L.append(f"| {k.split('/')[1]} | `{Q['report_md5']}` | {f(Q['termination']['all_pass'])} | {f(c['max_dev_F'], 2) if c else 'odd: not recomputed'} | {f(c['max_dev_purity'], 2) if c else 'n/a'} | {f(r['all_inside']) if r else 'n/a'} |")
    # V_X probes
    recs = vp['records']
    L += ['', '## 2. Clock-weighted V_X matrix elements in the retained basis (#138 / run spec sec. 1.1), from the saved controls job', '',
          f"Controls report `{vp['report']}` md5 `{vp['md5']}`; saved probe checks: " + ', '.join(f'{k} {v} pass' for k, v in vp['pass'].items()) + (', failures ' + json.dumps(vp['fail']) if vp['fail'] else ', no failures') + '. Local amputated targets (gate analytic values): collinear -32/5 (other fourteen zero), spin average 512/25, threshold 0, intervalley 4, slope -16; the perpendicular 256/25 is outside this axial kinematics.', '',
          '| n | g | probe | raw contraction | target | normalized interval | raw = g S_N local_C exactly |', '|---|---|---|---|---|---|---|']
    lastng = (None, None)
    for x in recs:
        if x['kind'] == 'weighted_probe':
            lastng = (x['n'], x['g'])
            if x['raw'].get('re') not in ('0', None):
                nz = x['normalized']
                L.append(f"| {x['n']} | {x['g']} | {x['label']} | {x['raw']['re']} | {x['target']['re']} | {I(*nz['re']) if 're' in nz else 'n/a'} | {f(x['raw_eq_g_SN_localC'])} |")
        elif x['kind'] == 'weighted_tangent' and x['raw'] and x['raw'].get('re') not in ('0', None):
            L.append(f"| {lastng[0]} | {lastng[1]} | full tangent (raw derivative) | {x['raw']['re']} | {x['target']['re']} | n/a | n/a |")
    L += ['', 'Rows with zero raw contraction (all g = 0 rows and the fourteen zero collinear spin rows, threshold rows) are omitted; their saved checks pass with exact zero.', '',
          '## 3. Conversion', '']
    L += t_conversion(['baseline'], (2, 3), 'Table B3. On-site coherent conversion', False)
    L += ['']
    L += t_populations('baseline', (2, 3), 'Table B4. On-site populations at t = 4 and the later-collected block before collection', False)
    L += ['', '## 4. Return exclusion', '']
    L += t_purification(['baseline'], 'Table B5. On-site bounded-return-exclusion purification (n = 3)', False)
    L += ['']
    L += t_wall(['baseline'], (2, 3), 'Table B6. On-site wall bound (uniform empty lead, sqrt(<n_last>)/2 joining identity)', False)
    L += ['']
    L += t_return(['baseline'], (2, 3), 'Table B7. On-site two-time return probability', False)
    L += ['', '## 5. Recovery and purification', '']
    L += t_recovery('baseline', (2, 3), 'Table B8. On-site register decoupling and fixed-receiver fidelity', False)
    L += ['']
    L += t_purity('baseline', (2, 3), 'Table B9. On-site fixed-cut purities', False)
    # odd
    L += ['', '**Table B10. Odd (N = 1) code control, common to all g**', '', '| n | clock | output | D_RK interval | status | F_e interval (icloud) |', '|---|---|---|---|---|---|']
    for nm in ('odd_n2_v1', 'odd_n3_v1'):
        P = ext['physical']['baseline/' + nm]; Q = ext['readouts'].get('baseline/' + nm)
        for r in P['register']:
            fr = [x for x in (Q['receiver'] if Q else []) if x['clock'] == r['clock'] and x['choice'] == r['role'] and bool(x['first']) == bool(r['first_collected'])]
            L.append(f"| {nm[5]} | {f(r['clock'])} | {('first block ' if r['first_collected'] else 'whole lead ') + r['role']} | {iv(r['D'])} | {r['status']} | {I(*fr[0]['fidelity_interval']) if fr else 'n/a'} |")
    # judgments
    jl = []
    allok = all(x[6] for x in rows)
    jl.append(('J-B-cohorts', f"All eight even on-site cohorts {'accepted' if allok else 'NOT all accepted'} on completion, predicates, sources and busi re-derivation; both odd cohorts complete with all predicates true. (icloud's original pair had already closed these judgments; this is busi's read of the same saved outputs.)"))
    allid = all(x['raw_eq_g_SN_localC'] for x in recs if x['kind'] == 'weighted_probe')
    from fractions import Fraction as _Fr
    la = [x for x in vp['rows'] if x['name'].startswith('local_amputated')]
    la_vals = sorted(set(x['actual']['re'] for x in la if isinstance(x['actual'], dict) and x['actual'].get('re') != '0'))
    la_ok = all(x['pass'] and x['actual'] == x['target'] for x in la)
    jl.append(('J-B-vertex', f"V_X probes: every saved probe check passes ({'yes' if not vp['fail'] else 'NO'}) and every raw contraction equals g S_N(z) local_C exactly ({'all' if allid else 'NOT all'} {sum(1 for x in recs if x['kind'] == 'weighted_probe')} probe records); the {len(la)} local amputated rows equal their targets ({f(la_ok)}), nonzero values {', '.join(la_vals)}."))
    cc = []
    for n in (2, 3):
        for gs, gl in GS:
            _, ncc, ntot, mx, _, _ = conv_summary(ext['physical'][key('baseline', n, gs)])
            cc.append((n, gl, ncc, ntot, mx))
    jl.append(('J-B-conversion', 'Conversion: ' + '; '.join(f"n = {x[0]}, g = {x[1]}: {x[2]} of {x[3]} clocks, max p_LL {f(x[4]['p_LL']['value'], 4)} at t = {f(x[4]['clock'])}" for x in cc) + '. Coherent (C_HL lower bound positive) wherever conversion is resolved; sampled clocks only.'))
    exc = [ind['cohorts'][key('baseline', 3, gs)]['purification'] for gs, _ in GS]
    bret = {k: max(x['joint_return']['interval'][0] for x in Q['return'][1:]) for k, Q in ext['readouts'].items() if k.startswith('baseline/even') and Q['return']}
    allret = all(v > 0 for v in bret.values())
    jl.append(('J-B-return-exclusion', f"Return exclusion: bounded-return-exclusion purification {'is NOT established in any n = 3 on-site cohort' if not any(p['bounded_return_exclusion_finite'] or p['bounded_return_exclusion_ext'] for p in exc) else 'is established somewhere (Table B5)'}; smallest upper bound {f(min(min(p['U_cond_finite'], p['U_cum_finite'], p['U_cond_ext'], p['U_cum_ext']) for p in exc), 6)}; status return_or_precision_unresolved; the two-time joint return probability has a strictly positive lower endpoint at some clock after the escape test in {'every' if allret else 'not every'} even cohort ({len(bret)} cohorts; smallest such maximum lower endpoint {f(min(bret.values()), 4)}; Table B7)."))
    allnd = all(r['status'] == 'not_decoupled_to_1e-4' for kk, P in ext['physical'].items() if kk.startswith('baseline') for r in P['register'])
    fmax = max(x['F_upper'] for kk, R in ind['cohorts'].items() if kk.startswith('baseline') for x in R['receiver'])
    jl.append(('J-B-recovery', f"Recovery: {'every' if allnd else 'not every'} on-site registered readout (even and odd) is not_decoupled_to_1e-4; the largest fixed-receiver F_e upper endpoint is {f(fmax, 9)}; the odd light-only D_RK stays at the 3/4 control value (Table B10)."))
    L += ['', '## 6. Judgments', '']
    for i, (tag, txt) in enumerate(jl, 1):
        L.append(f'{i}. **{tag}.** {txt}')
    L += ['', '## 7. For the second reader', '',
          f'Inputs are icloud box files under <internal>/MOCK_unitary_lead_run_v1 and MOCK_comparison_readouts_v1/ on compute-host (report md5s in Tables B1 and B2; every file busi read is in `{MAN}`, md5 `{MANMD5}`).', '',
          '1. **J-B-cohorts.** Per even report: termination complete, all_pass, sources_unchanged, checks equal to the count of check rows; recompute p_LL and C_HL from each conversion density array (row_keys = register index, occupation bitmask; species = (m mod 8) div 4), entropies from the primary raw vectors by graded partial trace, D_RK from the registered raw vectors (shape 2 x 276 or 2 x 120, R first), and U_cond, U_cum from the purification row (formula in the filtered file, sec. 9). Criterion: all deviations <= 1e-12.',
          f"2. **J-B-vertex.** Input `{vp['report']}` (md5 `{vp['md5']}`), rows of kind weighted_probe / weighted_tangent and checks local_amputated_* / weighted_probe_*. Quantity: raw, S_N, local_C, target. Criterion: raw == g * S_N * local_C exactly; raw == target; local amputated == -32/5 (collinear diagonal), 4 (valley rows).",
          '3. **J-B-conversion.** As J-F-conversion of the filtered file, on the on-site reports.',
          '4. **J-B-return-exclusion.** As J-F-return-exclusion, on the on-site n = 3 purification rows and the MOCK_comparison_readouts_v1 return rows.',
          '5. **J-B-recovery.** As J-F-recovery, on the on-site register rows (even and odd) and the MOCK_comparison_readouts_v1 receiver rows.', '',
          '## 8. Independent second read', '', f"Second read `{SR}` (md5 `{md5(SR)}`): every judgment AGREES. Repairs regenerated by cw17_write_reads_v2.py: largest F_e upper endpoint stated to full digits (J-B-recovery); Table B4 site-1 column labelled with its actual clock t1 = 1.", '',
          '## 9. Files', '', 'Same busi products and scripts as the filtered file (sec. 10 there): box <internal>/clockwell with MD5SUMS; code busi/code/clockwell/.', '']
    return '\n'.join(L) + '\n'

for name, fn in (('MOCK_filtered_clockwell_v1.md', filtered_doc), ('MOCK_unitary_lead_baseline_v1.md', baseline_doc)):
    p = f'{B}/{name}'
    open(p, 'w').write(fn())
    print('wrote', p, md5(p))
