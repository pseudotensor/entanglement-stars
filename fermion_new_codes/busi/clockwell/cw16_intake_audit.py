"""cw16: busi's audit of the three new retained-input intake jobs (filtered n = 3, g = 0, 1/2, 2), run with
icloud's unchanged adapter under the cw15 overlay, against icloud's own accepted n = 3, g = 1 intake.
Writes one assessment file per cohort (the review the adapter's readout gate binds: it must name the intake
journal sha256, report sha256 and the intake tuple/source binding) and prints their sha256.
Read-only on <internal>/solve_b_bulk. Outputs: <internal>/reviews_intake_v1
"""
import json, hashlib, os, collections, sys
OV = '<internal>/upper'
REF = '<internal>/intake_even_n3_g1_v1'
PHYS = '<internal>/MOCK_filtered_physical_v1'
OUT = '<internal>/reviews_intake_v1'
os.makedirs(OUT, exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def rows(d):
    return [json.loads(l) for l in open(f'{d}/records.ndjson')]

def single(rs, kind):
    x = [r['data'] for r in rs if r['kind'] == kind]
    assert len(x) == 1, (kind, len(x))
    return x[0]

def names(src):
    return {os.path.basename(p): s for p, s in src.items()}

ref = rows(REF); refc = single(ref, 'filtered_saved_intake_complete'); reft = single(ref, 'termination')
refctl = single(ref, 'accepted_filtered_adapter_controls')
summary = {}
for gl, g, psha in (('0', 0.0, 'ef6b41f742dde756fe5e1cce0e437daf514ef8efa3b355c0b20c4e80e309ac96'),
                    ('0p5', 0.5, 'd76b3a2a6db4383d65075e92f5b9567f0e480804ee745832babbbf176f65060f'),
                    ('2', 2.0, 'e3d9ee09f57014fca3278ff7e2c4438ac2fd649a4f909e56636e9df817a0697b')):
    d = f'{OV}/intake_even_n3_g{gl}_v1'
    rs = rows(d); c = single(rs, 'filtered_saved_intake_complete'); t = single(rs, 'termination'); i = single(rs, 'intake')
    ctl = single(rs, 'accepted_filtered_adapter_controls')
    jsha = sha(f'{d}/records.ndjson'); rsha = sha(f'{d}/written_report.md')
    parent_now = sha(f'{PHYS}/even_n3_g{gl}_v1/records.ndjson')
    checks = collections.OrderedDict()
    checks['termination complete, error none, all_pass, sources_unchanged, runtime_admitted, one thread'] = (
        t['status'] == 'complete' and t['error'] is None and t['all_pass'] is True and t['sources_unchanged'] is True
        and t['runtime_admitted'] is True and t['threads'] == 1)
    checks['terminal row is the unique last row'] = rs[-1]['kind'] == 'termination' and sum(r['kind'] == 'termination' for r in rs) == 1
    checks['record-kind roster identical to icloud n3 g1 intake'] = collections.Counter(r['kind'] for r in rs) == collections.Counter(r['kind'] for r in ref)
    checks['intake sources == termination sources'] = i['sources'] == t['sources']
    checks['source basenames and sha256 identical to icloud n3 g1 intake (39 files)'] = names(t['sources']) == names(reft['sources'])
    checks['accepted adapter-control gate identical to icloud n3 g1 (journal, report, review sha256)'] = all(
        ctl[k] == refctl[k] for k in ('journal_sha256', 'report_sha256', 'review_sha256'))
    checks['n = 3, g, parent journal sha256 as requested'] = c['n'] == 3 and c['g'] == g and c['parent_journal_sha256'] == psha
    checks['parent journal unchanged now (sha256 recomputed)'] = parent_now == psha
    checks['new_observables_evaluated is False'] = c['new_observables_evaluated'] is False
    checks['34 selected original Actions: 17 pure + 17 registered, clocks j/4'] = (
        sorted(map(tuple, c['actual_action_roster'])) == sorted([('pure', j/4) for j in range(17)] + [('registered', j/4) for j in range(17)]))
    checks['256 excluded original wall-midpoint Actions'] = len(c['excluded_wall_actions']) == 256 and all(
        'wall midpoint' in x['reason'] for x in c['excluded_wall_actions'])
    checks['locator count equal to icloud n3 g1'] = len(c['locators']) == len(refc['locators'])
    checks['physical producer source roster identical to icloud n3 g1 (30 files)'] = c['actual_sources'] == refc['actual_sources']
    checks['original role plans recorded for pure and registered'] = set(c['original_role_plans']) == {'pure', 'registered'}
    ok = all(checks.values())
    binding = c['intake_binding']
    text = [f'# busi assessment of the filtered retained-input intake, n = 3, g = {gl.replace("0p5", "1/2")} (endpoint-copy halo; N_-1 = 1/2, N_n = 1)', '',
            'Assessor: busi (analyst engineer), busi inbox #489, taking over icloud #145 under the coordinator ruling that the never-run return history is in scope.',
            'Job: icloud adapter run_mock_filtered_comparison_readouts_v1.py (sha256 d1b35e411f9ef950830fd6561e09e94df7995a56a591430503554499090728a8), unchanged, stage intake, run in a private overlay namespace (busi/code/clockwell/cw15_launch_overlay.sh); bytes physically in ' + d + '.', '',
            f'Intake journal sha256: {jsha}', f'Intake written report sha256: {rsha}', f'Intake tuple/source binding: {binding}',
            f'Parent physical journal sha256: {psha}', '', '## Checks (each against the actual intake rows; reference = icloud\'s accepted n = 3, g = 1 intake)', '']
    for k, v in checks.items():
        text.append(f'- {"PASS" if v else "FAIL"}: {k}')
    text += ['', f'Runtime: wall {t["wall"]:.3f} s, CPU {t["cpu"]:.3f} s, RSS {t["rss_kib"]} KiB, one thread, nice 10, box CPU.', '',
             f'**Disposition: {"accepted" if ok else "NOT accepted"}.** Scope: retained own physical model/code/source/action/eigensystem association only; no new observable was evaluated by the intake. This assessment admits the fixed new-readout job for this cohort under the same adapter.', '']
    p = f'{OUT}/busi_intake_review_n3_g{gl}_v1.md'
    open(p, 'w').write('\n'.join(text))
    summary[gl] = dict(ok=ok, review=p, review_sha256=sha(p), journal_sha256=jsha, report_sha256=rsha, binding=binding,
                       wall=t['wall'], cpu=t['cpu'], rss_kib=t['rss_kib'], checks={k: bool(v) for k, v in checks.items()})
    print(gl, 'accepted' if ok else 'NOT', p, summary[gl]['review_sha256'], 'fails:', [k for k, v in checks.items() if not v])
json.dump(summary, open('<internal>/cw16_intake_audit.json', 'w'), indent=1)
