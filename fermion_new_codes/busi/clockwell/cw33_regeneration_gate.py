"""cw33: reviewer A's regeneration gate for the n = 5 preparation (#491 (iii)), busi.
busi's own builder (cw32, no icloud code) must reproduce, for n = 3, both forms, g = 0 and 1:
  (a) the saved Hamiltonian array entrywise (diagnostic: max |H_busi - H_saved|);
  (b) the 12 saved registered states (t = 1, 2, 4) within the addendum tolerance: ||psi_busi - psi_saved|| <= eps_saved
      (eps_saved = the saved action's state-error radius; busi's own propagation error is ~1e-13);
  (c) the saved p_LL of the pure input at t = 1, 2, 4 (inside the saved interval);
  (d) the saved D_RK (light-only and complete whole-lead output) at t = 1, 2, 4 (inside the saved interval);
  (e) the saved protected conditional entropy Cq at t2 = 2 (inside the saved interval), protection = delete every
      monomial whose mode set meets both site 2 and its complement, from t1 = 1.
STOP flag if any fails. Output: <internal>/cw33_regeneration.json
"""
import json, os, sys, time, resource
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M

ROOT = '<internal>/solve_b_bulk'
EXT = '<internal>/cw05_extract.json'
OUTD = '<internal>/n5prep'
os.makedirs(OUTD, exist_ok=True)

def load(d, a):
    z = np.load(os.path.join(d, a)); return z[z.files[0]]

def main():
    t0 = time.process_time()
    ext = json.load(open(EXT))
    res = {'cohorts': [], 'stop': False}
    n = 3
    for arm, form in (('baseline', 'onsite'), ('filtered', 'filtered')):
        for gs, g in (('0', 0.0), ('1', 1.0)):
            k = f'{arm}/even_n3_g{gs}_v1'; P = ext['physical'][k]; d = f"{ROOT}/{P['top']}/{P['name']}"
            saved_keys = [tuple(b for b in range(24) if (mask >> b) & 1) for _, mask in P['conversion'][0]['row_keys']]
            H, B = M.assemble(n, g, form)
            same_basis = (saved_keys == B)
            Hs = None; reg = {}; pure = {}
            with open(P['report']) as f:
                for line in f:
                    if Hs is None and line.startswith('{"kind":"numerical_action"'):
                        r = json.loads(line)['data']
                        if r.get('kind') == 'operator':
                            Hs = load(d, r['hamiltonian']['array'])
                    if line.startswith('{"kind":"action"'):
                        r = json.loads(line)['data']; m = r['metadata']
                        if m.get('role') == 'primary' and m.get('start') == 0.0 and m.get('end') in (1.0, 2.0, 4.0):
                            if m.get('plan') == P['plans']['registered']:
                                reg[m['end']] = (load(d, r['raw_vector']['array']).reshape(2, -1), m['error'])
                            elif m.get('plan') == P['plans']['pure']:
                                pure[m['end']] = (load(d, r['raw_vector']['array']).ravel(), m['error'])
            dH = float(np.abs(H - Hs).max())
            psi0, reg0 = M.initial_states(B)
            row = dict(arm=arm, g=gs, cohort=k, report_md5=P['report_md5'], basis_identical=same_basis, max_abs_dH=dH, clocks=[])
            ok = same_basis and dH < 1e-12
            for t in (1.0, 2.0, 4.0):
                rb = np.vstack([M.propagate(H, reg0[0], t), M.propagate(H, reg0[1], t)])
                rs, eps = reg[t]
                dpsi = float(np.linalg.norm(rb - rs))
                pb = M.propagate(H, psi0, t)
                pHH, pLL, pHL = M.p_species(pb, B)
                cs = [c for c in P['conversion'] if c['clock'] == t and not c['dephased']][0]['p_LL']
                pin = cs['lower'] <= pLL <= cs['upper']
                dr = {}
                for role in ('light', 'all'):
                    out = P['plans']['outputs'][role]
                    Dv = M.D_RK(rb, B, out, n)
                    sv = [x for x in P['register'] if x['clock'] == t and x['role'] == role and not x['first_collected']][0]['D']
                    dr[role] = dict(busi=Dv, saved=[sv['lower'], sv['upper']], inside=sv['lower'] <= Dv <= sv['upper'])
                cok = dpsi <= eps and pin and all(v['inside'] for v in dr.values())
                ok = ok and cok
                row['clocks'].append(dict(t=t, state_distance=dpsi, eps_saved=eps, state_ok=dpsi <= eps, pLL_busi=pLL,
                                          pLL_saved=[cs['lower'], cs['upper']], pLL_inside=pin, D_RK=dr, ok=cok))
                print(f"{arm:8s} g={gs} t={t}: |dpsi|={dpsi:.2e} (eps {eps:.2e}) pLL {pLL:.10f} in {pin} D_L {dr['light']['busi']:.10f} {dr['light']['inside']} D_all {dr['all']['busi']:.10f} {dr['all']['inside']}", flush=True)
            # Cq
            site2 = set(range(16, 24))
            keep = lambda modes: (set(modes) <= site2) or not (set(modes) & site2)
            Hq, _ = M.assemble(n, g, form, keep=keep)
            p1 = M.propagate(H, psi0, 1.0)
            q2 = M.propagate(Hq, p1, 1.0)
            rhoA2, _ = M.reduce_rho(q2.reshape(1, -1), B, list(range(8, 24)))
            rhoA1, _ = M.reduce_rho(q2.reshape(1, -1), B, list(range(16, 24)))
            Cq = M.vn(rhoA2) - M.vn(rhoA1)
            sq = P['purification']['protected_conditional']
            cin = sq['lower'] <= Cq <= sq['upper']
            ok = ok and cin
            row.update(Cq_busi=Cq, Cq_saved=[sq['lower'], sq['upper']], Cq_inside=cin, admitted=ok)
            print(f"{arm:8s} g={gs}: max|dH|={dH:.2e} basis {same_basis} Cq {Cq:.10f} in {cin} -> {'REPRODUCED' if ok else 'FAIL'}", flush=True)
            res['cohorts'].append(row); res['stop'] = res['stop'] or not ok
    res['cpu_s'] = time.process_time() - t0; res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump(res, open(f'{OUTD}/cw33_regeneration.json', 'w'), indent=1)
    print('GATE', 'FAIL: STOP' if res['stop'] else 'PASS: all 12 states, p_LL, D_RK, Cq reproduced')

if __name__ == '__main__':
    main()
