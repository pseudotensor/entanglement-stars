"""cw12: verification of icloud's SAVED two-time return-probability records (addendum sec. 3) for the cohorts
where icloud's readout job exists. Read-only on <internal>/solve_b_bulk.

Route (independent of icloud's adapter): take the saved primary Hamiltonian array (first 'numerical_action'
operator record of the physical job), check it against the saved trajectory (exp(-iH) applied to the saved
t=0 pure vector reproduces the saved t=1 vector), then form eta = Q_E psi(1) with busi's own site-0 mask
from the basis bitmasks, propagate eta by busi's own eigh of the saved H, and compare p_escape and the joint
p_return(t) at every saved readout clock with icloud's saved intervals.
This is a re-derivation of saved readouts; it is NOT run for cohorts whose readout job is missing.
Output: <internal>/cw12_return_check.json
"""
import json, os, time
T0 = time.time()
import numpy as np

ROOT = '<internal>/solve_b_bulk'
EXT = '<internal>/cw05_extract.json'
OUT = '<internal>/cw12_return_check.json'

def load(d, a):
    z = np.load(os.path.join(d, a)); return z[z.files[0]]

ext = json.load(open(EXT))
res = {}
for key, Q in ext['readouts'].items():
    if '/even_' not in key or not Q['return']:
        continue
    P = ext['physical'][key]; n = P['n']; d = f"{ROOT}/{P['top']}/{P['name']}"
    Hrec = None; states = {}
    with open(P['report']) as f:
        for line in f:
            if Hrec is None and line.startswith('{"kind":"numerical_action"'):
                r = json.loads(line)['data']
                if r.get('kind') == 'operator':
                    Hrec = r['hamiltonian']['array']
            if line.startswith('{"kind":"action"'):
                r = json.loads(line)['data']; m = r['metadata']
                if m.get('plan') == P['plans']['pure'] and m.get('role') == 'primary' and m.get('start') == 0.0 and m.get('end') in (0.0, 1.0):
                    states.setdefault(m['end'], r['raw_vector']['array'])
    H = load(d, Hrec)
    keys = []
    for rr, mask in P['conversion'][0]['row_keys']:
        keys.append([b for b in range(8*n) if (mask >> b) & 1])
    inC = np.array([any(m < 8 for m in ms) for ms in keys])
    w, V = np.linalg.eigh((H + H.conj().T) / 2)
    psi0 = load(d, states[0.0]).ravel(); psi1 = load(d, states[1.0]).ravel()
    prop = lambda t, v: V @ (np.exp(-1j * w * t) * (V.conj().T @ v))
    traj_dev = float(np.linalg.norm(prop(1.0, psi0) - psi1))
    psi1n = psi1 / np.linalg.norm(psi1)
    eta = np.where(inC, 0, psi1n)
    pesc = float(np.vdot(eta, eta).real)
    rows = []
    for q in Q['return']:
        t = q['readout_clock']
        phi = prop(t - 1.0, eta)
        pret = float(np.sum(np.abs(phi[inC])**2))
        lo, hi = q['joint_return']['interval']; elo, ehi = q['escape']['interval']
        rows.append({'t': t, 'p_return_busi': pret, 'icloud_interval': [lo, hi],
                     'inside_or_within_1e-12': bool(lo - 1e-12 <= pret <= hi + 1e-12),
                     'p_escape_busi': pesc, 'icloud_escape': [elo, ehi], 'escape_inside': bool(elo - 1e-12 <= pesc <= ehi + 1e-12),
                     'conditional_busi': pret / pesc if pesc > 0 else None, 'icloud_conditional': q['conditional_interval'],
                     'icloud_status': q['conditional_status']})
    res[key] = {'H_array': Hrec, 'H_dim': int(H.shape[0]), 'trajectory_check_t1': traj_dev,
                'all_inside': all(x['inside_or_within_1e-12'] and x['escape_inside'] for x in rows), 'rows': rows}
    print(key, 'traj', f'{traj_dev:.2e}', 'escape', f'{pesc:.6f}', 'all inside', res[key]['all_inside'],
          'max p_return', max(x['p_return_busi'] for x in rows), flush=True)
res['_elapsed_wall_s'] = time.time() - T0
json.dump(res, open(OUT, 'w'), indent=1)
