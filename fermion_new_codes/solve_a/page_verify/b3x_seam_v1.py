#!/usr/bin/env python3
"""b3x_seam_v1.py (solve, page_verify B3 light kit): the seam a page_history chain has where the record's engine sites end and v6q's
appended vacuum exterior (N = 1, v = -(2 g E_k / R)^(1/2)) begins: the record's applied N, v at its last sites against that exterior, for
B3_ext200 (2048 sites) and B2 (4096 sites) at common clocks; and the relative departure over windows. Reads only.
    python3 b3x_seam_v1.py OUT.json"""
import json, sys
import numpy as np

RUNS = {'B3_ext200 (2048)': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_ext200_r100',
        'B2 (4096)': '<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096'}


def main():
    out = {}
    for lab, p in RUNS.items():
        J = json.load(open(p + '.json')); Z = np.load(p + '_fields.npz', allow_pickle=True)
        g = float(J['args']['g']); rs = float(J['args']['rs'])
        byT = {round(float(r['T']), 6): r for r in J['records'] if r.get('E') is not None}
        T = np.asarray(Z['T'], float); R = np.asarray(Z['R'], float); N = np.asarray(Z['N_applied'], float); v = np.asarray(Z['v_applied'], float)
        rows = []
        for t in (40.0, 60.0, 80.0, 100.0, 150.0, 200.0):
            k = int(np.argmin(np.abs(T / rs - t)))
            if abs(T[k] / rs - t) > 1e-6:
                continue
            E = float(byT[round(float(T[k]), 6)]['E'])
            vv = -np.sqrt(2 * g * E / R)
            ent = {'T_over_rs': t, 'E': E, 'last_site': float(R[-1]), 'N_last_minus_1': float(N[k, -1] - 1.0), 'v_last': float(v[k, -1]), 'v_vac_last': float(vv[-1]),
                   'v_last_rel_dep': float(v[k, -1] / vv[-1] - 1.0), 'R_sh': byT[round(float(T[k]), 6)].get('R_sh')}
            for lo, hi in ((16, 32), (24, 32), (32, 64), (48, 64)):
                m = (R >= lo * rs) & (R <= hi * rs)
                if m.any():
                    ent['window_%d_%d' % (lo, hi)] = {'max_dN': float(np.max(np.abs(N[k, m] - 1.0))), 'max_rel_dv': float(np.max(np.abs(v[k, m] / vv[m] - 1.0)))}
            rows.append(ent)
        out[lab] = rows
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
    for lab, rows in out.items():
        for e in rows:
            print(lab, e['T_over_rs'], 'R_sh', e['R_sh'], 'seam N-1 %.2e v/vvac-1 %+.3e' % (e['N_last_minus_1'], e['v_last_rel_dep']),
                  {k: '%.1e/%.1e' % (x['max_dN'], x['max_rel_dv']) for k, x in e.items() if k.startswith('window')})


if __name__ == '__main__':
    main()
