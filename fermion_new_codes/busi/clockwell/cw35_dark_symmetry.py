"""cw35: script-emitted checks for the dark-codeword theorem (#491 A(c)), busi, own builder cw32.

Theorem (proved in the v1.1 text): let q = diag(+1,-1,-1,+1) per site on the four components (single-particle
charge sigma_z x sigma_z) and r = 1 x sigma_x (single-particle unitary c0<->c1, c2<->c3), applied to every mode
of both species. Then (1) dGamma(h) commutes with Q = dGamma(q) and with R = Gamma(r), for every site profile N,
mass and size n; (2) the converter changes Q only by 0 or +-4 (K is nonzero only between pair channels whose
charges are equal or opposite with |q| = 2), so Q mod 4 is conserved by H; (3) K's columns for heavy pairs 03 and
12 are equal, so on the |Q| = 2 heavy sector the heavy factor of every converter monomial (on-site or filtered,
any stencil, kernel or halo weight) is proportional to H_03 + H_12, which is R-even; D is R-odd with |Q| = 2, so
V exp(-iH0 t) D = 0 for all t and D never acquires light; (4) B = |01>_H has Q = 0, so the B and D branches stay
in different Q mod 4 sectors: their O-vacuum components are orthogonal (c = 0) for every output O.
Consequence for any light output O (whole-lead light, or all light): F* = 1/4 + (1 - p0)/4, p0 = probability that
the B branch has no particle in O; hence F* <= 1/2 at every n, clock, form and coupling.
Checks here: (1)-(2) on the assembled matrices for n = 2, 3, 5, 7, both forms, g = 1 (commutator norms; Q-mod-4
block-off-diagonal norm), (3) the K columns, and the closed form against the 24 certified cw27 values.
Output: <internal>/cw35_dark_symmetry.json"""
import json, sys
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M

qc = np.array([1, -1, -1, 1])
res = {'K_col03_equals_col12': bool(np.array_equal(M.K[:, 2], M.K[:, 3])), 'checks': []}
res['K_charge_rule'] = all(M.K[P, Q] == 0 or (qc[a] + qc[b]) in ((qc[c] + qc[d]), -(qc[c] + qc[d]))
                           for P, (a, b) in enumerate(M.PAIR_ORDER) for Q, (c, d) in enumerate(M.PAIR_ORDER))
rperm = {0: 1, 1: 0, 2: 3, 3: 2}
for n in (2, 3, 5, 7):
    for form in ('onsite', 'filtered'):
        H, B = M.assemble(n, 1.0, form)
        H0, _ = M.assemble(n, 0.0, form)
        idx = {p: i for i, p in enumerate(B)}
        Qd = np.array([qc[a % 4] + qc[b % 4] for a, b in B])
        Rm = np.zeros((len(B), len(B)))
        for (a, b), i in idx.items():
            a2 = (a // 4) * 4 + rperm[a % 4]; b2 = (b // 4) * 4 + rperm[b % 4]
            j = idx[(min(a2, b2), max(a2, b2))]
            Rm[j, i] = 1.0 if a2 < b2 else -1.0
        comm_Q_H0 = float(np.abs(np.diag(Qd) @ H0 - H0 @ np.diag(Qd)).max())
        comm_R_H0 = float(np.abs(Rm @ H0 - H0 @ Rm).max())
        mod4 = (Qd % 4)
        off = float(np.abs(H[np.ix_(mod4 == 0, mod4 == 2)]).max())
        res['checks'].append(dict(n=n, form=form, comm_Q_H0=comm_Q_H0, comm_R_H0=comm_R_H0, H_offblock_Qmod4=off))
        print(n, form, 'max|[Q,H0]|', comm_Q_H0, 'max|[R,H0]|', comm_R_H0, 'max|H(Q=0 mod 4, Q=2 mod 4)|', off, flush=True)
# closed form vs certified cw27
sdp = json.load(open('<internal>/cw27_sdp.json'))
cf = []
for s in sdp['states']:
    for on, o in s['outputs'].items():
        w0 = o['blocks']['0']['weight']        # = (p0 + 1)/2 (R-traced O-vacuum weight)
        p0 = 2 * w0 - 1
        F = 0.25 + (1 - p0) / 4
        inside = o['F_lower_saved_state'] - 1e-9 <= F <= o['F_upper_saved_state'] + 1e-9
        cf.append(dict(arm=s['arm'], g=s['g'], t=s['t'], output=on, p0=p0, closed_form=F, cert=[o['F_lower_saved_state'], o['F_upper_saved_state']], inside=inside))
res['closed_form_vs_cw27'] = cf
res['closed_form_all_inside'] = all(x['inside'] for x in cf)
print('K col 03 == col 12:', res['K_col03_equals_col12'], 'K charge rule:', res['K_charge_rule'], 'closed form inside all 24 certified intervals:', res['closed_form_all_inside'])
json.dump(res, open('<internal>/cw35_dark_symmetry.json', 'w'), indent=1)
