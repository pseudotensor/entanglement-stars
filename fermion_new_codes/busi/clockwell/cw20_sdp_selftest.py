"""cw20: known-answer tests of cw19's SDP and certification on manufactured states (run before cw19).
  (a) rho_RO = |Phi><Phi| (O a perfect qubit copy, d_O = 2): F* = 1;
  (b) rho_RO = I_R/2 x rho_O (no information): F* = 1/4 for random rho_O (d_O = 5);
  (c) rho_RO = |Phi_rot><Phi_rot| with a local unitary on O: F* = 1 (decoder must undo it);
  (d) classical correlation (|00><00| + |11><11|)/2: F* = 1/2.
Each must have certified [LB, UB] containing the known value with gap < 1e-6.
Output: <internal>/cw20_selftest.json"""
import sys, json
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw19_recovery_sdp as S
from flint import arb

rng = np.random.default_rng(7)
res = {}
def run(name, C, d, target):
    rho = C @ C.conj().T
    sigma, W, info = S.solve_block(rho, d)
    Z, prim = S.W_to_Z(W, rho, 2 * d)
    cert = S.certify_block(C, d, arb(1), sigma, Z)
    LB = float(cert['LB'].lower()); UB = float(cert['UB'].upper())
    ok = LB <= target + 1e-12 and UB >= target - 1e-12 and UB - LB < 1e-6 and cert['psd_certified']
    res[name] = dict(LB=LB, UB=UB, target=target, ok=ok, solver=info)
    print(name, LB, UB, target, 'OK' if ok else 'FAIL')
# rows of C index (r, o) r-major; columns = purifying system
phi = np.zeros((4, 1), dtype=complex); phi[0, 0] = phi[3, 0] = 1 / np.sqrt(2)
run('a_perfect_copy', phi, 2, 1.0)
X = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)); sq = X / np.linalg.norm(X)
C = np.zeros((10, 10), dtype=complex); C[:5, :5] = sq / np.sqrt(2); C[5:, 5:] = sq / np.sqrt(2)
run('b_no_information', C, 5, 0.25)
U, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
M = np.zeros((4, 1), dtype=complex)
for r in range(2):
    e = np.zeros(2); e[r] = 1
    M[r * 2:(r + 1) * 2, 0] = (U @ e) / np.sqrt(2)
run('c_rotated_copy', M, 2, 1.0)
C = np.zeros((4, 2), dtype=complex); C[0, 0] = C[3, 1] = 1 / np.sqrt(2)
run('d_classical', C, 2, 0.5)
json.dump(res, open('<internal>/cw20_selftest.json', 'w'), indent=1)
print('ALL OK' if all(v['ok'] for v in res.values()) else 'SELFTEST FAILED')
