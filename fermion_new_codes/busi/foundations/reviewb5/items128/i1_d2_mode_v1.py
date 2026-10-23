#!/usr/bin/env python
"""i1_d2_mode_v1.py (busi, foundations reviewb5 item 1; agent A). READ script, box CPU, seconds.

The near-critical direction of the PM1 branch Jacobian at d = 2: K_d = chi_d A_d^-1, its eigenvalues closest to -1 for d = 4, 3, 2,
the make-up of the eigenvector (energy vs momentum test rows, radial vs tangential J, radius) and of the field response
A_d^-1 x (clock vs flow bumps), and two float stability checks of sigma_min(I + K_2): K from a linear solve instead of the
explicit inverse, and chi perturbed by its own float noise scale (relative 1e-12, random sign). Output: <out>.json."""
import json
import os

import numpy as np

D = "<internal>/items128"


def main():
    C = np.load(os.path.join(D, "i1_chi_v1_d4_d3_d2.npz"))
    I2 = np.load(os.path.join(D, "i2_projected_v1.npz"))
    res = {"version": "i1_d2_mode_v1"}
    for d in (4, 3, 2):
        chi = C["d%d_chi" % d]; A = I2["A_d%d_b+0" % d]; Y = C["d%d_Y" % d]
        K = chi @ np.linalg.inv(A)
        Ks = np.linalg.solve(A.T, chi.T).T
        ev, V = np.linalg.eig(K)
        k = int(np.argmin(np.abs(1 + ev)))
        x = V[:, k]; x = x / np.linalg.norm(x)
        xb = x.reshape(-1, 4)
        r = np.sqrt((Y ** 2).sum(axis=1)); yh = np.where(r[:, None] > 0, Y / np.where(r[:, None] > 0, r[:, None], 1), 0)
        Jr = (xb[:, 1:] * yh).sum(axis=1)
        c = np.linalg.solve(A, x).reshape(-1, 4)
        cr = (c[:, 1:] * yh).sum(axis=1)
        w_r = np.abs(xb) ** 2
        rng = np.random.default_rng(3)
        chi_p = chi * (1 + 1e-12 * rng.choice([-1, 1], size=chi.shape))
        ent = {"eig_closest_to_minus1": [float(ev[k].real), float(ev[k].imag)], "min_abs_1_plus_eig": float(np.abs(1 + ev[k])),
               "n_eigs_below_minus_half": int(np.sum(ev.real < -0.5)), "most_negative_real_eig": float(ev.real.min()),
               "eigvec_share_rho_rows": float((np.abs(xb[:, 0]) ** 2).sum()), "eigvec_share_J_radial": float((np.abs(Jr) ** 2).sum()),
               "eigvec_mean_radius": float((w_r.sum(axis=1) * r).sum()),
               "field_response_share_N": float((np.abs(c[:, 0]) ** 2).sum() / (np.abs(c) ** 2).sum()),
               "field_response_share_v_radial": float((np.abs(cr) ** 2).sum() / max((np.abs(c[:, 1:]) ** 2).sum(), 1e-300)),
               "field_response_norm": float(np.linalg.norm(c)),
               "sigma_min_I_plus_K_inv": float(np.linalg.svd(np.eye(len(K)) + K, compute_uv=False)[-1]),
               "sigma_min_I_plus_K_solve": float(np.linalg.svd(np.eye(len(K)) + Ks, compute_uv=False)[-1]),
               "sigma_min_I_plus_K_chi_noise_1e-12": float(np.linalg.svd(np.eye(len(K)) + chi_p @ np.linalg.inv(A), compute_uv=False)[-1])}
        res["d%d" % d] = ent
        print(d, json.dumps(ent), flush=True)
    json.dump(res, open(os.path.join(D, "i1_d2_mode_v1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
