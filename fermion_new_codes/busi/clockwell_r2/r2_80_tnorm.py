"""Second reader #498: own certified UPPER bound on ||T_ext||_2, T_ext = HH -> LL block of V_ext = H_ext(1) - H_ext(0) at 9 sites
(own builder r2_50), and the sector restriction. Route (different from an interval Gram matrix):
  ||T||_2 <= ||T_f||_2 + ||T - T_f||_F, with T the exact rational target (exact G, K, N; Fractions) and T_f the binary64 matrix;
  ||T - T_f||_F: exact rational, upper square root;
  ||T_f||_2 <= ||U||_2 sigma_max ||V||_2 + ||R||_2 from a floating SVD T_f = U S V^dag + R, with every floating quantity bounded by
  a priori constants gamma_m = m u/(1 - m u) (IEEE binary64, round to nearest, any summation order), stated in the code.
Sector checks: V_ext maps the heavy-pair sector only into the light-pair sector (other blocks exactly 0, float and exact);
H0_ext, H0 and S0 preserve species (so e0 = psi0_ext - J psi0 lies in HH). argv: out_json"""
import json, sys, os, time, math
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
OUT = sys.argv[1]
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
u = 2.0 ** -53
def gam(m): return m * u / (1 - m * u)
NE = 9; Be, idxe = M.basis(NE); De = len(Be)
sp = lambda m: (m % 8) // 4
HH = [i for i, (a, b) in enumerate(Be) if sp(a) == 0 and sp(b) == 0]; LL = [i for i, (a, b) in enumerate(Be) if sp(a) == 1 and sp(b) == 1]
other_rows = [i for i in range(De) if i not in set(LL)]
hpos = {v: k for k, v in enumerate(HH)}; lpos = {v: k for k, v in enumerate(LL)}
out = {"T_shape": [len(LL), len(HH)], "forms": {}}
for form in ("onsite", "filtered"):
    t0 = time.process_time()
    He0 = M.hamiltonian(NE, form, 0.0, K); He1 = M.hamiltonian(NE, form, 1.0, K); Vx = He1 - He0
    r = {"other_blocks_from_HH_float": float(np.abs(Vx[np.ix_(other_rows, HH)]).max()),
         "H0_ext_species_mixing": float(max(np.abs(He0[np.ix_([i for i in range(De) if i not in set(HH)], HH)]).max(), 0.0))}
    Tf = Vx[np.ix_(LL, HH)].copy(); del He0, He1, Vx
    # exact target T (and exact check that every converter term maps an H pair to an L pair)
    Tex = {}; bad = 0
    for coef, Lv, Hv in M.converter_terms(NE, form, K, exact=True):
        for li, la in Lv.items():
            for hi, ha in Hv.items():
                if li not in lpos or hi not in hpos: bad += 1; continue
                key = (lpos[li], hpos[hi]); Tex[key] = Tex.get(key, 0) + Fr(coef) * Fr(la) * Fr(ha)
    r["exact_terms_outside_HH_to_LL"] = bad
    # ||T - T_f||_F exactly (T real; T_f complex with zero imaginary part expected)
    d2 = Fr(0)
    nzf = np.argwhere(Tf != 0)
    keys = set(Tex) | {(int(a), int(b)) for a, b in nzf}
    for k_ in keys:
        x = Tf[k_]; dre = Fr(float(x.real)) - Tex.get(k_, 0); d2 += dre * dre + Fr(float(x.imag)) ** 2
    eT = math.sqrt(float(d2)) * (1 + 4 * u) + 1e-300
    r["e_T_frobenius_upper"] = eT
    # rigorous floating SVD bound
    n = Tf.shape[0]
    U, s, Vh = np.linalg.svd(Tf); smax = float(s[0])
    def orth_defect(Q):
        G = Q.conj().T @ Q; dfro = float(np.linalg.norm(G - np.eye(Q.shape[1]))) * (1 + gam(2 * Q.shape[1] ** 2 + 2))
        err = gam(4 * n + 4) * float(np.linalg.norm(np.abs(Q).T @ np.abs(Q))) * (1 + gam(4 * n + 4)) * (1 + gam(2 * Q.shape[1] ** 2 + 2))
        return dfro + err
    dU = orth_defect(U); dV = orth_defect(Vh.conj().T)
    Rhat = Tf - (U * s) @ Vh
    absrec = (np.abs(U) * s) @ np.abs(Vh)
    Rf = float(np.linalg.norm(Rhat)) * (1 + gam(2 * n * n + 2)) + gam(4 * n + 8) * (float(np.linalg.norm(absrec)) + float(np.linalg.norm(Tf))) * (1 + gam(2 * n * n + 2)) * (1 + gam(4 * n + 8))
    up_f = (math.sqrt(1 + dU) * smax * math.sqrt(1 + dV) + Rf) * (1 + 8 * u)
    up = (up_f + eT) * (1 + 4 * u)
    r.update({"sigma_max_float": smax, "U_orth_defect_upper": dU, "V_orth_defect_upper": dV, "residual_F_upper": Rf, "upper_Tf": up_f, "upper_T": up,
              "cpu_s": time.process_time() - t0})
    if form == "onsite":
        Kn = np.array(K, float); r["norm_K_2"] = float(np.linalg.norm(Kn, 2)); r["K_eigs"] = sorted(np.linalg.eigvalsh(Kn).round(12).tolist())
    out["forms"][form] = r; print(form, r, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
