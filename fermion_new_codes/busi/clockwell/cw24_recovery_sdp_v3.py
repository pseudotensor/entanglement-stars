"""cw24 (v3: the SDP of each block is solved on the support of rho_O (an exact reduction: compressing sigma
and Z to supp(rho_O) preserves feasibility and the objective), then lifted back and CERTIFIED ON THE FULL SPACE
exactly as before; cw21 was stopped at 4461 CPU s and 62 GB RSS on its first full-size g = 1 transport-diagnostic
block, before the 2 CPU-hour limit; cw21 itself was v2 of cw19: identical mathematics and certification; vectorized sparse SDP assembly only, cw19 was stopped
after 5 of 24 solves for speed; its partial stdout is kept as cw19_stdout.txt): channel-optimal recovery audit (#490), busi. Runs only if cw18 admitted every state.

For each admitted state psi (saved registered raw vector, R first, n = 3) and each output algebra O
  (i)  whole-lead light modes {12..15, 20..23}  (the receiver),
  (ii) all light modes incl. the core {4..7, 12..15, 20..23}  (transport diagnostic, not a receiver),
the optimal entanglement fidelity over ALL CPTP decoders O -> Q (qubit), with every O sector kept
(vacuum, one- and two-particle; no postselection, no branch renormalization) is
    F*(rho_RO) = max { <Phi|(id_R x D)(rho_RO)|Phi> : D CPTP }
              = (1/2) max { Tr[rho_RO Z] : Z >= 0 on R x O, Tr_R Z = I_O }          (primal)
              = (1/2) min { Tr sigma_O : I_R x sigma_O - rho_RO >= 0 }                  (dual)
(Koenig-Renner-Schaffner: 2^{-H_min(R|O)} = d_R max_D F). rho_RO = C C^dagger / N2 with C the graded
coefficient matrix (rows (r, O-configuration), columns complement-configuration; sign -1 when a complement
mode precedes an O mode), N2 the exact squared norm (cw18). rho_RO is block diagonal in the O particle
number k = 0, 1, 2 (the complement has definite number), so both optima split exactly into k blocks
(pinching argument); the decoder may read k, which keeps every sector, discards none, and renormalizes none.
Numerics: Clarabel interior point on the real embedding [[Re, -Im], [Im, Re]] of each Hermitian block.
CERTIFICATION (python-flint arb/acb ball arithmetic, 256-bit; inputs are exact dyadic binary64 values):
  upper: sigma' = sigma_num + delta I; write I x sigma' - rho = B B^dagger + eta I + E with B dyadic (from a
         float eigendecomposition of the numerical slack minus eta I, negatives clipped) and certify
         ||E||_F < eta  =>  I x sigma' - rho >= 0 exactly (Weyl); UB = (1/2) sum_k Tr sigma'_k (upper end).
  lower: Z = A A^dagger with A dyadic (clipped eigendecomposition of the numerical decoder variable), so Z >= 0
         exactly; s_k = Gershgorin upper bound of lambda_max(Tr_R Z_k); Z_k / s_k is feasible with
         Tr_R <= I (completing to equality only adds a nonnegative term); LB = (1/2) sum_k ||A_k^dagger C_k||_F^2 / (N2 s_k)
         (lower end). Both are bounds for the normalized SAVED state.
  physical target: widen by 2 eps (eps = saved state error; |psi_hat/||psi_hat|| - psi| <= 2 eps, and F is an
         expectation of a POVM element, so it moves by at most the trace distance <= the vector distance).
GATE (physical bounds): LB > 1/2 PASS (one-shot transfer witness only); UB <= 1/2 FAIL; else UNRESOLVED.
Outputs: <internal>/cw24_sdp.json (+ per-state npz of the numerical sigma and Z).
"""
import json, os, sys, time, resource, math, hashlib
import numpy as np
import scipy.sparse as sp
import clarabel
import flint
from flint import acb_mat, acb, arb

flint.ctx.prec = 256
OUTD = '<internal>/recovery'
ADM = f'{OUTD}/cw18_admission.json'
OUTPUTS = {'i_whole_lead_light': [12, 13, 14, 15, 20, 21, 22, 23],
           'ii_all_light_incl_core': [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]}
ETA = 1e-10

def keys_from_rowkeys(rk):
    out = []
    for rr, mask in rk:
        ms = [b for b in range(24) if (mask >> b) & 1]
        assert rr == 0 and len(ms) == 2
        out.append(tuple(ms))
    return out

def coefficient_blocks(v, keys, Omodes):
    """returns {k: (C_k float complex array (2*d_k, d_K_k), list of O configs)} with graded signs"""
    O = set(Omodes)
    oconf, kconf, ent = {}, {}, []
    for i, (m1, m2) in enumerate(keys):
        a = tuple(m for m in (m1, m2) if m in O); t = tuple(m for m in (m1, m2) if m not in O)
        sgn = -1.0 if (m1 not in O and m2 in O) else 1.0
        oconf.setdefault(a, None); kconf.setdefault(t, None); ent.append((i, a, t, sgn))
    blocks = {}
    for k in (0, 1, 2):
        A = sorted(a for a in oconf if len(a) == k); T = sorted(t for t in kconf if len(t) == 2 - k)
        ai = {a: j for j, a in enumerate(A)}; ti = {t: j for j, t in enumerate(T)}
        C = np.zeros((2 * len(A), len(T)), dtype=complex)
        for i, a, t, sgn in ent:
            if len(a) == k:
                for r in range(2):
                    C[r * len(A) + ai[a], ti[t]] += sgn * v[r, i]
        blocks[k] = (C, A)
    return blocks

def svec_index(i, j):  # upper triangle, column major (i <= j)
    return j * (j + 1) // 2 + i

def real_embed(H):
    return np.block([[H.real, -H.imag], [H.imag, H.real]])

def svec(Rm):
    n = Rm.shape[0]
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    mask = ii <= jj
    idx = jj[mask] * (jj[mask] + 1) // 2 + ii[mask]
    vals = Rm[mask] * np.where(ii[mask] != jj[mask], math.sqrt(2.0), 1.0)
    out = np.zeros(n * (n + 1) // 2); out[idx] = vals
    return out

def solve_block(rho, d):
    """dual: min Tr sigma s.t. I_2 x sigma - rho >= 0; returns sigma (d x d complex), W (real 4d PSD dual), info"""
    N = 2 * d; n = 2 * N; m = n * (n + 1) // 2
    params = [('re', i, i) for i in range(d)] + [('re', i, j) for j in range(d) for i in range(j)] + [('im', i, j) for j in range(d) for i in range(j)]
    rows, cols, vals = [], [], []
    sq2 = math.sqrt(2.0)
    for p, (kind, i, j) in enumerate(params):
        ents = []  # entries (a, b, value) of the real embedding of I_2 x B_p, full matrix
        for r in range(2):
            a, b = r * d + i, r * d + j
            if kind == 're':
                for (x, y) in {(a, b), (b, a)}:
                    ents += [(x, y, 1.0), (x + N, y + N, 1.0)]
            else:  # B = i E_ij - i E_ji  ->  Im part +1 at (a,b), -1 at (b,a)
                for (x, y, im) in ((a, b, 1.0), (b, a, -1.0)):
                    ents += [(x, y + N, -im), (x + N, y, im)]
        for (x, y, val) in ents:
            if x <= y:
                rows.append(svec_index(x, y)); cols.append(p); vals.append(-(val * (sq2 if x != y else 1.0)))
    A = sp.csc_matrix((vals, (rows, cols)), shape=(m, len(params)))
    bvec = -svec(real_embed(rho))
    q = np.array([1.0 if kind == 're' and i == j else 0.0 for (kind, i, j) in params])
    P = sp.csc_matrix((len(params), len(params)))
    settings = clarabel.DefaultSettings(); settings.verbose = False
    settings.tol_gap_abs = 1e-11; settings.tol_gap_rel = 1e-11; settings.tol_feas = 1e-11; settings.max_iter = 300
    solver = clarabel.DefaultSolver(P, q, A, bvec, [clarabel.PSDTriangleConeT(n)], settings)
    sol = solver.solve()
    x = np.array(sol.x); z = np.array(sol.z)
    sigma = np.zeros((d, d), dtype=complex)
    for p, (kind, i, j) in enumerate(params):
        if kind == 're':
            sigma[i, j] += x[p]
            if i != j: sigma[j, i] += x[p]
        else:
            sigma[i, j] += 1j * x[p]; sigma[j, i] -= 1j * x[p]
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    mask = ii <= jj
    idx = jj[mask] * (jj[mask] + 1) // 2 + ii[mask]
    W = np.zeros((n, n))
    W[ii[mask], jj[mask]] = z[idx] / np.where(ii[mask] != jj[mask], sq2, 1.0)
    W = W + np.triu(W, 1).T
    return sigma, W, dict(status=str(sol.status), iterations=int(sol.iterations), obj_primal=float(sol.obj_val), obj_dual=float(sol.obj_val_dual))

def W_to_Z(W, rho, N):
    """complex Hermitian Z with Tr[rho Z] = <E(rho), W>; choose the convention that reproduces it"""
    W11, W12, W21, W22 = W[:N, :N], W[:N, N:], W[N:, :N], W[N:, N:]
    target = float(np.sum(np.block([[rho.real, -rho.imag], [rho.imag, rho.real]]) * W))
    best = None
    for Z in ((W11 + W22) + 1j * (W21 - W12), (W11 + W22) - 1j * (W21 - W12)):
        Z = (Z + Z.conj().T) / 2
        val = float(np.real(np.trace(rho @ Z)))
        if best is None or abs(val - target) < best[0]:
            best = (abs(val - target), Z)
    return best[1], target

def solve_compressed(C, d, N2):
    """solve on supp(rho_O): M = [C_0 | C_1] (d x 2 d_K); Q = left singular vectors with s > 1e-13 s_max"""
    Mo = np.hstack([C[:d, :], C[d:, :]])
    U, sv, _ = np.linalg.svd(Mo, full_matrices=False)
    r = int(np.sum(sv > 1e-13 * max(sv.max(), 1e-300))) if sv.size else 0
    if r == 0:
        sigma = np.zeros((d, d), dtype=complex); Z = np.zeros((2 * d, 2 * d), dtype=complex)
        return sigma, Z, 0.0, dict(status='zero_block', rank=0)
    Q = U[:, :r]
    IQ = np.kron(np.eye(2), Q)
    Cs = IQ.conj().T @ C
    rho_s = Cs @ Cs.conj().T / N2
    sig_s, W, info = solve_block(rho_s, r)
    Z_s, prim = W_to_Z(W, rho_s, 2 * r)
    info['rank'] = r; info['d_full'] = d
    return Q @ sig_s @ Q.conj().T, IQ @ Z_s @ IQ.conj().T, prim, info

def acb_from(M):
    return acb_mat([[acb(float(x.real), float(x.imag)) for x in row] for row in np.atleast_2d(M)])

def frob2_upper(M):
    s = arb(0)
    for row in M.tolist():
        for e in row:
            s += e.real ** 2 + e.imag ** 2
    return s.upper()

def certify_block(C, d, N2arb, sigma, Z):
    n = 2 * d
    Ca = acb_from(C)
    rho_a = Ca * Ca.conjugate().transpose()
    rho_a = acb_mat([[e / N2arb for e in row] for row in rho_a.tolist()])
    rho_f = (C @ C.conj().T)
    # ---- upper bound
    sig = (sigma + sigma.conj().T) / 2
    M0 = np.kron(np.eye(2), sig) - rho_f / float(N2arb.mid())
    lam = np.linalg.eigvalsh(M0)
    delta = max(0.0, -lam.min()) + 2 * ETA
    sigp = sig + delta * np.eye(d)
    Mf = np.kron(np.eye(2), sigp) - rho_f / float(N2arb.mid()) - ETA * np.eye(n)
    w, V = np.linalg.eigh((Mf + Mf.conj().T) / 2)
    B = V * np.sqrt(np.clip(w, 0, None))
    Ma = acb_from(np.kron(np.eye(2), sigp)) - rho_a
    E = Ma - acb_from(B) * acb_from(B).conjugate().transpose() - acb_from(ETA * np.eye(n))
    eF2 = frob2_upper(E)
    psd_ok = bool(eF2 < arb(ETA) ** 2)
    tr = arb(0)
    for i in range(d):
        tr += arb(float(sigp[i, i].real))
    UB = (tr / 2).upper()
    # ---- lower bound
    Zh = (Z + Z.conj().T) / 2
    wz, Vz = np.linalg.eigh(Zh)
    Af = Vz * np.sqrt(np.clip(wz, 0, None))
    Aa = acb_from(Af)
    T = acb_mat(d, d)
    Aac = Aa
    A0 = acb_mat([Aac.tolist()[i] for i in range(d)]); A1 = acb_mat([Aac.tolist()[d + i] for i in range(d)])
    T = A0 * A0.conjugate().transpose() + A1 * A1.conjugate().transpose()
    # certified s >= lambda_max(T): s I - T = B B^dagger + eta I + E with ||E||_F < eta (Weyl); Gershgorin as fallback
    Tf = Af[:d] @ Af[:d].conj().T + Af[d:] @ Af[d:].conj().T
    lam_T = float(np.linalg.eigvalsh((Tf + Tf.conj().T) / 2).max())
    s_try = lam_T + 2 * ETA
    Mf = s_try * np.eye(d) - Tf - ETA * np.eye(d)
    wT, VT = np.linalg.eigh((Mf + Mf.conj().T) / 2)
    BT = acb_from(VT * np.sqrt(np.clip(wT, 0, None)))
    ET = acb_from(s_try * np.eye(d)) - T - BT * BT.conjugate().transpose() - acb_from(ETA * np.eye(d))
    if frob2_upper(ET) < arb(ETA) ** 2:
        s_up = arb(s_try)
    else:
        s = arb(0)
        for row in T.tolist():
            rs = arb(0)
            for e in row:
                rs += abs(e)
            s = rs if rs.upper() > s.upper() else s
        s_up = s.upper()
    Pm = Aa.conjugate().transpose() * Ca
    val = arb(0)
    for row in Pm.tolist():
        for e in row:
            val += e.real ** 2 + e.imag ** 2
    LB = (val / N2arb / (2 * s_up)).lower()
    return dict(UB=UB, LB=LB, psd_certified=psd_ok, delta=delta, E_frob_upper=float(eF2.sqrt().upper()) if True else None,
                gershgorin_s_upper=float(s_up))

def main():
    adm = json.load(open(ADM))
    if adm['stop']:
        print('ADMISSION FAILED: STOP'); sys.exit(2)
    ext = json.load(open('<internal>/cw05_extract.json'))
    res = {'admission_md5': hashlib.md5(open(ADM, 'rb').read()).hexdigest(), 'eta': ETA, 'states': []}
    for st in adm['states']:
        keys = keys_from_rowkeys(ext['physical'][st['cohort']]['conversion'][0]['row_keys'])
        z = np.load(st['array']); v = z[z.files[0]]
        from fractions import Fraction as Fr
        N2f = Fr(st['N2_exact'])
        N2arb = arb(N2f.numerator) / arb(N2f.denominator)
        row = dict(arm=st['arm'], g=st['g'], t=st['t'], array=st['array'], array_md5=st['array_md5'], state_error=st['state_error'], outputs={})
        for oname, Om in OUTPUTS.items():
            t0 = time.process_time(); w0 = time.time()
            blocks = coefficient_blocks(v, keys, Om)
            tot = dict(UB=arb(0), LB=arb(0), num_primal=0.0, num_dual=0.0, blocks={})
            ok = True
            for k, (C, Aconf) in blocks.items():
                d = len(Aconf)
                rho = C @ C.conj().T / float(N2arb.mid())
                sigma, Z, prim, info = solve_compressed(C, d, float(N2arb.mid()))
                cert = certify_block(C, d, N2arb, sigma, Z)
                ok = ok and cert['psd_certified']
                tot['UB'] += cert['UB']; tot['LB'] += cert['LB']
                tot['num_dual'] += float(np.real(np.trace(sigma))) / 2; tot['num_primal'] += prim / 2
                tot['blocks'][str(k)] = dict(d_O=d, weight=float(np.real(np.trace(rho))), solver=info, num_dual_half=float(np.real(np.trace(sigma))) / 2,
                                             num_primal_half=prim / 2, cert_LB=float(cert['LB'].lower()), cert_UB=float(cert['UB'].upper()),
                                             psd_certified=cert['psd_certified'], delta=cert['delta'], E_frob_upper=cert['E_frob_upper'], s_upper=cert['gershgorin_s_upper'])
                np.savez(f"{OUTD}/cw24_{st['arm']}_g{st['g']}_t{int(st['t'])}_{oname}_k{k}.npz", sigma=sigma, Z=Z)
            LBc = float(tot['LB'].lower()); UBc = float(tot['UB'].upper())
            eps2 = 2 * st['state_error']
            LBp = LBc - eps2; UBp = UBc + eps2
            gate = 'PASS' if LBp > 0.5 else ('FAIL' if UBp <= 0.5 else 'UNRESOLVED')
            if not ok:
                gate = 'UNCERTIFIED'
            row['outputs'][oname] = dict(modes=Om, F_lower_saved_state=LBc, F_upper_saved_state=UBc, F_lower_physical=LBp, F_upper_physical=UBp,
                                         numerical_primal=tot['num_primal'], numerical_dual=tot['num_dual'], certified_gap=UBc - LBc,
                                         all_blocks_psd_certified=ok, gate=gate if oname.startswith('i_') else gate + ' (diagnostic only)',
                                         cpu_s=time.process_time() - t0, wall_s=time.time() - w0,
                                         peak_rss_kib_so_far=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, blocks=tot['blocks'])
            print(f"{st['arm']:8s} g={st['g']} t={st['t']} {oname:24s} F in [{LBc:.10f}, {UBc:.10f}] phys [{LBp:.10f}, {UBp:.10f}] {row['outputs'][oname]['gate']} cpu {row['outputs'][oname]['cpu_s']:.1f}s", flush=True)
        res['states'].append(row)
        json.dump(res, open(f'{OUTD}/cw24_sdp.json', 'w'), indent=1)
    res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res['cpu_s_total'] = time.process_time()
    json.dump(res, open(f'{OUTD}/cw24_sdp.json', 'w'), indent=1)
    print('done; cpu', res['cpu_s_total'], 'peak RSS KiB', res['peak_rss_kib'])

if __name__ == '__main__':
    main()
