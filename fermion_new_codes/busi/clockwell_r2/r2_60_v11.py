"""Second reader #495 (n = 5 v1.1): own re-derivation, own builder r2_50 (validated in r2_51 against all saved n = 3 H).
stage 'recert': g = 1 pure-B Bwall at P = 64, 256, 1024 panels per unit (all midpoints recomputed), P rule, t*, frozen gate;
               two integrand variants: (a) plan formula in plain floating point; (b) rigorous floating-point bounds with stated
               a priori constants (Higham: any-order sum or dot product of m terms has |error| <= gamma_m sum|terms|,
               gamma_m = m u / (1 - m u), u = 2^-53, IEEE binary64 round-to-nearest, no overflow, underflow charged);
               plus an exact (integer) N2 check at sampled midpoints of the N2 lower bound used by the first reader.
stage 'bright': Psi_bright = (|0_R> B + |1_R> B')/sqrt 2, both forms, g = 0, 1, 2, t = 1..5: exact N2, Gram, D_RK (graded trace),
               light-only and complete whole-lead output; distance to the first reader's saved bright states (md5-checked).
stage 'pi':    Pi = (-1)^x beta; [Pi, h], [Pi_total, H0], [Pi_total, H]; dim(W cap {Pi_total = +-2}) and {0} exactly (rank mod p of the
               restricted unobservable pair) and by the float spectral route.
argv: stage out_json"""
import json, sys, os, time, math, hashlib, resource
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_50_model as M
stage, OUT = sys.argv[1], sys.argv[2]
K = json.load(open("<internal>/r2_51_validate_n3.json"))["K"]
n = 5; Bs, idx = M.basis(n); D = len(Bs)
u = 2.0 ** -53
def gam(m): return m * u / (1 - m * u)
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
t0 = time.process_time(); w0 = time.time(); out = {"stage": stage}
def e(a, b):
    v = np.zeros(D, complex); v[idx[(a, b)]] = 1; return v
s2 = 1 / math.sqrt(2)
Bv = e(0, 1); Bp = s2 * (e(0, 3) + e(1, 2))
def exact_norm2(v):
    """exact sum |v_i|^2 of binary64 entries, as a Fraction"""
    tot = 0; SH = 1100
    for x in v:
        for c in (x.real, x.imag):
            if c != 0.0:
                m, ex = math.frexp(c); mi = int(math.ldexp(m, 53)); tot += (mi * mi) << (2 * (ex - 53) + 2 * SH)
    return Fr(tot, 1 << (2 * SH))

if stage == "recert":
    out["arms"] = {}
    for form in ("onsite", "filtered"):
        ta = time.process_time()
        H = M.hamiltonian(n, form, 1.0, K); H = (H + H.conj().T) / 2
        w, V = np.linalg.eigh(H); c = V.conj().T @ Bv
        res = float(np.linalg.norm(H @ V - V * w, 2)); recon = float(np.linalg.norm(V @ c - Bv)); e_H = 1e-13
        def eps(t): return recon + t * res * float(np.linalg.norm(c)) + t * e_H
        He = M.hamiltonian(9, form, 1.0, K); Be, idxe = M.basis(9)
        emb = np.array([idxe[(a, b)] for a, b in Bs])
        S = He[:, emb].copy(); S[emb, :] -= H; del He
        Q = S.conj().T @ S; Cj = math.sqrt(float(np.abs(Q).sum(1).max())); L = Cj * float(np.abs(H).sum(1).max())
        absS = np.abs(S); nr = int((S != 0).sum(1).max())
        setup = time.process_time() - ta
        arm = {"C_join": Cj, "L": L, "row_nnz_max": nr, "eig_res": res, "recon": recon, "setup_cpu": setup, "P": {}}
        # P rule
        P_rule = next(P for P in (2 ** k for k in range(0, 20)) if 5 * L / (4 * P) <= 0.01)
        arm["P_rule_this_arm"] = P_rule
        exact_checks = []
        for P in (64, 256, 1024):
            tp = time.process_time()
            ts = (np.arange(5 * P) + 0.5) / P
            fa = np.empty(5 * P); fb = np.empty(5 * P); lowN2_fr = []
            for k, tm in enumerate(ts):
                phi = V @ (np.exp(-1j * w * tm) * c)
                N2hat = float(np.vdot(phi, phi).real)
                y = S @ phi
                fa[k] = float(np.linalg.norm(y)) / math.sqrt(N2hat) + Cj * 2 * eps(tm)
                # (b) rigorous floating bounds
                N2low = N2hat * (1 - gam(2 * D)) * (1 - 4 * u) - 2 * D * 2.0 ** -1022
                z = absS @ np.abs(phi)                                     # nonnegative terms
                num_err = gam(2 * nr + 2) * float(np.linalg.norm(z)) * (1 + gam(2 * len(z) + 4))
                yn2 = float(np.vdot(y, y).real) * (1 + gam(2 * len(y))) + 2 * len(y) * 2.0 ** -1022
                num_up = (math.sqrt(yn2) * (1 + 2 * u) + num_err) * (1 + 2 * u)
                fb[k] = num_up / math.sqrt(N2low) * (1 + 4 * u) + Cj * 2 * eps(tm) * (1 + 4 * u)
                if P == 1024 and k % 64 == 0:
                    ex = exact_norm2(phi); fr_bound = N2hat * (1 - (D * u) / (1 - D * u))
                    exact_checks.append([float(tm), float(ex - 1), N2hat - 1, bool(Fr(fr_bound) <= ex), bool(Fr(N2low) <= ex)])
            mids_cpu = time.process_time() - tp
            rows = {}
            for t in (1, 2, 3, 4, 5):
                sa = float(fa[:t * P].sum() / P); sb = float(fb[:t * P].sum() / P); qt = t * L / (4 * P)
                rows[t] = {"sampled_a": sa, "sampled_b": sb, "quad": qt, "Bwall_a": sa + qt, "Bwall_b": sb + qt}
            arm["P"][P] = {"rows": rows, "cpu": mids_cpu, "max_rel_b_minus_a": float(((fb - fa) / fa).max())}
        arm["exact_N2_checks"] = exact_checks
        arm["exact_N2_all_bounds_hold"] = {"first_reader_gamma_D": all(x[3] for x in exact_checks), "own_gamma_2D": all(x[4] for x in exact_checks)}
        out["arms"][form] = arm
        print(form, {P: {t: round(r["Bwall_b"], 7) for t, r in arm["P"][P]["rows"].items()} for P in arm["P"]}, flush=True)
    out["P_rule"] = max(a["P_rule_this_arm"] for a in out["arms"].values())

if stage == "bright":
    out["arms"] = {}
    sp = np.array([(b % 8) // 4 for b in range(40)]); st = np.array([b // 8 for b in range(40)])
    OUTS = {"light": [m for m in range(40) if sp[m] == 1 and st[m] >= 1], "all": [m for m in range(40) if st[m] >= 1]}
    masks = np.array([(1 << a) | (1 << b) for a, b in Bs])
    def kt(keep):
        keep = sorted(keep); ks = set(keep); tr = [m for m in range(40) if m not in ks]
        kmask = sum(1 << m for m in keep); tmask = sum(1 << m for m in tr)
        sgn = np.ones(D)
        for i, m in enumerate(masks):
            cnt = sum(1 for t_ in tr if (m >> t_) & 1 for k_ in keep if k_ > t_ and (m >> k_) & 1)
            sgn[i] = -1.0 if cnt % 2 else 1.0
        kp = masks & kmask; tp_ = masks & tmask
        ku = {v: i for i, v in enumerate(sorted(set(kp.tolist())))}; tu = {v: i for i, v in enumerate(sorted(set(tp_.tolist())))}
        return sgn, np.array([ku[v] for v in kp.tolist()]), np.array([tu[v] for v in tp_.tolist()]), len(ku), len(tu)
    KT = {k: kt([m for m in range(40) if m not in set(O)]) for k, O in OUTS.items()}
    def drk(C, key):
        C = C / np.linalg.norm(C); sgn, ki, ti, nk, nt = KT[key]
        Mx = np.zeros((2, nk, nt), complex)
        for r in range(2): Mx[r, ki, ti] = sgn * C[r]
        M2 = Mx.reshape(2 * nk, nt); rho = M2 @ M2.conj().T; R4 = rho.reshape(2, nk, 2, nk)
        rR = np.einsum("aibi->ab", R4); rK = np.einsum("aiaj->ij", R4)
        return 0.5 * float(np.abs(np.linalg.eigvalsh(rho - np.kron(rR, rK))).sum())
    SAVED = {("onsite", 0): "203383cfecc2f0276548d9e5c991cfa7", ("onsite", 1): "45b0dbc827fd5d8f50926f39ac60f6eb", ("onsite", 2): "50cfddd0aca86ff65f766a3472a48f39",
             ("filtered", 0): "203383cfecc2f0276548d9e5c991cfa7", ("filtered", 1): "36c0139356fa7313a4c24275da4208a3", ("filtered", 2): "04824a24d661874a99562ca395181047"}
    for form in ("onsite", "filtered"):
        for g in (0, 1, 2):
            ta = time.process_time()
            H = M.hamiltonian(n, form, float(g), K); H = (H + H.conj().T) / 2
            w, V = np.linalg.eigh(H)
            C0 = np.vstack([s2 * Bv, s2 * Bp]); cc = C0 @ V.conj()  # rows: V^dag r
            sp_path = "<internal>/%s_g%d_states_bright.npz" % (form, g)
            sm = md5(sp_path); saved = np.load(sp_path)
            rows = []
            for t in (1, 2, 3, 4, 5):
                Ct = (cc * np.exp(-1j * w * t)) @ V.T
                N2 = exact_norm2(Ct.ravel())
                G = 2 * np.array([[np.vdot(Ct[a], Ct[b]) for b in range(2)] for a in range(2)]) / float(N2)
                rows.append({"t": t, "N2_minus_1": float(N2 - 1), "gram_dist": float(np.linalg.norm(G - np.eye(2))),
                             "D_light": drk(Ct, "light"), "D_all": drk(Ct, "all"), "dist_saved": float(np.linalg.norm(Ct - saved["t%d" % t]))})
            out["arms"]["%s_g%d" % (form, g)] = {"rows": rows, "saved_md5_ok": sm == SAVED[(form, g)], "cpu": time.process_time() - ta}
            print(form, g, [(r["t"], round(r["D_light"], 10), round(r["D_all"], 10), r["dist_saved"]) for r in rows], flush=True)

if stage == "pi":
    def pi_mode(m): return (-1) ** (m // 8) * (1 if (m % 4) < 2 else -1)
    out["rows"] = []
    for nn in (3, 5):
        Bn, idn = M.basis(nn)
        for form in ("onsite", "filtered"):
            H0 = M.hamiltonian(nn, form, 0, K); H1 = M.hamiltonian(nn, form, 1, K)
            h = np.zeros((8 * nn, 8 * nn), complex)
            for (p, q), v in M.one_body(nn).items(): h[p, q] += v
            pim = np.diag([pi_mode(m) for m in range(8 * nn)]).astype(float)
            PiT = np.diag([pi_mode(a) + pi_mode(b) for a, b in Bn]).astype(float)
            row = {"n": nn, "form": form, "comm_pi_h": float(np.abs(pim @ h - h @ pim).max()),
                   "comm_PiT_H0": float(np.abs(PiT @ H0 - H0 @ PiT).max()), "comm_PiT_H": float(np.abs(PiT @ H1 - H1 @ PiT).max())}
            sp = lambda m: (m % 8) // 4
            HH = [i for i, (a, b) in enumerate(Bn) if sp(a) == 0 and sp(b) == 0]; LL = [i for i, (a, b) in enumerate(Bn) if sp(a) == 1 and sp(b) == 1]
            Hh = H0[np.ix_(HH, HH)]; T = (H1 - H0)[np.ix_(LL, HH)]
            piv = np.array([pi_mode(Bn[i][0]) + pi_mode(Bn[i][1]) for i in HH])
            # float spectral route on the full HH sector, then intersection dims via projector eigenvalues
            wv, U = np.linalg.eigh(Hh); cl = []; s0 = 0
            for k in range(1, len(wv) + 1):
                if k == len(wv) or wv[k] - wv[k - 1] > 1e-9: cl.append((s0, k)); s0 = k
            Wb = []; sc = np.linalg.norm(T, 2)
            for a, b in cl:
                Ue = U[:, a:b]; _, s, Vh = np.linalg.svd(T @ Ue); sv = np.zeros(b - a); sv[:len(s)] = s
                nl = [k for k in range(b - a) if sv[k] <= 1e-10 * sc]
                if nl: Wb.append(Ue @ Vh.conj().T[:, nl])
            Qw, _ = np.linalg.qr(np.hstack(Wb))
            for lab, sel in (("pm2", np.abs(piv) == 2), ("zero", piv == 0)):
                Ps = np.diag(sel.astype(float)); ev = np.linalg.eigvalsh(Qw.conj().T @ Ps @ Qw)
                row["float_dim_W_cap_%s" % lab] = int((ev > 1 - 1e-9).sum()); row["float_gap_%s" % lab] = [float(ev[ev <= 1 - 1e-9].max()) if (ev <= 1 - 1e-9).any() else None]
                # exact: unobservable subspace of (H restricted to the sector, T restricted), rank mod p (sector is H0-invariant)
                ids = [i for i, s_ in enumerate(sel) if s_]
                row["block_leak_HH"] = float(np.abs(Hh[np.ix_([i for i in range(len(HH)) if not sel[i]], ids)]).max()) if len(ids) < len(HH) else 0.0
            row["dimW_float"] = int(Qw.shape[1])
            out["rows"].append(row); print(row, flush=True)
out["cpu_s"] = time.process_time() - t0; out["wall_s"] = time.time() - w0; out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
json.dump(out, open(OUT, "w"), indent=1, default=str)
print("done", stage, out["cpu_s"])
