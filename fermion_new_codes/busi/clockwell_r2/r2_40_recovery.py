"""Second reader (clockwell_r2), #490 recovery audit: admission + channel-optimal entanglement fidelity
for the 12 saved n = 3 registered states (on-site / filtered, g = 0, 1, t = 1, 2, 4), outputs
(i) whole-lead light modes {12..15, 20..23} (receiver) and (ii) all light modes incl. core {4..7} added (diagnostic).

Route (independent of the first reader): rho_RO is block diagonal in the O particle number k (total N = 2).
  F* = 2^{-Hmin(R|O)} / 2 = (1/2) sum_k Opt_k,  Opt_k = min Tr sigma_k s.t. I_2 x sigma_k >= rho_k   (Koenig-Renner-Schaffner)
  k = 0 (O vacuum, 2x2):   Opt_0 = lambda_max(rho_0)                         closed form, interval arithmetic
  k = 2 (K vacuum, rank 1): Opt_2 = (s_0 + s_1)^2 = Tr G + 2 sqrt(det G),  G = V V^dag (2x2), V the 2 x d_O coefficient block
  k = 1 (general):        SDP (own Clarabel formulation, dual and primal separately), certified exactly:
                          UB: sigma' = sigma_num + delta I, exact LDL of the real embedding of N2 (I x sigma') - C C^dag > 0
                          LB: Z = A A^dag (dyadic A), s >= lambda_max(Tr_R Z) by exact LDL of s I - Tr_R Z, LB = ||A^dag C||_F^2 / (N2 s)
Cross-check: full SDP (dual) on every block, compressed to supp(rho_O) for k = 0, 2, compared with the closed forms.
Usage: python r2_40_recovery.py out_json
"""
import json, sys, os, math, hashlib, time, resource
import numpy as np
import scipy.sparse as sps
import clarabel
from flint import fmpq
from mpmath import iv, mp

iv.dps = 60; mp.dps = 60
t0 = time.process_time(); w0 = time.time()
OUT = sys.argv[1]
REPORTS = {("onsite", 0): "<internal>/even_n3_g0_v1",
           ("onsite", 1): "<internal>/even_n3_g1_v1",
           ("filtered", 0): "<internal>/even_n3_g0_v1",
           ("filtered", 1): "<internal>/even_n3_g1_v1"}
WIP_ARRAYS = {  # first reader's sec. 1 listing (array name, md5) -- compared, not trusted
    ("onsite", 0, 1): ("array_123.npz", "b8a38a65d0aa05f255830d67c562e0bd"), ("onsite", 0, 2): ("array_231.npz", "d3efbe6a94d9bb2e25609f158978d9dd"),
    ("onsite", 0, 4): ("array_405.npz", "8e77284ff16d6e5d58b19c6d97bdd11f"), ("onsite", 1, 1): ("array_123.npz", "29d9c0642f243a2e5911b43ec18a00c0"),
    ("onsite", 1, 2): ("array_231.npz", "44025b771dcc567323b1ae16b34602c1"), ("onsite", 1, 4): ("array_405.npz", "b64daa5e58dbde4642799f9e2116f20f"),
    ("filtered", 0, 1): ("array_121.npz", "b8a38a65d0aa05f255830d67c562e0bd"), ("filtered", 0, 2): ("array_229.npz", "d3efbe6a94d9bb2e25609f158978d9dd"),
    ("filtered", 0, 4): ("array_403.npz", "8e77284ff16d6e5d58b19c6d97bdd11f"), ("filtered", 1, 1): ("array_121.npz", "7a91f9a0983ed4b406de070872f3891d"),
    ("filtered", 1, 2): ("array_229.npz", "a787dd40e81c0597cab591b71f2d41cb"), ("filtered", 1, 4): ("array_403.npz", "5dd4adbced52187eab4c8b7b82925b92")}
REPORT_MD5 = {("onsite", 0): "84db4c2a9fe340d41bac5750db83a8c5", ("onsite", 1): "3325aebbc1cd4042b89d1ee0f7d9ec22",
              ("filtered", 0): "62df5e285a7763662983a0a4eaa9ddcc", ("filtered", 1): "ae7f8968500fdaab89fb0407ee02bdb6"}
OUTPUTS = {"i_whole_lead_light": [12, 13, 14, 15, 20, 21, 22, 23],
           "ii_all_light_incl_core": [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]}
NM = 24
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()

# ---------------- exact helpers ----------------
def q(x): return fmpq(*float(x).as_integer_ratio())
def ldl_pd(Mq):
    """exact symmetric Gaussian elimination (no pivoting) of a real symmetric fmpq matrix; True iff every pivot > 0
    (all leading principal minors > 0, i.e. positive definite)"""
    n = len(Mq); A = [row[:] for row in Mq]; minpiv = None
    for j in range(n):
        p = A[j][j]
        if p <= 0: return False, p
        minpiv = p if minpiv is None or p < minpiv else minpiv
        Aj = A[j]
        for i in range(j + 1, n):
            if A[i][j] != 0:
                f = A[i][j] / p; Ai = A[i]
                for k in range(j, n): Ai[k] -= f * Aj[k]
    return True, minpiv
def realembed_q(Hre, Him):
    n = len(Hre); M = [[fmpq(0)] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = Hre[i][j]; M[i + n][j + n] = Hre[i][j]; M[i][j + n] = -Him[i][j]; M[i + n][j] = Him[i][j]
    return M

# ---------------- SDP (own Clarabel formulation, real embedding, Clarabel svec) ----------------
S2 = math.sqrt(2.0)
def svec_idx(n):
    idx = {}; c = 0
    for j in range(n):
        for i in range(j + 1): idx[(i, j)] = c; c += 1
    return idx, c
def herm_basis(d):
    B = []
    for i in range(d):
        E = np.zeros((d, d), complex); E[i, i] = 1; B.append(("d", E))
    for i in range(d):
        for j in range(i + 1, d):
            E = np.zeros((d, d), complex); E[i, j] = E[j, i] = 1; B.append(("re", E))
            E = np.zeros((d, d), complex); E[i, j] = 1j; E[j, i] = -1j; B.append(("im", E))
    return B
def remb(H): return np.block([[H.real, -H.imag], [H.imag, H.real]])
def svec(S, idx, m):
    v = np.zeros(m)
    for (i, j), c in idx.items(): v[c] = S[i, j] * (1.0 if i == j else S2)
    return v
def settings():
    s = clarabel.DefaultSettings(); s.verbose = False
    s.tol_gap_abs = 1e-12; s.tol_gap_rel = 1e-12; s.tol_feas = 1e-12; s.tol_ktratio = 1e-10; s.max_iter = 500
    return s
def sdp_dual(rho):
    """min Tr sigma s.t. I_2 x sigma - rho >= 0; rho (2d x 2d) Hermitian, index r*d + a"""
    D2 = rho.shape[0]; d = D2 // 2; B = herm_basis(d)
    n = 2 * D2; idx, m = svec_idx(n)
    cols = [svec(remb(np.kron(np.eye(2), E)), idx, m) for _, E in B]
    A = -np.array(cols).T; b = -svec(remb(rho), idx, m)
    qv = np.array([1.0 if t == "d" else 0.0 for t, _ in B])
    sol = clarabel.DefaultSolver(sps.csc_matrix((len(B), len(B))), qv, sps.csc_matrix(A), b, [clarabel.PSDTriangleConeT(n)], settings()).solve()
    x = np.array(sol.x); sig = sum(xi * E for xi, (_, E) in zip(x, B))
    return (sig + sig.conj().T) / 2, float(np.trace(sig).real), str(sol.status)
def sdp_primal(rho):
    """max Tr rho Z s.t. Z >= 0, Tr_R Z = I_d"""
    D2 = rho.shape[0]; d = D2 // 2; B = herm_basis(D2); Bd = herm_basis(d)
    n = 2 * D2; idx, m = svec_idx(n)
    Apsd = -np.array([svec(remb(E), idx, m) for _, E in B]).T
    # equality: for each Hermitian basis element F of d: Re Tr(F^dag Tr_R Z) = Tr(F I)
    Aeq = np.zeros((len(Bd), len(B))); beq = np.zeros(len(Bd))
    for r_, (_, F) in enumerate(Bd):
        beq[r_] = np.trace(F.conj().T).real
        for c_, (_, E) in enumerate(B):
            TrR = E[:d, :d] + E[d:, d:]
            Aeq[r_, c_] = np.trace(F.conj().T @ TrR).real
    qv = -np.array([np.trace(rho @ E).real for _, E in B])
    A = sps.csc_matrix(np.vstack([Aeq, Apsd])); b = np.concatenate([beq, np.zeros(m)])
    sol = clarabel.DefaultSolver(sps.csc_matrix((len(B), len(B))), qv, A, b, [clarabel.ZeroConeT(len(Bd)), clarabel.PSDTriangleConeT(n)], settings()).solve()
    x = np.array(sol.x); Z = sum(xi * E for xi, (_, E) in zip(x, B))
    return (Z + Z.conj().T) / 2, -float(sol.obj_val), str(sol.status)

# ---------------- certified bounds for a general block from (C, N2) ----------------
def certify_general(C, N2q, N2f):
    """C: (2d x dK) complex float (dyadic) coefficient block with rho = C C^dag / N2. Returns (LB, UB) exact fmpq and numerics."""
    rho = C @ C.conj().T / N2f; D2 = rho.shape[0]; d = D2 // 2
    if not np.any(C):
        return fmpq(0), fmpq(0), {"zero_block": True}
    sig, dval, st_d = sdp_dual(rho)
    Z, pval, st_p = sdp_primal(rho)
    # --- UB: sigma' = sigma + delta I, prove N2 (I x sigma') - C C^dag > 0 exactly
    lam = np.linalg.eigvalsh(np.kron(np.eye(2), sig) - rho).min()
    delta = max(0.0, -lam) * 2 + 1e-13 * max(1.0, float(np.abs(rho).max())) + 1e-300
    for attempt in range(8):
        sp_ = sig + delta * np.eye(d)
        Sre = [[q(sp_[i, j].real) for j in range(d)] for i in range(d)]; Sim = [[q(sp_[i, j].imag) for j in range(d)] for i in range(d)]
        for i in range(d): Sim[i][i] = fmpq(0)
        for i in range(d):
            for j in range(i): Sre[i][j] = Sre[j][i]; Sim[i][j] = -Sim[j][i]
        Cre = [[q(C[i, j].real) for j in range(C.shape[1])] for i in range(D2)]; Cim = [[q(C[i, j].imag) for j in range(C.shape[1])] for i in range(D2)]
        Mre = [[fmpq(0)] * D2 for _ in range(D2)]; Mim = [[fmpq(0)] * D2 for _ in range(D2)]
        for i in range(D2):
            for j in range(D2):
                re = fmpq(0); im = fmpq(0)
                for t in range(C.shape[1]):  # (C C^dag)_ij = sum_t C_it conj(C_jt)
                    re += Cre[i][t] * Cre[j][t] + Cim[i][t] * Cim[j][t]
                    im += Cim[i][t] * Cre[j][t] - Cre[i][t] * Cim[j][t]
                blk_i, a = divmod(i, d); blk_j, b_ = divmod(j, d)
                sre = Sre[a][b_] if blk_i == blk_j else fmpq(0); sim = Sim[a][b_] if blk_i == blk_j else fmpq(0)
                Mre[i][j] = N2q * sre - re; Mim[i][j] = N2q * sim - im
        ok, piv = ldl_pd(realembed_q(Mre, Mim))
        if ok: break
        delta *= 10
    UB = sum((Sre[i][i] for i in range(d)), fmpq(0)) if ok else None
    # --- LB: Z = A A^dag from the numerical primal; s >= lambda_max(Tr_R Z) exactly
    w, V = np.linalg.eigh(Z); w = np.clip(w, 0, None); A = V * np.sqrt(w)
    Are = [[q(A[i, j].real) for j in range(D2)] for i in range(D2)]; Aim = [[q(A[i, j].imag) for j in range(D2)] for i in range(D2)]
    # T = Tr_R (A A^dag)
    Tre = [[fmpq(0)] * d for _ in range(d)]; Tim = [[fmpq(0)] * d for _ in range(d)]
    for a in range(d):
        for b_ in range(d):
            re = fmpq(0); im = fmpq(0)
            for r in range(2):
                i, j = r * d + a, r * d + b_
                for t in range(D2):
                    re += Are[i][t] * Are[j][t] + Aim[i][t] * Aim[j][t]
                    im += Aim[i][t] * Are[j][t] - Are[i][t] * Aim[j][t]
            Tre[a][b_] = re; Tim[a][b_] = im
    Tf = np.array([[float(Tre[a][b_]) + 1j * float(Tim[a][b_]) for b_ in range(d)] for a in range(d)])
    s = q(np.linalg.eigvalsh(Tf).max() * (1 + 1e-12) + 1e-14)
    for attempt in range(8):
        Mre = [[(s if a == b_ else fmpq(0)) - Tre[a][b_] for b_ in range(d)] for a in range(d)]
        Mim = [[-Tim[a][b_] for b_ in range(d)] for a in range(d)]
        okL, _ = ldl_pd(realembed_q(Mre, Mim))
        if okL: break
        s = s * fmpq(1001, 1000)
    # Tr(rho Z) = ||A^dag C||_F^2 / N2
    num = fmpq(0)
    for j in range(D2):          # column j of A, (A^dag C)_{j t} = sum_i conj(A_ij) C_it
        for t in range(C.shape[1]):
            re = fmpq(0); im = fmpq(0)
            for i in range(D2):
                re += Are[i][j] * Cre[i][t] + Aim[i][j] * Cim[i][t]
                im += Are[i][j] * Cim[i][t] - Aim[i][j] * Cre[i][t]
            num += re * re + im * im
    LB = num / (N2q * s) if okL else None
    return LB, UB, {"dual_num": dval, "primal_num": pval, "status": [st_d, st_p], "delta": delta, "UB_ok": ok, "LB_ok": okL}

def iv_of(x): return iv.mpf(int(x.p)) / iv.mpf(int(x.q))
def interval_k0(C0, N2q):
    """rho_0 = C0 C0^dag / N2 (2x2), Opt_0 = lambda_max, interval"""
    Cre = [[q(C0[i, t].real) for t in range(C0.shape[1])] for i in range(2)]; Cim = [[q(C0[i, t].imag) for t in range(C0.shape[1])] for i in range(2)]
    def ent(i, j):
        re = sum((Cre[i][t] * Cre[j][t] + Cim[i][t] * Cim[j][t] for t in range(C0.shape[1])), fmpq(0))
        im = sum((Cim[i][t] * Cre[j][t] - Cre[i][t] * Cim[j][t] for t in range(C0.shape[1])), fmpq(0))
        return re / N2q, im / N2q
    a, _ = ent(0, 0); d_, _ = ent(1, 1); bre, bim = ent(0, 1)
    disc = ((a - d_) / 2) ** 2 + bre * bre + bim * bim
    val = (iv_of(a) + iv_of(d_)) / 2 + iv.sqrt(iv_of(disc))
    return val, {"rho0": [float(a), float(d_), float(bre), float(bim)]}
def interval_k2(V, N2q):
    """rank one: Opt_2 = Tr G + 2 sqrt(det G), G = V V^dag / N2"""
    Vre = [[q(V[i, t].real) for t in range(V.shape[1])] for i in range(2)]; Vim = [[q(V[i, t].imag) for t in range(V.shape[1])] for i in range(2)]
    def ent(i, j):
        re = sum((Vre[i][t] * Vre[j][t] + Vim[i][t] * Vim[j][t] for t in range(V.shape[1])), fmpq(0))
        im = sum((Vim[i][t] * Vre[j][t] - Vre[i][t] * Vim[j][t] for t in range(V.shape[1])), fmpq(0))
        return re / N2q, im / N2q
    g00, _ = ent(0, 0); g11, _ = ent(1, 1); gre, gim = ent(0, 1)
    det = g00 * g11 - gre * gre - gim * gim
    assert det >= 0
    val = iv_of(g00 + g11) + 2 * iv.sqrt(iv_of(det))
    return val, {"G": [float(g00), float(g11), float(gre), float(gim)], "det": float(det)}

# ---------------- known-answer tests through the same code ----------------
def kat():
    res = {}
    s = 1 / math.sqrt(2)
    # (a) maximally entangled R-O qubit pair inside a k-block: C = 2d x 1, d = 2: |00>+|11>
    C = np.zeros((4, 1), complex); C[0, 0] = s; C[3, 0] = s
    lb, ub, _ = certify_general(C, fmpq(1), 1.0); res["a_max_entangled_block"] = [float(lb) / 2 * 2, float(ub)]
    # (b) product |0>|0>: Opt = 1 (F contribution 1/2)
    C = np.zeros((4, 1), complex); C[0, 0] = 1
    lb, ub, _ = certify_general(C, fmpq(1), 1.0); res["b_product_block"] = [float(lb), float(ub)]
    # (c) I/2 x rho_O with rho_O = diag(1/2,1/2): Opt = 1/2
    C = np.eye(4, dtype=complex) / 2
    lb, ub, _ = certify_general(C, fmpq(1), 1.0); res["c_maximally_mixed"] = [float(lb), float(ub)]
    # (d) classical correlated (|00><00| + |11><11|)/2: Opt = 1
    C = np.zeros((4, 2), complex); C[0, 0] = s; C[3, 1] = s
    lb, ub, _ = certify_general(C, fmpq(1), 1.0); res["d_classical"] = [float(lb), float(ub)]
    # (e) random rank-one, closed form vs certified SDP; (f) random 2x2 k=0 type vs lambda_max
    rng = np.random.default_rng(7)
    for tag, dk in (("e_rank1_closed_vs_sdp", 1), ("f_k0_closed_vs_sdp", 3)):
        if tag.startswith("e"):
            V = rng.normal(size=(2, 5)) + 1j * rng.normal(size=(2, 5)); V /= np.linalg.norm(V)
            ref, _ = interval_k2(V, fmpq(1)); C = V.reshape(10, 1)
            C = np.zeros((10, 1), complex)
            for r in range(2):
                for a in range(5): C[r * 5 + a, 0] = V[r, a]
        else:
            C0 = rng.normal(size=(2, dk)) + 1j * rng.normal(size=(2, dk)); C0 /= np.linalg.norm(C0)
            ref, _ = interval_k0(C0, fmpq(1)); C = C0
        lb, ub, _ = certify_general(C, fmpq(1), 1.0)
        res[tag] = [float(lb), float(ub), float(ref.a), float(ref.b)]
    return res

# ---------------- main ----------------
out = {"kat": kat(), "states": []}
for (arm, g), d in REPORTS.items():
    rp = os.path.join(d, "written_report.md")
    rmd5 = md5(rp)
    recs = [json.loads(l) for l in open(rp) if l.strip()]
    by = {}
    for r in recs: by.setdefault(r["kind"], []).append(r["data"])
    keys = [c for c in by["coherent_conversion"] if not c["dephased"]][0]["density"]["row_keys"]
    masks = [k[1] for k in keys]; assert all(k[0] == 0 for k in keys) and len(masks) == 276
    acts = {a["metadata"]["output"]: a for a in by["action"]}
    reg_plan = by["plans"][0]["registered"]
    for t in (1.0, 2.0, 4.0):
        regs = [x for x in by["register"] if x["state"]["end"] == t and x["role"] == "light" and not x.get("first_collected")]
        a = acts[regs[0]["state"]["output"]]
        m = a["metadata"]; assert m["plan"] == reg_plan and m["role"] == "primary" and m["start"] == 0.0 and m["end"] == t
        ap = os.path.join(d, a["raw_vector"]["array"]); amd5 = md5(ap)
        C = np.load(ap); C = C[C.files[0]]; assert C.shape == (2, 276)
        eps = float(m["error"])
        # ---- admission (exact) ----
        Cq = [[(q(C[r, i].real), q(C[r, i].imag)) for i in range(276)] for r in range(2)]
        gram = [[None, None], [None, None]]
        for r in range(2):
            for s_ in range(2):
                re = sum((Cq[r][i][0] * Cq[s_][i][0] + Cq[r][i][1] * Cq[s_][i][1] for i in range(276)), fmpq(0))
                im = sum((Cq[s_][i][0] * Cq[r][i][1] - Cq[s_][i][1] * Cq[r][i][0] for i in range(276)), fmpq(0))  # <v_s|v_r>
                gram[r][s_] = (re, im)
        N2q = gram[0][0][0] + gram[1][1][0]; N2f = float(N2q)
        dev = abs(N2q - 1); tol = (1 + q(eps)) ** 2 - 1
        # code Gram of the evolved codewords, G_rs = 2 <v_r, v_s>, exact; Frobenius distance from I_2 (upper bound via sqrt)
        G = [[(2 * gram[r][s_][0], 2 * gram[r][s_][1]) for s_ in range(2)] for r in range(2)]
        fro2 = sum(((G[r][s_][0] - (1 if r == s_ else 0)) ** 2 + G[r][s_][1] ** 2 for r in range(2) for s_ in range(2)), fmpq(0))
        fro_up = float((iv.sqrt(iv_of(fro2))).b)
        # own Gram tolerance: Psi = sum_r |r> v_r, v_r = codeword_r / sqrt2; sum_r ||v_r^ - v_r||^2 <= eps^2 (raw-state error eps), so
        # |<v_r^, v_s^> - <v_r, v_s>| <= eps (1/sqrt2 + eps) + eps / sqrt2 = sqrt2 eps + eps^2; G = 2 <v_r, v_s> deviates by <= 2 sqrt2 eps + 2 eps^2
        # per entry; Frobenius over the 2 x 2 matrix <= 2 x that
        gtol = 2 * (2 * math.sqrt(2) * eps + 2 * eps ** 2)
        adm = {"N2": [float((iv_of(N2q)).a), float((iv_of(N2q)).b)], "N2_exact_positive": N2q > 0, "abs_N2_minus_1": float(dev), "tol": float(tol),
               "within_tol": dev <= tol, "within_2e-12": dev <= fmpq(2, 10 ** 12), "gram_fro_upper": fro_up, "gram_tol": gtol, "gram_ok": fro_up <= gtol,
               "eps": eps, "divisors": ["N2 only (rho = C C^dag / N2)"]}
        adm["admitted"] = bool(adm["N2_exact_positive"] and adm["within_tol"] and adm["gram_ok"])
        st = {"arm": arm, "g": g, "t": t, "report_md5": rmd5, "report_md5_ok": rmd5 == REPORT_MD5[(arm, g)], "array": a["raw_vector"]["array"], "array_md5": amd5,
              "array_matches_wip": (a["raw_vector"]["array"], amd5) == WIP_ARRAYS[(arm, g, int(t))], "admission": adm, "outputs": {}}
        if not adm["admitted"]:
            out["states"].append(st); continue
        for oname, O in OUTPUTS.items():
            Oset = set(O); res = {}
            # graded split, O first: sign -1 iff the lower mode is in K and the higher in O
            blocks = {0: {}, 1: {}, 2: {}}
            for i, mask in enumerate(masks):
                ms = [b for b in range(NM) if (mask >> b) & 1]; m1, m2 = ms
                oa = tuple(x for x in ms if x in Oset); ka = tuple(x for x in ms if x not in Oset)
                sgn = -1.0 if (m1 not in Oset and m2 in Oset) else 1.0
                blocks[len(oa)].setdefault("rows", {}); blocks[len(oa)].setdefault("cols", {})
                blocks[len(oa)]["rows"].setdefault(oa, len(blocks[len(oa)]["rows"])); blocks[len(oa)]["cols"].setdefault(ka, len(blocks[len(oa)]["cols"]))
                blocks[len(oa)].setdefault("ent", []).append((oa, ka, i, sgn))
            mats = {}
            for k, bk in blocks.items():
                dO = len(bk["rows"]); dK = len(bk["cols"])
                M = np.zeros((2 * dO, dK), complex)
                for oa, ka, i, sgn in bk["ent"]:
                    for r in range(2): M[r * dO + bk["rows"][oa], bk["cols"][ka]] = sgn * C[r, i]
                mats[k] = M
            # closed forms
            v0, info0 = interval_k0(mats[0], N2q)
            dO2 = mats[2].shape[0] // 2; V = np.vstack([mats[2][:dO2, 0], mats[2][dO2:, 0]])
            v2, info2 = interval_k2(V, N2q)
            lb1, ub1, info1 = certify_general(mats[1], N2q, N2f)
            LB = (iv_of(lb1) + v0 + v2) / 2; UB = (iv_of(ub1) + v0 + v2) / 2
            lo, hi = float(LB.a), float(UB.b)
            # numerical cross-check: dual SDP on every block (k = 0, 2 compressed to supp(rho_O))
            xs = {}
            for k in (0, 2):
                Mk = mats[k]; dO = Mk.shape[0] // 2
                rhoO_cols = np.hstack([Mk[:dO], Mk[dO:]])
                U_, sv, _ = np.linalg.svd(rhoO_cols, full_matrices=False); Qb = U_[:, sv > 1e-13 * max(sv.max(), 1e-300)] if sv.max() > 0 else U_[:, :1]
                Mc = np.vstack([Qb.conj().T @ Mk[:dO], Qb.conj().T @ Mk[dO:]])
                rc = Mc @ Mc.conj().T / N2f
                _, dv, _ = sdp_dual(rc); xs[k] = dv
            closed = {0: float(v0.mid), 2: float(v2.mid)}
            phys = [max(0.0, lo - 2 * eps), hi + 2 * eps]
            gate = "PASS" if phys[0] > 0.5 else ("FAIL" if phys[1] <= 0.5 else "UNRESOLVED")
            res = {"F_saved": [lo, hi], "F_phys": phys, "gate": gate, "width": hi - lo,
                   "Opt0": [float(v0.a), float(v0.b)], "Opt2": [float(v2.a), float(v2.b)], "Opt1": [float(lb1), float(ub1)],
                   "k1_info": info1, "sdp_vs_closed": {k: xs[k] - closed[k] for k in (0, 2)},
                   "block_weights": {k: float(np.linalg.norm(mats[k]) ** 2 / N2f) for k in (0, 1, 2)}, "dims": {k: list(mats[k].shape) for k in mats}}
            st["outputs"][oname] = res
        out["states"].append(st)
        print(arm, g, t, {o: (r["F_saved"], r["gate"]) for o, r in st["outputs"].items()}, flush=True)
out["cpu_s"] = time.process_time() - t0; out["wall_s"] = time.time() - w0
out["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
json.dump(out, open(OUT, "w"), indent=1, default=str)
print(json.dumps(out["kat"]), out["cpu_s"], out["peak_rss_kib"])
