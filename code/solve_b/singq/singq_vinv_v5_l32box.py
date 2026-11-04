# singq_vinv_v5_l32box.py -- (vref_r1, lead #561: = mark3_r2 singq_vinv_v5.py 4eff3368 with ONLY the GPU lock path moved to the box (<internal>/vinv_gpu.lock) and vref l16/code (singq_tierR_tan3 e1148a93) added to sys.path; the light weight comes from the spec, set to 32 there) -- = singq_vinv_v3.py (cd348eba) + for r1 configs that carry a light chain (keys w_l, k_l; relcha_r2 #531 nodes with light ON) the light chain
# e2.Chain('light', ns_l, k_l, 0.0, w_l) is added to the reference Jacobian T (route-2 adapter, as for the twins' light chain). Otherwise identical.
# singq_vinv_v3.py -- = singq_vinv_v2.py (cee4f788) with ONLY the tangent validation replaced. v2's check (T x directions at h 5e-9/5e-10) was
# roundoff-dominated: singq_tanscan_v2 (476b9d0f) shows FD error growing like 1/h on T x directions (best 1.8e-8 at h 1e-2, trapped on_928;
# 1.9e-6 on on_0) while normal directions match to 1e-10. v3: 4 directions (2 standard normal, 2 T x), central FD over the h ladder
# 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10 with Fields' clamp flag per h; a direction passes iff min over clamp-free h of rel < 1e-4; valid iff all pass.
# singq_vinv_v2.py -- = singq_vinv_v1.py (b0facea4) with (i) BATCHED assembly (512 columns per pass, singq_tierR_tan3 batched passes, same per-column
# arithmetic), and (ii) the corrected tangent validation: directions = T x (the operator's own 1/V-scaled sources), central differences at h = 5e-9
# and 5e-10 with Fields' positive-mass clamp flag recorded (must be 0), valid iff rel < 1e-4 at 5e-10 with no clamp; plus min M and min M/y^3.
# singq_vinv_v1.py -- SINGQ amend4 (aa71c799): dense VERIFIED APPROXIMATE INVERSE of F_u = I + B T at stored records (tier R-fp).
# CPU: assemble F~ (analytic T, exact signed tangent B; singq_tierR_tan2) and the entrywise assembly-error bound Delta. GPU (fp64, serialized by
# a file lock, waits for >= 32 GB free): R0 = inv(F~), C = R0 F~, e = ||I-C||_b + gamma_n || |R0||F~| ||_b + ||R0||_b ||Delta||_b + 2u(1+||C||_b),
# ||X||_b = sqrt(||X||_1 ||X||_inf) from inflated absolute sums. Accept e < 1: beta_R = ||R0||_b/(1-e).
# Usage: <out.jsonl> <worker> <n workers> <spec file>; spec: r2|label|t|rec npz|prep meta|light weight   or   r1|label|t|rec npz|cfg json
import sys, os, json, time, fcntl
import numpy as np
import scipy.sparse as sp
import torch
sys.path.insert(0, '<internal>/code'); sys.path.insert(0, '<internal>/code')
import m3r2_rows_v1 as rw
import m3r2_engine_v2 as e2
import m3r2_w2adapter_v2 as AD
import singq_tierR_tan2 as TAN
import singq_tierR_tan3 as TAN3
outp, wi, nw, specf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
u_ = 2.0 ** -53
LOCK = '<internal>/vinv_gpu.lock'
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try:
            done.add(json.loads(l)['label'])
        except Exception:
            pass
fo = open(outp, 'a')
nrm = lambda v_: float(np.linalg.norm(v_))
specs = [l.strip().split('|') for l in open(specf) if l.strip()]
for idx, spl in enumerate(specs):
    if idx % nw != wi or spl[1] in done:
        continue
    T0w = time.time()
    kind, label, tclk, recp, cfgp = spl[:5]
    z = np.load(recp)
    if kind == 'r2':
        meta = json.load(open(cfgp)); P = meta['params']
        ns, sig, hd, G, ext = P['NS_L'], P['sig'], P['hd'], P['G'], 4.0
        chains = [e2.Chain('heavy', P['NS_H'], int(k), P['m'], float(w)) for k, w in meta['weights'].items()]
        chains.append(e2.Chain('light', P['NS_L'], P['KL'], 0.0, float(spl[5])))
        x_rho, x_J = np.asarray(z['rho'], float), np.asarray(z['J'], float)
    else:
        cfg = json.load(open(cfgp))
        ns, sig, hd, G, ext = cfg['ns_l'], cfg['sigma'], cfg['hd'], cfg['G'], cfg['ext']
        chains = [e2.Chain('heavy', cfg['ns_h'], int(k), cfg['m'], float(w)) for (k, cols, w) in cfg['chans']]
        if 'w_l' in cfg:
            chains.append(e2.Chain('light', cfg['ns_l'], int(cfg['k_l']), 0.0, float(cfg['w_l'])))
        x_rho, x_J = np.asarray(z['x'][:ns], float), np.asarray(z['x'][ns:], float)
    u = np.concatenate([np.asarray(z['Nb'], float), np.asarray(z['vb'], float)])
    kern = rw.Kernel(ns, 1.0, hd, sig, yend=ns + ext * sig); fs = e2.Fields(kern, G)
    rows, cols, vals = [], [], []
    def add(Ms, r0, c0):
        Mc = Ms.tocoo(); rows.append(Mc.row + r0); cols.append(Mc.col + c0); vals.append(Mc.data)
    for c in chains:
        if c.weight == 0:
            continue
        Cc = AD._coef(c); n = c.ns
        r = np.arange(1, n + 1, dtype=float); Vc = 4 * np.pi * r ** 2
        N = u[:n]; v = u[ns:ns + n]; Dm = sp.csr_matrix(Cc['D']); A, Bc, Cq = Cc['A'], Cc['B'], Cc['C']
        g = Dm @ v; F = A * g ** 2 - 2 * Bc * v * g + Cq * v ** 2
        Fv = sp.diags(2 * A * g - 2 * Bc * v) @ Dm + sp.diags(-2 * Bc * g + 2 * Cq * v)
        HNN = sp.diags(-2 * F / N ** 3); HNv = sp.diags(1 / N ** 2) @ Fv
        Hvv = -(2 * (Dm.T @ sp.diags(A / N) @ Dm) - 2 * (sp.diags(Bc / N) @ Dm + Dm.T @ sp.diags(Bc / N)) + 2 * sp.diags(Cq / N))
        S = sp.diags(c.weight / Vc)
        add(S @ HNN, 0, 0); add(S @ HNv, 0, ns); add(S @ HNv.T, ns, 0); add(S @ Hvv, ns, ns)
    Tm = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * ns, 2 * ns))
    f = fs.solve(x_rho, x_J); y = kern.y; M, w, lam = f['M'], f['w'], f['lam']; Jl = f['Jl']
    spv = np.zeros_like(M); pos = M > 0; spv[pos] = 4 * np.pi * y[pos] ** 2 * Jl[pos] * w[pos] / (2 * M[pos])
    tr, tt = f['tail']; MI = float(M[-1]); Tp = -tt * np.sqrt(2 * G) / (2 * np.sqrt(MI)) if MI > 0 else 0.0
    W0c, W1c = kern.W0.tocsc(), kern.W1.tocsc(); W0T, W1T = kern.W0T, kern.W1T; tau1 = kern.tau1; V = kern.V
    W0a, W1a, W0Ta, W1Ta = abs(W0c), abs(W1c), abs(W0T), abs(W1T); Nd = f['N']
    def Bsigned(xr, xJ):
        drl = W0c @ xr; dJl = W1c @ xJ; dtail = float(tau1 @ xJ)
        dM, dN, dw = TAN.tan_pass(drl, dJl, y, hd, M, w, spv, lam, Tp, tt, MI, G, dtail)
        dv = dw * Nd + w * dN
        return np.concatenate([(W0T @ dN) / V, (W1T @ dv - (np.sqrt(2 * G) * dM[-1] / (2 * np.sqrt(MI)) if MI > 0 else 0.0) * tau1) / V])
    def Bhat(xr, xJ):
        drl = W0a @ xr; dJl = W1a @ xJ; dtail = float(np.abs(tau1) @ xJ)
        dM, dN, dw = TAN.maj_pass(drl, dJl, y, hd, M, w, spv, lam, Tp, tt, MI, G, dtail)
        dv = dw * np.abs(Nd) + np.abs(w) * dN
        return np.concatenate([(W0Ta @ dN) / V, (W1Ta @ dv + (np.sqrt(2 * G) * dM[-1] / (2 * np.sqrt(MI)) if MI > 0 else 0.0) * np.abs(tau1)) / V])
    def Bfd(ds, h):
        sb = np.concatenate([x_rho, x_J]); sc = h * float(np.max(np.abs(sb))) / max(float(np.max(np.abs(ds))), 1e-300)
        p = fs.solve(sb[:ns] + sc * ds[:ns], sb[ns:] + sc * ds[ns:]); m = fs.solve(sb[:ns] - sc * ds[:ns], sb[ns:] - sc * ds[ns:])
        return np.concatenate([p['Nb'] - m['Nb'], p['vb'] - m['vb']]) / (2 * sc), int(p['flag']) | int(m['flag'])
    rng = np.random.default_rng(11); tchk = []
    for fam in ('normal', 'normal', 'Tx', 'Tx'):
        g = rng.standard_normal(2 * ns); xx = g if fam == 'normal' else Tm @ g
        bs = Bsigned(xx[:ns], xx[ns:]); lad = []
        for hh in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
            bf, fl = Bfd(xx, hh); lad.append(dict(h=hh, rel=nrm(bs - bf) / max(nrm(bf), 1e-300), clamp_flag=fl))
        ok_ = [r_['rel'] for r_ in lad if r_['clamp_flag'] == 0]
        tchk.append(dict(fam=fam, ladder=lad, best_rel=(min(ok_) if ok_ else None), passed=bool(ok_) and min(ok_) < 1e-4))
    tangent_valid = all(d_['passed'] for d_ in tchk)
    Mmin = float(np.min(M[1:])); Mr3min = float(np.min(M[1:] / y[1:] ** 3))
    nd = 2 * ns; rnd = 64 * y.size * u_
    Fm = np.eye(nd); Dl = np.zeros((nd, nd))
    Td = Tm.tocsc(); KB = 512
    W0c_, W1c_ = W0c, W1c
    for j0 in range(0, nd, KB):
        j1 = min(nd, j0 + KB); Tb = Td[:, j0:j1].toarray()
        if not Tb.any():
            continue
        Xr, XJ = Tb[:ns], Tb[ns:]
        dM, dN, dw = TAN3.tan_pass_b(np.ascontiguousarray(W0c_ @ Xr), np.ascontiguousarray(W1c_ @ XJ), y, hd, M, w, spv, lam, Tp, tt, MI, G, tau1 @ XJ)
        dv = dw * Nd[:, None] + w[:, None] * dN
        colB = np.vstack([(W0T @ dN) / V[:, None], (W1T @ dv - (np.sqrt(2 * G) / (2 * np.sqrt(MI)) if MI > 0 else 0.0) * np.outer(tau1, dM[-1])) / V[:, None]])
        aM, aN, aw = TAN3.maj_pass_b(np.ascontiguousarray(W0a @ np.abs(Xr)), np.ascontiguousarray(W1a @ np.abs(XJ)), y, hd, M, w, spv, lam, Tp, tt, MI, G, np.abs(tau1) @ np.abs(XJ))
        av = aw * np.abs(Nd)[:, None] + np.abs(w)[:, None] * aN
        bh = np.vstack([(W0Ta @ aN) / V[:, None], (W1Ta @ av + (np.sqrt(2 * G) / (2 * np.sqrt(MI)) if MI > 0 else 0.0) * np.outer(np.abs(tau1), aM[-1])) / V[:, None]])
        Fm[:, j0:j1] += colB
        Dl[:, j0:j1] = bh * rnd + 64 * u_ * np.abs(colB)
    t_asm = time.time() - T0w
    def nb_(X):              # ||X||_b on the GPU from absolute sums, inflated
        Xa = X.abs(); n1 = float(Xa.sum(dim=0).max()); ni = float(Xa.sum(dim=1).max())
        return float(np.sqrt(n1 * ni) * (1 + nd * u_))
    lf = open(LOCK, 'w'); fcntl.flock(lf, fcntl.LOCK_EX)
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free >= 32:
            break
        time.sleep(20)
    T1 = time.time()
    dev = torch.device('cuda')
    F_ = torch.as_tensor(Fm, device=dev)
    R0 = torch.linalg.inv(F_)
    C = R0 @ F_
    I_ = torch.eye(nd, device=dev, dtype=torch.float64)
    ec = nb_(I_ - C); cn = nb_(C); del I_
    absprod = R0.abs() @ F_.abs(); gam = nd * u_ / (1 - nd * u_); gterm = gam * nb_(absprod); del absprod, C
    r0n = nb_(R0); del F_, R0
    Dn = nb_(torch.as_tensor(Dl, device=dev))
    torch.cuda.empty_cache()
    fcntl.flock(lf, fcntl.LOCK_UN); lf.close()
    e = ec + gterm + r0n * Dn + 2 * u_ * (1 + cn)
    ok = bool(e < 1) and tangent_valid
    beta = r0n / (1 - e) if ok else None
    row = dict(label=label, t=float(tclk), kind=kind, tier='R-fp', tangent_check=tchk, tangent_valid=tangent_valid, M_min=Mmin, M_over_y3_min=Mr3min, e=e, e_comp=ec, gemm_term=gterm, R0_norm_b=r0n,
               Delta_norm_b=Dn, accept=ok, beta_R=beta, sigma_min_lower=(1 / beta if ok else None), t_assembly=t_asm, t_gpu=time.time() - T1,
               wall=time.time() - T0w)
    fo.write(json.dumps(row, default=float) + '\n'); fo.flush(); done.add(label)
    print(label, f"e {e:.3e} beta_R {beta} asm {t_asm:.0f}s gpu {row['t_gpu']:.1f}s", flush=True)
