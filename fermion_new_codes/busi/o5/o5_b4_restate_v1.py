# o5_b4_restate_v1: the restated B4 check of RUN_O5_controls_v1 section 5 (O5_controls_reading_v1 section 4),
# read on the box's CPU before any control-4 cube run. Rc20_k1 only: R_c 20 a, sigma 8 a, the chain grid r = 1 .. 1024 a.
import argparse, sys
import numpy as np

RC, SIG, NGRID, JMAX = 20.0, 8.0, 1024, 80
CUTS = (("r_h", 22.250080), ("1.25rs", 36.25))
KNOWN_STRICT, KNOWN_MID = 0.329149, 0.345711   # O5_witness_v1.2 section 10 T3; the lapse-scaling md's chain t 0 row
LIMIT_A, LIMIT_C = 1e-6, 1e-8                  # the RUN's B4 limit; this note's rule for part (c)

def load_src(path):
    z = np.load(path, allow_pickle=True)
    for k in ("coords", "values"):
        if k not in z.files:
            print("STOP: key %s absent in %s; keys %s" % (k, path, list(z.files))); sys.exit(3)
    c = np.asarray(z["coords"], dtype=float); v = np.asarray(z["values"])
    if c.ndim == 2 and c.shape[0] == 3 and c.shape[1] != 3: c = c.T
    if v.ndim == 2 and v.shape[0] == 4 and v.shape[1] != 4: v = v.T
    if c.ndim != 2 or v.ndim != 2 or c.shape[1] != 3 or v.shape[1] != 4 or c.shape[0] != v.shape[0]:
        print("STOP: shapes coords %s values %s in %s" % (c.shape, v.shape, path)); sys.exit(3)
    return c, v.astype(complex)

def chain_cells():
    j = np.arange(1, NGRID + 1, dtype=float)
    p = np.exp(-(j - RC) ** 2 / SIG ** 2)
    return j, p / p.sum()

def rules(j, p, cut):
    strict = p[j > cut].sum(); jc = int(np.rint(cut)); frac = (jc + 0.5) - cut; pc = p[j == jc][0]
    return strict, strict + frac * pc, frac, jc, pc

def shells(c, v):
    x = c - 0.5 * (c.max(axis=0) + c.min(axis=0))
    r = np.sqrt((x ** 2).sum(axis=1))
    return r, np.rint(r).astype(int), (np.abs(v) ** 2).sum(axis=1)

def finish(out, L, status):
    open(out + ".md", "w").write("# o5_b4_restate_v1\n\n" + "\n".join(l if l.startswith("|") else "- " + l for l in L) + "\n")
    print("\n".join(L)); sys.exit(status)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True); ap.add_argument("--default", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = []; status = 0
    j, p = chain_cells()
    s, m, frac, jc, pc = rules(j, p, CUTS[0][1])
    ok = abs(s - KNOWN_STRICT) <= 2e-6 and abs(m - KNOWN_MID) <= 2e-6
    L.append("closed-form chain cells p_j = exp(-(j - 20)^2/64) normalized over j = 1 .. 1024: strict beyond r_h %.6f (0.329149), midpoint %.6f (0.345711), cell %d weight %.6f, fraction beyond the cut %.6f: %s" % (s, m, jc, pc, frac, "ok" if ok else "STOP"))
    if not ok: finish(a.out, L, 3)
    ci, vi = load_src(a.iso); cd, vd = load_src(a.default)
    ri, ji, di = shells(ci, vi)
    W = np.array([di[ji == k].sum() for k in range(JMAX + 1)])
    w0 = W[0] / W.sum(); Wn = W[1:] / W[1:].sum(); pj = p[:JMAX]
    d = Wn - pj; tw = np.cumsum(Wn[::-1])[::-1]; tp = np.cumsum(pj[::-1])[::-1]; D = tw[1:] - tp[1:]
    kd = int(np.argmax(np.abs(d))); kD = int(np.argmax(np.abs(D)))
    ok_a = np.abs(d).max() <= LIMIT_A and np.abs(D).max() <= LIMIT_A
    L.append("(a) origin shell weight as written %.3e; after renormalization over shells 1 .. %d: largest |W_j - p_j| %.3e at j %d, largest cumulative difference beyond a cell boundary %.3e at %.1f a; limit %.0e: %s" % (w0, JMAX, abs(d[kd]), kd + 1, abs(D[kD]), kD + 1.5, LIMIT_A, "PASS" if ok_a else "FAIL"))
    if not ok_a: status = 2
    tot1 = W[1:].sum()
    for name, cut in CUTS:
        s_, m_, f_, jc_, pc_ = rules(j, p, cut)
        cw = di[ri > cut].sum() / di.sum(); cr = di[(ri > cut) & (ji >= 1)].sum() / tot1
        L.append("(b) %s %.6f a: cube beyond the cut as written %.6f (%+.2e to the midpoint rule), origin excluded %.6f (%+.2e); chain strict %.6f, midpoint %.6f (fraction %.4f of cell %d, weight %.3e); implied fraction of the straddling cell beyond the cut on the cube %.4f (print, no limit)" % (name, cut, cw, cw - m_, cr, cr - m_, s_, m_, f_, jc_, pc_, (cr - s_) / pc_))
    oi = np.lexsort(ci.T[::-1]); od = np.lexsort(cd.T[::-1])
    if ci.shape != cd.shape or not np.allclose(ci[oi], cd[od], atol=1e-9):
        L.append("(c) STOP: the site sets of the 2 files differ (%d against %d sites)" % (len(ci), len(cd))); finish(a.out, L, 2)
    vi_s, vd_s, ji_s = vi[oi], vd[od], ji[oi]
    ni, nd = np.abs(vi_s) ** 2, np.abs(vd_s) ** 2
    vmax = np.abs(vd_s).max(); worst_q = 0.0; worst_site = 0.0; mixing = 0; rows = []
    for k in range(1, JMAX):
        msk = ji_s == k
        if not msk.any() or p[k - 1] < 1e-12: continue
        Ni, Nd = ni[msk].sum(axis=0), nd[msk].sum(axis=0)
        use = Nd > 1e-20 * Nd.max()
        mixing += int(((~use) & (Ni > 1e-20 * Ni.max())).sum())
        q = Ni[use] / Nd[use]; spread = q.max() / q.min() - 1.0; worst_q = max(worst_q, float(spread))
        for c in range(4):
            mc = msk & (np.abs(vd_s[:, c]) > 1e-10 * vmax)
            if mc.sum() < 2: continue
            rho = vi_s[mc, c] / vd_s[mc, c]
            worst_site = max(worst_site, float(np.abs(rho / rho.mean() - 1.0).max()))
        rows.append((k, float(q.mean()), float(spread)))
    ok_c = worst_q <= LIMIT_C and worst_site <= LIMIT_C and mixing == 0
    L.append("(c) per shell, the 4 component weight ratios iso/default: largest spread among a shell's components %.3e; largest within-shell variation of the site-wise ratio iso/default %.3e; slots zero in the default and not in iso %d; limit %.0e: %s (section 4's derivation expects PASS: step (i) is the identity for kappa 1)" % (worst_q, worst_site, mixing, LIMIT_C, "PASS" if ok_c else "FAIL"))
    L.append("| shell j | weight ratio iso/default | component spread |"); L.append("|---|---|---|")
    for k, qm, sp in rows: L.append("| %d | %.6f | %.2e |" % (k, qm, sp))
    if not ok_c: status = 2
    L.append("VERDICT: %s" % ("PROCEED to control 4's cube run (convention I)" if status == 0 else "STOP: read before any control-4 cube run"))
    finish(a.out, L, status)

if __name__ == "__main__":
    main()
