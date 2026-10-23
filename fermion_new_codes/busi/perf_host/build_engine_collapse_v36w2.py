#!/usr/bin/env python
"""build_engine_collapse_v36w2.py (busi lane PERF build, inbox #370 (1); final4/PERF/PERF_TRACKER.md row SYM-W1; final4/PERF/REVIEWA_symmetry_audit_v1.md
sec. 2 W2-S1): writes engine_collapse_v36w2.py from engine_collapse_v36v.py (md5 29111a34, the W2 lineage's production engine) by exact string
replacements in build_engine_collapse_v36w.py's form. Each OLD is copied exactly from engine_collapse_v36v.py and must occur exactly once (else the
build aborts); replacing every NEW by its OLD restores v36v byte for byte (checked here). Every new or changed line carries the marker "v36w2 cheb"
(checked here); the two propagation blocks are re-indented one level under an else (the v36v statements unchanged, each line marked).

SYM-W1: the H-bar eigendecompositions of the dg step (10 per fixed-point iterate, one per channel at the Gauss-averaged fields (Nbar, vbar), used for
nothing but the propagation X -> exp(-i tau_s H-bar) X of the sea and dust columns) replaced, under one option off by default, by the Chebyshev series
on the banded H-bar that the Page engine uses (page_ops_v6.cheb_order and cheb_expm):
  --hbar-prop eigh|chebyshev   (default eigh: v36v's statements, bitwise in every record, printed line, stored array and entry key; the json's "args"
                                dict gains hbar_prop, hbar_cheb_tol and hbar_prop_log, as v36p's and v36v's builds added theirs)
  --hbar-cheb-tol T            (default 1e-16) the Chebyshev degree K per call: the smallest K >= |z| + 4 with 2 sum_{k>K} |J_k(z)| <= T, z = w tau_s,
                                w the Gershgorin bound of H-bar (the largest absolute row sum, an upper bound on its spectral radius). Error bound: for a
                                Hermitian H with spectrum in [-w, w], || exp(-i tau H) - sum_{k<=K} c_k T_k(H/w) ||_2 <= 2 sum_{k>K} |J_k(w tau)| <= T
                                (Jacobi-Anger, |T_k| <= 1 on [-1, 1]), per unit column; rounding adds about K eps ||X|| (measured 4e-15 to 8e-15 absolute,
                                1e-14 relative against the eigh route, cheb_vs_eigh_v1.txt).
  --hbar-prop-log on|off       (default off) on: each sub-step entry gains "hbar_prop_v36w2" (the mode, the calls, the summed and largest Chebyshev
                                orders, the largest and summed tails, the block's wall with the device synchronized) and one printed "v36w2 cheb" line,
                                in either mode (the eigh arm of an A/B measures its own propagation block the same way). Under chebyshev it is always on.
H-bar's entries are formed exactly as clocked_gpu forms them (the channel's sparse h and Q in coordinate form, h_ij (N_i + N_j)/2 + Q_ij (v_i + v_j)/2,
duplicates summed), stored by diagonal (bandwidth 4 in the interleaved index), on the engine's device. The propagated columns differ from the eigh
route's at the rounding level (tolerance class; not bitwise): the eigendecompositions counted per sub-step and step fall by 10 per iterate.
Usage: python build_engine_collapse_v36w2.py [engine_collapse_v36v.py path] [engine_collapse_v36w2.py path]   (defaults: beside this script)."""
import hashlib
import os
import py_compile
import sys

MARK = "v36w2 cheb"
SRC_MD5 = "29111a348fd3443a9fdc1f94c8f36f2b"

OLD_K = '''        eig = eigh_all(chs, Nbar, vbar, bool(a.streams)); cols1 = {}   # v36k chart
        for ch in chs:   # v36k chart
            e_b, U_b = eig.pop(ch.kappa)   # v36k chart
            ph = torch.exp(-1j * e_b.to(torch.complex128) * tau_s)   # v36k chart
            prop = lambda X, U_b=U_b, ph=ph: U_b @ (ph[:, None] * (U_b.conj().T @ X))   # v36k chart
            sea0, dust0 = cols0[ch.kappa]   # v36k chart
            cols1[ch.kappa] = (prop(sea0), prop(dust0) if dust0.shape[1] else dust0)   # v36k chart
            del e_b, U_b   # v36k chart
        del eig   # v36k chart
'''

OLD_F = '''        eig = eigh_all(chs, Nbar, vbar, bool(a.streams))   # v36f dg
        cols1 = {}   # v36f dg
        for ch in chs:   # v36f dg
            e_b, U_b = eig.pop(ch.kappa)   # v36f dg
            ph = torch.exp(-1j * e_b.to(torch.complex128) * a.tau)   # v36f dg
            prop = lambda X, U_b=U_b, ph=ph: U_b @ (ph[:, None] * (U_b.conj().T @ X))   # v36f dg
            sea0, dust0 = cols0[ch.kappa]   # v36f dg
            cols1[ch.kappa] = (prop(sea0), prop(dust0) if dust0.shape[1] else dust0)   # v36f dg
            del e_b, U_b   # v36f dg
        del eig   # v36f dg
'''


def indent_marked(block):
    return "".join("    " + l_ + "; [v36w2 cheb: indented under the else]\n" for l_ in block.splitlines())


NEW_K = ('''        hbp_t0 = hbar_prop_t0_v36w2()   # v36w2 cheb: the propagation block's clock (None unless --hbar-prop-log on or chebyshev)
        if HBP["mode"] == "chebyshev":   # v36w2 cheb: exp(-i tau_s H-bar) X by the Chebyshev series on the banded H-bar (no eigendecomposition)
            cols1 = hbar_cheb_cols_v36w2(chs, Nbar, vbar, tau_s, cols0)   # v36w2 cheb
        else:   # v36w2 cheb: v36v's route (the eigendecompositions at H-bar, two products per block), its statements unchanged
''' + indent_marked(OLD_K) + '''        hbar_prop_t1_v36w2(hbp_t0)   # v36w2 cheb
''')

NEW_F = ('''        hbp_t0 = hbar_prop_t0_v36w2()   # v36w2 cheb
        if HBP["mode"] == "chebyshev":   # v36w2 cheb
            cols1 = hbar_cheb_cols_v36w2(chs, Nbar, vbar, a.tau, cols0)   # v36w2 cheb
        else:   # v36w2 cheb
''' + indent_marked(OLD_F) + '''        hbar_prop_t1_v36w2(hbp_t0)   # v36w2 cheb
''')

HELPERS = r'''HBP = {"mode": "eigh", "tol": 1e-16, "log": False, "calls": 0, "K_sum": 0, "K_max": 0, "tail_max": 0.0, "tail_sum": 0.0, "blocks": 0, "seconds": 0.0, "w_max": 0.0}   # v36w2 cheb: --hbar-prop, --hbar-cheb-tol, --hbar-prop-log and the running counters (per process)


def hbar_cheb_order_v36w2(z, tol):   # v36w2 cheb: page_ops_v6.cheb_order verbatim
    """v36w2 cheb (page_ops_v6.cheb_order): the smallest K >= |z| + 4 with 2 sum_{j>K} |J_j(|z|)| <= tol, and that tail; the tail summed from the computed coefficients up to |z| + 120."""   # v36w2 cheb
    from scipy.special import jv   # v36w2 cheb
    az = abs(float(z)); jmax = int(az + 120)   # v36w2 cheb
    J = np.abs(jv(np.arange(jmax + 1), az))   # v36w2 cheb
    tail = 2.0 * np.concatenate([np.cumsum(J[::-1])[::-1][1:], [0.0]])   # v36w2 cheb: tail[K] = 2 sum_{j>K} |J_j|
    K = int(math.ceil(az + 4))   # v36w2 cheb
    while K < jmax and tail[K] > tol:   # v36w2 cheb
        K += 1   # v36w2 cheb
    return K, float(tail[min(K, jmax)])   # v36w2 cheb


def hbar_bands_v36w2(ch, N, v):   # v36w2 cheb
    """v36w2 cheb: H-bar = h_m[N, v] by diagonals on the device, its entries formed exactly as clocked_gpu forms them (the same coordinate data, the same products, duplicates summed by index_put_ accumulate): returns (offsets, D) with D[j] the offset-d_j diagonal indexed by row (D[j][i] = H[i, i + d_j], zero outside), and the Gershgorin bound w = max_i sum_j |H_ij|."""   # v36w2 cheb
    torch = BK["torch"]; dev = BK["dev"]   # v36w2 cheb
    if getattr(ch, "_coo", None) is None:   # v36w2 cheb: clocked_gpu's cache, built the same way
        hc = ch.hs.tocoo(); qc = ch.Qs.tocoo()   # v36w2 cheb
        ch._coo = tuple(torch.as_tensor(x, dtype=dt, device=dev) for x, dt in ((hc.row, torch.int64), (hc.col, torch.int64), (hc.data, torch.complex128), (qc.row, torch.int64), (qc.col, torch.int64), (qc.data, torch.complex128)))   # v36w2 cheb
    hr, hcl, hv, qr, qcl, qv = ch._coo   # v36w2 cheb
    if getattr(ch, "_bands_v36w2", None) is None:   # v36w2 cheb: the offsets present in h and Q, and each entry's diagonal slot
        offs = sorted(set((hcl - hr).tolist()) | set((qcl - qr).tolist()))   # v36w2 cheb
        pos = {d_: j_ for j_, d_ in enumerate(offs)}   # v36w2 cheb
        ch._bands_v36w2 = (offs, torch.as_tensor([pos[int(d_)] for d_ in (hcl - hr).tolist()], dtype=torch.int64, device=dev), torch.as_tensor([pos[int(d_)] for d_ in (qcl - qr).tolist()], dtype=torch.int64, device=dev))   # v36w2 cheb
    offs, hslot, qslot = ch._bands_v36w2   # v36w2 cheb
    n = ch.hs.shape[0]   # v36w2 cheb
    Nd = torch.as_tensor(np.repeat(np.asarray(N, float), 2), dtype=torch.float64, device=dev)   # v36w2 cheb
    Vd = torch.as_tensor(np.repeat(np.asarray(v, float), 2), dtype=torch.float64, device=dev)   # v36w2 cheb
    D = torch.zeros((len(offs), n), dtype=torch.complex128, device=dev)   # v36w2 cheb
    D.index_put_((hslot, hr), hv * (0.5 * (Nd[hr] + Nd[hcl])).to(torch.complex128), accumulate=True)   # v36w2 cheb: clocked_gpu's h term
    D.index_put_((qslot, qr), qv * (0.5 * (Vd[qr] + Vd[qcl])).to(torch.complex128), accumulate=True)   # v36w2 cheb: clocked_gpu's Q term
    w = float(D.abs().sum(dim=0).max())   # v36w2 cheb: the Gershgorin bound (row i's entries are D[j][i] over j)
    return offs, D, w   # v36w2 cheb


def hbar_apply_v36w2(offs, D, X, out):   # v36w2 cheb
    """v36w2 cheb: out = H X for H by diagonals (offs, D); out is overwritten (a preallocated buffer of X's shape)."""   # v36w2 cheb
    n = X.shape[0]   # v36w2 cheb
    out.zero_()   # v36w2 cheb
    for j_, d_ in enumerate(offs):   # v36w2 cheb
        if d_ >= 0:   # v36w2 cheb
            out[: n - d_].addcmul_(D[j_, : n - d_, None], X[d_:])   # v36w2 cheb
        else:   # v36w2 cheb
            out[-d_:].addcmul_(D[j_, -d_:, None], X[: n + d_])   # v36w2 cheb
    return out   # v36w2 cheb


def hbar_cheb_expm_v36w2(offs, D, w, X, tau):   # v36w2 cheb: page_ops_v6.cheb_expm's series on preallocated buffers
    """v36w2 cheb: exp(-i tau H) X by the Chebyshev series on [-w, w] (page_ops_v6.cheb_expm): c_k = (2 - delta_k0) (-i)^k J_k(w tau), T_0 = X, T_1 = (H/w) X, T_{k+1} = 2 (H/w) T_k - T_{k-1}, the order by hbar_cheb_order_v36w2 at --hbar-cheb-tol. X is not written. Returns (Y, K, tail)."""   # v36w2 cheb
    from scipy.special import jv   # v36w2 cheb
    torch = BK["torch"]   # v36w2 cheb
    if X.shape[1] == 0 or tau == 0.0:   # v36w2 cheb
        return X.clone(), 0, 0.0   # v36w2 cheb
    z = w * tau   # v36w2 cheb
    K, tail = hbar_cheb_order_v36w2(z, HBP["tol"])   # v36w2 cheb
    k = np.arange(K + 1)   # v36w2 cheb
    ck = jv(k, z) * ((-1j) ** k) * np.where(k == 0, 1.0, 2.0)   # v36w2 cheb
    Dn = D / w   # v36w2 cheb: H/w
    T0 = X   # v36w2 cheb
    T1 = hbar_apply_v36w2(offs, Dn, X, torch.empty_like(X))   # v36w2 cheb
    Y = T0 * complex(ck[0]); Y.add_(T1 * complex(ck[1]))   # v36w2 cheb
    T2 = torch.empty_like(X)   # v36w2 cheb
    for kk in range(2, K + 1):   # v36w2 cheb
        hbar_apply_v36w2(offs, Dn, T1, T2)   # v36w2 cheb
        T2.mul_(2.0).sub_(T0)   # v36w2 cheb: T_kk = 2 (H/w) T_{kk-1} - T_{kk-2}
        Y.add_(T2 * complex(ck[kk]))   # v36w2 cheb
        spare = T0 if kk > 2 else torch.empty_like(X)   # v36w2 cheb: X itself is never reused as a buffer
        T0, T1, T2 = T1, T2, spare   # v36w2 cheb
    return Y, K, tail   # v36w2 cheb


def hbar_cheb_cols_v36w2(chs, N, v, tau, cols0):   # v36w2 cheb
    """v36w2 cheb: the dg step's propagated columns {kappa: (exp(-i tau H-bar) sea0, exp(-i tau H-bar) dust0)} without an eigendecomposition (H-bar = h_m[N, v] per channel), the counters in HBP."""   # v36w2 cheb
    out = {}   # v36w2 cheb
    for ch in chs:   # v36w2 cheb
        offs, D, w = hbar_bands_v36w2(ch, N, v)   # v36w2 cheb
        sea0, dust0 = cols0[ch.kappa]   # v36w2 cheb
        ys, Ks, ts = hbar_cheb_expm_v36w2(offs, D, w, sea0, tau)   # v36w2 cheb
        if dust0.shape[1]:   # v36w2 cheb
            yd, Kd, td = hbar_cheb_expm_v36w2(offs, D, w, dust0, tau)   # v36w2 cheb
        else:   # v36w2 cheb
            yd, Kd, td = dust0, 0, 0.0   # v36w2 cheb
        out[ch.kappa] = (ys, yd)   # v36w2 cheb
        for K_, t_ in ((Ks, ts), (Kd, td)):   # v36w2 cheb
            if K_:   # v36w2 cheb
                HBP["calls"] += 1; HBP["K_sum"] += K_; HBP["K_max"] = max(HBP["K_max"], K_); HBP["tail_max"] = max(HBP["tail_max"], t_); HBP["tail_sum"] += t_   # v36w2 cheb
        HBP["w_max"] = max(HBP["w_max"], w)   # v36w2 cheb
        del D   # v36w2 cheb
    return out   # v36w2 cheb


def hbar_prop_t0_v36w2():   # v36w2 cheb
    """v36w2 cheb: the propagation block's start clock (device synchronized) when it is logged, else None (v36v's statements alone)."""   # v36w2 cheb
    if not (HBP["log"] or HBP["mode"] == "chebyshev"):   # v36w2 cheb
        return None   # v36w2 cheb
    sync_dev()   # v36w2 cheb
    return time.time()   # v36w2 cheb


def hbar_prop_t1_v36w2(t0):   # v36w2 cheb
    if t0 is None:   # v36w2 cheb
        return   # v36w2 cheb
    sync_dev(); HBP["seconds"] += time.time() - t0; HBP["blocks"] += 1   # v36w2 cheb


def hbar_prop_snap_v36w2():   # v36w2 cheb
    return {k_: HBP[k_] for k_ in ("calls", "K_sum", "K_max", "tail_max", "tail_sum", "blocks", "seconds", "w_max")}   # v36w2 cheb


def hbar_prop_entry_v36w2(before):   # v36w2 cheb
    """v36w2 cheb: the sub-step's propagation record since `before` (a hbar_prop_snap_v36w2() copy): the mode, the blocks (one per iterate, all channels), their wall, the Chebyshev calls, the summed and largest orders, the summed and largest tails, the largest Gershgorin bound so far."""   # v36w2 cheb
    now = hbar_prop_snap_v36w2()   # v36w2 cheb
    return {"mode": HBP["mode"], "tol": HBP["tol"], "blocks": now["blocks"] - before["blocks"], "seconds": now["seconds"] - before["seconds"], "cheb_calls": now["calls"] - before["calls"], "cheb_K_sum": now["K_sum"] - before["K_sum"], "cheb_K_max": now["K_max"], "cheb_tail_sum": now["tail_sum"] - before["tail_sum"], "cheb_tail_max": now["tail_max"], "w_max": now["w_max"]}   # v36w2 cheb


def eigh_all(chs, N, v, streams=False):
'''

REPLACEMENTS = [
("docstring",
'''#!/usr/bin/env python
"""engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1): ''',
'''#!/usr/bin/env python
"""engine_collapse_v36w2.py (busi PERF build SYM-W1, inbox #370; built by build_engine_collapse_v36w2.py over engine_collapse_v36v.py md5 29111a34) [v36w2 cheb]
version 36w2 = version 36v with --hbar-prop chebyshev (default eigh: v36v bitwise): the dg step's H-bar propagation exp(-i tau_s H-bar) X by the [v36w2 cheb]
Chebyshev series on the banded H-bar (the Page engine's propagator; degree by the tail 2 sum_{k>K} |J_k(w tau_s)| <= --hbar-cheb-tol, default 1e-16, [v36w2 cheb]
w the Gershgorin bound) in place of the 10 per-iterate eigendecompositions; --hbar-prop-log on records the propagation block per sub-step. [v36w2 cheb]
[v36w2 cheb: v36v's docstring follows] engine_collapse_v36p.py (reviewer A for the busi lane; PRIM_ruling_v1): '''),
("the helpers (before eigh_all)", "def eigh_all(chs, N, v, streams=False):\n", HELPERS),
("the chart sub-step's propagation (dgk_substep_v36k)", OLD_K, NEW_K),
("the v36f step's propagation (dg_step_v36f)", OLD_F, NEW_F),
("the sub-step's snapshot (dgk_substep_v36k start)",
'''    t0 = time.time(); n_eigh0 = TM["eigh_n"]; n = len(R)   # v36k chart
    s_nodes = [q[0] for q in GL3_V36F]; w_nodes = [q[1] for q in GL3_V36F]   # v36k chart
    dust_only = (anchor is None)   # v36k chart
''',
'''    t0 = time.time(); n_eigh0 = TM["eigh_n"]; n = len(R)   # v36k chart
    hbp0 = hbar_prop_snap_v36w2()   # v36w2 cheb
    s_nodes = [q[0] for q in GL3_V36F]; w_nodes = [q[1] for q in GL3_V36F]   # v36k chart
    dust_only = (anchor is None)   # v36k chart
'''),
("the sub-step's entry (dgk_substep_v36k end)",
'''    ent["converged"] = conv; ent["why"] = why; ent["eigendecompositions"] = TM["eigh_n"] - n_eigh0; ent["solves"] = n_solves   # v36k chart
''',
'''    ent["converged"] = conv; ent["why"] = why; ent["eigendecompositions"] = TM["eigh_n"] - n_eigh0; ent["solves"] = n_solves   # v36k chart
    if HBP["log"] or HBP["mode"] == "chebyshev":   # v36w2 cheb: the propagation record (no key, no line under the defaults)
        ent["hbar_prop_v36w2"] = hbar_prop_entry_v36w2(hbp0); q_ = ent["hbar_prop_v36w2"]   # v36w2 cheb
        print("  v36w2 cheb T/rs %.4f sub-step %d/%d: H-bar propagation by %s: %d blocks in %.3f s (%.3f s a block); Chebyshev calls %d, order mean %s max %d, tail max %.1e (sum %.1e), w_max %.4g" % (T0 / rs, j + 1, nsub, q_["mode"], q_["blocks"], q_["seconds"], q_["seconds"] / max(q_["blocks"], 1), q_["cheb_calls"], ("%.1f" % (q_["cheb_K_sum"] / q_["cheb_calls"])) if q_["cheb_calls"] else "-", q_["cheb_K_max"], q_["cheb_tail_max"], q_["cheb_tail_sum"], q_["w_max"]), flush=True)   # v36w2 cheb
'''),
("the options",
'''    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:''',
'''    ap.add_argument("--hbar-prop", dest="hbar_prop", choices=["eigh", "chebyshev"], default="eigh", help="(v36w2) the dg step's H-bar propagation exp(-i tau_s H-bar) X: eigh (version 36v bitwise: the per-iterate eigendecompositions) or chebyshev (the Chebyshev series on the banded H-bar, the Page engine's propagator; tolerance class)")   # v36w2 cheb
    ap.add_argument("--hbar-cheb-tol", dest="hbar_cheb_tol", type=float, default=1e-16, help="(v36w2) the Chebyshev degree's tail bound: the smallest K >= |z| + 4 with 2 sum_{k>K} |J_k(z)| <= tol, z = w tau_s, w the Gershgorin bound (the operator-norm truncation error per unit column)")   # v36w2 cheb
    ap.add_argument("--hbar-prop-log", dest="hbar_prop_log", choices=["on", "off"], default="off", help="(v36w2) on: each dg sub-step entry gains hbar_prop_v36w2 and one printed line (the propagation block's wall, the Chebyshev orders and tails); always on under --hbar-prop chebyshev")   # v36w2 cheb
    a = ap.parse_args()
    HBP["mode"] = a.hbar_prop; HBP["tol"] = float(a.hbar_cheb_tol); HBP["log"] = (a.hbar_prop_log == "on")   # v36w2 cheb
    if a.m == 0.0 and not a.massless_ball:'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36w2: the OLD string of '%s' occurs %d times in the source (1 required): the source is not engine_collapse_v36v.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36w2: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36v.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36w2.py")
    raw = open(src, "rb").read()
    if md5(raw) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36w2: the source md5 is %s, not v36v's %s" % (md5(raw), SRC_MD5))
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text or md5(back.encode("utf-8")) != SRC_MD5:
        raise SystemExit("build_engine_collapse_v36w2: the reverse replacements do not restore v36v byte for byte")
    srcset = set(text.splitlines())
    unmarked = [l_ for l_ in out.splitlines() if l_ not in srcset and l_.strip() and MARK not in l_]
    if unmarked:
        raise SystemExit("build_engine_collapse_v36w2: %d new lines without the marker, the first: %r" % (len(unmarked), unmarked[0][:200]))
    marked = sum(1 for l_ in out.splitlines() if MARK in l_)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements, %d marked lines, every new non-blank line marked; source md5 %s; built md5 %s; the reverse replacements restore v36v byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), marked, md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
