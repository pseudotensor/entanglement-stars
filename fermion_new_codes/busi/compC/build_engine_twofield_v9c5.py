#!/usr/bin/env python
"""build_engine_twofield_v9c5.py (busi lane, PERF tracker rows SYM-C1, SYM-C2; reviewer A symmetry audit final4/PERF/REVIEWA_symmetry_audit_v1.md
secs. 1(b) and 3, C-S1 and C-S2): engine_twofield_v9c5.py over engine_twofield_v9c4.py (md5 152ec1c6193f452922d343196dbf9b18) in the lane's
patch form: exact replacements, each OLD occurring once in v9c4; every line of v9c5 not in v9c4 carries "[v9c5 perf]"; --reverse restores v9c4
byte for byte; the source md5 is checked; an existing output is never overwritten.

The structure (audit sec. 1(b)): a light (massless) channel's clocked h_m = (1/2){N, h_0} + (1/2){v, Q} is purely imaginary and commutes with
S = sum_n (-1)^n sigma_y; in the basis p_i = (1, i (-1)^i)/sqrt 2 (S = +1) and q_i = conj(p_i) (S = -1) the off-sector block is zero and the
- block is -conj(H_+): the - sector's eigenpairs are (-e_+, conj(P_+ u_+)). For K = h - a S (the walks' operator) the same holds with
K_+ = H_+ - a and K_- = -conj(K_+), so |spec K_-| = |spec K_+| and K^2 - d^2 is positive definite exactly when K_+^2 - d^2 is.

Two options, each off by default (then every statement executed is v9c4's), consumed by an argparse action into the module dict _SYMC and never
written into the parsed namespace (the json's args equal v9c4's; H3's plumbing):

  --perf-sym-light-sector (SYM-C1, audit C-S1): every light-channel eigendecomposition (v9c2_eig_light with its cache under its own key,
      LightPathV9.eig, light_continue_sector_v9's recomputation at the previous fields, light_init_v36, light_resume_v36) is the n x n eigh of
      H_+ = P_+^H h_m P_+ (built on the device from clocked_gpu's matrix), lifted: e = sorted(e_+ U -e_+), U the lifted columns P_+ u_+ and
      conj(P_+ u_+) in that order, and inside every numerically degenerate block (engine_ops_v3.blocks_of_v3 at --light-tol-block, the
      rotation's own blocks) the S = -1 columns first: the order light_sectorize_v36's rotation leaves (torch eigh of S on the block is ascending).
      The off-sector block is computed on every call; above 1e-14 of max |H_+| the call falls back to v9c4's full eigh (counted).
      light_sectorize_v36 then returns sigma = sign(<u|S|u>) (exactly +-1), no rotated block, the measured impurity, whenever the measured
      impurity is below 1e-10; otherwise v9c4's rotation runs unchanged.
  --perf-sym-ladder-sector (SYM-C2, audit C-S2): the light walks' banded Cholesky ladders (light_walk_v36 and light_sector_walk_v9, with H3's
      B1 jobs, which read the same seg entries) test K_+(s)^2 - d^2 (n x n, bandwidth 2, K_+^2 bandwidth 4) in place of the full K(s)^2 - d^2
      (2n x 2n, bandwidth 4 and 8). Deviation from the audit's text, kept conservative: the Gershgorin quantities (||dK||_G, G_K, the rounding
      floor F = 64 eps G_K^2, d_floor, the slack) stay those of the full matrices, as in v9c4, so the step rule, floors and certificate numbers
      are v9c4's and the walk's labels, pieces and Cholesky counts are equal whenever every ladder verdict is. The pair's off-sector blocks are
      checked on every build (exact zero required; otherwise that anchor keeps the full matrices, counted).

Diagnostics only: counters (sector eighs, their seconds, fallbacks, the largest off-sector ratio; sector ladder pairs, full-matrix fallbacks)
printed on one "v9c5 perf (the run)" line at the end, and a "v9c5 perf:" settings line at the start, both only when an option is on.

Not covered (left on v9c4's full path, counted nowhere): the record-scale fallback's continuation (engine_ops_v3.continue_sea_v3 calls eig_at on
the light channel inside the flagged LZ fallback), the heavy channels (massive: audit sec. 1(a), no exact reduction).

Usage. python build_engine_twofield_v9c5.py engine_twofield_v9c4.py [engine_twofield_v9c5.py]
         --check: the build in memory with the marks check and the reverse round trip (nothing written)
         --reverse engine_twofield_v9c5.py restored.py [--against engine_twofield_v9c4.py]
"""
import argparse
import hashlib
import os
import sys

MARK = "[v9c5 perf]"
SOURCE_MD5 = "152ec1c6193f452922d343196dbf9b18"   # engine_twofield_v9c4.py

BLOCK = r'''# >>> v9c5 perf: PERF SYM-C1, SYM-C2 (reviewer A symmetry audit secs. 1(b), 3): the light channels' eigendecomposition per sector (the n x n eigh of H_+, lifted) and the walks' Cholesky ladders on K_+; both off by default, and then every statement executed is v9c4's
import argparse as _symc_argparse
import threading as _symc_threading

V9C5_VERSION = "engine_twofield_v9c5"
_SYMC = {"light": False, "ladder": False, "tol_block": 1e-8}
_SYMC_STATS = {}
_SYMC_LOCK = _symc_threading.Lock()
_SYMC_SIGN = {}


def _symc_inc(k, n=1):
    # diagnostic counters only (no numerical effect)
    with _SYMC_LOCK:
        _SYMC_STATS[k] = _SYMC_STATS.get(k, 0) + n


def _symc_max(k, x):
    with _SYMC_LOCK:
        _SYMC_STATS[k] = max(_SYMC_STATS.get(k, 0.0), float(x))


class _SymcAction(_symc_argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        _SYMC[self.dest] = True
        # Deliberately do not modify namespace: output args stay v9c4's.


def _symc_add_arguments(ap):
    ap.add_argument("--perf-sym-light-sector", dest="light", nargs=0, action=_SymcAction, default=_symc_argparse.SUPPRESS,
                    help="SYM-C1: the light eigendecomposition by the S = +1 sector block (tolerance class, exact labels); default off")
    ap.add_argument("--perf-sym-ladder-sector", dest="ladder", nargs=0, action=_SymcAction, default=_symc_argparse.SUPPRESS,
                    help="SYM-C2: the light walks' Cholesky ladders on the S = +1 sector block K_+ (tolerance class, same verdicts); default off")


def symc_set(a):
    _SYMC["tol_block"] = float(getattr(a, "light_tol_block", 1e-8))
    if _SYMC["light"] or _SYMC["ladder"]:
        print("v9c5 perf: %s; light_sector %s, ladder_sector %s, tol_block %.1e (tolerance class with exact labels; the options are kept out of the parsed namespace, so the json's args equal v9c4's)" % (
            V9C5_VERSION, "on" if _SYMC["light"] else "off", "on" if _SYMC["ladder"] else "off", _SYMC["tol_block"]), flush=True)


def symc_close():
    if (_SYMC["light"] or _SYMC["ladder"]) and _SYMC_STATS:
        print("  v9c5 perf (the run): %s" % ", ".join("%s %s" % (k, (("%.3g" % v) if isinstance(v, float) else v)) for k, v in sorted(_SYMC_STATS.items())), flush=True)


def _symc_sign_dev(n, device):
    torch = BK["torch"]
    key = (int(n), str(device))
    s = _SYMC_SIGN.get(key)
    if s is None:
        s = torch.as_tensor(light_sign_v36(n), dtype=torch.float64, device=device).to(torch.complex128)
        _SYMC_SIGN[key] = s
    return s


def _symc_eig_light(ch, N, v):
    """SYM-C1: (e, U) as eigT(clocked_gpu(ch, N, v)) returns them, from the n x n eigh of the S = +1 block (the module docstring's order)."""
    torch = BK["torch"]
    t_ = time.time()
    M = clocked_gpu(ch, N, v)
    n2 = int(M.shape[0]); n = n2 // 2
    sc = _symc_sign_dev(n, M.device)
    h00 = M[0::2, 0::2]; h01 = M[0::2, 1::2]; h10 = M[1::2, 0::2]; h11 = M[1::2, 1::2]
    a01 = h01 * sc[None, :]; a10 = sc[:, None] * h10; a11 = (sc[:, None] * h11) * sc[None, :]
    Hp = 0.5 * (h00 + 1j * a01 - 1j * a10 + a11)
    off = 0.5 * (h00 + 1j * a01 + 1j * a10 - a11)
    offmax = float(off.abs().max()); scale = float(Hp.abs().max())
    del off, a01, a10, a11, h00, h01, h10, h11
    _symc_max("light_offblock_ratio_max", offmax / max(scale, 1e-300))
    if offmax > 1e-14 * scale:
        _symc_inc("light_offblock_fallbacks")
        return eigT(M)
    del M
    ep, up = torch.linalg.eigh(Hp)
    ep = ep.real
    del Hp
    r = 1.0 / math.sqrt(2.0)
    Up = torch.empty((n2, n), dtype=torch.complex128, device=up.device)
    Up[0::2, :] = up * r
    Up[1::2, :] = (1j * sc[:, None]) * (up * r)
    del up
    e_all = torch.cat([ep, -ep])
    order = torch.argsort(e_all, stable=True)
    e_s = e_all[order]
    e_np = e_s.cpu().numpy().astype(float)
    order_np = order.cpu().numpy().astype(np.int64)
    sg = np.where(order_np < n, 1.0, -1.0)
    perm = order_np.copy()
    for (i, j) in OPS3.blocks_of_v3(e_np, _SYMC["tol_block"]):
        if j - i > 1:
            sub = np.argsort(sg[i:j], kind="stable")
            perm[i:j] = perm[i:j][sub]
    Uf = torch.empty((n2, n2), dtype=torch.complex128, device=Up.device)
    Uf[:, :n] = Up
    Uf[:, n:] = Up.conj()
    del Up
    U = Uf.index_select(1, torch.as_tensor(perm, dtype=torch.int64, device=Uf.device))
    del Uf
    _symc_inc("light_sector_eigh"); _symc_inc("light_sector_eigh_s", time.time() - t_)
    return e_s, U


def _symc_sectorize(e, U, sgn_dev):
    """SYM-C1: (sigma, rotated blocks, impurity) for a sector-built U (impurity below 1e-10): sigma the sign of <u|S|u>, exactly +-1; None otherwise."""
    torch = BK["torch"]
    sigma = light_sigma_v36(U, sgn_dev)
    imp = float((sigma.abs() - 1.0).abs().max()) if sigma.numel() else 0.0
    if imp > 1e-10:
        _symc_inc("sectorize_rotation_kept")
        return None
    _symc_inc("sectorize_exact"); _symc_max("sectorize_impurity_max", imp)
    return torch.sign(sigma), 0, imp


def _symc_plus_host(K):
    """SYM-C2: the S = +1 block K_+ = P_+^H K P_+ (n x n host CSR) of a 2n x 2n host CSR K, and its off-sector block's largest modulus."""
    n = K.shape[0] // 2
    K = K.tocsr()
    D = sp.diags(light_sign_v36(n))
    K00 = K[0::2, 0::2]; K01 = K[0::2, 1::2] @ D; K10 = D @ K[1::2, 0::2]; K11 = D @ K[1::2, 1::2] @ D
    Kp = (0.5 * (K00 + 1j * K01 - 1j * K10 + K11)).tocsr()
    Ko = (0.5 * (K00 + 1j * K01 + 1j * K10 - K11)).tocsr()
    Kp.sort_indices()
    Kp = Kp.astype(np.complex128)
    return Kp, (float(abs(Ko).max()) if Ko.nnz else 0.0)


def _symc_plus_pair(K0, K1, dK):
    """SYM-C2: (K0_+, dK_+) for the ladders, K_+(t) = K0_+ + t dK_+; the full (K0, dK) when an off-sector block is not exactly zero."""
    K0p, o0 = _symc_plus_host(K0)
    K1p, o1 = _symc_plus_host(K1)
    if o0 != 0.0 or o1 != 0.0:
        _symc_inc("ladder_offblock_fallbacks"); _symc_max("ladder_offblock_max", max(o0, o1))
        return K0, dK
    _symc_inc("ladder_sector_pairs")
    return K0p, (K1p - K0p).tocsr()
# <<< v9c5 perf'''

OLD_SECT = '    float64), rotated blocks, the sector impurity max | |sigma| - 1 |)."""\n'
SECT_INS = '    if _SYMC["light"]:\n        out_symc = _symc_sectorize(e, U, sgn_dev)\n        if out_symc is not None:\n            return out_symc\n'

OLD_WALK_GK = '    GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))                                                 # v3\n'
WALK_INS = '    if _SYMC["ladder"]:\n        K0, dK = _symc_plus_pair(K0, K1, dK)\n'

OLD_SECW_GK = '                nG = OPS3.gershgorin(dK); GK = max(OPS3.gershgorin(K0), OPS3.gershgorin(K1))              # v9 sector\n'
SECW_INS = '                if _SYMC["ladder"]:\n                    K0, dK = _symc_plus_pair(K0, K1, dK)\n'

OLD_PATH_EIG = '        e, U = eigT(clocked_gpu(self.ch, N, v)); self.eigh_n += 1                                        # v9 sector\n'
PATH_HEAD = '        if _SYMC["light"]:\n            e, U = _symc_eig_light(self.ch, N, v); self.eigh_n += 1\n        else:\n'

OLD_PREV_EIG = '        e0t, U0 = eigT(clocked_gpu(ch, prev["f"][0], prev["f"][1])); sg0, _n0, _i0 = light_sectorize_v36(e0t, U0, sgn_dev, a.light_tol_block)   # v9 sector\n'
PREV_HEAD = '        if _SYMC["light"]:\n            e0t, U0 = _symc_eig_light(ch, prev["f"][0], prev["f"][1]); sg0, _n0, _i0 = light_sectorize_v36(e0t, U0, sgn_dev, a.light_tol_block)\n        else:\n'

OLD_INIT_EIG = '        e, U = eigT(clocked_gpu(chL, N, v))\n        sigma, n_rot, imp = light_sectorize_v36(e, U, sgn_dev, a.light_tol_block)\n'
INIT_HEAD = '        if _SYMC["light"]:\n            e, U = _symc_eig_light(chL, N, v)\n        else:\n'

OLD_RES_EIG = '        e, U = eigT(clocked_gpu(chL, N_s, v_s))\n'
RES_HEAD = '        if _SYMC["light"]:\n            e, U = _symc_eig_light(chL, N_s, v_s)\n        else:\n'

OLD_C2EIG = 'def v9c2_eig_light(chL, N, v):  # [v9c2 perf]\n'
C2EIG_INS = ('    if _SYMC["light"]:\n'
             '        if not V9C2["eig_cap"]:\n'
             '            return _symc_eig_light(chL, N, v)\n'
             '        try:\n'
             '            key_symc = ("light-sym", chL.key, _v9c2_akey(N), _v9c2_akey(v))\n'
             '        except _V9c2NoCache:\n'
             '            key_symc = None\n'
             '        return _v9c2_spectral(key_symc, lambda: _symc_eig_light(chL, N, v), 1, None)\n')

ANCHOR_BLOCK = "\n\n" + "def light_walk_v36(ch, f_prev, f_cur, mu0, mu1, g0, S, d_min=1e-7, max_pieces=4096):\n"


class BuildStop(Exception):
    pass


class Rep(object):
    def __init__(self, name, old, new):
        self.name = name; self.old = old; self.new = new


def mark(text):
    out = []
    for line in text.split("\n"):
        out.append(line if (line.strip() == "" or MARK in line) else line + "  # " + MARK)
    return "\n".join(out)


def indent4(t):
    return "\n".join(("    " + x) if x.strip() else x for x in t.rstrip("\n").split("\n")) + "\n"


def after(old, ins):
    return old + mark(ins.rstrip("\n")) + "\n"


def branch(old, head):
    # head's last line is "else:"; the original statement(s) re-indented under it, marked (the reverse restores them verbatim)
    return mark(head.rstrip("\n")) + "\n" + mark(indent4(old).rstrip("\n")) + "\n"


def replacements():
    M = "  # " + MARK
    first, rest = OLD_INIT_EIG.split("\n", 1)
    return [
        Rep("P1 block", ANCHOR_BLOCK, "\n\n" + mark(BLOCK) + "\n\n" + ANCHOR_BLOCK),
        Rep("P2 args", "    _h3_add_arguments(ap)  # [v9c4 perf]\n", "    _h3_add_arguments(ap)  # [v9c4 perf]\n    _symc_add_arguments(ap)" + M + "\n"),
        Rep("P3 set", "    h3_report()  # [v9c4 perf]\n", "    h3_report()  # [v9c4 perf]\n    symc_set(a)" + M + "\n"),
        Rep("P3 close", "        h3_close()  # [v9c4 perf]\n", "        symc_close()" + M + "\n        h3_close()  # [v9c4 perf]\n"),
        Rep("C1 sectorize", OLD_SECT, after(OLD_SECT, SECT_INS)),
        Rep("C1 v9c2_eig_light", OLD_C2EIG, after(OLD_C2EIG, C2EIG_INS)),
        Rep("C1 LightPathV9.eig", OLD_PATH_EIG, branch(OLD_PATH_EIG, PATH_HEAD)),
        Rep("C1 previous fields", OLD_PREV_EIG, branch(OLD_PREV_EIG, PREV_HEAD)),
        Rep("C1 init", OLD_INIT_EIG, branch(first + "\n", INIT_HEAD) + rest),
        Rep("C1 resume", OLD_RES_EIG, branch(OLD_RES_EIG, RES_HEAD)),
        Rep("C2 light_walk_v36", OLD_WALK_GK, after(OLD_WALK_GK, WALK_INS)),
        Rep("C2 light_sector_walk_v9", OLD_SECW_GK, after(OLD_SECW_GK, SECW_INS)),
    ]


def precheck(text, reps, reverse=False):
    return [(r, text.count(r.new if reverse else r.old)) for r in reps if text.count(r.new if reverse else r.old) != 1]


def apply(text, reps, reverse=False):
    bad = precheck(text, reps, reverse)
    if bad:
        raise BuildStop("; ".join("%s: %s occurs %d time(s), not once" % (r.name, "NEW" if reverse else "OLD", n) for r, n in bad))
    for r in (reversed(reps) if reverse else reps):
        old, new = (r.new, r.old) if reverse else (r.old, r.new)
        text = text.replace(old, new, 1)
    return text


def marks_check(src, out):
    old = set(src.split("\n")); bad = []
    for q, line in enumerate(out.split("\n"), 1):
        if line.strip() == "" or line in old:
            continue
        if MARK not in line:
            bad.append((q, line))
    return bad


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read(path):
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def write_new(path, text):
    if os.path.exists(path):
        raise BuildStop("the output %s exists; it is never overwritten" % path)
    with open(path, "xb") as f:
        f.write(text.encode("utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="engine_twofield_v9c5 over engine_twofield_v9c4 (exact replacements; fails closed)")
    ap.add_argument("source"); ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--against", default=None)
    a = ap.parse_args(argv)
    src = read(a.source); reps = replacements()
    try:
        if a.reverse:
            back = apply(src, reps, reverse=True)
            out = a.out or (os.path.splitext(a.source)[0] + ".restored.py")
            write_new(out, back)
            print("reverse: %s (md5 %s) -> %s (md5 %s)" % (a.source, md5(src), out, md5(back)))
            if a.against:
                ok = read(a.against) == back
                print("against %s: %s" % (a.against, "BYTE FOR BYTE" if ok else "DIFFERS"))
                return 0 if ok else 2
            return 0
        if md5(src) != SOURCE_MD5:
            print("STOP: the source md5 is %s, not v9c4's %s" % (md5(src), SOURCE_MD5)); return 2
        out_text = apply(src, reps)
    except BuildStop as stop:
        print("STOP: %s; nothing written" % stop); return 2
    marks = marks_check(src, out_text)
    if marks:
        print("STOP: %d new line(s) lack %r:" % (len(marks), MARK))
        for q, line in marks[:20]:
            print("  %6d: %s" % (q, line[:160]))
        return 2
    if apply(out_text, reps, reverse=True) != src:
        print("STOP: the reverse does not restore v9c4 byte for byte"); return 2
    try:
        compile(out_text, "engine_twofield_v9c5.py", "exec")
    except SyntaxError as exc:
        print("STOP: the built text does not compile: %s" % exc); return 2
    n_marked = sum(1 for line in out_text.split("\n") if MARK in line)
    print("build: %s (md5 %s) -> v9c5 (md5 %s); %d replacements applied; %d lines carry %r; marks ok; reverse byte for byte; compiled" % (
        a.source, md5(src), md5(out_text), len(reps), n_marked, MARK))
    if a.check:
        print("CHECK PASS (nothing written)"); return 0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), "engine_twofield_v9c5.py")
    try:
        write_new(out, out_text)
    except BuildStop as stop:
        print("STOP: %s" % stop); return 2
    print("written: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
