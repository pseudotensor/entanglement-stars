#!/usr/bin/env python
"""engine_collapse_v36l_selftest.py (reviewer A for the busi lane; W2 dg ruling v3): the checks of version 36l.
  --build-check V36K V36L [--builder PATH]:
        (b1) the builder's forward replacements on V36K reproduce V36L byte for byte; (b2) the reverse replacements restore V36K byte for byte;
        (b3) every line of V36L that is not a line of V36K carries "v36l attr" (nothing of the default path changed); V36L compiles.
  --unit --code DIR [--gpu N]   (no engine run):
        (ix)   DgkAnderson_v36k on a linear map with the logs' multipliers (-0.65, +0.70; 58 modes within 0.1): plain needs more than 60 iterations to
               1e-12, Anderson m 4 at most 25, m 2 at most 40, m 4 with a purge at iteration 5 at most 30, each at the exact fixed point within 1e-9
               (the review's test, carried over to the v36l module);
        (x)    dgk_track_v36k on a synthetic 4-level avoided crossing (coupling 0.05, two levels crossing at theta 0.5): the identity at the anchor, the
               count held, exactly one set change (at theta 0.5), the set at theta 1 following the character against the energy order (the review's test);
        (xi)   dgk_track_v36k(..., return_w=True): the anchor weights lie in [0, 1], sum to the count, and the tracked set is their top set;
        (xii)  dgk_attr_pairs_v36l on the synthetic crossing between theta 0.3 and 0.7: coupling 0.05 gives one pair (level 1 -> 2), both gap_min estimates
               within 0.003 of the exact 0.100 (= 2 x coupling), s* within 0.02 of 0.5, the first-order jump -w gap_min within 0.05 of -1.6 for w 16;
               coupling 0 (levels crossing in energy order, a relabeling) gives estimates below 1e-6, s* 0.5, jump below 1e-5.
  --fixture --code DIR --out OUT [--threads N] [--gpu N] [extra args]:
        the v36k fixture self-test (engine_collapse_v36k_selftest.py, with the review's edit B2 applied) rerun on a scratch copy of DIR in which
        engine_collapse_v36l.py stands as engine_collapse_v36k.py: its checks (i)-(viii) ((iv) then includes E_start - E_rec at the floor) and its (ix),
        (x) run against the v36l code. The production step 12.5 -> 12.625 is the live test of the switching lines (RUN_W2_v3.md 3.4)."""
import argparse
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys

import numpy as np


def say(ok, title, detail):
    print("%s %s: %s" % ("PASS" if ok else "FAIL", title, detail), flush=True)
    return bool(ok)


def load_builder(path):
    spec = importlib.util.spec_from_file_location("build_engine_collapse_v36l", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_check(a):
    v36k, v36l = a.build_check
    B = load_builder(a.builder)
    src = open(v36k, "rb").read().decode("utf-8"); dst = open(v36l, "rb").read().decode("utf-8")
    out = {}
    try:
        built = B.build(src)
        out["b1"] = say(built == dst, "(b1) the forward replacements reproduce the built file", "%d replacements; %s" % (len(B.REPLACEMENTS), "byte for byte" if built == dst else "DIFFER (rebuild with the builder, or the built file was edited)"))
    except SystemExit as e_:
        out["b1"] = say(False, "(b1) the forward replacements reproduce the built file", str(e_))
    try:
        back = B.reverse(dst)
        out["b2"] = say(back == src, "(b2) the reverse replacements restore the source", "byte for byte" if back == src else "DIFFER")
    except SystemExit as e_:
        out["b2"] = say(False, "(b2) the reverse replacements restore the source", str(e_))
    kl = set(src.splitlines()); extra = [l_ for l_ in dst.splitlines() if l_ not in kl]
    bad = [l_ for l_ in extra if "v36l attr" not in l_]
    out["b3"] = say(not bad, "(b3) every new line is marked", "%d new lines, %d unmarked%s" % (len(extra), len(bad), (": " + bad[0][:120]) if bad else ""))
    try:
        py_compile.compile(v36l, doraise=True); out["compile"] = say(True, "compile", v36l)
    except Exception as e_:
        out["compile"] = say(False, "compile", str(e_))
    return out


def unit_tests(a):
    sys.path.insert(0, os.path.abspath(a.code))
    import engine_collapse_v5 as C5
    import engine_collapse_v36l as E
    out = {}
    rng = np.random.default_rng(3); n = 60
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))); lam = np.concatenate([[-0.65, 0.70], rng.uniform(-0.1, 0.1, n - 2)])
    A = Q @ np.diag(lam) @ Q.T; b = rng.standard_normal(n); xs = np.linalg.solve(np.eye(n) - A, b)

    def iterate(m, reset_at=None):
        x = np.zeros(n); mx = E.DgkAnderson_v36k(m, 1.0) if m else None
        for it in range(1, 201):
            gx = A @ x + b
            if float(np.max(np.abs(gx - x))) < 1e-12:
                return it, float(np.max(np.abs(x - xs)))
            if mx is None:
                x = gx
            else:
                if reset_at and it == reset_at:
                    mx.reset()
                x, _u = mx.step(x, gx)
        return 201, float(np.max(np.abs(x - xs)))
    ip, ep = iterate(0); i4, e4 = iterate(4); i2, e2 = iterate(2); ir, er = iterate(4, reset_at=5)
    out["ix"] = say(ip > 60 and i4 <= 25 and i2 <= 40 and ir <= 30 and max(e4, e2, er) < 1e-9, "(ix) Anderson type II on the logs' multipliers (-0.65, +0.70; 58 modes within 0.1)",
                    "iterations to 1e-12: plain %d, Anderson m 4 %d, m 2 %d, m 4 with a purge at iteration 5 %d; errors at the fixed point %.1e %.1e %.1e" % (ip, i4, i2, ir, e4, e2, er))
    if not C5.BK.get("torch"):
        C5.backend(a.gpu)
    torch = C5.BK["torch"]
    if hasattr(E, "BK") and E.BK is not C5.BK:
        E.BK.setdefault("torch", torch)

    def h(th, V=0.05):
        H = torch.zeros((4, 4), dtype=torch.complex128); H[0, 0] = -3.0; H[3, 3] = 3.0; H[1, 1] = -0.5 + th; H[2, 2] = 0.5 - th; H[1, 2] = V; H[2, 1] = V
        return torch.linalg.eigh(H)
    e0, U0 = h(0.0); anchor = (U0[:, :2], 2)
    _r, mask, ti = E.dgk_track_v36k(e0, U0, anchor)
    ok_id = bool(mask.tolist() == [True, True, False, False]) and ti["min_occupied_weight"] > 1.0 - 1e-12 and bool(ti["mask_equals_negative_set"])
    prev = mask; changes = 0; where = None; counts = set()
    for th in np.linspace(0.0, 1.0, 201)[1:]:
        e, U = h(float(th)); _r, mask, ti = E.dgk_track_v36k(e, U, anchor); counts.add(int(mask.sum()))
        if not np.array_equal(mask, prev):
            changes += 1; where = float(th)
        prev = mask
    ok_end = bool(mask.tolist() == [True, False, True, False]) and ti["inversions"] == 1 and not ti["mask_equals_negative_set"]
    out["x"] = say(ok_id and counts == {2} and changes == 1 and where is not None and abs(where - 0.5) < 0.02 and ok_end, "(x) the tracker on a synthetic avoided crossing (count held, maximum overlap, one switch)",
                   "identity at the anchor %s; counts %s; set changes %d (at theta %s; 0.5 expected); at theta 1 the set %s follows the character (inversions %d, equals the negative set %s)" % (
                       ok_id, sorted(counts), changes, where, mask.tolist(), ti["inversions"], ti["mask_equals_negative_set"]))
    e, U = h(0.4); ref, mask, ti, w = E.dgk_track_v36k(e, U, anchor, return_w=True)
    w = np.asarray(w, float)
    ok_w = abs(float(w.sum()) - 2.0) < 1e-10 and bool((w >= -1e-12).all()) and bool((w <= 1.0 + 1e-12).all()) and set(np.argsort(-w)[:2].tolist()) == set(np.flatnonzero(mask).tolist()) and int(ref.shape[1]) == 2
    out["xi"] = say(ok_w, "(xi) return_w: the anchor weights", "sum %.12f (2 expected), range [%.3e, %.6f], top set %s = tracked set %s" % (float(w.sum()), float(w.min()), float(w.max()), sorted(np.argsort(-w)[:2].tolist()), np.flatnonzero(mask).tolist()))
    res = {}
    for V in (0.05, 0.0):
        e0v, U0v = h(0.0, V); anc = (U0v[:, :2], 2)
        eP, UP = h(0.3, V); _r, mP, _t, wP = E.dgk_track_v36k(eP, UP, anc, return_w=True)
        eQ, UQ = h(0.7, V); _r, mQ, _t, wQ = E.dgk_track_v36k(eQ, UQ, anc, return_w=True)
        pairs, s_star, jsum, foP, foQ = E.dgk_attr_pairs_v36l({0: (mP, mQ)}, {0: (eP.cpu().numpy(), np.asarray(wP, float))}, {0: (eQ.cpu().numpy(), np.asarray(wQ, float))}, 0.3, 0.7, {0: 16.0}, 1.0)
        res[V] = (mP.tolist(), mQ.tolist(), pairs, float(s_star), float(jsum), float(foP), float(foQ))
    p05 = [p_ for p_ in res[0.05][2] if "old" in p_]; p0 = [p_ for p_ in res[0.0][2] if "old" in p_]
    ok_masks = all(res[V][0] == [True, True, False, False] and res[V][1] == [True, False, True, False] for V in (0.05, 0.0))
    ok_pair = len(p05) == 1 and p05[0]["old"] == 1 and p05[0]["new"] == 2 and all(abs(g - 0.1) < 0.003 for g in p05[0]["gap_min_estimates"]) and abs(res[0.05][3] - 0.5) < 0.02 and abs(res[0.05][4] + 1.6) < 0.05 and res[0.05][5] < -6.0 and res[0.05][6] < -6.0
    ok_zero = len(p0) == 1 and all(abs(g) < 1e-6 for g in p0[0]["gap_min_estimates"]) and abs(res[0.0][3] - 0.5) < 1e-6 and abs(res[0.0][4]) < 1e-5
    out["xii"] = say(ok_masks and ok_pair and ok_zero, "(xii) the chord-switch pairs on the synthetic crossing (theta 0.3 to 0.7)",
                     "masks %s; coupling 0.05: pairs %s, gap_min estimates %s (0.100 expected), s* %.4f (0.5), first-order jump %.4f (-1.6), -w gap at the points %.3f | %.3f; coupling 0: estimates %s, s* %.6f, jump %.1e" % (
                         ok_masks, [(p_["old"], p_["new"]) for p_ in p05], [round(g, 5) for g in p05[0]["gap_min_estimates"]] if p05 else "-", res[0.05][3], res[0.05][4], res[0.05][5], res[0.05][6],
                         [round(g, 9) for g in p0[0]["gap_min_estimates"]] if p0 else "-", res[0.0][3], res[0.0][4]))
    return out


def fixture(a, extra):
    tmp = os.path.join(a.out, "code_v36l_as_v36k")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for f in os.listdir(a.code):
        p = os.path.join(a.code, f)
        if os.path.isfile(p) and f != "engine_collapse_v36k.py":
            shutil.copy2(p, os.path.join(tmp, f))
    shutil.copy2(os.path.join(a.code, "engine_collapse_v36l.py"), os.path.join(tmp, "engine_collapse_v36k.py"))
    cmd = [sys.executable, os.path.join(tmp, "engine_collapse_v36k_selftest.py"), "--fixture", "--code", tmp, "--out", os.path.join(a.out, "fixture")]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    if a.gpu is not None:
        cmd += ["--gpu", str(a.gpu)]
    cmd += extra
    print("the v36k fixture self-test with the v36l engine in place: " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    return {"fixture": say(rc == 0, "(fixture) engine_collapse_v36k_selftest.py --fixture on the v36l engine", "exit code %d (its own PASS/FAIL lines above; (iv) includes E_start - E_rec after the review's B2)" % rc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-check", dest="build_check", nargs=2, metavar=("V36K", "V36L"), default=None)
    ap.add_argument("--builder", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_engine_collapse_v36l.py"))
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--code", default=".")
    ap.add_argument("--out", default=os.path.join(".", "selftest_v36l"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    a, extra = ap.parse_known_args()
    results = {}
    if a.build_check:
        results.update(build_check(a))
    if a.unit:
        results.update(unit_tests(a))
    if a.fixture:
        os.makedirs(a.out, exist_ok=True)
        results.update(fixture(a, extra))
    if not results:
        ap.error("nothing to do: --build-check V36K V36L, --unit --code DIR, or --fixture --code DIR --out OUT")
    bad = [k for k, v in results.items() if not v]
    print("engine_collapse_v36l_selftest: %d checks, %d failed%s" % (len(results), len(bad), (": " + ", ".join(bad)) if bad else ""), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
