#!/usr/bin/env python
"""engine_collapse_v36m_selftest.py (busi, reviewer A; reorientation 5): the host-offload engine's fixture on the CPU and on a GPU.
The engine's own Newton code runs (newton_record_v32 -> nk_stage_v32 -> rounds_v36c -> nk_loop_v36c, the exact J.v, confirm_v32, the
release and the restore) on synthetic channels: a gapped Dirac chain (hopping -1, staggered mass) whose clocked Hamiltonian
0.5{N, h} + 0.5{v, Q} is diagonalized for real (torch.linalg.eigh), so every cache holds real device tensors; the field solve is a stub
map with two regions of the fields (the occupied set toggles its top level where N[0] > 1), so that one record is quiet (one round) and
one is an episode: round 1 refuses a CROSS-LOWER trial and converges, the v36d rule runs a second round on the other set from the far-side
start, both meet the stop rule and the record keeps round 1 by the larger gap: round 1 was released to the host and comes back.
Checked per leg: with --nk-host-release off, on and test the record is bitwise the same (the kept fields, the sources, the history,
the solve entry, every channel's e, U, R2 with sizes, strides and the view relation of R2 to U); on a GPU the peak device memory
(torch.cuda.max_memory_allocated) falls with the release on the episode record and does not rise on the quiet one.
    python engine_collapse_v36m_selftest.py --code <internal>/code_v36d [--engine .../engine_collapse_v36m.py] [--gpu 0 | --cpu-only]
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
import types

import numpy as np
import scipy.sparse as sp


def load_engine(path, code_dir):
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("engine_collapse_v36m_under_test", path)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    return E


class Chan(object):
    """the fixture's channel: the chain's h (hopping -1, staggered mass m: a gap of 2m about zero) and the current Q, host CSR (hs, Qs)
    and device dense (h, Q), as the engine's jvp and the fixture's sources use them."""

    def __init__(self, kappa, n, m, torch, dev):
        self.kappa = kappa; self.w = 1.0
        n2 = 2 * n
        off = -np.ones(n2 - 1)
        mass = m * np.where(np.arange(n2) % 2 == 0, 1.0, -1.0)
        self.hs = sp.diags([off, mass, off], [-1, 0, 1]).astype(np.complex128).tocsr()
        self.Qs = sp.diags([1j * np.ones(n2 - 1), -1j * np.ones(n2 - 1)], [-1, 1]).astype(np.complex128).tocsr()
        self.h = torch.as_tensor(self.hs.toarray(), dtype=torch.complex128, device=dev)
        self.Q = torch.as_tensor(self.Qs.toarray(), dtype=torch.complex128, device=dev)


class Fixture(object):
    """the stub map: region A (N[0] <= t) has the target yA with the hidden factor aA, region B (N[0] > t) the target yB with aB;
    G(x) = y + a (x - y) + eps (sources of R2); the exact J.v sees the eps term only, so the Newton step is the plain F step."""

    def __init__(self, E, torch, dev, n, kappa, scenario):
        self.E, self.torch, self.dev, self.n = E, torch, dev, n
        self.chs = [Chan(k, n, 0.5 + 0.1 * k, torch, dev) for k in range(1, kappa + 1)]
        i = np.arange(n)
        N0 = 1.0 + 0.001 * np.cos(2 * np.pi * i / n); v0 = 0.001 * np.sin(2 * np.pi * i / n)
        self.t = 1.0; self.eps = 1e-12
        d0 = 0.01
        if scenario == "episode":
            yA = (N0.copy(), v0.copy()); yA[0][0] = self.t - 0.1 * d0
            yB = (N0.copy(), v0.copy()); yB[0][0] = self.t + 0.3 * d0
            x0 = (yA[0].copy(), yA[1].copy()); x0[0][0] -= d0; x0[0][3] += 0.3 * d0; x0[1][7] -= 0.3 * d0
            self.yA, self.yB, self.aA, self.aB = yA, yB, -0.25, 0.0
        else:
            yB = (N0.copy(), v0.copy()); yB[0][0] = self.t + 0.15
            x0 = (yB[0] + 0.005, yB[1] + 0.0005)
            self.yA, self.yB, self.aA, self.aB = yB, yB, 0.0, 0.0
        self.x0 = x0
        self.xinit = (N0.copy(), v0.copy())
        self.x = None

    def eigh_all(self, chs, N, v, streams=False):
        self.x = (np.asarray(N, float).copy(), np.asarray(v, float).copy())
        out = {}
        for ch in chs:
            H = self.torch.as_tensor(self.E.clocked_host_sparse(ch, N, v).toarray(), dtype=self.torch.complex128, device=self.dev)
            e, U = self.torch.linalg.eigh(H)
            out[ch.kappa] = (e, U)
            self.E.TM["eigh_n"] += 1
        return out

    def continue_ref(self, a, ch, window, f_cur, e, U, n_sea):
        e_np = e.cpu().numpy(); nneg = int((e_np < 0).sum())
        mask = e_np < 0
        if float(f_cur[0][0]) > self.t:
            mask = mask.copy(); mask[nneg - 1] = False                      # the other set: the top occupied level emptied; R2 a copy
            ref = U[:, self.torch.as_tensor(mask, device=U.device)]
        else:
            ref = U[:, :nneg]                                               # the negative set; R2 a view of U (continue_fast_info's form)
        cinfo = {"mode": "fixture", "nsub": 1, "verified": True, "n_pocket": 0, "nneg": nneg, "gap_between_blocks": float(e_np[nneg] - e_np[nneg - 1]), "_mask": mask}
        return ref, cinfo, mask

    def sources_with(self, base_, cache_):
        n = self.n; eps = np.zeros(n); J = np.zeros(n)
        for ch in self.chs:
            ref = cache_[ch.kappa][3]
            d = (ref.conj() * ref).real.sum(dim=1).reshape(n, 2).sum(dim=1).cpu().numpy()
            q = (ref.conj() * (ch.Q @ ref)).real.sum(dim=1).reshape(n, 2).sum(dim=1).cpu().numpy()
            eps += ch.w * d; J += ch.w * q
        return eps, J, eps.copy(), np.zeros(n), J.copy()

    def do_solve(self, srcs):
        eps, J = np.asarray(srcs[0], float), np.asarray(srcs[1], float)
        N_, v_ = self.x
        tgt, al = (self.yB, self.aB) if N_[0] > self.t else (self.yA, self.aA)
        N = tgt[0] + al * (N_ - tgt[0]) + self.eps * eps
        v = tgt[1] + al * (v_ - tgt[1]) + self.eps * J
        self.E.TM["solve_n"] += 1
        return N, v, None, {"E": 1.0 + 1e-6 * float(eps.sum())}

    def state(self):
        st, cache0 = {}, {}
        eig = self.eigh_all(self.chs, self.xinit[0], self.xinit[1])
        for ch in self.chs:
            e, U = eig[ch.kappa]
            nneg = int((e < 0).sum())
            ref0 = U[:, :nneg].clone()
            st[ch.kappa] = {"sea": ref0.clone(), "dust": None, "f": np.zeros(0), "ref": ref0, "n_sea": nneg,
                            "window": [{"ref": ref0, "f": (self.xinit[0].copy(), self.xinit[1].copy()), "gap": None}]}
            cache0[ch.kappa] = (None, None, None, ref0, {"mode": "lagged"})
        return st, cache0


def install(E, fx):
    E.eigh_all = fx.eigh_all
    E.continue_ref_v34 = fx.continue_ref
    E.dev_csr = lambda A: fx.torch.as_tensor(A.tocsr().toarray(), dtype=fx.torch.complex128, device=fx.dev)
    E.FLD = types.SimpleNamespace(charge=lambda info, g: info["E"], __name__="fixture")


def make_args(n, mode):
    return types.SimpleNamespace(nsites=n, g=0.25, rs=64.0, ell=4, tau=8.0, reference="R2", streams=0, mix="newton", stop_fields=1e-10, stop_E=1e-3,
                                 nk_maxiter=25, nk_inner=15, nk_krylov=10, nk_restarts=2, nk_halvings=6, nk_line_search="halving", nk_jvp="exact",
                                 nk_set="resolve", source_continuation=None, end_above=None, pair_workers=0, nk_host_release=mode)


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "seconds"}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def summarize(out, it_hist, other, torch):
    n, conv, b, cache, srcs, Nn, vn, Wn, info, N_app, v_app, e32 = out

    def by(x):
        return np.ascontiguousarray(np.asarray(x, float)).tobytes()

    def tb(t):
        return (tuple(t.shape), tuple(t.stride()), np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()) if torch.is_tensor(t) else repr(t)

    def same(x, y):
        try:
            return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001
            return x.storage().data_ptr() == y.storage().data_ptr()
    S = {"n": n, "conv": bool(conv), "kept": (float(b["resid"]), int(b["k_it"]), float(b["E"]), float(b["dE"])),
         "fields": tuple(by(x) for x in (Nn, vn, N_app, v_app)), "srcs": tuple(by(x) for x in srcs),
         "it_hist": json.dumps(it_hist, default=float), "e32": json.dumps(strip(e32), sort_keys=True, default=float),
         "cache": {k: (tb(e), tb(U), tb(ref), same(ref, U), np.asarray(ci["_mask"], bool).tobytes()) for k, (hm, e, U, ref, ci) in cache.items()},
         "other": None if other is None else tuple(by(other[k]) for k in ("N", "v", "N_applied", "v_applied", "eps", "J")) + (json.dumps(other["meta"], sort_keys=True, default=float),)}
    return S


def structure(S):
    e32 = json.loads(S["e32"]); v = e32["solves"][0]["v36c"]
    return {"rounds": len(v["rounds"]), "kept_round": v["kept_round"], "label": v["label"], "meets": e32["meets_stop_rule"], "kept_residual": e32["kept_residual"]}


def run_once(E, fx, mode, log):
    torch = fx.torch
    E.OFF.update({"on": mode != "off", "test": mode == "test", "pinned": None, "rec": None})
    E.NSR.update({"on": True, "prev_sig": None, "rec": None, "prev_m": None, "other": None})
    E.V36["fixes"] = True
    for k_ in E.JV:
        E.JV[k_] = None if k_ == "ref_mismatch" else (0.0 if k_ == "seconds" else 0)
    E.tm_reset()
    a = make_args(fx.n, mode)
    st, cache0 = fx.state()
    base = {ch.kappa: None for ch in fx.chs}
    it_hist = []
    t_ = time.time()
    with contextlib.redirect_stdout(log):
        out = E.newton_record_v32(a, fx.chs, st, cache0, fx.sources_with, base, None, None, fx.do_solve, fx.x0[0].copy(), fx.x0[1].copy(), 1.0, it_hist, 8.0)
    rec = E.offload_record_v36m() if E.OFF["on"] else None
    E.OFF.update({"on": False, "test": False})
    other = E.NSR["other"]; E.NSR["other"] = None
    S = summarize(out, it_hist, other, torch)
    del out, st, cache0
    return S, rec, time.time() - t_


def leg(E, torch, name, dev, n, kappa, verbose):
    E.BK["torch"] = torch; E.BK["dev"] = dev
    cuda = dev is not None and dev.type == "cuda"
    ok = True
    print("--- leg %s (%s) ---" % (name, torch.cuda.get_device_name(dev) if cuda else "cpu"))
    for scenario in ("quiet", "episode"):
        fx = Fixture(E, torch, dev, n, kappa, scenario)
        install(E, fx)
        S, rec, peak, secs = {}, {}, {}, {}
        for mode in ("off", "on", "test"):
            log = io.StringIO()
            gc.collect()
            if cuda:
                torch.cuda.synchronize(dev); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
                base = torch.cuda.memory_allocated(dev)
            try:
                S[mode], rec[mode], secs[mode] = run_once(E, fx, mode, log)
            except Exception:  # noqa: BLE001
                print(log.getvalue()); raise
            if cuda:
                torch.cuda.synchronize(dev); peak[mode] = torch.cuda.max_memory_allocated(dev) - base
            gc.collect()
            if verbose:
                print(log.getvalue())
        st_ = {m: structure(S[m]) for m in S}
        bit = {m: [k for k in S["off"] if S["off"][k] != S[m][k]] for m in ("on", "test")}
        exp_rounds, exp_kept = (1, 1) if scenario == "quiet" else (2, 1)
        exp_rel = {"on": 0 if scenario == "quiet" else 1, "test": 1}
        for m in ("off", "on", "test"):
            r_ = rec[m] or {}
            print("%s %-7s %-4s rounds %d kept %d%s; kept residual %.1e; released %s restored %s memo-dropped %s pinned %s; peak %s; %.1f s; bitwise vs off: %s" % (
                name, scenario, m, st_[m]["rounds"], st_[m]["kept_round"], (" (%s)" % st_[m]["label"]) if st_[m]["label"] else "", st_[m]["kept_residual"],
                r_.get("rounds_released", "-"), r_.get("rounds_restored", "-"), r_.get("memo_dropped", "-"), r_.get("pinned", "-"),
                ("%.2f MiB" % (peak[m] / 2.0 ** 20)) if cuda else "-", secs[m], "same" if m == "off" else ("yes" if not bit[m] else "NO: %s" % bit[m])))
        for m in ("on", "test"):
            if bit[m]:
                ok = False
            if st_[m]["rounds"] != exp_rounds or st_[m]["kept_round"] != exp_kept or st_["off"]["rounds"] != exp_rounds:
                ok = False; print("  FAIL: expected %d round(s) with round %d kept" % (exp_rounds, exp_kept))
            if rec[m]["rounds_released"] != exp_rel[m] or rec[m]["rounds_restored"] != exp_rel[m]:
                ok = False; print("  FAIL: expected %d release(s) and restore(s) in mode %s" % (exp_rel[m], m))
            if scenario == "episode" and rec[m]["memo_dropped"] < 1:
                ok = False; print("  FAIL: the memo was never dropped on the episode record")
        if scenario == "episode" and st_["off"]["label"] != "two self-consistent sets":
            ok = False; print("  FAIL: the episode did not produce two self-consistent sets")
        if cuda:
            if scenario == "episode":
                good = peak["on"] < peak["off"] and peak["test"] < peak["off"]
                print("  peak device memory: off %.2f MiB, on %.2f MiB (%.0f%% of off), test %.2f MiB: %s" % (
                    peak["off"] / 2.0 ** 20, peak["on"] / 2.0 ** 20, 100.0 * peak["on"] / max(peak["off"], 1), peak["test"] / 2.0 ** 20, "falls" if good else "DOES NOT FALL"))
            else:
                good = peak["on"] <= peak["off"] and peak["test"] <= peak["off"]
                print("  peak device memory: off %.2f MiB, on %.2f MiB, test %.2f MiB: %s" % (peak["off"] / 2.0 ** 20, peak["on"] / 2.0 ** 20, peak["test"] / 2.0 ** 20, "does not rise" if good else "RISES"))
            ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="the engine's import directory (engine_collapse_v5, engine_fields_v16, engine_ops_v3, ...)")
    ap.add_argument("--engine", default=None, help="engine_collapse_v36m.py (default: --code/engine_collapse_v36m.py)")
    ap.add_argument("--gpu", type=int, default=None, help="the GPU leg's card (default: 0 when CUDA is available)")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--nsites", type=int, default=96)
    ap.add_argument("--kappa", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print the engine's own lines")
    a = ap.parse_args()
    path = a.engine or os.path.join(a.code, "engine_collapse_v36m.py")
    print("engine %s md5 %s" % (path, hashlib.md5(open(path, "rb").read()).hexdigest()))
    E = load_engine(path, a.code)
    import torch
    print("torch %s; numpy %s" % (torch.__version__, np.__version__))
    ok = leg(E, torch, "cpu", None, a.nsites, a.kappa, a.verbose)
    if not a.cpu_only and torch.cuda.is_available():
        dev = torch.device("cuda", a.gpu if a.gpu is not None else 0)
        ok = leg(E, torch, "gpu", dev, a.nsites, a.kappa, a.verbose) and ok
    elif not a.cpu_only:
        print("no CUDA device: the GPU leg did not run (the peak-memory check needs it)")
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
