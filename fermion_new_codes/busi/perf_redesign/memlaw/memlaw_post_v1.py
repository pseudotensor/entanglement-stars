#!/usr/bin/env python
"""memlaw_post_v1.py (busi, inbox #384): the per-channel device transients of the record's post phase (the handoff stress, the M1
balance's two sums, the commutator, the R2 overlap) at the engine's call shapes, one channel (kappa 1) at 4096 and 8192 sites, with the
engine's own functions (engine_collapse_v36nq_rd13 and engine_stress_v2 from the memlaw code copy, unmodified). Also a lean form of
stress_parts (the same tensor operations in the same order, the 2n x 2n intermediates deleted as soon as their last use is past): its
peak and its bitwise equality with the original on the same input (every returned array compared with np.array_equal); and a column-
chunked form (the column-local work in chunks of 1024 or 4096 columns into full-width site arrays, the column reductions at full width
as the original): its peak and its bitwise equality.
    python memlaw_post_v1.py OUT CODE_DIR --sizes 4096,8192"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("code_dir"); ap.add_argument("--sizes", default="4096,8192")
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.code_dir)
import torch  # noqa: E402
import engine_collapse_v36nq_rd13 as E  # noqa: E402
import engine_stress_v2 as STR  # noqa: E402
import engine_vacflow_v1 as VF  # noqa: E402

os.makedirs(a.out, exist_ok=True)
E.C5.backend(0)
dev = E.BK["dev"]; torch.cuda.set_device(dev)
lines = []; res = {"device": torch.cuda.get_device_name(dev), "sizes": {}}


def say(s):
    lines.append(s); print(s, flush=True)


def measure(fn):
    torch.cuda.synchronize(dev); base = torch.cuda.memory_allocated(dev); torch.cuda.reset_peak_memory_stats(dev)
    t_ = time.time(); out = fn(); torch.cuda.synchronize(dev); dt = time.time() - t_
    return out, {"peak": int(torch.cuda.max_memory_allocated(dev) - base), "kept": int(torch.cuda.memory_allocated(dev) - base), "seconds": dt}


def stress_parts_lean(cs, cols, f):
    """engine_stress_v2.stress_parts with early deletes (the same operations in the same order)."""
    fw = torch.as_tensor(np.asarray(f, float), device=cs.dev, dtype=torch.float64)
    site = lambda X: X.reshape(-1, 2, X.shape[1]).sum(dim=1)
    A = cs.hbond @ cols; Qc = cs.Q @ cols; QA = cs.Q @ A
    t1 = A.conj() * Qc; del A
    t2 = QA.conj() * cols; del QA
    g = 0.5 * site(t1 + t2); del t1, t2
    H = cs.h @ cols; QH = cs.Q @ H
    t1 = H.conj() * Qc
    t2 = QH.conj() * cols; del QH
    g0 = 0.5 * site(t1 + t2); del t1, t2
    Sc = cs.hsite @ cols
    t1 = cs.hsite @ Qc; t2 = cs.Q @ Sc; del Sc
    Xc = 1j * (t1 - t2); del t1, t2
    fcn = site((cols.conj() * Xc).real); del Xc
    b = site((cols.conj() * cols).real * cs.sz[:, None])
    c0 = cols[0::2, :]; c1 = cols[1::2, :]
    sy = 2.0 * (c0.conj() * c1).imag
    eps = site((cols.conj() * H).real); del H
    Jd = site((cols.conj() * Qc).real); del Qc
    red = lambda X: (X * fw[None, :]).sum(dim=1).cpu().numpy()
    gw = (g * fw[None, :].to(g.dtype)).sum(dim=1).cpu().numpy(); g0w = (g0 * fw[None, :].to(g0.dtype)).sum(dim=1).cpu().numpy()
    return {"Gr": gw.real, "Gi": gw.imag, "G0i": g0w.imag, "fc": red(fcn), "b": red(b), "sy": red(sy), "eps": red(eps), "J": red(Jd)}


def stress_parts_chunked(cs, cols, f, width=1024):
    """stress_parts with the per-column work done in column chunks of `width` (the sparse products, the elementwise products and the
    two-component site sums: column-local) into full-width site arrays, then the column reductions over the full width as the original."""
    fw = torch.as_tensor(np.asarray(f, float), device=cs.dev, dtype=torch.float64)
    nsite, nc = cols.shape[0] // 2, cols.shape[1]
    site = lambda X: X.reshape(-1, 2, X.shape[1]).sum(dim=1)
    g = torch.empty((nsite, nc), dtype=torch.complex128, device=cs.dev); g0 = torch.empty_like(g)
    fcn = torch.empty((nsite, nc), dtype=torch.float64, device=cs.dev); b = torch.empty_like(fcn); eps = torch.empty_like(fcn); Jd = torch.empty_like(fcn)
    sy = torch.empty((nsite, nc), dtype=torch.float64, device=cs.dev)
    for j0 in range(0, nc, width):
        c = cols[:, j0:j0 + width]
        A = cs.hbond @ c; Qc = cs.Q @ c; QA = cs.Q @ A
        H = cs.h @ c; QH = cs.Q @ H
        g[:, j0:j0 + width] = 0.5 * site(A.conj() * Qc + QA.conj() * c)
        g0[:, j0:j0 + width] = 0.5 * site(H.conj() * Qc + QH.conj() * c)
        Sc = cs.hsite @ c
        Xc = 1j * (cs.hsite @ Qc - cs.Q @ Sc)
        fcn[:, j0:j0 + width] = site((c.conj() * Xc).real)
        b[:, j0:j0 + width] = site((c.conj() * c).real * cs.sz[:, None])
        c0 = c[0::2, :]; c1 = c[1::2, :]
        sy[:, j0:j0 + width] = 2.0 * (c0.conj() * c1).imag
        eps[:, j0:j0 + width] = site((c.conj() * H).real)
        Jd[:, j0:j0 + width] = site((c.conj() * Qc).real)
        del A, Qc, QA, H, QH, Sc, Xc, c0, c1, c
    red = lambda X: (X * fw[None, :]).sum(dim=1).cpu().numpy()
    gw = (g * fw[None, :].to(g.dtype)).sum(dim=1).cpu().numpy(); g0w = (g0 * fw[None, :].to(g0.dtype)).sum(dim=1).cpu().numpy()
    return {"Gr": gw.real, "Gi": gw.imag, "G0i": g0w.imag, "fc": red(fcn), "b": red(b), "sy": red(sy), "eps": red(eps), "J": red(Jd)}


for ns in [int(x) for x in a.sizes.split(",")]:
    n2 = 2 * ns; nsq = float(ns) ** 2
    ch = E.C5.Chan(1, 0.2, ns)
    R = ch.R; rs = ns / 32.0
    v = VF.profile(R, rs, "exact"); N = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
    g_ = torch.Generator(device=dev); g_.manual_seed(7)
    sea = torch.randn(n2, ns, dtype=torch.complex128, device=dev, generator=g_) / np.sqrt(n2)
    ref = torch.randn(n2, ns, dtype=torch.complex128, device=dev, generator=g_) / np.sqrt(n2)
    dust = torch.randn(n2, 64, dtype=torch.complex128, device=dev, generator=g_) / np.sqrt(n2)
    fd = np.full(64, 0.5)
    cs = STR.ChanStress(ch, torch, dev)
    rec = {}
    say("== %d sites (2n = %d): sea, R2 2n x n; dust 64 columns; n^2 = %.4e" % (ns, n2, nsq))
    fpp = np.concatenate([np.ones(ns), -np.ones(ns)])
    try:
        o1, rec["stress_pairs"] = measure(lambda: STR.stress_parts(cs, torch.cat([sea, ref], dim=1), fpp))
    except torch.OutOfMemoryError as ex:   # the original's transient above the card's free memory: recorded, not measured
        o1 = None; rec["stress_pairs_oom"] = str(ex)[:160]
    torch.cuda.empty_cache()
    o2, rec["stress_pairs_lean"] = measure(lambda: stress_parts_lean(cs, torch.cat([sea, ref], dim=1), fpp))
    rec["stress_lean_bitwise"] = bool(all(np.array_equal(o1[k_], o2[k_]) for k_ in o1)) if o1 is not None else None
    torch.cuda.empty_cache()
    for wd in (1024, 4096):
        o3, rec["stress_pairs_chunk%d" % wd] = measure(lambda: stress_parts_chunked(cs, torch.cat([sea, ref], dim=1), fpp, width=wd))
        ref_o = o1 if o1 is not None else o2
        rec["stress_chunk%d_bitwise" % wd] = bool(all(np.array_equal(ref_o[k_], o3[k_]) for k_ in ref_o))
        rec["stress_chunk%d_maxrel" % wd] = float(max(np.max(np.abs(ref_o[k_] - o3[k_])) / max(np.max(np.abs(ref_o[k_])), 1e-300) for k_ in ref_o))
        del o3
        torch.cuda.empty_cache()
    del o1, o2
    torch.cuda.empty_cache()
    _, rec["stress_dust"] = measure(lambda: STR.stress_parts(cs, dust, fd))
    cut = int(0.75 * ns)
    H_new = E.clocked_host_sparse(ch, N, v)

    def m1_inside():
        Hr = E.dev_csr(H_new[:2 * cut, :])
        cols = torch.cat([sea, dust, ref], dim=1)
        fs = np.concatenate([np.ones(ns), fd, -np.ones(ns)])
        return E.inside_energy_sp(Hr, cols, fs, 2 * cut)
    _, rec["M1_inside"] = measure(m1_inside)

    def m1_cut():
        cols = torch.cat([sea, dust, ref], dim=1)
        fs = np.concatenate([np.ones(ns), fd, -np.ones(ns)])
        return E.energy_cut_current_sp(H_new, cols, fs, cut)
    _, rec["M1_cut"] = measure(m1_cut)
    _, rec["commutator"] = measure(lambda: E.commutator_norm(ch, N, v, ref))
    torch.cuda.empty_cache()
    for k_, r_ in rec.items():
        if isinstance(r_, dict):
            say("  %-18s peak %8.3f GiB (%6.1f n^2 B)  kept %6.1f n^2  %6.2f s" % (k_, r_["peak"] / 2 ** 30, r_["peak"] / nsq, r_["kept"] / nsq, r_["seconds"]))
    say("  chunked stress_parts against the original (or the lean form where the original did not fit): %s" % "; ".join("width %d bitwise %s max rel %.1e" % (w_, rec["stress_chunk%d_bitwise" % w_], rec["stress_chunk%d_maxrel" % w_]) for w_ in (1024, 4096)))
    say("  lean stress_parts bitwise equal to the original: %s%s" % (rec["stress_lean_bitwise"], ("; the original: " + rec["stress_pairs_oom"]) if "stress_pairs_oom" in rec else ""))
    res["sizes"][str(ns)] = rec
    del sea, ref, dust, cs
    torch.cuda.empty_cache()
    with open(os.path.join(a.out, "memlaw_post_v1.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
with open(os.path.join(a.out, "memlaw_post_v1.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
say("done")
