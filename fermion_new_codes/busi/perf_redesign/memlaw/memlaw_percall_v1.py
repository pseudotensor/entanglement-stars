#!/usr/bin/env python
"""memlaw_percall_v1.py (busi, inbox #384): the per-channel device transients of the Newton engine's phases, measured one call at a time
with torch's caching-allocator counters, at 2048, 4096 and 8192 sites (2n = 4096, 8192, 16384), with the engine's own functions
(engine_collapse_v36nq_rd13 imported unmodified from the memlaw code copy; the parent's code paths are its defaults).

Per size, one channel (kappa 1) on the lad family's synthetic fields (the RD-1 percall's vacuum profile, rs = nsites/32):
  hm        clocked_gpu: the dense h_m the engine assembles (index_put)
  eig_dense eigT_t(clocked_gpu(...)) with torch.linalg.eigh (the parent's route), peak above the baseline incl. the dense h_m
  eig_band  engine_banded_eigh_v1.eigT_banded(clocked_gpu(...)) (the --eigh banded route), the same accounting
  cont      engine_ops_v3.assign_max_overlap_v3 (the continuation's one-step assignment) from a reference at perturbed fields
  jv_eig    the exact J.v's channel body as v36nq writes it (Uo, Uu, Hm, den, M, X, the two density products)
  jv_st     the --jvp sternheimer channel body (engine_banded_eigh_v1.sternheimer_X)
  prop      the record's propagation of the sea (U ph U^H X) and the densities dens(h, ref)
  prep      one channel of prepare_dust's pass (the dense diagonal well U_d, hw, two eigendecompositions, the kept columns), banded route
  xfer      pinned and pageable host <-> device copies of one 2n x 2n complex128 U (the host-release route's unit)
Writes OUT/memlaw_percall_v1.json and .txt.   python memlaw_percall_v1.py OUT CODE_DIR --sizes 2048,4096,8192 [--skip-dense-above N]"""
import argparse
import json
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("code_dir")
ap.add_argument("--sizes", default="2048,4096,8192")
ap.add_argument("--kappa", type=int, default=1)
ap.add_argument("--skip-dense-above", type=int, default=0, help="skip the dense eigh above this many sites (0: never)")
ap.add_argument("--no-prep", action="store_true")
a = ap.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, a.code_dir)
import torch  # noqa: E402
import engine_collapse_v36nq_rd13 as E  # noqa: E402
import engine_banded_eigh_v1 as EBE  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402
import engine_vacflow_v1 as VF  # noqa: E402

os.makedirs(a.out, exist_ok=True)
E.C5.backend(0)
dev = E.BK["dev"]; torch.cuda.set_device(dev)
tw = EBE.warmup(dev)
lines = []; res = {"device": torch.cuda.get_device_name(dev), "torch": torch.__version__, "warmup_s": tw, "sizes": {}}


def say(s):
    lines.append(s); print(s, flush=True)


def measure(fn):
    torch.cuda.synchronize(dev); base = torch.cuda.memory_allocated(dev); torch.cuda.reset_peak_memory_stats(dev)
    t_ = time.time(); out = fn(); torch.cuda.synchronize(dev); dt = time.time() - t_
    pk = torch.cuda.max_memory_allocated(dev) - base; kept = torch.cuda.memory_allocated(dev) - base
    return out, {"peak": int(pk), "kept": int(kept), "seconds": dt}


for ns in [int(x) for x in a.sizes.split(",")]:
    n2 = 2 * ns; nsq = float(ns) ** 2
    ch = E.C5.Chan(a.kappa, 0.2, ns)
    R = ch.R; rs = ns / 32.0
    v = VF.profile(R, rs, "exact"); N = 1.0 - 0.3 * np.exp(-(R / (2 * rs)) ** 2)
    bump = np.exp(-((R - 2 * rs) / rs) ** 2)
    rec = {}
    say("== %d sites (2n = %d), kappa %d; n^2 = %.4e; U = 2n x 2n complex128 = %.3f GiB" % (ns, n2, a.kappa, nsq, n2 * n2 * 16 / 2 ** 30))
    hm, rec["hm"] = measure(lambda: E.clocked_gpu(ch, N, v)); del hm
    if not (a.skip_dense_above and ns > a.skip_dense_above):
        (e_d, U_d), rec["eig_dense"] = measure(lambda: E.C5.eigT(E.clocked_gpu(ch, N, v)))
        del e_d, U_d
    torch.cuda.empty_cache()
    (e, U), rec["eig_band"] = measure(lambda: EBE.eigT_banded(E.clocked_gpu(ch, N, v)))
    say("  banded route stats: %s" % {k_: EBE.STATS[k_] for k_ in ("calls", "banded", "banded_all", "dense_fallback")})
    # the reference at perturbed fields (the previous record's R2)
    e_p, U_p = EBE.eigT_banded(E.clocked_gpu(ch, N - 1e-3 * bump, v - 1e-3 * bump))
    ref_prev = U_p[:, e_p < 0].contiguous(); n_sea = int(ref_prev.shape[1]); del e_p, U_p
    torch.cuda.empty_cache()
    (ref, info, mask), rec["cont"] = measure(lambda: OPS3.assign_max_overlap_v3(ref_prev, e, U, n_sea))
    say("  n_sea %d, n_occ %d" % (n_sea, int(np.sum(mask))))
    dN = 1e-3 * bump; dv = 1e-3 * bump

    def jv_eig():
        mask_t = torch.as_tensor(np.asarray(mask, bool), device=U.device)
        Uo = U[:, mask_t]; Uu = U[:, ~mask_t]
        eo = e[mask_t].to(torch.float64); eu = e[~mask_t].to(torch.float64)
        dh = E.dev_csr(E.clocked_host_sparse(ch, dN, dv))
        Hm = Uu.conj().T @ (dh @ Uo)
        den = eo[None, :] - eu[:, None]
        small = den.abs() < 1e-12
        M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
        X = Uu @ M
        out_ = []
        for A_ in (ch.h, ch.Q):
            d_ = (Uo.conj() * (A_ @ X)).real.sum(dim=1) + (X.conj() * (A_ @ Uo)).real.sum(dim=1)
            out_.append(d_.reshape(-1, 2).sum(dim=1).cpu().numpy())
        return out_
    _, rec["jv_eig"] = measure(jv_eig)
    torch.cuda.empty_cache()

    def jv_st():
        mask_t = torch.as_tensor(np.asarray(mask, bool), device=U.device)
        Uo = U[:, mask_t]
        eo = e[mask_t].to(torch.float64); eu = e[~mask_t].to(torch.float64)
        dh = E.dev_csr(E.clocked_host_sparse(ch, dN, dv))
        band_ = EBE.band_from_csr(E.clocked_host_sparse(ch, N, v))
        Y_, inf_ = EBE.sternheimer_X(band_, eo, eu, Uo, -(dh @ Uo))
        if Y_ is None:
            return None
        out_ = []
        for A_ in (ch.h, ch.Q):
            d_ = (Uo.conj() * (A_ @ Y_)).real.sum(dim=1) + (Y_.conj() * (A_ @ Uo)).real.sum(dim=1)
            out_.append(d_.reshape(-1, 2).sum(dim=1).cpu().numpy())
        return inf_.get("fallback")
    fb, rec["jv_st"] = measure(jv_st)
    rec["jv_st"]["fallback"] = fb
    torch.cuda.empty_cache()
    sea = ref_prev

    def prop():
        ph = torch.exp(-1j * e.to(torch.complex128) * 1.0)
        return U @ (ph[:, None] * (U.conj().T @ sea))
    s2, rec["prop"] = measure(prop); del s2
    _, rec["dens"] = measure(lambda: E.dens(ch.h, ref))
    torch.cuda.empty_cache()
    # the host-release unit: one U, pinned and pageable
    xf = {}
    hp = torch.empty(U.shape, dtype=U.dtype, pin_memory=True)
    for rep in range(3):
        torch.cuda.synchronize(dev); t_ = time.time(); hp.copy_(U); torch.cuda.synchronize(dev); xf.setdefault("d2h_pinned_s", []).append(time.time() - t_)
        Ud2 = torch.empty_like(U); torch.cuda.synchronize(dev); t_ = time.time(); Ud2.copy_(hp, non_blocking=True); torch.cuda.synchronize(dev); xf.setdefault("h2d_pinned_s", []).append(time.time() - t_); del Ud2
    hq = torch.empty(U.shape, dtype=U.dtype)
    torch.cuda.synchronize(dev); t_ = time.time(); hq.copy_(U); torch.cuda.synchronize(dev); xf["d2h_pageable_s"] = time.time() - t_
    Ud2 = torch.empty_like(U); t_ = time.time(); Ud2.copy_(hq); torch.cuda.synchronize(dev); xf["h2d_pageable_s"] = time.time() - t_; del Ud2
    nb = U.numel() * U.element_size(); xf["bytes"] = nb
    xf["GBps"] = {k_: nb / 1e9 / min(v_) if isinstance(v_, list) else nb / 1e9 / v_ for k_, v_ in xf.items() if k_.endswith("_s")}
    rec["xfer"] = xf
    del hp, hq
    del ref, sea, ref_prev
    if not a.no_prep:
        del U, e
        torch.cuda.empty_cache()

        def prep_ch():
            Uw = 0.1 * np.ones(ns)
            Ud = E.T_(np.diag(np.repeat(Uw, 2)))
            hw = ch.clocked_dense(N, v) + Ud
            ew, Uwv = E.eigT(hw)
            em, Um = E.eigT(ch.clocked_dense(N, v))
            pos = ew > 0
            kept_ = (ew[pos], Uwv[:, pos], Um[:, em > 0])
            del hw, ew, Uwv, em, Um
            return Ud, kept_
        globals()["eigT_saved"] = E.eigT
        E.eigT = EBE.eigT_banded
        (Ud_, kept_), rec["prep"] = measure(prep_ch)
        E.eigT = eigT_saved
        rec["prep"]["Ud_bytes"] = Ud_.numel() * Ud_.element_size()
        del Ud_, kept_
    else:
        del U, e
    torch.cuda.empty_cache()
    for k_, r_ in rec.items():
        if "peak" in r_:
            say("  %-10s peak %8.3f GiB (%6.1f n^2 B)  kept %8.3f GiB (%6.1f n^2 B)  %7.2f s%s" % (
                k_, r_["peak"] / 2 ** 30, r_["peak"] / nsq, r_["kept"] / 2 ** 30, r_["kept"] / nsq, r_["seconds"],
                ("  fallback %s" % r_.get("fallback")) if k_ == "jv_st" else ""))
    say("  xfer one U (%.3f GB): %s" % (rec["xfer"]["bytes"] / 1e9, ", ".join("%s %.1f GB/s" % (k_, v_) for k_, v_ in rec["xfer"]["GBps"].items())))
    res["sizes"][str(ns)] = rec
    with open(os.path.join(a.out, "memlaw_percall_v1.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
with open(os.path.join(a.out, "memlaw_percall_v1.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
say("done")
