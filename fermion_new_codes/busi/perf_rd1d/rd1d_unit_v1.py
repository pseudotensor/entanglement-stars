#!/usr/bin/env python
"""rd1d_unit_v1.py (busi, analyst; PERF RD-1d self-test (b)): the forced dense fallback of engine_banded_eigh_v1 (73ee46fa) against v2,
at the flat member's size (2n = 8192 for 4096 sites; --n to change), on one CUDA device (never cuda:0; CUDA_VISIBLE_DEVICES picks it).
The matrix: a Hermitian complex128 band of half-bandwidth 4 (the clocked chain's shape, seeded) plus ONE entry outside the band, so that
guard G1 sends eigT_banded to the dense route ("offband"). "Resident" tensors stand for the Newton caches: R units of 2n x 2n complex128
(a U each) and, on each, an R2-like view U[:, :n] (the view relation is checked after the restore). The device is capped by
torch.cuda.set_per_process_memory_fraction so that the dense call cannot fit beside the residents, as on F256_s4's card.
Lines (each PASS/FAIL):
  L1 no pressure: v2._dense == v1._dense bitwise (e and U) and v2.eigT_banded == v1.eigT_banded bitwise (the offband matrix and a pure band
     matrix: the banded route), STATS equal, no "memsafe" key;
  L2 the fallback's own need, measured with no pressure: peak allocated during v1's dense call above its entry (GiB), the output's size;
  L3 under the cap: v1._dense raises torch.OutOfMemoryError (the F256_s4 failure reproduced in miniature: the bytes requested, free);
  L4 under the cap: v2._dense returns, and its e and U equal L1's v1 result bitwise; v2.eigT_banded (offband) the same;
  L5 every resident restored byte for byte (sha of each storage against its value before), every view still a view of its base (same
     storage, same offset, same values), the device's allocated bytes after the call = before + the output;
  L6 the peak allocated during v2's call under the cap (at most the cap), the bytes parked, the seconds (park, restore);
  L7 the cap below residents + output (the result cannot be kept beside the residents): v2 raises torch.OutOfMemoryError and every
     resident is intact (as L5).
    CUDA_VISIBLE_DEVICES=1 python rd1d_unit_v1.py --dir <dir with engine_banded_eigh_v1.py and v2> [--n 8192] [--residents 6] --out OUT.txt
"""
import argparse
import hashlib
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--residents", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-gib", type=float, default=12.0, help="the process's ceiling for the whole test (shared card)")
    a = ap.parse_args()
    sys.path.insert(0, a.dir)
    import torch
    import engine_banded_eigh_v1 as V1
    import engine_banded_eigh_v2 as V2
    out = open(a.out, "w", buffering=1)
    res = []

    def P(s):
        print(s, flush=True); out.write(s + "\n")

    def line(name, ok, txt):
        res.append((name, bool(ok)))
        P("%s %s: %s" % (name, "PASS" if ok else "FAIL", txt))

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    GiB = 2.0 ** 30
    tot = torch.cuda.get_device_properties(dev).total_memory
    CAP0 = a.cap_gib * GiB                                               # the process's ceiling on a shared card (the coordinator's 12 GiB)
    torch.cuda.set_per_process_memory_fraction(CAP0 / tot, dev)
    P("rd1d_unit_v1: v1 %s, v2 %s, device %s, torch %s, n %d, residents %d" % (V1.VERSION, V2.VERSION, torch.cuda.get_device_name(dev), torch.__version__, a.n, a.residents))
    for M in (V1, V2):
        M.warmup(dev)
    n = a.n
    rng = np.random.default_rng(20261022)
    band = np.zeros((5, n), dtype=np.complex128)
    band[0] = rng.standard_normal(n)
    for m in range(1, 5):
        band[m, :n - m] = rng.standard_normal(n - m) + 1j * rng.standard_normal(n - m)
    Hb = np.zeros((n, n), dtype=np.complex128)
    for m in range(5):
        idx = np.arange(n - m)
        Hb[idx + m, idx] = np.conj(band[m, :n - m])
        Hb[idx, idx + m] = band[m, :n - m]
    np.fill_diagonal(Hb, band[0].real)
    Ho = Hb.copy(); Ho[n - 1, 0] = 1e-3; Ho[0, n - 1] = 1e-3          # one entry outside the band: guard G1 -> the dense route
    H_band = torch.as_tensor(Hb, device=dev); H_off = torch.as_tensor(Ho, device=dev)
    del Hb, Ho

    def snap(M):
        s = M.stats_snapshot(reset=True)
        return {k: v for k, v in s.items() if "seconds" not in k and k not in ("warmup_s", "device", "devnum_patch")}

    def eq(x, y):
        return bool(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]))

    def tohost(x):
        return (x[0].cpu(), x[1].cpu())

    # ---------------------------------------------------------------- L1, L2 (no pressure)
    snap(V1); snap(V2)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev); a0 = torch.cuda.memory_allocated(dev)
    r1 = V1._dense(H_off); torch.cuda.synchronize()
    need = torch.cuda.max_memory_allocated(dev) - a0; outb = r1[0].numel() * r1[0].element_size() + r1[1].numel() * r1[1].element_size()
    r1h = tohost(r1); del r1
    r2 = V2._dense(H_off); torch.cuda.synchronize(); r2h = tohost(r2); del r2
    e1 = tohost(V1.eigT_banded(H_off)); s1 = snap(V1)
    e2 = tohost(V2.eigT_banded(H_off)); s2 = snap(V2)
    b1 = tohost(V1.eigT_banded(H_band)); sb1 = snap(V1)
    b2 = tohost(V2.eigT_banded(H_band)); sb2 = snap(V2)
    line("L1", eq(r1h, r2h) and eq(e1, e2) and eq(r1h, e1) and eq(b1, b2) and s1 == s2 and sb1 == sb2 and "memsafe" not in s2 and "memsafe" not in sb2,
         "no pressure: _dense v2 == v1 bitwise %s; eigT_banded offband v2 == v1 %s (= _dense %s; route %s); band matrix v2 == v1 %s (route banded %d, dense %d); STATS equal %s/%s; no memsafe key %s" % (
             eq(r1h, r2h), eq(e1, e2), eq(r1h, e1), s1.get("fallback_why"), eq(b1, b2), sb2.get("banded", -1), sb2.get("dense_fallback", -1), s1 == s2, sb1 == sb2,
             "memsafe" not in s2 and "memsafe" not in sb2))
    line("L2", need > 0, "the dense fallback's own need at 2n = %d: peak %.3f GiB above its entry (the output e, U %.3f GiB; the solver's workspace %.3f GiB); the input %.3f GiB" % (
        n, need / GiB, outb / GiB, (need - outb) / GiB, H_off.numel() * 16 / GiB))
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- residents and the cap
    unit = n * n * 16
    resid = [torch.randn(n, n, dtype=torch.complex128, device=dev) for _ in range(a.residents)]
    views = [U[:, : n // 2] for U in resid]                              # R2-like views of U (continue_fast_info's U[:, :nneg])
    small = torch.randn(1024, 1024, dtype=torch.complex128, device=dev)  # 16 MiB: below the park threshold, must stay where it is
    torch.cuda.synchronize()

    def sha(t):
        return hashlib.sha1(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def sha_view(U):     # a view's bytes read through its base's host copy (a strided device-to-host copy would allocate on the device)
        return sha(U.detach().cpu()[:, : n // 2])
    sh0 = [sha(U) for U in resid]; shv0 = [sha_view(U) for U in resid]; shs0 = sha(small); ptr_small = small.data_ptr()
    base_alloc = torch.cuda.memory_allocated(dev)
    cap = base_alloc + need - unit // 2                                  # the dense call cannot fit beside the residents
    if cap > CAP0:
        raise SystemExit("the test's cap %.3f GiB exceeds the ceiling %.3f GiB: use fewer residents" % (cap / GiB, CAP0 / GiB))
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(cap / tot, dev)
    P("residents: %d x %.3f GiB (+ views, + 16 MiB small); allocated %.3f GiB; cap %.3f GiB (allocated + need - 0.5 unit): free under the cap %.3f GiB < need %.3f GiB" % (
        a.residents, unit / GiB, base_alloc / GiB, cap / GiB, (cap - base_alloc) / GiB, need / GiB))

    # ---------------------------------------------------------------- L3 v1 under the cap
    msg = None
    try:
        x = V1._dense(H_off); del x
        oom1 = False
    except torch.cuda.OutOfMemoryError as ex:
        oom1 = True; msg = str(ex).split("GPU")[0].strip()
    torch.cuda.empty_cache()
    line("L3", oom1, "v1 under the cap: torch.OutOfMemoryError %s (%s)" % (oom1, msg))
    try:
        x = V1.eigT_banded(H_off); del x
        oom1b = False
    except torch.cuda.OutOfMemoryError:
        oom1b = True
    snap(V1); torch.cuda.empty_cache()
    P("  v1.eigT_banded (offband) under the cap raises torch.OutOfMemoryError: %s (the F256_s4 traceback's last frames)" % oom1b)

    def intact():
        ok_r = all(sha(U) == h for U, h in zip(resid, sh0))
        ok_v = all(v.untyped_storage().data_ptr() == U.untyped_storage().data_ptr() and v.storage_offset() == 0 and v.data_ptr() == U.data_ptr()
                   and sha_view(U) == h for v, U, h in zip(views, resid, shv0))
        ok_s = sha(small) == shs0 and small.data_ptr() == ptr_small
        return ok_r, ok_v, ok_s

    # ---------------------------------------------------------------- L4 to L6 v2 under the cap
    logs = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev); a_in = torch.cuda.memory_allocated(dev)
    t0 = time.time()
    r3 = V2._dense(H_off, log=logs.append); torch.cuda.synchronize(); dt = time.time() - t0
    pk = torch.cuda.max_memory_allocated(dev); a_out = torch.cuda.memory_allocated(dev)
    ms = dict(V2.MEMSAFE)
    r3h = tohost(r3); del r3
    ok_r, ok_v, ok_s = intact()
    line("L4", eq(r3h, r1h), "v2._dense under the cap returns; e and U equal v1's no-pressure dense result bitwise: %s (%s)" % (eq(r3h, r1h), "; ".join(logs)))
    line("L5", ok_r and ok_v and ok_s and a_out - a_in == outb, "residents restored byte for byte %s, views still views of their bases with equal bytes %s, the 16 MiB tensor untouched (same address) %s; allocated after - before = %.6f GiB = the output %.6f GiB: %s" % (
        ok_r, ok_v, ok_s, (a_out - a_in) / GiB, outb / GiB, a_out - a_in == outb))
    line("L6", pk <= cap, "v2's peak allocated during the call %.3f GiB (cap %.3f; v1 needed %.3f = residents %.3f + the fallback's %.3f); parked %d storages %.3f GiB; park %.2f s, restore %.2f s, call %.2f s; MEMSAFE %s" % (
        pk / GiB, cap / GiB, (base_alloc + need) / GiB, base_alloc / GiB, need / GiB, ms["tensors_parked"], ms["bytes_parked"] / GiB, ms["park_s"], ms["restore_s"], dt, ms))
    logs2 = []
    V2.stats_snapshot(reset=True)
    r4 = tohost(V2.eigT_banded(H_off, log=logs2.append)); s4 = V2.stats_snapshot(reset=True)
    ok_r, ok_v, ok_s = intact()
    line("L4b", eq(r4, r1h) and ok_r and ok_v and ok_s and "memsafe" in s4,
         "v2.eigT_banded (offband) under the cap: equal to v1's no-pressure result bitwise %s, residents intact %s/%s/%s, snapshot memsafe %s (%s)" % (
             eq(r4, r1h), ok_r, ok_v, ok_s, s4.get("memsafe"), "; ".join(logs2)))
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- L7 the result cannot be kept beside the residents
    cap2 = base_alloc + outb // 2
    torch.cuda.set_per_process_memory_fraction(cap2 / tot, dev)
    logs3 = []
    ms0 = dict(V2.MEMSAFE); msg3 = None
    try:
        x = V2._dense(H_off, log=logs3.append); del x
        oom3 = False
    except torch.cuda.OutOfMemoryError as ex:
        oom3 = True; msg3 = str(ex).split("GPU")[0].strip()
        import traceback; traceback.print_exc()
    ms3 = {k: V2.MEMSAFE[k] - ms0[k] for k in ms0}
    ok_r, ok_v, ok_s = intact()
    line("L7", oom3 and ok_r and ok_v and ok_s, "cap %.3f GiB (residents + half the output): v2 raises torch.OutOfMemoryError %s (%s); residents intact %s, views %s, small %s; MEMSAFE change %s (%s)" % (
        cap2 / GiB, oom3, msg3, ok_r, ok_v, ok_s, ms3, "; ".join(logs3)))
    torch.cuda.set_per_process_memory_fraction(CAP0 / tot, dev)
    nf = sum(1 for _n, ok in res if not ok)
    P("rd1d_unit_v1: %d lines, %d FAIL -> %s" % (len(res), nf, "PASS" if nf == 0 else "FAIL"))


if __name__ == "__main__":
    main()
