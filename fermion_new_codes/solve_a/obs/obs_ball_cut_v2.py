#!/usr/bin/env python3
"""obs_ball_cut_v2.py (builder analyst; S6 = J4): version 2 of obs_ball_cut_v1.py for the large radii (R 48, 62 on the H100).  The only
addition: the irrep blocks are assembled by torch on a device (--assemble-device cuda:N, or cpu:0 for tests), one block at a time, and
solved there (--solver gram: the smaller Gram matrix and eigvalsh; svd: torch.linalg.svdvals), so neither all ten blocks (about 43 GB at
R 62 + 7/16) nor the numpy gather pass (about 7 min per (offset, box) at R 62 on the box CPU) is needed on the host.  The kernel tables,
the orbit data, the mask-correlation Frobenius check, the entropy formula and the record format are obs_ball_cut_v1.py's (imported
unmodified as a library); the per-block arithmetic is the same: K_lambda[(x, k), (y, l)] = (At_x^T U(x, y) At_y)_kl,
U(x, y) = sum_g D^lambda(g) phi_g(g y) K(x, g y), K(x, z) = eta_i(x) g_i(x - z).
Modes: run (as v1, plus --assemble-device and --max-elems; the device's peak memory per (offset, box) is recorded on CUDA); compare (S of
v2 records against v1 records at the same (R, offset, L)).
Validation: run with --assemble-device cpu:0 reproduces v1's records (compare mode)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_ball_cut_v1 as B  # noqa: E402

SCRIPT = "obs_ball_cut_v2.py"


def torch_mod():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
    return torch


class DevBall:
    """the device-side orbit data shared by all ten blocks of one ball."""

    def __init__(self, bs, cub, rmax, dev, torch):
        oe, oo = bs["oe"], bs["oo"]
        W = 2 * rmax + 1
        off = int(B.lin_index(np.array([rmax] * 3), W))
        Y = oo["reps"]
        GY = np.einsum("gij,yj->ygi", cub.G, Y)
        self.ny = len(Y)
        self.PhY = torch.as_tensor(cub.phi[np.arange(48)[None, :], B.parity_index(GY)], dtype=torch.float64, device=dev)
        self.linGY = torch.as_tensor(B.lin_index(GY, W), dtype=torch.int64, device=dev)
        Xr = oe["reps"]
        self.q = (Xr[:, 0] & 1) * 2 + (Xr[:, 1] & 1)
        self.baseX = torch.as_tensor(B.lin_index(Xr, W) + off, dtype=torch.int64, device=dev)
        self.F = torch.as_tensor(cub.F, dtype=torch.float64, device=dev)


def assemble_block_torch(bs, cub, nm, db, sgt_dev, dev, torch, max_elems=5e7):
    oe, oo = bs["oe"], bs["oo"]
    d = cub.dim[nm]
    me, mo = bs["m"][nm]
    K = torch.zeros((me, mo), dtype=torch.float64, device=dev)
    if me == 0 or mo == 0:
        return K
    Fl = db.F[:, cub.cols[nm]]
    Ate = torch.as_tensor(oe["At"][nm], dtype=torch.float64, device=dev)
    Ato = torch.as_tensor(oo["At"][nm], dtype=torch.float64, device=dev)
    colmask = torch.as_tensor((np.arange(d)[None, :] < oo["mult"][nm][:, None]).ravel(), device=dev)
    rowmask_all = (np.arange(d)[None, :] < oe["mult"][nm][:, None])
    ny = db.ny
    cx = max(1, int(max_elems // (48 * ny)))
    rowoff = 0
    for qq in range(4):
        sel = np.nonzero(db.q == qq)[0]
        tab = sgt_dev[qq]
        for c0 in range(0, len(sel), cx):
            chn = sel[c0:c0 + cx]
            ch = torch.as_tensor(chn, dtype=torch.int64, device=dev)
            idx = db.baseX[ch][:, None, None] - db.linGY[None, :, :]
            v = tab[idx]
            del idx
            v.mul_(db.PhY[None, :, :])
            U = (v.reshape(-1, 48) @ Fl).reshape(len(chn), ny, d, d)
            del v
            U = torch.matmul(Ate[ch].transpose(1, 2)[:, None], U)
            U = torch.matmul(U, Ato[None])
            Kc = U.permute(0, 2, 1, 3).reshape(len(chn) * d, ny * d)
            del U
            rm = torch.as_tensor(rowmask_all[chn].ravel(), device=dev)
            Kc = Kc[rm][:, colmask]
            K[rowoff:rowoff + Kc.shape[0]] = Kc
            rowoff += Kc.shape[0]
            del Kc
    assert rowoff == me
    return K


def solve_block_torch(K, solver, torch):
    me, mo = K.shape
    if min(me, mo) == 0:
        return np.zeros(0), "empty"
    if solver == "svd":
        return torch.linalg.svdvals(K).cpu().numpy(), "torch.svdvals"
    G = K @ K.T if me <= mo else K.T @ K
    mu = torch.linalg.eigvalsh(G).cpu().numpy()
    del G
    return np.sqrt(np.clip(mu, 0.0, None)), "torch.gram_eigvalsh"


def mode_run(a):
    torch = torch_mod()
    dev = a.assemble_device
    cub = B.Cubic()
    gchk = cub.checks()
    B.log("group checks %s; assembling and solving on %s" % (json.dumps(gchk), dev))
    os.makedirs(a.out_dir, exist_ok=True)
    jl = open(os.path.join(a.out_dir, "ballcut_%s.jsonl" % a.tag), "a")
    for R in a.R:
        Rmaxeff = R + max(abs(o) for o in a.offsets16) / 16.0
        rmax = 2 * int(math.ceil(Rmaxeff)) + 2
        Ls = []
        for c in a.box_ratios:
            L = int(round(c * R))
            L += L % 2
            Ls.append((c, L))
        tabs = {}
        for c, L in Ls:
            t0 = time.time()
            tb = B.kernel_tables(L, rmax)
            sgt = B.signed_tables(tb["gtab"], tb["itab"])
            tabs[L] = dict(sgt_dev=[torch.as_tensor(s.ravel(), dtype=torch.float64, device=dev) for s in sgt], gtab=tb["gtab"],
                           forbidden=tb["forbidden"], t=time.time() - t0)
            del tb, sgt
            B.log("R %g: kernel table L %d (rmax %d) in %.1f s, forbidden-parity max %.2e" % (R, L, rmax, tabs[L]["t"], tabs[L]["forbidden"]))
        for j in a.offsets16:
            Reff = R + j / 16.0
            t0 = time.time()
            bs = B.ball_structure(Reff, cub)
            db = DevBall(bs, cub, rmax, dev, torch)
            t_orb = time.time() - t0
            for c, L in Ls:
                rows, Ssum, frob_b, t_asm, t_sol = {}, 0.0, 0.0, 0.0, 0.0
                peak = 0
                for nm in B.IRREPS:
                    if str(dev).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(dev)
                    t1 = time.time()
                    K = assemble_block_torch(bs, cub, nm, db, tabs[L]["sgt_dev"], dev, torch, a.max_elems)
                    if str(dev).startswith("cuda"):
                        torch.cuda.synchronize(dev)
                    ta = time.time() - t1
                    fb = float((K * K).sum().item())
                    t1 = time.time()
                    s, route = solve_block_torch(K, a.solver, torch)
                    ts = time.time() - t1
                    me, mo = K.shape
                    del K
                    if str(dev).startswith("cuda"):
                        peak = max(peak, int(torch.cuda.max_memory_allocated(dev)))
                        torch.cuda.empty_cache()
                    Sb = B.S_from_svals(s) + abs(me - mo) * B.LN2
                    rows[nm] = dict(d=cub.dim[nm], m_e=me, m_o=mo, S_block=Sb, smax=float(s.max()) if len(s) else 0.0,
                                    smin=float(s.min()) if len(s) else 0.0, n_above_1=int((s > 1.0).sum()), sum_s2=float((s * s).sum()),
                                    frob2=fb, t_assemble=ta, t_solve=ts, route=route)
                    Ssum += cub.dim[nm] * Sb
                    frob_b += cub.dim[nm] * fb
                    t_asm += ta
                    t_sol += ts
                S = 4.0 * Ssum
                t2 = time.time()
                fd, npairs, rnd = B.frob_direct(bs, tabs[L]["gtab"], rmax)
                t_frob = time.time() - t2
                rec = dict(script=SCRIPT, host=socket.gethostname(), utc=B.utc(), R=R, delta16=j, Reff=Reff, L=L, box_ratio=c,
                           n=bs["n"], n_e=bs["n_e"], n_o=bs["n_o"], max_x2=bs["max_x2"], mask_sha1=bs["sha1"], S=S,
                           S_over_4piR2=S / (4 * math.pi * Reff * Reff), blocks=rows,
                           frob=dict(blocks=frob_b, mask_correlation=fd, rel_diff=(frob_b - fd) / fd, fft_rounding=rnd),
                           checks=dict(orbit_sum=bs["orbit_sum_check"], mult_sum=bs["mult_sum_check"],
                                       gram_dev=[bs["oe"]["gram_dev"], bs["oo"]["gram_dev"]],
                                       orbit_dim=[bs["oe"]["orbit_dim_check"], bs["oo"]["orbit_dim_check"]],
                                       kernel_forbidden_parity=tabs[L]["forbidden"]),
                           solver=a.solver, device=str(dev), assemble_device=str(dev), threads=os.environ.get("OMP_NUM_THREADS"),
                           t_orbits=t_orb, t_assemble=t_asm, t_solve=t_sol, t_frob=t_frob, t_table=tabs[L]["t"], gpu_peak_bytes=peak)
                jl.write(json.dumps(rec) + "\n")
                jl.flush()
                big = max(rows.values(), key=lambda r: r["m_e"] * r["m_o"])
                B.log("R %g d %+d/16 L %d: n %d S %.10f S/4piR^2 %.6f frob rel %.1e | assemble %.1f s, solve %.1f s (largest %dx%d %.1f s), "
                      "device peak %.2f GB" % (R, j, L, bs["n"], S, rec["S_over_4piR2"], rec["frob"]["rel_diff"], t_asm, t_sol, big["m_e"],
                                               big["m_o"], big["t_solve"], peak / 1e9))
            del db
    jl.close()


def mode_compare(a):
    """S of v2 records against v1 records at the same (R, offset, L)."""
    ref = {}
    for f in a.ref:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                ref[(float(r["R"]), int(r["delta16"]), int(r["L"]))] = r
    worst = 0.0
    n = 0
    for f in a.new:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            p = ref.get((float(r["R"]), int(r["delta16"]), int(r["L"])))
            if p is None:
                continue
            d = r["S"] - p["S"]
            worst = max(worst, abs(d))
            n += 1
            print("R %g%+d/16 L %d: v2 (%s, %s) %.10f, v1 (%s) %.10f, difference %.2e" % (r["R"], r["delta16"], r["L"], r["assemble_device"],
                                                                                        r["solver"], r["S"], p["solver"], p["S"], d))
    print("compared %d records; largest |difference| %.2e" % (n, worst))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--R", type=float, nargs="+", required=True)
    r.add_argument("--offsets16", type=int, nargs="+", default=list(B.OFFSETS16))
    r.add_argument("--box-ratios", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    r.add_argument("--solver", choices=["svd", "gram"], default="gram")
    r.add_argument("--assemble-device", required=True)
    r.add_argument("--max-elems", type=float, default=5e7)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--tag", default="v2")
    c = sub.add_parser("compare")
    c.add_argument("--new", nargs="+", required=True)
    c.add_argument("--ref", nargs="+", required=True)
    a = ap.parse_args()
    B.log("%s %s on %s, threads %s" % (SCRIPT, a.mode, socket.gethostname(), os.environ.get("OMP_NUM_THREADS")))
    {"run": mode_run, "compare": mode_compare}[a.mode](a)


if __name__ == "__main__":
    main()
