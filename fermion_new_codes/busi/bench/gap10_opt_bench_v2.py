#!/usr/bin/env python
"""gap10_opt_bench_v2.py -- gate and timing of an optimized GAP10 core (gap10_core_opt2 by default; --opt_module
gap10_core_opt1 for the record) against gap10_core_v2 on fixed cases (busi/notes/GPU_opt_judgment_v2.md, section 3).
Self-contained: imports the lane's modules by path (--code_dir).

Cases, initial states, receivers and the sweep as v1 (flat T6, ball0 T1, ball02 T2, lapse T7-like; B random complex unit
vectors; 2 sites and 1 packet group; tau = rs/2 = 16 at the production Chebyshev order, dt_read 0.5).

Gate (every case at --size, every requested backend; the accuracy thresholds are v1's, far above the roundoff of about
1e-15 and far below the printed precision of the run): apply relative 1e-14; sweep state 1e-13; projections P, readouts
A = Cm P, face residuals R and certificate Ik R 1e-12 (relative to the largest entry); vector norms Nr 1e-13; |norm - 1|
within 1e-12 of v2's and below 1e-9; energies 1e-11; shares, boundary-layer mass and the region Gram matrices 1e-10; the
exact face residual of the end state 1e-12; the face list of the certificate test and a lambda projection 1e-12; the
Taylor step 1e-13 / 1e-12; the 1-element-list input released; the backend used is the one requested.
What v2 changes (the v1 criterion 'peak memory <= v2' compared peaks over different baselines and on a first sweep that
carries the compile and the autotune transient; busi/notes/GPU_opt_judgment_v2.md section 1):
  memory  : the in-sweep increment (peak minus the memory allocated when the sweep starts), measured on the backend's
            SECOND sweep (steady state) in units of S = 1 state buffer = 64 B L^3 bytes, at most v2's increment + 1;
            and the projected peak at the matrix's largest run (L 225, B 10: S 7.29 GB) = increment x 7.29 GB + the
            operator's fields scaled to L 225 + 1 GB, below 70 GB (the pod's H100 has 80; the matrix runs alone on it).
  compile : at most 2 dynamo graphs compiled in a backend's first sweep at a new box shape (the step and the norm),
            0 in any repeated or timed sweep (opt1's duplicate-input graph shows as 3 and 1 here).
Any failure prints GATE FAIL and the backend must not be adopted. The [rule] lines at the end apply the adoption rule
(gate, at least 1.5x at the timing size on every timing case, projected peak) per backend.

Timing (--timing_size; ball0 and lapse): 1 warm-up sweep per core (the compile is in it, printed separately), then --reps
timed sweeps; ms per application = sweep wall / (n + 1); min and median; the speedup against v2's median; the warm-up
excess (warm-up wall minus the compile minus a steady sweep: the second compile and the autotune) is printed.
--production adds L 225 at --B (needs about 30 GB free; never beside J4's stage E).

Usage (a few minutes of GPU at the defaults, under 8 GB; safe beside production):
  python gap10_opt_bench_v2.py --device cuda --size 65 --timing_size 129 --json opt_bench_pod_v2.json
  python gap10_opt_bench_v2.py --device cpu --size 17 --timing_size 0 --backends eager            (box smoke pass)
  GAP10_OPT2_ALLOW_CPU_COMPILE=1 python gap10_opt_bench_v2.py --device cpu --size 9 --timing_size 0 --backends compiled --cases flat,lapse
"""
import argparse
import importlib
import json
import os
import platform
import sys
import time
import numpy as np
import torch

ap = argparse.ArgumentParser(description="gate and timing of an optimized GAP10 core against gap10_core_v2")
ap.add_argument('--device', default='auto')
ap.add_argument('--code_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gap10'))
ap.add_argument('--opt_module', default='gap10_core_opt2', help="the optimized core module (gap10_core_opt2 or gap10_core_opt1)")
ap.add_argument('--size', type=int, default=65, help="box side L of the gate cases (odd)")
ap.add_argument('--timing_size', type=int, default=129, help="box side of the timing cases (0 = skip)")
ap.add_argument('--production', action='store_true', help="also time L 225 at --B (needs >= 30 GB free; not beside J4)")
ap.add_argument('--B', type=int, default=4)
ap.add_argument('--rs', type=float, default=32.0)
ap.add_argument('--tau', type=float, default=16.0)
ap.add_argument('--dt_read', type=float, default=0.5)
ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--cases', default='flat,ball0,ball02,lapse')
ap.add_argument('--timing_cases', default='ball0,lapse')
ap.add_argument('--backends', default='compiled,eager')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--L_max', type=int, default=225, help="box side of the matrix's largest run (the projection)")
ap.add_argument('--B_max', type=int, default=10, help="batch of the matrix's largest run (the projection)")
ap.add_argument('--device_gb', type=float, default=70.0, help="the projected peak must stay below this")
ap.add_argument('--json', default='gap10_opt_bench_v2.json')
args = ap.parse_args()

sys.path.insert(0, os.path.abspath(args.code_dir))
import gap10_core_v2 as C2    # noqa: E402
C1 = importlib.import_module(args.opt_module)

if args.device == 'auto':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    dev = torch.device(args.device)
CUDA = dev.type == 'cuda'
S_MAX = args.B_max * 4 * args.L_max ** 3 * 16 / 1e9
REPORT = dict(host=platform.node(), torch=torch.__version__, device=(torch.cuda.get_device_name(dev) if CUDA else 'cpu'),
              opt_module=args.opt_module, args=vars(args), gate=[], timing=[], memory=[], failures=[], S_max_gb=S_MAX)
try:
    import triton
    REPORT['triton'] = triton.__version__
except Exception:
    REPORT['triton'] = None
OPT = args.opt_module.replace('gap10_core_', '')
print(f"[bench] torch {torch.__version__} triton {REPORT['triton']} device {REPORT['device']} core v2 {C2.__file__} {OPT} {C1.__file__}", flush=True)
print(f"[bench] projection target: L {args.L_max}, B {args.B_max}: 1 state buffer S = {S_MAX:.2f} GB; limit {args.device_gb:.0f} GB", flush=True)


def sync():
    if CUDA:
        torch.cuda.synchronize(dev)


def free_gb():
    return torch.cuda.mem_get_info(dev)[0] / 1e9 if CUDA else float('nan')


def allocated_gb():
    return torch.cuda.memory_allocated(dev) / 1e9 if CUDA else float('nan')


def peak_reset():
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def peak_gb():
    return torch.cuda.max_memory_allocated(dev) / 1e9 if CUDA else float('nan')


def graphs():
    """dynamo graphs compiled so far in this process (0 if the counter is unavailable)."""
    try:
        from torch._dynamo.utils import counters
        return int(counters['stats']['unique_graphs'])
    except Exception:
        return 0


def state_gb(B, L):
    return B * 4 * L ** 3 * 16 / 1e9


def gate(name, value, tol, info=""):
    ok = bool(np.isfinite(value)) and value <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} (tol {tol:.0e}) {info}", flush=True)
    REPORT['gate'].append(dict(name=name, value=float(value), tol=tol, ok=ok))
    if not ok:
        REPORT['failures'].append(name)
    return ok


def rel_state(a, b):
    """max over the batch of ||a_b - b_b|| / ||b_b||."""
    a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    num = torch.linalg.vector_norm(a2 - b2, dim=1)
    den = torch.linalg.vector_norm(b2, dim=1).clamp_min(1e-300)
    return float((num / den).max().item())


def rel_max(a, b):
    """max |a - b| / max |b| (torch or numpy)."""
    a = np.asarray(a.cpu().numpy() if torch.is_tensor(a) else a)
    b = np.asarray(b.cpu().numpy() if torch.is_tensor(b) else b)
    den = float(np.abs(b).max()) if b.size else 0.0
    return float(np.abs(a - b).max() / max(den, 1e-300)) if a.size else 0.0


def lapse_profile(rs, Rb):
    r = np.linspace(0.0, 3.0 * rs, 601)
    ball = C2.RadialProfile.ball(rs, Rb)
    N = 1.0 - 0.3 * np.exp(-(r / Rb) ** 2)
    return C2.RadialProfile(rs, Rb, 'file', 'bench lapse table', False, (r, N, ball.u(r)))


def build_case(name, L, rs):
    t0 = time.time()
    c = L // 2
    ell, Rb = 4.0, 0.6 * rs
    m = 0.2 if name == 'ball02' else 0.0
    if name == 'flat':
        v_full, N_full, _ = C2.build_site_fields(L + 4, c + 2, 1.0, None, 'point', mode='flat')
    else:
        prof = lapse_profile(rs, Rb) if name == 'lapse' else C2.RadialProfile.ball(rs, Rb)
        v_full, N_full, _ = C2.build_site_fields(L + 4, c + 2, ell, prof, 'averaged')
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    N = None if N_full is None else N_full[2:-2, 2:-2, 2:-2].copy()
    ghost = C2.ghost_weights_np(v_full, L, N_full)
    wp, wm = C2.bond_weights_np(v)
    lp, lm = (None, None) if N is None else C2.link_weights_np(N)
    dE = 1.01 * C2.spectral_bound(m, wp, wm, N)
    a0 = allocated_gb()
    H2 = C2.LatticeH(m, wp, wm, dev, ghost_np=ghost, N_np=N, lp_np=lp, lm_np=lm)
    f2 = allocated_gb() - a0
    a0 = allocated_gb()
    H1 = C1.LatticeH(m, wp, wm, dev, ghost_np=ghost, N_np=N, lp_np=lp, lm_np=lm)
    f1 = allocated_gb() - a0
    scale = ((args.L_max + 4) / (L + 4)) ** 3
    print(f"[case {name}] L={L} m={m} N_is_one={N is None} dE={dE:.5f} built in {time.time() - t0:.1f} s; operator fields v2 {f2:.3f} GB, "
          f"{OPT} {f1:.3f} GB (about {f1 * scale:.2f} GB at L {args.L_max})", flush=True)
    return dict(name=name, L=L, c=c, m=m, dE=dE, H2=H2, H1=H1, fields2=f2, fields1=f1, fields1_max=f1 * scale)


def build_receivers(L, c, rs, m):
    rec = C2.Receivers(L, c, dev)
    rec.add_site(np.array([0, 0, 0]), 'src', 0.0)
    p2 = np.array([min(5, c - 1), min(1, c - 1), -min(2, c - 1)])
    rec.add_site(p2, 'rec', float(np.linalg.norm(p2)))
    nhat = np.array([1.0, 0.0, 0.0])
    k = 0.5
    spn = C2.helicity_spinors(k, nhat, m)
    pos = np.array([-min(3, c - 1), min(2, c - 1), min(1, c - 1)])
    idx_ref, vals, metas = None, [], []
    for hel, sp in zip((+1, -1), spn):
        idx, values, coords, prob, meta = C2.packet_mode(L, c, pos, nhat, k, sp, 2.0, 3.0, rs, restriction='raw')
        if idx_ref is None:
            idx_ref = idx
        assert np.array_equal(idx_ref, idx)
        meta.update(kind='pkt', role='bench', label=f'pkt|bench|h={hel}')
        vals.append(values)
        metas.append(meta)
    rec.add_packet_group(idx_ref, vals, metas)
    rec.finalize()
    return rec


def initial_state(B, L, seed):
    rng = np.random.default_rng(seed)
    psi = torch.empty((B, 4, L, L, L), dtype=C2.CD, device=dev)
    for b in range(B):
        z = rng.normal(size=(4, L, L, L)) + 1j * rng.normal(size=(4, L, L, L))
        z /= np.linalg.norm(z)
        psi[b] = torch.tensor(z, dtype=C2.CD, device=dev)
    return psi


def sweep(core, H, psi0, dE, c_end, n, project, residual=True, faces=None, as_list=True):
    """Returns (out, info): wall, base (allocated at the start), peak, inc = peak - base (GB), graphs compiled, released."""
    peak_reset()
    sync()
    base, g0 = allocated_gb(), graphs()
    t0 = time.time()
    arg = [psi0.clone()] if as_list else psi0.clone()
    out = core.cheb_sweep(H, arg, dE, c_end, n, project, residual=residual, faces=faces)
    sync()
    wall = time.time() - t0
    pk = peak_gb()
    return out, dict(wall=wall, base=base, peak=pk, inc=pk - base, graphs=graphs() - g0, released=(len(arg) == 0 if as_list else None))


def memory_gates(tag, case, bk, m_steady, m_first, m_v2, n_apps):
    """The 2 memory criteria from a steady sweep's increment, against v2's increment at the same size."""
    L = case['L']
    S = state_gb(args.B, L)
    row = dict(case=case['name'], L=L, B=args.B, backend=bk, tag=tag, S_gb=S, inc_v2_S=m_v2['inc'] / S, inc_first_S=(m_first['inc'] / S if m_first else float('nan')),
               inc_steady_S=m_steady['inc'] / S, fields_gb=case['fields1'], fields_max_gb=case['fields1_max'],
               projected_gb=m_steady['inc'] / S * S_MAX + case['fields1_max'] + 1.0, graphs_first=(m_first['graphs'] if m_first else 0), graphs_steady=m_steady['graphs'])
    REPORT['memory'].append(row)
    if not CUDA:
        print(f"[memory {tag}] skipped (needs CUDA)", flush=True)
        return
    first = f" (first sweep {row['inc_first_S']:.2f} S)" if m_first else ""
    print(f"[memory {tag}] in-sweep increment: v2 {row['inc_v2_S']:.2f} S, {OPT} steady {row['inc_steady_S']:.2f} S{first}; "
          f"S = {S:.3f} GB; fields {row['fields_gb']:.2f} GB (-> {row['fields_max_gb']:.2f} at L {args.L_max}); "
          f"projected peak at L {args.L_max}, B {args.B_max}: {row['projected_gb']:.1f} GB", flush=True)
    gate(f"steady in-sweep memory (state buffers) <= v2 + 1 {tag}", row['inc_steady_S'] - row['inc_v2_S'], 1.0,
         f"{OPT} {row['inc_steady_S']:.2f} S v2 {row['inc_v2_S']:.2f} S")
    gate(f"projected peak at L {args.L_max}, B {args.B_max} < {args.device_gb:.0f} GB {tag}", row['projected_gb'], args.device_gb)


def run_gate(case, rec, psi0, backends):
    H2, H1, dE, L, c = case['H2'], case['H1'], case['dE'], case['L'], case['c']
    tag = f"[{case['name']} L={L}]"
    n, tr, Cm, Ik, tail = C2.cheb_coefficients(dE, args.tau, args.dt_read)
    c_end = Cm[-1]
    y2 = H2.apply(psi0)
    y1 = H1.apply(psi0)
    gate(f"apply {tag}", rel_state(y1, y2), 1e-14)
    del y1, y2
    (psi2, P2, R2, N2), m2 = sweep(C2, H2, psi0, dE, c_end, n, rec.project)
    print(f"[v2 {tag}] n={n} sweep {m2['wall']:.2f} s ({1e3 * m2['wall'] / (n + 1):.2f} ms/application) peak {m2['peak']:.2f} GB (increment {m2['inc']:.2f} GB)", flush=True)
    geo = C2.Geometry(L, c, args.rs, args.rs, dev)
    d2 = geo.measure(psi2)
    e2, _ = H2.energy(psi2)
    G2 = geo.gram(psi2)
    res2 = H2.boundary_residual(psi2)
    A2 = np.einsum('jk,kbr->jbr', Cm, P2.cpu().numpy())
    cert2 = Ik @ R2.cpu().numpy()
    n2 = np.abs(d2[0] - 1.0)
    lam = lambda phi: phi[:, 0, c, c, c].reshape(-1, 1)  # noqa: E731
    faces2 = []
    (psi2f, P2f, R2f, _), _ = sweep(C2, H2, psi0, dE, c_end, n, lam, faces=faces2, as_list=False)
    J = C2.taylor_order(dE, 1.0)
    t2, r2, Pt2 = C2.taylor_step(H2, psi0, 1.0, J, project=rec.project)
    for bk in backends:
        try:
            H1.set_backend(bk)
        except RuntimeError as e:
            print(f"[skip] backend {bk} {tag}: {e}", flush=True)
            REPORT['gate'].append(dict(name=f"backend {bk} {tag}", skipped=str(e)))
            continue
        btag = f"[{case['name']} L={L} {bk}]"
        (psi1, P1, R1, N1), m1 = sweep(C1, H1, psi0, dE, c_end, n, rec.project)
        used = H1.backend_used
        print(f"[{OPT} {btag}] used {used} first sweep {m1['wall']:.2f} s ({1e3 * m1['wall'] / (n + 1):.2f} ms/application, compile {H1.compile_s:.1f} s, "
              f"{m1['graphs']} graphs) peak {m1['peak']:.2f} GB (increment {m1['inc']:.2f} GB)", flush=True)
        if used != bk:
            gate(f"backend used = requested {btag}", 1.0, 0.0, f"used {used}")
        gate(f"sweep state {btag}", rel_state(psi1, psi2), 1e-13)
        gate(f"projections P {btag}", rel_max(P1, P2), 1e-12)
        gate(f"readout A = Cm P {btag}", rel_max(np.einsum('jk,kbr->jbr', Cm, P1.cpu().numpy()), A2), 1e-12)
        gate(f"face residuals R {btag}", rel_max(R1, R2), 1e-12)
        gate(f"certificate Ik R {btag}", rel_max(Ik @ R1.cpu().numpy(), cert2), 1e-12)
        gate(f"vector norms Nr {btag}", float((N1 - N2).abs().max().item()), 1e-13)
        d1 = geo.measure(psi1)
        n1 = np.abs(d1[0] - 1.0)
        gate(f"|norm - 1| difference {btag}", float(np.abs(n1 - n2).max()), 1e-12, f"{OPT} {n1.max():.1e} v2 {n2.max():.1e}")
        gate(f"|norm - 1| below the run's abort level {btag}", float(n1.max()), 1e-9)
        e1, _ = H1.energy(psi1)
        gate(f"energy {btag}", float((e1 - e2).abs().max().item()), 1e-11)
        gate(f"shares {btag}", float(np.abs(d1[2] - d2[2]).max()), 1e-10)
        gate(f"boundary-layer mass {btag}", rel_max(d1[3], d2[3]), 1e-10)
        gate(f"region Gram {btag}", float(np.abs(geo.gram(psi1) - G2).max()), 1e-10)
        gate(f"exact face residual of the end state {btag}", rel_max(H2.boundary_residual(psi1), res2), 1e-12)
        gate(f"list input released {btag}", 0.0 if m1['released'] else 1.0, 0.0)
        gate(f"graphs compiled in the first sweep <= 2 {btag}", float(m1['graphs']), 2.0, "the step and the norm; a third is a recompile inside the sweep")
        del psi1, P1, R1, N1
        (psi1b, P1b, R1b, N1b), m1b = sweep(C1, H1, psi0, dE, c_end, n, rec.project)   # the steady sweep: memory and recompiles
        print(f"[{OPT} {btag}] second sweep {m1b['wall']:.2f} s ({1e3 * m1b['wall'] / (n + 1):.2f} ms/application, {m1b['graphs']} graphs) "
              f"peak {m1b['peak']:.2f} GB (increment {m1b['inc']:.2f} GB)", flush=True)
        gate(f"repeated sweep identical {btag}", rel_state(psi1b, psi2), 1e-13)
        gate(f"no recompilation in a repeated sweep {btag}", float(m1b['graphs']), 0.0)
        memory_gates(btag, case, bk, m1b, m1, m2, n + 1)
        del psi1b, P1b, R1b, N1b
        faces1 = []
        (psi1f, P1f, R1f, _), _ = sweep(C1, H1, psi0, dE, c_end, n, lam, faces=faces1, as_list=False)
        gate(f"lambda projection path: state {btag}", rel_state(psi1f, psi2f), 1e-13)
        gate(f"lambda projection path: P {btag}", rel_max(P1f, P2f), 1e-12)
        gate(f"lambda projection path: R {btag}", rel_max(R1f, R2f), 1e-12)
        fd = max(rel_max(a, b) for fa, fb in zip(faces1, faces2) for a, b in zip(fa, fb)) if faces2 and faces2[0] else 0.0
        gate(f"face vectors (certificate test path) {btag}", fd, 1e-12, f"{len(faces1)} x {len(faces1[0]) if faces1 else 0}")
        t1, r1, Pt1 = C2.taylor_step(H1, psi0, 1.0, J, project=rec.project)
        gate(f"taylor step state {btag}", rel_state(t1, t2), 1e-13)
        gate(f"taylor boundary integral {btag}", rel_max(r1, r2), 1e-12)
        gate(f"taylor projections {btag}", rel_max(Pt1, Pt2), 1e-12)
        del psi1f, P1f, R1f, faces1, t1, Pt1
    del psi2, P2, R2, N2, psi2f, faces2, t2, Pt2


def run_timing(case, rec, psi0, backends, reps):
    H2, H1, dE, L = case['H2'], case['H1'], case['dE'], case['L']
    n, tr, Cm, Ik, tail = C2.cheb_coefficients(dE, args.tau, args.dt_read)
    c_end = Cm[-1]
    rows = []
    infos = {}
    for core, H, bk in [(C2, H2, None)] + [(C1, H1, b) for b in backends]:
        label = 'v2' if bk is None else f'{OPT}-{bk}'
        if bk is not None:
            try:
                H1.set_backend(bk)
            except RuntimeError as e:
                print(f"[skip] timing {label}: {e}", flush=True)
                continue
        _, mw = sweep(core, H, psi0, dE, c_end, n, rec.project)
        walls, gsum = [], 0
        for r in range(reps):
            _, mt = sweep(core, H, psi0, dE, c_end, n, rec.project)
            walls.append(mt['wall'])
            gsum += mt['graphs']
        ms = [1e3 * w / (n + 1) for w in walls]
        med = float(np.median(walls))
        row = dict(case=case['name'], L=L, B=psi0.shape[0], core=label, used=(H1.backend_used if bk else 'v2'), n_apps=n + 1,
                   warmup_s=mw['wall'], compile_s=(H1.compile_s if bk else 0.0), warmup_excess_s=mw['wall'] - (H1.compile_s if bk else 0.0) - med,
                   graphs_warmup=mw['graphs'], graphs_timed=gsum, ms_per_app_min=min(ms), ms_per_app_median=float(np.median(ms)),
                   peak_gb=mt['peak'], inc_gb=mt['inc'], inc_S=mt['inc'] / state_gb(psi0.shape[0], L))
        rows.append(row)
        infos[label] = mt
        print(f"[timing {case['name']} L={L} B={psi0.shape[0]}] {label:14s} used {row['used']:9s} {row['ms_per_app_median']:8.2f} ms/application "
              f"(min {row['ms_per_app_min']:.2f}; warm-up sweep {mw['wall']:.1f} s, compile {row['compile_s']:.1f} s, warm-up excess {row['warmup_excess_s']:.1f} s, "
              f"graphs {mw['graphs']} warm-up / {gsum} timed) peak {mt['peak']:.2f} GB (increment {mt['inc']:.2f} GB = {row['inc_S']:.2f} S)", flush=True)
        if bk is not None:
            btag = f"[{case['name']} L={L} {bk}]"
            gate(f"no recompilation in the timed sweeps {btag}", float(gsum), 0.0)
            if 'v2' in infos:
                memory_gates(btag, case, bk, mt, mw, infos['v2'], n + 1)
    base = next((r for r in rows if r['core'] == 'v2'), None)
    for r in rows:
        r['speedup_vs_v2'] = (base['ms_per_app_median'] / r['ms_per_app_median']) if base else float('nan')
        if r['core'] != 'v2':
            print(f"[speedup {case['name']} L={L}] {r['core']}: {r['speedup_vs_v2']:.2f}x", flush=True)
    REPORT['timing'].extend(rows)


def rule_summary(backends):
    """The adoption rule per backend: gate (all criteria naming the backend), speed, projected peak."""
    verdicts = {}
    other = [f for f in REPORT['failures'] if not (f.endswith(" compiled]") or f.endswith(" eager]"))]   # e.g. apply: counts against every backend
    for bk in backends:
        fails = [f for f in REPORT['failures'] if f.endswith(f" {bk}]")] + other
        sp = [r['speedup_vs_v2'] for r in REPORT['timing'] if r['core'] == f'{OPT}-{bk}']
        pj = [r['projected_gb'] for r in REPORT['memory'] if r['backend'] == bk]
        spmin = min(sp) if sp else float('nan')
        pjmax = max(pj) if pj else float('nan')
        ok = (not fails) and bool(sp) and spmin >= 1.5 and (not CUDA or (bool(pj) and pjmax < args.device_gb))
        verdicts[bk] = dict(failures=fails, speedup_min=spmin, projected_max_gb=pjmax, adoptable=bool(ok))
        print(f"[rule] {OPT} {bk}: {len(fails)} failed criteria; speedup at the timing size min {spmin:.2f}x (1.5 required); "
              f"projected peak at L {args.L_max}, B {args.B_max} max {pjmax:.1f} GB ({args.device_gb:.0f} limit) -> {'ADOPTABLE' if ok else 'NOT adoptable'}"
              f"{' (timing or memory not measured here)' if not sp or (CUDA and not pj) else ''}", flush=True)
    REPORT['rule'] = verdicts
    order = [bk for bk in ('compiled', 'eager') if bk in verdicts and verdicts[bk]['adoptable']]
    print(f"[rule] adopt: {order[0] if order else 'nothing'} (preference compiled, then eager, then nothing)", flush=True)


def main():
    backends = [b for b in args.backends.split(',') if b]
    cases = [c for c in args.cases.split(',') if c]
    # ---- gate
    for name in cases:
        case = build_case(name, args.size, args.rs)
        rec = build_receivers(case['L'], case['c'], args.rs, case['m'])
        psi0 = initial_state(args.B, case['L'], args.seed)
        run_gate(case, rec, psi0, backends)
        del case, rec, psi0
        if CUDA:
            torch.cuda.empty_cache()
    # ---- timing
    tcases = [c for c in args.timing_cases.split(',') if c]
    sizes = []
    if args.timing_size > 0:
        sizes.append(args.timing_size)
    if args.production:
        need = 8.0 * args.B * 4 * 225 ** 3 * 16 / 1e9 + 4.0
        if CUDA and free_gb() >= max(30.0, need):
            sizes.append(225)
        else:
            print(f"[timing] production size skipped: free {free_gb():.1f} GB, need about {max(30.0, need):.0f} GB (never beside J4's stage E)", flush=True)
    for L in sizes:
        for name in tcases:
            case = build_case(name, L, args.rs)
            rec = build_receivers(case['L'], case['c'], args.rs, case['m'])
            psi0 = initial_state(args.B, L, args.seed)
            run_timing(case, rec, psi0, backends, args.reps if L < 225 else max(2, args.reps - 1))
            del case, rec, psi0
            if CUDA:
                torch.cuda.empty_cache()
    ok = not REPORT['failures']
    print(f"[gate] {'GATE PASS' if ok else 'GATE FAIL: ' + '; '.join(REPORT['failures'])}", flush=True)
    rule_summary(backends)
    REPORT['gate_pass'] = ok
    with open(args.json, 'w') as f:
        json.dump(REPORT, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"[done] {'GATE PASS' if ok else 'GATE FAIL'}; report {args.json}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
