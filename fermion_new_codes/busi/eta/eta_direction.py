#!/usr/bin/env python3
"""eta_direction.py -- the planar coefficient eta(p, m) of the sine-Dirac sea for a primitive normal p.

Method (specification ETA_spec_v1.md sections 2 to 5): rational-plane chain with reduced unimodular rows;
pi-zone midpoint grid, nearly isotropic in physical momentum, halved by the inversion symmetry; per
transverse momentum the half-infinite chain entropy from a slab of L = Lmin + ceil(cL |p|/Delta) layers
(2 cuts; dense Hermitian eigenvalues; Weyl block for m = 0); the node logarithm subtracted exactly with
the pi-periodic model D and its closed-form integral; 2 grid levels (h0 and h0/2) per case; the per-direction
value eta = eta_1 + (eta_1 - eta_0)/10 and half-width |eta_1 - eta_0|/4 + chain_err + ref_err.
One JSON per (p, mass, level); resumable; --combine merges the JSON files into the hand-off tables.
"""
import os
import sys
import argparse
import json
import math
import time
import glob


def _early_threads():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--threads', type=int, default=1)
    a, _ = pre.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)
    return a.threads


THREADS = _early_threads()
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_common as ec  # noqa: E402

HW_METHOD = ("levels(h0,h0/2): eta = eta1 + (eta1-eta0)/10; hw = |eta1-eta0|/4 + chain_err(1.5 cL recompute) "
             "+ ref_err/(3|p|) + 1e-9; valid if the grid error decreases at least as h^2 between the levels "
             "(expected h^4 log h after the node subtraction; tested on the exact [100] and [111] values)")


def case_name(p, mass, level):
    return f"dir_p{p[0]}_{p[1]}_{p[2]}_m{mass:g}_L{level}.json"


def run_level(p, mass, level, h0=0.1, cL=10.0, Lmin=24, device='cpu', check_near=4, check_random=4,
              seed=12345, verbose=True):
    """Compute eta(p, mass) at one grid level; returns a JSON-serializable record."""
    t_start = time.time()
    p = tuple(int(v) for v in p)
    pn = math.sqrt(sum(v * v for v in p))
    r, s, G = ec.plane_basis(p)
    y = mass * mass
    N1, N2 = ec.grid_sizes(G, h0, level)
    h_eff = (math.pi * math.sqrt(G[0, 0]) / N1, math.pi * math.sqrt(G[1, 1]) / N2)
    R, Rerr = ec.reference_log_integral(G, y)
    phis, psis, wts = ec.grid_arrays(N1, N2)
    npts = len(phis)
    Deltas = np.array([ec.chain_gap(p, r, s, phis[i], psis[i], mass) for i in range(npts)])
    Ls = Lmin + np.ceil(cL * pn / Deltas).astype(int)
    nc = 2 if mass == 0.0 else 4
    order = np.argsort(-Ls)                    # largest slabs first
    est = float(np.sum(ec.eig_seconds(nc * Ls, device)))
    if verbose:
        print(f"[{ec.now()}] p={p} m={mass:g} level={level}: r={r} s={s} G=[[{G[0,0]:.6f},{G[0,1]:.6f}],[.,{G[1,1]:.6f}]] "
              f"N1xN2={N1}x{N2} points={npts} h_eff=({h_eff[0]:.4f},{h_eff[1]:.4f}) Delta_min={Deltas.min():.5f} "
              f"L_max={int(Ls.max())} n_max={int(nc*Ls.max())} R={R:.12f} est={est/60:.1f} min", flush=True)
    S4 = np.zeros(npts)
    secs = np.zeros(npts)
    done = 0
    t_last = time.time()
    for idx in order:
        t0 = time.time()
        S4[idx] = ec.slab_entropy(p, r, s, phis[idx], psis[idx], mass, int(Ls[idx]), device=device)
        secs[idx] = time.time() - t0
        done += 1
        if verbose and (nc * Ls[idx] >= 2000 or time.time() - t_last > 60.0):
            print(f"  [{ec.now()}] {done}/{npts} point Delta={Deltas[idx]:.5f} L={int(Ls[idx])} n={int(nc*Ls[idx])} "
                  f"S4={S4[idx]:.9f} ({secs[idx]:.1f} s)", flush=True)
            t_last = time.time()
    Dm = ec.node_model(G, y, phis, psis)
    Stilde = S4 + np.log(Dm) / 3.0
    avg_sub = float(np.sum(wts * Stilde) / (N1 * N2))
    eta = (avg_sub - R / 3.0) / pn
    eta_raw = float(np.sum(wts * S4) / (N1 * N2)) / pn
    # chain-length checks: the smallest-gap points and random others recomputed with 1.5 cL
    near = [int(i) for i in np.argsort(Deltas)[:check_near]]
    rng = np.random.default_rng(seed)
    rest = [i for i in range(npts) if i not in near]
    rnd = [int(i) for i in rng.choice(rest, size=min(check_random, len(rest)), replace=False)] if rest else []
    checks = []
    for idx in near + rnd:
        L2 = Lmin + int(math.ceil(1.5 * cL * pn / Deltas[idx]))
        t0 = time.time()
        S2 = ec.slab_entropy(p, r, s, phis[idx], psis[idx], mass, L2, device=device)
        checks.append(dict(index=idx, kind='near' if idx in near else 'random', Delta=float(Deltas[idx]),
                           L=int(Ls[idx]), L_check=int(L2), S4=float(S4[idx]), S4_check=float(S2),
                           diff=float(S2 - S4[idx]), seconds=time.time() - t0))
    near_w = float(sum(wts[c['index']] for c in checks if c['kind'] == 'near'))
    near_abs = float(sum(wts[c['index']] * abs(c['diff']) for c in checks if c['kind'] == 'near'))
    rnd_max = max([abs(c['diff']) for c in checks if c['kind'] == 'random'] + [0.0])
    chain_err = (near_abs + (N1 * N2 - near_w) * rnd_max) / (N1 * N2 * pn)
    rec = dict(p=list(p), pnorm=pn, mass=mass, y=y, level=level, h0=h0, cL=cL, Lmin=Lmin, device=device,
               r=list(r), s=list(s), G=G.tolist(), N1=N1, N2=N2, h_eff=list(h_eff), n_points=npts,
               n_components=nc, R=R, ref_err=Rerr, avg_subtracted=avg_sub, eta=eta, eta_raw_unsubtracted=eta_raw,
               Delta_min=float(Deltas.min()), L_max=int(Ls.max()), n_max=int(nc * Ls.max()),
               seconds=time.time() - t_start, seconds_estimate=est, chain_checks=checks, chain_err=chain_err,
               hw_method=HW_METHOD, threads=THREADS,
               points=dict(phi=phis.tolist(), psi=psis.tolist(), weight=wts.tolist(), Delta=Deltas.tolist(),
                           L=Ls.tolist(), S4=S4.tolist(), seconds=secs.tolist()))
    if verbose:
        print(f"[{ec.now()}] p={p} m={mass:g} level={level}: eta={eta:.10f} (raw {eta_raw:.8f}) chain_err={chain_err:.2e} "
              f"max|chain diff|={max([abs(c['diff']) for c in checks] + [0.0]):.2e} ({rec['seconds']/60:.1f} min)", flush=True)
    return rec


def combine_levels(recs):
    """recs: dict level -> record for one (p, mass). Returns the reported value and half-width."""
    ls = sorted(recs)
    if len(ls) < 2:
        return None
    l0, l1 = ls[-2], ls[-1]
    e0, e1 = recs[l0]['eta'], recs[l1]['eta']
    d = e1 - e0
    rec1 = recs[l1]
    ref_err = rec1['ref_err'] / (3.0 * rec1['pnorm'])
    hw = abs(d) / 4.0 + rec1['chain_err'] + ref_err + 1e-9
    out = dict(value=e1 + d / 10.0, hw=hw, levels=ls, eta_levels=[recs[l]['eta'] for l in ls], diff=d,
               chain_err=rec1['chain_err'], ref_err=ref_err, h_lev1=max(rec1['h_eff']), N1=rec1['N1'], N2=rec1['N2'],
               cL=rec1['cL'], Lmin=rec1['Lmin'], seconds=sum(recs[l]['seconds'] for l in ls),
               eta_raw_levels=[recs[l]['eta_raw_unsubtracted'] for l in ls])
    if len(ls) >= 3:
        d_prev = recs[ls[-2]]['eta'] - recs[ls[-3]]['eta']
        out['ratio'] = d_prev / d if d != 0.0 else None
    return out


def fit_b(ys, etas, hws, eta0, hw0):
    """r(y) = [eta(y) - eta(0) - y log y/(3 pi)]/y fitted by b + c y log y + d y; linear worst-case error on b."""
    ys = np.asarray(ys, dtype=float)
    etas = np.asarray(etas, dtype=float)
    hws = np.asarray(hws, dtype=float)
    r = (etas - eta0 - ys * np.log(ys) / (3.0 * math.pi)) / ys
    hr = (hws + hw0) / ys
    A = np.column_stack([np.ones_like(ys), ys * np.log(ys), ys])
    w = 1.0 / hr
    Aw = A * w[:, None]
    coef, *_ = np.linalg.lstsq(Aw, r * w, rcond=None)
    pinv = np.linalg.pinv(Aw)
    row0 = pinv[0] * w
    b_hw = float(np.sum(np.abs(row0) * hr))
    resid = r - A @ coef
    return dict(b=float(coef[0]), b_hw=b_hw, c=float(coef[1]), d=float(coef[2]),
                resid_max=float(np.max(np.abs(resid))), r=r.tolist(), ys=ys.tolist())


def load_records(outdir):
    cases = {}
    for path in glob.glob(os.path.join(outdir, 'dir_p*_m*_L*.json')):
        with open(path) as f:
            rec = json.load(f)
        key = (tuple(rec['p']), float(rec['mass']))
        cases.setdefault(key, {})[int(rec['level'])] = rec
    return cases


def combine(outdir, table_path, btable_path=None, verbose=True):
    cases = load_records(outdir)
    rows = []
    incomplete = []
    for key in sorted(cases, key=lambda k: (k[1], sum(v * v for v in k[0]), k[0])):
        p, mass = key
        c = combine_levels(cases[key])
        if c is None:
            incomplete.append(key)
            continue
        rows.append(dict(p=p, pnorm=math.sqrt(sum(v * v for v in p)), mass=mass, **c))
    cols = ['p1', 'p2', 'p3', 'pnorm', 'mass', 'eta', 'eta_hw', 'eta_lev0', 'eta_lev1', 'levels', 'h_lev1', 'N1', 'N2',
            'cL', 'Lmin', 'chain_err', 'ref_err', 'ratio_if_3_levels', 'seconds', 'hw_method']
    with open(table_path, 'w') as f:
        f.write("# busi lane, task 2: planar coefficient eta(p, ma) of the flat sine-Dirac sea; nats; one 4-component field;\n")
        f.write("# one face; per unit area a^2; a = 1; eta_hw is the measured half-width (hw_method column).\n")
        f.write("# eta_lev0, eta_lev1: the 2 finest grid levels used; levels: all levels present.\n")
        f.write('\t'.join(cols) + '\n')
        for rw in rows:
            e_levels = rw['eta_levels']
            vals = [rw['p'][0], rw['p'][1], rw['p'][2], f"{rw['pnorm']:.12f}", f"{rw['mass']:g}", f"{rw['value']:.12f}",
                    f"{rw['hw']:.3e}", f"{e_levels[-2]:.12f}", f"{e_levels[-1]:.12f}", ','.join(str(l) for l in rw['levels']),
                    f"{rw['h_lev1']:.5f}", rw['N1'], rw['N2'], f"{rw['cL']:g}", rw['Lmin'], f"{rw['chain_err']:.2e}",
                    f"{rw['ref_err']:.2e}", f"{rw['ratio']:.3f}" if rw.get('ratio') is not None else 'na',
                    f"{rw['seconds']:.0f}", rw.get('hw_method', HW_METHOD) if False else 'see header']
            f.write('\t'.join(str(v) for v in vals) + '\n')
        f.write(f"# hw_method: {HW_METHOD}\n")
    if verbose:
        print(f"combined {len(rows)} (p, mass) cases into {table_path}; {len(incomplete)} incomplete (fewer than 2 levels)")
        m0 = [rw for rw in rows if rw['mass'] == 0.0]
        if m0:
            hws = np.array([rw['hw'] for rw in m0])
            print(f"  massless: {len(m0)} directions, half-widths max {hws.max():.2e} median {np.median(hws):.2e}")
            for rw in m0:
                key = tuple(rw['p'])
                tag = ''
                if key == (1, 0, 0):
                    tag = f" exact {ec.TARGET_ETA100:.12f} diff {rw['value'] - ec.TARGET_ETA100:+.2e}"
                if key == (1, 1, 1):
                    tag = f" exact {ec.TARGET_ETA111:.12f} diff {rw['value'] - ec.TARGET_ETA111:+.2e}"
                if key in ec.SOLVE_LANE_J7:
                    tag += f" J7 {ec.SOLVE_LANE_J7[key]:.8f} diff {rw['value'] - ec.SOLVE_LANE_J7[key]:+.2e}"
                if tag or rw.get('ratio') is not None:
                    rat = f" ratio {rw['ratio']:.2f}" if rw.get('ratio') is not None else ''
                    print(f"    {key}: eta {rw['value']:.10f} +/- {rw['hw']:.1e}{rat}{tag}")
        for key in incomplete:
            print(f"  incomplete: p={key[0]} m={key[1]:g} levels present {sorted(cases[key])}")
    # finite-mass coefficient per direction
    if btable_path:
        by_p = {}
        for rw in rows:
            by_p.setdefault(tuple(rw['p']), {})[rw['mass']] = rw
        brows = []
        for p in sorted(by_p, key=lambda q: (sum(v * v for v in q), q)):
            d = by_p[p]
            masses = sorted(m for m in d if m > 0.0)
            if 0.0 not in d or len(masses) < 3:
                continue
            ys = [m * m for m in masses]
            fit = fit_b(ys, [d[m]['value'] for m in masses], [d[m]['hw'] for m in masses], d[0.0]['value'], d[0.0]['hw'])
            brows.append(dict(p=p, pnorm=math.sqrt(sum(v * v for v in p)), masses=masses, **fit))
        with open(btable_path, 'w') as f:
            f.write("# busi lane: finite-mass coefficient b(p) of eta(p, y) = eta(p, 0) + y log y/(3 pi) + b y + ..., y = (ma)^2;\n")
            f.write("# fit of r(y) = [eta(y) - eta(0) - y log y/(3 pi)]/y by b + c y log y + d y over the listed masses;\n")
            f.write("# b_hw: linear worst-case propagation of the directional half-widths; resid_max: fit residual (model indicator).\n")
            f.write('\t'.join(['p1', 'p2', 'p3', 'pnorm', 'b', 'b_hw', 'c', 'd', 'resid_max', 'masses']) + '\n')
            for br in brows:
                f.write('\t'.join(str(v) for v in [br['p'][0], br['p'][1], br['p'][2], f"{br['pnorm']:.12f}", f"{br['b']:.8f}",
                                                   f"{br['b_hw']:.2e}", f"{br['c']:.5f}", f"{br['d']:.5f}", f"{br['resid_max']:.2e}",
                                                   ','.join(f"{m:g}" for m in br['masses'])]) + '\n')
        if verbose:
            print(f"wrote {len(brows)} finite-mass rows to {btable_path}")
            for br in brows:
                if tuple(br['p']) == (1, 0, 0):
                    print(f"    b(1,0,0) = {br['b']:.6f} +/- {br['b_hw']:.1e} (fit resid {br['resid_max']:.1e}); exact b100 = {ec.TARGET_B100:.10f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--p', nargs=3, type=int, help='primitive integer normal')
    ap.add_argument('--masses', nargs='+', type=float, default=[0.0], help='values of ma')
    ap.add_argument('--levels', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--h0', type=float, default=0.1, help='physical transverse grid spacing at level 0')
    ap.add_argument('--cL', type=float, default=10.0, help='slab length L = Lmin + ceil(cL |p|/Delta)')
    ap.add_argument('--Lmin', type=int, default=24)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--force', action='store_true', help='recompute existing cases')
    ap.add_argument('--check-near', type=int, default=4)
    ap.add_argument('--check-random', type=int, default=4)
    ap.add_argument('--combine', action='store_true', help='merge the JSON files in --outdir into tables')
    ap.add_argument('--table', default='eta_table.tsv')
    ap.add_argument('--btable', default='b_table.tsv')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.combine:
        combine(args.outdir, args.table, args.btable)
        return
    if args.p is None:
        ap.error('--p a b c is required unless --combine')
    p = tuple(args.p)
    if not ec.is_primitive(p):
        ap.error('p must be primitive')
    for mass in args.masses:
        for level in args.levels:
            path = os.path.join(args.outdir, case_name(p, mass, level))
            if os.path.exists(path) and not args.force:
                print(f"skip existing {path}")
                continue
            rec = run_level(p, mass, level, h0=args.h0, cL=args.cL, Lmin=args.Lmin, device=args.device,
                            check_near=args.check_near, check_random=args.check_random, verbose=not args.quiet)
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(rec, f)
            os.replace(tmp, path)
            print(f"wrote {path}: eta={rec['eta']:.10f} ({rec['seconds']:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
