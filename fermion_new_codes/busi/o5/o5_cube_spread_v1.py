#!/usr/bin/env python
"""o5_cube_spread_v1.py -- RUN_O5_controls_v1: the spread among propagations of the same source read without the chain (control 4: the default,
isometric and m_j -1/2 lifts of 1 column on the cube), the regression check of 2 runs of the same files (the spread must vanish to rounding), and
the lapse scaling check (control 1, S2 against S1): every --run or --chain entry may carry a clock factor as 'path:factor'; a reading of a curve
with factor f at its clock t is matched to the other curves' clock f t (S2 at N = 0.5 carries 0.5).
Curves: cube runs (gap10_run_v2_o5's json and npz: shares at the sweep ends, cert, A, labels, sweep_t, times) and chain exports (o5_chain_export_v2's
json and npz: p_mid at the sweep ends and, with --dense, at every readout time; prop_err; amp; the detector labels). For every shell (r_h, 1.5rs,
2rs, 3rs) and every matched sweep end: each curve's value and band (2 sqrt(p) delta + delta^2 with the curve's own delta: the cube's certificate,
the chain's propagation error), the spread max - min, the spread's ratio to the summed bands, the residuals to the first curve; the half-weight
crossing clocks of p beyond 1.5 r_s per curve (own reference, the comparator's rule) and at the first curve's half level (common level); the
detector peak moduli and clocks per curve and label. Writes <out>.json and <out>.md; no plots. Clocks print in sweeps of --rs a. No digit here is
the object's.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o5_chain_export_v1 import half_weight_clock  # noqa: E402

RADII = ['r_h', '1.5rs', '2rs', '3rs']


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _f(x, fmt='%.6f'):
    try:
        return fmt % float(x) if x is not None and np.isfinite(float(x)) else '-'
    except (TypeError, ValueError):
        return '-'


def weight_band(p, d):
    s = math.sqrt(max(float(p), 0.0))
    return 2.0 * s * float(d) + float(d) ** 2


def parse_entry(e):
    if os.path.exists(e):
        return e, 1.0
    if ':' in e:
        path, f = e.rsplit(':', 1)
        return path, float(f)
    return e, 1.0


def level_clock(t, p, level):
    """The first upward crossing of `level` by p(t), interpolated linearly on the given clocks; nan if none."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    for i in range(1, t.size):
        if p[i - 1] < level <= p[i]:
            return float(t[i - 1] + (t[i] - t[i - 1]) * (level - p[i - 1]) / (p[i] - p[i - 1]))
    return float('nan')


def load_cube(path, factor):
    with open(path) as fh:
        K = json.load(fh)
    Z = np.load(path[:-5] + '.npz')
    names = [str(x[0]) for x in K['share_radii']]
    sweep_t = np.asarray(Z['sweep_t'], dtype=float)
    times = np.asarray(Z['times'], dtype=float)
    S = sweep_t.size - 1
    idx = np.array([int(np.argmin(np.abs(times - t))) for t in sweep_t], dtype=int)
    shares = np.asarray(Z['shares'], dtype=float)
    cert = np.asarray(Z['cert'], dtype=float)
    A = np.asarray(Z['A'])
    labels = [str(x) for x in K['labels']]
    return dict(kind='cube', path=path, name=os.path.basename(path)[:-5], factor=float(factor), sweep_t=sweep_t, times=times,
                p={nm: shares[:S + 1, 0, names.index(nm)] for nm in RADII if nm in names}, delta=cert[idx, 0], delta_last=float(cert[idx[-1], 0]),
                amps={lab: np.abs(A[:, 0, i]) for i, lab in enumerate(labels)}, H=K.get('H'), wall=K.get('wall_total'), napply=K.get('napply'),
                lattice_scale=((K.get('src') or [{}])[0].get('band') or {}).get('lattice_scale'), p_dense=None)


def load_chain(path, factor):
    with open(path) as fh:
        J = json.load(fh)
    npz = J['files']['npz'] if os.path.exists(J['files']['npz']) else path[:-5] + '.npz'
    Z = np.load(npz)
    names = list(J['readout']['radii_names'])
    sweep_idx = np.asarray(Z['sweep_idx'], dtype=int)
    sweep_t = np.asarray(Z['sweep_t'], dtype=float)
    times = np.asarray(Z['times'], dtype=float)
    p_mid = np.asarray(Z['p_mid'], dtype=float)
    prop = np.asarray(Z['prop_err'], dtype=float)
    amp = np.asarray(Z['amp'])
    labels = [m['label'] for m in J['detector']['members']]
    return dict(kind='chain', path=path, name=os.path.basename(path)[:-5], factor=float(factor), sweep_t=sweep_t, times=times,
                p={nm: p_mid[sweep_idx, names.index(nm)] for nm in RADII if nm in names}, delta=prop[sweep_idx], delta_last=float(prop[sweep_idx[-1]]),
                amps={lab: np.abs(amp[:, i]) for i, lab in enumerate(labels)}, H=None, wall=J.get('wall'), napply=None, lattice_scale=J.get('lattice_scale_weight'),
                p_dense={nm: p_mid[:, names.index(nm)] for nm in RADII if nm in names})


def common_clocks(curves, key, tol=1e-6):
    scaled = [np.asarray(c[key], dtype=float) * c['factor'] for c in curves]
    common = scaled[0]
    for s in scaled[1:]:
        common = np.array([t for t in common if np.any(np.abs(s - t) <= tol)], dtype=float)
    return common


def at_clock(c, key, t, tol=1e-6):
    s = np.asarray(c[key], dtype=float) * c['factor']
    i = int(np.argmin(np.abs(s - t)))
    if abs(s[i] - t) > tol:
        raise ValueError(f"{c['name']}: no reading at the scaled clock {t}")
    return i


def main(argv=None):
    ap = argparse.ArgumentParser(description="RUN_O5_controls_v1: the spread among propagations of the same source (cube runs and chain exports), with clock factors")
    ap.add_argument('--run', action='append', default=[], help="a cube run's json, optionally 'path:factor' (the clock factor of a lapse-scaled run, 0.5 for N = 0.5); repeatable")
    ap.add_argument('--chain', action='append', default=[], help="a chain export's json, optionally 'path:factor'; repeatable")
    ap.add_argument('--rs', type=float, default=29.0, help="the sweep in a (clocks print in sweeps)")
    ap.add_argument('--dense', action='store_true', help="chain curves: also compare the shares at every matched readout time (the lapse scaling check at the readout resolution)")
    ap.add_argument('--out', required=True, help="writes <out>.json and <out>.md")
    args = ap.parse_args(argv)
    curves = [load_cube(*parse_entry(e)) for e in args.run] + [load_chain(*parse_entry(e)) for e in args.chain]
    if len(curves) < 2:
        raise SystemExit("give at least 2 curves (--run / --chain)")
    rs = float(args.rs)
    radii = [nm for nm in RADII if all(nm in c['p'] for c in curves)]
    clocks = common_clocks(curves, 'sweep_t')
    if clocks.size == 0:
        raise SystemExit("no common sweep end after the clock factors")
    rows, summary = {}, {}
    for nm in radii:
        rows[nm] = []
        worst = None
        for t in clocks:
            vals, bands = [], []
            for c in curves:
                i = at_clock(c, 'sweep_t', t)
                vals.append(float(c['p'][nm][i]))
                bands.append(weight_band(c['p'][nm][i], c['delta'][i]))
            spread = max(vals) - min(vals)
            sb = float(sum(bands))
            row = dict(t_scaled=float(t), t_rs=float(t / rs), values=vals, bands=bands, spread=float(spread), band_sum=sb,
                       ratio=(float(spread / sb) if sb > 0.0 else float('inf')), residuals_to_first=[float(v - vals[0]) for v in vals])
            rows[nm].append(row)
            if worst is None or spread > worst['spread']:
                worst = row
        summary[nm] = dict(max_spread=worst['spread'], at_t_rs=worst['t_rs'], max_ratio=max(r_['ratio'] for r_ in rows[nm]), n_clocks=int(clocks.size))
    clk = []
    if '1.5rs' in radii:
        ref = curves[0]
        level = 0.5 * float(ref['p']['1.5rs'][-1])
        for c in curves:
            ts = np.asarray(c['sweep_t'], dtype=float) * c['factor']
            own = half_weight_clock(ts, np.asarray(c['p']['1.5rs'], dtype=float))
            com = level_clock(ts, c['p']['1.5rs'], level)
            clk.append(dict(name=c['name'], kind=c['kind'], factor=c['factor'], own_reference_rs=float(own / rs), common_level_rs=float(com / rs),
                            terminal_share=float(c['p']['1.5rs'][-1]), common_level=level))
    dets = []
    labels = sorted(set.intersection(*[set(c['amps']) for c in curves]))
    for lab in labels:
        peaks = []
        for c in curves:
            a = np.asarray(c['amps'][lab], dtype=float)
            i = int(np.argmax(a))
            peaks.append(dict(name=c['name'], peak=float(a[i]), t_rs=float(c['times'][i] * c['factor'] / rs)))
        dets.append(dict(label=lab, peaks=peaks, peak_spread=max(p['peak'] for p in peaks) - min(p['peak'] for p in peaks),
                         clock_spread_rs=max(p['t_rs'] for p in peaks) - min(p['t_rs'] for p in peaks)))
    dense = None
    if args.dense:
        ch = [c for c in curves if c['kind'] == 'chain']
        if len(ch) >= 2:
            tc = common_clocks(ch, 'times')
            dense = {}
            for nm in radii:
                worst, worst_t = 0.0, None
                for t in tc:
                    vs = [float(c['p_dense'][nm][at_clock(c, 'times', t)]) for c in ch]
                    if max(vs) - min(vs) > worst:
                        worst, worst_t = max(vs) - min(vs), float(t / rs)
                dense[nm] = dict(n_clocks=int(tc.size), max_spread=worst, at_t_rs=worst_t)
    out = dict(script='o5_cube_spread_v1.py', rs=rs, curves=[dict(name=c['name'], kind=c['kind'], path=c['path'], factor=c['factor'], H=c['H'], wall=c['wall'],
                                                                    napply=c['napply'], delta_last=c['delta_last'], lattice_scale=c['lattice_scale']) for c in curves],
               radii=radii, clocks_scaled=clocks.tolist(), shells=rows, summary=summary, half_weight_clocks=clk, detectors=dets, dense=dense,
               note="the spread among propagations of the same source: a measurement of what the curves differ by, with each curve's own band; no digit is the object's")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    L = ["# o5_cube_spread_v1: the spread among propagations of the same source", "",
         "Curves (name; kind; clock factor; delta at the last sweep end): " + '; '.join(f"{c['name']} ({c['kind']}; {c['factor']:g}; {c['delta_last']:.2e})" for c in curves) + ".",
         f"Clocks in sweeps of {rs:g} a; a curve with factor f is read at its clock t against the others' f t. Bands: 2 sqrt(p) delta + delta^2 with each curve's own delta.", ""]
    for nm in radii:
        L += [f"## weight beyond {nm}: largest spread {summary[nm]['max_spread']:.3e} at {summary[nm]['at_t_rs']:.2f} sweeps; largest spread / band sum {summary[nm]['max_ratio']:.2f}", "",
              "| t (sweeps) | " + ' | '.join(c['name'] for c in curves) + " | spread | band sum | spread / band sum |", "|---|" + "---|" * (len(curves) + 3)]
        for r_ in rows[nm]:
            L.append(f"| {r_['t_rs']:.2f} | " + ' | '.join(f"{v:.6f} ({b:.1e})" for v, b in zip(r_['values'], r_['bands'])) + f" | {r_['spread']:.3e} | {r_['band_sum']:.2e} | {_f(r_['ratio'], '%.2f')} |")
        L.append("")
    if clk:
        L += ["## half-weight crossing of p beyond 1.5 r_s (own reference: half the curve's last sweep-end value; common level: half the first curve's)", "",
              "| curve | own reference (sweeps) | common level (sweeps) | terminal share |", "|---|---|---|---|"]
        for c in clk:
            L.append(f"| {c['name']} | {_f(c['own_reference_rs'], '%.2f')} | {_f(c['common_level_rs'], '%.2f')} | {c['terminal_share']:.4f} |")
        L.append("")
    if dets:
        L += ["## detector peaks (modulus at its clock) per curve and label", "", "| label | " + ' | '.join(c['name'] for c in curves) + " | peak spread | clock spread (sweeps) |",
              "|---|" + "---|" * (len(curves) + 2)]
        for d in dets:
            L.append(f"| {d['label']} | " + ' | '.join(f"{p['peak']:.4e} ({p['t_rs']:.2f})" for p in d['peaks']) + f" | {d['peak_spread']:.2e} | {d['clock_spread_rs']:.2f} |")
        L.append("")
    if dense is not None:
        L += ["## chain curves at every matched readout time (--dense)", ""] + [f"- beyond {nm}: {dense[nm]['n_clocks']} matched clocks, largest spread {dense[nm]['max_spread']:.3e}"
                                                                                + (f" at {dense[nm]['at_t_rs']:.2f} sweeps" if dense[nm]['at_t_rs'] is not None else "") for nm in radii] + [""]
    with open(args.out + '.md', 'w') as fh:
        fh.write("\n".join(L) + "\n")
    for nm in radii:
        print(f"[spread beyond {nm}] largest {summary[nm]['max_spread']:.3e} at {summary[nm]['at_t_rs']:.2f} sweeps (spread / band sum up to {summary[nm]['max_ratio']:.2f})", flush=True)
    if clk:
        print("[clocks] " + '; '.join(f"{c['name']}: own {_f(c['own_reference_rs'], '%.2f')}, common {_f(c['common_level_rs'], '%.2f')} sweeps" for c in clk), flush=True)
    if dense is not None:
        print("[dense] " + '; '.join(f"beyond {nm}: {dense[nm]['max_spread']:.3e} over {dense[nm]['n_clocks']} clocks" for nm in radii), flush=True)
    print(f"[done] {args.out}.json and .md", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
