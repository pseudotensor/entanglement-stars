#!/usr/bin/env python
"""o5_lift_v2.py -- O5 (GAP10_T7_final_v1.1 section 9): P21's lift of the chain column to the cubic lattice, reading the column's declared
radial profile (o5_chain_export_v2's column.profile: HORIZON's envelope exp(-(r - R_c)^2/(2 sigma^2)) with the carrier exp(i k_0 r) and no cut,
or v1's density-width Gaussian with the cut) instead of v1's fixed form, so that the lifted column is the exported column evaluated at the
cube's site radii. The angular functions, the Gram defect, the reduction check (the numpy copy of gap10_core_v2's operator on the record's
point-sampled fields), the detector lift, GAP10's axis packets and their channel projections are o5_lift_v1's functions, imported unchanged;
the files written have v1's names and keys (gap10_run_v2_o5.py and o5_compare_v2.py read them) plus the profile dicts. A chain json without
column.profile (o5_chain_export_v1's) is read with v1's convention, and a detector member without a profile with v1's detector convention.
The lift's support for an uncut profile ends where the envelope's amplitude falls below --support_eps of its peak (1e-10: a density below
1e-20, beyond the Gram defect by far); the declared profile is re-evaluated on the chain's own grid against the exported column and the
difference printed (the lift's first check; above 1e-10 the lift stops). Writes <out>/o5_lift_Rc<Rc>_k<kappa>_H<H>_src.npz, ..._det.npz and
..._H<H>.json. Resumable (skipped unless --force). No plots. CPU, 1 to 2 minutes per H (the reduction check holds about 10 copies of a 4 x L^3
complex array, L = 2 (R_c + support + 3) + 1: about 2.5 GB for HORIZON's sigma 8 at R_c 20).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_chain_export_v2 as CH2  # noqa: E402  (the declared profiles; no torch import)
import o5_lift_v1 as L1           # noqa: E402  (the angular functions, Gram, reduction, axis packets, channel projection)

md5 = L1.md5
_json_default = L1._json_default


def profile_of_column(J):
    col = J['column']
    if col.get('profile'):
        return dict(col['profile'])
    return CH2.profile_lane(col['Rc'], col['sigma'], col['ak0'], col['Z'], col['cut'])


def profile_of_detector(J, md):
    if md.get('profile'):
        return dict(md['profile'])
    det = J['detector']
    return CH2.profile_lane(det['r_det'], det['sigma_det'], md['k'], md['Z'], det['cut'])


def support_bounds(prof, eps):
    """(R_lo, R_hi, half-width): the cut's for a cut profile; for an uncut one the radius where the amplitude falls below eps of its peak."""
    Rc, sigma, den = float(prof['Rc']), float(prof['sigma']), float(prof['envelope_denominator'])
    cut = prof.get('cut')
    half = float(cut) * sigma if cut is not None else sigma * math.sqrt(den * math.log(1.0 / eps))
    return max(0.0, Rc - half), Rc + half, half


def lift_profile_column(prof, u, H, kappa_H, mj, lower_sign, kappa_assign, p_chain=None, support_floor=1e-6, pair=None, support_eps=1e-10):
    """The lift of one declared radial profile times the chain 2-spinor u (the source or a detector); pair = (Oa, Ob) reuses the angular
    functions on the same sites. With a v1 profile dict this equals o5_lift_v1.lift_shell_column to rounding (the self-test checks it)."""
    R_lo, R_hi, half = support_bounds(prof, support_eps)
    coords, r = L1.shell_sites(R_lo, R_hi, H)
    if coords.shape[0] == 0:
        raise ValueError("empty lift support")
    f_r = CH2.radial_profile(r, prof, apply_cut=False)          # the support selection is the cut
    Oa, Ob = L1.angular_pair(coords, r, kappa_H, mj, kappa_assign) if pair is None else pair
    vals = L1.lift_vector(r, f_r, u, Oa, Ob, lower_sign)
    W = float(np.sum(np.abs(vals) ** 2))
    gram = L1.shell_gram_dev(r, Oa, Ob)
    out = dict(coords=coords, r=r, values=vals / math.sqrt(W), values_raw=vals, W_lift=W, gram=gram, R_lo=R_lo, R_hi=R_hi, support_half_width=half,
               box_clipped=bool(R_hi > H), n_sites=int(coords.shape[0]), Oa=Oa, Ob=Ob, f_r=f_r, profile=prof)
    if p_chain is not None:
        p = np.asarray(p_chain, dtype=float)
        devj = np.ones(p.size + 1)
        k = min(gram['dev'].size, devj.size)
        devj[:k] = gram['dev'][:k]
        out['gram_defect'] = float(np.sum(p * devj[1:p.size + 1]))
        sup = p > support_floor * float(p.max())
        out['gram_dev_max_support'] = float(np.max(devj[1:p.size + 1][sup])) if np.any(sup) else float('nan')
        out['weight_defect_bound_check'] = float(abs(W - 1.0))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="O5 (v2): P21's lift of the chain column (its declared radial profile) and its detectors to the cube of half-width H, in the form gap10_run_v2_o5.py reads")
    ap.add_argument('--chain', required=True, help="the chain export's json (o5_chain_export_v2.py; v1's is read with v1's convention); its npz beside it")
    ap.add_argument('--H', type=int, required=True, help="the cube's half-width (232 for the 10 r_s window, 103 for the 3 r_s check)")
    ap.add_argument('--mj', type=float, default=0.5, help="the channel member m_j (1/2: 1 member of the 2|kappa|-fold channel)")
    ap.add_argument('--lower_sign', type=float, default=1.0, choices=[1.0, -1.0], help="+1 the derived convention; -1 flips the lower pair (the reduction check decides)")
    ap.add_argument('--kappa_assign', type=int, default=1, choices=[1, -1], help="+1 (upper Omega_{-kappa}, lower Omega_{+kappa}), the derived convention; -1 swaps")
    ap.add_argument('--record', default='', help="the record npz for the reduction check (default: the chain json's record path)")
    ap.add_argument('--no_reduction', action='store_true', help="skip the reduction check (it needs about 10 x the column's sub-box in memory)")
    ap.add_argument('--reduction_margin', type=int, default=3, help="the reduction sub-box extends this many sites beyond the column's support")
    ap.add_argument('--no_axis', action='store_true', help="do not write the channel projections of GAP10's axis packets")
    ap.add_argument('--direction', choices=['100', '111'], default='100', help="the cube run's direction (the axis packets' site)")
    ap.add_argument('--axis_sigma', type=float, default=6.0)
    ap.add_argument('--axis_cut', type=float, default=5.0)
    ap.add_argument('--support_floor', type=float, default=1e-6, help="shells with chain weight below this fraction of the maximum do not enter the maximal Gram deviation")
    ap.add_argument('--support_eps', type=float, default=1e-10, help="an uncut profile is lifted out to the radius where its amplitude falls below this fraction of the peak")
    ap.add_argument('--out', default='o5_lift')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.chain) as fh:
        J = json.load(fh)
    npz_path = J['files']['npz'] if os.path.exists(J['files']['npz']) else args.chain[:-5] + '.npz'
    Zc = np.load(npz_path)
    col, det, rec = J['column'], J['detector'], J['record']
    kappa_H = int(J['chain']['kappa'])
    m = float(J['chain']['m'])
    rs = float(J['readout']['rs'])
    H = int(args.H)
    prof = profile_of_column(J)
    stem = f"o5_lift_Rc{col['Rc']:g}_k{kappa_H}_H{H}{args.tag}"
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, stem)
    src_path, det_path, json_path = base + '_src.npz', base + '_det.npz', base + '.json'
    if (not args.force) and all(os.path.exists(p) for p in (src_path, det_path, json_path)):
        print(f"[skip] {base}_src.npz, _det.npz and .json exist (--force to redo)", flush=True)
        return 0
    tw = time.time()
    psi0 = np.asarray(Zc['psi0'])
    r_chain = np.asarray(Zc['r'], dtype=float)
    n_chain = psi0.shape[0]
    p_chain = np.sum(np.abs(psi0) ** 2, 1)
    u = np.array([complex(*col['spinor'][0]), complex(*col['spinor'][1])])
    grid_diff = float(np.max(np.abs(CH2.radial_profile(r_chain, prof)[:, None] * u[None, :] - psi0)))
    cut_txt = 'none' if prof.get('cut') is None else f"{float(prof['cut']):g} sigma"
    print(f"[chain] {args.chain} (md5 {md5(args.chain)}; npz {npz_path} md5 {md5(npz_path)}; script {J.get('script')}): kappa={kappa_H}, R_c={col['Rc']:g}, a k_0={col['ak0']:+.6f}, "
          f"<h>={col['E_mean']:.6f}, rms {col['E_rms']:.5f}, n={n_chain}; profile: envelope exp(-(r - R_c)^2/({float(prof['envelope_denominator']):g} sigma^2)) with sigma {float(prof['sigma']):g} a "
          f"({prof['sigma_meaning']}; density std {CH2.density_std(prof):.4f} a), carrier exp(i k_0 (r - {float(prof['carrier_reference']):g})), cut {cut_txt}, source {prof.get('source')}; "
          f"the declared profile on the chain's own grid reproduces the exported column to {grid_diff:.1e}; lift to H={H} with m_j={args.mj:g}, lower_sign={args.lower_sign:+g}, kappa_assign={args.kappa_assign:+d}", flush=True)
    if grid_diff > 1e-10:
        raise SystemExit(f"the chain json's declared profile does not reproduce its exported column (max difference {grid_diff:.2e}): the lift would not be this column; stop")

    # ---- the source column, 4 conventions (the chosen one is written)
    t0 = time.time()
    R_lo, R_hi, half = support_bounds(prof, args.support_eps)
    coords, r = L1.shell_sites(R_lo, R_hi, H)
    Oa, Ob = L1.angular_pair(coords, r, kappa_H, args.mj, +1)
    lifts = {}
    for ls_ in (1.0, -1.0):
        for ka in (1, -1):
            pair = (Oa, Ob) if ka > 0 else (Ob, Oa)
            lifts[f"lower{ls_:+g}_kappa{ka:+d}"] = lift_profile_column(prof, u, H, kappa_H, args.mj, ls_, ka, p_chain=p_chain, support_floor=args.support_floor, pair=pair, support_eps=args.support_eps)
    key = f"lower{args.lower_sign:+g}_kappa{args.kappa_assign:+d}"
    S = lifts[key]
    rad_names = J['readout']['radii_names']
    radii = J['readout']['radii']
    t0_check = []
    for nm, R in zip(rad_names, radii):
        t0_check.append(dict(name=nm, R=float(R), lift_strict=L1.weight_beyond(S['r'], S['values'], R), chain_mid=float(J['sweeps'][0]['p_mid'][nm]),
                             chain_strict=float(J['sweeps'][0]['p_strict'][nm])))
    print(f"[lift source] {S['n_sites']} sites on {R_lo:.1f} <= r <= {R_hi:.1f} a (support half-width {half:.1f} a; box clipped: {S['box_clipped']}); weight before renormalization {S['W_lift']:.6f}; "
          f"Gram defect delta_G = {S['gram_defect']:.3e} (column-weighted ||G_j - I||; the lift's own error for shell weights), maximal shell deviation on the support "
          f"{S['gram_dev_max_support']:.3e} (the lift error for amplitudes); ({time.time() - t0:.1f}s)", flush=True)
    for c_ in t0_check:
        print(f"[lift t0] beyond {c_['name']} ({c_['R']:.4f} a): lifted {c_['lift_strict']:.6f} against the chain's cell rule {c_['chain_mid']:.6f} (strict {c_['chain_strict']:.6f}); "
              f"difference to the cell rule {c_['lift_strict'] - c_['chain_mid']:+.2e} (delta_G {S['gram_defect']:.1e})", flush=True)
    sg = S['gram']
    jshow = [j for j in (1, 2, 3, 5, 10, 14, 20, 30, 44, 58, 87) if j < sg['dev'].size]
    print("[lift gram] shell j: sites, ||G_j - I||: " + ', '.join(f"{j}: {int(sg['count'][j])}, {sg['dev'][j]:.2e}" for j in jshow), flush=True)

    # ---- the reduction check
    red = None
    if not args.no_reduction:
        rec_path = args.record or rec['path']
        if os.path.exists(rec_path):
            t0 = time.time()
            rt, Nt, ut = L1.load_table(rec_path)
            Hc = min(int(math.ceil(R_hi)) + int(args.reduction_margin), H)
            rc = L1.reduction_check(lifts, rt, Nt, ut, m, Hc)
            diffs = {k_: abs(v_['E_mean'] - col['E_mean']) for k_, v_ in rc.items()}
            closest = min(diffs, key=diffs.get)
            red = dict(record=rec_path, record_md5=md5(rec_path), Hc=Hc, L=2 * Hc + 1, chain_E_mean=col['E_mean'], chain_E_rms=col['E_rms'],
                       conventions={k_: dict(v_, abs_diff=diffs[k_], rel_diff=diffs[k_] / max(abs(col['E_mean']), 1e-300)) for k_, v_ in rc.items()},
                       default_key=key, closest_key=closest, default_is_closest=(closest == key), wall=time.time() - t0,
                       note="the numpy copy of gap10_core_v2's operator on the record's point-sampled fields; a convention error shows as an O(1) relative <h> difference, the lattice difference between the 2 descriptions as a small one")
            print(f"[reduction] sub-box half-width {Hc} (L {2 * Hc + 1}), {red['wall']:.1f}s: chain <h> {col['E_mean']:.6f} rms {col['E_rms']:.5f}; cube <Psi, h Psi> per convention: "
                  + '; '.join(f"{k_}: {v_['E_mean']:.6f} (rms {v_['E_rms']:.5f}, |diff| {diffs[k_]:.2e})" for k_, v_ in rc.items())
                  + f"; closest {closest}; the chosen convention {key} is the closest: {closest == key}" + ("" if closest == key else " -- WARNING: rerun the lift with the closest convention's --lower_sign/--kappa_assign"), flush=True)
        else:
            print(f"[reduction] record {rec_path} not found here: skipped (give --record)", flush=True)

    # ---- the detectors (the chain's 2 members, both carrier signs), lifted the same way
    det_lifts = []
    coords_d = None
    for md in det['members']:
        ud = np.array([complex(*md['spinor'][0]), complex(*md['spinor'][1])])
        pd_ = np.sum(np.abs(np.asarray(Zc['det_vecs'])[len(det_lifts)]) ** 2, 1)
        prof_d = profile_of_detector(J, md)
        D = lift_profile_column(prof_d, ud, H, kappa_H, args.mj, args.lower_sign, args.kappa_assign, p_chain=pd_, support_floor=args.support_floor,
                                pair=None if coords_d is None else det_lifts[0]['pair'], support_eps=args.support_eps)
        if coords_d is None:
            coords_d = D['coords']
            D['pair'] = (D['Oa'], D['Ob'])
        else:
            assert np.array_equal(coords_d, D['coords'])
        D['label'] = md['label']
        D['meta'] = dict(label=md['label'], sign=int(md['sign']), k=float(md['k']), r_det=float(det['r_det']), sigma_det=float(det['sigma_det']), cut=float(det['cut']),
                         omega_target=float(det['omega_target']), matched=bool(det['matched']), W_lift=D['W_lift'], gram_defect=D['gram_defect'],
                         gram_dev_max_support=D['gram_dev_max_support'], n_sites=D['n_sites'], profile=prof_d)
        det_lifts.append(D)
        print(f"[lift detector] {md['label']}: {D['n_sites']} sites, weight before renormalization {D['W_lift']:.6f}, Gram defect {D['gram_defect']:.3e}, maximal shell deviation {D['gram_dev_max_support']:.3e}", flush=True)
    amp_lift_error = float(max([S['gram_dev_max_support']] + [D['gram_dev_max_support'] for D in det_lifts]))

    # ---- GAP10's axis packets at the +nhat 2 r_s site with the chain's k_det: their channel projections for the chain side
    axis = None
    axis_proj, axis_labels = np.zeros((0, n_chain, 2), dtype=np.complex128), []
    if not args.no_axis and det['k_det'] > 0.0:
        nhat = L1.direction_vector(args.direction)
        pos, r_site = L1.site_on_direction(det['r_det'], args.direction)
        if int(np.max(np.abs(pos))) <= H:
            projs = []
            for k in (+float(det['k_det']), -float(det['k_det'])):
                spn = L1.helicity_spinors(k, nhat, m)
                for hel, sp in zip((+1, -1), spn):
                    cc, vv = L1.gap10_axis_packet(H, pos, nhat, k, sp, args.axis_sigma, args.axis_cut)
                    projs.append(L1.channel_projection(cc, vv, kappa_H, args.mj, n_chain, args.lower_sign, args.kappa_assign))
                    axis_labels.append(f"o5axis|k={k:+.5f}|h={'+' if hel > 0 else '-'}|sig={args.axis_sigma:g}")
            axis_proj = np.stack(projs, 0)
            axis = dict(direction=args.direction, site=[int(x) for x in pos], r=r_site, nhat=[float(x) for x in nhat], k=float(det['k_det']), sigma=float(args.axis_sigma),
                        cut=float(args.axis_cut), labels=axis_labels, n_sites=int(cc.shape[0]),
                        projection_norms=[float(np.sum(np.abs(g) ** 2)) for g in projs],
                        note="GAP10's own 3-dimensional receiver packets (packet_mode, raw) at the axis site; their channel projection (the adjoint lift) is the chain's reading of them, not a same-object detector")
            print(f"[axis] 4 packets at site {axis['site']} (r {r_site:.3f}, direction {args.direction}), k=+-{det['k_det']:.5f}, width {args.axis_sigma:g}, cut {args.axis_cut:g}; "
                  f"channel-projection norms {['%.4f' % x for x in axis['projection_norms']]} (the channel's share of each packet)", flush=True)
        else:
            print(f"[axis] the 2 r_s site is outside the box H={H}: no axis packets", flush=True)

    # ---- write
    chain_info = dict(json=args.chain, md5_json=md5(args.chain), npz=npz_path, md5_npz=md5(npz_path), stem=os.path.basename(args.chain)[:-5], kappa=kappa_H, Rc=float(col['Rc']),
                      sigma=float(col['sigma']), sigma_density=CH2.density_std(prof), ak0=float(col['ak0']), cut=col.get('cut'), E_mean=float(col['E_mean']), E_rms=float(col['E_rms']),
                      spinor=col['spinor'], Z=float(col['Z']), n=int(n_chain), r_h=float(rec['r_h']), rs=rs, chain_script=J.get('script'), profile=prof,
                      operator=((J.get('chain') or {}).get('operator') or {}).get('source'))
    source_info = dict(n_sites=S['n_sites'], R_lo=S['R_lo'], R_hi=S['R_hi'], support_half_width=S['support_half_width'], support_eps=args.support_eps, box_clipped=S['box_clipped'],
                       W_lift=S['W_lift'], gram_defect=S['gram_defect'], gram_dev_max_support=S['gram_dev_max_support'], t0_check=t0_check, profile=prof, chain_grid_max_diff=grid_diff)
    src_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE, chain=chain_info, source=source_info)
    np.savez_compressed(src_path, coords=S['coords'].astype(np.int32), values=S['values'], meta_json=json.dumps(src_meta, default=_json_default))
    det_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, chain=chain_info, members=[D['meta'] for D in det_lifts],
                    axis_k=(float(det['k_det']) if axis is not None else 0.0), axis_sigma=float(args.axis_sigma), axis_cut=float(args.axis_cut),
                    axis_direction=(args.direction if axis is not None else None), axis_site=(axis['site'] if axis is not None else None), axis_labels=axis_labels)
    np.savez_compressed(det_path, coords=coords_d.astype(np.int32), values=np.stack([D['values'] for D in det_lifts], 0), meta_json=json.dumps(det_meta, default=_json_default),
                        axis_proj=axis_proj, axis_labels=np.array(axis_labels, dtype=str))
    out = dict(script='o5_lift_v2.py', args=vars(args), chain=chain_info, H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=L1.CONVENTION_NOTE,
               spinor_convention_chain=J.get('spinor_convention'),
               source=dict(source_info, shell_gram=dict(j=sg['j'].tolist(), dev=sg['dev'].tolist(), count=sg['count'].tolist(), gaa=sg['gaa'].tolist(), gbb=sg['gbb'].tolist(),
                                                        gab_abs=np.abs(sg['gab']).tolist()),
                           alternatives={k_: dict(W_lift=v_['W_lift'], gram_defect=v_['gram_defect']) for k_, v_ in lifts.items()}),
               reduction=red,
               detector=dict(r_det=float(det['r_det']), sigma_det=float(det['sigma_det']), cut=float(det['cut']), k_det=float(det['k_det']), matched=bool(det['matched']),
                             omega_target=float(det['omega_target']), n_sites=int(coords_d.shape[0]), members=[D['meta'] for D in det_lifts]),
               amp_lift_error=amp_lift_error, axis=axis,
               files=dict(src_npz=src_path, det_npz=det_path, json=json_path, src_md5=md5(src_path), det_md5=md5(det_path)), wall=time.time() - tw)
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print(f"[done] {src_path} (md5 {out['files']['src_md5']}), {det_path} (md5 {out['files']['det_md5']}), {json_path} ({time.time() - tw:.0f}s)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
