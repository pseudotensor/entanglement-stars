#!/usr/bin/env python
"""o5_build_driver_v1.py -- builds gap10_run_v2_o5.py from gap10_run_v2_wl.py by the replacement list below (each 'old' must match
exactly once; the original is never edited). The additions, each marked O5 in the built driver:
  --o5_source_npz   the lifted radial-chain column (o5_lift_v1.py) replaces the wavelength classes as the single source class 'o5' (B 1 with
                    --wl_components raw); the driver's Killing moments, threshold filter, receivers, certificate, regions, shares and planes apply to it unchanged;
  --o5_det_npz      the lifted detector packets (both carrier signs) as an extra receiver packet group, their complex amplitudes in A;
  --o5_axis_k       4 explicit axis packets (GAP10's packet_mode, both signs, both helicities) at the +nhat 2 r_s receiver with the chain's k_det
                    (-1 = read it from the det npz; 0 = none);
  --o5_current      the fixed-cut number current through the cut r_h at every sweep end, J_> = 2 Im <P_> psi, h P_< psi> (1 extra application each);
  --o5_rh           the cut radius of the shares and regions (the chain's table root, so both descriptions read the same cut);
  --o5_tag          the stem's label.
Usage: python o5_build_driver_v1.py --wl <internal>/gap10_run_v2_wl.py --out <internal>/gap10_run_v2_o5.py
(the output must sit beside gap10_core_opt2.py and gap10_sources_wl.py, which the driver imports from its own directory).
"""
import argparse
import hashlib
import os
import py_compile
import sys

O5_BLOCK = r'''
# ------------------------------------------------------------------ O5 (busi/code/o5, GAP10_T7_final_v1.1 section 9): the lifted radial-chain column
def _o5_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _o5_load(path):
    """A lift file of o5_lift_v1.py: coords (n, 3) int, values (4, n) or (ndet, 4, n) complex, meta_json (a json string)."""
    z = np.load(path, allow_pickle=False)
    return z, json.loads(str(z['meta_json']))


def _o5_flat(coords, L, c):
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size and int(np.max(np.abs(coords))) > c:
        raise ValueError(f"O5: a lifted site lies outside this box (half-width {c}); the lift must be made for this H or a smaller one")
    return (((coords[:, 0] + c) * L + (coords[:, 1] + c)) * L + (coords[:, 2] + c)).astype(np.int64)


def _o5_source_class(args, L, c, rs, m, nhat, src_r, u_src, N_src, u_src_prep, N_src_prep, r_zero):
    """O5: the lifted column as the single source class 'o5' (the shell about the origin; its band weights on its own bounding cube)."""
    z, lm = _o5_load(args.o5_source_npz)
    coords = np.asarray(z['coords'], dtype=np.int64)
    values = np.asarray(z['values'], dtype=np.complex128)
    idx = _o5_flat(coords, L, c)
    prob = np.sum(np.abs(values) ** 2, 0)
    nrm = float(prob.sum())
    r = np.linalg.norm(coords, axis=1)
    hw = int(np.max(np.abs(coords)))
    bw = C.band_weights(values, coords, np.zeros(3, dtype=int), m, -u_src * nhat, nhat, hw, N_src)
    tails = {f'{R:.4f}': float(prob[r > R].sum() / nrm) for R in (r_zero, 1.25 * rs, 1.5 * rs, 2.0 * rs)}
    ch = lm['chain']
    k0 = float(ch['ak0'])
    E_mean = float(ch['E_mean'])
    mean = (prob @ coords) / nrm
    pmeta = dict(kind='pkt', role='src', pos=[0, 0, 0], r=float(ch['Rc']), omega=E_mean, sigma=float(ch['sigma']), res='raw', hel=1, k=k0, root_sign_changes=0,
                 vg=float('nan'), vg_run=float('nan'), u_prep=float(u_src_prep), N_prep=float(N_src_prep), u_run=float(u_src), N_run=float(N_src),
                 prep_mismatch=float(u_src - u_src_prep), predicted_shift=0.0, E_carrier_run=E_mean, band=bw, wl_class='o5',
                 wavelength=(2.0 * math.pi / abs(k0) if k0 != 0.0 else float('inf')), semiclassical=None, initial_tails=tails, retained=1.0, clipped=False,
                 n_sites=int(values.shape[1]), exterior_probability=float(prob[r > rs].sum() / nrm), interior_probability=float(prob[r < rs].sum() / nrm),
                 mean_r=float(prob @ r / nrm), mean_pos=[float(x) for x in mean],
                 position_std=[float(x) for x in np.sqrt(np.maximum(prob @ (coords - mean) ** 2 / nrm, 0.0))], band_projected=False,
                 upper_band_raw=float(bw['upper_band']), o5=dict(lift_npz=args.o5_source_npz, lift_md5=_o5_md5(args.o5_source_npz), lift_meta=lm))
    print(f"[o5 source] {args.o5_source_npz} (md5 {pmeta['o5']['lift_md5']}): chain column Rc={ch['Rc']:g} a kappa={ch['kappa']} sigma={ch['sigma']:g} a "
          f"ak0={k0:+.5f} <h>_chain={E_mean:.5f}; lifted on {values.shape[1]} sites, norm {nrm:.12f}, lift weight before renormalization {lm['source']['W_lift']:.6f}, "
          f"Gram defect {lm['source']['gram_defect']:.3e} (the lift's own error); band weights on the bounding cube: upper band {bw['upper_band']:.4f}, "
          f"outgoing pos. freq {bw['outgoing_positive_frequency']:.4f}, lattice scale {bw['lattice_scale']:.4f}; initial tails beyond r_h {tails[f'{r_zero:.4f}']:.4f}, "
          f"1.5rs {tails[f'{1.5 * rs:.4f}']:.4f}, 2rs {tails[f'{2.0 * rs:.4f}']:.4f}", flush=True)
    return dict(cls='o5', idx=idx, values=values, meta=pmeta)


def _o5_add_detectors(args, rec, L, c, rs, m):
    """O5: the lifted detector packets (the chain's own detectors, both carrier signs) as 1 receiver packet group."""
    z, lm = _o5_load(args.o5_det_npz)
    coords = np.asarray(z['coords'], dtype=np.int64)
    vals = np.asarray(z['values'], dtype=np.complex128)
    if vals.ndim == 2:
        vals = vals[None]
    idx = _o5_flat(coords, L, c)
    metas = []
    for i, md in enumerate(lm['members']):
        metas.append(dict(kind='pkt', role='o5det', r=float(md['r_det']), omega=float(md['omega_target']), sigma=float(md['sigma_det']), res='raw', hel=1,
                          k=float(md['k']), label=str(md['label']), matched_to=0, frac=float(md['r_det']) / rs, side=+1, o5=md))
    rec.add_packet_group(idx, [vals[i] for i in range(vals.shape[0])], metas)
    print(f"[o5 detectors] {args.o5_det_npz} (md5 {_o5_md5(args.o5_det_npz)}): {vals.shape[0]} lifted detector packets on {coords.shape[0]} sites: "
          + '; '.join(f"{md['label']} (energy matched: {md['matched']})" for md in lm['members']), flush=True)
    return lm


def _o5_add_axis_packets(args, rec, rec_sites, L, c, rs, m, nhat, common_half):
    """O5: 4 explicit GAP10 packets (packet_mode, raw) at the +nhat 2 r_s receiver: k = +-k_det, both helicities (the chain reads their channel projections)."""
    k_abs = float(args.o5_axis_k)
    sig = float(args.o5_axis_sigma)
    if k_abs < 0.0:
        if not args.o5_det_npz:
            return
        _, lm = _o5_load(args.o5_det_npz)
        k_abs = float(lm.get('axis_k', 0.0))
        sig = float(lm.get('axis_sigma', sig))
        if lm.get('axis_direction') not in (None, args.direction):
            raise ValueError(f"O5: the lift's axis packets were made for direction {lm.get('axis_direction')}, this run is {args.direction}")
    if k_abs <= 0.0:
        return
    site = [d for d in rec_sites if abs(d['frac'] - 2.0) < 1e-9 and d['side'] > 0]
    if not site:
        print("[o5 axis] no +nhat receiver at 2 rs in this box: axis packets skipped", flush=True)
        return
    site = site[0]
    for k in (+k_abs, -k_abs):
        spn = C.helicity_spinors(k, nhat, m)
        idx_ref, vals, metas = None, [], []
        for hel, sp in zip((+1, -1), spn):
            idx, values, coords, prob, meta = C.packet_mode(L, c, site['pos'], nhat, k, sp, sig, args.cut, rs, restriction='raw', common_half=common_half)
            if idx_ref is None:
                idx_ref = idx
            else:
                assert np.array_equal(idx_ref, idx)
            meta.update(kind='pkt', role='o5axis', r=float(site['r']), omega=float(C.disp_omega(k, site['u_run'], nhat, m, site['N_run'])[0]), sigma=sig, res='raw',
                        hel=int(hel), k=float(k), label=f"o5axis|k={k:+.5f}|h={'+' if hel > 0 else '-'}|sig={sig:g}", matched_to=0, frac=2.0, side=+1)
            vals.append(values)
            metas.append(meta)
        rec.add_packet_group(idx_ref, vals, metas)
    print(f"[o5 axis] 4 explicit packets at the +nhat 2 rs site {np.asarray(site['pos']).tolist()} (r={site['r']:.3f}): k=+-{k_abs:.5f}, both helicities, width {sig:g} a, cut {args.cut:g}", flush=True)


def _o5_cut_current(Hop, psi, mask_out):
    """O5: J_>(r_h) = <psi, i[h, P_>] psi> = 2 Im <P_> psi, h P_< psi> per batch element (outward positive; GAP10 section 3, HORIZON's rate (1)):
    1 application of h on the interior part; the transient is 1 masked copy plus apply's temporaries, after the sweep's buffers are released."""
    m_in = (~mask_out).to(C.FD)
    phi = psi * m_in
    del m_in
    hphi = Hop.apply(phi)
    del phi
    hphi.mul_(mask_out.to(C.FD))
    out = torch.empty(psi.shape[0], dtype=C.FD, device=psi.device)
    for b in range(psi.shape[0]):
        out[b] = 2.0 * torch.vdot(psi[b].reshape(-1), hphi[b].reshape(-1)).imag
    del hphi
    return out


'''

REPLACEMENTS = [
    # 1 the header line
    ('"""gap10_run_v2_wl.py -- GAP10 T7, the wavelength-class test',
     '"""gap10_run_v2_o5.py -- GAP10 T7, O5\'s witness exchange (GAP10_T7_final_v1.1 section 9): gap10_run_v2_wl.py with the lifted radial-chain\n'
     'column of o5_lift_v1.py as the single member (B 1), its lifted detector packets and explicit axis packets as extra receiver groups,\n'
     'the fixed-cut number current through r_h at every sweep end and the cut radius overridable to the chain\'s table root. Built by\n'
     'o5_build_driver_v1.py from its replacement list; never edited by hand; the original\'s docstring follows.\n'
     'gap10_run_v2_wl.py -- GAP10 T7, the wavelength-class test',
     'header'),
    # 2 hashlib
    ('import argparse\n', 'import argparse\nimport hashlib\n', 'import hashlib'),
    # 3 the arguments
    ("    ap.add_argument('--selftest', action='store_true')\n",
     "    ap.add_argument('--selftest', action='store_true')\n"
     "    ap.add_argument('--o5_source_npz', default='', help=\"O5: the lifted radial-chain column (o5_lift_v1.py) as the single source; replaces the wl classes\")\n"
     "    ap.add_argument('--o5_det_npz', default='', help=\"O5: the lifted detector packets at 2 rs (o5_lift_v1.py) as an extra receiver packet group\")\n"
     "    ap.add_argument('--o5_axis_k', type=float, default=-1.0, help=\"O5: |k| of 4 explicit axis packets (both signs, both helicities) at the +nhat 2 rs receiver; -1 = from the det npz; 0 = none\")\n"
     "    ap.add_argument('--o5_axis_sigma', type=float, default=6.0, help=\"O5: width of those axis packets (the det npz's value when --o5_axis_k is -1)\")\n"
     "    ap.add_argument('--o5_current', action='store_true', help=\"O5: the fixed-cut number current through the cut at every sweep end (1 extra application each)\")\n"
     "    ap.add_argument('--o5_rh', type=float, default=0.0, help=\"O5: the cut radius of the shares and regions (0 = the direction's crossing r_h as before)\")\n"
     "    ap.add_argument('--o5_tag', default='', help=\"O5: appended to the stem's o5 label\")\n",
     'arguments'),
    # 4 the stem
    ("f\"_{args.direction}_d{args.depth:g}_wl{''.join(args.wl_classes)}_{comps}_cheb{args.tag}\")",
     "f\"_{args.direction}_d{args.depth:g}_\" + (f\"o5{args.o5_tag}\" if args.o5_source_npz else f\"wl{''.join(args.wl_classes)}_{comps}\") + f\"_cheb{args.tag}\")",
     'stem'),
    # 5 the wl classes are not built when the o5 source is given
    ("    for cls in args.wl_classes:\n", "    for cls in ([] if args.o5_source_npz else args.wl_classes):\n", 'wl loop'),
    # 6 the o5 source class
    ("    # the source packets as receivers (their own columns carry the Killing moments)\n",
     "    if args.o5_source_npz:\n"
     "        classes.append(_o5_source_class(args, L, c, rs, m, nhat, src_r, u_src, N_src, u_src_prep, N_src_prep, r_zero))\n"
     "    # the source packets as receivers (their own columns carry the Killing moments)\n",
     'source class'),
    # 7 the detector and axis groups before the final finalize
    ("    rec.finalize()\n    if hasattr(rec, '_opt2_cache'):\n",
     "    if args.o5_det_npz:\n"
     "        _o5_add_detectors(args, rec, L, c, rs, m)\n"
     "    if args.o5_axis_k != 0.0:\n"
     "        _o5_add_axis_packets(args, rec, rec_sites, L, c, rs, m, nhat, common_half)\n"
     "    rec.finalize()\n    if hasattr(rec, '_opt2_cache'):\n",
     'detector groups'),
    # 8 the cut radius of the geometry
    ("    geo = C.Geometry(L, c, rs, r_zero, dev, region_fracs=tuple(args.regions))\n",
     "    geo = C.Geometry(L, c, rs, (float(args.o5_rh) if args.o5_rh > 0.0 else r_zero), dev, region_fracs=tuple(args.regions))\n"
     "    if args.o5_rh > 0.0:\n"
     "        print(f\"[o5 cut] the shares' and regions' cut r_h overridden to {args.o5_rh:.6f} a (the chain's table root; the direction's crossing {r_zero:.6f} a stays in the json as r_h)\", flush=True)\n",
     'geometry cut'),
    # 9 the current at t 0
    ("    grams, norm_excess = [geo.gram(psi)], []\n",
     "    grams, norm_excess = [geo.gram(psi)], []\n"
     "    currents = [_o5_cut_current(Hop, psi, geo.region_masks[0]) if args.o5_current else None]\n",
     'current init'),
    # 10 the current at every sweep end
    ("        G = geo.gram(psi)\n",
     "        G = geo.gram(psi)\n"
     "        J_rh = _o5_cut_current(Hop, psi, geo.region_masks[0]) if args.o5_current else None\n",
     'current per sweep'),
    # 11 its print
    ("        if s == 0:\n            print(f\"[projection] this run about {wall * nsweeps / 60:.1f} min\", flush=True)\n",
     "        if args.o5_current:\n"
     "            print(f\"[o5 current] t={t:.1f} ({t / rs:.2f} rs): fixed-cut number current through the cut {geo.share_radii[0][1]:.4f} a, J_> = {float(J_rh[0]):+.6e} (outward positive); \"\n"
     "                  f\"the weight beyond it {float(dd[2][0, 0]):.6f}, its change since t 0 {float(dd[2][0, 0] - shares[0][0, 0]):+.6e}\", flush=True)\n"
     "        if s == 0:\n            print(f\"[projection] this run about {wall * nsweeps / 60:.1f} min\", flush=True)\n",
     'current print'),
    # 12 store the current
    ("        res_exact.append(rex); energies.append(es); grams.append(G)\n",
     "        res_exact.append(rex); energies.append(es); grams.append(G); currents.append(J_rh)\n",
     'current store'),
    # 13 the output arrays
    ("    for gi, g in enumerate(rec.groups):\n        arrays[f'group_S_{gi}'] = g['S']\n",
     "    if args.o5_current:\n"
     "        arrays['o5_current_rh'] = np.array([[float(x) for x in jj.cpu().numpy()] for jj in currents])\n"
     "    for gi, g in enumerate(rec.groups):\n        arrays[f'group_S_{gi}'] = g['S']\n",
     'output arrays'),
    # 14 the helper block
    ("def _make_state(cl, L, dev):\n", O5_BLOCK.lstrip('\n') + "def _make_state(cl, L, dev):\n", 'helper block'),
]


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build(src_text):
    text = src_text
    for old, new, note in REPLACEMENTS:
        k = text.count(old)
        if k != 1:
            raise SystemExit(f"replacement '{note}': the old text matches {k} times, not once; the driver changed: stop and inspect")
        text = text.replace(old, new)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="build gap10_run_v2_o5.py from gap10_run_v2_wl.py by the replacement list")
    ap.add_argument('--wl', required=True, help="gap10_run_v2_wl.py (read only)")
    ap.add_argument('--out', required=True, help="gap10_run_v2_o5.py, beside gap10_core_opt2.py and gap10_sources_wl.py")
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    if os.path.abspath(args.wl) == os.path.abspath(args.out):
        raise SystemExit("the output must not be the original")
    if os.path.exists(args.out) and not args.force:
        print(f"[skip] {args.out} exists (--force to rebuild)", flush=True)
        return 0
    with open(args.wl) as fh:
        src = fh.read()
    out = build(src)
    with open(args.out, 'w') as fh:
        fh.write(out)
    py_compile.compile(args.out, doraise=True)
    print(f"[built] {args.out} (md5 {md5(args.out)}) from {args.wl} (md5 {md5(args.wl)}): {len(REPLACEMENTS)} replacements, each matched once; compiles", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
