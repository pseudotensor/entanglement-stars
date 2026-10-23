#!/usr/bin/env python
"""n2_retained_gram_v1.py (reviewer A, foundations item N2: ReviewA_foundations_v1.md sec. 3.2 N2; paper App. app:rows first
paragraph, Eq. dm-average; busi's N3 Gram findings in FOUNDATIONS_N3_N4_N5_v1.md).

DATA script (box CPU, numpy only). For the paper's kernel k_{ell,a} = K_ell * B_a and the periodic parent's plane-wave tests:
(1) the Fourier transform khat(K) of the kernel and the zone sums gamma_e(K), gamma_pp(K) of the energy and momentum vertices,
    which give the Gram matrix of the retained tests, diagonal in K with 1 + 3 blocks: Gamma = |khat(K)|^2 gamma(K);
(2) the condition number of the retained Gram over the band |K| <= kappa_* / ell for kappa_* = pi (the band the spacing-ell
    shifted-kernel tests resolve) and 2 pi (spacing ell/2), at ell = 2 a and 4 a, m a = 0.2: the numbers behind busi's 6.8e5
    and 1e15;
(3) the fraction of a smooth source's norm beyond the band, for Gaussian profiles of width L_geom = 2, 4, 8, 16 ell and for the
    kernel-smoothed uniform ball of radius 4 ell (busi's N3 parent at ell = 2): the size of the bridge's remainder.

Output: <out>/n2_retained_gram_v1.json. Never edits its inputs."""
import argparse
import itertools
import json
import os

import numpy as np

VERSION = "n2_retained_gram_v1"


def khat_K(k, ell):
    """3D Fourier transform of K_ell(z) = (9009/(1024 pi ell^3)) (1 - |z|^2/ell^2)_+^5, normalized to 1 at k = 0."""
    k = np.atleast_1d(np.asarray(k, float))
    r = np.linspace(0, 1, 20001)
    w = (1 - r ** 2) ** 5 * r ** 2
    out = np.empty_like(k)
    for i, kk in enumerate(k):
        x = kk * ell * r
        j0 = np.where(x > 1e-12, np.sin(x) / np.where(x > 1e-12, x, 1.0), 1.0)
        out[i] = np.trapz(w * j0, r) / np.trapz(w, r)
    return out


def bhat(Kvec, a=1.0):
    """transform of the cell function B_a: product of sinc(K_d a / 2)."""
    x = np.asarray(Kvec, float) * a / 2
    return float(np.prod(np.where(np.abs(x) > 1e-12, np.sin(x) / np.where(np.abs(x) > 1e-12, x, 1.0), 1.0)))


def zone_sums(Kvec, m, npts=48):
    """gamma_e(K) = <4 m^2 + |d(p+K) + d(p)|^2>_p, gamma_pp(K)_{jj'} = <(q_j(p+K) + q_j(p))(q_j'(p+K) + q_j'(p))>_p, a = 1."""
    g = (np.arange(npts) + 0.5) * 2 * np.pi / npts - np.pi
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    PK = P + np.asarray(Kvec, float)[None, :]
    d, dK = np.sin(P), np.sin(PK)
    q, qK = np.sin(2 * P) / 2, np.sin(2 * PK) / 2
    ge = float(np.mean(4 * m ** 2 + np.sum((d + dK) ** 2, axis=1)))
    S = q + qK
    gpp = (S.T @ S) / S.shape[0]
    return ge, gpp


def tail_fraction_radial(profile, rgrid, kmax, kgrid):
    """fraction of int |shat|^2 k^2 dk beyond kmax for a radial profile s(r): shat(k) = 4 pi int s(r) sin(kr)/(kr) r^2 dr."""
    sh = np.empty_like(kgrid)
    for i, k in enumerate(kgrid):
        x = k * rgrid
        j0 = np.where(x > 1e-12, np.sin(x) / np.where(x > 1e-12, x, 1.0), 1.0)
        sh[i] = 4 * np.pi * np.trapz(profile * j0 * rgrid ** 2, rgrid)
    power = sh ** 2 * kgrid ** 2
    tot = np.trapz(power, kgrid)
    tail = np.trapz(np.where(kgrid > kmax, power, 0.0), kgrid)
    return float(tail / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = dict(version=VERSION, m=args.m)
    # (1), (2): the retained Gram over the band
    for ell in (2.0, 4.0):
        rec = {}
        for kappa in (np.pi, 2 * np.pi):
            kmax = kappa / ell
            # sample the band on a 3D grid of K (periodic parent L -> infinity limit: continuous K)
            nK = 9
            ks = np.linspace(-kmax, kmax, nK)
            vals = []
            for Kv in itertools.product(ks, repeat=3):
                Kv = np.array(Kv)
                if np.linalg.norm(Kv) > kmax + 1e-12:
                    continue
                kh = khat_K(np.linalg.norm(Kv), ell)[0] * bhat(Kv)
                ge, gpp = zone_sums(Kv, args.m)
                eig = np.linalg.eigvalsh(gpp)
                vals.append((kh ** 2 * ge, kh ** 2 * eig.min(), kh ** 2 * eig.max(), kh ** 2))
            vals = np.array(vals)
            gram_entries = np.concatenate([vals[:, 0], vals[:, 1], vals[:, 2]])
            rec["kappa_%.3f" % kappa] = dict(
                kmax=kmax, n_K_sampled=int(len(vals)),
                khat2_min=float(vals[:, 3].min()), khat2_max=float(vals[:, 3].max()),
                khat2_at_band_edge_axis=float((khat_K(kmax, ell)[0] * bhat([kmax, 0, 0])) ** 2),
                gram_min=float(gram_entries.min()), gram_max=float(gram_entries.max()),
                cond=float(gram_entries.max() / gram_entries.min()))
        out["ell_%g" % ell] = rec
    # the kernel transform along the axis for reference
    ell = 2.0
    out["khat_axis_ell2"] = {("kl_%.3f" % x): float(khat_K(x / ell, ell)[0] * bhat([x / ell, 0, 0])) for x in (0.5, 1, 2, np.pi, 4, 2 * np.pi, 8)}
    # (3): tails beyond pi/ell of smooth sources
    tails = {}
    for ell in (2.0,):
        kmax = np.pi / ell
        kgrid = np.linspace(1e-4, 6.0, 6000)
        for Lg in (2, 4, 8, 16):
            Lgeom = Lg * ell
            r = np.linspace(0, 12 * Lgeom, 24001)
            prof = np.exp(-r ** 2 / (2 * Lgeom ** 2))
            tails["gaussian_Lgeom_%dell" % Lg] = tail_fraction_radial(prof, r, kmax, kgrid)
        # busi's N3 parent: uniform ball of radius 4 ell (R_b = 8 at ell = 2), smoothed by the kernel (in k: ball transform x khat)
        Rb = 4 * ell
        r = np.linspace(0, 6 * Rb, 24001)
        ball = (r <= Rb).astype(float)
        sh = np.empty_like(kgrid)
        for i, k in enumerate(kgrid):
            x = k * r
            j0 = np.where(x > 1e-12, np.sin(x) / np.where(x > 1e-12, x, 1.0), 1.0)
            sh[i] = 4 * np.pi * np.trapz(ball * j0 * r ** 2, r) * khat_K(k, ell)[0] * bhat([k, 0, 0])
        power = sh ** 2 * kgrid ** 2
        tails["smoothed_ball_R4ell"] = float(np.trapz(np.where(kgrid > kmax, power, 0.0), kgrid) / np.trapz(power, kgrid))
        tails["smoothed_ball_R4ell_amplitude"] = float(np.sqrt(tails["smoothed_ball_R4ell"]))
    out["tail_fraction_of_norm2_beyond_pi_over_ell"] = tails
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, VERSION + ".json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
