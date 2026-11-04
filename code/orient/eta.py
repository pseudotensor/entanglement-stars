#!/usr/bin/env python3
"""Rational planar eta, Eq. (1.14). Numerical errors are estimates, not bounds."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from lattice_entropy import (
    primitive, interval_covariance, entropy
)


def channel(job):
    p, q, L, F, mass, gpu = job
    C = interval_covariance(p, q, L, F, mass)
    return entropy(C, gpu)/(2*np.linalg.norm(p))


def value(p, L=32, nk=16, nfft=512, mass=0., workers=1, gpu=-1):
    p = tuple(map(int, primitive(p)))
    if gpu >= 0 and workers != 1:
        raise ValueError("Use one process per GPU")
    # Full primitive transverse zone, Eq. (1.13).
    q = 2*np.pi*(np.arange(nk)+.5)/nk
    jobs = [(p, (a, b), L, nfft, mass, gpu) for a in q for b in q]
    if workers == 1:
        vals = list(map(channel, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            vals = list(ex.map(channel, jobs, chunksize=1))
    return float(np.mean(vals))


def converge(p, depth=16., nk=12, mass=0., workers=1, gpu=-1):
    p = tuple(map(int, primitive(p)))
    L = max(8, 2*int(np.ceil(depth*np.linalg.norm(p)/2)))
    F = 16*L
    configs = [
        (L, nk, F),
        (2*L, nk, F),
        (2*L, 2*nk, F),
        (2*L, 2*nk, 2*F),
        (4*L, 2*nk, 2*F),
    ]
    rows = []
    for l, k, f in configs:
        z = value(p, l, k, f, mass, workers, gpu)
        rows.append(dict(length=l, nk=k, nfft=f, eta=z))
    z = np.array([r["eta"] for r in rows])
    # Eq. (1.14); these differences do NOT constitute a rigorous enclosure.
    errors = dict(
        length=abs(z[4]-z[3]),
        transverse=abs(z[2]-z[1]),
        Fourier=abs(z[3]-z[2]),
    )
    return dict(
        normal=p, eta=float(z[4]),
        estimated_error=float(2*sum(errors.values())),
        error_components=errors, records=rows,
        certified=False,
        extrapolated_length_p2=float(z[4]+(z[4]-z[3])/3),
        note="p=2 extrapolation is diagnostic, not assumed in eta/error."
    )


def folding_test(p, L=16, nk=6, fold=2, nfft=512):
    """Exact zone folding of an enlarged transverse cell, Eq. (1.13).

    A folded channel is the direct sum of primitive channels at
    q=(Q+2*pi*r)/fold. Entropy is additive on that direct sum.
    """
    coarse = 2*np.pi*(np.arange(nk)+.5)/nk
    folded = 0.
    for a in coarse:
        for b in coarse:
            for r in range(fold):
                for s in range(fold):
                    q = ((a+2*np.pi*r)/fold, (b+2*np.pi*s)/fold)
                    folded += channel((tuple(p), q, L, nfft, 0., -1))
    folded /= (nk*fold)**2
    direct = value(p, L, nk*fold, nfft)
    return abs(folded-direct)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("normal", nargs=3, type=int)
    ap.add_argument("--depth", type=float, default=16)
    ap.add_argument("--nk", type=int, default=12)
    ap.add_argument("--mass", type=float, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--output", default="eta.json")
    a = ap.parse_args()
    out = converge(a.normal, a.depth, a.nk, a.mass, a.workers, a.gpu)
    text = json.dumps(out, indent=2)
    print(text)
    with open(a.output, "w") as f:
        f.write(text+"\n")
