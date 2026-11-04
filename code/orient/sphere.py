#!/usr/bin/env python3
"""Solid-angle quadrature: Eqs. (1.16), (1.21), (1.22).

The geometric bound is rigorous, conditional on exact vertex values.
Planar uncertainties supplied by eta.py are empirical. Never conflated.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from eta import converge
from lattice_entropy import canonical

ETA100 = .7865834


def triangles(M):
    def p(i, j):
        return (M, i, j)
    out = []
    for i in range(M):
        for j in range(i):
            A, B = p(i, j), p(i+1, j)
            C, D = p(i+1, j+1), p(i, j+1)
            out.extend([(A, B, C), (A, C, D)])
        out.append((p(i, i), p(i+1, i), p(i+1, i+1)))
    return out


def geometry(T):
    V = np.asarray(T, float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    a, b, c = V
    area = 2*np.arctan2(
        abs(np.dot(a, np.cross(b, c))),
        1+np.dot(a,b)+np.dot(b,c)+np.dot(c,a)
    )
    diameter = max(np.linalg.norm(a-b), np.linalg.norm(b-c),
                   np.linalg.norm(c-a))
    return area, diameter


def run(M, cache, depth, nk, workers):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    Ts = triangles(M)
    normals = sorted({canonical(p) for T in Ts for p in T})
    data = {}
    for p in normals:
        file = cache / ("eta_"+"_".join(map(str,p))+".json")
        if file.exists():
            rec = json.loads(file.read_text())
        else:
            rec = converge(p, depth, nk, workers=workers)
            file.write_text(json.dumps(rec, indent=2)+"\n")
        data[p] = rec

    total = numerical = geometric = weight = 0.
    for T in Ts:
        area, diameter = geometry(T)
        w = 12*area/np.pi
        rows = [data[canonical(p)] for p in T]
        total += w*np.mean([r["eta"] for r in rows])
        numerical += w*np.mean([r["estimated_error"] for r in rows])
        geometric += w*np.sqrt(3)*ETA100*diameter  # Eq. (1.22)
        weight += w
    assert abs(weight-1) < 1e-12
    return dict(
        mesh=M, distinct_normals=len(normals), weight_sum=weight,
        eta_sphere=total,
        geometric_error_bound=geometric,
        planar_error_estimate=numerical,
        conditional_interval=[total-geometric-numerical,
                              total+geometric+numerical],
        certified=False,
        ratio_BH=4*.2745414*total,
        warning="Interval uses empirical planar errors; not a certified enclosure."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=2)
    ap.add_argument("--cache", default="eta_cache")
    ap.add_argument("--depth", type=float, default=16)
    ap.add_argument("--nk", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", default="sphere.json")
    a = ap.parse_args()
    out = run(a.mesh, a.cache, a.depth, a.nk, a.workers)
    text = json.dumps(out, indent=2)
    print(text)
    Path(a.output).write_text(text+"\n")
