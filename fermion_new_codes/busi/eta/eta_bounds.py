#!/usr/bin/env python3
"""eta_bounds.py -- convex (support-function) bounds on the sphere average of the planar coefficient from a
table of directional intervals eta_i +/- hw_i (ETA_spec_v1.md section 8).

Upper bound: W is inside the polytope P = {x : u_i.x <= eta_i + hw_i} (all 48 cubic images of every direction);
the sphere average of h_P is (1/8 pi) sum_edges L_e alpha_e (exterior dihedral angles).
Lower bound: the support point of W in direction u_i lies in the cap C_i = P and {u_i.x >= eta_i - hw_i}, so
h_W(n) >= g_i(n) = min over C_i of n.x; on each spherical triangle of the gnomonic mesh of the fundamental cubic
triangle the linear interpolant of ONE g_i from its vertex values is a lower bound (g_i concave, homogeneous),
integrated with the exact vector area; the maximum over candidate caps is taken per triangle.
Also: orientation-range bounds, the solid-angle vertex quadrature on nested meshes (the solve lane's rule),
the 3-direction artifact, the finite-mass estimate b_sphere from b_table.tsv, sensitivity to hw x 2 and x 10.
"""
import os
import sys
import argparse
import json
import math
import time


def _early_threads():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--threads', type=int, default=1)
    a, _ = pre.parse_known_args()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = str(a.threads)
    return a.threads


THREADS = _early_threads()
import numpy as np  # noqa: E402
from scipy.spatial import HalfspaceIntersection, ConvexHull, cKDTree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_common as ec  # noqa: E402


# ---------------------------------------------------------------------------------------------- input
def read_table(path, mass=0.0):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if header is None:
                header = parts
                continue
            d = dict(zip(header, parts))
            if abs(float(d['mass']) - mass) > 1e-12:
                continue
            rows.append(dict(p=(int(d['p1']), int(d['p2']), int(d['p3'])), pnorm=float(d['pnorm']),
                             eta=float(d['eta']), hw=float(d['eta_hw'])))
    if not rows:
        raise SystemExit(f"no rows with mass {mass} in {path}")
    return rows


def read_btable(path):
    out = {}
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if header is None:
                header = parts
                continue
            d = dict(zip(header, parts))
            out[(int(d['p1']), int(d['p2']), int(d['p3']))] = dict(b=float(d['b']), b_hw=float(d['b_hw']),
                                                                   resid=float(d['resid_max']))
    return out


def build_normals(rows, hw_scale=1.0, use_exact=False):
    """Unit normals of all cubic images with their interval, plus the orbit index of each normal."""
    U, eta, hw, orb = [], [], [], []
    for k, rw in enumerate(rows):
        e, w = rw['eta'], rw['hw'] * hw_scale
        if use_exact and ec.canonical_orbit(rw['p']) == (1, 0, 0):
            e, w = ec.TARGET_ETA100, 1e-12
        if use_exact and ec.canonical_orbit(rw['p']) == (1, 1, 1):
            e, w = ec.TARGET_ETA111, 1e-12
        for q in ec.cubic_images(rw['p']):
            v = np.array(q, dtype=float)
            U.append(v / np.linalg.norm(v))
            eta.append(e)
            hw.append(w)
            orb.append(k)
    return np.array(U), np.array(eta), np.array(hw), np.array(orb)


# ---------------------------------------------------------------------------------------------- polytope
def polytope_from_upper(U, upper):
    hs = HalfspaceIntersection(np.column_stack([U, -upper]), np.zeros(3))
    V = np.unique(np.round(hs.intersections, 11), axis=0)
    try:
        hull = ConvexHull(V)
    except Exception:
        hull = ConvexHull(V, qhull_options='QJ')
    return V, hull


def hull_edges(hull):
    """Edges of the triangulated hull as (a, b, face_f, face_g), each once."""
    simplices = hull.simplices
    nb = hull.neighbors
    E = []
    for f in range(len(simplices)):
        for i in range(3):
            g = nb[f, i]
            if g > f:
                a, b = [int(simplices[f, k]) for k in range(3) if k != i]
                E.append((a, b, f, int(g)))
    return np.array(E, dtype=int)


def mean_support_polytope(V, hull):
    """(1/4 pi) int_S2 h_P = (1/8 pi) sum_e L_e alpha_e; coplanar triangle pairs contribute 0 (atan2 of cross, dot)."""
    E = hull_edges(hull)
    n = hull.equations[:, :3]
    nf, ng = n[E[:, 2]], n[E[:, 3]]
    cross = np.linalg.norm(np.cross(nf, ng), axis=1)
    dot = np.sum(nf * ng, axis=1)
    ang = np.arctan2(cross, dot)
    Le = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(np.sum(Le * ang) / (8.0 * math.pi)), E


def cap_points(V, E, u, lower, tol=1e-11):
    """Vertices of the cap P and {u.x >= lower}: polytope vertices above the plane plus edge intersections."""
    d = V @ u - lower
    parts = [V[d >= -tol]]
    a, b = E[:, 0], E[:, 1]
    cross = d[a] * d[b] < 0
    if np.any(cross):
        ia, ib = a[cross], b[cross]
        t = d[ia] / (d[ia] - d[ib])
        parts.append(V[ia] + (V[ib] - V[ia]) * t[:, None])
    P = np.concatenate(parts)
    if len(P) == 0:
        raise ValueError("empty cap: the directional intervals are inconsistent with convexity")
    return np.unique(np.round(P, 12), axis=0)


# ---------------------------------------------------------------------------------------------- sphere mesh
def fundamental_mesh(M):
    """Gnomonic mesh of the cubic fundamental triangle: unit vertices (1, j/M, k/M)/norm, 2 M^2 triangles."""
    idx = {}
    verts = []
    for j in range(M + 1):
        for k in range(j + 1):
            idx[(j, k)] = len(verts)
            verts.append((1.0, j / M, k / M))
    tris = []
    for j in range(M):
        for k in range(j + 1):
            tris.append((idx[(j, k)], idx[(j + 1, k)], idx[(j + 1, k + 1)]))
            if k < j:
                tris.append((idx[(j, k)], idx[(j + 1, k + 1)], idx[(j, k + 1)]))
    V = np.array(verts, dtype=float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    return V, np.array(tris, dtype=int)


def mesh_integer_vertices(M):
    return [(M, j, k) for j in range(M + 1) for k in range(j + 1)]


def triangle_vector_areas(N):
    """V_T = int_T n dOmega for spherical triangles with unit vertex rows N (ntri, 3, 3)."""
    vec = np.zeros((len(N), 3))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cr = np.cross(N[:, i], N[:, j])
        sn = np.linalg.norm(cr, axis=1)
        dt = np.sum(N[:, i] * N[:, j], axis=1)
        ang = np.arctan2(sn, dt)
        vec += 0.5 * (ang / sn)[:, None] * cr
    cen = N.mean(axis=1)
    sign = np.sign(np.sum(vec * cen, axis=1))
    return vec * sign[:, None]


def triangle_weights(N, vec):
    """w with sum_j w_j n_j = V_T: then sum_j w_j g(n_j) = a.V_T for the linear interpolant a.n of g."""
    return np.linalg.solve(np.transpose(N, (0, 2, 1)), vec[..., None])[..., 0]


def spherical_triangle_areas(N):
    n1, n2, n3 = N[:, 0], N[:, 1], N[:, 2]
    num = np.abs(np.sum(n1 * np.cross(n2, n3), axis=1))
    den = 1.0 + np.sum(n1 * n2, axis=1) + np.sum(n2 * n3, axis=1) + np.sum(n3 * n1, axis=1)
    return 2.0 * np.arctan2(num, den)


def lower_bound_mesh(V, E, U, lower_vals, M, K=32, batch=1024, hP_vertices=True):
    """Cap lower bound on the sphere average, the convex vertex interpolation of h_P (an upper bound), and
    a lower bound on the minimum of eta over the sphere."""
    Vm, T = fundamental_mesh(M)
    N = Vm[T]
    vec = triangle_vector_areas(N)
    W = triangle_weights(N, vec)
    if W.min() < -1e-12:
        raise RuntimeError("negative integration weight in the mesh")
    cen = N.mean(axis=1)
    cen /= np.linalg.norm(cen, axis=1)[:, None]
    K = min(K, len(U))
    _, ids = cKDTree(U).query(cen, k=K)
    ids = np.atleast_2d(ids)
    needed = np.unique(ids)
    caps = {int(i): cap_points(V, E, U[i], lower_vals[i]) for i in needed}
    maxlen = max(len(c) for c in caps.values())
    cap_arr = np.zeros((len(U), maxlen, 3))
    for i, c in caps.items():
        cap_arr[i, :len(c)] = c
        cap_arr[i, len(c):] = c[0]
    best = np.full(len(T), -np.inf)
    vmin = np.full(len(T), -np.inf)
    for st in range(0, len(T), batch):
        sl = slice(st, min(st + batch, len(T)))
        pts = cap_arr[ids[sl]]                                  # (B, K, maxlen, 3)
        dots = np.einsum('bkmd,bjd->bkjm', pts, N[sl])          # (B, K, 3, maxlen)
        g = dots.min(axis=3)                                    # g_i(n_j)
        integ = np.einsum('bkj,bj->bk', g, W[sl])
        best[sl] = integ.max(axis=1)
        vmin[sl] = g.min(axis=2).max(axis=1)
    lower_avg = float((12.0 / math.pi) * best.sum())
    out = dict(M=M, n_triangles=int(len(T)), caps_used=int(len(needed)),
               cap_points_total=int(sum(len(c) for c in caps.values())),
               eta_lower=lower_avg, min_eta_lower=float(vmin.min()))
    if hP_vertices:
        hP = (Vm @ V.T).max(axis=1)
        out['eta_upper_interp'] = float((12.0 / math.pi) * np.sum(W * hP[T]))
    return out


def vertex_quadrature(values_at, M):
    """Solid-angle vertex rule on mesh M (the solve lane's rule): weights 12 area_T/pi, split equally."""
    Vm, T = fundamental_mesh(M)
    areas = spherical_triangle_areas(Vm[T])
    vals = np.array([values_at(v) for v in mesh_integer_vertices(M)])
    tot = 0.0
    for t, a in zip(T, areas):
        tot += (12.0 * a / math.pi) * np.mean(vals[t])
    return float(tot), float(np.sum(12.0 * areas / math.pi))


def reduced(q):
    g = math.gcd(math.gcd(q[0], q[1]), q[2])
    return tuple(sorted((abs(v) // g for v in q), reverse=True))


def run_bounds(rows, hw_scale=1.0, mesh_list=(64, 128, 256), K=32, use_exact=False, verbose=True):
    t0 = time.time()
    U, eta, hw, orb = build_normals(rows, hw_scale, use_exact)
    upper_vals = eta + hw
    lower_vals = eta - hw
    V, hull = polytope_from_upper(U, upper_vals)
    mean_hP, E = mean_support_polytope(V, hull)
    hP_at_U = (U @ V.T).max(axis=1)
    violations = int(np.sum(lower_vals > hP_at_U + 1e-9))
    out = dict(n_orbits=len(rows), n_normals=int(len(U)), n_vertices=int(len(V)), hw_scale=hw_scale, use_exact=use_exact,
               max_hw=float(hw.max()), eta_upper=mean_hP, support_violations=violations,
               max_eta_upper=float(np.linalg.norm(V, axis=1).max()), max_eta_lower=float(lower_vals.max()),
               min_eta_upper=float(upper_vals.min()), meshes=[])
    for M in mesh_list:
        lb = lower_bound_mesh(V, E, U, lower_vals, M, K=K)
        out['meshes'].append(lb)
        if verbose:
            print(f"    mesh {M}: lower {lb['eta_lower']:.10f} upper(exact edges) {mean_hP:.10f} upper(interp) {lb['eta_upper_interp']:.10f} "
                  f"min_eta_lower {lb['min_eta_lower']:.8f} caps {lb['caps_used']}")
    out['eta_lower'] = max(m['eta_lower'] for m in out['meshes'])
    out['min_eta_lower'] = max(m['min_eta_lower'] for m in out['meshes'])
    out['midpoint'] = 0.5 * (out['eta_lower'] + out['eta_upper'])
    out['halfwidth'] = 0.5 * (out['eta_upper'] - out['eta_lower'])
    out['a_over_lP'] = [2.0 * math.sqrt(out['eta_lower']), 2.0 * math.sqrt(out['eta_upper'])]
    out['G_over_a2'] = [1.0 / (4.0 * out['eta_upper']), 1.0 / (4.0 * out['eta_lower'])]
    out['seconds'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--table', required=True)
    ap.add_argument('--mass', type=float, default=0.0, help='use the rows of this ma (eta(n, y) is a support function too)')
    ap.add_argument('--hw-scales', nargs='+', type=float, default=[1.0, 2.0, 10.0])
    ap.add_argument('--pmax-list', nargs='*', type=int, default=None, help='bounds for the subsets max component <= pmax')
    ap.add_argument('--mesh', nargs='+', type=int, default=[64, 128, 256])
    ap.add_argument('--neighbors', type=int, default=32)
    ap.add_argument('--use-exact', action='store_true', help='substitute the exact [100] and [111] values (hw 1e-12)')
    ap.add_argument('--btable', default=None)
    ap.add_argument('--out', default='eta_bounds.json')
    ap.add_argument('--handoff', default='handoff_eta_table.tsv')
    args = ap.parse_args()

    rows = read_table(args.table, mass=args.mass)
    allp = {rw['p']: rw for rw in rows}
    result = dict(table=os.path.abspath(args.table), mass=args.mass, n_rows=len(rows), neighbors=args.neighbors,
                  conventions='nats; one 4-component sine-Dirac field; one face; per unit area a^2', runs=[])
    pmax_all = max(max(rw['p']) for rw in rows)
    pmax_list = args.pmax_list if args.pmax_list else [pmax_all]
    for pm in pmax_list:
        sub = [rw for rw in rows if max(rw['p']) <= pm]
        if len(sub) < 3:
            continue
        for sc in args.hw_scales:
            print(f"pmax {pm} ({len(sub)} orbits), half-widths x {sc:g}:")
            res = run_bounds(sub, hw_scale=sc, mesh_list=args.mesh, K=args.neighbors, use_exact=args.use_exact)
            res['pmax'] = pm
            result['runs'].append(res)
            print(f"  => {res['eta_lower']:.10f} < eta_sphere < {res['eta_upper']:.10f}  (midpoint {res['midpoint']:.8f}, "
                  f"half-width {res['halfwidth']:.2e}; a/l_P in [{res['a_over_lP'][0]:.6f}, {res['a_over_lP'][1]:.6f}]; "
                  f"max eta in [{res['max_eta_lower']:.6f}, {res['max_eta_upper']:.6f}]; min eta in [{res['min_eta_lower']:.6f}, {res['min_eta_upper']:.6f}]; "
                  f"support violations {res['support_violations']}; {res['seconds']:.0f} s)")
    # the solve lane's vertex rule on nested meshes, for comparison with q2, q3, q4 (0.9011, 0.9110, 0.9145)
    def eta_at(q):
        return allp[reduced(q)]['eta']

    quad = {}
    for M in (2, 3, 4, 6, 8, 10):
        try:
            qv, wsum = vertex_quadrature(eta_at, M)
            quad[M] = qv
            print(f"vertex rule mesh {M}: {qv:.8f} (weights sum {wsum:.12f})")
        except KeyError:
            pass
    result['vertex_rule'] = quad
    # the 3-direction artifact (exact weights 10/35, 16/35, 9/35 on the span of 1, p4, p6)
    try:
        c0 = (10 * eta_at((1, 0, 0)) + 16 * eta_at((1, 1, 0)) + 9 * eta_at((1, 1, 1))) / 35.0
        result['three_direction_artifact'] = c0
        print(f"3-direction harmonic interpolation (10 eta100 + 16 eta110 + 9 eta111)/35 = {c0:.6f}: an artifact, not a bound")
    except KeyError:
        pass
    # finite-mass estimate
    if args.btable and os.path.exists(args.btable):
        bt = read_btable(args.btable)
        base = [r for r in result['runs'] if r['hw_scale'] == 1.0]
        eta_mid = base[-1]['midpoint'] if base else None
        bq = {}
        for M in (2, 3, 4, 6):
            try:
                bv, _ = vertex_quadrature(lambda q: bt[reduced(q)]['b'], M)
                ev, _ = vertex_quadrature(eta_at, M)
                bq[M] = dict(Q_b=bv, Q_eta=ev)
            except KeyError:
                pass
        if bq:
            ps = [p for p in bt if p in allp]
            bb = np.array([bt[p]['b'] for p in ps])
            ee = np.array([allp[p]['eta'] for p in ps])
            A = np.column_stack([np.ones_like(ee), ee])
            coef, *_ = np.linalg.lstsq(A, bb, rcond=None)
            kappa = float(coef[1])
            for M, d in bq.items():
                d['b_cv'] = d['Q_b'] - kappa * (d['Q_eta'] - eta_mid) if eta_mid is not None else None
                print(f"b_sphere vertex rule mesh {M}: Q[b] = {d['Q_b']:.5f}; control variate (kappa {kappa:.3f}): "
                      f"{d['b_cv'] if d['b_cv'] is None else round(d['b_cv'], 5)}")
            result['b_sphere'] = dict(meshes=bq, kappa=kappa, n_directions=len(ps),
                                      status='estimate by angular quadrature with mesh differences as indicator; not a bound')
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    # hand-off: the input table with a provenance header
    with open(args.handoff, 'w') as f:
        f.write("# hand-off from the busi lane to the hreduce lane: directional planar coefficients with measured half-widths;\n")
        f.write(f"# source table {os.path.abspath(args.table)}; bounds in {os.path.abspath(args.out)} (eta_bounds.py of this directory).\n")
        f.write("# columns: p1 p2 p3 (primitive normal), pnorm = |p|, mass = ma, eta, eta_hw (half-width), method in the source header.\n")
        f.write('\t'.join(['p1', 'p2', 'p3', 'pnorm', 'mass', 'eta', 'eta_hw']) + '\n')
        for rw in sorted(rows, key=lambda r: (r['pnorm'], r['p'])):
            f.write('\t'.join([str(rw['p'][0]), str(rw['p'][1]), str(rw['p'][2]), f"{rw['pnorm']:.12f}", f"{args.mass:g}",
                               f"{rw['eta']:.12f}", f"{rw['hw']:.3e}"]) + '\n')
    print(f"wrote {args.out} and {args.handoff}")


if __name__ == '__main__':
    main()
