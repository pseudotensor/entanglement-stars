#!/usr/bin/env python3
"""H1.11/C8: outward support geometry, conditional on supplied intervals.

This consumes a busi TSV; it never computes directional fermion entropies.
Floating LPs propose witnesses only. All accepted dual equalities and signs
are checked with exact fractions; integration uses Arb spherical vector areas.
The original table's measured-error premise is retained in the result.
The original pair's C11d refinement retains checked coarse affine bounds,
uses their cap and current upper-dual normals as extra proposals, and scores
every enumerated feasible recovery with its exact objective.
Run only on the authorized box in qi2, nice10, with private python-flint.
"""
import argparse
import csv
from decimal import Decimal
from fractions import Fraction as Q
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import time
import warnings

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.spatial import cKDTree
from flint import arb, ctx
from certify_eta111 import ball_record, exact_fraction


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def determinant(rows):
    return dot(rows[0], cross(rows[1], rows[2]))


def ball(q):
    q = Q(q)
    return arb(q.numerator) / arb(q.denominator)


def solve_columns(columns, target):
    """Solve a consistent 3-by-k system, k<=3, exactly; reject free columns."""
    n = len(columns)
    if not 1 <= n <= 3:
        return None
    a = [[Q(columns[j][i]) for j in range(n)] + [Q(target[i])]
         for i in range(3)]
    pivots = []
    row = 0
    for col in range(n):
        pivot = next((i for i in range(row, 3) if a[i][col]), None)
        if pivot is None:
            return None
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][col]
        a[row] = [x / scale for x in a[row]]
        for i in range(3):
            if i != row:
                scale = a[i][col]
                a[i] = [a[i][j] - scale * a[row][j] for j in range(n+1)]
        pivots.append(row)
        row += 1
    if any(not any(a[i][:n]) and a[i][n] for i in range(3)):
        return None
    answer = [a[pivots[j]][n] for j in range(n)]
    if any(sum(Q(columns[j][i])*answer[j] for j in range(n)) != target[i]
           for i in range(3)):
        return None
    return answer


def interpolate(rays, values):
    columns = [tuple(v[j] for v in rays) for j in range(3)]
    answer = solve_columns(columns, values)
    if answer is None:
        raise ArithmeticError("singular exact interpolation system")
    return answer


def last_place_half(text):
    exponent = Decimal(text).as_tuple().exponent
    return Q(10) ** exponent / 2


def cubic_images(p):
    return sorted({tuple(sign[i]*q[i] for i in range(3))
                   for q in itertools.permutations(p)
                   for sign in itertools.product((-1, 1), repeat=3)})


def read_table(path, mass):
    raw = Path(path).read_text()
    lines = [x for x in raw.splitlines() if x.strip() and not x.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    normals, upper, lower, converted = [], [], [], []
    seen = set()
    for row in rows:
        if Q(row["mass"]) != mass:
            continue
        p = tuple(int(row[f"p{i}"]) for i in (1, 2, 3))
        if not any(p):
            raise ValueError("zero normal")
        orbit = tuple(sorted(map(abs, p), reverse=True))
        if orbit in seen:
            raise ValueError("duplicate cubic orbit at selected mass")
        seen.add(orbit)
        centre, width = Q(row["eta"]), Q(row["eta_hw"])
        if width < 0:
            raise ValueError("negative half-width")
        formatting = last_place_half(row["eta"]) + last_place_half(row["eta_hw"])
        lo, hi = centre-width-formatting, centre+width+formatting
        if lo <= 0:
            raise ValueError("this consumer requires positive support intervals")
        norm = arb(dot(p, p)).sqrt()
        ll = exact_fraction((ball(lo)*norm).lower())
        uu = exact_fraction((ball(hi)*norm).upper())
        converted.append({"original_row": row, "formatting_allowance": str(formatting),
                          "homogeneous_lower": str(ll), "homogeneous_upper": str(uu)})
        for q in cubic_images(p):
            normals.append(q)
            lower.append(ll)
            upper.append(uu)
    if not converted:
        raise ValueError("no input rows at requested mass")
    return normals, upper, lower, {
        "source": str(path), "mass": str(mass), "raw_table": raw,
        "scope": "geometry conditional on the supplied measured directional errors",
        "anchor_checks": "separate required original-table checks; not performed here",
        "converted_orbits": converted,
        "rounded_pnorm_column_used": False,
    }


def cube_input():
    normals, upper, lower = [], [], []
    for p, support in (((1, 0, 0), Q(1)), ((1, 1, 1), Q(3))):
        for q in cubic_images(p):
            normals.append(q)
            upper.append(support)
            lower.append(support)
    return normals, upper, lower, {"scope": "exact synthetic side2 cube; no native eta data"}


def fundamental_mesh(m):
    if m < 1:
        raise ValueError("mesh must be positive")
    rays, ids = [], {}
    for j in range(m+1):
        for k in range(j+1):
            ids[j, k] = len(rays)
            rays.append((m, j, k))
    triangles = []
    for j in range(m):
        for k in range(j+1):
            triangles.append((ids[j, k], ids[j+1, k], ids[j+1, k+1]))
            if k < j:
                triangles.append((ids[j, k], ids[j+1, k+1], ids[j, k+1]))
    if len(triangles) != m*m:
        raise ArithmeticError("wrong wedge triangle count")
    edges = {}
    for t in triangles:
        v = [rays[i] for i in t]
        if determinant(v) != m:
            raise ArithmeticError("wrong mesh orientation or planar area")
        for a, b in zip(t, t[1:]+t[:1]):
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append((a, b))
    boundary = []
    for entries in edges.values():
        if len(entries) == 2:
            if entries[0] != entries[1][::-1]:
                raise ArithmeticError("internal edge orientation mismatch")
        elif len(entries) == 1:
            a, b = (rays[i] for i in entries[0])
            if not ((a[2] == b[2] == 0) or (a[1] == a[2] and b[1] == b[2])
                    or (a[1] == b[1] == m)):
                raise ArithmeticError("nonboundary edge left uncancelled")
            boundary.append(entries[0])
        else:
            raise ArithmeticError("nonmanifold mesh edge")
    if len(boundary) != 3*m:
        raise ArithmeticError("wrong wedge boundary")
    return rays, triangles, {"triangles": len(triangles), "expected_triangles": m*m,
                             "boundary_edges": len(boundary),
                             "exact_orientation_and_planar_coverage": True}


def vector_area(rays):
    result = [arb(0), arb(0), arb(0)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cr = cross(rays[i], rays[j])
        sn = arb(dot(cr, cr)).sqrt()
        if not sn > 0:
            raise ArithmeticError("degenerate spherical edge")
        angle = arb.atan2(sn, arb(dot(rays[i], rays[j])))
        for k in range(3):
            result[k] += angle*cr[k]/(2*sn)
    if not dot(result, [sum(v[k] for v in rays) for k in range(3)]) > 0:
        raise ArithmeticError("spherical vector area orientation failed")
    return result


def solid_angle(rays):
    norms = [arb(dot(v, v)).sqrt() for v in rays]
    den = norms[0]*norms[1]*norms[2]
    den += dot(rays[0], rays[1])*norms[2]
    den += dot(rays[1], rays[2])*norms[0]
    den += dot(rays[2], rays[0])*norms[1]
    det = determinant(rays)
    if det <= 0 or not den > 0:
        raise ArithmeticError("solid angle branch is not certified")
    return 2*arb.atan2(arb(det), den)


class Geometry:
    def __init__(self, normals, upper, lower, deadline):
        self.a, self.u, self.l = normals, upper, lower
        self.af = np.asarray(normals, dtype=float)
        self.uf = np.asarray([float(x) for x in upper])
        self.unit = self.af/np.linalg.norm(self.af, axis=1)[:, None]
        self.tree = cKDTree(self.unit)
        self.deadline = deadline
        self.lp_calls = self.recovered = self.rejected = self.upper_fallbacks = 0
        self.feasible_supports = 0
        self.lower_cache = {}
        self.indices = {p: i for i, p in enumerate(normals)}
        # Serial LPs reuse their fixed rows; do not rebuild thousands of
        # exact tuples and dense arrays for every vertex/cap combination.
        self.lower_rows = [tuple(-x for x in a) for a in normals] + [None]
        self.lower_objective = [-x for x in upper] + [None]
        self.lower_af = np.empty((len(normals)+1, 3), dtype=float)
        self.lower_af[:-1] = self.af
        self.lower_uf = np.empty(len(normals)+1, dtype=float)
        self.lower_uf[:-1] = self.uf

    def proposal(self, c, aa, bb):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("geometry wall cap reached")
        self.lp_calls += 1
        # scipy forwards this native HiGHS option; every accepted numerical
        # proposal is still checked independently with exact arithmetic.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning,
                                    message="Unrecognized options detected.*")
            return linprog(c, A_ub=aa, b_ub=bb, bounds=[(None, None)]*3,
                           method="highs", options={"threads": 1,
                                                     "time_limit": min(30., remaining)})

    def recover(self, proposal, rows, target, objective, maximize):
        if not proposal.success:
            self.rejected += 1
            return None
        weights = -np.asarray(proposal.ineqlin.marginals)
        order = np.argsort(-weights)
        candidates = [int(i) for i in order[:12] if weights[i] > 0]
        best_terms, best_value = None, None
        for k in range(1, min(3, len(candidates))+1):
            for ids in itertools.combinations(candidates, k):
                coeff = solve_columns([rows[i] for i in ids], target)
                if coeff is not None and all(x >= 0 for x in coeff):
                    terms = [(i, x) for i, x in zip(ids, coeff) if x]
                    if all(sum(rows[i][j]*x for i, x in terms) == target[j]
                           for j in range(3)):
                        value = sum(objective[i]*x for i, x in terms)
                        self.feasible_supports += 1
                        if (best_value is None or
                                (value > best_value if maximize else value < best_value)):
                            best_terms, best_value = terms, value
        if best_terms is not None:
            self.recovered += 1
            return best_terms
        self.rejected += 1
        return None

    def upper(self, v):
        proposal = self.proposal(-np.asarray(v, dtype=float), self.af, self.uf)
        terms = self.recover(proposal, self.a, v, self.u, False)
        if terms is None:
            # All actual accepted tables must contain the100 anchor. This
            # fallback is a verified box bound, never an optimizer estimate.
            terms = []
            for j in range(3):
                if v[j]:
                    e = tuple((1 if v[j] > 0 else -1) if i == j else 0
                              for i in range(3))
                    terms.append((self.indices[e], Q(abs(v[j]))))
            if any(sum(self.a[i][j]*x for i, x in terms) != v[j] for j in range(3)):
                raise ArithmeticError("upper fallback equality failed")
            self.upper_fallbacks += 1
        return sum(self.u[i]*x for i, x in terms), terms

    def lower(self, vertex_id, v, cap):
        key = (vertex_id, cap)
        if key in self.lower_cache:
            return self.lower_cache[key]
        p = self.a[cap]
        self.lower_af[-1] = -np.asarray(p, dtype=float)
        self.lower_uf[-1] = -float(self.l[cap])
        proposal = self.proposal(np.asarray(v, dtype=float), self.lower_af, self.lower_uf)
        # For minimization v = -A_aug^T lambda, lambda>=0.
        self.lower_rows[-1] = p
        self.lower_objective[-1] = self.l[cap]
        terms = self.recover(proposal, self.lower_rows, v, self.lower_objective, True)
        if terms is None:
            result = None
        else:
            value = sum((self.l[cap] if i == len(self.a) else -self.u[i])*x
                        for i, x in terms)
            result = value, terms
        self.lower_cache[key] = result
        return result


def witness(terms):
    return [[i, str(x)] for i, x in terms]


def checked_objective(terms, rows, target, objective):
    parsed = [(int(i), Q(x)) for i, x in terms]
    if any(i < 0 or i >= len(rows) or x < 0 for i, x in parsed):
        raise ArithmeticError("invalid inherited dual index or sign")
    if any(sum(rows[i][j]*x for i, x in parsed) != target[j] for j in range(3)):
        raise ArithmeticError("inherited exact dual equality failed")
    return sum(objective[i]*x for i, x in parsed)


class CoarseEnvelope:
    """Reconstruct existing exact affine certificates, not optimizer guesses."""
    def __init__(self, enclosure, normals, upper, lower, source):
        if ([tuple(p) for p in enclosure["integer_normals"]] != normals or
                list(map(Q, enclosure["homogeneous_upper"])) != upper or
                list(map(Q, enclosure["homogeneous_lower"])) != lower):
            raise ArithmeticError("coarse envelope has different actual input bounds")
        self.mesh = enclosure["mesh"]
        self.rays, self.triangles, _ = fundamental_mesh(self.mesh)
        if (len(enclosure["upper_vertex_duals"]) != len(self.rays) or
                len(enclosure["selected_triangle_cap_duals"]) != len(self.triangles)):
            raise ArithmeticError("incomplete coarse witness arrays")
        hi = [checked_objective(t, normals, v, upper)
              for t, v in zip(enclosure["upper_vertex_duals"], self.rays)]
        self.upper_affine, self.lower_affine, self.caps = [], [], []
        self.lookup = {}
        negative_rows = [tuple(-x for x in a) for a in normals]
        negative_upper = [-x for x in upper]
        for ti, ids in enumerate(self.triangles):
            vv = [self.rays[i] for i in ids]
            self.lookup[tuple(sorted(vv))] = ti
            self.upper_affine.append(interpolate(vv, [hi[i] for i in ids]))
            record = enclosure["selected_triangle_cap_duals"][ti]
            if record is None:
                # Original coarse outputs use None only for the zero baseline.
                # A repaired output with inherited selections must not silently
                # be interpreted as that baseline.
                if ("triangle_bound_records" in enclosure and
                        enclosure["triangle_bound_records"][ti]["lower_source"] != "zero"):
                    raise ValueError("use an original fresh-witness coarse envelope")
                self.lower_affine.append([Q(0)]*3)
                self.caps.append(None)
                continue
            cap = int(record["cap"])
            if not 0 <= cap < len(normals) or len(record["vertex_duals"]) != 3:
                raise ArithmeticError("invalid inherited cap record")
            rows = negative_rows + [normals[cap]]
            objective = negative_upper + [lower[cap]]
            values = [checked_objective(t, rows, v, objective)
                      for t, v in zip(record["vertex_duals"], vv)]
            self.lower_affine.append(interpolate(vv, values))
            self.caps.append(cap)
        self.scalar_lower = Q(enclosure["lower"]["lower"])
        self.scalar_upper = Q(enclosure["upper"]["upper"])
        self.report = {"source": str(source), "mesh": self.mesh,
                       "upper_vertices_exactly_checked": len(self.rays),
                       "triangle_witnesses_exactly_checked": len(self.triangles)}

    def parent(self, vv, fine_mesh):
        if fine_mesh <= self.mesh or fine_mesh % self.mesh:
            raise ValueError("coarse mesh must strictly divide the fine mesh")
        centre = [sum(v[j] for v in vv) for j in range(3)]
        u, v = (Q(self.mesh*centre[j], centre[0]) for j in (1, 2))
        j, k = u.numerator//u.denominator, v.numerator//v.denominator
        if u-j >= v-k:
            key = ((self.mesh, j, k), (self.mesh, j+1, k), (self.mesh, j+1, k+1))
        else:
            key = ((self.mesh, j, k), (self.mesh, j+1, k+1), (self.mesh, j, k+1))
        ti = self.lookup[tuple(sorted(key))]
        columns = [self.rays[i] for i in self.triangles[ti]]
        # The index is only a proposal: certify all three cone inclusions.
        for vertex in vv:
            coefficients = solve_columns(columns, vertex)
            if coefficients is None or any(x < 0 for x in coefficients):
                raise ArithmeticError("fine triangle not exactly inside coarse parent")
        return ti


def intersect_scalar(enclosure, previous):
    """Intersect already valid bounds on the same conditional quantity."""
    if "before_scalar_intersection" not in enclosure:
        enclosure["before_scalar_intersection"] = {
            k: enclosure[k] for k in ("lower", "upper", "width")}
    ll = max([Q(enclosure["lower"]["lower"])]+[x[1] for x in previous])
    uu = min([Q(enclosure["upper"]["upper"])]+[x[2] for x in previous])
    if ll > uu:
        raise ArithmeticError("completed conditional envelopes have empty intersection")
    lo, hi = ball(ll).lower(), ball(uu).upper()
    enclosure.update(lower=ball_record(lo), upper=ball_record(hi), width=ball_record(hi-lo))
    enclosure.setdefault("scalar_intersections", []).extend(
        {"source": str(s), "lower": str(l), "upper": str(u)} for s, l, u in previous)


def enclose(normals, upper, lower, mesh, caps, deadline, checkpoint, coarse=None):
    geom = Geometry(normals, upper, lower, deadline)
    rays, triangles, coverage = fundamental_mesh(mesh)
    hi_values, hi_witnesses = [], []
    for v in rays:
        value, terms = geom.upper(v)
        hi_values.append(value)
        hi_witnesses.append(witness(terms))
    sum_lo, sum_hi, omega = arb(0), arb(0), arb(0)
    sum_vec = [arb(0), arb(0), arb(0)]
    accepted, bound_records = [], []
    inherited_lower_count = inherited_upper_count = max_candidates = 0
    for ti, ids in enumerate(triangles):
        if time.monotonic() >= deadline:
            raise TimeoutError("geometry wall cap reached")
        vv = [rays[i] for i in ids]
        area = vector_area(vv)
        omega += solid_angle(vv)
        sum_vec = [x+y for x, y in zip(sum_vec, area)]
        aa = interpolate(vv, [hi_values[i] for i in ids])
        triangle_hi = dot([ball(x) for x in aa], area)
        fresh_hi = ball_record(triangle_hi)
        upper_source = "fresh_vertex_duals"
        parent = coarse.parent(vv, mesh) if coarse is not None else None
        if parent is not None:
            old_hi = dot([ball(x) for x in coarse.upper_affine[parent]], area)
            if old_hi.upper() < triangle_hi.upper():
                triangle_hi = old_hi
                upper_source = "inherited_coarse_affine"
                inherited_upper_count += 1
        centre = np.asarray([sum(v[j] for v in vv) for j in range(3)], dtype=float)
        centre /= np.linalg.norm(centre)
        _, candidates = geom.tree.query(centre, k=min(caps, len(normals)))
        candidates = set(map(int, np.atleast_1d(candidates)))
        if parent is not None:
            if coarse.caps[parent] is not None:
                candidates.add(coarse.caps[parent])
            for vertex in ids:
                candidates.update(int(i) for i, _ in hi_witnesses[vertex])
        max_candidates = max(max_candidates, len(candidates))
        best = arb(0)
        best_record = None
        lower_source = "zero"
        if parent is not None:
            old_lo = dot([ball(x) for x in coarse.lower_affine[parent]], area)
            if old_lo.lower() > best.lower():
                best = old_lo
                lower_source = "inherited_coarse_affine"
        fresh_best = arb(0)
        for cap in sorted(candidates):
            cap = int(cap)
            vals = [geom.lower(i, rays[i], cap) for i in ids]
            if any(x is None for x in vals):
                continue
            bb = interpolate(vv, [x[0] for x in vals])
            candidate = dot([ball(x) for x in bb], area)
            if candidate.lower() > fresh_best.lower():
                fresh_best = candidate
            if candidate.lower() > best.lower():
                best = candidate
                best_record = {"cap": cap, "vertex_duals": [witness(x[1]) for x in vals]}
                lower_source = "fresh_cap_duals"
        if lower_source == "inherited_coarse_affine":
            inherited_lower_count += 1
        sum_lo += best.lower()
        sum_hi += triangle_hi.upper()
        accepted.append(best_record)
        bound_records.append({"parent_triangle": parent, "candidate_caps": sorted(candidates),
                              "lower_source": lower_source, "upper_source": upper_source,
                              "fresh_lower_integral": ball_record(fresh_best),
                              "fresh_upper_integral": fresh_hi,
                              "selected_lower_integral": ball_record(best),
                              "selected_upper_integral": ball_record(triangle_hi)})
        if (ti+1) % 8 == 0 or ti+1 == len(triangles):
            checkpoint({"mesh": mesh, "completed_triangles": ti+1, "expected_triangles": len(triangles),
                        "covered_solid_angle": ball_record(omega), "lp_calls": geom.lp_calls})
    difference = omega-arb.pi()/12
    expected_vec = vector_area(((1, 0, 0), (1, 1, 0), (1, 1, 1)))
    if not difference.contains(0) or any(not (x-y).contains(0) for x, y in zip(sum_vec, expected_vec)):
        raise ArithmeticError("spherical wedge coverage check failed")
    factor = 12/arb.pi()
    lo = (factor*sum_lo).lower()
    hi = (factor*sum_hi).upper()
    if not lo <= hi:
        raise ArithmeticError("inverted enclosure")
    result = {"lower": ball_record(lo), "upper": ball_record(hi),
            "width": ball_record(hi-lo), "mesh": mesh, "caps_per_triangle": caps,
            "coverage": {**coverage, "solid_angle": ball_record(omega),
                         "solid_angle_error": ball_record(difference),
                         "vector_area": [ball_record(x) for x in sum_vec]},
            "integer_normals": normals, "homogeneous_upper": list(map(str, upper)),
            "homogeneous_lower": list(map(str, lower)),
            "upper_vertex_duals": hi_witnesses, "selected_triangle_cap_duals": accepted,
            "triangle_bound_records": bound_records,
            "lp_calls": geom.lp_calls, "exact_duals_recovered": geom.recovered,
            "exact_feasible_supports_compared": geom.feasible_supports,
            "best_exact_recovered_objective_used": True,
            "maximum_candidate_caps_per_triangle": max_candidates,
            "inherited_lower_triangles_selected": inherited_lower_count,
            "inherited_upper_triangles_selected": inherited_upper_count,
            "rejected_proposals": geom.rejected, "upper_box_fallbacks": geom.upper_fallbacks,
            "every_retained_dual_equality_and_sign_checked_exactly": True}
    if coarse is not None:
        result["coarse_affine_input"] = coarse.report
        intersect_scalar(result, [(coarse.report["source"], coarse.scalar_lower, coarse.scalar_upper)])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--control", action="store_true")
    mode.add_argument("--table", type=Path)
    ap.add_argument("--mass", default="0")
    ap.add_argument("--mesh", type=int, default=4,
                    help="table mesh; control checks this and twice this mesh")
    ap.add_argument("--caps", type=int, default=4)
    ap.add_argument("--bits", type=int, default=160)
    ap.add_argument("--coarse-result", type=Path,
                    help="original completed coarse output with fresh exact cap witnesses")
    ap.add_argument("--intersect-result", type=Path, action="append", default=[],
                    help="additional completed output on exactly the same input intervals")
    ap.add_argument("--wall-seconds", type=int, default=600)
    ap.add_argument("--memory-mib", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.caps < 1 or args.bits < 128:
        ap.error("need caps>=1 and bits>=128")
    if args.control and (args.coarse_result or args.intersect_result):
        ap.error("synthetic mode uses its own same-run coarse envelope")
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))
    available = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(available)})
    cap = args.memory_mib*1024**2
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    resource.setrlimit(resource.RLIMIT_CPU, (args.wall_seconds, args.wall_seconds+1))
    def timed_out(signum, frame):
        raise TimeoutError("geometry wall cap reached")
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.wall_seconds)
    ctx.prec, ctx.threads = args.bits, 1
    started = time.monotonic()
    result = {"status": "incomplete", "purpose": "H1.11/C8 enclosure arithmetic",
              "pid": os.getpid(), "bits": args.bits, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "wall_cap_seconds": args.wall_seconds, "memory_cap_bytes": cap,
              "directional_entropies_computed": False, "certification_scope": "pending"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(update=None):
        if update:
            result["progress"] = update
        result["elapsed_seconds"] = time.monotonic()-started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["cpu_seconds"] = usage.ru_utime+usage.ru_stime
        result["peak_rss_KiB"] = usage.ru_maxrss
        temporary = args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(result, indent=2)+"\n")
        temporary.replace(args.output)
    save()
    try:
        if args.control:
            normals, upper, lower, source = cube_input()
        else:
            normals, upper, lower, source = read_table(args.table, Q(args.mass))
        result["input"] = source
        coarse = None
        if args.coarse_result:
            old = json.loads(args.coarse_result.read_text())
            if (old["status"] != "conditional_enclosure" or
                    old["input"]["raw_table"] != source["raw_table"]):
                raise ValueError("coarse result is incomplete or uses a different actual table")
            coarse = CoarseEnvelope(old["enclosure"], normals, upper, lower, args.coarse_result)
        intersections = []
        for path in args.intersect_result:
            old = json.loads(path.read_text())
            e = old["enclosure"]
            if (old["status"] != "conditional_enclosure" or
                    old["input"]["raw_table"] != source["raw_table"] or
                    [tuple(p) for p in e["integer_normals"]] != normals or
                    list(map(Q, e["homogeneous_upper"])) != upper or
                    list(map(Q, e["homogeneous_lower"])) != lower):
                raise ValueError("intersection result has different or incomplete actual inputs")
            intersections.append((str(path), Q(e["lower"]["lower"]), Q(e["upper"]["upper"])))
        result["enclosure"] = enclose(normals, upper, lower, args.mesh, args.caps,
                                      started+args.wall_seconds, save, coarse=coarse)
        if intersections:
            intersect_scalar(result["enclosure"], intersections)
        if args.control:
            synthetic_coarse = CoarseEnvelope(result["enclosure"], normals, upper, lower,
                                               "same-run synthetic coarse enclosure")
            result["refined_control_enclosure"] = enclose(
                normals, upper, lower, 2*args.mesh, args.caps,
                started+args.wall_seconds, save, coarse=synthetic_coarse)
            octant = vector_area(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
            errors = [x-arb.pi()/4 for x in octant]
            for key in ("enclosure", "refined_control_enclosure"):
                lo = Q(result[key]["lower"]["lower"])
                hi = Q(result[key]["upper"]["upper"])
                if not (lo <= Q(3, 2) <= hi and hi-lo < Q(2)**-80):
                    raise ArithmeticError("cube mean3/2 or precision check failed")
            if any(not x.contains(0) for x in errors):
                raise ArithmeticError("octant vector area check failed")
            if last_place_half("0.123456789012") != Q(1, 2*10**12):
                raise ArithmeticError("TSV centre last-place check failed")
            if last_place_half("1.234e-05") != Q(1, 2*10**8):
                raise ArithmeticError("TSV half-width last-place check failed")
            result["control_checks"] = {"cube_side": 2, "expected_cube_mean": "3/2",
                                         "octant_vector_area": [ball_record(x) for x in octant],
                                         "octant_errors": [ball_record(x) for x in errors],
                                         "TSV_last_place_checks": True,
                                         "native_anchor_check": "not this synthetic arithmetic control"}
            result["certification_scope"] = "passed synthetic geometry arithmetic only"
            result["status"] = "control_passed"
        else:
            result["certification_scope"] = "outward geometry conditional on measured input intervals"
            result["status"] = "conditional_enclosure"
        signal.alarm(0)
        save()
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("input", "enclosure", "refined_control_enclosure", "control_checks")}, indent=2))


if __name__ == "__main__":
    main()
