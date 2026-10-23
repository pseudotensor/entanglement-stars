#!/usr/bin/env python3
"""Bounded original-H_* open3x2x2 n3 ENGINEERING DIAGNOSTIC ONLY.

Author specification: solve/strand_A_cold_three_body.tex, TB1--TB19.
Independent specification review: solve/review_A_cold_three_body.tex.
Only the explicitly approved floating diagnostic fallback is implemented:
no proved arithmetic, assembly or spectral radii are available here.
Assembly, internal-ground-gap and CS-source certificate flags stay FALSE.

Allocation contract (complex128, one live rebuild, no triple eigensolve):
 one stored S3: 2024*2024*16 = 65,545,216 bytes;
 at most 32 triple-column blocks: 32*2024*56*16 = 58,032,128 bytes;
 at most 12 annihilation/output blocks: 12*24*276*56*16 = 71,221,248 bytes;
 at most 128 pair matrices: 128*276*276*16 = 156,008,448 bytes;
 Kramers tensor/workspace: 6*24**3*56*16 = 74,317,824 bytes;
 small roots/Ritz/source/archive arrays: reserved 64 MiB;
 library temporaries and serialization buffers: reserved 512 MiB.
 Their conservative sum is below 1 GiB and the hard working-array cap
 is 4 GiB. Fixed Python/BLAS/CUDA library overhead is not this array sum.
 S3 is released before another rebuild; dS3 and all triple exponentials,
 H3 and clocks are actions only. Pair middle always has all276 rows.

No model math at import. Ops alone admits the one remote process, with
all six worker/BLAS pools1 and max4 threads; no midrun foreign-PID abort.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import time

from remote_runtime import remote_execution


NP = None
SHAPE = (3, 2, 2)
MODE_COUNT, PAIR_COUNT, TRIPLE_COUNT, TRIAL_COUNT = 24, 276, 2024, 56
P_EXP, P_GRAM = 8, 12
STEPS = (Fraction(1, 2**8), Fraction(1, 2**9), Fraction(1, 2**10))
IDENTITY_TARGET = 1e-10
WORKING_ARRAY_CAP = 4 * 1024**3
ALLOCATION_BYTES = (65545216 + 58032128 + 71221248 + 156008448
                    + 74317824 + 64 * 1024**2 + 512 * 1024**2)
EXACT_INPUT = {"shape": [3, 2, 2], "a": "1", "m": "6741313/100000000",
               "gM": "1/5", "gB": "1", "Lambda_seed": "1/2", "ell": "2",
               "N_base": "1", "psi_base": "1", "boundary": "open zero extension",
               "sector": "complete hole-empty n=3; empty invariant spectators",
               "site_mode_order": "lexicographic (x1,x2,x3,spin)",
               "CAR_basis_order": "ascending lexicographic pairs and triples",
               "law": "exp(S3)*sum(Ci^dagger*Bik*exp(-2*S2)*Ck)*exp(S3)",
               "dictionary": "canonical Gaussian/Walsh; no extra taste factor or old-bond rescaling"}
MISSING_RADII = ["directed scalar roots/exponentials and fixed-input rounding",
                 "polar<internal>/seed assembly and Sylvester-solve arithmetic",
                 "CAR-lift and polynomial matrix/action arithmetic",
                 "Gram normalization and inverse-root arithmetic",
                 "Ritz Hermitian assembly, eigenvalues, eigenvectors and residual arithmetic",
                 "physical residual/complement norm arithmetic",
                 "ruler/native connection and clock-row assembly arithmetic"]


class Unresolved(Exception):
    pass


class Cap(Unresolved):
    pass


@dataclass
class Work:
    seconds: int
    start: float
    rss_mib: int = 5120
    actions: int = 0
    s3_live: bool = False
    archive: dict | None = None
    peak_rss_kib: int = 0
    peak_working_array_majorant_bytes: int = 0

    def check(self):
        self.actions += 1
        if time.perf_counter() - self.start > self.seconds:
            raise Cap("Predeclared one-point wall-time cap reached")
        if len(list(Path("/proc/self/task").iterdir())) > 4:
            raise Cap("Own process exceeds the admitted four-thread cap")
        status = Path("/proc/self/status").read_text()
        rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
        self.peak_rss_kib = max(self.peak_rss_kib, rss)
        if rss > self.rss_mib*1024:
            raise Cap("Predeclared process RSS cap reached")
        # The audited transient-array plan is charged in full at every
        # checkpoint. Also charge every retained archive array separately;
        # this deliberately double-counts arrays shared with the plan.
        owned = {} if self.archive is None else {id(value): value for value in self.archive.values() if hasattr(value, "nbytes")}
        majorant = ALLOCATION_BYTES + sum(value.nbytes for value in owned.values())
        self.peak_working_array_majorant_bytes = max(self.peak_working_array_majorant_bytes, majorant)
        if majorant > WORKING_ARRAY_CAP:
            raise Cap("Four-GiB working-array contract reached")

    def claim_s3(self):
        self.check()
        if self.s3_live:
            raise Cap("Allocation contract forbids two live triple dressing matrices")
        self.s3_live = True


def adj(matrix):
    return matrix.conj().T


def herm(matrix):
    return (matrix + adj(matrix)) / 2


def fro(matrix):
    return float(NP.sqrt(NP.vdot(matrix, matrix).real))


def norm2(matrix):
    # Only native24, pair276, Ritz56 or a rectangular2024x56 block.
    if matrix.shape == (TRIPLE_COUNT, TRIPLE_COUNT):
        raise RuntimeError("Dense triple spectral operations are outside this scope")
    return float(NP.linalg.svd(matrix, compute_uv=False)[0])


def comparison(left, right):
    absolute = fro(left - right)
    scale = max(1.0, fro(left), fro(right))
    return {"absolute_Frobenius_difference": absolute, "comparison_scale": scale,
            "scaled_difference": absolute / scale,
            "observed_below_engineering_target": absolute / scale <= IDENTITY_TARGET,
            "certified_arithmetic_radius": None}


def positive_root(matrix, power=0.5):
    values, vectors = NP.linalg.eigh(herm(matrix))
    if not NP.all(NP.isfinite(values)) or not float(values[0]) > 0:
        raise Unresolved("Ordinary positive-root control failed; no certified interpretation")
    return (vectors * values**power) @ adj(vectors)


def sylvester_positive(root, rhs):
    values, vectors = NP.linalg.eigh(herm(root))
    if not float(values[0]) > 0:
        raise Unresolved("Ordinary positive Sylvester denominator failed")
    coordinates = adj(vectors) @ rhs @ vectors
    return vectors @ (coordinates / (values[:, None] + values[None, :])) @ adj(vectors)


def exp_frechet_small(matrix, derivative=None):
    """Native24 Hermitian exponential and its noncommuting Duhamel jet."""
    if matrix.shape != (MODE_COUNT, MODE_COUNT):
        raise RuntimeError("Only the native24 regulator uses this spectral exponential")
    values, vectors = NP.linalg.eigh(herm(matrix))
    exponentials = NP.exp(values)
    result = (vectors * exponentials) @ adj(vectors)
    if derivative is None:
        return result, None
    difference = values[:, None] - values[None, :]
    # exp(xb)*expm1(xa-xb)/(xa-xb), with the exact removable limit.
    divided = NP.empty_like(difference)
    nonzero = difference != 0
    divided[nonzero] = NP.expm1(difference[nonzero]) / difference[nonzero]
    divided[~nonzero] = 1.0
    divided *= exponentials[None, :]
    tangent = vectors @ (divided * (adj(vectors) @ derivative @ vectors)) @ adj(vectors)
    return result, tangent


def polynomial_action(action, block, degree=P_EXP, tangent=None):
    """Explicit Taylor words; tangent never assumes commuting matrices."""
    result, term = block.copy(), block.copy()
    if tangent is None:
        for k in range(1, degree + 1):
            term = action(term) / k
            result += term
        return result
    derivative, dterm = NP.zeros_like(block), NP.zeros_like(block)
    for k in range(1, degree + 1):
        dterm = (tangent(term) + action(dterm)) / k
        term = action(term) / k
        derivative += dterm
        result += term
    return result, derivative


def matrix_polynomial(matrix, degree=P_EXP, derivative=None):
    identity = NP.eye(len(matrix), dtype=complex)
    return polynomial_action(lambda block: matrix @ block, identity, degree,
                             None if derivative is None else lambda block: derivative @ block)


def tail_formula(s, derivative_norm=None):
    result = {"formula": "exp(s)*s^9/9!", "ordinary_evaluation": math.exp(s) * s**9 / math.factorial(9),
              "norm_majorant_ordinary_evaluation": s, "scalar_arithmetic_radius": None}
    if derivative_norm is not None:
        result.update({"derivative_formula": "dX*exp(s)*s^8/8!",
                       "derivative_ordinary_evaluation": derivative_norm * math.exp(s) * s**8 / math.factorial(8)})
    return result


class CAR:
    def __init__(self):
        self.pairs = list(itertools.combinations(range(MODE_COUNT), 2))
        self.triples = list(itertools.combinations(range(MODE_COUNT), 3))
        self.pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        self.triple_index = {triple: i for i, triple in enumerate(self.triples)}
        self.maps = []
        for mode in range(MODE_COUNT):
            columns, rows, signs = [], [], []
            for column, triple in enumerate(self.triples):
                if mode in triple:
                    position = triple.index(mode)
                    rows.append(self.pair_index[triple[:position] + triple[position + 1:]])
                    columns.append(column)
                    signs.append((-1)**position)
            self.maps.append((NP.array(rows), NP.array(columns), NP.array(signs)))
        self.pair_array = NP.array(self.pairs, dtype=int)

    def annihilate(self, block):
        result = NP.zeros((MODE_COUNT, PAIR_COUNT, block.shape[1]), dtype=complex)
        for mode, (rows, columns, signs) in enumerate(self.maps):
            result[mode, rows] = signs[:, None] * block[columns]
        return result

    def create_sum(self, outputs):
        result = NP.zeros((TRIPLE_COUNT, outputs.shape[2]), dtype=complex)
        for mode, (rows, columns, signs) in enumerate(self.maps):
            result[columns] += signs[:, None] * outputs[mode, rows]
        return result

    def one_body_action(self, one_body, block):
        removed = self.annihilate(block)
        return self.create_sum(NP.einsum("ik,kbc->ibc", one_body, removed, optimize=True))

    def pair_lift_action(self, pair_matrix, block):
        removed = self.annihilate(block)
        return self.create_sum(NP.einsum("ab,ibc->iac", pair_matrix, removed, optimize=True))

    def pair_lift_matrix(self, pair_matrix, work):
        work.claim_s3()
        result = NP.zeros((TRIPLE_COUNT, TRIPLE_COUNT), dtype=complex)
        for rows, columns, signs in self.maps:
            work.check()
            result[NP.ix_(columns, columns)] += signs[:, None] * pair_matrix[NP.ix_(rows, rows)] * signs[None, :]
        return result

    def pair_one_body(self, one_body):
        a, b = self.pair_array.T
        c, d = a, b
        return (one_body[a[:, None], c[None, :]] * (b[:, None] == d[None, :])
                - one_body[a[:, None], d[None, :]] * (b[:, None] == c[None, :])
                - one_body[b[:, None], c[None, :]] * (a[:, None] == d[None, :])
                + one_body[b[:, None], d[None, :]] * (a[:, None] == c[None, :]))

    def q2(self, left, right):
        a, b = self.pair_array.T
        c, d = a, b
        return (left[a[:, None], c[None, :]] * right[b[:, None], d[None, :]]
                + right[a[:, None], c[None, :]] * left[b[:, None], d[None, :]]
                - left[a[:, None], d[None, :]] * right[b[:, None], c[None, :]]
                - right[a[:, None], d[None, :]] * left[b[:, None], c[None, :]])

    def theta3(self, one_body, block):
        # Exterior cube via an antisymmetric tensor, not a stored2024-square.
        tensor = NP.zeros((MODE_COUNT, MODE_COUNT, MODE_COUNT, block.shape[1]), dtype=complex)
        permutations = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        for permutation, sign in zip(permutations, (1, -1, -1, 1, 1, -1)):
            indices = NP.array([[triple[p] for p in permutation] for triple in self.triples])
            tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = sign * block.conj() / math.sqrt(6)
        tensor = NP.einsum("ia,abcq->ibcq", one_body, tensor, optimize=True)
        tensor = NP.einsum("jb,ibcq->ijcq", one_body, tensor, optimize=True)
        tensor = NP.einsum("kc,ijcq->ijkq", one_body, tensor, optimize=True)
        indices = NP.array(self.triples)
        return math.sqrt(6) * tensor[indices[:, 0], indices[:, 1], indices[:, 2]]


class Geometry:
    def __init__(self, car):
        self.sites = list(itertools.product(range(3), range(2), range(2)))
        self.site_index = {site: i for i, site in enumerate(self.sites)}
        sigma = [NP.array([[0, 1], [1, 0]], dtype=complex),
                 NP.array([[0, -1j], [1j, 0]], dtype=complex),
                 NP.array([[1, 0], [0, -1]], dtype=complex)]
        self.dirac = NP.zeros((MODE_COUNT, MODE_COUNT), dtype=complex)
        for axis in range(3):
            shift = NP.zeros((12, 12), dtype=complex)
            for row, site in enumerate(self.sites):
                neighbor = list(site)
                neighbor[axis] += 1
                if neighbor[axis] < SHAPE[axis]:
                    shift[row, self.site_index[tuple(neighbor)]] = 1
            self.dirac += NP.kron((shift - adj(shift)) / (2j), sigma[axis])
        square = self.dirac @ self.dirac
        self.pl, self.ph = 2 * NP.eye(MODE_COUNT) - 2 * square, 2 * square - NP.eye(MODE_COUNT)
        self.beta = NP.kron(sigma[0], NP.eye(2))
        self.currents = [NP.kron(sigma[1], matrix) for matrix in sigma]
        self.theta2 = NP.kron(NP.eye(12), 1j * sigma[1])
        self.filters, self.cells = [], []
        for cell in range(2):
            sites = [self.site_index[(cell + dx, dy, dz)]
                     for dx, dy, dz in itertools.product((0, 1), repeat=3)]
            self.cells.append(sites)
            masks = []
            for nu in itertools.product((0, 1), repeat=3):
                matrix = NP.zeros((4, 2 * MODE_COUNT), dtype=complex)
                for delta, site in zip(itertools.product((0, 1), repeat=3), sites):
                    weight = (-1)**sum(n * d for n, d in zip(nu, delta)) / 8
                    for block in range(2):
                        for spin in range(2):
                            matrix[2 * block + spin, block * MODE_COUNT + 2 * site + spin] = weight
                masks.append(matrix)
            self.filters.append(masks)
        low = NP.zeros((MODE_COUNT, 8), dtype=complex)
        for column, (y, z, spin) in enumerate(itertools.product(range(2), repeat=3)):
            for x in (0, 2):
                low[2 * self.site_index[(x, y, z)] + spin, column] = 1 / math.sqrt(2)
        self.low = low
        low_triples = list(itertools.combinations(range(8), 3))
        self.i0 = NP.zeros((TRIPLE_COUNT, TRIAL_COUNT), dtype=complex)
        a, b, c = NP.array(car.triples).T
        for column, (i, j, k) in enumerate(low_triples):
            self.i0[:, column] = (low[a, i] * (low[b, j] * low[c, k] - low[b, k] * low[c, j])
                                  - low[a, j] * (low[b, i] * low[c, k] - low[b, k] * low[c, i])
                                  + low[a, k] * (low[b, i] * low[c, j] - low[b, j] * low[c, i]))
        self.probes = {"uniform": NP.ones(12),
                       "left_slice": NP.array([int(site[0] == 0) for site in self.sites], dtype=float)}

    def ruler_geometry(self, psi, f):
        squares, derivatives = psi**2, 2 * psi**2 * f
        uc = NP.array([NP.mean(squares[sites]) for sites in self.cells])
        duc = NP.array([NP.mean(derivatives[sites]) for sites in self.cells])
        plane = NP.array([NP.mean([squares[self.site_index[(x, y, z)]]
                                   for y, z in itertools.product(range(2), repeat=2)]) for x in range(3)])
        dplane = NP.array([NP.mean([derivatives[self.site_index[(x, y, z)]]
                                    for y, z in itertools.product(range(2), repeat=2)]) for x in range(3)])
        # Exact multilinear integral along(.5,.5,.5)--(1.5,.5,.5).
        weights = NP.array([1 / 8, 3 / 4, 1 / 8])
        distance, ddistance = float(weights @ plane), float(weights @ dplane)
        gaussian = NP.full((2, 2), math.exp(-distance**2 / 4) / 8)
        dgaussian = -distance * ddistance * gaussian / 2
        NP.fill_diagonal(gaussian, 1 / 8)
        NP.fill_diagonal(dgaussian, 0)
        return uc, duc, gaussian, dgaussian, distance, ddistance


def analytic_constants():
    m = float(Fraction(6741313, 100000000))
    el, eh = math.sqrt(m*m + 0.5), math.sqrt(m*m + 1)
    sigma = 23 * math.exp(-4) / (320 * el)
    a1, u1 = m + 3, (m + 3) / el
    e1, b1 = eh * u1, eh * u1 + eh
    zeta = 8 + u1 / math.sqrt(2)
    v1 = math.exp(-4) * ((1 / 10) * (6 + 4*zeta) + (3 / 16) * (1 + 4*zeta))
    sigma1 = (v1 + 4*b1*sigma) / (4*el)
    eta, delta = 3*eh*math.expm1(8*sigma), eh-el
    a0 = math.expm1(3*sigma)
    complement = 3*el + delta*(1-a0*a0) - eta
    rstar = math.exp(6*sigma) * (3*el*math.expm1(6*sigma) + 3*eh*math.expm1(2*sigma))
    exprel = lambda x: math.expm1(x) / x if x else 1
    kappa_b = b1/(2*el)
    kappa_s = sigma1*exprel(2*sigma) + 3*sigma1*math.sqrt(eh/el)*math.exp(8*sigma)*exprel(6*sigma)
    return {"m": m, "epsilon_l": el, "epsilon_h": eh, "E_l": 3*el, "Delta": delta,
            "sigma": sigma, "sigma1": sigma1, "eta": eta, "a0": a0, "L": complement,
            "r_star": rstar, "a1": a1, "u1": u1, "e1": e1, "b1": b1,
            "zeta": zeta, "v1": v1, "kappa_B": kappa_b, "kappa_S": kappa_s,
            "metric_condition_bound": math.exp(12*sigma),
            "frozen_phi_tangent_bound": 3*sigma1*exprel(6*sigma),
            "light_native_tangent_bound": math.sqrt(30)*u1/2,
            "protected_analytic_rational_margins": {"sigma": "<1/450", "eta": "<11/200", "Delta": ">7/25",
                 "rank56_cluster_gap": ">17/100", "complement_above_trial": ">9/50",
                 "r_star": "<1/20", "kappa_B_plus_kappa_S": "<5", "state_tangent": "<2/5",
                 "light_native_tangent": "<13", "metric_condition": "<75/73"},
            "scalar_formula_values_are_ordinary_not_directed": True}


def offbase_norms(psi, constants):
    pmin, pmax, m = float(NP.min(psi)), float(NP.max(psi)), constants["m"]
    amin = max(m*pmin**2/pmax**3, 1/(math.sqrt(2)*pmax**3)-m/pmin)
    bmin, bmax = pmin*amin, pmax*(m/pmin + 1/pmin**3)
    rnorm = math.exp(-pmax**-4)
    vbar = ((1/10)*pmin**-6 + 3/16)*rnorm**4
    sigma = vbar/(4*bmin)
    return {"psi_min": pmin, "psi_max": pmax, "A_min": amin, "B_min": bmin, "B_max": bmax,
            "R_max": rnorm, "V2_max": vbar, "S2_max": sigma, "S3_max": 3*sigma,
            "formula_source": "reviewed off-base TB norm clarification",
            "scalar_rounding_radius": None}


class Native:
    def __init__(self, geometry, car, psi, constants, work, f=None):
        work.check()
        self.geometry, self.psi = geometry, psi.copy()
        self.f = NP.zeros(12) if f is None else f.copy()
        self.base = bool(NP.all(psi == 1))
        d, m = geometry.dirac, constants["m"]
        ruler = NP.diag(NP.repeat(psi, 2))
        fd = NP.diag(NP.repeat(self.f, 2))
        inverse = NP.diag(NP.repeat(psi**-1, 2))
        inverse3 = NP.diag(NP.repeat(psi**-3, 2))
        self.a = m*inverse + 1j*inverse3@d
        self.da = -m*inverse@fd - 3j*inverse3@fd@d
        self.e = (constants["epsilon_l"]*geometry.pl + constants["epsilon_h"]*geometry.ph
                  if self.base else positive_root(adj(self.a)@self.a))
        self.u = self.a @ NP.linalg.inv(self.e)
        self.de = herm(sylvester_positive(self.e, adj(self.da)@self.a + adj(self.a)@self.da))
        self.k0 = sylvester_positive(self.e, adj(self.u)@self.da - adj(self.da)@self.u)
        self.du = self.u @ self.k0
        self.root_e = positive_root(self.e)
        self.droot_e = herm(sylvester_positive(self.root_e, self.de))
        self.b = herm(self.root_e@ruler@self.root_e)
        self.db = herm(self.droot_e@ruler@self.root_e + self.root_e@ruler@fd@self.root_e
                       + self.root_e@ruler@self.droot_e)
        self.root_b = positive_root(self.b)
        self.droot_b = herm(sylvester_positive(self.root_b, self.db))
        kpsi = inverse@d@inverse
        dkpsi = -inverse@fd@d@inverse - inverse@d@inverse@fd
        generator, dgenerator = -2*kpsi@kpsi, -2*(dkpsi@kpsi+kpsi@dkpsi)
        r_spectral, self.dr = exp_frechet_small(generator, dgenerator)
        self.r = math.exp(-1)*geometry.pl + math.exp(-2)*geometry.ph if self.base else r_spectral
        rblock = NP.zeros((2*MODE_COUNT, 2*MODE_COUNT), dtype=complex)
        drblock = NP.zeros_like(rblock)
        rblock[:MODE_COUNT, :MODE_COUNT] = rblock[MODE_COUNT:, MODE_COUNT:] = self.r
        drblock[:MODE_COUNT, :MODE_COUNT] = drblock[MODE_COUNT:, MODE_COUNT:] = self.dr
        wp = NP.vstack((NP.eye(MODE_COUNT), self.u))/math.sqrt(2)
        dwp = NP.vstack((NP.zeros((MODE_COUNT, MODE_COUNT)), self.du))/math.sqrt(2)
        self.uc, self.duc, self.gaussian, self.dgaussian, self.distance, self.ddistance = geometry.ruler_geometry(psi, self.f)
        vertices, dvertices = [], []
        for cell in range(2):
            cell_v, cell_dv = [], []
            for mask in geometry.filters[cell]:
                leg = mask@rblock@wp
                dleg = mask@(drblock@wp + rblock@dwp)
                cell_v.append([herm(adj(leg)@gamma@leg) for gamma in [geometry.beta]+geometry.currents])
                cell_dv.append([herm(adj(dleg)@gamma@leg + adj(leg)@gamma@dleg)
                                for gamma in [geometry.beta]+geometry.currents])
            vertices.append(cell_v)
            dvertices.append(cell_dv)
        self.v2, self.dv2 = NP.zeros((PAIR_COUNT, PAIR_COUNT), dtype=complex), NP.zeros((PAIR_COUNT, PAIR_COUNT), dtype=complex)
        for nu in range(8):
            for cell in range(2):
                work.check()
                vertex, derivative = vertices[cell][nu][0], dvertices[cell][nu][0]
                coefficient = -0.2*self.uc[cell]**-3
                dcoefficient = 0.6*self.uc[cell]**-4*self.duc[cell]
                term = car.q2(vertex, vertex)
                self.v2 += coefficient*term
                self.dv2 += dcoefficient*term + coefficient*(car.q2(derivative, vertex)+car.q2(vertex, derivative))
            for cell in range(2):
                for other in range(2):
                    for current in range(1, 4):
                        left, right = vertices[cell][nu][current], vertices[other][nu][current]
                        dleft, dright = dvertices[cell][nu][current], dvertices[other][nu][current]
                        self.v2 -= 0.5*self.gaussian[cell, other]*car.q2(left, right)
                        self.dv2 -= 0.5*(self.dgaussian[cell, other]*car.q2(left, right)
                            + self.gaussian[cell, other]*(car.q2(dleft, right)+car.q2(left, dright)))
        self.v2, self.dv2 = herm(self.v2), herm(self.dv2)
        self.bp, self.dbp = herm(car.pair_one_body(self.b)), herm(car.pair_one_body(self.db))
        if self.base:
            count = herm(car.pair_one_body(geometry.ph))
            identity = NP.eye(PAIR_COUNT)
            projectors = [(count-identity)@(count-2*identity)/2, count@(2*identity-count), count@(count-identity)/2]
            energies = [(2-r)*constants["epsilon_l"]+r*constants["epsilon_h"] for r in range(3)]
            def solve(rhs):
                return sum((projectors[r]@rhs@projectors[s]/(energies[r]+energies[s])
                            for r in range(3) for s in range(3)), NP.zeros_like(rhs))
            self.s2 = herm(solve(self.v2))
            self.ds2 = herm(solve(self.dv2-self.dbp@self.s2-self.s2@self.dbp))
        else:
            self.s2 = herm(sylvester_positive(self.bp, self.v2))
            self.ds2 = herm(sylvester_positive(self.bp, self.dv2-self.dbp@self.s2-self.s2@self.dbp))
        self.r2, self.dr2 = matrix_polynomial(-self.s2, derivative=-self.ds2)
        self.w2_reference, self.dw2_reference = matrix_polynomial(-2*self.s2, derivative=-2*self.ds2)
        self.w2 = herm(adj(self.r2)@self.r2)
        self.dw2 = herm(adj(self.dr2)@self.r2 + adj(self.r2)@self.dr2)
        self.bounds = ({"S2_max": constants["sigma"], "S3_max": 3*constants["sigma"],
                        "B_min": constants["epsilon_l"], "B_max": constants["epsilon_h"],
                        "formula_source": "TB7 flat finite-box bound", "scalar_rounding_radius": None}
                       if self.base else offbase_norms(psi, constants))
        self.controls = {"polar_factor": comparison(self.a, self.u@self.e),
            "polar_unitarity": comparison(adj(self.u)@self.u, NP.eye(MODE_COUNT)),
            "polar_root": comparison(self.e@self.e, adj(self.a)@self.a),
            "polar_tangent": comparison(self.da, self.du@self.e+self.u@self.de),
            "root_tangent": comparison(self.e@self.de+self.de@self.e, adj(self.da)@self.a+adj(self.a)@self.da),
            "band_connection_antihermitian": comparison(self.k0+adj(self.k0), NP.zeros_like(self.k0)),
            "energy_positive_square_root": comparison(self.root_e@self.root_e, self.e),
            "energy_root_tangent": comparison(self.root_e@self.droot_e+self.droot_e@self.root_e, self.de),
            "allocated_B_positive_square_root": comparison(self.root_b@self.root_b, self.b),
            "allocated_B_root_tangent": comparison(self.root_b@self.droot_b+self.droot_b@self.root_b, self.db),
            "regulator_Duhamel_spectral": comparison(self.r, r_spectral),
            "regulator_generator_commutation": comparison(generator@self.r, self.r@generator),
            "regulator_Duhamel_commutator_identity": comparison(generator@self.dr-self.dr@generator,
                self.r@dgenerator-dgenerator@self.r),
            "pair_Sylvester": comparison(self.bp@self.s2+self.s2@self.bp, self.v2),
            "pair_derivative_Sylvester": comparison(self.bp@self.ds2+self.ds2@self.bp,
                self.dv2-self.dbp@self.s2-self.s2@self.dbp),
            "positive_middle_vs_p8_reference": comparison(self.w2, self.w2_reference),
            "middle_jet_vs_p8_reference": comparison(self.dw2, self.dw2_reference)}


class Model:
    def __init__(self, native, car, work):
        self.native, self.car, self.work = native, car, work
        self.s3 = car.pair_lift_matrix(native.s2, work)

    def release(self):
        self.s3 = None
        self.work.s3_live = False

    def dress(self, block, sign=1, derivative=False):
        if self.s3 is None:
            raise RuntimeError("Released triple matrix cannot be reused")
        def action(value):
            self.work.check()
            return sign*(self.s3@value)
        tangent = None if not derivative else lambda value: sign*self.car.pair_lift_action(self.native.ds2, value)
        return polynomial_action(action, block, tangent=tangent)

    def middle(self, block, derivative=False, reference=False):
        self.work.check()
        n = self.native
        removed = self.car.annihilate(block)
        w, dw = (n.w2_reference, n.dw2_reference) if reference else (n.w2, n.dw2)
        outputs = NP.einsum("ab,kbc->kac", w, removed, optimize=True)
        mixed = NP.einsum("ik,kbc->ibc", n.b, outputs, optimize=True)
        if derivative:
            doutputs = NP.einsum("ab,kbc->kac", dw, removed, optimize=True)
            mixed = NP.einsum("ik,kbc->ibc", n.db, outputs, optimize=True) + NP.einsum("ik,kbc->ibc", n.b, doutputs, optimize=True)
        return self.car.create_sum(mixed)

    def h_action(self, block, reference=False):
        return self.dress(self.middle(self.dress(block), reference=reference))

    def q_blocks(self, block, derivative=False, clock=False):
        n = self.native
        dressed, ddressed = self.dress(block, derivative=True) if derivative else (self.dress(block), None)
        removed = self.car.annihilate(dressed)
        root = NP.sqrt(NP.repeat(n.psi, 2))[:, None]*n.root_e if clock else n.root_b
        outputs = NP.einsum("ik,kbc->ibc", root, removed, optimize=True)
        q = NP.einsum("ab,ibc->iac", n.r2, outputs, optimize=True)
        if not derivative:
            return q
        droot = (NP.sqrt(NP.repeat(n.psi, 2))[:, None]*(n.droot_e+NP.repeat(n.f, 2)[:, None]*n.root_e/2)
                 if clock else n.droot_b)
        dremoved = self.car.annihilate(ddressed)
        doutputs = NP.einsum("ik,kbc->ibc", droot, removed, optimize=True) + NP.einsum("ik,kbc->ibc", root, dremoved, optimize=True)
        dq = NP.einsum("ab,ibc->iac", n.dr2, outputs, optimize=True) + NP.einsum("ab,ibc->iac", n.r2, doutputs, optimize=True)
        return q, dq

    def h_jet(self, fixed_block):
        dressed, tangent = self.dress(fixed_block, derivative=True)
        middle = self.middle(dressed)
        _, first = self.dress(middle, derivative=True)
        second = self.dress(self.middle(dressed, derivative=True))
        third = self.dress(self.middle(tangent))
        return first+second+third


def gram_of_outputs(outputs, derivative=None):
    count = outputs.shape[-1]
    flat = outputs.reshape(-1, count)
    if derivative is None:
        return herm(adj(flat)@flat)
    dflat = derivative.reshape(-1, count)
    return herm(adj(dflat)@flat+adj(flat)@dflat)


def clock_matrices(outputs, derivative=None):
    return NP.array([gram_of_outputs(outputs[2*x:2*x+2], None if derivative is None else derivative[2*x:2*x+2])
                     for x in range(12)])


def gram_inverse_reference(gram):
    displacement = gram-NP.eye(TRIAL_COUNT)
    result, power, coefficient = NP.eye(TRIAL_COUNT, dtype=complex), NP.eye(TRIAL_COUNT, dtype=complex), 1.0
    for k in range(1, P_GRAM+1):
        power = power@displacement
        coefficient *= -(2*k-1)/(2*k)
        result += coefficient*power
    return herm(result)


def base_reference(model, geometry, constants, archive):
    model.work.check()
    a_i0 = model.dress(geometry.i0, sign=-1)
    gram = herm(adj(a_i0)@a_i0)
    inverse12 = gram_inverse_reference(gram)
    preliminary = a_i0@inverse12
    preliminary_gram = herm(adj(preliminary)@preliminary)
    correction = positive_root(preliminary_gram, power=-0.5)
    phi_basis = geometry.i0@inverse12@correction
    y = model.dress(phi_basis, sign=-1)
    h_y = model.h_action(y)
    q = model.q_blocks(y)
    h_ritz = gram_of_outputs(q)
    values, vectors = NP.linalg.eigh(h_ritz)
    phi = phi_basis@vectors
    a_phi = model.dress(phi, sign=-1)
    z = NP.sum(NP.abs(a_phi)**2, axis=0).real
    if not NP.all(z > 0):
        raise Unresolved("Ordinary trial metric normalization failed")
    psi = a_phi/NP.sqrt(z)[None, :]
    h_psi = model.h_action(psi)
    energies = NP.sum(psi.conj()*h_psi, axis=0).real
    residual = h_psi-psi*energies[None, :]
    physical_norms = NP.sqrt(NP.sum(NP.abs(residual)**2, axis=0)).real
    m_phi = model.middle(phi)
    w_phi = model.dress(a_phi, sign=-1)
    pencil = m_phi-w_phi*energies[None, :]
    metric_residual = model.dress(pencil)/NP.sqrt(z)[None, :]
    pencil_energies = NP.sum(phi.conj()*m_phi, axis=0).real/z
    complement_residual = h_y-y@h_ritz
    beta = norm2(complement_residual)
    # Ordinary TB12 formula evaluations ONLY, never certified endpoints.
    l_preview = (values+constants["L"]-NP.sqrt((constants["L"]-values)**2+4*beta**2))/2
    q_clock = model.q_blocks(y, clock=True)
    clocks = clock_matrices(q_clock)
    q_selected = model.q_blocks(psi, clock=True)
    selected_clocks = NP.array([NP.sum(NP.abs(q_selected[2*x:2*x+2])**2, axis=(0, 1)).real for x in range(12)])
    eig_residual = h_ritz@vectors-vectors*values[None, :]
    eig_gram = adj(vectors)@vectors-NP.eye(TRIAL_COUNT)
    d = norm2(eig_gram)
    polar_correction = d/(1+math.sqrt(1-d)) if d < 1 else None
    spectral_ordinary = (norm2(eig_residual)+(norm2(h_ritz)+float(NP.max(NP.abs(values))))*polar_correction
                        if polar_correction is not None else None)
    controls = {"I0_orthonormal": comparison(adj(geometry.i0)@geometry.i0, NP.eye(TRIAL_COUNT)),
        "Gram_p8_minus2_reference": comparison(gram, adj(geometry.i0)@polynomial_action(lambda b: -2*(model.s3@b), geometry.i0)),
        "p12_inverse_root": comparison(inverse12, positive_root(gram, power=-0.5)),
        "Y_observed_orthonormality": comparison(adj(y)@y, NP.eye(TRIAL_COUNT)),
        "T3Y_vs_undressed_PhiBasis": comparison(model.dress(y), phi_basis),
        "H_action_vs_positive_Q": comparison(adj(y)@h_y, h_ritz),
        "H_positive_middle_vs_p8_middle_reference": comparison(h_y, model.h_action(y, reference=True)),
        "complete_clock_sum": comparison(NP.sum(clocks, axis=0), h_ritz),
        "physical_action_vs_metric_pencil_residual": comparison(residual, metric_residual),
        "action_vs_pencil_Rayleigh": comparison(energies[:, None], pencil_energies[:, None]),
        "S3_CAR_action": comparison(model.s3@geometry.i0, model.car.pair_lift_action(model.native.s2, geometry.i0))}
    archive.update({"I0": geometry.i0, "S2_base": model.native.s2, "V2_base": model.native.v2,
                    "Gram_p8": gram, "Gram_inverse12": inverse12, "Gram_polar_correction": correction,
                    "PhiBasis_frozen": phi_basis, "Y_base_normalized": y, "H3Y_full": h_y,
                    "h_Ritz": h_ritz, "Ritz_values": values, "Ritz_vectors": vectors,
                    "full_complement_residual": complement_residual, "named_phi_all56": phi,
                    "named_psi_all56": psi, "Z_all56": z, "E_action_all56": energies,
                    "full_physical_residual_all56": residual, "full_pencil_residual_all56": pencil,
                    "clock_matrices_all12": clocks, "clock_matrices_Ritz_all12": NP.array([adj(vectors)@row@vectors for row in clocks])})
    saved = {"ordinary_Ritz_values_all56": values.tolist(), "certified_Ritz_intervals": None,
             "ordinary_TB12_formula_preview_all56_not_lower_endpoints": l_preview.tolist(),
             "ordinary_l3_minus_alpha2_preview_not_a_gap_certificate": float(l_preview[2]-values[1]),
             "ground_pair_gate_certified": False, "retained_cluster_dimension": 56,
             "cluster_internal_splitting_status": "unresolved; no exact ground-pair adoption",
             "named_trial_vectors": "all56 saved ordinary Ritz columns; each renormalized in its p8 metric",
             "named_ensemble": "uniform trace ensemble of all56 approximate low Ritz columns, distinct from a ground state",
             "Z_all56": z.tolist(), "E_action_all56": energies.tolist(), "E_pencil_all56": pencil_energies.tolist(),
             "r_physical_all56": physical_norms.tolist(), "ordinary_beta_R": beta,
             "ordinary_low_cluster_width": float(values[-1]-values[0]),
             "clock_expectations_all12_all56": selected_clocks.tolist(),
             "Gram_condition_observed": float(NP.linalg.cond(gram)),
             "Gram_preliminary_deviation": comparison(preliminary_gram, NP.eye(TRIAL_COUNT)),
             "G8_observed_norm_distance_from_identity": norm2(gram-NP.eye(TRIAL_COUNT)),
             "G8_observed_norm_is_not_a_certified_q8": True,
             "G8_q8_formula": "q+delta_Z*(2*exp(3*sigma)+delta_Z); delta_Z includes polynomial, assembly and arithmetic error",
             "G8_missing_delta_Z_arithmetic_and_assembly_contribution": None,
             "G8_certified_q8": None, "G8_certified_p12_radius": None,
             "p12_polar_correction_change": fro(correction-NP.eye(TRIAL_COUNT)),
             "trial_range": "same approximate range of the p8 exp(-S3) I0; no exact-range or exact-metric certificate",
             "ordinary_spectral_residual": norm2(eig_residual), "ordinary_eigenvector_Gram_error": d,
             "ordinary_spectral_polar_formula_without_missing_delta_h": spectral_ordinary,
             "missing_delta_h_assembly_radius": None,
             "CS_clock_source_bound": None, "CS_transported_ruler_bound": None, "CS_native_ruler_bound": None,
             "controls": controls}
    return {"Y": y, "HY": h_y, "PhiBasis": phi_basis, "phi": phi, "psi": psi, "Hpsi": h_psi,
            "Z": z, "E": energies, "r": physical_norms, "pencil": pencil, "vectors": vectors,
            "h_Ritz": h_ritz, "clocks": clocks, "saved": saved}


def polynomial_error_report(native, constants, derivative=False):
    sigma = native.bounds["S2_max"]
    dsigma = constants["sigma1"] if derivative else None
    reports = {"pair_exp_minusS": tail_formula(sigma, dsigma),
               "pair_exp_minus2S_reference": tail_formula(2*sigma, None if dsigma is None else 2*dsigma),
               "triple_exp_plus_or_minusS": tail_formula(3*sigma, None if dsigma is None else 3*dsigma),
               "triple_exp_minus2S_reference": tail_formula(6*sigma, None if dsigma is None else 6*dsigma)}
    r, t = math.exp(sigma), math.exp(3*sigma)
    er, et = reports["pair_exp_minusS"]["ordinary_evaluation"], reports["triple_exp_plus_or_minusS"]["ordinary_evaluation"]
    b = native.bounds["B_max"]
    qnorm = math.sqrt(3*b)*r*t
    qerror = math.sqrt(3*b)*(er*t+(r+er)*et)
    reports["positive_map_polynomial_H_error_formula"] = "(2*Qnorm+Qerror)*Qerror; Qerror=sqrt(3Bmax)*(eR*T+(R+eR)*eT)"
    reports["positive_map_polynomial_H_error_ordinary_evaluation"] = (2*qnorm+qerror)*qerror
    reports["middle_positive_square_error_ordinary_evaluation"] = 2*r*er+er*er
    if derivative:
        dr, dt = dsigma*r, 3*dsigma*t
        der = reports["pair_exp_minusS"]["derivative_ordinary_evaluation"]
        det = reports["triple_exp_plus_or_minusS"]["derivative_ordinary_evaluation"]
        root, droot = math.sqrt(constants["epsilon_h"]), constants["b1"]/(2*math.sqrt(constants["epsilon_l"]))
        dqnorm = math.sqrt(3)*(droot*r*t+root*dr*t+root*r*dt)
        dqerror = math.sqrt(3)*(droot*(er*t+(r+er)*et)
                    + root*(der*t+(dr+der)*et) + root*(er*dt+(r+er)*det))
        reports["positive_map_H_derivative_error_formula"] = "2*(dQerror*Qnorm+dQnorm*Qerror+dQerror*Qerror)"
        reports["positive_map_H_derivative_error_ordinary_evaluation"] = 2*(dqerror*qnorm+dqnorm*qerror+dqerror*qerror)
    q = math.expm1(6*sigma)
    delta_z_polynomial = reports["triple_exp_plus_or_minusS"]["ordinary_evaluation"]
    q8_polynomial_only = q+delta_z_polynomial*(2*math.exp(3*sigma)+delta_z_polynomial)
    reports["Gram_inverse12_tail"] = {"formula": "q^13/(1-q); q=exp(6*sigma)-1",
                                      "ordinary_evaluation": q**13/(1-q), "q_ordinary_evaluation": q,
                                      "applies_to_exact_G_only_current_background": True,
                                      "not_a_bound_for_implemented_G8_polynomial": True,
                                      "scalar_arithmetic_radius": None}
    reports["implemented_G8_reference"] = {"q8_formula": "q+delta_Z*(2*exp(3*sigma)+delta_Z)",
        "delta_Z_polynomial_only_ordinary_evaluation": delta_z_polynomial,
        "q8_approximation_only_ordinary_evaluation": q8_polynomial_only,
        "p12_approximation_only_tail_formula_evaluation": q8_polynomial_only**13/(1-q8_polynomial_only) if q8_polynomial_only < 1 else None,
        "delta_Z_missing_assembly_and_arithmetic_contribution": None,
        "certified_q8": None, "certified_implemented_p12_radius": None,
        "approximation_only_formula_evaluation_is_not_a_G8_arithmetic_certificate": True}
    reports["status"] = "analytical approximation formulas evaluated ordinarily; missing arithmetic radii not supplied"
    return reports


def free_and_car_controls(car, geometry, model, constants):
    work = model.work
    exact_count = [0]*TRIPLE_COUNT
    sign_correct = True
    for mode, (rows, columns, signs) in enumerate(car.maps):
        for row, column, sign in zip(rows, columns, signs):
            triple = car.triples[int(column)]
            position = triple.index(mode)
            sign_correct = sign_correct and int(sign) == (-1)**position and car.pairs[int(row)] == triple[:position]+triple[position+1:]
            exact_count[int(column)] += 1
    if not sign_correct or any(count != 3 for count in exact_count):
        raise RuntimeError("Exact exterior CAR removal/count construction failed")
    controls = {"CAR_integer_removal_signs_and_sum_CdaggerC_equals3I": True,
        "free_Pl_projector": comparison(geometry.pl@geometry.pl, geometry.pl),
        "free_Ph_projector": comparison(geometry.ph@geometry.ph, geometry.ph),
        "free_projector_completeness": comparison(geometry.pl+geometry.ph, NP.eye(MODE_COUNT)),
        "free_projector_ranks_observed": [float(NP.trace(geometry.pl).real), float(NP.trace(geometry.ph).real)],
        "free_low_basis": comparison(geometry.low@adj(geometry.low), geometry.pl),
        "free_triple_energy": comparison(car.one_body_action(model.native.e, geometry.i0), constants["E_l"]*geometry.i0),
        "Walsh_norm": [comparison(mask@adj(mask), NP.eye(4)/8) for masks in geometry.filters for mask in masks],
        "quartic_identity_lift_counts_three_pairs": comparison(car.pair_lift_action(NP.eye(PAIR_COUNT), geometry.i0), 3*geometry.i0),
        "zero_seed_control": {"kind": "algebraic S2=S3=0; no altered-coupling result",
            "energies_formula": "E_l+r*Delta, r=0,1,2,3", "ordinary_values": [constants["E_l"]+r*constants["Delta"] for r in range(4)],
            "exact_multiplicities": [56, 448, 960, 560], "full_sector_dimension": 2024}}
    # Independent Q2 product/subtraction identity on fixed native matrices.
    left, right = geometry.pl, model.native.e
    product = car.pair_one_body(left)@car.pair_one_body(right)-car.pair_one_body(left@right)
    controls["Q2_independent_dGamma_product"] = comparison(car.q2(left, right), product)
    # n2 has S1=0. This control is not the n3 middle construction.
    work.check()
    t2 = matrix_polynomial(model.native.s2)
    annihilated = NP.zeros((MODE_COUNT, MODE_COUNT, PAIR_COUNT), dtype=complex)
    for column, (i, j) in enumerate(car.pairs):
        annihilated[i, j, :] = t2[column, :]
        annihilated[j, i, :] = -t2[column, :]
    q2 = NP.einsum("ik,kbc->ibc", model.native.root_b, annihilated, optimize=True)
    controls["separate_n2_positive_congruence"] = comparison(gram_of_outputs(q2), t2@model.native.bp@t2)
    return controls


def kramers_controls(model, geometry, reference):
    work, car, n = model.work, model.car, model.native
    work.check()
    j = adj(n.u)@geometry.theta2
    a, b = car.pair_array.T
    pair_j = j[a[:, None], a[None, :]]*j[b[:, None], b[None, :]] - j[a[:, None], b[None, :]]*j[b[:, None], a[None, :]]
    theta_y = car.theta3(j, reference["Y"])
    theta_hy = car.theta3(j, reference["HY"])
    square_y = car.theta3(j, theta_y)
    controls = {"particle_Theta_square_minusI": comparison(j@j.conj(), -NP.eye(MODE_COUNT)),
                "pair_seed_Theta": comparison(pair_j@n.v2.conj()@adj(pair_j), n.v2),
                "pair_Sylvester_Theta": comparison(pair_j@n.s2.conj()@adj(pair_j), n.s2),
                "triple_Theta_square_minusI": comparison(square_y, -reference["Y"]),
                "complete_H3_Theta_action": comparison(model.h_action(theta_y), theta_hy),
                "approximate_trial_range_Theta": comparison(theta_y, reference["Y"]@(adj(reference["Y"])@theta_y)),
                "ordinary_adjacent_Ritz_pair_splittings": [float(reference["saved"]["ordinary_Ritz_values_all56"][k+1]-reference["saved"]["ordinary_Ritz_values_all56"][k]) for k in range(0, 56, 2)],
                "exact_pairing_source": "TB13 flat analytic symmetry; arithmetic control does not certify an internal ground pair",
                "nonuniform_probe_symmetry_assumed": False}
    return controls


def diagonal_products(left, right):
    return NP.sum(left.conj()*right, axis=0)


def selected_native_connection(model, psi, hpsi):
    car, t = model.car, model.native.k0/2
    tangent = car.one_body_action(t, psi)
    mean = diagonal_products(psi, tangent)
    removed = car.annihilate(psi)
    number_variance = NP.sum(NP.abs(tangent)**2, axis=0).real-NP.abs(mean)**2
    pair_variance, densities = [], []
    for column in range(TRIAL_COUNT):
        gamma = removed[:, :, column]@removed[:, :, column].conj().T
        densities.append(gamma)
        pair_variance.append(float(NP.trace(adj(t)@(NP.eye(MODE_COUNT)-gamma)@t).real))
    pair_variance = NP.array(pair_variance)
    total = number_variance+pair_variance
    norms = [math.sqrt(float(value)) if value >= 0 else None for value in total]
    commutator = -2*diagonal_products(tangent, hpsi).real
    return tangent, NP.array(densities), {"ordinary_centered_number_variance": number_variance.tolist(),
        "ordinary_pair_creation_CAR_norm_squared": pair_variance.tolist(),
        "ordinary_full_light_centered_tangent_norm": norms,
        "ordinary_connection_commutator": commutator.tolist(),
        "n5_Hamiltonian_constructed": False, "pair_tangent_omitted": False,
        "spectator_connection": "empty spectator tangents orthogonal to light residual; no total packet-connection norm inferred",
        "certified_connection_arithmetic_radius": None}


def ruler_jet(model, geometry, reference, constants, name, archive):
    work, original_native = model.work, model.native
    probe = geometry.probes[name]
    n = Native(geometry, model.car, NP.ones(12), constants, work, f=probe)
    model.native = n
    try:
        y, psi, phi = reference["Y"], reference["psi"], reference["phi"]
        q, dq = model.q_blocks(y, derivative=True)
        transported = gram_of_outputs(q, dq)
        dot_h_y = model.h_jet(y)
        q_clock, dq_clock = model.q_blocks(y, derivative=True, clock=True)
        dot_clocks = clock_matrices(q_clock, dq_clock)
        k_y = model.car.one_body_action(n.k0/2, y)
        commutator_matrix = herm(-adj(k_y)@reference["HY"]-adj(reference["HY"])@k_y)
        native_matrix = transported+commutator_matrix
        a_phi, da_phi = model.dress(phi, sign=-1, derivative=True)
        z = NP.sum(NP.abs(a_phi)**2, axis=0).real
        dz = 2*diagonal_products(a_phi, da_phi).real
        dot_psi = da_phi/NP.sqrt(z)[None, :]-psi*(dz/(2*z))[None, :]
        fixed_q, fixed_dq = model.q_blocks(psi, derivative=True)
        fixed_source = 2*NP.sum(fixed_q.conj()*fixed_dq, axis=(0, 1)).real
        state_work = 2*diagonal_products(dot_psi, reference["Hpsi"]).real
        dot_energy = fixed_source+state_work
        m_dot_phi = model.middle(phi, derivative=True)
        m_phi = model.middle(phi)
        energy_pencil = diagonal_products(phi, m_phi).real/z
        dot_energy_pencil = (diagonal_products(phi, m_dot_phi).real-energy_pencil*dz)/z
        xi = m_phi-model.dress(a_phi, sign=-1)*energy_pencil[None, :]
        frame = model.dress(da_phi)
        cs_work = 2*diagonal_products(xi, frame).real/z
        tangent_centered = dot_psi-psi*diagonal_products(psi, dot_psi)[None, :]
        geometric = NP.sqrt(NP.sum(NP.abs(tangent_centered)**2, axis=0)).real
        _, density, connection = selected_native_connection(model, psi, reference["Hpsi"])
        selected_qclock, selected_dqclock = model.q_blocks(psi, derivative=True, clock=True)
        tangent_qclock = model.q_blocks(dot_psi, clock=True)
        frozen_clock_derivatives = NP.array([2*NP.sum(selected_qclock[2*x:2*x+2].conj()
            *(selected_dqclock[2*x:2*x+2]+tangent_qclock[2*x:2*x+2]), axis=(0, 1)).real for x in range(12)])
        expected_duc = NP.array([2, 2] if name == "uniform" else [1, 0])
        expected_dg = NP.array([[0, -math.exp(-0.25)/8], [-math.exp(-0.25)/8, 0]])
        if name == "left_slice":
            expected_dg /= 8
        controls = {"same_base_S2_under_jet_reassembly": comparison(n.s2, original_native.s2),
            "positive_map_vs_full_three_term_jet": comparison(transported, adj(y)@dot_h_y),
            "allocated_clock_derivative_sum": comparison(NP.sum(dot_clocks, axis=0), transported),
            "explicit_B_ruler_leg": comparison(n.db, n.de+n.root_e@NP.diag(NP.repeat(probe, 2))@n.root_e),
            "geometry_du": comparison(n.duc[:, None], expected_duc[:, None]),
            "geometry_G_derivative": comparison(n.dgaussian, expected_dg),
            "geometry_distance_derivative": comparison(NP.array([n.ddistance]), NP.array([2.0 if name == "uniform" else 0.25])),
            "geometry_contact_weight_derivative": comparison(-3*n.uc**-4*n.duc,
                NP.array([-6, -6] if name == "uniform" else [-3, 0])),
            "frozen_phi_action_vs_pencil_energy_derivative": comparison(dot_energy[:, None], dot_energy_pencil[:, None]),
            "frozen_phi_state_work_vs_CS_metric_identity": comparison(state_work[:, None], cs_work[:, None]),
            "state_tangent_normalization": comparison(diagonal_products(psi, dot_psi).real[:, None], NP.zeros((56, 1)))}
        archive.update({name+"_dS2": n.ds2, name+"_dV2": n.dv2, name+"_full_HdotY_fixed": dot_h_y,
            name+"_transported_ruler_matrix": transported, name+"_native_ruler_matrix": native_matrix,
            name+"_transported_ruler_Ritz": adj(reference["vectors"])@transported@reference["vectors"],
            name+"_native_ruler_Ritz": adj(reference["vectors"])@native_matrix@reference["vectors"],
            name+"_clock_derivative_matrices": dot_clocks, name+"_phi_frozen_state_tangent": dot_psi,
            name+"_particle_density_all56": density})
        saved = {"frozen_undressed_input": "the saved matching PhiBasis(0)*V; no G(t), free-band or Ritz reoptimization",
            "fixed_source_state": "the saved base normalized Y(0); its coordinates and orbitals held fixed",
            "transported_ruler_expectation_all56": fixed_source.tolist(),
            "native_ruler_expectation_all56": (fixed_source+NP.array(connection["ordinary_connection_commutator"])).tolist(),
            "frozen_phi_E_derivative_all56": dot_energy.tolist(), "frozen_phi_pencil_E_derivative_all56": dot_energy_pencil.tolist(),
            "frozen_phi_state_work_all56": state_work.tolist(), "CS_metric_state_work_all56": cs_work.tolist(),
            "ordinary_centered_state_tangent_norm_all56": geometric.tolist(),
            "analytic_state_work_bound_formula": "4*r/5; scalar/arithmetic residual enclosure missing",
            "light_native_connection": connection, "native_controls": n.controls,
            "polynomial_error_formulas": polynomial_error_report(n, constants, derivative=True), "controls": controls,
            "nonuniform_Kramers_preservation_assumed": False}
        targets = {"fixedY_H_matrix": transported, "fixedY_clock_matrices": dot_clocks,
                   "frozen_phi_Z": dz, "frozen_phi_E": dot_energy,
                   "frozen_phi_clocks": frozen_clock_derivatives, "frozen_phi_state": dot_psi,
                   "native_R": n.dr, "native_E": n.de, "native_U": n.du,
                   "native_B": n.db, "native_S2": n.ds2, "native_V2": n.dv2,
                   "S3_I0": model.car.pair_lift_action(n.ds2, geometry.i0),
                   "geometry_u": n.duc, "geometry_distance": NP.array([n.ddistance]), "geometry_G": n.dgaussian}
        return saved, targets
    finally:
        model.native = original_native


def rebuild_readouts(model, reference, geometry):
    y, phi = reference["Y"], reference["phi"]
    q = model.q_blocks(y)
    fixed_clock = model.q_blocks(y, clock=True)
    a_phi = model.dress(phi, sign=-1)
    z = NP.sum(NP.abs(a_phi)**2, axis=0).real
    if not NP.all(z > 0):
        raise Unresolved("Ordinary rebuilt frozen-phi normalization failed")
    psi = a_phi/NP.sqrt(z)[None, :]
    selected_clock = model.q_blocks(psi, clock=True)
    h_psi = model.h_action(psi)
    n = model.native
    return {"fixedY_H_matrix": gram_of_outputs(q), "fixedY_clock_matrices": clock_matrices(fixed_clock),
            "frozen_phi_Z": z, "frozen_phi_E": diagonal_products(psi, h_psi).real,
            "frozen_phi_clocks": NP.array([NP.sum(NP.abs(selected_clock[2*x:2*x+2])**2, axis=(0, 1)).real for x in range(12)]),
            "frozen_phi_state": psi, "native_R": n.r.copy(), "native_E": n.e.copy(), "native_U": n.u.copy(),
            "native_B": n.b.copy(), "native_S2": n.s2.copy(), "native_V2": n.v2.copy(),
            "S3_I0": model.s3@geometry.i0, "geometry_u": n.uc.copy(),
            "geometry_distance": NP.array([n.distance]), "geometry_G": n.gaussian.copy()}


def finite_difference_controls(car, geometry, constants, reference, targets, work, result, archive):
    result["finite_difference_controls"] = {"status": "attempted", "exact_steps": [str(step) for step in STEPS],
        "only_probes": ["uniform", "left_slice"], "state_conventions": "fixed baseY; separately frozen actual undressed phi",
        "maximum_nonzero_rebuilds": 12, "attempted_nonzero_rebuilds": 0, "completed_nonzero_rebuilds": 0,
        "observed_differences_are_not_certified_derivative_radii": True, "rows": []}
    saved = result["finite_difference_controls"]
    for name in ("uniform", "left_slice"):
        previous = {}
        for exact_step in STEPS:
            h = float(exact_step)
            row = {"probe": name, "exact_step": str(exact_step), "status": "attempted", "rebuilds": []}
            saved["rows"].append(row)
            readouts = []
            for sign in (-1, 1):
                work.check()
                saved["attempted_nonzero_rebuilds"] += 1
                psi = NP.exp(sign*h*geometry.probes[name])
                rebuild = {"sign": sign, "status": "attempted", "psi_site_values": psi.tolist()}
                row["rebuilds"].append(rebuild)
                native = Native(geometry, car, psi, constants, work)
                model = Model(native, car, work)
                try:
                    values = rebuild_readouts(model, reference, geometry)
                    readouts.append(values)
                    rebuild.update({"status": "ordinary_rebuild_complete", "offbase_norm_bounds": native.bounds,
                        "polynomial_error_formulas": polynomial_error_report(native, constants),
                        "native_identity_controls": native.controls,
                        "offbase_allocated_clock_sum": comparison(NP.sum(values["fixedY_clock_matrices"], axis=0), values["fixedY_H_matrix"])})
                    archive[name+"_"+str(exact_step.denominator)+"_"+str(sign)+"_S2"] = native.s2
                    archive[name+"_"+str(exact_step.denominator)+"_"+str(sign)+"_V2"] = native.v2
                    saved["completed_nonzero_rebuilds"] += 1
                finally:
                    model.release()
            comparisons = {}
            for key, target in targets[name].items():
                difference = readouts[1][key]-readouts[0][key]
                measured = difference/(2*h)
                control = comparison(measured, target)
                subtraction_scale = max(1.0, fro(readouts[1][key]), fro(readouts[0][key]))
                indicator = float(NP.finfo(float).eps)*subtraction_scale/(2*h)
                resolved = fro(difference) > 64*float(NP.finfo(float).eps)*subtraction_scale
                control.update({"subtraction_input_scale": subtraction_scale,
                    "central_difference_denominator": 2*h,
                    "roundoff_scale_indicator_only_not_a_bound": indicator,
                    "subtraction_signal_resolved_heuristically": resolved,
                    "cancellation_status": "observed signal above heuristic floor" if resolved else "unresolved cancellation; no derivative/sign credit",
                    "previous_error_over_current_error": (previous[key]/control["absolute_Frobenius_difference"]
                        if key in previous and control["absolute_Frobenius_difference"] > 0 else None)})
                previous[key] = control["absolute_Frobenius_difference"]
                comparisons[key] = control
                if key != "frozen_phi_state" and key != "S3_I0":
                    archive[name+"_FD_"+str(exact_step.denominator)+"_"+key] = measured
            row.update({"status": "all_fixed_readouts_attempted", "readouts": comparisons})
        del readouts
    saved["status"] = "all12_nonzero_rebuilds_and_fixed_controls_completed; engineering_only"


def certificate_flags():
    return {"assembly_certified": False, "arithmetic_certified": False, "ground_gap_certified": False,
            "ground_pair_certified": False, "CS_source_certified": False,
            "same_sector_minimizes_fixed_charge_with_holes_certified": False,
            "dense_medium_EOS_certified": False, "stellar_or_interior_source_certified": False}


def result_template(args, backend, admission):
    return {"scope": "one-point full-coupling finite-parent n3 engineering diagnostic only",
        "source": "solve/strand_A_cold_three_body.tex", "independent_specification_review": "solve/review_A_cold_three_body.tex",
        "rebuild_addendum": "solve/strand_A_three_body_rebuilds.tex",
        "independent_rebuild_addendum_review": "solve/review_A_three_body_rebuilds.tex",
        "implementation_independent_review": "solve/review_A_cold_three_body_code.tex",
        "exact_input": EXACT_INPUT, "backend": backend, "admission": admission,
        "dimensions": {"sites": 12, "particle_modes": 24, "complete_pair_middle": 276, "complete_triple_sector": 2024, "trial_columns": 56},
        "spectator_packets_and_bosons": "invariant empty excitation sectors; protected-vacuum source rows separate",
        "requested_stage": args.stage, "status": "unresolved", "model_evaluation_attempted": False,
        "model_evaluated": False, "base_model_evaluated": False, "certificates": certificate_flags(),
        "diagnostic_only_fallback": True, "missing_radii": MISSING_RADII,
        "arithmetic_missing_inputs_are_not_replaced_by_agreement": True,
        "working_array_plan": {"hard_cap_bytes": WORKING_ARRAY_CAP, "charged_transient_majorant_bytes": ALLOCATION_BYTES,
            "one_live_stored_S3_bytes": 65545216, "triple_exp_H_source_clock_matrices_stored": False,
            "dS3_action_only": True, "full_pair_middle_streams_all276_outputs": True,
            "archive_arrays_charged_separately_conservatively": True, "dense_triple_eigensolve": False,
            "full_Fock_or_n5_Hamiltonian": False, "fixed_library_overhead_excluded_from_array_plan": True},
        "predeclared_runtime_caps": {"seconds": args.seconds, "process_RSS_MiB": args.rss_mib,
            "working_array_bytes": WORKING_ARRAY_CAP, "maximum_own_threads": 4, "all_six_worker_pools": 1,
            "nonzero_field_rebuilds": 12, "one_input_point": True, "polynomial_degrees": {"exponential": 8, "Gram_inverse_root": 12}},
        "unexecuted_controls": ["base_full_sector", "both_analytic_ruler_probes", "all_fixed_finite_differences"],
        "scientific_source_bounds": {"clock": None, "transported_ruler": None, "native_ruler": None}}


def mark_failed_stage(result, status, message):
    for row in result.get("ruler_probes", {}).values():
        if row.get("status") == "attempted":
            row.update({"status": status, "unresolved_item": message})
    controls = result.get("finite_difference_controls")
    if controls is not None and controls.get("status") == "attempted":
        controls.update({"status": status, "unresolved_item": message})
        for row in controls["rows"]:
            if row.get("status") == "attempted":
                row.update({"status": status, "unresolved_item": message})
                for rebuild in row["rebuilds"]:
                    if rebuild.get("status") == "attempted":
                        rebuild.update({"status": status, "unresolved_item": message})


def serialization_ready(value, path, nonfinite):
    if isinstance(value, NP.generic):
        value = value.item()
    if isinstance(value, NP.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: serialization_ready(item, path+"."+str(key), nonfinite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialization_ready(item, path+"["+str(index)+"]", nonfinite) for index, item in enumerate(value)]
    if isinstance(value, complex):
        return {"real": serialization_ready(value.real, path+".real", nonfinite),
                "imag": serialization_ready(value.imag, path+".imag", nonfinite)}
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append({"path": path, "ordinary_value": str(value), "status": "unresolved; not a radius"})
        return None
    return value


def save_outputs(args, result, archive):
    # Distinct resolved paths are checked before backend import/admission;
    # exclusive file creation also refuses a post-admission path race.
    result["numerical_archive"] = str(args.archive)
    result["text_companion"] = str(args.text)
    nonfinite = []
    data = serialization_ready(result, "result", nonfinite)
    if nonfinite:
        data["nonfinite_ordinary_scalar_paths"] = nonfinite
        data["status"] = "unresolved_nonfinite_ordinary_diagnostic"
        data["model_evaluated"] = False
        data["unresolved_item"] = data.get("unresolved_item", "Nonfinite ordinary scalars retained as explicit unresolved/null diagnostics")
    # Validate the full sanitized JSON before creating any artifact. The
    # archive retains raw arrays, including any nonfinite entries, without
    # turning them into accepted ordinary or certified quantities.
    json_body = json.dumps(data, indent=2, allow_nan=False)+"\n"
    archive_error = None
    try:
        with args.archive.open("xb") as stream:
            NP.savez(stream, **archive)
    except BaseException as error:
        archive_error = error
        data.update({"status": "output_archive_error_partial_summary", "model_evaluated": False,
                     "output_archive_error": type(error).__name__+": "+str(error)})
        json_body = json.dumps(data, indent=2, allow_nan=False)+"\n"
    with args.output.open("x") as stream:
        stream.write(json_body)
    lines = [data["scope"], "status: "+data["status"],
        "assembly / ground-gap / CS-source certified: false / false / false",
        "full pair276, triple2024, trial56; one fixed input point",
        "missing arithmetic inputs: "+"; ".join(MISSING_RADII)]
    if "unresolved_item" in data:
        lines.append("unresolved item: "+data["unresolved_item"])
    if nonfinite:
        lines.append("nonfinite ordinary scalar paths: "+"; ".join(item["path"] for item in nonfinite))
    if archive_error is not None:
        lines.append("archive error: "+data["output_archive_error"])
    with args.text.open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    if archive_error is not None:
        raise archive_error
    return data["status"]


def run(args, backend, admission):
    archive = {}
    work = Work(args.seconds, time.perf_counter(), rss_mib=args.rss_mib, archive=archive)
    result = result_template(args, backend, admission)
    model = None
    fatal = None
    try:
        work.check()
        result["model_evaluation_attempted"] = True
        result["active_operation"] = "exact CAR/geometry and ordinary native seed construction"
        car = CAR()
        geometry = Geometry(car)
        constants = analytic_constants()
        result["analytic_constants_ordinary_formula_evaluations"] = constants
        native = Native(geometry, car, NP.ones(12), constants, work)
        archive.update({"S2_base": native.s2, "V2_base": native.v2})
        result["base_native_controls"] = native.controls
        result["base_polynomial_error_formulas"] = polynomial_error_report(native, constants)
        model = Model(native, car, work)
        result["active_operation"] = "complete-sector correlated trial/Ritz/residual/clock construction"
        reference = base_reference(model, geometry, constants, archive)
        result["base_diagnostic"] = reference["saved"]
        result["free_CAR_and_n2_controls"] = free_and_car_controls(car, geometry, model, constants)
        result["flat_Kramers_controls"] = kramers_controls(model, geometry, reference)
        result["base_model_evaluated"] = True
        result["unexecuted_controls"].remove("base_full_sector")
        if args.stage in ("jets", "full"):
            result["active_operation"] = "two fixed analytic ruler jets and frozen-phi state work"
            result["ruler_probes"] = {}
            targets = {}
            for name in ("uniform", "left_slice"):
                work.check()
                result["ruler_probes"][name] = {"status": "attempted"}
                saved, target = ruler_jet(model, geometry, reference, constants, name, archive)
                result["ruler_probes"][name] = {"status": "ordinary_full_jet_complete", **saved}
                targets[name] = target
            result["unexecuted_controls"].remove("both_analytic_ruler_probes")
        model.release()
        if args.stage == "full":
            result["active_operation"] = "twelve fixed nonzero-field central-difference rebuilds"
            finite_difference_controls(car, geometry, constants, reference, targets, work, result, archive)
            result["unexecuted_controls"].remove("all_fixed_finite_differences")
        work.check()
        result.update({"model_evaluated": True, "status": "completed_requested_engineering_stage",
                       "independent_saved_output_review_status": "pending; no numerical certificate or physical adoption"})
        result.pop("active_operation", None)
    except (Unresolved, Cap) as error:
        result["unresolved_item"] = str(error)
        result["unresolved_kind"] = "engineering cap or ordinary numerical control; no physics classification"
        mark_failed_stage(result, "unresolved", str(error))
    except BaseException as error:
        fatal = error
        result.update({"status": "backend_or_arithmetic_error", "unresolved_item": type(error).__name__+": "+str(error)})
        mark_failed_stage(result, "backend_or_arithmetic_error", result["unresolved_item"])
    finally:
        if model is not None and model.s3 is not None:
            model.release()
    result.update({"elapsed_seconds": time.perf_counter()-work.start, "cooperative_check_count": work.actions,
                   "peak_process_RSS_KiB_observed": work.peak_rss_kib,
                   "peak_charged_working_array_majorant_bytes": work.peak_working_array_majorant_bytes})
    # Certificate flags are immutable false even after observed agreement.
    result["certificates"] = certificate_flags()
    try:
        saved_status = save_outputs(args, result, archive)
    except BaseException as output_error:
        if fatal is not None:
            raise fatal from output_error
        raise
    if fatal is not None:
        raise fatal
    return saved_status


def load_backend():
    global NP
    import numpy
    NP = numpy
    return {"NumPy_version_actual": numpy.__version__, "dtype": "float64/complex128",
            "directed_or_proved_arithmetic_backend": False,
            "small_eigh_native_pair_Ritz_only": True, "no_triple_eigensolve": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--stage", choices=("base", "jets", "full"), default="full")
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--rss-mib", type=int, default=5120)
    args = parser.parse_args()
    if args.seconds <= 0 or args.rss_mib <= 0:
        parser.error("Time/RSS caps must be fixed positive integers before the model")
    paths = (args.output, args.archive, args.text)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Summary/archive/text outputs must be distinct resolved paths")
    if any(path.exists() or not path.parent.is_dir() for path in paths):
        parser.error("All output paths must be new files in existing directories")
    if ALLOCATION_BYTES > WORKING_ARRAY_CAP:
        parser.error("The frozen allocation plan exceeds the four-GiB cap")
    pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in pools):
        parser.error("All six worker/BLAS pools must be declared as one")
    backend = load_backend()  # Finish every host-only import before final admission.
    admission = remote_execution(max_threads=4)
    print(run(args, backend, admission), flush=True)


if __name__ == "__main__":
    main()
