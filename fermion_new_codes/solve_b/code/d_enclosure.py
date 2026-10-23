#!/usr/bin/env python3
"""Original-A open3^3 D load-one ground-state candidate and enclosure.

This is precisely the finite diagnostic released by
solve/strand_D_enclosure_spec.tex and review_D_enclosure_spec.tex.
The full Q=+2, number=2 reducing sector has dimension1431; no other
physical-charge sectors or stellar/operator claim is supplied.

Run exclusively oncompute-host with the qi2 interpreter, nice10, a
freshly free reserved GPU and the lane-wide <=16-thread allocation.
No sampling or finite difference enters the certificate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import d_verified as verified
from d_verified import (Ball, Taylor3, add_up, sub_down, mul_up, div_up,
                        sqrt_up, sqrt_down, gamma, underflow, norm_upper,
                        absolute_norm_upper, scalar_real, trace_block, complex_product_center,
                        expectation, squared_norm, validate_eigenpair,
                        validate_response, sylvester_taylor,
                        congruence_from_delta)
from lattice_seed import ALPHA, BETA, pairs_for, pair_product, second_quantized_pair
from lattice_trace import MASSES_SQUARED, GRADING, sine_lattice
from polar_law import polar_jet, diagonal_field
from remote_runtime import remote_execution


PARAMETERS = {"law": "original_A_sine_polar", "shape": [3, 3, 3],
              "a": "1", "mass_squared": ["1/4", "3/4", "1", "3/2"],
              "eta": [1, -1, -1, 1], "Lambda": "1/2", "ell": "2",
              "g_M": "1/80", "g_B": "1/200", "source_load": "1",
              "complete_cells": 8, "ordered_cell_pairs": 64,
              "particle_modes": 54, "pair_dimension": 1431,
              "interior": [1, 1, 1], "collar_s_n": "1",
              "domain": {"s": ["999/1000", "1001/1000"],
                         "n": ["999/1000", "1001/1000"]}}
SCOPE = ("Full1431-dimensional original-A open3^3 charge+2/excitation2 finite "
         "ground-state diagnostic with one clamped-collar interior site. "
         "No full-charge comparison, stellar state, Thomas-Fermi result, "
         "internal ruler operator, production couplings or physical spacing claim.")


def xp():
    return verified.XP


def as_float(value):
    return float(value.item() if hasattr(value, "item") else value)


def point_bounds(value):
    low, high = value.bounds()
    return {"center": as_float(value.data.real), "lower": low,
            "upper": high, "radius": value.radius}


def lower_product(*values):
    from decimal import localcontext, ROUND_FLOOR, Decimal
    with localcontext() as context:
        context.prec = verified.DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        value = Decimal(1)
        for factor in values:
            value *= verified.decimal(factor)
        return verified.float_down(value)


def center_field(value, width=0):
    """Only the central ruler varies; the26 collar entries stay exact."""
    array = np.ones(27)
    array[13] = value
    first = np.zeros(27)
    first[13] = 1
    return Taylor3(Ball(array, width), [Ball(first), Ball(np.zeros(27)), Ball(np.zeros(27))])


def scalar_ruler(value, width=0):
    return Taylor3(Ball(value, width), [Ball(1), Ball(0), Ball(0)])


def exterior_onebody(matrix: Ball, pairs):
    """dGamma2 has operator-norm gain2, with explicit center rounding."""
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    m = matrix.data
    value = (m[i, k] * (j == ell) - m[i, ell] * (j == k)
             - m[j, k] * (i == ell) + m[j, ell] * (i == k))
    rounding = add_up(mul_up(gamma(16), 2, matrix.absolute_norm()),
                      underflow(16, int(value.size)))
    return Ball(value, add_up(2 * matrix.radius, rounding))


def exterior_jet(matrix: Taylor3, pairs):
    return Taylor3.from_coefficients([exterior_onebody(x, pairs) for x in matrix.coefficients])


def exterior_product(first: Ball, second: Ball, pairs):
    """Reviewed four-external-leg map on the entire exterior square."""
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    k, ell = pairs[None, :, 0], pairs[None, :, 1]
    m, n = first.data, second.data
    value = (complex_product_center(m[i, k], n[j, ell]) - complex_product_center(m[i, ell], n[j, k])
             - complex_product_center(m[j, k], n[i, ell]) + complex_product_center(m[j, ell], n[i, k]))
    # On the antisymmetric tensor square the bilinear map is the
    # compression of M tensor N + N tensor M, hence its norm is <=2MN.
    # The absolute-entry version has the same bound before subtraction.
    propagation = mul_up(2, add_up(mul_up(first.center_norm(), second.radius),
                                   mul_up(second.center_norm(), first.radius),
                                   mul_up(first.radius, second.radius)))
    rounding = add_up(mul_up(gamma(40), 2, first.absolute_norm(), second.absolute_norm()),
                      underflow(40, int(value.size)))
    return Ball(value, add_up(propagation, rounding))


def exterior_product_jet(first: Taylor3, second: Taylor3, pairs):
    result = []
    for degree in range(4):
        value = exterior_product(first.coefficients[0], second.coefficients[degree], pairs)
        for j in range(1, degree + 1):
            value = value + exterior_product(first.coefficients[j], second.coefficients[degree - j], pairs)
        result.append(value)
    return Taylor3.from_coefficients(result)


def rectangular_annihilator(matrix: Ball, site, pairs):
    """[(Pi_x C)d] maps pairs to2 spin components times54 survivors.

    Its operator-norm gain in C is <=sqrt2, by the exact exterior-power
    annihilator identity. The center map uses copies and signs only.
    """
    modes, dimension = matrix.data.shape[0], len(pairs)
    result = xp().zeros((2 * modes, dimension), dtype=xp().complex128)
    columns = xp().arange(dimension)
    for spin in range(2):
        row = 2 * site + spin
        result[spin * modes + pairs[:, 1], columns] = matrix.data[row, pairs[:, 0]]
        result[spin * modes + pairs[:, 0], columns] = -matrix.data[row, pairs[:, 1]]
    return Ball(result, mul_up(sqrt_up(2), matrix.radius))


def annihilator_jet(root: Taylor3, scalar_s: Taylor3, pairs):
    raw = Taylor3.from_coefficients([rectangular_annihilator(x, 13, pairs) for x in root.coefficients])
    return scalar_s.positive_root() * raw


def source_amplitude(annihilator: Ball, delta: Ball, state: Ball):
    dressed_state = state + delta @ state
    return annihilator @ dressed_state


def exact_gap_budget():
    """Evaluate E11/E12 with exact rational inputs and outward bounds."""
    a_minus, a_plus = Ball.rational(999, 1000), Ball.rational(1001, 1000)
    minus = Taylor3(a_minus)
    ratio = a_minus / a_plus
    d_a = Ball.rational(1, 2) * (minus ** -1 - 1).coefficients[0]
    d_a += Ball.rational(3, 2).scalar_sqrt() * (minus ** -3 - 1).coefficients[0]
    sqrt7 = Ball(7).scalar_sqrt()
    denominator = Ball.rational(1, 2) + Ball.rational(1, 2) * a_minus * a_minus / (a_plus * a_plus * a_plus)
    d_e = d_a * (sqrt7 + d_a) / denominator
    d_h = 2 * (d_e + (sqrt7 / 2 + d_e) * (a_plus * a_plus - 1))
    contact = (Taylor3((7 + a_minus * a_minus) / 8) ** -3).coefficients[0]
    v_star = Ball.rational(1, 40) * contact + Ball.rational(3, 200)
    sigma = v_star / (2 * ratio * ratio * ratio)
    eta = d_h + ((2 * sigma).scalar_exp() - 1) * (sqrt7 + d_h)
    delta0 = (Ball(3).scalar_sqrt() - 1) / 2
    gap = delta0 - 2 * eta
    eta_upper = eta.bounds()[1]
    gap_lower = gap.bounds()[0]
    eta_limit = Ball.rational(7, 50).bounds()[0]
    gap_limit = Ball.rational(2, 25).bounds()[1]
    if eta_upper >= eta_limit or gap_lower <= gap_limit:
        raise ArithmeticError("Reviewed original-A whole-domain gap inequalities failed.")
    second_level = (1 + delta0 - eta).bounds()[0]
    return {"d_A": point_bounds(d_a), "d_E": point_bounds(d_e),
            "d_H": point_bounds(d_h), "v_star": point_bounds(v_star),
            "sigma_star": point_bounds(sigma), "eta_star": point_bounds(eta),
            "free_gap": point_bounds(delta0), "ground_gap": point_bounds(gap),
            "ground_gap_lower": gap_lower, "second_level_lower": second_level,
            "eta_below_7_over50": sub_down(eta_limit, eta_upper),
            "gap_above_2_over25": sub_down(gap_lower, gap_limit)}


def build(s, width=0, label="center"):
    """Rebuild every polar/sea/seed/denominator/dressing derivative."""
    started = time.monotonic()
    print(f"D {label}: rebuilding original-A full1431 sector at s={s:.17g}, width={width:.3g}.", flush=True)
    coordinates, dirac = sine_lattice((3, 3, 3), 1, periodic=False)
    if not np.array_equal(MASSES_SQUARED, [0.25, 0.75, 1.0, 1.5]) or not np.array_equal(GRADING, [1, -1, -1, 1]):
        raise RuntimeError("The reviewed original-A D inventory changed.")
    if len(dirac) != 54 or not np.array_equal(coordinates[13], [1, 1, 1]):
        raise RuntimeError("The reviewed D parent geometry changed.")
    # The sine entries here are exactly representable zeros and half units.
    if not np.array_equal(dirac * 2, np.round(dirac.real * 2) + 1j * np.round(dirac.imag * 2)):
        raise RuntimeError("The assumed exact sine input entries changed.")
    pairs = xp().asarray(pairs_for(54))
    if len(pairs) != 1431:
        raise RuntimeError("Full-sector dimension changed.")
    ruler, scalar_s = center_field(s, width), scalar_ruler(s, width)
    light = polar_jet(ruler, dirac, Ball.rational(1, 2), uv_r=None)
    root = light["rootE0"]
    b = (root @ diagonal_field(ruler) @ root).hermitian()
    projector = np.zeros((54, 54))
    projector[26:28, 26:28] = np.eye(2)
    a_center = (scalar_s * (root @ projector @ root)).hermitian()
    h0, ac = exterior_jet(b, pairs).hermitian(), exterior_jet(a_center, pairs).hermitian()
    wplus = light["U"].map(lambda x: np.vstack([np.zeros_like(x), x]), gain=1)
    wplus += np.vstack([np.eye(54), np.zeros((54, 54))])
    wplus /= Ball(2).scalar_sqrt()
    inverse = diagonal_field(ruler ** -1)
    kernel = (inverse @ dirac @ inverse).hermitian()
    regulator = (-(kernel @ kernel) * 2).exponential().hermitian()
    native_regulator = regulator.map(lambda x: np.kron(np.eye(2), x), gain=1)
    origins = list(itertools.product((0, 1), repeat=3))
    indices = {tuple(x): k for k, x in enumerate(coordinates.astype(int))}
    contact = ((7 + scalar_s ** 2) / 8) ** -3
    integrals = [Ball.rational(1, 8), Ball.rational(3, 16),
                 Ball.rational(7, 24), Ball.rational(15, 32)]
    bond = {}
    for h in range(4):
        d_squared = h * (1 + (scalar_s ** 2 - 1) * integrals[h]) ** 2
        bond[h] = (-d_squared / 4).exponential() / 8
    seed = Taylor3(xp().zeros((1431, 1431)))
    for valley_index, valley in enumerate(itertools.product((0, 1), repeat=3)):
        vertices = []
        for origin in origins:
            native_filter = np.zeros((4, 108), complex)
            for corner in itertools.product((0, 1), repeat=3):
                site = indices[tuple(np.array(origin) + corner)]
                sign = (-1) ** int(np.dot(valley, corner)) / 8
                native_filter[:2, 2 * site:2 * site + 2] = sign * np.eye(2)
                native_filter[2:, 54 + 2 * site:54 + 2 * site + 2] = sign * np.eye(2)
            field = native_filter @ native_regulator @ wplus
            vertices.append([(field.dagger() @ gamma_matrix @ field).hermitian()
                             for gamma_matrix in [BETA] + ALPHA])
        for r, origin in enumerate(origins):
            scalar_vertex = vertices[r][0]
            seed += exterior_product_jet(scalar_vertex,
                       -Ball.rational(1, 80) * contact * scalar_vertex, pairs)
            for direction in range(1, 4):
                # Bilinearity collects the8 small external-leg matrices
                # before the exterior lift; all64 ordered cell pairs stay.
                weighted = Taylor3(np.zeros((54, 54)))
                for q, other in enumerate(origins):
                    h = sum(a != b for a, b in zip(origin, other))
                    weighted += -Ball.rational(1, 400) * bond[h] * vertices[q][direction]
                seed += exterior_product_jet(vertices[r][direction], weighted, pairs)
        print(f"D {label}: seed valley{valley_index + 1}/8 complete.", flush=True)
    seed = seed.hermitian()
    s_min, s_max = sub_down(s, width), add_up(s, width)
    # min b >=m*(s_min/s_max)^3 and min dGamma2(b)>=2 min b.
    # Include the exact clamped collar extrema when the center is above1.
    low, high = min(1.0, s_min), max(1.0, s_max)
    ratio = Ball(low) / Ball(high)
    h0_lower = (ratio * ratio * ratio).bounds()[0]
    completion = sylvester_taylor(h0, seed, h0_lower).hermitian()
    delta = completion.exponential(delta=True).hermitian()
    dressed_b = congruence_from_delta(delta, h0)
    dressed_e = congruence_from_delta(delta, ac)
    grouped = Taylor3(np.zeros((54, 54)))
    packet_errors = []
    for mass_index, (mass_squared, orientation) in enumerate(zip(MASSES_SQUARED, GRADING)):
        square = [Ball.rational(1, 4), Ball.rational(3, 4), Ball(1), Ball.rational(3, 2)][mass_index]
        packet = light if mass_index == 0 else polar_jet(ruler, dirac, square.scalar_sqrt(), uv_r=None)
        grouped += int(orientation) * (packet["E1"] - packet["E0"])
        packet_errors.append({"mass_squared": float(mass_squared),
                              "root0_radius": packet["E0"].coefficients[0].radius,
                              "root1_radius": packet["E1"].coefficients[0].radius})
    t = Taylor3.from_coefficients([trace_block(x, 26) for x in grouped.coefficients])
    trace_all = Taylor3.from_coefficients([trace_block(x, 0, 54) for x in grouped.coefficients])
    annihilator = annihilator_jet(root, scalar_s, pairs)
    elapsed = time.monotonic() - started
    print(f"D {label}: all roots/seed/Sylvester/exp derivatives built in{elapsed:.1f}s.", flush=True)
    return {"s": s, "width": width, "coordinates": coordinates, "pairs": pairs,
            "C": root, "H0": h0, "Ac": ac, "V": seed, "S2": completion,
            "delta": delta, "B": dressed_b, "E": dressed_e,
            "t": t, "grouped": grouped, "trace_all": trace_all,
            "annihilator": annihilator, "packet_errors": packet_errors,
            "h0_lower": h0_lower, "build_seconds": elapsed}


def reusable_seed_gate():
    """Small independent sea-Wick/normal-product gate; diagnostic only.

    The reviewed full-sea A builder keeps every charge block on open2^3.
    Its pp block is compared with C's independently ordered normal
    product and this dedicated arithmetic/contracted external-leg path.
    This control is not the final nonlinear/root arithmetic certificate.
    """
    from lattice_seed import assemble as sea_assemble
    from pair_response import assemble_pair
    print("D reusable gate: open2^3 full-sea Wick versus both pp implementations.", flush=True)
    coordinates, dirac = sine_lattice((2, 2, 2), 1, periodic=False)
    signs = 2 * coordinates - 1
    ruler = 1 + signs[:, 0] / 32 + signs[:, 1] / 64 + signs[:, 2] / 128
    lapse = np.ones(8)
    # This dyadic profile has exact binary64 squares and exact8-entry mean.
    mean_square = float(np.mean(ruler * ruler))
    particle = polar_jet(Taylor3(ruler), dirac, Ball.rational(1, 2), uv_r=None)
    inverse = diagonal_field(Taylor3(ruler) ** -1)
    kernel = (inverse @ dirac @ inverse).hermitian()
    regulator = (-(kernel @ kernel) * 2).exponential().hermitian()
    native_regulator = regulator.map(lambda x: np.kron(np.eye(2), x), gain=1)
    wplus = particle["U"].map(lambda x: np.vstack([np.zeros_like(x), x]), gain=1)
    wplus += np.vstack([np.eye(16), np.zeros((16, 16))])
    wplus /= Ball(2).scalar_sqrt()
    contact = Taylor3(Ball(mean_square)) ** -3
    pairs = xp().asarray(pairs_for(16))
    seed = Taylor3(np.zeros((120, 120)))
    for valley in itertools.product((0, 1), repeat=3):
        native_filter = np.zeros((4, 32), complex)
        for site, position in enumerate(coordinates.astype(int)):
            sign = (-1) ** int(np.dot(valley, position)) / 8
            native_filter[:2, 2 * site:2 * site + 2] = sign * np.eye(2)
            native_filter[2:, 16 + 2 * site:16 + 2 * site + 2] = sign * np.eye(2)
        field = native_filter @ native_regulator @ wplus
        for direction, gamma_matrix in enumerate([BETA] + ALPHA):
            vertex = (field.dagger() @ gamma_matrix @ field).hermitian()
            weight = -Ball.rational(1, 80) * contact if direction == 0 else Taylor3(-Ball.rational(1, 3200))
            seed += exterior_product_jet(vertex, weight * vertex, pairs)
    pp = assemble_pair(ruler, lapse, coordinates, dirac, shape=(2, 2, 2), spacing=1,
                       modes=np.zeros((2, 8)), analytic=False, uv_r=None,
                       cutoff=0.5, length=2, couplings=(1 / 80, 1 / 200),
                       coupling_scale=1, check_seed=True)
    sea = sea_assemble(ruler, lapse, coordinates, dirac, uv_r=None)
    pp_indices = np.flatnonzero(sea["pairs"][:, 1] < 16)
    sea_pp = sea["V2"][np.ix_(pp_indices, pp_indices)] / 4
    center = verified.host(seed.coefficients[0].data)
    pp_error = float(np.linalg.norm(center - pp["V2"].value))
    sea_error = float(np.linalg.norm(center - sea_pp))
    projected_product_error = float(pp["independent_seed_error"])
    passed = max(pp_error, sea_error, projected_product_error) < 1e-10
    return {"scope": "Ordinary small reusable algebra gate; not an enclosure or arithmetic bound on the final1431 matrices.",
            "shape": [2, 2, 2], "pp_dimension": 120,
            "all_charge_number2_dimension": len(sea["pairs"]),
            "dedicated_vs_shared_pp_Frobenius": pp_error,
            "dedicated_vs_full_sea_Wick_pp_Frobenius": sea_error,
            "shared_pp_vs_independent_normal_product_Frobenius": projected_product_error,
            "dedicated_seed_operator_radius": seed.coefficients[0].radius,
            "passed": passed}


def ordinary_fields(data, s, n):
    """A Taylor candidate search; final residuals always use a full rebuild."""
    ds = s - data["s"]
    def evaluate(jet, order=0):
        answer = xp().zeros_like(jet.value)
        for k in range(order, 4):
            answer += math.factorial(k) / math.factorial(k - order) * ds ** (k - order) * jet.coefficients[k].data
        return answer
    b, e = evaluate(data["B"]), evaluate(data["E"])
    h = b + (n - 1) * e
    hs = evaluate(data["B"], 1) + (n - 1) * evaluate(data["E"], 1)
    hss = evaluate(data["B"], 2) + (n - 1) * evaluate(data["E"], 2)
    en, es = e, evaluate(data["E"], 1)
    t, tp, tpp = (as_float(evaluate(data["t"], k).real) for k in range(3))
    values, vectors = verified.candidate_eigh((h + h.conj().T) * 0.5)
    energy = as_float(values[0])
    u = vectors[:, :1]
    def mean(matrix):
        return as_float((u.conj().T @ matrix @ u).real)
    def response(matrix):
        slopes = vectors[:, 1:].conj().T @ matrix @ u
        return -vectors[:, 1:] @ (slopes / (values[1:, None] - values[0]))
    chis, chin = response(hs), response(en)
    ss = mean(hss) + 2 * as_float((u.conj().T @ hs @ chis).real)
    sn = mean(es) + 2 * as_float((u.conj().T @ hs @ chin).real)
    nn = 2 * as_float((u.conj().T @ en @ chin).real)
    raw_source = evaluate(data["annihilator"])
    delta = evaluate(data["delta"])
    amplitude = raw_source @ (u + delta @ u)
    source_square = as_float(xp().sum(amplitude.real ** 2 + amplitude.imag ** 2))
    residual = np.array([s * t + source_square, n * t + (n * s - 1) * tp + mean(hs)])
    jacobian = np.array([[t + s * tp + sn, nn],
                         [2 * n * tp + (n * s - 1) * tpp + ss, t + s * tp + sn]])
    return {"residual": residual, "jacobian": jacobian, "energy": energy,
            "state": u[:, 0], "responses": (chis[:, 0], chin[:, 0]),
            "computed_gap_diagnostic": as_float(values[1] - values[0])}


def source_profile(data, state):
    delta = data["delta"].coefficients[0]
    root = data["C"].coefficients[0]
    values, certificates = [], []
    for site in range(27):
        raw = rectangular_annihilator(root, site, data["pairs"])
        ruler = Ball(data["s"]) if site == 13 else Ball(1)
        source_map = ruler.scalar_sqrt() * raw
        amplitude = source_amplitude(source_map, delta, state)
        square = squared_norm(amplitude)
        certificates.append(point_bounds(square))
        values.append(as_float(square.data.real))
    total = Ball(0)
    for point in certificates:
        total += Ball(point["center"], point["radius"])
    fraction = Ball(values[13], certificates[13]["radius"]) / total
    corners = [all(c in (0, 2) for c in position) for position in data["coordinates"]]
    return {"scope": "Certified center ground state at fixed candidate fields; root-box envelopes are separately reported when executed.",
            "coordinates": data["coordinates"].astype(int).tolist(),
            "excitation_sources": values, "source_intervals": certificates,
            "sum_sources": point_bounds(total), "f_I": point_bounds(fraction),
            "free_control_sources": [1 / 8 if corner else 0 for corner in corners],
            "free_control_f_I": 0.0}


def source_profile_over_box(profile, box, center, n_width, gap):
    """Extend all27 center source intervals to the entire root rectangle."""
    n_ball = Ball(center["n_center"], n_width)
    h_s = (box["B"].derivative(1) + (n_ball - 1) * box["E"].derivative(1)).norm()
    h_n = box["E"].coefficients[0].norm()
    delta = box["delta"].coefficients[0].norm()
    delta_prime = box["delta"].derivative(1).norm()
    dressing = add_up(1, delta)
    root = box["C"].coefficients[0]
    root_prime = box["C"].derivative(1)
    envelopes, amplitude_lipschitz = [], []
    for site in range(27):
        if site == 13:
            raw = box["annihilator"].coefficients[0].norm()
            raw_prime = box["annihilator"].derivative(1).norm()
        else:
            raw = rectangular_annihilator(root, site, box["pairs"]).norm()
            raw_prime = rectangular_annihilator(root_prime, site, box["pairs"]).norm()
        z = mul_up(raw, dressing)
        z_prime = add_up(mul_up(raw_prime, dressing), mul_up(raw, delta_prime))
        l_s = add_up(z_prime, div_up(mul_up(z, h_s), gap))
        l_n = div_up(mul_up(z, h_n), gap)
        variation = add_up(mul_up(l_s, box["width"]), mul_up(l_n, n_width))
        point = profile["source_intervals"][site]
        amplitude_lower = sqrt_down(max(0.0, point["lower"]))
        amplitude_upper = sqrt_up(max(0.0, point["upper"]))
        lower = max(0.0, sub_down(amplitude_lower, variation))
        upper = add_up(amplitude_upper, variation)
        envelopes.append([lower_product(lower, lower), mul_up(upper, upper)])
        amplitude_lipschitz.append({"s": l_s, "n": l_n, "variation": variation})
    from decimal import localcontext, ROUND_FLOOR, Decimal
    with localcontext() as context:
        context.prec = verified.DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        total_lower = verified.float_down(sum((verified.decimal(x[0]) for x in envelopes), Decimal(0)))
    total_upper = add_up(*(x[1] for x in envelopes))
    if total_lower <= 0:
        raise ArithmeticError("The root-box total excitation source was not resolved.")
    with localcontext() as context:
        context.prec = verified.DECIMAL_DIGITS
        context.rounding = ROUND_FLOOR
        fraction_lower = verified.float_down(verified.decimal(envelopes[13][0]) / verified.decimal(total_upper))
    fraction_upper = div_up(envelopes[13][1], total_lower)
    profile["root_box_source_intervals"] = envelopes
    profile["root_box_sum_sources"] = [total_lower, total_upper]
    profile["root_box_f_I"] = [fraction_lower, fraction_upper]
    profile["root_box_amplitude_lipschitz"] = amplitude_lipschitz
    return profile


def rectangle_in_D0(candidate, widths):
    from decimal import Decimal
    if not all(math.isfinite(float(x)) for x in candidate):
        return False
    return all(verified.decimal(sub_down(float(candidate[i]), widths[i])) >= Decimal("0.999")
               and verified.decimal(add_up(float(candidate[i]), widths[i])) <= Decimal("1.001")
               for i in range(2))


def certified_center(data, n, gap_budget, search=None):
    if search is None:
        search = ordinary_fields(data, data["s"], n)
    s = data["s"]
    h = data["B"].coefficients[0] + (Ball(n) - 1) * data["E"].coefficients[0]
    hs = data["B"].derivative(1) + (Ball(n) - 1) * data["E"].derivative(1)
    hss = data["B"].derivative(2) + (Ball(n) - 1) * data["E"].derivative(2)
    en, es = data["E"].coefficients[0], data["E"].derivative(1)
    ground = validate_eigenpair(h, search["state"], gap_budget["second_level_lower"])
    psi, epsilon = ground["state"], ground["energy"]
    chi_s, check_s = validate_response(h, hs, psi, epsilon, search["responses"][0], gap_budget["ground_gap_lower"])
    chi_n, check_n = validate_response(h, en, psi, epsilon, search["responses"][1], gap_budget["ground_gap_lower"])
    ss = expectation(psi, hss) + 2 * scalar_real((psi.dagger() @ (hs @ chi_s)).map(lambda x: x.reshape(()), gain=1))
    sn = expectation(psi, es) + 2 * scalar_real((psi.dagger() @ (hs @ chi_n)).map(lambda x: x.reshape(()), gain=1))
    nn = 2 * scalar_real((psi.dagger() @ (en @ chi_n)).map(lambda x: x.reshape(()), gain=1))
    t, tp, tpp = (data["t"].derivative(k) for k in range(3))
    center_map = data["annihilator"].coefficients[0]
    amplitude = source_amplitude(center_map, data["delta"].coefficients[0], psi)
    source = squared_norm(amplitude)
    r1 = Ball(s) * t + source
    r2 = Ball(n) * t + (Ball(n) * Ball(s) - 1) * tp + expectation(psi, hs)
    j_common = t + Ball(s) * tp + sn
    jss = 2 * Ball(n) * tp + (Ball(n) * Ball(s) - 1) * tpp + ss
    jacobian = [[j_common, nn], [jss, j_common]]
    return {"R": [scalar_real(r1), scalar_real(r2)], "J": jacobian,
            "H": h, "Hs": hs, "E": en, "ground": ground,
            "chi_s": chi_s, "chi_n": chi_n,
            "response_validation": {"s": check_s, "n": check_n},
            "source_amplitude": amplitude, "source": source,
            "ordinary": search}


def third_residual_bounds(data, n, hn, gap):
    """E17 with whole-ruler-box balls and exact lapse affinity."""
    n_ball = Ball(n, hn)
    nm1 = n_ball - 1
    operators = {
        (0,): data["B"].derivative(1) + nm1 * data["E"].derivative(1),
        (1,): data["E"].coefficients[0],
        (0, 0): data["B"].derivative(2) + nm1 * data["E"].derivative(2),
        (0, 1): data["E"].derivative(1),
        (1, 1): Ball(np.zeros((1, 1))),
        (0, 0, 0): data["B"].derivative(3) + nm1 * data["E"].derivative(3),
        (0, 0, 1): data["E"].derivative(2),
        (0, 1, 1): Ball(np.zeros((1, 1))),
        (1, 1, 1): Ball(np.zeros((1, 1)))}
    norms = {key: value.norm() for key, value in operators.items()}
    def operator_norm(indices):
        return norms[tuple(sorted(indices))]
    def eigen_third(a, b, c):
        first = [operator_norm((x,)) for x in (a, b, c)]
        pair = add_up(mul_up(operator_norm((a, b)), first[2]),
                      mul_up(operator_norm((a, c)), first[1]),
                      mul_up(operator_norm((b, c)), first[0]))
        return add_up(operator_norm((a, b, c)), div_up(2 * pair, gap),
                      div_up(mul_up(12, *first), lower_product(gap, gap)))
    s_ball = Ball(data["s"], data["width"])
    tp, tpp, tppp = (data["t"].derivative(k) for k in (1, 2, 3))
    vacuum_nss = (2 * tp + s_ball * tpp).norm()
    vacuum_sss = (3 * n_ball * tpp + (n_ball * s_ball - 1) * tppp).norm()
    vacuum = {(0, 0, 0): vacuum_sss, (0, 0, 1): vacuum_nss,
              (0, 1, 1): 0.0, (1, 1, 1): 0.0}
    bound = np.zeros((2, 2, 2))
    # R1=energy_n and R2=energy_s; variable columns are(s,n).
    for row, energy_index in enumerate((1, 0)):
        for j in range(2):
            for ell in range(2):
                triple = tuple(sorted((energy_index, j, ell)))
                bound[row, j, ell] = add_up(vacuum[triple], eigen_third(*triple))
    return bound, {"operator_norms": {"".join("s" if x == 0 else "n" for x in key): value
                                        for key, value in norms.items()},
                   "vacuum_nss": vacuum_nss, "vacuum_sss": vacuum_sss,
                   "residual_second_derivative_bounds": bound.tolist()}


def newton_inequalities(center, widths, inverse, second_bounds):
    """Entrywise E18/E19; all inequality arithmetic is rounded outward."""
    beta = []
    defect = np.zeros((2, 2))
    for i in range(2):
        transformed = inverse[i, 0] * center["R"][0] + inverse[i, 1] * center["R"][1]
        beta.append(transformed.norm())
        for j in range(2):
            value = Ball(int(i == j)) - inverse[i, 0] * center["J"][0][j] - inverse[i, 1] * center["J"][1][j]
            defect[i, j] = value.norm()
    z = defect.copy()
    for i in range(2):
        for j in range(2):
            correction = []
            for k in range(2):
                for ell in range(2):
                    correction.append(mul_up(abs(inverse[i, k]), second_bounds[k, j, ell], widths[ell]))
            z[i, j] = add_up(defect[i, j], *correction)
    mapped, contractions, margins = [], [], []
    for i in range(2):
        variation = add_up(*(mul_up(z[i, j], widths[j]) for j in range(2)))
        mapped.append(add_up(beta[i], variation))
        contractions.append(div_up(variation, widths[i]))
        margins.append(sub_down(widths[i], mapped[i]))
    contraction_margin = sub_down(1, max(contractions))
    return {"Y": inverse.tolist(), "beta": beta, "center_defect": defect.tolist(),
            "Z": z.tolist(), "image_radii": mapped,
            "inward_margins": margins, "row_contractions": contractions,
            "contraction_margin": contraction_margin,
            "passed": bool(min(margins) > 0 and contraction_margin > 0)}


def source_box_certificate(center, box, n_width, gap):
    """E16 directly for the108-component sum-of-squares source map."""
    amplitudes = center["source_amplitude"]
    center_square_lower = max(0.0, squared_norm(Ball(amplitudes.data)).bounds()[0])
    center_norm_lower = max(0.0, sub_down(sqrt_down(center_square_lower), amplitudes.radius))
    raw, raw_prime = (box["annihilator"].derivative(k).norm() for k in (0, 1))
    delta, delta_prime = (box["delta"].derivative(k).norm() for k in (0, 1))
    dressing = add_up(1, delta)
    z_norm = mul_up(raw, dressing)
    z_s_norm = add_up(mul_up(raw_prime, dressing), mul_up(raw, delta_prime))
    n_ball = Ball(center["ordinary"].get("n", 1.0), n_width)
    # Use the actual final center n, supplied explicitly by the caller.
    if "n_center" in center:
        n_ball = Ball(center["n_center"], n_width)
    h_s = (box["B"].derivative(1) + (n_ball - 1) * box["E"].derivative(1)).norm()
    h_n = box["E"].coefficients[0].norm()
    l_s = add_up(z_s_norm, div_up(mul_up(z_norm, h_s), gap))
    l_n = div_up(mul_up(z_norm, h_n), gap)
    variation = add_up(mul_up(l_s, box["width"]), mul_up(l_n, n_width))
    margin = sub_down(center_norm_lower, variation)
    lower = lower_product(margin, margin) if margin > 0 else 0.0
    return {"center_amplitude_lower": center_norm_lower,
            "Z_norm_bound": z_norm, "Z_s_norm_bound": z_s_norm,
            "amplitude_L_s": l_s, "amplitude_L_n": l_n,
            "amplitude_variation": variation, "amplitude_margin": margin,
            "source_lower_throughout_box": lower, "passed": margin > 0}


def free_ground(data):
    coordinates, pairs = data["coordinates"], verified.host(data["pairs"])
    u = np.zeros((54, 2), complex)
    for site, point in enumerate(coordinates):
        if all(x in (0, 2) for x in point):
            u[2 * site, 0] = 1
            u[2 * site + 1, 1] = 1
    # Products of the1/sqrt8 one-body amplitudes are exact rationals1/8.
    phi = (u[pairs[:, 0], 0] * u[pairs[:, 1], 1]
           - u[pairs[:, 0], 1] * u[pairs[:, 1], 0]) / 8
    return Ball(phi[:, None])


def mechanism_gate(data, gap_budget):
    """E14/E14a controls at the actual flat-background nonzero seed."""
    phi = free_ground(data)
    v, generator = data["V"].coefficients[0], data["S2"].coefficients[0]
    seed_expectation = expectation(phi, v)
    seed_target = -Ball.rational(1, 80 * 2048)
    derivative_expectation = expectation(phi, data["V"].derivative(1))
    derivative_target = Ball.rational(3, 80 * 8192)
    acted = generator @ phi
    projected = acted - phi * expectation(phi, generator)
    h0 = data["H0"].coefficients[0]
    basis = xp().eye(1431) - phi.data @ phi.data.conj().T
    bordered = h0.data - xp().eye(1431) + phi.data @ phi.data.conj().T
    candidate = verified.candidate_solve(bordered, projected.data)
    candidate = basis @ candidate
    approximate = Ball(candidate)
    residual = h0 @ approximate - approximate - projected
    gap = gap_budget["free_gap"]["lower"]
    orthogonality = (phi.dagger() @ approximate).norm()
    response = Ball(candidate, add_up(div_up(residual.norm(), gap), orthogonality))
    amplitude = -2 * (data["annihilator"].coefficients[0] @ response)
    coefficient = squared_norm(amplitude)
    v_phi = v @ phi
    v_phi_center_lower = sqrt_down(max(0.0, squared_norm(Ball(v_phi.data)).bounds()[0]))
    acted_center_lower = sqrt_down(max(0.0, squared_norm(Ball(acted.data)).bounds()[0]))
    seed_difference = seed_expectation - seed_target
    derivative_difference = derivative_expectation - derivative_target
    seed_identity_passed = seed_difference.bounds()[0] <= 0 <= seed_difference.bounds()[1]
    derivative_identity_passed = derivative_difference.bounds()[0] <= 0 <= derivative_difference.bounds()[1]
    return {"V_norm_upper": v.norm(), "S2_norm_upper": generator.norm(),
            "V_phi_norm_lower": max(0.0, sub_down(v_phi_center_lower, v_phi.radius)),
            "S2_phi_norm_lower": max(0.0, sub_down(acted_center_lower, acted.radius)),
            "seed_ground_expectation": point_bounds(seed_expectation),
            "exact_E14a_seed_target": point_bounds(seed_target),
            "seed_identity_difference": point_bounds(seed_difference),
            "flat_seed_ruler_derivative": point_bounds(derivative_expectation),
            "exact_E14a_derivative_target": point_bounds(derivative_target),
            "derivative_identity_difference": point_bounds(derivative_difference),
            "seed_identity_passed": seed_identity_passed,
            "derivative_identity_passed": derivative_identity_passed,
            "E14_tau2_coefficient": point_bounds(coefficient),
            "free_reduced_resolvent_residual": residual.norm(),
            "nonzero_gate_passed": bool(seed_identity_passed and derivative_identity_passed
                                         and coefficient.bounds()[0] > 0
                                         and seed_expectation.bounds()[1] < 0)}


def affine_boundary_check(data, center, n, profile):
    """Separate full trace/site annihilator sum, compared with total energy."""
    grouped = data["grouped"].coefficients[0]
    boundary = Ball(0)
    vacuum_direct = Ball(0)
    for site in range(27):
        t_site = trace_block(grouped, 2 * site)
        s_site, n_site = (Ball(data["s"]), Ball(n)) if site == 13 else (Ball(1), Ball(1))
        vacuum_direct += n_site * s_site * t_site
        if site != 13:
            source = profile["source_intervals"][site]
            boundary += t_site + Ball(source["center"], source["radius"])
    simplified_vacuum = (Ball(n) * Ball(data["s"]) - 1) * data["t"].coefficients[0]
    total = center["ground"]["energy"] + simplified_vacuum
    direct_total = center["ground"]["energy"] + vacuum_direct
    difference = total - boundary
    affine_remainder = difference - Ball(n) * center["R"][0]
    return {"vacuum_full_trace": point_bounds(vacuum_direct),
            "vacuum_E6": point_bounds(simplified_vacuum),
            "vacuum_full_minus_E6": point_bounds(vacuum_direct - simplified_vacuum),
            "grouped_full_trace": point_bounds(data["trace_all"].coefficients[0]),
            "total_energy": point_bounds(total), "direct_total_energy": point_bounds(direct_total),
            "collar_energy": point_bounds(boundary),
            "total_minus_collar": point_bounds(difference),
            "total_minus_collar_minus_nR1": point_bounds(affine_remainder)}


def archive_candidate(directory, data, center, candidate, profile):
    """Retain the actual normalized pair chart/state for independent B probes."""
    ruler, lapse = np.ones(27), np.ones(27)
    ruler[13], lapse[13] = candidate
    energy_lower, energy_upper = center["ground"]["energy"].bounds()
    np.savez_compressed(directory / "d_candidate.npz",
                        state_center=verified.host(center["ground"]["state"].data).reshape(-1),
                        state_radius=np.array(center["ground"]["state"].radius),
                        pair_ordering=verified.host(data["pairs"]),
                        ruler_profile=ruler, lapse_profile=lapse,
                        candidate=np.asarray(candidate),
                        eigenvalue_interval=np.array([energy_lower, energy_upper]),
                        excitation_source_profile=np.array(profile["excitation_sources"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--decimal-digits", type=int, default=80)
    parser.add_argument("--candidate-steps", type=int, default=8)
    parser.add_argument("--rebuilds", type=int, default=2)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--minimum-radius", type=float, default=1e-12)
    parser.add_argument("--radius-factor", type=float, default=4.0)
    args = parser.parse_args()
    if os.environ.get("QI2_REMOTE_RUN") != "1" or Path(sys.executable).resolve() != Path("<internal>/python").resolve():
        raise SystemExit("Run this numerical job only with the authorized remote qi2 interpreter.")
    if args.rebuilds < 1 or args.candidate_steps < 1 or args.minimum_radius <= 0 or args.radius_factor <= 1:
        raise SystemExit("Use at least one rebuild/step and positive certificate radii with factor>1.")
    if args.backend == "cupy":
        # CuPy's host module import does not create numerical arrays or a
        # CUDA allocation. Complete it before the shared same-process
        # final fresh-device query and sole-PID allocation check.
        import cupy
    runtime = remote_execution(max_threads=8)
    verified.configure(args.backend, certified=True, digits=args.decimal_digits)
    backend = verified.verify_backend()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gap_budget = exact_gap_budget()
    seed_gate = reusable_seed_gate()
    if not seed_gate["passed"]:
        write_result(args.output_dir, {"scope": SCOPE, "parameters": PARAMETERS,
                                      "status": "small independent seed gate failed; no enclosure",
                                      "certificate_passed": False, "reusable_seed_gate": seed_gate,
                                      "backend_verification": backend, "remote_runtime": runtime,
                                      "gap_budget": gap_budget, "unproved_constants": []})
        return
    initial = build(1.0, label="flat control")
    mechanism = mechanism_gate(initial, gap_budget)
    initial_search = ordinary_fields(initial, 1.0, 1.0)
    initial_center = certified_center(initial, 1.0, gap_budget, initial_search)
    alpha = -initial["t"].derivative(1)
    alpha_lower = alpha.bounds()[0]
    alpha_graph = Ball.rational(9, 32) / Ball(3).scalar_sqrt()
    if alpha_lower <= 0:
        raise ArithmeticError("The actual grouped vacuum coercivity was not resolved.")
    yg = np.array([[-1 / as_float(alpha.data.real), 0],
                   [2 / as_float(alpha.data.real), -1 / as_float(alpha.data.real)]])
    candidate = np.ones(2) - yg @ initial_search["residual"]
    history = [{"s": 1.0, "n": 1.0, "R": initial_search["residual"].tolist(),
                "J": initial_search["jacobian"].tolist(), "kind": "actual flat rebuild"}]
    candidate_gate = {"V_action_resolved": mechanism["V_phi_norm_lower"] > 0,
                      "S2_action_resolved": mechanism["S2_phi_norm_lower"] > 0,
                      "mechanism_and_exact_seed_identities": mechanism["nonzero_gate_passed"],
                      "load1_interior_source_resolved": initial_center["source"].bounds()[0] > 0}
    candidate_gate["passed"] = all(candidate_gate.values())
    if not candidate_gate["passed"]:
        result = {"scope": SCOPE, "parameters": PARAMETERS,
                  "status": "predeclared load1 nontrivial-source candidate gate unresolved; no enclosure",
                  "gap_budget": gap_budget, "mechanism": mechanism,
                  "candidate_gate": candidate_gate,
                  "initial_residual": [point_bounds(x) for x in initial_center["R"]],
                  "initial_source": point_bounds(initial_center["source"]),
                  "arithmetic": verified.EVENTS,
                  "unproved_constants": []}
        write_result(args.output_dir, result)
        return
    current = initial
    for rebuild in range(args.rebuilds):
        for iteration in range(args.candidate_steps):
            if not rectangle_in_D0(candidate, (0, 0)):
                result = {"scope": SCOPE, "parameters": PARAMETERS,
                          "status": "candidate left declared D0; no enclosure",
                          "candidate": candidate.tolist(), "history": history,
                          "gap_budget": gap_budget, "mechanism": mechanism,
                          "unproved_constants": []}
                write_result(args.output_dir, result)
                return
            search = ordinary_fields(current, *candidate)
            history.append({"s": float(candidate[0]), "n": float(candidate[1]),
                            "R": search["residual"].tolist(),
                            "J": search["jacobian"].tolist(),
                            "kind": "third-order Taylor candidate search"})
            step = np.linalg.solve(search["jacobian"], search["residual"])
            candidate -= step
            if not rectangle_in_D0(candidate, (0, 0)):
                result = {"scope": SCOPE, "parameters": PARAMETERS,
                          "status": "Newton step left declared D0 before rebuild; no enclosure",
                          "candidate": candidate.tolist(), "history": history,
                          "gap_budget": gap_budget, "mechanism": mechanism,
                          "unproved_constants": []}
                write_result(args.output_dir, result)
                return
            print(f"D candidate step: s={candidate[0]:.17g}, n={candidate[1]:.17g}, |step|={np.max(np.abs(step)):.3g}.", flush=True)
            if np.max(np.abs(step)) <= 3e-15:
                break
        current = build(float(candidate[0]), label=f"actual candidate rebuild{rebuild + 1}")
        search = ordinary_fields(current, *candidate)
        center = certified_center(current, float(candidate[1]), gap_budget, search)
        history.append({"s": float(candidate[0]), "n": float(candidate[1]),
                        "R": search["residual"].tolist(), "J": search["jacobian"].tolist(),
                        "kind": "actual final-load rebuild"})
    center["n_center"] = float(candidate[1])
    profile = source_profile(current, center["ground"]["state"])
    archive_candidate(args.output_dir, current, center, candidate, profile)
    boundary = affine_boundary_check(current, center, float(candidate[1]), profile)
    inverse = np.linalg.inv(np.array([[as_float(value.data.real) for value in row] for row in center["J"]]))
    beta = newton_inequalities(center, np.ones(2), inverse, np.zeros((2, 2, 2)))["beta"]
    widths = np.array([max(args.minimum_radius, mul_up(args.radius_factor, value)) for value in beta])
    result = {"scope": SCOPE, "parameters": PARAMETERS,
              "status": "candidate only; enclosure not executed", "certificate_passed": False,
              "candidate": {"s": float(candidate[0]), "n": float(candidate[1]),
                            "energy": point_bounds(center["ground"]["energy"])},
              "gap_budget": gap_budget, "actual_alpha": point_bounds(alpha),
              "independent_alpha_graph_lower": point_bounds(alpha_graph),
              "alpha_margin_above_graph_bound": sub_down(alpha_lower, alpha_graph.bounds()[1]),
              "initial_residual": [point_bounds(x) for x in initial_center["R"]],
              "initial_J": [[point_bounds(x) for x in row] for row in initial_center["J"]],
              "initial_source": point_bounds(initial_center["source"]),
              "mechanism": mechanism, "center_residual": [point_bounds(x) for x in center["R"]],
              "candidate_gate": candidate_gate,
              "reusable_seed_gate": seed_gate,
              "center_J": [[point_bounds(x) for x in row] for row in center["J"]],
              "eigenpair_validation": {k: value for k, value in center["ground"].items()
                                       if k not in ("state", "energy", "rayleigh")},
              "response_validation": center["response_validation"],
              "center_source": point_bounds(center["source"]),
              "source_profile": profile, "affine_boundary_check": boundary,
              "candidate_state_archive": "d_candidate.npz",
              "candidate_history": history,
              "arithmetic": {"center_significand_bits": 53,
                             "scalar_decimal_digits": args.decimal_digits,
                             "roundoff_model": "explicit sequential gamma_(8k+4) complex binary64 products; directed scalar bounds; underflow allowance2^-1022",
                             "implementation_events": dict(verified.EVENTS)},
              "backend_verification": backend, "remote_runtime": runtime,
              "unproved_constants": [],
              "arithmetic_hypotheses": ["The deterministic CUDA double or NumPy real ufuncs implement IEEE elementary operations; CUDA fast math/FMA are disabled and CuPy's added FTZ is covered by the2^-1022 absolute underflow allowance.",
                                        "Explicit NumPy longdouble ufunc accumulation uses its reported>=64-bit significand; each dot has8k elementary operations."]}
    # Save the scoped center/source deliverable before the more expensive box.
    write_result(args.output_dir, result)
    if args.candidate_only:
        return
    contained = rectangle_in_D0(candidate, widths)
    if not contained:
        result["status"] = "proposed rectangle leaves declared D0; no enclosure"
        result["proposed_radii"] = widths.tolist()
        write_result(args.output_dir, result)
        return
    box = build(float(candidate[0]), float(widths[0]), label="whole root rectangle")
    second_bounds, derivative_details = third_residual_bounds(box, float(candidate[1]), float(widths[1]), gap_budget["ground_gap_lower"])
    newton = newton_inequalities(center, widths, inverse, second_bounds)
    source = source_box_certificate(center, box, float(widths[1]), gap_budget["ground_gap_lower"])
    result["source_profile"] = source_profile_over_box(profile, box, center, float(widths[1]), gap_budget["ground_gap_lower"])
    energy_center = center["ground"]["energy"]
    h_s_bound = derivative_details["operator_norms"]["s"]
    h_n_bound = derivative_details["operator_norms"]["n"]
    energy_variation = add_up(mul_up(h_s_bound, widths[0]), mul_up(h_n_bound, widths[1]))
    energy_low, energy_high = energy_center.bounds()
    passed = bool(backend["passed"] and seed_gate["passed"] and candidate_gate["passed"]
                  and newton["passed"] and source["passed"])
    result.update({"status": "strict finite root/source enclosure" if passed else "rigorous sufficient enclosure inequalities did not close",
                   "certificate_passed": passed,
                   "root_rectangle": {"s": [sub_down(float(candidate[0]), widths[0]), add_up(float(candidate[0]), widths[0])],
                                      "n": [sub_down(float(candidate[1]), widths[1]), add_up(float(candidate[1]), widths[1])],
                                      "radii": widths.tolist(), "contained_in_D0": contained},
                   "energy_throughout_root_rectangle": [sub_down(energy_low, energy_variation), add_up(energy_high, energy_variation)],
                   "newton_certificate": newton, "source_certificate": source,
                   "whole_box_derivatives": derivative_details})
    result["arithmetic"]["implementation_events"] = dict(verified.EVENTS)
    write_result(args.output_dir, result)


def write_result(directory, result):
    (directory / "d_enclosure.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [result["scope"], result["status"]]
    if "candidate" in result:
        lines.append(f"candidate: {result['candidate']}")
    if "root_rectangle" in result:
        lines.append(f"root rectangle: {result['root_rectangle']}")
        lines.append(f"Newton inward margins: {result['newton_certificate']['inward_margins']}")
        lines.append(f"contraction margin: {result['newton_certificate']['contraction_margin']}")
        lines.append(f"positive interior source lower: {result['source_certificate']['source_lower_throughout_box']}")
    if "source_profile" in result:
        lines.append(f"f_I: {result['source_profile']['f_I']}")
    lines.append(f"unproved arithmetic constants: {result.get('unproved_constants', [])}")
    lines.extend(result.get("arithmetic_hypotheses", []))
    summary = "\n".join(lines) + "\n"
    (directory / "d_enclosure.txt").write_text(summary)
    print(summary, end="", flush=True)


if __name__ == "__main__":
    main()
