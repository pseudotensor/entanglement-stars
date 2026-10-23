#!/usr/bin/env python3
"""Prepared original-H_* Ramsey control; separate from frozen ground code.

Source: strand_E_clock_spec.tex and its completed analytical review, plus
strand_E_bound_clock.tex.  The required prior ground certificate supplies
the same exact mu2/gamma and point identity.  This evaluator reconstructs
its own certified reference g/e, retains the exact free continuum, and
does not change the coupling, preparation, detector or source.  Its own
author/independent actual-code audits and Ops admission precede execution.

Only the directed primitives and ground-stage helpers of e_bound_clock
are reused.  Its main/run, draft clock dispatcher, source jets, Ramsey
and excited-state routines are never called.  That file remains frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import time

import e_bound_clock as base
from remote_runtime import remote_execution


F = Fraction
a, c, upper, lower = base.a, base.c, base.upper, base.lower
abs_upper, positive, encoded = base.abs_upper, base.positive, base.encoded
Unresolved, EngineeringCap, AnalyticDomain = base.Unresolved, base.EngineeringCap, base.AnalyticDomain
SOURCE = "solve/strand_E_clock_spec.tex"
SOURCE_REVIEW = "solve/review_E_clock_spec.tex"
CODE_REVIEW = "solve/review_E_ramsey_code.tex"


class CertificateConflict(Unresolved):
    """Contradictory certified source inputs/enclosures; never a physics result."""


def exact_endpoint(record):
    if set(record) != {"mantissa", "exponent2"} or not isinstance(record["mantissa"], str) or not isinstance(record["exponent2"], int):
        raise ValueError("A certificate endpoint requires exact mantissa-string/exponent2 fields")
    return int(record["mantissa"]), record["exponent2"]


def check_ball_record(record):
    if not isinstance(record, dict) or "lower_dyadic" not in record or "upper_dyadic" not in record:
        raise ValueError("A certificate value lacks its exact directed endpoints")
    exact_endpoint(record["lower_dyadic"])
    exact_endpoint(record["upper_dyadic"])


def decode_ball(record):
    # Constructor dyadics and outward hulls retain all input widths.  The
    # display string is never an input and never stripped of a radius.
    lo = base.arb(exact_endpoint(record["lower_dyadic"]))
    hi = base.arb(exact_endpoint(record["upper_dyadic"]))
    if not lo.is_finite() or not hi.is_finite() or lower(lo) > upper(hi):
        raise Unresolved("Ground certificate endpoint ordering/finiteness unresolved")
    if record["lower_dyadic"] == record["upper_dyadic"]:
        if not lo.is_exact():
            raise Unresolved("An exact input dyadic was not preserved by the backend")
        return lo
    return base.interval_hull(lo, hi)


def read_ground_contract(path):
    """Host-only input shape/identity checks; no ball math before admission."""
    data = json.loads(path.read_text())
    if data.get("stage") != "ground" or data.get("status") != "certified_ground_preparation":
        raise ValueError("The prerequisite must be a certified ground-only result")
    if data.get("ground_preparation_certified") is not True:
        raise ValueError("The prerequisite ground certification flag is unset")
    export = data.get("ground_export", {})
    if export.get("certified_after_measure_kernel_and_next_level_controls") is not True:
        raise ValueError("The prerequisite ground measure/kernel/refinement controls are unset")
    params = data.get("fixed_parameters", {})
    expected = {"m_over_mP": "1/1000", "a_over_lP": "1/1000", "alpha": "1/1000000",
                "Lambda_over_m": "1", "ell_times_m": "1", "gB": "0",
                "sector": "even standard-spinor singlet, half-period character+, K=0, positive-charge two excitations",
                "law": "H_t=exp(-gamma*T_t)*(2*epsilon_t)*exp(-gamma*T_t)",
                "gamma_rule": "log(4*Emax)/(2*mu2), same exact trial compression at every refinement"}
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError("Ground and Ramsey physical point/sector/law/coupling rule differ")
    if export.get("radial_chart") != "L2(nu_alpha), alpha=1/1000000; implemented chart L2(dr)":
        raise ValueError("Ground and Ramsey measure charts differ")
    for key in ("mu2", "gamma", "gamma_midpoint_exact", "gamma_midpoint_mismatch_upper", "E_g", "strict_original_ground_positive_bound"):
        check_ball_record(data.get(key))
    check_ball_record(export.get("full_embedding_phase_aligned_norm_error_upper"))
    radial = data.get("ground_radial_function", {})
    for key in ("original_full_operator_residual_upper", "original_spectral_separation_lower",
                "phase_aligned_vector_error_upper", "original_declared_sign_margin_lower"):
        check_ball_record(radial.get(key))
    if data.get("same_exact_coupling_rule_across_all_refinements") is not True:
        raise ValueError("Prerequisite does not certify the same exact coupling across refinements")
    return data


@dataclass
class SourceJet:
    """Order-four ordinary Taylor coefficients; no commuting assumption."""
    terms: list

    def coerce(self, value):
        if isinstance(value, SourceJet):
            return value
        scalar = value if isinstance(value, (base.arb, base.acb)) else a(value)
        return SourceJet([scalar] + [a(0)] * (len(self.terms) - 1))

    def __add__(self, other):
        other = self.coerce(other)
        return SourceJet([x + y for x, y in zip(self.terms, other.terms)])

    __radd__ = __add__

    def __neg__(self):
        return SourceJet([-value for value in self.terms])

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return SourceJet([sum((self.terms[k] * other.terms[n - k] for k in range(n + 1)), a(0))
                          for n in range(len(self.terms))])

    __rmul__ = __mul__

    def reciprocal(self):
        zero = self.terms[0]
        if not zero.is_finite() or zero.contains(0):
            raise AnalyticDomain("Source-jet reciprocal is not analytic on the requested ball")
        result = [1 / zero]
        for n in range(1, len(self.terms)):
            result.append(-sum((self.terms[k] * result[n - k] for k in range(1, n + 1)), a(0)) / zero)
        return SourceJet(result)

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) * self.reciprocal()

    def sqrt(self, analytic=False):
        value = self.terms[0]
        if isinstance(value, base.acb):
            root = base.guarded_root(value, analytic)
        else:
            root = positive(value, "Real source-jet square root").sqrt()
        result = [root]
        for n in range(1, len(self.terms)):
            result.append((self.terms[n] - sum((result[k] * result[n - k] for k in range(1, n)), a(0))) / (2 * root))
        return SourceJet(result)

    def exp(self):
        result = [self.terms[0].exp()]
        for n in range(1, len(self.terms)):
            result.append(sum((k * self.terms[k] * result[n - k] for k in range(1, n + 1)), a(0)) / n)
        return SourceJet(result)


def leg_jets(radius, source_t=0, degree=4, analytic=False):
    u0 = (-2 * a(source_t)).exp() * radius**2
    u = SourceJet([u0 * (-2)**j / math.factorial(j) for j in range(degree + 1)])
    epsilon, regulator = (1 + u).sqrt(analytic), (-u).exp()
    f = regulator * (1 + 1 / epsilon) / 2
    j = regulator * (1 - 1 / epsilon) / 2
    return epsilon, f, j


def pair_kernel_jets(r, s, source_t=0, degree=4):
    er, fr, jr = leg_jets(r, source_t, degree)
    es, fs, js = leg_jets(s, source_t, degree)
    scale = (-3 * a(source_t)).exp()
    prefactor = SourceJet([scale * (-3)**j / math.factorial(j) for j in range(degree + 1)])
    return prefactor * (fr * fs + jr * js) / (er + es)


class RamseyDiscretization(base.Discretization):
    def kernel_matrices(self, source_t=0, degree=4):
        if degree == 0:
            return super().kernel_matrices(source_t, degree=0)
        if degree != 4:
            raise ValueError("Only the specified reference order-four source jet is constructed")
        nodes, count = self.node_data, len(self.node_data)
        self.work.reserve_matrix_entries(5 * count**2 + 4 * self.rank * count + 8 * self.rank**2,
                                         "Ramsey positive interaction/source jets")
        kernels = [base.arb_mat(count, count) for _ in range(5)]
        for row, (_, r, _, _, qr) in enumerate(nodes):
            for column in range(row, count):
                self.work.check(1)
                _, s, _, _, qs = nodes[column]
                values = pair_kernel_jets(r, s, source_t).terms
                for k, value in enumerate(values):
                    kernels[k][row, column] = kernels[k][column, row] = qr * qs * math.factorial(k) * value
        embedding = base.arb_mat(self.rank, count)
        for column, (panel, _, weight, values, _) in enumerate(nodes):
            for n, value in enumerate(values):
                embedding[panel * self.order + n, column] = weight * value
        result = []
        for kernel in kernels:
            self.work.check(self.rank**2)
            matrix = embedding * kernel * base.transpose(embedding)
            for row in range(self.rank):
                for column in range(row + 1, self.rank):
                    lo = max(lower(matrix[row, column]), lower(matrix[column, row]))
                    hi = min(upper(matrix[row, column]), upper(matrix[column, row]))
                    matrix[row, column] = matrix[column, row] = base.interval_hull(lo, hi)
            self.matrix_radii.append(base.matrix_enclosure_radius(matrix))
            result.append(matrix)
        return result


def minus_toeplitz_exponential(matrices, gamma_midpoint, work, tolerance):
    rank = matrices[0].nrows()
    work.reserve_matrix_entries(8 * (5 * rank)**2, "Order-four minus-Toeplitz completion jet")
    block = base.arb_mat(5 * rank, 5 * rank)
    for row in range(5):
        for column in range(row, 5):
            matrix = -gamma_midpoint * matrices[column - row] / math.factorial(column - row)
            for i in range(rank):
                for j in range(rank):
                    block[row * rank + i, column * rank + j] = matrix[i, j]
    # The audited general expm1 helper supplies a signed-matrix Taylor norm
    # remainder, every squaring error and outward arithmetic.  Adding I
    # recovers the exponential of MINUS the entire Toeplitz matrix.
    difference, certificate = base.certified_matrix_expm1(block, work, tolerance)
    exponential = base.identity(5 * rank) + difference
    derivatives = []
    for k in range(5):
        matrix = base.arb_mat(rank, rank)
        for i in range(rank):
            for j in range(rank):
                matrix[i, j] = math.factorial(k) * exponential[i, k * rank + j]
        derivatives.append(matrix)
    return derivatives, {"entire_block_has_minus_sign": True, "block_count": 5,
                         "ordinary_derivatives_recovered_by_factorials": True, **certificate}


class ResolventModel(base.FiniteModel):
    """Exact free multiplication and complete analytic Woodbury enclosure."""
    def __init__(self, discretization, matrices, gamma_midpoint, source_t=0, lapse=1):
        super().__init__(discretization, matrices[:1], gamma_midpoint, degree=0, source_t=source_t, lapse=lapse)
        self.matrices = matrices

    def resolvent_bilinear(self, z, left, right, analytic=False):
        if not left.cutoff_supported or not right.cutoff_supported:
            raise Unresolved("No exterior free-block representation supplied for Dunford vectors")
        d, n = self.d, self.d.rank
        g0 = d.free_gram(z, self.source_t, self.lapse, analytic=analytic)
        g = base.acb_mat(2 * n, 2 * n)
        for i in range(n):
            for j in range(n):
                unit = int(i == j)
                g[i, j] = g0[i, j]
                g[i, n + j] = g[n + i, j] = unit + z * g0[i, j]
                g[n + i, n + j] = self.b[i, j] + z * unit + z**2 * g0[i, j]
        cc = base.as_complex_matrix(self.coupling_matrix)
        inverse, residual = base.verified_inverse(base.identity(2 * n, True) + cc * g, d.work, analytic=analytic)

        def free_bilinear(u, v):
            total = c(0)
            for p, panel in enumerate(d.panels):
                if u.support_panels is not None and p not in u.support_panels:
                    continue
                if v.support_panels is not None and p not in v.support_panels:
                    continue
                def integrand(r, radial_analytic):
                    denominator = base.free_multiplier(r, self.source_t, self.lapse, radial_analytic) - z
                    if not denominator.is_finite() or denominator.contains(0):
                        raise AnalyticDomain("Complete exact free denominator not certified on cell")
                    return u(p, r, radial_analytic) * v(p, r, radial_analytic) / denominator
                total += d.integrals.integrate(integrand, c(panel.left), c(panel.right), "Ramsey complete free bilinear")
            return total
        value = free_bilinear(left, right)
        pl, pr = self.projection(left), self.projection(right)
        row, column = base.acb_mat(1, 2 * n), base.acb_mat(2 * n, 1)
        for i in range(n):
            li, ri = free_bilinear(left, d.basis(i)), free_bilinear(d.basis(i), right)
            row[0, i], column[i, 0] = li, ri
            row[0, n + i], column[n + i, 0] = pl[i] + z * li, pr[i] + z * ri
        return value - (row * inverse * cc * column)[0, 0], residual


class JetModel(ResolventModel):
    def __init__(self, discretization, matrices, gamma_midpoint):
        if len(matrices) != 5:
            raise ValueError("The complete order-four source jet is required")
        super().__init__(discretization, matrices, gamma_midpoint)
        self.exponential_derivatives, self.jet_exponential_certificate = minus_toeplitz_exponential(
            matrices, gamma_midpoint, self.d.work, self.d.tolerance)

    def exponent_action(self, function, derivative):
        projection = self.projection(function)
        matrix = self.z if derivative == 0 else self.exponential_derivatives[derivative]
        coefficients = [sum((matrix[i, j] * projection[j] for j in range(self.d.rank)), a(0))
                        for i in range(self.d.rank)]
        correction = self.d.polynomial(coefficients, "Ramsey exp^(" + str(derivative) + ")")
        return function + correction if derivative == 0 else correction

    def h_action(self, function, derivative):
        def evaluate(panel, r, analytic):
            epsilon, _, _ = leg_jets(r, self.source_t, degree=derivative, analytic=analytic)
            scalar = 2 * c(self.lapse) * math.factorial(derivative) * epsilon.terms[derivative]
            return scalar * function(panel, r, analytic)
        return base.RadialFunction(evaluate, "Ramsey h^(" + str(derivative) + ")" + function.label,
                                   function.support_panels)

    def apply_derivative(self, function, derivative):
        if not 0 <= derivative <= 4:
            raise ValueError("Source derivative exceeds the frozen order-four jet")
        result = None
        for i in range(derivative + 1):
            for l in range(derivative - i + 1):
                k = derivative - i - l
                coefficient = math.factorial(derivative) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                value = self.exponent_action(self.h_action(self.exponent_action(function, k), l), i).scaled(coefficient)
                result = value if result is None else result + value
        return result

    def mixed_derivative(self, function, derivative):
        # N_lambda=1-lambda: N^(0) H^(j)-j H^(j-1), without dropping lapse jets.
        value = self.apply_derivative(function, derivative).scaled(1 - a(self.source_t))
        return value if derivative == 0 else value - self.apply_derivative(function, derivative - 1).scaled(derivative)


def completion_error_contract(d, gamma, gamma_midpoint):
    """All four operator jets, contractive Bell words, full lattice h norms."""
    tc, eta = a(base.T0 + base.ETA), a(base.ETA)
    q = positive((-2 * tc).exp() * (2 * eta).cos(), "Ramsey complex Gaussian rate")
    def gaussian_integrand(r, analytic):
        qa = base.absorption(r, d.coefficients, analytic)
        return qa**2 * (-2 * q * r**2).exp()
    integral = sum((d.integrals.real(gaussian_integrand, panel.left, panel.right, "Ramsey complex T norm")
                    for panel in d.panels), a(0))
    full_mass = 1 / (8 * a(base.ALPHA)**3)
    tnorm = upper((3 * tc).exp() * (integral + full_mass * (-2 * q * d.cutoff**2).exp()))
    terr = upper(d.errors["tail_projection"] + d.errors["quadrature"])
    tqnorm = upper(tnorm + terr)
    gamma_error = upper(abs(gamma - gamma_midpoint))
    jnorm = max(upper(upper(gamma) * tnorm), upper(gamma_midpoint * tqnorm))
    p = [a(0)] + [upper(math.factorial(j) * eta**(-j) * jnorm) for j in range(1, 5)]
    bell = [a(1)]
    for j in range(1, 5):
        bell.append(upper(sum((math.comb(j - 1, k - 1) * p[k] * bell[j - k] for k in range(1, j + 1)), a(0))))
    full_h_complex = upper(2 * (1 + 3 / a(base.ALPHA)**2 * (2 * tc).exp()).sqrt())
    full_h_real = upper(2 * (1 + 3 / a(base.ALPHA)**2 * (2 * a(base.T0)).exp()).sqrt())
    h = [full_h_real] + [upper(math.factorial(j) * eta**(-j) * full_h_complex) for j in range(1, 5)]
    category_errors, j_errors, exp_errors = {}, {}, {}
    for name in ("tail_projection", "quadrature", "coupling"):
        dj = [upper(math.factorial(j) * eta**(-j)
                     * (gamma_error * tqnorm if name == "coupling" else upper(gamma) * d.errors[name]))
              for j in range(5)]
        ej = [upper(dj[0] * bell[j] + sum((math.comb(j, k) * bell[j - k] * dj[k] for k in range(1, j + 1)), a(0)))
              for j in range(5)]
        hj = []
        for j in range(5):
            value = a(0)
            for i in range(j + 1):
                for l in range(j - i + 1):
                    k = j - i - l
                    multiplier = math.factorial(j) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                    value += multiplier * h[l] * (ej[i] * bell[k] + bell[i] * ej[k])
            hj.append(upper(value))
        category_errors[name], j_errors[name], exp_errors[name] = hj, dj, ej
    hnorm = []
    for j in range(5):
        value = a(0)
        for i in range(j + 1):
            for l in range(j - i + 1):
                k = j - i - l
                multiplier = math.factorial(j) // (math.factorial(i) * math.factorial(l) * math.factorial(k))
                value += multiplier * bell[i] * h[l] * bell[k]
        hnorm.append(upper(value))
    hj_error = [upper(sum((category_errors[name][j] for name in category_errors), a(0))) for j in range(5)]
    # Mixed path differentiates N_lambda=1-lambda as well as H_t.
    mixed_norm = [upper((1 + a(base.T0)) * hnorm[j] + (j * hnorm[j - 1] if j else 0)) for j in range(5)]
    mixed_error = [upper((1 + a(base.T0)) * hj_error[j] + (j * hj_error[j - 1] if j else 0)) for j in range(5)]
    return {"T_complex_norm": tnorm, "TQ_complex_norm": tqnorm, "T_errors_total": terr,
            "gamma_mismatch": gamma_error, "J_error_categories": j_errors, "exp_error_categories": exp_errors,
            "Bell": bell, "p_j": p, "full_free_derivative_norms": h,
            "categories": category_errors, "H_errors": hj_error, "H_norms": hnorm,
            "mixed_norms": mixed_norm, "mixed_errors": mixed_error}


def serialize_error_contract(contract):
    result = {}
    for key, value in contract.items():
        if isinstance(value, dict):
            result[key] = {name: [encoded(x) for x in xs] for name, xs in value.items()}
        elif isinstance(value, list):
            result[key] = [encoded(x) for x in value]
        else:
            result[key] = encoded(value)
    result["full_fixed_lattice_free_norms_retained"] = True
    result["mixed_source_lapse_jets_retained"] = True
    return result


def original_levels(intervals, guard, contract, gamma, matrix):
    error = contract["H_errors"][0]
    norm_t = upper(base.norm_bound(matrix) + contract["T_errors_total"])
    strict_positive = positive(2 * (-2 * upper(gamma) * norm_t).exp(), "Original ground strict positive lower bound")
    ground_lo = max(lower(strict_positive), lower(lower(intervals[0]) - error))
    ground_hi = min(upper(upper(intervals[0]) + error), a(F(1, 4)))
    if ground_lo > ground_hi:
        raise Unresolved("Original ground interval and fixed binding theorem are inconsistent")
    eg = base.interval_hull(ground_lo, ground_hi)
    ee = base.widen(intervals[1], error)
    gap = positive(ee - eg, "Full original-H simple doublet gap")
    higher_lower = lower(lower(guard) - error)
    positive(higher_lower - max(upper(ee), upper(intervals[1])), "Original excited separation from higher levels")
    positive(2 - max(upper(ee), upper(intervals[1])), "Original breakup margin")
    return [eg, ee], gap, higher_lower, strict_positive


def normalize_excited(model, g, e, eq, original, higher_lower, contract, metadata):
    """Actual normalized eigenfamily, mutual Gram, full residual and sign."""
    d = model.d
    cross = d.inner(g, e)
    if not cross.contains(0):
        raise Unresolved("Distinct selected H_Q eigenfamilies fail exact orthogonality enclosure")
    changed = e - g.scaled(cross)
    norm2 = positive(d.inner(changed, changed), "Excited Gram--Schmidt norm")
    changed = changed.scaled(1 / norm2.sqrt())
    change_upper = d.norm(changed - e)
    norm_after, cross_after = d.inner(changed, changed), d.inner(g, changed)
    if not norm_after.contains(1) or not cross_after.contains(0):
        raise Unresolved("Certified continuous mutual orthonormalization unresolved")
    reference_lo = min(lower(eq[1]), lower(original[1]))
    reference_hi = max(upper(eq[1]), upper(original[1]))
    separation = positive(min(lower(reference_lo - upper(original[0])),
                              lower(higher_lower - reference_hi), lower(2 - reference_hi)),
                          "Excited residual-reference distance to every other original-H level")
    residual_function = model.apply_derivative(changed, 0) - changed.scaled(eq[1])
    numerical, norm_certificate = d.norm(residual_function, certificate=True)
    components = {name: values[0] for name, values in contract["categories"].items()}
    components.update({"joint_scalar_quadrature_allocation": numerical, "joint_matrix_arithmetic_allocation": numerical,
                       "spectral_energy_enclosure": upper(upper(eq[1]) - lower(eq[1]))})
    full_residual = upper(sum(components.values(), a(0)))
    ratio = upper(full_residual / separation)
    if not ratio < a(F(1, 2)):
        raise Unresolved("Excited full residual/separation angle test unresolved")
    vector_error = upper(2 * ratio + change_upper)
    trial_norm = upper(positive(d.inner(d.trial(0), d.trial(0)) + d.trial_tail(0), "Full reference f norm").sqrt())
    overlap = d.inner(d.trial(0), changed)
    margin = positive(lower(overlap) - trial_norm * vector_error, "First excited f-projection and declared sign margin")
    metadata.update({"continuous_Gram_cross_before": encoded(cross), "continuous_Gram_cross_after": encoded(cross_after),
                     "continuous_norm_after": encoded(norm_after), "orthonormalization_vector_change_upper": encoded(change_upper),
                     "full_original_operator_residual_upper": encoded(full_residual),
                     "residual_reference_energy_HQ": encoded(eq[1]), "radial_residual_reference_separation_lower": encoded(separation),
                     "phase_aligned_vector_error_upper": encoded(vector_error), "f_projection_positive_sign_margin": encoded(margin),
                     "joint_numerical_residual_certificate": norm_certificate,
                     "residual_error_allocation": {name: encoded(value) for name, value in components.items()},
                     "joint_residual_allocated_twice_conservatively": True,
                     "multiple_excited_cluster_or_skipped_sequence_member": False})
    return changed, vector_error, metadata


def f_time(x, tau):
    x = x if isinstance(x, base.acb) else c(x)
    half_argument = x * c(tau) / 2
    # Official python-flint0.9.0 acb.sinc is the entire sin(z)/z.
    return c(tau) * (-c(0, 1) * half_argument).exp() * half_argument.sinc()


def j_time(x, y, tau, work, tolerance):
    difference = x - y
    if difference.is_finite() and not difference.contains(0):
        return (f_time(y, tau) - f_time(x, tau)) / (c(0, 1) * difference)
    duration = upper(tau)
    maximum = upper(duration * max(abs_upper(x), abs_upper(y)))
    if not maximum.is_finite() or maximum > base.ctx.prec:
        raise AnalyticDomain("Entire J coincidence cell requires subdivision beyond its series work range")
    result, degree, remainder = c(0), -1, a(1)
    while degree < 0 or not upper(remainder) < lower(a(tolerance)):
        degree += 1
        work.check(degree + 1)
        polynomial = sum((x**k * y**(degree - k) for k in range(degree + 1)), c(0))
        result += c(tau)**2 * (-c(0, 1) * c(tau))**degree * polynomial / math.factorial(degree + 2)
        # Taylor-expand the integrand on the whole triangle, whose area is
        # tau^2/2 and whose |x u+y v| is at most max(|x|,|y|)*tau.
        remainder = upper(duration**2 / 2 * maximum.exp() * maximum**(degree + 1) / math.factorial(degree + 1))
        if degree > 16 * base.ctx.prec:
            raise Unresolved("Entire J directed Taylor remainder did not reach its cap")
    return base.widen(result, remainder)


def minimum_interval(x, y):
    return base.interval_hull(min(lower(x), lower(y)), min(upper(x), upper(y)))


def dunford(model, left, right, entire, tau, source_radius=0):
    if not left.cutoff_supported or not right.cutoff_supported:
        raise Unresolved("Dunford exterior free block is required but not represented")
    d = model.d
    height = positive(minimum_interval(a(F(1, 4)), 1 / positive(tau, "Reference duration")),
                      "Full Dunford contour height")
    if source_radius:
        bound = (1 + a(source_radius)) * 2 * (1 + (2 * a(source_radius)).exp() * d.cutoff**2).sqrt()
    else:
        bound = 2 * a(1 + d.cutoff**2).sqrt()
    right_endpoint = bound + 1
    vertices = [base.acb(-1, -height), base.acb(right_endpoint, -height),
                base.acb(right_endpoint, height), base.acb(-1, height)]
    values, radii, inverse_residual_max = [], [], a(0)
    for segment in range(4):
        def integrand(z, analytic):
            nonlocal inverse_residual_max
            # Exact free integrals and an inverse proof over this WHOLE z
            # ball precede evaluation.  Merely sampling an inverse is never
            # accepted as an analytic-cell certificate.
            resolvent, residual = model.resolvent_bilinear(z, left, right, analytic=analytic)
            inverse_residual_max = max(inverse_residual_max, upper(residual))
            value = entire(z) * resolvent
            if not value.is_finite():
                raise AnalyticDomain("Complete four-segment Dunford integrand not finite")
            return value
        value = d.integrals.integrate(integrand, vertices[segment], vertices[(segment + 1) % 4],
                                      "Ramsey Dunford segment " + str(segment))
        values.append(value)
        radii.append(upper(value.rad()) / (2 * lower(base.arb.pi())))
    result = -sum(values, c(0)) / (2 * base.acb.pi() * c(0, 1))
    certificate = {"CCW_segment_enclosures": [encoded(value) for value in values],
                   "segment_radius_bounds_after_1_over_2pi": [encoded(upper(value)) for value in radii],
                   "summed_segment_radius_bound": encoded(upper(sum(radii, a(0)))),
                   "complete_integral_enclosure": encoded(result), "contour_height": encoded(height),
                   "right_endpoint": encoded(right_endpoint), "maximum_verified_inverse_Neumann_residual": encoded(inverse_residual_max),
                   "exact_free_continuum_retained": True, "all_vectors_cutoff_supported": True,
                   "no_continuation_or_horizontal_pole_shortcut": True,
                   "entire_functions_keep_removable_limits": True}
    return result, certificate


def coefficient_model_errors(contract, vector_errors, gap, tau):
    astar, bstar, duration = contract["H_norms"][1], contract["H_norms"][2], upper(tau)
    epsilon = upper(sum(vector_errors, a(0)))
    epsilon_o, epsilon_xi = epsilon, upper(epsilon / a(2).sqrt())
    output = {}
    for name, hj in contract["categories"].items():
        d_delta_a, d_delta_b = upper(2 * hj[1]), upper(2 * hj[2])
        d_i = upper(4 * astar * hj[1] * duration**2 + 4 * astar**2 * hj[0] * duration**3)
        output[name] = [upper(d_delta_a / lower(gap)),
                        upper(d_delta_b / (2 * lower(gap)) + 8 / lower(base.arb.pi()) * d_i)]
    d_delta_a, d_delta_b = upper(2 * astar * epsilon), upper(2 * bstar * epsilon)
    time_radius = upper(tau.rad())
    d_i = upper(2 * astar**2 * duration**2 * epsilon_o + 4 * astar**2 * duration**2 * epsilon_xi
                + 4 * astar**2 * duration * time_radius)
    output["spectral_vectors_gap_time"] = [upper(d_delta_a / lower(gap)),
                                           upper(d_delta_b / (2 * lower(gap)) + 8 / lower(base.arb.pi()) * d_i)]
    return output, {"A_star_full_lattice": encoded(astar), "B_star_full_lattice": encoded(bstar),
                    "epsilon_O_upper": encoded(epsilon_o), "epsilon_xi_upper": encoded(epsilon_xi),
                    "tau_enclosure_radius": encoded(time_radius), "I_time_radius_bound": encoded(upper(4 * astar**2 * duration * time_radius)),
                    "I_coefficient_exact": "8/pi", "positive_gap_divisor": encoded(gap)}


def spectral_coefficients(model, vectors, energies_hq, gap, tau, contract, vector_errors):
    d = model.d
    a_images = [model.apply_derivative(function, 1) for function in vectors]
    b_images = [model.apply_derivative(function, 2) for function in vectors]
    aa = [[d.inner(vectors[i], a_images[j]) for j in range(2)] for i in range(2)]
    bb = [d.inner(vectors[i], b_images[i]) for i in range(2)]
    if not aa[0][1].overlaps(aa[1][0]):
        raise Unresolved("Complete A self-adjoint expectation intervals disagree")
    delta_a, delta_b = aa[1][1] - aa[0][0], bb[1] - bb[0]
    amplitudes, ordered, contours = [], [], []
    for i in range(2):
        amplitude = c(0)
        ordered_value = c(0)
        for j in range(2):
            difference = c(0) if i == j else c(energies_hq[j] - energies_hq[i])
            amplitude += -c(0, 1) * c(aa[i][j]) * f_time(difference, tau)
            def entire(z, i=i, difference=difference):
                return j_time(z - c(energies_hq[i]), difference, tau, d.work, d.tolerance)
            value, certificate = dunford(model, a_images[i], a_images[j], entire, tau)
            ordered_value += value
            contours.append({"i": i, "j": j, **certificate})
        amplitudes.append(amplitude)
        ordered.append(ordered_value)
    bare_a1 = delta_a / gap
    bare_a2 = delta_b / (2 * gap) + 4 / base.arb.pi() * (ordered[1] - ordered[0]
                                                       + (amplitudes[1]**2 - amplitudes[0]**2) / 2).imag
    errors, propagation = coefficient_model_errors(contract, vector_errors, gap, tau)
    joint_numerical = [upper(bare_a1.rad()), upper(bare_a2.rad())]
    model_sums = [upper(sum((values[k] for values in errors.values()), a(0))) for k in range(2)]
    # Base coefficient balls carry every arithmetic, spectral-parameter,
    # radial-integral and contour radius.  Conservatively allocate that
    # joint numerical radius twice, not as two independent estimates, and
    # explicitly add the duplicate once in the final enclosure.
    errors["joint_matrix_arithmetic_allocation"] = joint_numerical
    errors["joint_functional_calculus_quadrature_allocation"] = joint_numerical
    a1 = base.widen(bare_a1, model_sums[0] + joint_numerical[0])
    a2 = base.widen(bare_a2, model_sums[1] + joint_numerical[1])
    return [a1, a2], {"A_complete_expectations": [[encoded(value) for value in row] for row in aa],
                     "B_complete_diagonal": [encoded(value) for value in bb],
                     "Dyson_b": [encoded(value) for value in amplitudes],
                     "ordered_R_full_intermediate_spectrum": [encoded(value) for value in ordered],
                     "bare_directed_coefficient_enclosures": [encoded(bare_a1), encoded(bare_a2)],
                     "coefficient_radius_allocation": {name: [encoded(value) for value in values] for name, values in errors.items()},
                     "joint_numerical_radius_is_deliberately_duplicated": True,
                     "numeric_allocations_are_not_independent_component_estimates": True,
                     "model_propagation": propagation, "contours": contours,
                     "Omega_times_tau_exactly_pi_over_4": True}


def coherence(model, vectors, tau, model_error, vector_errors, source_radius):
    preparation = (vectors[0] + vectors[1]).scaled(1 / a(2).sqrt())
    amplitudes, contours = [], []
    for vector in vectors:
        value, certificate = dunford(model, vector, preparation,
                                     lambda z: (-c(0, 1) * z * c(tau)).exp(), tau, source_radius)
        amplitudes.append(value)
        contours.append(certificate)
    raw = 2 * amplitudes[0].conjugate() * amplitudes[1]
    epsilon = upper(sum(vector_errors, a(0)))
    # C=2<xi,U*OU xi>, ||O||=||xi||=1.  Unitary Duhamel plus
    # epsilon_O<=epsilon_g+epsilon_e and epsilon_xi<=epsilon/sqrt2.
    model_radius = upper(2 * epsilon + 4 * epsilon / a(2).sqrt() + 4 * upper(tau) * model_error)
    value = base.widen(raw, model_radius)
    if not lower(abs(value)) >= a(F(1, 2)):
        raise Unresolved("Direct finite-source coherence signal lower bound1/2 unresolved")
    rotated = value * (c(0, 1) * base.acb.pi() / 4).exp()
    positive(rotated.real, "Direct phase branch through exp(-i*pi/4)")
    logarithm = rotated.log(analytic=True)
    if not logarithm.is_finite():
        raise Unresolved("Direct coherence logarithm branch not certified")
    nop = 1 - 4 / base.arb.pi() * logarithm.imag
    return nop, {"raw_coherence_enclosure": encoded(raw), "corrected_original_coherence_enclosure": encoded(value),
                 "model_vector_coherence_error_upper": encoded(model_radius),
                 "positive_rotated_phase_margin": encoded(lower(rotated.real)),
                 "same_fixed_reference_vectors_and_time": True, "contours": contours}


def require_width(value, target, label):
    if not upper(base.width(value)) <= lower(a(target)):
        raise Unresolved(label + " actual outward width exceeds or overlaps its declared target")


def finite_source_controls(model, vectors, tau, gap, contract, vector_errors, coefficients):
    d, hats = model.d, contract["mixed_norms"]
    # Bounds are exact outward dyadics.  The directed lower endpoint picks
    # one fixed safe delta, then the required h sequence is exactly delta/2^j.
    delta = positive(lower(minimum_interval(a(base.T0), 1 / (16 * upper(tau) * hats[1]))),
                     "Fixed finite-source control radius")
    duration = upper(tau)
    m1 = upper(4 * duration * hats[1])
    m2 = upper(4 * duration * hats[2] + 8 * duration**2 * hats[1]**2)
    m3 = upper(4 * duration * hats[3] + 24 * duration**2 * hats[1] * hats[2] + 16 * duration**3 * hats[1]**3)
    m4 = upper(4 * duration * hats[4] + 32 * duration**2 * hats[1] * hats[3] + 24 * duration**2 * hats[2]**2
               + 96 * duration**3 * hats[1]**2 * hats[2] + 32 * duration**4 * hats[1]**4)
    l3 = upper(4 / lower(base.arb.pi()) * (2 * m3 + 12 * m2 * m1 + 16 * m1**3))
    l4 = upper(4 / lower(base.arb.pi()) * (2 * m4 + 16 * m3 * m1 + 12 * m2**2
                                         + 96 * m2 * m1**2 + 96 * m1**4))
    uniform_signal_bound = upper(duration * delta * hats[1])
    if not uniform_signal_bound <= a(F(1, 16)):
        raise Unresolved("Uniform source signal proof tau*delta*hhat1<=1/16 unresolved")
    baseline, baseline_certificate = coherence(model, vectors, tau, contract["H_errors"][0], vector_errors, delta)
    if not baseline.contains(1):
        raise Unresolved("Independent original reference identity Nop(0)=1 fails enclosure")
    require_width(baseline, F(1, 10**4), "Reference phase control")
    output = {"delta_fixed": encoded(delta), "mixed_full_lattice_derivative_bounds": [encoded(value) for value in hats],
              "tau_delta_hhat1_upper": encoded(uniform_signal_bound), "M1": encoded(m1), "M2": encoded(m2),
              "M3": encoded(m3), "M4": encoded(m4), "L3": encoded(l3), "L4": encoded(l4),
              "baseline_Nop": encoded(baseline), "baseline_control": baseline_certificate,
              "mixed_path": "N_lambda=1-lambda, t=lambda", "rows": []}
    for divisor in (2, 4, 8):
        h = delta / divisor
        values, direct_rows = [], []
        for sign in (-1, 1):
            source = sign * h
            positive(1 - source, "Finite-source lapse")
            matrices = d.kernel_matrices(source, degree=0)
            changed = ResolventModel(d, matrices, model.gamma, source_t=source, lapse=1 - source)
            nop, certificate = coherence(changed, vectors, tau, contract["mixed_errors"][0], vector_errors, delta)
            values.append(nop - (1 - source))
            direct_rows.append({"lambda": encoded(source), "lapse": encoded(1 - source), "Nop": encoded(nop), **certificate})
        first_tail, second_tail = upper(l3 * h**2 / 6), upper(l4 * h**2 / 24)
        first = base.widen((values[1] - values[0]) / (2 * h), first_tail)
        second = base.widen((values[1] + values[0] - 2 * (baseline - 1)) / (2 * h**2), second_tail)
        if not first.overlaps(coefficients[0]) or not second.overlaps(coefficients[1] - coefficients[0]):
            raise Unresolved("Direct phase and spectral coefficient intervals do not overlap")
        require_width(first, F(1, 10**4), "Independent first coefficient control")
        require_width(second, F(2, 10**4), "Independent mixed quadratic control")
        output["rows"].append({"h": encoded(h), "first_a1_enclosure": encoded(first),
                               "second_a2_minus_a1_enclosure": encoded(second),
                               "first_L3_truncation_upper": encoded(first_tail), "second_L4_truncation_upper": encoded(second_tail),
                               "finite_source_direct_rows": direct_rows})
    # Constant lapse is an independent full contour evaluation; its result
    # is compared with N, never replaced by the exact identity as a value.
    lapse = 1 - delta / 2
    lapse_model = ResolventModel(d, model.matrices[:1], model.gamma, source_t=0, lapse=lapse)
    lapse_readout, certificate = coherence(lapse_model, vectors, tau, upper(abs(lapse)) * contract["H_errors"][0],
                                           vector_errors, delta)
    discrepancy = lapse_readout - lapse
    if not discrepancy.contains(0):
        raise Unresolved("Independent constant-lapse Nop=N identity does not enclose zero discrepancy")
    require_width(discrepancy, F(1, 10**4), "Constant-lapse phase control")
    output["constant_lapse"] = {"N": encoded(lapse), "Nop": encoded(lapse_readout),
                                "Nop_minus_N": encoded(discrepancy), **certificate}
    return output


def source_primitive_controls(d):
    """Independent first/second elementary formulas from the retained source."""
    records = []
    for panel in d.panels:
        r = a((panel.left + panel.right) / 2)
        for source in (-a(base.T0), a(0), a(base.T0)):
            epsilon, f, j = leg_jets(r, source, degree=4)
            u, e = (-2 * source).exp() * r**2, (1 + (-2 * source).exp() * r**2).sqrt()
            regulator = (-u).exp()
            e1, e2 = -u / e, 2 * u / e - u**2 / e**3
            r1, r2 = 2 * u * regulator, (4 * u**2 - 4 * u) * regulator
            checks = [epsilon.terms[1] - e1, 2 * epsilon.terms[2] - e2]
            for sign, jet in ((1, f), (-1, j)):
                leg = (1 + sign / e) / 2
                leg1 = sign * u / (2 * e**3)
                leg2 = sign * (-u / e**3 + 3 * u**2 / (2 * e**5))
                checks.extend([jet.terms[1] - (r1 * leg + regulator * leg1),
                               2 * jet.terms[2] - (r2 * leg + 2 * r1 * leg1 + regulator * leg2)])
            if any(not value.contains(0) or not abs_upper(value) < d.tolerance * 100 for value in checks):
                raise Unresolved("Independent source primitive derivative identities unresolved")
            records.append({"radius": encoded(r), "t": encoded(source), "identity_discrepancies": [encoded(value) for value in checks]})
    return {"fixed_measure_and_coupling_not_differentiated": True, "rows": records}


@dataclass(frozen=True)
class Configuration:
    cutoff: int
    order: int
    quadrature: int
    digits: int
    weight_order: int
    energy_digits: int

    def saved(self):
        return {"cutoff": self.cutoff, "Legendre_order_per_panel": self.order,
                "positive_Gauss_order_per_panel": self.quadrature, "precision_decimal_digits": self.digits,
                "weight_series_last_power": self.weight_order, "root_absolute_width_goal": "1e-" + str(self.energy_digits),
                "total_rank": len(base.panels_for(self.cutoff)) * self.order}


def configurations():
    for cutoff_index, cutoff in enumerate(base.CUTOFFS):
        for order_index, order in enumerate(base.ORDERS):
            if len(base.panels_for(cutoff)) * order > base.MAX_RANK:
                continue
            for precision_index, digits in enumerate(base.PRECISIONS):
                weight_order = 8 * 2**min(precision_index, 4)
                energy_digits = min(digits - 32, 14 + 4 * order_index + 8 * precision_index + 4 * cutoff_index)
                yield Configuration(cutoff, order, 2 * order, digits, weight_order, energy_digits)


def refinement_configurations(configuration):
    ni, pi = base.ORDERS.index(configuration.order), base.PRECISIONS.index(configuration.digits)
    if ni + 1 == len(base.ORDERS) or pi + 1 == len(base.PRECISIONS):
        raise Unresolved("Required next projection/precision control exceeds the frozen ladders")
    return [
        ("projection", Configuration(configuration.cutoff, base.ORDERS[ni + 1], max(configuration.quadrature, base.ORDERS[ni + 1]),
                                     configuration.digits, configuration.weight_order, configuration.energy_digits + 2)),
        ("quadrature", Configuration(configuration.cutoff, configuration.order, 2 * configuration.quadrature,
                                     configuration.digits, configuration.weight_order, configuration.energy_digits + 2)),
        ("precision", Configuration(configuration.cutoff, configuration.order, configuration.quadrature,
                                    base.PRECISIONS[pi + 1], configuration.weight_order, configuration.energy_digits + 2))]


def raw_integration_controls(d):
    return {"documented_directed_backend": "python-flint0.9.0 acb.integral with analytic root/denominator/inverse guards",
            "finite_enclosed_call_count": d.integrals.calls,
            "unpropagated_radius_sum": encoded(upper(d.integrals.radius_sum)),
            "largest_returned_radius": encoded(d.integrals.radius_max),
            "requested_tolerance_goal_only": encoded(d.integrals.tolerance),
            "raw_radius_sum_is_not_a_propagated_residual_or_coefficient_bound": True}


def evaluate_configuration(configuration, work, prior, shared, saved, controls):
    base.ctx.dps = configuration.digits
    tolerance = F(1, 10**max(64, configuration.digits // 2))
    saved.update({"configuration": configuration.saved(), "model_evaluation_attempted": True,
                  "model_evaluated": False,
                  "two_level_reference_certified": False, "ramsey_point_certified": False,
                  "ramsey_certified": False, "full_clock_certified": False,
                  "source_jets_status": "unexecuted", "direct_spectral_status": "unexecuted",
                  "finite_source_controls_status": "unexecuted"})
    d = RamseyDiscretization(configuration.cutoff, configuration.order, configuration.quadrature,
                              configuration.weight_order, work, tolerance)
    matrices0 = d.kernel_matrices(degree=0)
    saved["trial_compression"] = {}
    mu2, gamma, compression = base.trial_compression(d, matrices0[0], saved["trial_compression"])
    if not shared["mu2"].overlaps(mu2) or not shared["gamma"].overlaps(gamma):
        raise CertificateConflict("Certified ground/clock enclosures of the same exact mu2/gamma are disjoint")
    shared["mu2"] = base.intersect_enclosures(shared["mu2"], mu2, "Same exact ground/clock mu2")
    shared["gamma"] = base.intersect_enclosures(shared["gamma"], gamma, "Same exact ground/clock gamma")
    mu2, gamma = shared["mu2"], shared["gamma"]
    gamma_midpoint = positive(gamma.mid(), "Deterministic Ramsey dyadic gamma midpoint")
    saved.update({"mu2": encoded(mu2), "gamma": encoded(gamma), "gamma_midpoint_exact": encoded(gamma_midpoint),
                  "gamma_midpoint_mismatch_upper": encoded(upper(abs(gamma - gamma_midpoint))),
                  "trial_compression": compression,
                  "interaction_error_bounds": {key: encoded(value) for key, value in d.errors.items()},
                  "same_exact_physical_coupling_rule": True})
    contract = completion_error_contract(d, gamma, gamma_midpoint)
    saved["complete_operator_error_contract"] = serialize_error_contract(contract)
    # This is a pre-response engineering gate, not a missing spectral test:
    # an inadequate norm-error certificate cannot meet either energy target.
    if not upper(contract["H_errors"][0]) < lower(a(F(1, 10**13))):
        projection = contract["categories"]["tail_projection"][0]
        other = sum((values[0] for name, values in contract["categories"].items() if name != "tail_projection"), a(0))
        if lower(projection) >= upper(a(F(1, 10**13))) and lower(projection) >= upper(other):
            raise base.RefineOrder("Fixed analytic tail/projection allocation dominates the Ramsey energy gate")
        raise Unresolved("Full operator norm-error allocation cannot yet meet the energy target")
    saved["source_jets_status"] = "attempted"
    matrices = d.kernel_matrices(degree=4)
    # The independently assembled degree-zero enclosure and the zero block
    # of the full jet enclose the same ideal matrix; intersect each entry.
    for i in range(d.rank):
        for j in range(d.rank):
            matrices[0][i, j] = base.intersect_enclosures(matrices0[0][i, j], matrices[0][i, j], "Reference/source-jet M0")
    model = JetModel(d, matrices, gamma_midpoint)
    saved.update({"source_jets_status": "enclosed", "minus_Toeplitz_completion_certificate": model.jet_exponential_certificate,
                  "negative_reference_exponential_certificates": model.exp_metadata,
                  "interaction_matrix_enclosure_radii": [encoded(value) for value in d.matrix_radii]})
    intervals, guard, pencil, counts = base.spectral_data(model, F(1, 10**configuration.energy_digits), target_count=2)
    saved["two_level_BS_count_certificate"] = counts
    original, gap, higher_lower, strict_positive = original_levels(intervals, guard, contract, gamma, matrices[0])
    if not original[0].overlaps(prior["E_g"]):
        raise CertificateConflict("New original-H ground interval disagrees with the certified prior ground interval")
    require_width(original[0], F(1, 10**12), "Original ground energy")
    require_width(original[1], F(1, 10**12), "Original first excited energy")
    if not upper(base.width(gap)) <= lower(a(F(1, 10**6)) * lower(gap)):
        raise Unresolved("Original positive-gap relative width exceeds its target")
    tau = positive(base.arb.pi() / (4 * gap), "External Ramsey reference duration")
    saved.update({"E_g": encoded(original[0]), "E_e": encoded(original[1]), "Omega": encoded(gap), "tau": encoded(tau),
                  "strict_original_ground_positive_bound": encoded(strict_positive),
                  "original_breakup_margin_lower": encoded(lower(2 - upper(original[1]))),
                  "prior_original_ground_interval_agreement": True,
                  "HQ_reference_energy_intervals": [encoded(value) for value in intervals]})
    g, g_metadata = base.eigenfunction(model, intervals[0], pencil, "Ramsey_ground")
    e, e_metadata = base.eigenfunction(model, intervals[1], pencil, "Ramsey_first_excited_f_projection")
    epsilon_g, g_metadata = base.certify_ground(model, g, intervals[0], original[0], original[1], contract, g_metadata)
    e, epsilon_e, e_metadata = normalize_excited(model, g, e, intervals, original, higher_lower, contract, e_metadata)
    saved.update({"two_level_reference_certified": True, "Ramsey_ground_function": g_metadata,
                  "Ramsey_excited_function": e_metadata,
                  "fixed_readout": "xi=(g+e)/sqrt2; O=|g><e|; first excited f projection; positive real phases",
                  "native_transport": "the frozen standard-spinor reference transport"})
    if controls:
        saved["exact_measure_controls"] = base.measure_controls(d)
        saved["kernel_and_first_coupling_controls"] = base.kernel_completion_controls(model, contract)
        saved["source_primitive_derivative_controls"] = source_primitive_controls(d)
    saved["direct_spectral_status"] = "attempted"
    coefficients, spectral_certificate = spectral_coefficients(model, [g, e], intervals, gap, tau, contract, [epsilon_g, epsilon_e])
    saved.update({"a1": encoded(coefficients[0]), "a2": encoded(coefficients[1]),
                  "a2_minus_a1": encoded(coefficients[1] - coefficients[0]),
                  "direct_spectral_coefficient_certificate": spectral_certificate, "direct_spectral_status": "enclosed"})
    require_width(coefficients[0], F(1, 10**4), "Fully corrected a1")
    require_width(coefficients[1], F(1, 10**4), "Fully corrected a2")
    saved["ramsey_point_certified"] = True
    if controls:
        saved["finite_source_controls_status"] = "attempted"
        saved["independent_direct_phase_controls"] = finite_source_controls(model, [g, e], tau, gap, contract,
                                                                            [epsilon_g, epsilon_e], coefficients)
        saved["finite_source_controls_status"] = "all_prescribed_controls_enclosed"
    saved["scalar_integration_raw_controls"] = raw_integration_controls(d)
    saved["model_evaluated"] = True
    return {"mu2": mu2, "gamma": gamma, "E_g": original[0], "E_e": original[1], "Omega": gap, "tau": tau,
            "a1": coefficients[0], "a2": coefficients[1], "a2_minus_a1": coefficients[1] - coefficients[0],
            "_g": g, "_e": e, "_epsilon_g": epsilon_g, "_epsilon_e": epsilon_e, "_discretization": d}


def corrected_refinement_controls(configuration, baseline, work, prior, shared, saved):
    rows = []
    saved["next_level_controls"] = rows
    for axis, next_configuration in refinement_configurations(configuration):
        row = {"axis": axis, "status": "attempted"}
        rows.append(row)
        try:
            candidate = evaluate_configuration(next_configuration, work, prior, shared, row, controls=False)
            for key, value in baseline.items():
                if key.startswith("_"):
                    continue
                if not value.overlaps(candidate[key]):
                    raise Unresolved("Fully corrected next-" + axis + " " + key + " intervals fail overlap")
            for name in ("g", "e"):
                difference = candidate["_discretization"].norm(baseline["_" + name] - candidate["_" + name])
                allowance = upper(baseline["_epsilon_" + name] + candidate["_epsilon_" + name])
                if not upper(difference) <= lower(allowance):
                    raise Unresolved("Phase-aligned next-" + axis + " " + name + " functions fail corrected L2 overlap")
                row[name + "_function_L2_difference_upper"] = encoded(difference)
                row[name + "_combined_original_vector_error_upper"] = encoded(allowance)
        except (Unresolved, AnalyticDomain) as error:
            row.update({"status": "unresolved", "unresolved_item": str(error)})
            raise
        except Exception as error:
            row.update({"status": "backend_or_arithmetic_error", "unresolved_item": type(error).__name__ + ": " + str(error)})
            raise
        row["status"] = "all_fully_corrected_requested_enclosures_and_functions_overlap"
    return rows


def decode_prior(data):
    values = {name: decode_ball(data[name]) for name in
              ("mu2", "gamma", "gamma_midpoint_exact", "gamma_midpoint_mismatch_upper", "E_g", "strict_original_ground_positive_bound")}
    positive(values["mu2"], "Prerequisite exact mu2")
    positive(values["gamma"], "Prerequisite exact gamma")
    positive(values["strict_original_ground_positive_bound"], "Prerequisite strict original ground positivity")
    if not values["gamma_midpoint_exact"].is_exact():
        raise Unresolved("Prerequisite gamma midpoint is not an exact retained dyadic")
    derived_gamma = (4 * 2 * (1 + 3 / a(base.ALPHA)**2).sqrt()).log() / (2 * values["mu2"])
    if not values["gamma"].overlaps(derived_gamma):
        raise CertificateConflict("Prerequisite exact mu2/gamma enclosures violate the frozen coupling rule")
    return values


def new_result(args, ground_data, backend, admission):
    return {"scope": "fixed transported Ramsey readout of the original sine-polar K=0 pair control",
            "source": SOURCE, "independent_source_review": SOURCE_REVIEW,
            "independent_Ramsey_code_review": CODE_REVIEW,
            "backend": backend, "admission": admission,
            "fixed_parameters": ground_data["fixed_parameters"],
            "ground_certificate_input": str(args.ground_certificate),
            "prior_ground_certificate_preserved": ground_data,
            "ground_preparation_certified": True, "ground_dependency_certified": True,
            "ramsey_certified": False, "full_clock_certified": False,
            "COM_response_or_triplet_gap_certified": False, "dense_medium_or_autonomous_clock_certified": False,
            "status": "unresolved", "model_evaluated": False, "model_evaluation_attempted": False,
            "model_evaluated_semantics": "A numerical configuration point completed; certification requires separate gates and refinement controls",
            "mu2": None, "gamma": None, "E_g": None, "E_e": None, "Omega": None, "tau": None,
            "a1": None, "a2": None, "a2_minus_a1": None,
            "unexecuted_outputs": ["mu2", "gamma", "E_g", "E_e", "Omega", "tau", "a1", "a2", "a2_minus_a1"],
            "predeclared_engineering_caps": {"evaluation_work_units": args.max_evaluations,
                                             "integrator_evaluations_per_call": args.integral_evaluations,
                                             "seconds": args.seconds, "resident_memory_MB": args.memory_mb,
                                             "maximum_threads": 4, "worker_pools": 1, "polynomial_rank": base.MAX_RANK,
                                             "last_cutoff": base.CUTOFFS[-1], "last_precision_decimal_digits": base.PRECISIONS[-1]},
            "configuration_outcomes": []}


def retain_partial(result, row):
    result["last_partial_point"] = row
    attempted_rows = [row] + row.get("next_level_controls", [])
    result["model_evaluation_attempted"] = result["model_evaluation_attempted"] or any(
        candidate.get("model_evaluation_attempted", False) for candidate in attempted_rows)
    result["model_evaluated"] = result["model_evaluated"] or any(
        candidate.get("model_evaluated", False) for candidate in attempted_rows)
    keys = ("mu2", "gamma", "gamma_midpoint_exact", "gamma_midpoint_mismatch_upper", "E_g", "E_e", "Omega", "tau",
            "a1", "a2", "a2_minus_a1", "strict_original_ground_positive_bound",
            "complete_operator_error_contract", "Ramsey_ground_function", "Ramsey_excited_function",
            "two_level_BS_count_certificate", "direct_spectral_coefficient_certificate")
    for key in keys:
        if key in row:
            result[key] = row[key]
    result["unexecuted_outputs"] = [key for key in result["unexecuted_outputs"] if result.get(key) is None]


def run(args, ground_data, backend, admission):
    work, result = base.Work(args), new_result(args, ground_data, backend, admission)
    row = None
    try:
        base.ctx.dps = base.PRECISIONS[0]
        prior = decode_prior(ground_data)
        shared = {"mu2": prior["mu2"], "gamma": prior["gamma"]}
        advance_orders = set()
        for configuration in configurations():
            if (configuration.cutoff, configuration.order) in advance_orders:
                continue
            row = {"status": "attempted", "configuration": configuration.saved()}
            try:
                baseline = evaluate_configuration(configuration, work, prior, shared, row, controls=True)
                corrected_refinement_controls(configuration, baseline, work, prior, shared, row)
            except (EngineeringCap, CertificateConflict):
                raise
            except (Unresolved, AnalyticDomain) as error:
                row.update({"status": "unresolved", "unresolved_item": str(error)})
                if isinstance(error, base.RefineOrder):
                    advance_orders.add((configuration.cutoff, configuration.order))
                retain_partial(result, row)
                result["configuration_outcomes"].append({"configuration": configuration.saved(), "status": "unresolved", "item": str(error),
                                                         "model_evaluation_attempted": row.get("model_evaluation_attempted", False),
                                                         "model_evaluated": row.get("model_evaluated", False)})
                result["unresolved_item"] = str(error)
                continue
            row.update({"status": "all_prescribed_full_clock_controls_passed", "ramsey_certified": True, "full_clock_certified": True})
            retain_partial(result, row)
            for key, value in row.items():
                if key not in ("status", "configuration_outcomes"):
                    result[key] = value
            result["status"] = "certified_prepared_original_H_Ramsey_control"
            result["unexecuted_outputs"] = []
            result["configuration_outcomes"].append({"configuration": configuration.saved(), "status": "all_prescribed_controls_passed",
                                                     "model_evaluation_attempted": row.get("model_evaluation_attempted", False),
                                                     "model_evaluated": row.get("model_evaluated", False)})
            result["independent_saved_output_review_status"] = "pending; source audits and internal flags do not adopt a paper result"
            result.pop("unresolved_item", None)
            break
        else:
            result["unresolved_item"] = "Frozen cutoff/order/precision ladder exhausted without every Ramsey control"
    except (EngineeringCap, CertificateConflict, Unresolved, AnalyticDomain) as error:
        result["unresolved_item"] = str(error)
        if isinstance(error, CertificateConflict):
            result["unresolved_kind"] = "certificate_input_or_enclosure_conflict; no physics classification"
        elif isinstance(error, EngineeringCap):
            result["unresolved_kind"] = "predeclared_engineering_cap; no physics classification"
        else:
            result["unresolved_kind"] = "directed_math_or_API_enclosure_not_resolved; no physics classification"
        if row is not None:
            row.update({"status": "unresolved", "unresolved_item": str(error)})
            retain_partial(result, row)
    except Exception as error:
        result.update({"status": "backend_or_arithmetic_error", "unresolved_item": type(error).__name__ + ": " + str(error),
                       "elapsed_seconds": time.perf_counter() - work.start, "evaluation_work_units": work.evaluations})
        if row is not None:
            row.update({"status": "backend_or_arithmetic_error", "unresolved_item": result["unresolved_item"]})
            retain_partial(result, row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    result.update({"elapsed_seconds": time.perf_counter() - work.start, "evaluation_work_units": work.evaluations})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result["status"]


def load_backend():
    backend = base.load_backend()
    for owner, name in ((base.acb, "sinc"), (base.acb, "conjugate"), (base.acb, "log"),
                        (base.acb, "sqrt"), (base.acb_mat, "inv")):
        if not hasattr(owner, name):
            raise RuntimeError("Pinned directed Ramsey backend lacks documented API: " + name)
    return backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=2_000_000)
    parser.add_argument("--integral-evaluations", type=int, default=20_000)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--memory-mb", type=int, default=4096)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.max_evaluations, args.integral_evaluations, args.seconds, args.memory_mb)):
        parser.error("All engineering caps must be declared positive integers before response evaluation")
    if not args.ground_certificate.is_file():
        parser.error("A prior admitted certified ground result is required")
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("Output must be a new file in an existing admitted result directory")
    pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in pools):
        parser.error("All six worker/BLAS pools must be fixed as one")
    try:
        ground_data = read_ground_contract(args.ground_certificate)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    backend = load_backend()  # Complete host-only imports and input checks first.
    if hasattr(base.ctx, "threads"):
        base.ctx.threads = 1
    admission = remote_execution(max_threads=4)  # Fresh free-device allocation/sole-own-PID guard.
    print(run(args, ground_data, backend, admission), flush=True)


if __name__ == "__main__":
    main()
