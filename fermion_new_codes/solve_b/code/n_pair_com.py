#!/usr/bin/env python3
"""Bounded original-H_* complete-channel COM evaluator, pending code audit.

Mathematical inputs are strand_N_pair_spec.tex, its independent review,
strand_N_pair_transport.tex, and the frozen E ground-preparation contract.
The separately reviewed NK1--NK11 supplement fixes the coefficient
assembly and exact feature-1 response restriction.  Four physical triplet
features remain in the model; three have exactly zero first response to
the radial ground.  This source has not been imported, tested or run.

All scientific arithmetic, including imports/tests, is remote-only under
root/Ops admission.  A numerical caller must finish host-only imports,
then pass the same-process fresh-device/visible-allocation/sole-PID guard
before activating this backend.  No other-lane process is queried or altered.
The native free multipliers remain functions on the complete continuum.
No Gram threshold, full-fiber jet replacement, or N1 drift member is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from typing import Any


F = Fraction
SOURCE_SPEC = "solve/strand_N_pair_spec.tex"
SOURCE_REVIEW = "solve/review_N_pair_spec.tex"
ASSEMBLY_SPEC = "solve/strand_N_pair_kernel_assembly.tex"
ASSEMBLY_REVIEW = "solve/review_N_pair_kernel_assembly.tex"
CONTROL_SPEC = "solve/strand_N_pair_controls.tex"
CONTROL_REVIEW = "solve/review_N_pair_controls.tex"
E_SPEC = "solve/strand_E_clock_spec.tex"
E_CODE_REVIEW = "solve/review_E_ground_code.tex"
CHANNELS = ("radial_singlet", "triplet_F0", "triplet_F1", "triplet_F2", "triplet_F3")
ACTIVE_CHANNELS = ("radial_singlet", "triplet_F1")
CUTOFF_LADDER = (16, 24, 32, 48, 64)
POLYNOMIAL_LADDER = (16, 32, 64, 128)
DIGIT_LADDER = (128, 256, 512, 1024, 2048)
ANGULAR_LADDER = (2, 4, 8, 16, 32)
NEUMANN_LADDER = (8, 16, 32, 64, 128)
PANEL_ENDPOINTS = ("0", "1/2", "1", "2", "4", "8", "16", "24", "32", "48", "64")
GROUND_FORMULA = (
    "normalization*(sum(a_i*v_i)/(2*sqrt(1+r^2)-E)+sum(b_i*v_i)) on[0,L], zero outside"
)


class Unresolved(RuntimeError):
    """A missing prerequisite or a finite certified budget has not closed."""


class EngineeringCap(Unresolved):
    """An own-job work cap; never intercepted as a quadrature-domain failure."""


class AnalyticDomain(Exception):
    """An integration cell has not proved the requested holomorphic branch."""


@dataclass(frozen=True)
class Panel:
    left: F
    right: F

    def __post_init__(self):
        if not self.left < self.right:
            raise Unresolved("Panel endpoints are not strictly ordered")

    def saved(self):
        return [str(self.left), str(self.right)]


def panels_for(cutoff):
    if cutoff not in CUTOFF_LADDER:
        raise Unresolved("Cutoff is outside the frozen N ladder")
    edges = [F(value) for value in PANEL_ENDPOINTS if F(value) <= cutoff]
    return tuple(Panel(left, right) for left, right in zip(edges, edges[1:]))


@dataclass(frozen=True)
class Configuration:
    cutoff: int
    counts: tuple[int, ...]
    digits: int = 128
    angular_degree: int = 2
    neumann_degree: int = 8

    def validate(self):
        if len(self.counts) != len(panels_for(self.cutoff)):
            raise Unresolved("Common panel assignment has the wrong length")
        if any(value not in POLYNOMIAL_LADDER for value in self.counts):
            raise Unresolved("Polynomial count is outside the frozen N ladder")
        if self.digits not in DIGIT_LADDER or self.angular_degree not in ANGULAR_LADDER:
            raise Unresolved("Precision or angular degree is outside the frozen ladder")
        if self.neumann_degree not in NEUMANN_LADDER:
            raise Unresolved("Neumann degree is outside the frozen N ladder")
        if self.physical_rank > 2048 or max(self.gauss_orders) > 256:
            raise Unresolved("Global five-channel rank or positive-Gauss cap exceeded")
        return self

    @property
    def scalar_rank(self):
        return sum(self.counts)

    @property
    def physical_rank(self):
        # NK9--NK11 prove an exact reducing response block.  This counts
        # the actually allocated singlet and feature-1 polynomial ranges.
        return len(ACTIVE_CHANNELS) * self.scalar_rank

    @property
    def gauss_orders(self):
        return tuple(2 * value for value in self.counts)

    def saved(self):
        self.validate()
        return {
            "cutoff": self.cutoff, "channels": list(CHANNELS),
            "panels": [panel.saved() for panel in panels_for(self.cutoff)],
            "polynomial_counts_including_constant": list(self.counts),
            "positive_Gauss_orders": list(self.gauss_orders),
            "physical_feature_polynomial_rank_global": self.physical_rank,
            "physical_model_channels": list(CHANNELS),
            "actually_allocated_response_channels": list(ACTIVE_CHANNELS),
            "inactive_first_response_blocks": {"triplet_F0": "exact zero",
                                                "triplet_F2": "exact zero",
                                                "triplet_F3": "exact zero"},
            "inactive_triplet_Hamiltonian_declared_zero": False,
            "decimal_digits": self.digits, "angular_series_last_degree": self.angular_degree,
            "Neumann_last_degree": self.neumann_degree,
            "coefficient_projection_measure": "unweighted dr in all five channels",
            "physical_Gram_weight": "diag(1,W00,W11,W22,W33)",
            "free_multiplier_compressed": False, "Gram_threshold_used": False,
        }


class Refinements:
    """At most twelve distinct same-law tuples, with a declared error driver."""

    def __init__(self):
        self.attempted = []

    def admit(self, configuration, limiting_error):
        configuration.validate()
        if configuration in self.attempted:
            raise Unresolved("A repeated tuple does not create a new refinement")
        if len(self.attempted) >= 12:
            raise Unresolved("The twelve-tuple N refinement cap is exhausted")
        self.attempted.append(configuration)
        return {"tuple": configuration.saved(), "driver": limiting_error,
                "tuple_number": len(self.attempted), "law_or_coupling_refitted": False}

    @staticmethod
    def advance(configuration, category, panel_index=None):
        """The caller supplies the largest enclosed error, never a sample fit."""
        configuration.validate()
        values = dict(cutoff=configuration.cutoff, counts=configuration.counts,
                      digits=configuration.digits, angular_degree=configuration.angular_degree,
                      neumann_degree=configuration.neumann_degree)
        if category == "tail":
            index = CUTOFF_LADDER.index(configuration.cutoff)
            if index + 1 == len(CUTOFF_LADDER):
                raise Unresolved("Cutoff64 reached with an unresolved tail")
            values["cutoff"] = CUTOFF_LADDER[index + 1]
            extra = len(panels_for(values["cutoff"])) - len(configuration.counts)
            values["counts"] = configuration.counts + (16,) * extra
        elif category in ("projection", "quadrature"):
            if panel_index is None or not 0 <= panel_index < len(configuration.counts):
                raise Unresolved("A projection refinement needs its enclosed limiting panel")
            counts = list(configuration.counts)
            index = POLYNOMIAL_LADDER.index(counts[panel_index])
            if index + 1 == len(POLYNOMIAL_LADDER):
                raise Unresolved("Per-panel count128 reached with unresolved projection")
            counts[panel_index] = POLYNOMIAL_LADDER[index + 1]
            values["counts"] = tuple(counts)
        else:
            ladder, name = {
                "angular": (ANGULAR_LADDER, "angular_degree"),
                "arithmetic": (DIGIT_LADDER, "digits"),
                "Neumann": (NEUMANN_LADDER, "neumann_degree"),
            }.get(category, (None, None))
            if ladder is None:
                raise Unresolved("Upstream/coupling/ground input needs same-law refinement: " + category)
            index = ladder.index(values[name])
            if index + 1 == len(ladder):
                raise Unresolved("Declared " + category + " ladder exhausted")
            values[name] = ladder[index + 1]
        return Configuration(**values).validate()


class BallBackend:
    """Pinned documented Arb/Acb APIs; no scientific work before admission."""

    def __init__(self, module):
        self.module = module
        self.arb, self.acb = module.arb, module.acb
        self.arb_mat, self.acb_mat, self.ctx = module.arb_mat, module.acb_mat, module.ctx
        self.admission = None

    @classmethod
    def load_host(cls):
        import flint
        if getattr(flint, "__version__", None) != "0.9.0":
            raise Unresolved("Isolated pinned python-flint0.9.0 is required; no installation here")
        for owner, name in ((flint.arb, "legendre_p_root"), (flint.arb, "man_exp"),
                            (flint.acb, "integral"), (flint.acb_mat, "exp")):
            if not hasattr(owner, name):
                raise Unresolved("Pinned backend API is missing: " + name)
        return cls(flint)

    def activate(self, admission, digits):
        required = ("same_process_fresh_device_query_before_allocation",
                    "cuda_context_in_this_single_job_process",
                    "post_allocation_single_owned_compute_pid_verified")
        if any(admission.get(key) is not True for key in required):
            raise RuntimeError("N mathematical work lacks the complete same-process admission")
        if admission.get("single_process_pid") != os.getpid() or os.nice(0) != 10:
            raise RuntimeError("N admission does not identify this nice10 process")
        if os.environ.get("QI2_REMOTE_RUN") != "1" or digits not in DIGIT_LADDER:
            raise RuntimeError("N remote/precision contract is not active")
        self.admission = admission
        self.ctx.dps = digits
        self.ctx.threads = 1

    def require(self):
        if self.admission is None:
            raise RuntimeError("Arb scientific arithmetic attempted before admission")

    def real(self, value=0):
        self.require()
        if isinstance(value, float):
            raise TypeError("A binary64 value is not an exact scientific N input")
        if isinstance(value, F):
            return self.arb(value.numerator) / value.denominator
        return self.arb(value)

    def complex(self, value=0):
        self.require()
        if isinstance(value, self.acb):
            return value
        return self.acb(self.real(value))

    def upper(self, value):
        return value.upper()

    def lower(self, value):
        return value.lower()

    def absolute_upper(self, value):
        return abs(value).upper()

    def positive(self, value, label):
        if not value.is_finite() or not value.lower() > 0:
            raise Unresolved(label + " lacks a certified positive lower endpoint")
        return value

    def hull(self, left, right):
        lo, hi = self.real(left).lower(), self.real(right).upper()
        if lo > hi:
            raise Unresolved("Disjoint or unordered certified interval")
        middle = (lo + hi) / 2
        return middle + self.arb(0, ((hi - lo) / 2).upper())

    def intersect(self, left, right, label):
        lo, hi = max(left.lower(), right.lower()), min(left.upper(), right.upper())
        if lo > hi:
            raise Unresolved(label + " certified intervals are disjoint")
        return self.hull(lo, hi)

    def widen(self, value, error):
        radius = self.real(error).abs_upper()
        if isinstance(value, self.acb):
            return value + self.acb(self.arb(0, radius), self.arb(0, radius))
        return value + self.arb(0, radius)

    def decode(self, data):
        """Only exact dyadic endpoints; display_ball is never an input."""
        self.require()
        if isinstance(data, dict) and set(("real", "imag")) <= set(data):
            return self.acb(self.decode(data["real"]), self.decode(data["imag"]))
        if not isinstance(data, dict) or "lower_dyadic" not in data or "upper_dyadic" not in data:
            raise Unresolved("Missing exact dyadic ball endpoints")
        bounds = []
        for key in ("lower_dyadic", "upper_dyadic"):
            endpoint = data[key]
            if not isinstance(endpoint, dict) or not isinstance(endpoint.get("mantissa"), str):
                raise Unresolved("Invalid dyadic mantissa encoding")
            exponent = endpoint.get("exponent2")
            if type(exponent) is not int:
                raise Unresolved("Dyadic exponent must be an exact integer")
            try:
                mantissa = int(endpoint["mantissa"])
            except ValueError as exc:
                raise Unresolved("Dyadic mantissa is not an integer string") from exc
            bounds.append(self.arb((mantissa, exponent)))
        if data["lower_dyadic"] == data["upper_dyadic"]:
            # The tuple constructor denotes the exact dyadic datum.  Do
            # not round a saved gamma midpoint through a working-precision
            # arithmetic hull, or silently choose a different midpoint.
            if not bounds[0].is_exact():
                raise Unresolved("Backend did not preserve the exact dyadic input datum")
            return bounds[0]
        return self.hull(*bounds)

    def encode(self, value):
        if isinstance(value, self.acb):
            return {"real": self.encode(value.real), "imag": self.encode(value.imag)}
        if not value.is_finite():
            return {"status": "non-finite/unresolved"}
        def endpoint(bound):
            mantissa, exponent = bound.man_exp()
            return {"mantissa": str(mantissa), "exponent2": int(exponent)}
        return {"lower_dyadic": endpoint(value.lower()), "upper_dyadic": endpoint(value.upper()),
                "display_ball": value.str(30)}


class Work:
    """Own-job planning/checks; no perpetual device lease or foreign-PID abort."""

    def __init__(self, backend, seconds=21600, host_bytes=64 * 1024**3,
                 batch_elapsed_seconds=0, max_evaluations=10**8):
        if not 0 < seconds <= 21600 or not 0 < host_bytes <= 64 * 1024**3:
            raise Unresolved("Requested job caps exceed the reviewed planning limits")
        if not 0 <= batch_elapsed_seconds < 86400 or max_evaluations <= 0:
            raise Unresolved("Requested batch/work cap is invalid")
        backend.require()
        self.backend, self.start = backend, time.perf_counter()
        self.seconds, self.host_bytes = seconds, host_bytes
        self.batch_elapsed_seconds, self.max_evaluations = batch_elapsed_seconds, max_evaluations
        self.evaluations, self.rules = 0, {}
        # Shared by every Integrals object in this job, so an exception
        # in a nested C integration cannot be cleared by its inner scope.
        self.integration_depth = 0
        self.integration_exception = None

    def check(self, increment=0):
        self.evaluations += increment
        elapsed = time.perf_counter() - self.start
        if elapsed > self.seconds or self.batch_elapsed_seconds + elapsed > 86400:
            raise EngineeringCap("The reviewed own-job/cumulative wall-time cap was reached")
        if self.evaluations > self.max_evaluations:
            raise EngineeringCap("The declared integrand/matrix-work cap was reached")
        if len(os.listdir("/proc/self/task")) > 4 or os.nice(0) != 10:
            raise RuntimeError("The admitted single-worker N job left its thread/nice contract")
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:") and int(line.split()[1]) * 1024 > self.host_bytes:
                    raise EngineeringCap("The reviewed64GiB own resident-memory cap was reached")
        # A foreign context appearing after valid admission is not a stop.

    def storage_estimate(self, dense_entries, coefficient_entries=0):
        """Conservative planning bytes, explicitly not an arithmetic certificate.

        Every simultaneously retained matrix/workspace must contribute
        to dense_entries.  Actual resident memory is checked separately.
        Backend scratch estimates remain subject to Ops/source audit.
        """
        self.check()
        bits = self.backend.ctx.prec
        per_complex_ball = 256 + 2 * ((bits + 7) // 8)
        estimate = (dense_entries + coefficient_entries) * per_complex_ball
        if estimate > self.host_bytes:
            raise EngineeringCap("Declared matrix/function workspaces exceed the host planning cap")
        return {"dense_entries_all_workspaces": dense_entries,
                "weighted_function_coefficient_entries": coefficient_entries,
                "working_precision_bits": bits, "estimated_bytes": estimate,
                "estimate_is_a_feasibility_certificate": False,
                "device_arrays_allocated_by_Arb": False,
                "host_limit_bytes": self.host_bytes, "device_limit_bytes": 40 * 1024**3}

    def gauss(self, order):
        if type(order) is not int or order <= 0 or order > 256:
            raise Unresolved("Positive Gauss order leaves the frozen N cap")
        key = self.backend.ctx.prec, order
        if key not in self.rules:
            self.check(order)
            b = self.backend
            values = tuple(b.arb.legendre_p_root(order, index, weight=True) for index in range(order))
            for node, weight in values:
                b.positive(weight, "Positive Gauss weight")
                if not node.lower() > -1 or not node.upper() < 1:
                    raise Unresolved("Gauss node ball leaves its open reference panel")
            if not sum((weight for _, weight in values), b.real(0)).contains(2):
                raise Unresolved("Directed Gauss total-measure control failed")
            self.rules[key] = values
        return self.rules[key]


def double_factorial_odd(value):
    if value < -1 or value % 2 != 1:
        raise ValueError("Odd sphere factorial index is invalid")
    result = 1
    for factor in range(1, value + 1, 2):
        result *= factor
    return result


def binomial_middle(value):
    numerator, denominator = 1, 1
    for j in range(1, value + 1):
        numerator *= value + j
        denominator *= j
    return numerator // denominator


@lru_cache(maxsize=None)
def sphere_coefficients(moment, last_degree):
    """NS4 exact sphere/binomial coefficients, evaluated only after admission."""
    if len(moment) != 3 or any(type(value) is not int or value < 0 for value in moment):
        raise ValueError("Angular moment is not a nonnegative three-index")
    if last_degree not in ANGULAR_LADDER:
        raise Unresolved("Angular-series degree is outside the N ladder")
    coefficients = []
    for degree in range(last_degree + 1):
        value = F(0)
        for first in range(degree + 1):
            for second in range(degree - first + 1):
                third = degree - first - second
                orders = (first, second, third)
                numerator = 1
                for n, k in zip(moment, orders):
                    numerator *= binomial_middle(k) * double_factorial_odd(2 * (n + k) - 1)
                denominator = 4**degree * double_factorial_odd(2 * (sum(moment) + degree) + 1)
                value += F(numerator, denominator)
        if not 0 <= value <= 1:
            raise ArithmeticError("NS4 exact coefficient violates its proved bound")
        coefficients.append(value)
    return tuple(coefficients)


class NativeWeights:
    """Exact eight-preimage moments, with separately retained angular radii."""

    def __init__(self, backend, degree):
        backend.require()
        if degree not in ANGULAR_LADDER:
            raise Unresolved("Angular degree is outside the frozen ladder")
        self.b, self.degree = backend, degree
        self.alpha = backend.real(F(1, 10**6))

    def moment(self, moment, z, enclose_remainder=True):
        b = self.b
        bound = b.absolute_upper(z)
        if not bound < 1:
            raise AnalyticDomain("Native angular series leaves its holomorphic |z|<1 domain")
        coefficients = sphere_coefficients(tuple(moment), self.degree)
        value = b.complex(0)
        for coefficient in reversed(coefficients):
            value = value * z + b.real(coefficient)
        if not enclose_remainder:
            return value
        remainder = (bound**(self.degree + 1) / (1 - bound)).upper()
        if isinstance(z, b.arb) or (isinstance(z, b.acb) and z.imag.is_zero()):
            # Exact real coefficients and a real argument make the
            # entire conditional moment real, including its tail.
            return b.acb(b.widen(value.real, remainder))
        return b.widen(value, remainder)

    def absorption(self, radius, analytic=False):
        b = self.b
        z = self.alpha**2 * radius**2
        w = self.moment((0, 0, 0), z)
        return radius * w.sqrt(analytic=analytic) / (b.real(2).sqrt() * b.arb.pi())

    def joint(self, radius, analytic=False):
        """NS3 physical weight: singlet1 and all four exact diagonal triplets."""
        b = self.b
        r, z = b.complex(radius), self.alpha**2 * radius**2
        epsilon2 = 1 + r**2
        if epsilon2.contains(0):
            raise AnalyticDomain("Native energy denominator contains zero")
        w0 = self.moment((0, 0, 0), z)
        common = (-2 * r**2).exp() * r**4 / (4 * b.arb.pi()**2 * epsilon2)
        weights = [b.complex(1), common * (1 - z / 3) * w0]
        for moment in ((2, 0, 0), (1, 1, 0), (1, 0, 1)):
            weights.append(common * ((b.real(F(2, 3)) - z / 3) * w0
                                      + z * self.moment(moment, z)))
        return tuple(weights)

    def free(self, radius, power=1, analytic=False):
        h = 2 * (1 + self.b.complex(radius)**2).sqrt(analytic=analytic)
        if h.contains(0):
            raise AnalyticDomain("Free multiplier touches zero")
        exponent = F(power)
        if exponent.denominator == 1:
            return h**exponent.numerator
        if exponent.denominator == 2:
            root = h.sqrt(analytic=analytic)
            if not root.is_finite() or root.contains(0):
                raise AnalyticDomain("Exact free half-power branch is unresolved")
            return root**exponent.numerator
        raise RuntimeError("Undeclared free multiplication power")

    def free_second_radial(self, radius, analytic=False):
        b = self.b
        r = b.complex(radius)
        epsilon = (1 + r**2).sqrt(analytic=analytic)
        z = self.alpha**2 * r**2
        w0 = self.moment((0, 0, 0), z)
        if epsilon.contains(0) or w0.contains(0):
            raise AnalyticDomain("Native conditional free curvature denominator touches zero")
        w20 = self.moment((2, 0, 0), z)
        return ((1 - 2 * self.alpha**2 * r**2 / 3) / epsilon
                - (r**2 / 3 - self.alpha**2 * r**4 * w20 / w0) / epsilon**3) / 2


@dataclass
class ImportedGround:
    """Correlated E function balls; no componentwise midpoint state is created."""
    backend: BallBackend
    raw: dict[str, Any]
    panels: tuple[Panel, ...]
    order: int
    a_coefficients: tuple[Any, ...]
    b_coefficients: tuple[Any, ...]
    normalization: Any
    energy_HQ: Any
    energy_exact: Any
    energy_approximate: Any
    energy_error: Any
    gamma_exact: Any
    gamma_midpoint: Any
    gamma_mismatch: Any
    mu2: Any
    vector_error: Any
    residual: Any
    radial_separation: Any
    sign_margin: Any

    @staticmethod
    def read_host(path):
        with Path(path).open() as stream:
            data = json.load(stream)
        if data.get("stage") != "ground" or data.get("status") != "certified_ground_preparation":
            raise Unresolved("N requires the actual certified E ground-stage artifact, not a partial or another stage")
        if data.get("model_evaluated") is not True or data.get("ground_point_certified") is not True:
            raise Unresolved("The imported E ground point has not completed its declared model/ground gates")
        expected_parameters = {
            "m_over_mP": "1/1000", "a_over_lP": "1/1000", "alpha": "1/1000000",
            "Lambda_over_m": "1", "ell_times_m": "1", "gB": "0",
            "sector": "even standard-spinor singlet, half-period character+, K=0, positive-charge two excitations",
            "law": "H_t=exp(-gamma*T_t)*(2*epsilon_t)*exp(-gamma*T_t)",
            "gamma_rule": "log(4*Emax)/(2*mu2), same exact trial compression at every refinement",
            "sea": "W0=0; the unique fixed symmetric Newton-match rectangle is not refitted"}
        actual_parameters = data.get("fixed_parameters", {})
        if any(actual_parameters.get(key) != value for key, value in expected_parameters.items()):
            raise Unresolved("E fixed_parameters do not identify the same frozen original-sine contact law/control")
        if (data.get("same_exact_coupling_rule_across_all_refinements") is not True
                or data.get("ramsey_certified") is not False or data.get("full_clock_certified") is not False
                or data.get("COM_response_or_triplet_gap_certified") is not False):
            raise Unresolved("The E input's same-law/ground-only status does not match this consumer contract")
        export = data.get("ground_export", {})
        required_flags = (data.get("ground_preparation_certified"),
                          export.get("certified_after_measure_kernel_and_next_level_controls"),
                          export.get("full_embedding_is_isometric"),
                          export.get("interval_coefficients_enclose_actual_correlated_normalized_HQ_ground"),
                          export.get("arbitrary_componentwise_midpoint_is_not_a_certified_state"))
        if any(flag is not True for flag in required_flags):
            raise Unresolved("The required frozen E ground-export certificate/control gates have not passed")
        function = export.get("radial_function", {})
        if function.get("formula") != GROUND_FORMULA or function.get(
                "positive_common_E_factor_cancelled_before_normalization") is not True:
            raise Unresolved("E normalized function contract is missing its approved common-E cancellation")
        if export.get("COM_response_or_triplet_gap_certified") is not False:
            raise Unresolved("E's radial certificate cannot supply a triplet/COM certificate")
        expected_measure = {
            "radial_chart": "L2(nu_alpha), alpha=1/1000000; implemented chart L2(dr)",
            "undo_measure_absorption": "u_E(r)=u_E_dr(r)/q_alpha(r), almost everywhere; q_alpha=r*sqrt(W(alpha^2*r^2))/(sqrt(2)*pi)",
            "full_momentum_embedding": "u_full(k)=u_E(sqrt(sum_i sin(alpha*k_i)^2)/alpha)*chi_s/sqrt(8)",
            "full_momentum_measure": "d^3k/(2*pi)^3; pushforward=8*nu_alpha"}
        if any(export.get(key) != value for key, value in expected_measure.items()):
            raise Unresolved("E native measure/q-alpha/sqrt8 embedding contract does not match the frozen N chart")
        return data

    @classmethod
    def prepare(cls, backend, data):
        b, export = backend, data["ground_export"]
        metadata, function = data["ground_radial_function"], export["radial_function"]
        if metadata.get("normalized_actual_HQ_family") is not True or metadata.get(
                "separation_subtracts_both_original_interval_and_residual_reference") is not True:
            raise Unresolved("E normalized-family or intrinsic residual-reference gate is missing")
        order = function["Legendre_order_per_panel"]
        if type(order) is not int or order <= 0:
            raise Unresolved("Invalid E Legendre polynomial count")
        panels = tuple(Panel(F(left), F(right)) for left, right in function["panel_endpoints"])
        if not panels or panels[0].left != 0 or any(p.right != q.left for p, q in zip(panels, panels[1:])):
            raise Unresolved("Exported E support is not a contiguous zero-based panel interval")
        if panels[-1].right not in CUTOFF_LADDER:
            raise Unresolved("Exported ground support leaves the common cutoff ladder")
        aa, bb = tuple(b.decode(value) for value in function["a"]), tuple(b.decode(value) for value in function["b"])
        if len(aa) != len(panels) * order or len(bb) != len(aa):
            raise Unresolved("Exported E continuous coefficients disagree with its panels/count")
        if any(not isinstance(value, b.arb) or not value.is_finite() for value in (*aa, *bb)):
            raise Unresolved("The frozen E real continuous coefficient contract was not preserved")
        normalization = b.decode(function["normalization"])
        if normalization.contains(0) or not normalization.is_finite():
            raise Unresolved("Exported E normalization is non-finite or contains zero")
        energy_hq = b.decode(function["E"])
        b.positive(2 - energy_hq.upper(), "E continuous free-resolvent denominator")
        mu2 = b.positive(b.decode(data["mu2"]), "Exact trial mu2")
        gamma = b.positive(b.decode(data["gamma"]), "Exact same-law gamma")
        alpha = b.real(F(1, 10**6))
        emax = 2 * (1 + 3 / alpha**2).sqrt()
        coupling_rule = (4 * emax).log() / (2 * mu2)
        gamma = b.intersect(gamma, coupling_rule, "Exact E binding-rule coupling")
        midpoint = b.positive(b.decode(data["gamma_midpoint_exact"]), "E exact dyadic midpoint")
        if not midpoint.is_exact():
            raise Unresolved("E gamma midpoint is not an exact dyadic value")
        mismatch = max(b.absolute_upper(gamma - midpoint),
                       b.decode(data["gamma_midpoint_mismatch_upper"]).upper())
        positive_lower = b.positive(b.decode(data["strict_original_ground_positive_bound"]),
                                    "Strict original-H ground positivity")
        energy = b.intersect(b.decode(data["E_g"]), b.hull(positive_lower.lower(), b.real(F(1, 4))),
                             "Original ground and analytic0<E<=1/4 bounds")
        b.positive(energy, "Original ground")
        representative = energy.mid()
        energy_error = b.absolute_upper(energy - representative)
        vector_error = max(b.decode(export["full_embedding_phase_aligned_norm_error_upper"]).upper(),
                           b.decode(metadata["phase_aligned_vector_error_upper"]).upper())
        if vector_error < 0:
            raise Unresolved("Negative phase-aligned input-error bound")
        radial_gap = b.positive(b.decode(metadata["original_spectral_separation_lower"]),
                                "E intrinsic radial residual-reference separation")
        sign = b.positive(b.decode(metadata["original_declared_sign_margin_lower"]), "E ground phase margin")
        residual = b.decode(metadata["original_full_operator_residual_upper"])
        norm = b.positive(b.decode(metadata["continuous_normalization_enclosure"]), "E normalization integral")
        if not norm.contains(1):
            raise Unresolved("E's correlated normalized family does not enclose its unit-norm identity")
        return cls(b, data, panels, order, aa, bb, normalization, energy_hq, energy,
                   representative, energy_error, gamma, midpoint, mismatch, mu2,
                   vector_error, residual, radial_gap, sign)

    @property
    def support(self):
        return self.panels[-1].right

    def value_on_panel(self, panel_index, radius, analytic=False):
        """Exact advertised L2(dr) formula, without another E or normalization."""
        b, panel = self.backend, self.panels[panel_index]
        x = (2 * radius - b.real(panel.left + panel.right)) / b.real(panel.right - panel.left)
        polynomials = [b.complex(1)]
        if self.order > 1:
            polynomials.append(x)
        for degree in range(1, self.order - 1):
            polynomials.append(((2 * degree + 1) * x * polynomials[-1]
                                - degree * polynomials[-2]) / (degree + 1))
        va, vb = b.complex(0), b.complex(0)
        for degree, polynomial in enumerate(polynomials):
            legendre = polynomial * b.real(F(2 * degree + 1, 1) / (panel.right - panel.left)).sqrt()
            index = panel_index * self.order + degree
            va += self.a_coefficients[index] * legendre
            vb += self.b_coefficients[index] * legendre
        denominator = 2 * (1 + b.complex(radius)**2).sqrt(analytic=analytic) - self.energy_HQ
        if denominator.contains(0):
            raise AnalyticDomain("E continuous exported function touches a free pole")
        return self.normalization * (va / denominator + vb)

    def value_full_singlet_on_panel(self, panel_index, radius, native_weights, analytic=False):
        """NS1: undo exact q_alpha and divide by sqrt8 almost everywhere."""
        q = native_weights.absorption(radius, analytic=analytic)
        if q.contains(0):
            raise AnalyticDomain("Use the canceled unweighted chart at the measure-zero origin")
        return self.value_on_panel(panel_index, radius, analytic) / (q * self.backend.real(8).sqrt())


def native_cutoff_certificate(backend, gamma, cutoff):
    """NS5 full-torus analytic bound, not a spherical continuation at folds."""
    b = backend
    if cutoff not in CUTOFF_LADDER:
        raise Unresolved("Native cutoff is outside the frozen ladder")
    alpha, eta = b.real(F(1, 10**6)), b.real(F(1, 32))
    bb = (alpha * eta / 2).sinh() / alpha
    delta = b.positive(1 - 3 * bb**2, "Complex COM positive-root margin")
    q = b.positive((alpha * eta).cos(), "Complex COM Gaussian margin")
    u2 = b.real(F(3, 2)) * (1 + 1 / delta.sqrt())
    aa = (alpha * eta / 2).cosh()**3 * (3 * ((alpha * eta).cosh() - 1) / (2 * alpha**2)).exp()
    kernel_bound = 4 * b.absolute_upper(gamma) * aa**2 * u2**2 / delta.sqrt()
    tail = (2 * b.real(2).sqrt() * kernel_bound / alpha**3 * (-q * cutoff**2).exp()).upper()
    return {"seed": tail, "first": (tail / eta).upper(),
            "second": (2 * tail / eta**2).upper(), "eta": eta,
            "root_margin_delta": delta, "Gaussian_margin_q": q,
            "analytic_full_spin_kernel_bound": kernel_bound.upper(),
            "high_radius_and_van_Hove_regions_included": True}


def bernstein_block_error(backend, analytic_bound, left_count, right_count):
    """NS projection coefficient bound; physical diag(1,W) is added separately."""
    b, rho = backend, backend.real(F(5, 2))
    return (4 * b.real(analytic_bound) * (rho**(-left_count) + rho**(-right_count))
            / (1 - 1 / rho)**2).upper()


def common_physical_error(backend, panels, weights, coefficient_errors):
    """Weighted physical HS bound, retaining the singlet baseline w_i>=1."""
    b = backend
    if len(weights) != len(panels) or any(not value >= 1 for value in weights):
        raise Unresolved("Common physical panel weights must include the singlet1")
    total = b.real(0)
    for i, first in enumerate(panels):
        for j, second in enumerate(panels):
            total += weights[i] * weights[j] * b.real((first.right - first.left) * (second.right - second.left)) * coefficient_errors[i][j]**2
    return total.sqrt().upper()


def real_scalar(backend, value, label):
    """Restrict a known real quantity only after its imaginary enclosure test."""
    if not value.is_finite() or not value.imag.contains(0):
        raise Unresolved(label + " is nonfinite or fails its exact-real control")
    return value.real


def matrix_finite(matrix):
    return all(matrix[i, j].is_finite()
               for i in range(matrix.nrows()) for j in range(matrix.ncols()))


def matrix_norm(backend, matrix):
    """Directed Frobenius bound, valid for rectangular/nonnormal matrices."""
    total = backend.real(0)
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            total += matrix[i, j].abs_upper()**2
    return total.sqrt().upper()


def matrix_radius(backend, matrix):
    """An error bound only: these midpoints never replace a scientific state."""
    total = backend.real(0)
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            total += (matrix[i, j] - matrix[i, j].mid()).abs_upper()**2
    return total.sqrt().upper()


def block(matrix, row, column, size, backend):
    return backend.acb_mat([[matrix[row + i, column + j] for j in range(size)]
                            for i in range(size)])


def identity(backend, size):
    result = backend.acb_mat(size, size)
    for i in range(size):
        result[i, i] = 1
    return result


class Kernels:
    """Literal NK4--NK11 coefficient assembly in the canceled radial chart."""

    def __init__(self, weights, gamma, truncate_moments=False):
        self.weights, self.b, self.gamma = weights, weights.b, gamma
        self.truncate_moments = truncate_moments

    def amplitudes(self, radius, analytic=False):
        b, alpha, r = self.b, self.weights.alpha, self.b.complex(radius)
        epsilon = (1 + r**2).sqrt(analytic=analytic)
        if not epsilon.is_finite() or epsilon.contains(0) or (epsilon + 1).contains(0):
            raise AnalyticDomain("NK4 positive-root/epsilon+1 branch margin is unresolved")
        z = alpha**2 * r**2
        w0 = self.weights.moment((0, 0, 0), z, not self.truncate_moments)
        w20 = self.weights.moment((2, 0, 0), z, not self.truncate_moments)
        if w0.contains(0) or not w0.is_finite():
            raise AnalyticDomain("NK4 native conditional-moment denominator touches zero")
        root_w = w0.sqrt(analytic=analytic)
        q = r * root_w / (b.real(2).sqrt() * b.arb.pi())
        R2 = (-r**2).exp()
        f, t = (epsilon + 1) / (2 * epsilon), (epsilon - 1) / (2 * epsilon)
        if f.contains(0):
            raise AnalyticDomain("NK5 unmixed upper-spinor factor touches zero")
        g = (2 * epsilon * (epsilon + 1)).sqrt(analytic=analytic)
        if not g.is_finite() or g.contains(0):
            raise AnalyticDomain("NK4 lower-spinor branch margin is unresolved")
        g = 1 / g
        g_epsilon = -(2 * epsilon + 1) * g / (2 * epsilon * (epsilon + 1))
        A = 1 - z / 3
        B = r**2 / 3 - alpha**2 * r**4 * w20 / w0
        C = B / epsilon**2
        D = (1 - 2 * z / 3) / epsilon - B / epsilon**3
        FF = C / epsilon**3 - D / (2 * epsilon**2)
        leg_average = -b.real(F(1, 2)) + z / 3 - alpha**2 / 4
        aU, aL = q * R2 * f, q * R2 * t
        bb = -q * R2 * (A - B / (epsilon * (epsilon + 1))) / (2 * b.real(2).sqrt() * epsilon)
        bbar = -(A - B / (epsilon * (epsilon + 1))) / (2 * b.real(2).sqrt() * epsilon)
        cUbar = leg_average * f + FF / 4 - C / (16 * f * epsilon**4)
        cLbar = (leg_average * t - FF / 4 - g**2 * A
                 - (g_epsilon**2 * r**2 + 2 * epsilon * g * g_epsilon) * C)
        cU, cL = q * R2 * cUbar, q * R2 * cLbar
        return {"epsilon": epsilon, "q": q, "aU": aU, "aL": aL,
                "b": bb, "cU": cU, "cL": cL, "zU": D * aU / 2,
                "zL": D * aL / 2, "h11": D / 2,
                "normalized_without_qR2": {"epsilon": epsilon, "aU": f, "aL": t,
                   "b": bbar, "cU": cUbar, "cL": cLbar, "zU": D * f / 2,
                   "zL": D * t / 2}}

    def from_amplitudes(self, output, input_):
        b, gamma = self.b, self.gamma
        denominator = output["epsilon"] + input_["epsilon"]
        if not denominator.is_finite() or denominator.contains(0):
            raise AnalyticDomain("NK9--NK11 four-leg denominator touches zero")
        a = output["aU"] * input_["aU"] + output["aL"] * input_["aL"]
        numerator = (output["cU"] * input_["aU"] + output["cL"] * input_["aL"]
                     + output["aU"] * input_["cU"] + output["aL"] * input_["cL"]
                     - 2 * output["b"] * input_["b"])
        second_denominator = (output["zU"] * input_["aU"] + output["zL"] * input_["aL"]
                              + output["aU"] * input_["zU"] + output["aL"] * input_["zL"])
        k = gamma / denominator * (input_["b"]
                 - input_["aL"] / (b.real(2).sqrt() * (output["epsilon"] + 1)))
        return {"Sr": -gamma * a / denominator, "St": gamma / denominator,
                "K": k, "L11": -gamma * numerator / denominator
                + gamma * second_denominator / (2 * denominator**2)}

    def values(self, radius, other, analytic=False):
        return self.from_amplitudes(self.amplitudes(radius, analytic),
                                    self.amplitudes(other, analytic))


class Integrals:
    """Directed Acb quadrature with a job-shared, nesting-aware exception latch.

    AnalyticDomain requests local subdivision.  Every other callback
    exception is preserved without crossing a C callback boundary.
    Once latched, later callbacks do no scientific work.  Only after
    the outermost enclosing C integration returns is the original
    exception rethrown/cleared; no finite or partial result is accepted.
    Requested tolerances are accuracy goals, never certificates.
    """

    def __init__(self, backend, work, digits, eval_limit=10**6):
        if type(eval_limit) is not int or not 0 < eval_limit <= 10**6:
            raise Unresolved("Per-call integration cap must be a genuine positive integer at most 1000000")
        self.b, self.work = backend, work
        self.eval_limit = eval_limit
        self.goal = backend.real(F(1, 10**(digits // 2)))
        self.count, self.maximum_radius = 0, backend.real(0)
        self.invalid = backend.acb("nan")

    def raise_outer_latch(self):
        if self.work.integration_depth != 0:
            return self.invalid
        error, traceback = self.work.integration_exception
        self.work.integration_exception = None
        raise error.with_traceback(traceback) from None

    def latch(self, error):
        if self.work.integration_exception is None:
            self.work.integration_exception = error, error.__traceback__

    def integrate(self, function, panel, label):
        b = self.b
        if self.work.integration_exception is not None:
            return self.raise_outer_latch()
        def callback(z, analytic):
            if self.work.integration_exception is not None:
                return self.invalid
            try:
                self.work.check(1)
                value = function(z, bool(analytic))
                if self.work.integration_exception is not None:
                    return self.invalid
                return value if value.is_finite() else self.invalid
            except AnalyticDomain:
                return self.invalid
            except BaseException as error:
                self.latch(error)  # Includes EngineeringCap; first error wins.
                return self.invalid
        self.work.check()
        outermost = self.work.integration_depth == 0
        self.work.integration_depth += 1
        try:
            try:
                value = b.acb.integral(callback, b.complex(panel.left), b.complex(panel.right),
                                       abs_tol=self.goal, rel_tol=self.goal, eval_limit=self.eval_limit)
            except BaseException as error:
                # Acb's wrapper error cannot replace a previously latched
                # original callback error.  A backend error without an
                # earlier latch is itself preserved across enclosing C.
                self.latch(error)
                value = self.invalid
        finally:
            self.work.integration_depth -= 1
        if self.work.integration_exception is not None:
            if outermost:
                return self.raise_outer_latch()
            return self.invalid
        self.count += 1
        if not value.is_finite():
            if self.work.integration_depth:
                # A normal inner domain failure may ask an enclosing
                # analytic callback to subdivide its complex cell.
                raise AnalyticDomain(label + " did not produce a finite directed inner integral")
            raise Unresolved(label + " did not produce a finite directed integral")
        self.maximum_radius = max(self.maximum_radius, b.absolute_upper(value - value.mid()))
        return value


def ellipse(backend, panel):
    """A rectangle containing the entire rho=5/2 Bernstein ellipse."""
    b, rho = backend, F(5, 2)
    center, half = (panel.left + panel.right) / 2, (panel.right - panel.left) / 2
    horizontal, vertical = half * (rho + 1 / rho) / 2, half * (rho - 1 / rho) / 2
    rectangle = b.acb(b.arb(b.real(center), b.real(horizontal)), b.arb(0, b.real(vertical)))
    root_squared = 1 + rectangle**2
    margin = b.positive(root_squared.real, "Complete panel ellipse positive-energy branch")
    z_bound = (b.real(F(1, 10**12)) * rectangle.abs_upper()**2).upper()
    if not z_bound < b.real(F(1, 2)):
        raise Unresolved("Complete panel ellipse leaves the native angular branch margin")
    return rectangle, margin.lower(), z_bound


def approximation_budgets(backend, configuration, ground, weights, kernels):
    """Common-space projection/Gauss/complex-tail bounds, never sampled extrema."""
    b, panels = backend, panels_for(configuration.cutoff)
    rectangles, margins, physical_weights = [], [], []
    for panel in panels:
        rectangle, margin, zz = ellipse(b, panel)
        amplitudes = kernels.amplitudes(rectangle, analytic=True)
        if any(not value.is_finite() for name, value in amplitudes.items()
               if name != "normalized_without_qR2"):
            raise Unresolved("Complete panel ellipse NK4--NK5 enclosure failed")
        if not amplitudes["epsilon"].real.lower() > 0:
            raise Unresolved("Complete panel ellipse epsilon has no positive real margin")
        center, half = (panel.left + panel.right) / 2, (panel.right - panel.left) / 2
        real_interval = b.acb(b.arb(b.real(center), b.real(half)))
        wp = max([b.real(1)] + [value.abs_upper() for value in weights.joint(real_interval)])
        rectangles.append(amplitudes)
        margins.append({"root_squared_real_lower": margin, "angular_abs_z_upper": zz})
        physical_weights.append(wp)
    errors, panel_errors = {}, {}
    names = ("Sr", "St", "S1", "S11")
    for name in names:
        projection, quadrature = [], []
        for i, first in enumerate(rectangles):
            prow, qrow = [], []
            for j, second in enumerate(rectangles):
                forward = kernels.from_amplitudes(first, second)
                reverse = kernels.from_amplitudes(second, first)
                if name == "S1":
                    analytic_bound = (forward["K"].abs_upper()**2
                                      + reverse["K"].abs_upper()**2).sqrt().upper()
                else:
                    analytic_bound = forward[{"S11": "L11"}.get(name, name)].abs_upper()
                if not analytic_bound.is_finite():
                    raise Unresolved("A full product ellipse has no finite coefficient bound")
                ni, nj = configuration.counts[i], configuration.counts[j]
                qi, qj = configuration.gauss_orders[i], configuration.gauss_orders[j]
                prow.append(bernstein_block_error(b, analytic_bound, ni, nj))
                qrow.append(bernstein_block_error(b, analytic_bound, 2 * qi - ni + 1, 2 * qj - nj + 1))
            projection.append(prow)
            quadrature.append(qrow)
        dproj = common_physical_error(b, panels, physical_weights, projection)
        dquad = 2 * common_physical_error(b, panels, physical_weights, quadrature)
        errors[name] = {"projection": dproj, "quadrature": dquad.upper()}
        # Each row is an enclosed contribution; it selects a panel for the
        # next permitted tuple and is not a pointwise sample maximum.
        panel_errors[name] = tuple(
            (sum((physical_weights[i] * physical_weights[j]
                    * b.real((panels[i].right - panels[i].left)
                             * (panels[j].right - panels[j].left))
                    * (projection[i][j]**2 + 4 * quadrature[i][j]**2)
                  for j in range(len(panels))), b.real(0))).sqrt().upper()
            for i in range(len(panels)))
    tail = native_cutoff_certificate(b, ground.gamma_exact, configuration.cutoff)
    for name, order in (("Sr", "seed"), ("St", "seed"), ("S1", "first"), ("S11", "second")):
        errors[name]["tail"] = tail[order]
    return errors, panel_errors, physical_weights, margins, tail


def legendre_values(backend, panel, count, radius):
    b, length = backend, panel.right - panel.left
    x = (2 * radius - b.real(panel.left + panel.right)) / b.real(length)
    values = [b.complex(1)]
    if count > 1:
        values.append(x)
    for degree in range(1, count - 1):
        values.append(((2 * degree + 1) * x * values[-1] - degree * values[-2]) / (degree + 1))
    return tuple(value * b.real(F(2 * degree + 1, 1) / length).sqrt()
                 for degree, value in enumerate(values))


class Discretization:
    """Unweighted polynomial projection, exact physical Gram, fixed positive rule."""

    def __init__(self, backend, work, configuration, ground):
        self.b, self.work, self.configuration, self.ground = backend, work, configuration, ground
        self.panels = panels_for(configuration.cutoff)
        if self.panels[-1].right < ground.support:
            raise Unresolved("N support must retain the entire exported ground without a new truncation")
        self.ground_panel_indices = []
        for panel in self.panels:
            if panel.left >= ground.support:
                self.ground_panel_indices.append(None)
                continue
            index = next((j for j, p in enumerate(ground.panels)
                          if p.left <= panel.left and panel.right <= p.right), None)
            if index is None:
                raise Unresolved("N/E polynomial panels are incompatible: an N panel crosses an exported E boundary")
            self.ground_panel_indices.append(index)
        self.weights = NativeWeights(backend, configuration.angular_degree)
        self.integrals = Integrals(backend, work, configuration.digits)
        self.offsets, offset = [], 0
        for count in configuration.counts:
            self.offsets.append(offset)
            offset += count
        self.dimension, self.nodes = offset, []
        for index, panel in enumerate(self.panels):
            center, half = (panel.left + panel.right) / 2, (panel.right - panel.left) / 2
            for x, w in work.gauss(configuration.gauss_orders[index]):
                r, weight = backend.real(center) + backend.real(half) * x, backend.real(half) * w
                self.nodes.append((index, r, weight, legendre_values(backend, panel, configuration.counts[index], r)))
        self.gram = backend.acb_mat(self.dimension, self.dimension)
        for index, panel in enumerate(self.panels):
            count, offset = configuration.counts[index], self.offsets[index]
            for first in range(count):
                for second in range(first + 1):
                    def integrand(r, analytic, first=first, second=second):
                        values = legendre_values(backend, panel, count, r)
                        return values[first] * values[second] * self.weights.joint(r, analytic)[2]
                    value = self.integrals.integrate(integrand, panel, "Exact physical feature-1 Gram")
                    real_scalar(backend, value, "Physical feature-1 Gram")
                    self.gram[offset + first, offset + second] = value
                    self.gram[offset + second, offset + first] = value
        self.gram_norm = matrix_norm(backend, self.gram)
        self.gram_R = identity(backend, self.dimension)
        self.symmetrization_bounds = []
        self.coefficient_matrices, self.full_moment_matrices = self.assemble()

    def assemble(self):
        b, d, nodes = self.b, self.dimension, self.nodes
        coefficients = b.acb_mat(d, len(nodes))
        for j, (panel_index, _, weight, values) in enumerate(nodes):
            for n, value in enumerate(values):
                coefficients[self.offsets[panel_index] + n, j] = weight * value
        outputs = []
        for truncate in (True, False):
            kernels = Kernels(self.weights, self.ground.gamma_midpoint, truncate)
            amplitudes = [kernels.amplitudes(b.complex(radius)) for _, radius, _, _ in nodes]
            matrices = {name: b.acb_mat(len(nodes), len(nodes)) for name in ("Sr", "St", "K", "L11")}
            for i, first in enumerate(amplitudes):
                self.work.check(len(nodes))
                for j, second in enumerate(amplitudes):
                    values = kernels.from_amplitudes(first, second)
                    for name, matrix in matrices.items():
                        matrix[i, j] = values[name]
            projected = {name: coefficients * matrix * coefficients.transpose()
                         for name, matrix in matrices.items()}
            symmetry = {}
            for name in ("Sr", "St", "L11"):
                # The exact positive-rule Sr/St matrices are signed
                # positive/negative Cauchy compressions.  No eigenvalue
                # is clipped.  Symmetry follows from the literal kernel;
                # outward symmetrization retains the change below.
                matrix = projected[name]
                projected[name] = (matrix + matrix.transpose().conjugate()) / 2
                symmetry[name] = matrix_norm(b, projected[name] - matrix)
            if any(not matrix_finite(matrix) for matrix in projected.values()):
                raise Unresolved("Finite positive-rule coefficient assembly is nonfinite")
            outputs.append(projected)
            self.symmetrization_bounds.append(symmetry)
        return outputs[0], outputs[1]

    def physical_coefficient_norm(self, matrix, output_channel, input_channel):
        gain = self.b.real(1)
        if output_channel == 1:
            gain *= self.gram_norm.sqrt()
        if input_channel == 1:
            gain *= self.gram_norm.sqrt()
        return (gain * matrix_norm(self.b, matrix)).upper()

    def norm_and_error_allocations(self):
        b, errors = self.b, {}
        full_kernels = Kernels(self.weights, self.ground.gamma_exact)
        budget, rows, weights, margins, tail = approximation_budgets(
            b, self.configuration, self.ground, self.weights, full_kernels)
        norms = {}
        for name, coefficient_name, out_, in_ in (("Sr", "Sr", 0, 0), ("St", "St", 1, 1),
                                                   ("S1", "K", 1, 0), ("S11", "L11", 0, 0)):
            candidate = self.coefficient_matrices[coefficient_name]
            full = self.full_moment_matrices[coefficient_name]
            norm = self.physical_coefficient_norm(candidate, out_, in_)
            # S_exact=gamma*A, while these two Gauss matrices both use
            # the retained exact midpoint.  Scale the full-minus-series
            # discrepancy back to gamma_+ before coupling telescoping.
            angular = (self.physical_coefficient_norm(full - candidate, out_, in_)
                       * self.ground.gamma_exact.upper() / self.ground.gamma_midpoint).upper()
            arithmetic = matrix_radius(b, candidate)
            if out_ == 1:
                arithmetic *= self.gram_norm.sqrt()
            if in_ == 1:
                arithmetic *= self.gram_norm.sqrt()
            coupling = self.ground.gamma_mismatch * norm / self.ground.gamma_midpoint
            errors[name] = dict(budget[name], angular=angular, arithmetic=arithmetic.upper(),
                                coupling=coupling.upper())
            # The underlying literal kernels are exactly Hermitian.
            # Retain the outward symmetrization change as part of the
            # arithmetic budget rather than dropping rounded entries.
            if coefficient_name in self.symmetrization_bounds[0]:
                symmetry = self.symmetrization_bounds[0][coefficient_name]
                gain = self.gram_norm if out_ == in_ == 1 else b.real(1)
                errors[name]["arithmetic"] = (errors[name]["arithmetic"] + gain * symmetry).upper()
            # Addition is conservative: the same ball rounding also
            # accompanies every later matrix and integral operation.
            errors[name]["total"] = sum(errors[name].values(), b.real(0)).upper()
            norms[name] = norm
        return norms, errors, rows, weights, margins, tail


def exponential_insertions(discretization, scale):
    """Entire redundant-feature calculus for the required observable blocks.

    Use the R,T,R upper triangle.  Its first superdiagonal contains the
    two adjoint first insertions; its (0,2) block contains S11/2.  The
    exponential's (0,2) block is therefore half the ordinary RR second
    derivative.  With G=diag(I,Gt,I), exp(V C V*)=I+V C phi1(GC) V*.
    phi1 of a possibly nonnormal GC is the upper-right block of
    exp([[GC,I],[0,0]]), an entire matrix function.  No Gram inverse or
    zero-eigenvalue test occurs.
    """
    b, d = discretization.b, discretization.dimension
    matrices, gram = discretization.coefficient_matrices, discretization.gram
    size = 3 * d
    triangle, G = b.acb_mat(size, size), b.acb_mat(size, size)
    for i in range(d):
        for j in range(d):
            for index, name in ((0, "Sr"), (1, "St"), (2, "Sr")):
                triangle[index * d + i, index * d + j] = scale * matrices[name][i, j]
                G[index * d + i, index * d + j] = gram[i, j] if index == 1 else int(i == j)
            triangle[i, d + j] = scale * matrices["K"][j, i].conjugate()
            triangle[d + i, 2 * d + j] = scale * matrices["K"][i, j]
            triangle[i, 2 * d + j] = scale * matrices["L11"][i, j] / 2
    generator = G * triangle
    augmented = b.acb_mat(2 * size, 2 * size)
    for i in range(size):
        for j in range(size):
            augmented[i, j] = generator[i, j]
        augmented[i, size + i] = 1
    discretization.work.check()
    exponential = augmented.exp()
    discretization.work.check()
    if not matrix_finite(exponential):
        raise Unresolved("Documented Acb nonnormal augmented exponential is nonfinite")
    phi = b.acb_mat([[exponential[i, size + j] for j in range(size)] for i in range(size)])
    D = triangle * phi
    result = []
    for order in range(3):
        matrix = b.acb_mat(2 * d, 2 * d)
        for i in range(d):
            for j in range(d):
                if order == 0:
                    matrix[i, j], matrix[d + i, d + j] = D[i, j], D[d + i, d + j]
                elif order == 1:
                    matrix[i, d + j], matrix[d + i, j] = D[i, d + j], D[d + i, 2 * d + j]
                else:
                    matrix[i, j] = 2 * D[i, 2 * d + j]
        result.append(matrix)
    return tuple(result)


class Functions:
    """Shared formula DAG on the exact continuum; no weighted reprojection."""

    def __init__(self, discretization):
        self.d, self.b = discretization, discretization.b
        self.nodes = [("zero", (), None, frozenset())]
        self.ancestors_cache, self.projections = {}, {}
        self.ground = self.put("ground", (), None, (0,))

    def put(self, kind, parents, data, channels):
        index = len(self.nodes)
        self.nodes.append((kind, tuple(parents), data, frozenset(channels)))
        return index

    def channels(self, index):
        return self.nodes[index][3]

    def add(self, *terms):
        terms = tuple(index for index in terms if index != 0)
        if not terms:
            return 0
        if len(terms) == 1:
            return terms[0]
        channels = set().union(*(self.channels(index) for index in terms))
        return self.put("add", terms, None, channels)

    def scaled(self, factor, function):
        if function == 0:
            return 0
        return self.put("scale", (function,), self.b.complex(factor), self.channels(function))

    def weighted(self, multiplier, function):
        if multiplier == "h11" and self.channels(function) - {0}:
            raise RuntimeError("Only the declared radial h11 block is a free-second multiplier")
        return self.put("multiply", (function,), multiplier, self.channels(function)) if function else 0

    def polynomial(self, coefficients, channels):
        if coefficients.nrows() != 2 * self.d.dimension or coefficients.ncols() != 1:
            raise RuntimeError("A function coefficient column leaves the declared feature range")
        sparse = tuple((index, coefficients[index, 0])
                       for index in range(coefficients.nrows()) if not coefficients[index, 0].is_zero())
        return self.put("polynomial", (), sparse, channels) if sparse else 0

    def needed(self, index):
        if index not in self.ancestors_cache:
            ancestors, pending = set(), [index]
            while pending:
                node = pending.pop()
                if node not in ancestors:
                    ancestors.add(node)
                    pending.extend(self.nodes[node][1])
            self.ancestors_cache[index] = tuple(sorted(ancestors))
        return self.ancestors_cache[index]

    def value(self, index, panel_index, radius, analytic=False, reflected=False):
        b, d, panel = self.b, self.d, self.d.panels[panel_index]
        local_values, basis_values, scalar_weights = {}, None, {}
        for node in self.needed(index):
            kind, parents, data, channels = self.nodes[node]
            if kind == "zero":
                value = (b.complex(0), b.complex(0))
            elif kind == "ground":
                # All ground formula panels are original E data.  A new
                # polynomial or normalization is never fitted here.
                e_index = d.ground_panel_indices[panel_index]
                if e_index is None:
                    if panel.left < d.ground.support:
                        raise Unresolved("An internal E boundary cannot be treated as zero ground support")
                    value = (b.complex(0), b.complex(0))  # Wholly beyond the certified E support only.
                else:
                    # The exported a,b,E,normalization are real balls;
                    # their holomorphic reflected action equals itself.
                    value = (d.ground.value_on_panel(e_index, radius, analytic), b.complex(0))
            elif kind == "polynomial":
                if basis_values is None:
                    basis_values = legendre_values(b, panel, d.configuration.counts[panel_index], radius)
                values = [b.complex(0), b.complex(0)]
                offset, count = d.offsets[panel_index], d.configuration.counts[panel_index]
                for coefficient_index, coefficient in data:
                    channel, global_index = divmod(coefficient_index, d.dimension)
                    if offset <= global_index < offset + count:
                        constant = coefficient.conjugate() if reflected else coefficient
                        values[channel] += constant * basis_values[global_index - offset]
                value = tuple(values)
            elif kind == "add":
                value = tuple(sum((local_values[parent][channel] for parent in parents), b.complex(0))
                              for channel in (0, 1))
            elif kind == "scale":
                constant = data.conjugate() if reflected else data
                value = tuple(constant * v for v in local_values[parents[0]])
            elif kind == "multiply":
                if data not in scalar_weights:
                    if data == "h11":
                        multiplier = d.weights.free_second_radial(radius, analytic)
                    else:
                        multiplier = d.weights.free(radius, F(data), analytic)
                    scalar_weights[data] = multiplier
                value = tuple(scalar_weights[data] * v for v in local_values[parents[0]])
            else:
                raise RuntimeError("Unknown continuous-function formula node")
            if any(not v.is_finite() for v in value):
                raise AnalyticDomain("A continuous formula node is not analytic on this integration ball")
            local_values[node] = value
        return local_values[index]

    def inner(self, left, right):
        b, d, result = self.b, self.d, self.b.complex(0)
        common = self.channels(left) & self.channels(right)
        if not common:
            return result
        for panel_index, panel in enumerate(d.panels):
            def integrand(r, analytic):
                # Reflected holomorphic adjoint of constants, never
                # conjugation of the complex integration variable.
                v = self.value(left, panel_index, r, analytic, reflected=True)
                w = self.value(right, panel_index, r, analytic)
                weight = d.weights.joint(r, analytic)[2] if 1 in common else b.complex(0)
                return ((v[0] * w[0] if 0 in common else b.complex(0))
                        + (v[1] * weight * w[1] if 1 in common else b.complex(0)))
            result += d.integrals.integrate(integrand, panel, "Exact weighted source inner product")
        return result

    def norm_squared(self, function):
        value = real_scalar(self.b, self.inner(function, function), "Continuous squared norm")
        if value.upper() < 0:
            raise Unresolved("A continuous squared norm enclosure is strictly negative")
        return self.b.hull(max(self.b.real(0), value.lower()), max(self.b.real(0), value.upper()))

    def norm(self, function):
        return self.norm_squared(function).sqrt().upper()

    def project(self, function):
        if function in self.projections:
            return self.projections[function]
        b, d = self.b, self.d
        column = b.acb_mat(2 * d.dimension, 1)
        for panel_index, panel in enumerate(d.panels):
            count, offset = d.configuration.counts[panel_index], d.offsets[panel_index]
            for channel in self.channels(function):
                for degree in range(count):
                    def integrand(r, analytic, degree=degree, channel=channel):
                        v = legendre_values(b, panel, count, r)[degree]
                        value = self.value(function, panel_index, r, analytic)[channel]
                        weight = d.weights.joint(r, analytic)[2] if channel == 1 else b.complex(1)
                        return v * weight * value
                    column[channel * d.dimension + offset + degree, 0] = d.integrals.integrate(
                        integrand, panel, "Physical feature/source projection")
        self.projections[function] = column
        return column


class FiniteModel:
    """Required exponential insertions acting on continuous weighted sources."""

    def __init__(self, discretization):
        self.d, self.b = discretization, discretization.b
        self.functions = Functions(discretization)
        self.T = exponential_insertions(discretization, self.b.real(1))
        self.M = exponential_insertions(discretization, self.b.real(-2))
        # The same triangular identity gives exp(-S); only its triplet
        # diagonal is used for Y.  No candidate BLAS arithmetic is used.
        self.Y = exponential_insertions(discretization, self.b.real(-1))[0]
        self.coefficient_workspace_entries = sum(matrix.nrows() * matrix.ncols()
                                                 for matrix in (*self.T, *self.M, self.Y))

    def apply(self, family, order, function):
        f, b, d = self.functions, self.b, self.d.dimension
        if family == "Y":
            if order != 0 or f.channels(function) - {1}:
                raise RuntimeError("Y is used only in the triplet response")
            matrix, channels = self.Y, (1,)
        else:
            matrix = {"T": self.T, "M": self.M}[family][order]
            if order == 2:
                if f.channels(function) - {0}:
                    raise RuntimeError("Only the radial second insertion is supplied")
                channels = (0,)
            elif order == 1:
                channels = tuple(1 - value for value in f.channels(function))
            else:
                channels = tuple(f.channels(function))
        coefficients = matrix * f.project(function)
        correction = f.polynomial(coefficients, channels)
        return f.add(function, correction) if order == 0 else correction

    def H1(self, function):
        f = self.functions
        return f.add(self.apply("T", 1, f.weighted("1", self.apply("T", 0, function))),
                     self.apply("T", 0, f.weighted("1", self.apply("T", 1, function))))

    def H11(self, function):
        f = self.functions
        if f.channels(function) - {0}:
            raise RuntimeError("Only the radial H11 expectation contract is supplied")
        return f.add(self.apply("T", 2, f.weighted("1", self.apply("T", 0, function))),
                     self.apply("T", 0, f.weighted("1", self.apply("T", 2, function))),
                     f.scaled(2, self.apply("T", 1, f.weighted("1", self.apply("T", 1, function)))),
                     self.apply("T", 0, f.weighted("h11", self.apply("T", 0, function))))

    def block_norm(self, family, order, output_channel, input_channel):
        matrices = {"T": self.T, "M": self.M, "Y": (self.Y,)}[family]
        matrix = block(matrices[order], output_channel * self.d.dimension,
                       input_channel * self.d.dimension, self.d.dimension, self.b)
        return self.d.physical_coefficient_norm(matrix, output_channel, input_channel)

    def neumann(self, source, energy, include_Y, degree):
        """Positive-pencil inverse with exact h^(-1/2) at every occurrence."""
        f, b = self.functions, self.b
        b.positive(energy, "Approximate positive-pencil energy")
        q = (energy / 2).upper()
        b.positive(1 - q, "Actual approximate Neumann ratio margin")
        transformed = self.apply("Y", 0, source) if include_Y else source
        zeta = f.weighted("-1/2", transformed)
        norm_squared = f.norm_squared(zeta)
        current, summed, scalar = zeta, 0, b.complex(0)
        for n in range(degree + 1):
            self.d.work.check()
            scalar += f.inner(zeta, current)
            summed = f.add(summed, current)
            if n < degree:
                current = f.scaled(energy, f.weighted("-1/2", self.apply(
                    "M", 0, f.weighted("-1/2", current))))
        inverse_vector = f.weighted("-1/2", summed)
        if include_Y:
            inverse_vector = self.apply("Y", 0, inverse_vector)
        geometric = q**(degree + 1) / (1 - q)
        scalar_tail = (norm_squared.upper() * geometric).upper()
        vector_tail = (norm_squared.sqrt().upper() * geometric / b.real(2).sqrt()).upper()
        value = real_scalar(b, scalar, "Positive-pencil Neumann quadratic form")
        # Omitted powers are nonnegative in this scalar form.  The tail
        # is one-sided; directed integration radii remain in value.
        value += b.hull(b.real(0), scalar_tail)
        return value, inverse_vector, {"ratio_upper": q, "last_degree": degree,
                "transformed_source_norm_squared": norm_squared,
                "scalar_geometric_remainder_upper": scalar_tail,
                "vector_geometric_remainder_upper": vector_tail}


def propagation(model, norms, errors):
    """Reviewed norm balls and product telescoping, with all global free energies."""
    b, g = model.b, model.d.ground
    totals = {name: values["total"] for name, values in errors.items()}
    dS = max(totals["Sr"], totals["St"])
    s = max(norms["Sr"] + totals["Sr"], norms["St"] + totals["St"])
    s1, s11 = norms["S1"] + totals["S1"], norms["S11"] + totals["S11"]
    es, e2s = s.exp(), (2 * s).exp()
    hmax = (2 * (1 + 3 / model.d.weights.alpha**2).sqrt()).upper()
    # From the literal native h11=(cos(2 alpha k1)/epsilon
    # -q1^2/epsilon^3)/2, |q1|<=r and epsilon>=1 give |h11|<=1
    # on the full torus, including regions outside the spherical chart.
    h11max = b.real(1)
    nhTR = b.real(1)  # Structural Sr<=0; normalized E input is retained.
    nhTT = min((1 + model.block_norm("T", 0, 1, 1)).upper(), norms["St"].exp().upper())
    nT1 = max(model.block_norm("T", 1, 1, 0), model.block_norm("T", 1, 0, 1))
    nT11 = model.block_norm("T", 2, 0, 0)
    nM1 = max(model.block_norm("M", 1, 1, 0), model.block_norm("M", 1, 0, 1))
    nM11 = model.block_norm("M", 2, 0, 0)
    nhH1 = (hmax * nT1 * (nhTR + nhTT)).upper()
    nhH11 = (2 * hmax * nT11 * nhTR + 2 * hmax * nT1**2
             + h11max * nhTR**2).upper()

    def elementary(ds, ds1, ds11):
        dT = es * ds
        dT1 = es * (ds1 + s1 * ds)
        dT11 = es * (ds11 + s11 * ds + 2 * s1 * ds1 + s1**2 * ds)
        dM = 2 * e2s * ds
        dM1 = e2s * (2 * ds1 + 4 * s1 * ds)
        dM11 = e2s * (2 * ds11 + 4 * s11 * ds + 8 * s1 * ds1 + 8 * s1**2 * ds)
        return dT, dT1, dT11, dM, dM1, dM11

    all_elementary = elementary(dS, totals["S1"], totals["S11"])
    dT, dT1, dT11, dM, dM1, dM11 = all_elementary
    tn = min(es.upper(), (max(nhTR, nhTT) + dT).upper())
    t1n = min((es * s1).upper(), (nT1 + dT1).upper())
    t11n = min((es * (s11 + s1**2)).upper(), (nT11 + dT11).upper())

    def products(ds, ds1, ds11):
        dt, dt1, dt11, dm, dm1, dm11 = elementary(ds, ds1, ds11)
        dh1 = 2 * hmax * (dt1 * tn + t1n * dt)
        dh11 = (2 * hmax * (dt11 * tn + t11n * dt)
                + 4 * hmax * t1n * dt1 + 2 * h11max * tn * dt)
        return {"T": dt.upper(), "T1": dt1.upper(), "T11": dt11.upper(),
                "M": dm.upper(), "M1": dm1.upper(), "M11": dm11.upper(),
                "H1": dh1.upper(), "H11": dh11.upper()}

    total = products(dS, totals["S1"], totals["S11"])
    allocations = {}
    for category in ("tail", "projection", "quadrature", "angular", "coupling", "arithmetic"):
        # Sum, rather than max, makes these category bounds add to an
        # upper bound for the total common diagonal perturbation.
        ds = errors["Sr"][category] + errors["St"][category]
        allocations[category] = products(ds, errors["S1"][category], errors["S11"][category])
        allocations[category]["Sr"] = errors["Sr"][category]
        allocations[category]["St"] = errors["St"][category]
    d = b.positive(2 - g.energy_exact.upper(), "Full physical triplet response gap").lower()
    dhat = b.positive(2 - g.energy_approximate, "Approximate positive-pencil gap").lower()
    dY = totals["St"]
    dR = (2 * dY / d
          + (g.energy_error + 2 * g.energy_approximate.abs_upper() * dY) / (d * dhat)).upper()
    # The prescribed general dM bound is retained above.  In the positive
    # triplet block M_t=Y^2, contraction additionally gives dM_t<=2dS_t.
    dMt = min((2 * totals["St"]).upper(), (2 * e2s * totals["St"]).upper())
    dA = (g.energy_error + g.energy_approximate.abs_upper() * dMt).upper()
    dAinv = (dA / (d * dhat)).upper()
    return {"s": s.upper(), "s1": s1.upper(), "s11": s11.upper(),
            "exp_s": es.upper(), "exp_2s": e2s.upper(),
            "h_full_native_norm_upper": hmax, "h11_full_native_norm_upper": h11max,
            "hat_T_radial_norm_upper": nhTR, "hat_T_triplet_norm_upper": nhTT,
            "hat_T1_norm_upper": nT1, "hat_T11_RR_norm_upper": nT11,
            "hat_H1_norm_upper": nhH1, "hat_H11_RR_norm_upper": nhH11,
            "hat_M1_CR_norm_upper": nM1, "hat_M11_RR_norm_upper": nM11,
            "total": total, "allocations": allocations,
            "triplet_gap_lower": d, "approximate_gap_lower": dhat,
            "Y_error_upper": dY, "M_triplet_error_upper": dMt,
            "resolvent_error_upper": dR, "pencil_error_upper": dA,
            "pencil_inverse_error_upper": dAinv}


def curvature_evaluation(model, propagated):
    """Two separately evaluated/enclosed curvature routes, using the same E input."""
    b, f, g, p = model.b, model.functions, model.d.ground, propagated
    u, energy, eh = f.ground, g.energy_exact, g.energy_approximate
    norm_identity = f.norm_squared(u)
    if not norm_identity.contains(1):
        raise Unresolved("Actual correlated E input's unit-norm identity is not enclosed")
    source = model.H1(u)
    source_norm = f.norm(source)
    direct76 = real_scalar(b, f.inner(u, model.H11(u)), "N76 direct radial expectation")
    response76, inverse_source, tail76 = model.neumann(
        source, eh, True, model.d.configuration.neumann_degree)
    dchi = (p["total"]["H1"] + p["hat_H1_norm_upper"] * g.vector_error).upper()
    response_error76 = (dchi * (2 * source_norm + dchi) / p["triplet_gap_lower"]
                       + p["resolvent_error_upper"] * source_norm**2).upper()
    direct_error76 = (p["total"]["H11"] + 2 * p["hat_H11_RR_norm_upper"] * g.vector_error).upper()
    error76 = (direct_error76 + 2 * response_error76).upper()
    derivative76 = b.widen(direct76 - 2 * response76, error76)
    COM76 = energy * derivative76
    u1 = f.scaled(-1, inverse_source)
    u1_error = (dchi / p["triplet_gap_lower"]
                + p["resolvent_error_upper"] * source_norm
                + tail76["vector_geometric_remainder_upper"]).upper()

    # N79 is evaluated from its own metric derivatives and source.  The
    # imported vector is not assumed an eigenvector of this compact Hhat.
    y = model.apply("T", 0, u)
    metric_identity = real_scalar(b, f.inner(y, model.apply("M", 0, y)), "N79 metric normalization")
    if not metric_identity.contains(1):
        raise Unresolved("N79 metric unit identity is not enclosed; y must not be renormalized")
    z = model.apply("M", 1, y)
    z_norm = f.norm(z)
    direct_function79 = f.add(f.weighted("h11", y),
                               f.scaled(-eh, model.apply("M", 2, y)))
    direct79 = real_scalar(b, f.inner(y, direct_function79), "N79 direct pencil expectation")
    response79, _, tail79 = model.neumann(z, eh, False, model.d.configuration.neumann_degree)
    dy = (model.d.norm_errors["Sr"]["total"] + g.vector_error).upper()
    dz = (p["total"]["M1"] + p["hat_M1_CR_norm_upper"] * dy).upper()
    response_error79 = (dz * (2 * z_norm + dz) / p["triplet_gap_lower"]
                       + p["pencil_inverse_error_upper"] * z_norm**2).upper()
    direct_norm79 = (p["h11_full_native_norm_upper"]
                    + eh.abs_upper() * p["hat_M11_RR_norm_upper"]).upper()
    direct_error79 = (2 * direct_norm79 * dy + g.energy_error * p["hat_M11_RR_norm_upper"]
                     + energy.upper() * p["total"]["M11"]).upper()
    error79 = (direct_error79 + 2 * energy.upper()**2 * response_error79).upper()
    derivative79 = b.widen(direct79 - 2 * energy**2 * response79, error79)
    COM79 = energy * derivative79
    overlap = COM76.overlaps(COM79)
    final = b.intersect(COM76, COM79, "Independently propagated N76/N79 COM intervals") if overlap else None
    d, dhat = p["triplet_gap_lower"], p["approximate_gap_lower"]
    allocations, drivers = {}, {}
    for category, values in p["allocations"].items():
        dc = values["H1"]
        dR = (2 * values["St"] / d + 2 * eh.abs_upper() * values["St"] / (d * dhat)).upper()
        e76 = (values["H11"] + 2 * (dc * (2 * source_norm + dchi) / d
                                            + dR * source_norm**2)).upper()
        dycat = values["Sr"]
        dzcat = values["M1"] + p["hat_M1_CR_norm_upper"] * dycat
        e79 = (2 * direct_norm79 * dycat + energy.upper() * values["M11"]
               + 2 * energy.upper()**2 * (dzcat * (2 * z_norm + dz) / d
                    + 2 * eh.abs_upper() * values["St"] * z_norm**2 / (d * dhat))).upper()
        allocations[category] = {"N76_curvature_error_upper": e76, "N79_curvature_error_upper": e79}
        drivers[category] = (energy.upper() * max(e76, e79)).upper()
    dc_ground = p["hat_H1_norm_upper"] * g.vector_error
    e76_ground = (2 * p["hat_H11_RR_norm_upper"] * g.vector_error
                 + 2 * dc_ground * (2 * source_norm + dchi) / d).upper()
    e79_ground = (2 * direct_norm79 * g.vector_error + 2 * energy.upper()**2
                 * p["hat_M1_CR_norm_upper"] * g.vector_error * (2 * z_norm + dz) / d).upper()
    drivers["ground_input"] = (energy.upper() * max(e76_ground, e79_ground)).upper()
    e76_energy = 2 * g.energy_error * source_norm**2 / (d * dhat)
    e79_energy = (g.energy_error * p["hat_M11_RR_norm_upper"]
                  + 2 * energy.upper()**2 * g.energy_error * z_norm**2 / (d * dhat))
    outer_energy = g.energy_error * max(derivative76.abs_upper(), derivative79.abs_upper())
    # The N79 outer E^2 factor is already interval propagated.  Its
    # contribution is recorded independently for refinement selection.
    e79_energy += 2 * g.energy_error * (2 * eh.abs_upper() + g.energy_error) * response79.abs_upper()
    drivers["energy_input"] = (energy.upper() * max(e76_energy, e79_energy) + outer_energy).upper()
    drivers["Neumann"] = (2 * energy.upper() * max(tail76["scalar_geometric_remainder_upper"],
                         energy.upper()**2 * tail79["scalar_geometric_remainder_upper"])).upper()
    # This additional measured *ball* radius is certified, but combines
    # rounding, angular-integral and imported-family uncertainty.  No
    # point center is substituted to separate correlated data.
    numerical_radius = max((energy * (direct76 - 2 * response76)
                             - (energy * (direct76 - 2 * response76)).mid()).abs_upper(),
                            (energy * (direct79 - 2 * energy**2 * response79)
                             - (energy * (direct79 - 2 * energy**2 * response79)).mid()).abs_upper())
    drivers["arithmetic"] = (drivers["arithmetic"] + numerical_radius).upper()
    return {"ground_norm_identity": norm_identity, "metric_norm_identity": metric_identity,
            "N76_direct": direct76, "N76_response": response76,
            "N76_direct_error_upper": direct_error76,
            "N76_response_error_upper": response_error76,
            "N76_curvature": derivative76, "N76_COM": COM76,
            "N79_direct": direct79, "N79_response": response79,
            "N79_direct_error_upper": direct_error79,
            "N79_response_error_upper": response_error79,
            "N79_curvature": derivative79, "N79_COM": COM79,
            "independent_intervals_overlap": overlap, "COM_intersection": final,
            "N76_Neumann": tail76, "N79_Neumann": tail79,
            "source_chi_norm_upper": source_norm, "source_chi_error_upper": dchi,
            "source_z_norm_upper": z_norm, "source_z_error_upper": dz,
            "ground_phase_aligned_tail_upper_beyond_cutoff": g.vector_error,
            "first_derivative_vector_error_and_true_tail_upper": u1_error,
            "compact_derivative_norm_upper": f.norm(u1),
            "compact_ground_and_derivative_support_at_most_cutoff": True,
            "environment_window_symbolic": "|p_env,i| <= a^-1 arcsin(alpha L)",
            "full_interacting_Ward_window_proved": False,
            "operator_error_allocations": allocations, "refinement_driver_bounds": drivers,
            "numerical_ball_radius_upper_including_imported_correlations": numerical_radius}


def encode_tree(backend, value):
    if isinstance(value, (backend.arb, backend.acb)):
        return backend.encode(value)
    if isinstance(value, dict):
        return {key: encode_tree(backend, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_tree(backend, item) for item in value]
    return value


def scalar_product(left, right):
    # Avoid relying on a C-extension scalar's coercion/fallback when
    # the right operand belongs to our formal polynomial algebra.
    return right * left if isinstance(right, Polynomial) and not isinstance(left, Polynomial) else left * right


def tensor_entries(left, right):
    return [[scalar_product(left[i][j], right[k][ell])
             for j in range(len(left[0])) for ell in range(len(right[0]))]
            for i in range(len(left)) for k in range(len(right))]


def matrix_entries_add(*matrices):
    return [[sum((matrix[i][j] for matrix in matrices), 0)
             for j in range(len(matrices[0][0]))] for i in range(len(matrices[0]))]


def matrix_entries_scale(matrix, factor):
    return [[scalar_product(value, factor) for value in row] for row in matrix]


def matrix_entries_product(left, right):
    return [[sum((scalar_product(left[i][k], right[k][j]) for k in range(len(right))), 0)
             for j in range(len(right[0]))] for i in range(len(left))]


def spinor_entries(backend, top, vector):
    ii = backend.acb(0, 1)
    return [[top, 0], [0, top],
            [vector[2], vector[0] - vector[1] * ii],
            [vector[0] + vector[1] * ii, -vector[2]]]


def literal_spinor_jets(backend, sine_vector, cosine_vector, epsilon):
    """NC3; also usable over finite polynomials for the independent shell path."""
    b, d, C = backend, sine_vector, cosine_vector
    c = ((epsilon + 1) / (2 * epsilon)).sqrt()
    g = 1 / (2 * epsilon * (epsilon + 1)).sqrt()
    epsilon1 = d[0] * C[0] / epsilon
    epsilon11 = ((C[0]**2 - d[0]**2 * b.real(F(1, 10**12))) / epsilon
                  - d[0]**2 * C[0]**2 / epsilon**3)
    ce = -1 / (4 * c * epsilon**2)
    cee = 1 / (2 * c * epsilon**3) - 1 / (16 * c**3 * epsilon**4)
    ag = -1 / (2 * epsilon) - 1 / (2 * (epsilon + 1))
    ge = g * ag
    gee = g * (ag**2 + 1 / (2 * epsilon**2) + 1 / (2 * (epsilon + 1)**2))
    lower0 = tuple(value * g for value in d)
    lower1 = tuple(epsilon1 * ge * value + (C[0] * g if j == 0 else 0)
                   for j, value in enumerate(d))
    lower11 = tuple((epsilon1**2 * gee + epsilon11 * ge) * value
                    + (epsilon1 * (2 * ge) * C[0] - d[0] * (g * b.real(F(1, 10**12))) if j == 0 else 0)
                    for j, value in enumerate(d))
    return (spinor_entries(b, c, lower0), spinor_entries(b, epsilon1 * ce, lower1),
            spinor_entries(b, epsilon1**2 * cee + epsilon11 * ce, lower11), epsilon11)


def literal_pair_jets(backend, sine_vector, cosine_vector, epsilon):
    plus = literal_spinor_jets(backend, sine_vector, cosine_vector, epsilon)
    minus = literal_spinor_jets(backend, tuple(-value for value in sine_vector), cosine_vector, epsilon)
    U0 = tensor_entries(plus[0], minus[0])
    U1 = matrix_entries_scale(matrix_entries_add(tensor_entries(plus[1], minus[0]),
                              tensor_entries(plus[0], minus[1])), backend.real(F(1, 2)))
    U11 = matrix_entries_scale(matrix_entries_add(tensor_entries(plus[2], minus[0]),
             matrix_entries_scale(tensor_entries(plus[1], minus[1]), 2),
             tensor_entries(plus[0], minus[2])), backend.real(F(1, 4)))
    return U0, U1, U11, plus[3] / 2


def bits(index):
    return tuple((index >> j) & 1 for j in range(3))


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def literal_filter_jets(backend, cosine_vector, cosine_twice):
    """NC4 without division by a possibly vanishing filter factor; R2 removed."""
    legs, legs11 = [], []
    for index in range(8):
        nu = bits(index)
        factors = tuple((1 + (-1)**nu[j] * cosine_vector[j]) / 2 for j in range(3))
        base = product(factors)
        legs.append(base)
        legs11.append(-scalar_product(cosine_twice, base) / 2
                       - factors[1] * factors[2] * backend.real(F(1, 10**12)) / 8)
    return legs, legs11


def hadamard_legs(backend, legs, legs11):
    scale = 1 / backend.real(8).sqrt()
    return tuple((sum(((-1)**sum(nu[j] * mu[j] for j in range(3)) * legs[index]
                         for index in range(8) for nu in (bits(index),)), 0) * scale,
                  sum(((-1)**sum(nu[j] * mu[j] for j in range(3)) * legs11[index]
                         for index in range(8) for nu in (bits(index),)), 0) * scale)
                 for mu in (bits(index) for index in range(8)))


class NativeVertex:
    """Full coupling-normalized native spin matrix, independent of NK amplitudes."""

    def __init__(self, backend, work):
        self.b, self.work = backend, work
        self.alpha = backend.real(F(1, 10**6))
        signs = (1, 1, -1, -1)
        self.BD = tuple(signs[i] * signs[j] for i in range(4) for j in range(4))

    def angles(self, sine_vector, tau):
        values = []
        for value, sign in zip(sine_vector, tau):
            angle = (self.alpha * self.b.real(value)).asin()
            if sign == -1:
                angle = self.b.arb.pi() - angle
                if value < 0:
                    angle -= 2 * self.b.arb.pi()
            values.append(angle)
        return tuple(values)

    def endpoint(self, angles, kappa):
        b, alpha = self.b, self.alpha
        plus = tuple((angle + alpha * kappa[j] / 2).sin() / alpha for j, angle in enumerate(angles))
        minus = tuple((-angle + alpha * kappa[j] / 2).sin() / alpha for j, angle in enumerate(angles))
        rplus2, rminus2 = sum((x**2 for x in plus), b.real(0)), sum((x**2 for x in minus), b.real(0))
        ep, em = (1 + rplus2).sqrt(), (1 + rminus2).sqrt()
        def spinor(d, e):
            c = ((e + 1) / (2 * e)).sqrt()
            g = 1 / (2 * e * (e + 1)).sqrt()
            return b.acb_mat(spinor_entries(b, c, tuple(g * x for x in d)))
        U = b.acb_mat(tensor_entries(spinor(plus, ep).tolist(), spinor(minus, em).tolist()))
        regulator = (-(rplus2 + rminus2) / 2).exp()
        matrices = []
        for index in range(8):
            nu = bits(index)
            leg = regulator * product(((alpha * kappa[j] / 2).cos()
                            + (-1)**nu[j] * angles[j].cos()) / 2 for j in range(3))
            matrices.append(U * leg)
        return ep + em, tuple(matrices)

    def jets(self, angles):
        b, alpha = self.b, self.alpha
        d = tuple(angle.sin() / alpha for angle in angles)
        C = tuple(angle.cos() for angle in angles)
        r2 = sum((value**2 for value in d), b.real(0))
        epsilon = (1 + r2).sqrt()
        U0, U1, U11, h11 = literal_pair_jets(b, d, C, epsilon)
        U0, U1, U11 = (b.acb_mat(value) for value in (U0, U1, U11))
        legs, legs11 = literal_filter_jets(b, C, (2 * angles[0]).cos())
        regulator = (-r2).exp()
        return 2 * epsilon, h11, tuple((U0 * (regulator * leg), U1 * (regulator * leg),
                         (U0 * leg11 + U11 * leg) * regulator) for leg, leg11 in zip(legs, legs11))

    def contraction(self, left, right):
        b = self.b
        return b.acb_mat([[sum((left[k, i].conjugate() * self.BD[k] * right[k, j]
                               for k in range(16)), b.complex(0)) for j in range(4)] for i in range(4)])

    def matrix(self, out_, in_):
        denominator = self.b.positive(out_[0] + in_[0], "NC2 real complete excitation denominator")
        total = self.b.acb_mat(4, 4)
        for first, second in zip(out_[1], in_[1]):
            total += self.contraction(first, second)
        return total * (-2 / denominator)

    def matrix_jets(self, out_, in_):
        denominator = self.b.positive(out_[0] + in_[0], "NC5 real rest excitation denominator")
        zeroth, first, second = (self.b.acb_mat(4, 4) for _ in range(3))
        for a, c in zip(out_[2], in_[2]):
            zeroth += self.contraction(a[0], c[0])
            first += self.contraction(a[1], c[0]) + self.contraction(a[0], c[1])
            second += (self.contraction(a[2], c[0]) + 2 * self.contraction(a[1], c[1])
                        + self.contraction(a[0], c[2]))
        return zeroth * (-2 / denominator), first * (-2 / denominator), (
            second * (-2 / denominator) + zeroth * (2 * (out_[1] + in_[1]) / denominator**2))


def native_difference_controls(backend, work):
    b, native = backend, NativeVertex(backend, work)
    vectors = tuple(tuple(F(value) for value in vector) for vector in
                    (("0", "0", "0"), ("1/3", "1/4", "1/5"),
                     ("-2/5", "1/6", "1/7"), ("2", "-1", "1/2")))
    preimages = tuple(tuple((-1)**value for value in bits(index)) for index in range(8))
    alpha, eta = b.real(F(1, 10**6)), b.real(F(1, 32))
    bb = (alpha * eta / 2).sinh() / alpha
    delta = b.positive(1 - 3 * bb**2, "NC6 full native Cauchy root margin")
    q = b.positive((alpha * eta).cos(), "NC6 Gaussian margin")
    u2 = b.real(F(3, 2)) * (1 + 1 / delta.sqrt())
    aa = (alpha * eta / 2).cosh()**3 * (3 * ((alpha * eta).cosh() - 1) / (2 * alpha**2)).exp()
    unit_bound = 4 * aa**2 * u2**2 / delta.sqrt()  # Whole exact constant ball, not its upper endpoint.
    cache = {}
    for index, vector in enumerate(vectors):
        for preimage, tau in enumerate(preimages):
            angles = native.angles(vector, tau)
            cache[index, preimage] = angles, native.jets(angles)
    largest_ratio, largest_radius, checks = b.real(0), b.real(0), []
    all_pass, count = True, 0
    for out_vector, in_vector in ((0, 0), (1, 2), (3, 1)):
        radial_sum = sum((b.real(value)**2 for value in (*vectors[out_vector], *vectors[in_vector])), b.real(0))
        M = unit_bound * (-q * radial_sum).exp()
        for output_preimage in range(8):
            for input_preimage in range(8):
                work.check(1)
                out_angles, out_jets = cache[out_vector, output_preimage]
                in_angles, in_jets = cache[in_vector, input_preimage]
                exact_jets = native.matrix_jets(out_jets, in_jets)
                point_pass, point_ratios = True, []
                for power in (20, 24):
                    step = b.real(F(1, 2**power))
                    zeros = (b.real(0), b.real(0))
                    plus = native.matrix(native.endpoint(out_angles, (step, *zeros)),
                                         native.endpoint(in_angles, (step, *zeros)))
                    minus = native.matrix(native.endpoint(out_angles, (-step, *zeros)),
                                          native.endpoint(in_angles, (-step, *zeros)))
                    differences = ((plus - minus) / (2 * step),
                                   (plus - 2 * exact_jets[0] + minus) / step**2)
                    R1 = M * step**2 / (eta**3 * (1 - step**2 / eta**2))
                    R2 = 2 * M * step**2 / (eta**4 * (1 - step**2 / eta**2))
                    ratios = []
                    for order, (difference, target, remainder) in enumerate(zip(differences, exact_jets[1:], (R1, R2)), 1):
                        allowance = b.positive(2 * remainder, "Exact NC6 twice-remainder allowance").lower()
                        ratio = b.real(0)
                        for i in range(4):
                            for j in range(4):
                                error = (difference[i, j] - target[i, j]).abs_upper()
                                point_pass &= bool(error <= allowance)
                                ratio = max(ratio, (error / allowance).upper())
                                largest_radius = max(largest_radius,
                                    (difference[i, j] - difference[i, j].mid()).abs_upper(),
                                    (target[i, j] - target[i, j].mid()).abs_upper())
                        ratios.append(ratio)
                        largest_ratio = max(largest_ratio, ratio)
                    point_ratios.append({"step_exact": str(F(1, 2**power)), "entry_ratios_to_lower_2R": ratios,
                                         "R1": R1, "R2": R2})
                checks.append({"output_sine_vector": out_vector, "input_sine_vector": in_vector,
                               "output_cos_sign_preimage": output_preimage, "input_cos_sign_preimage": input_preimage,
                               "passed": point_pass, "steps": point_ratios})
                count += 1
                all_pass &= point_pass
    return {"pointwise_only_not_operator_error": True, "unit_coupling_factored_only_for_control": True,
            "all_filters": 8, "native_point_pairs": count, "expected_native_point_pairs": 192,
            "all_independent_cos_sign_preimages_retained": count == 192,
            "both_steps_pass": all_pass, "largest_entry_ratio_to_lower_exact_2R": largest_ratio,
            "largest_arithmetic_ball_radius_upper": largest_radius, "points": checks}


class Polynomial:
    """Finite native monomials in d1,d2,d3,C1,C2,C3, with exact structural parity."""

    def __init__(self, backend, coefficients=None):
        self.b = backend
        self.coefficients = {powers: value for powers, value in (coefficients or {}).items()
                             if not value.is_zero()}

    @classmethod
    def constant(cls, backend, value):
        return cls(backend, {(0,) * 6: backend.complex(value)})

    @classmethod
    def variable(cls, backend, index):
        powers = tuple(int(j == index) for j in range(6))
        return cls(backend, {powers: backend.complex(1)})

    def coerce(self, value):
        return value if isinstance(value, Polynomial) else Polynomial.constant(self.b, value)

    def __add__(self, other):
        other, result = self.coerce(other), dict(self.coefficients)
        for key, value in other.coefficients.items():
            result[key] = result.get(key, self.b.complex(0)) + value
        return Polynomial(self.b, result)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial(self.b, {key: -value for key, value in self.coefficients.items()})

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other, result = self.coerce(other), {}
        for first, a in self.coefficients.items():
            for second, c in other.coefficients.items():
                key = tuple(i + j for i, j in zip(first, second))
                result[key] = result.get(key, self.b.complex(0)) + a * c
        return Polynomial(self.b, result)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        if isinstance(scalar, Polynomial):
            raise RuntimeError("A native polynomial has no variable denominator")
        return self * (1 / self.b.complex(scalar))

    def __pow__(self, power):
        if type(power) is not int or power < 0:
            raise RuntimeError("Native polynomial powers are nonnegative integers")
        result = Polynomial.constant(self.b, 1)
        for _ in range(power):
            result = result * self
        return result

    def conjugate(self):
        # d,C are real formal variables; only coefficient constants are reflected.
        return Polynomial(self.b, {key: value.conjugate() for key, value in self.coefficients.items()})

    def odd_part(self):
        return Polynomial(self.b, {key: value for key, value in self.coefficients.items() if sum(key[:3]) % 2})

    def coefficient(self, powers):
        return self.coefficients.get(tuple(powers), self.b.complex(0))


def choose(n, k):
    result = 1
    for j in range(1, k + 1):
        result = result * (n - j + 1) // j
    return result


class ShellMean:
    """NC7: exact sign/reflection sum followed by certified native conditional moments."""

    def __init__(self, weights, radius):
        self.weights, self.b, self.radius = weights, weights.b, radius
        self.z, self.moments = weights.alpha**2 * radius**2, {}
        self.w0 = weights.moment((0, 0, 0), self.z)
        if self.w0.contains(0):
            raise Unresolved("NC7 native shell normalization contains zero")

    def moment(self, index):
        if index not in self.moments:
            self.moments[index] = self.weights.moment(index, self.z) / self.w0
        return self.moments[index]

    def __call__(self, polynomial):
        if not isinstance(polynomial, Polynomial):
            polynomial = Polynomial.constant(self.b, polynomial)
        total = self.b.complex(0)
        for powers, coefficient in polynomial.coefficients.items():
            ds, cs = powers[:3], powers[3:]
            if any(value % 2 for value in cs) or any(value % 2 for value in ds):
                continue  # Exact cos-preimage/parity identities, no threshold.
            ns = tuple(value // 2 for value in cs)
            for i in range(ns[0] + 1):
                for j in range(ns[1] + 1):
                    for k in range(ns[2] + 1):
                        expansion = (i, j, k)
                        factor = coefficient * product(choose(n, a) for n, a in zip(ns, expansion))
                        factor *= (-self.weights.alpha**2)**sum(expansion)
                        factor *= self.radius**(sum(ds) + 2 * sum(expansion))
                        moment = tuple(d // 2 + a for d, a in zip(ds, expansion))
                        total += factor * self.moment(moment)
        return total


def normalized_spin_wedges(backend):
    b, ii, root2 = backend, backend.acb(0, 1), backend.real(2).sqrt()
    epsilon_s = ((0, 1), (-1, 0))
    pauli = (((0, 1), (1, 0)), ((0, -ii), (ii, 0)), ((1, 0), (0, -1)))
    B0 = [[b.complex(value) / root2 for value in row] for row in epsilon_s]
    triplet_matrices = tuple(matrix_entries_scale(matrix_entries_product(sigma, epsilon_s), 1 / root2)
                             for sigma in pauli)
    singlet = tuple(B0[i][j] for i in range(2) for j in range(2))
    Bt = [[triplet_matrices[column][i][j] for column in range(3)]
          for i in range(2) for j in range(2)]
    wedges = {}
    for name, shift in (("UU", 0), ("LL", 2)):
        vector = [b.complex(0) for _ in range(16)]
        for i in range(2):
            for j in range(2):
                vector[(i + shift) * 4 + j + shift] = B0[i][j]
        wedges[name] = tuple(vector)
    for index, B in enumerate((B0, *triplet_matrices)):
        vector = [b.complex(0) for _ in range(16)]
        for i in range(2):
            for j in range(2):
                vector[i * 4 + j + 2] = B[i][j] / root2
                vector[(i + 2) * 4 + j] = -B[j][i] / root2
        wedges[index] = tuple(vector)
    return singlet, Bt, wedges


def raw_shell(weights, radius):
    """Literal spinor/filter polynomial route; no NK closed G/F identity here."""
    b = weights.b
    d = tuple(Polynomial.variable(b, j) for j in range(3))
    C = tuple(Polynomial.variable(b, j + 3) for j in range(3))
    epsilon = (1 + radius**2).sqrt()
    U0, U1, U11, h11 = literal_pair_jets(b, d, C, epsilon)
    legs, legs11 = literal_filter_jets(b, C, 1 - d[0]**2 * (2 * weights.alpha**2))
    transformed = hadamard_legs(b, legs, legs11)
    singlet, Bt, wedges = normalized_spin_wedges(b)
    columns = tuple(matrix_entries_product(U, [[value] for value in singlet]) for U in (U0, U1, U11))
    triplet_columns = tuple(matrix_entries_product(U, Bt) for U in (U0, U1))
    mean, root8 = ShellMean(weights, radius), b.real(8).sqrt()
    vectors, outputs = [], []
    for leg, leg11 in transformed:
        a = tuple(root8 * mean(leg * columns[0][j][0]) for j in range(16))
        bb = tuple(root8 * mean(leg * columns[1][j][0]) for j in range(16))
        c = tuple(root8 * mean(leg11 * columns[0][j][0] + leg * columns[2][j][0]) for j in range(16))
        zz = tuple(root8 * mean(h11 * leg * columns[0][j][0]) for j in range(16))
        vectors.append({"a": a, "b": bb, "c": c, "z": zz})
        raw0 = tuple(tuple((leg * triplet_columns[0][row][column].conjugate()).odd_part()
                           for row in range(16)) for column in range(3))
        raw1 = tuple(tuple((leg * triplet_columns[1][row][column].conjugate()).odd_part()
                           for row in range(16)) for column in range(3))
        outputs.append((raw0, raw1))
    return epsilon, vectors, outputs, wedges


def shell_controls(weights, work):
    b, tolerance = weights.b, weights.b.real(F(1, 10**20))
    radii = tuple(F(value) for value in ("0", "1/2", "1", "2", "4"))
    kernels = Kernels(weights, b.real(1))
    cache = {}
    largest_amp = largest_K = largest_L = largest_Sr = largest_poly = b.real(0)
    BD = tuple((1 if i < 2 else -1) * (1 if j < 2 else -1) for i in range(4) for j in range(4))
    amplitude_pass = True
    for exact_radius in radii:
        work.check(1)
        radius = b.real(exact_radius)
        raw = raw_shell(weights, radius)
        epsilon, vectors, outputs, wedges = raw
        closed = kernels.amplitudes(b.complex(radius))["normalized_without_qR2"]
        for mu, vector in enumerate(vectors):
            for key, terms in (("a", ((closed["aU"], wedges["UU"]), (closed["aL"], wedges["LL"])) if mu == 0 else ()),
                               ("b", ((closed["b"], wedges[1]),) if mu == 1 else ()),
                               ("c", ((closed["cU"], wedges["UU"]), (closed["cL"], wedges["LL"])) if mu == 0 else None),
                               ("z", ((closed["zU"], wedges["UU"]), (closed["zL"], wedges["LL"])) if mu == 0 else ())):
                if terms is None:
                    continue  # Other c_mu exist; no omitted full block is set to zero.
                for index in range(16):
                    expected = sum((scalar * wedge[index] for scalar, wedge in terms), b.complex(0))
                    error = (vector[key][index] - expected).abs_upper()
                    amplitude_pass &= bool(error <= tolerance.lower())
                    largest_amp = max(largest_amp, error)
        cache[exact_radius] = raw, closed
    pair_pass, checks = True, []
    for output_radius in radii:
        for input_radius in radii:
            work.check(1)
            (epsilon_out, out_vectors, raw_outputs, _), out_closed = cache[output_radius]
            (epsilon_in, in_vectors, _, _), in_closed = cache[input_radius]
            denominator = epsilon_out + epsilon_in
            closed_values = kernels.from_amplitudes(out_closed, in_closed)
            raw_K = []
            for component in range(3):
                polynomial = Polynomial.constant(b, 0)
                for mu in range(8):
                    raw0, raw1 = raw_outputs[mu]
                    polynomial += sum((raw0[component][j] * BD[j] * in_vectors[mu]["b"][j]
                                      + raw1[component][j] * BD[j] * in_vectors[mu]["a"][j]
                                       for j in range(16)), Polynomial.constant(b, 0))
                raw_K.append(-polynomial / denominator)
            # Extract the coefficient of C1*d3 in component2 while d,C
            # remain formal.  Divide only by its nonzero radial constant
            # -i/(4 epsilon), not by r, a feature value or a Gram entry.
            feature_coefficient = -b.acb(0, 1) / (4 * epsilon_out)
            extracted_K = raw_K[1].coefficient((0, 0, 1, 1, 0, 0)) / feature_coefficient
            error_K = (extracted_K - closed_values["K"]).abs_upper()
            expected_K = (Polynomial.constant(b, 0),
                          Polynomial.variable(b, 2) * Polynomial.variable(b, 3) * feature_coefficient * closed_values["K"],
                          -Polynomial.variable(b, 1) * Polynomial.variable(b, 3) * feature_coefficient * closed_values["K"])
            polynomial_error = b.real(0)
            for component in range(3):
                for value in (raw_K[component] - expected_K[component]).coefficients.values():
                    polynomial_error = max(polynomial_error, value.abs_upper())

            def contraction(first, second):
                return sum((first[j].conjugate() * BD[j] * second[j] for j in range(16)), b.complex(0))
            numerator, second_denominator, zeroth = b.complex(0), b.complex(0), b.complex(0)
            for out_, in_ in zip(out_vectors, in_vectors):
                zeroth += contraction(out_["a"], in_["a"])
                numerator += (contraction(out_["c"], in_["a"]) + contraction(out_["a"], in_["c"])
                              + 2 * contraction(out_["b"], in_["b"]))
                second_denominator += contraction(out_["z"], in_["a"]) + contraction(out_["a"], in_["z"])
            raw_L = -numerator / denominator + second_denominator / (2 * denominator**2)
            raw_Sr = -zeroth / denominator
            error_L, error_Sr = (raw_L - closed_values["L11"]).abs_upper(), (raw_Sr - closed_values["Sr"]).abs_upper()
            # E's normalized radial coefficient, independently bound to
            # its published f/t kernel, includes the same q/sqrt8 pullback.
            E_coefficient = -(out_closed["aU"] * in_closed["aU"]
                              + out_closed["aL"] * in_closed["aL"]) / denominator
            error_E = (raw_Sr - E_coefficient).abs_upper()
            passed = all(value <= tolerance.lower() for value in (error_K, polynomial_error, error_L, error_Sr, error_E))
            pair_pass &= passed
            largest_K, largest_L = max(largest_K, error_K), max(largest_L, error_L)
            largest_Sr, largest_poly = max(largest_Sr, error_Sr, error_E), max(largest_poly, polynomial_error)
            checks.append({"output_radius_exact": str(output_radius), "input_radius_exact": str(input_radius),
                           "K_normalized_coefficient_error_upper": error_K,
                           "K_all_raw_output_polynomial_residual_coefficients_upper": polynomial_error,
                           "L11_normalized_error_upper": error_L, "Sr_E_normalized_error_upper": max(error_Sr, error_E),
                           "passed": passed})
    return {"pointwise_only_not_operator_error": True, "exact_native_shell_radii": [str(value) for value in radii],
            "all_25_ordered_pairs": len(checks) == 25, "normalized_tolerance_exact": str(F(1, 10**20)),
            "amplitude_and_exact_zero_block_controls_pass": amplitude_pass,
            "kernel_pair_controls_pass": pair_pass, "independent_raw_triplet_P0_P1_used": True,
            "closed_G_F_identity_used_in_raw_path": False, "both_filter_jets_Hadamard_transformed": True,
            "formal_output_coefficients_extracted_before_zero_radius": True,
            "inverse_Gram_or_numerical_r_division_used": False,
            "native_conditional_moments_used_not_spherical_mean": True,
            "largest_normalized_amplitude_error_upper": largest_amp,
            "largest_K_coefficient_error_upper": largest_K, "largest_L11_error_upper": largest_L,
            "largest_Sr_E_error_upper": largest_Sr, "largest_K_polynomial_residual_coefficient_upper": largest_poly,
            "pairs": checks}


def free_current_controls(backend, work):
    """NC8/N70 at every prescribed preimage and finite-COM argument; no bound branch."""
    b, native = backend, NativeVertex(backend, work)
    alpha, tolerance = native.alpha, b.real(F(1, 10**20))
    vectors = tuple(tuple(F(value) for value in vector) for vector in
                    (("0", "0", "0"), ("1/3", "1/4", "1/5"),
                     ("-2/5", "1/6", "1/7"), ("2", "-1", "1/2")))
    largest, count, passed = b.real(0), 0, True
    for vector in vectors:
        for preimage in range(8):
            tau = tuple((-1)**value for value in bits(preimage))
            angles = native.angles(vector, tau)
            for direction in range(3):
                for step_power in (20, 24):
                    for sign in (-1, 1):
                        work.check(1)
                        kappa = tuple(b.real(F(sign, 2**step_power)) if j == direction else b.real(0)
                                      for j in range(3))
                        theta1 = tuple(theta + alpha * kappa[j] / 2 for j, theta in enumerate(angles))
                        theta2 = tuple(-theta + alpha * kappa[j] / 2 for j, theta in enumerate(angles))
                        sine1 = tuple(theta.sin() / alpha for theta in theta1)
                        sine2 = tuple(theta.sin() / alpha for theta in theta2)
                        omega1 = (1 + sum((value**2 for value in sine1), b.real(0))).sqrt()
                        omega2 = (1 + sum((value**2 for value in sine2), b.real(0))).sqrt()
                        H0, relative_b = omega1 + omega2, (omega1 - omega2) / 2
                        for component in range(3):
                            q1 = (2 * theta1[component]).sin() / (2 * alpha)
                            q2 = (2 * theta2[component]).sin() / (2 * alpha)
                            v1, v2 = q1 / omega1, q2 / omega2
                            H_K, H_k = (v1 + v2) / 2, v1 - v2
                            error = (H0 * H_K + relative_b * H_k - q1 - q2).abs_upper()
                            passed &= bool(error <= tolerance.lower())
                            largest, count = max(largest, error), count + 1
    return {"gamma_exactly_zero_only_for_free_control": True,
            "free_bound_branch_assumed": False, "N1_bound_pair_drift_used": False,
            "all_native_cos_sign_preimages": 8, "comparisons": count,
            "expected_comparisons": 1152, "all_1152_compared": count == 1152,
            "all_directions_components_steps_signs_pass": passed,
            "largest_difference_absolute_upper": largest, "tolerance_exact": str(F(1, 10**20)),
            "identity": "H0*d_kappa H0+b*d_k H0=q1+q2, including the relative dipole term"}


def backend_exponential_gate(backend, work):
    """API/identity prerequisite; rigor is supplied by the directed backend contract."""
    b = backend
    generator = b.acb_mat([[0, 1], [0, 0]])  # Nonnormal, nilpotent; no spectral estimate.
    augmented = b.acb_mat(4, 4)
    for i in range(2):
        for j in range(2):
            augmented[i, j] = generator[i, j]
        augmented[i, 2 + i] = 1
    work.check()
    result = augmented.exp()
    target = b.acb_mat([[1, b.real(F(1, 2))], [0, 1]])
    radius = b.real(F(1, 10**60))
    checks = [result[i, 2 + j].contains(target[i, j])
              and (result[i, 2 + j] - result[i, 2 + j].mid()).abs_upper() < radius
              for i in range(2) for j in range(2)]
    if not matrix_finite(result) or not all(checks):
        raise Unresolved("Directed entire nonnormal phi1 API prerequisite did not pass")
    return {"passed": True, "non_normal_nilpotent_phi1_identity_enclosed": True,
            "requested_radius_goal_is_not_the_arithmetic_proof": True,
            "matrix_function_contract": "python-flint0.9.0 directed acb_mat.exp; entire augmented exponential",
            "primary_documentation": "https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.exp"}


def sign_controls(discretization):
    b, d = discretization.b, discretization
    kernels = Kernels(d.weights, d.ground.gamma_midpoint, truncate_moments=True)
    checked = 0
    for _, radius, weight, _ in d.nodes:
        amplitudes = kernels.amplitudes(b.complex(radius))
        b.positive(weight, "Signed positive-rule Gauss weight")
        epsilon = real_scalar(b, amplitudes["epsilon"], "Signed Cauchy energy")
        b.positive(epsilon, "Signed Cauchy energy")
        for name in ("aU", "aL"):
            real_scalar(b, amplitudes[name], "Signed Cauchy real factor")
        physical_weight = real_scalar(b, d.weights.joint(radius)[2], "Physical feature-1 node density")
        b.positive(physical_weight, "Physical feature-1 density at every open node")
        checked += 1
    b.positive(d.ground.gamma_midpoint, "Signed positive-rule midpoint coupling")
    return {"passed": True, "positive_open_nodes_checked": checked,
            "Sr_nonpositive_by_signed_positive_Cauchy_compression": True,
            "St_nonnegative_by_positive_Cauchy_compression": True,
            "physical_Gram_positive_by_exact_weighted_integral": True,
            "eigenvalue_clipping_Gram_threshold_or_basis_deletion_used": False,
            "scope": "Structural signs of the exact finite positive-rule constructions; not sampled spectral signs"}


def exchange_chart_controls(backend):
    """Normalized spin wedges and exact odd feature parity, without a Gram test."""
    b, tolerance = backend, backend.real(F(1, 10**20))
    singlet, Bt, wedges = normalized_spin_wedges(b)
    largest = b.real(0)
    for i in range(2):
        for j in range(2):
            largest = max(largest, (singlet[2 * i + j] + singlet[2 * j + i]).abs_upper())
            for channel in range(3):
                largest = max(largest, (Bt[2 * i + j][channel] - Bt[2 * j + i][channel]).abs_upper())
    basis = [wedges[name] for name in ("UU", "LL", 0, 1, 2, 3)]
    for first, v in enumerate(basis):
        for second, w in enumerate(basis):
            inner = sum((v[j].conjugate() * w[j] for j in range(16)), b.complex(0))
            largest = max(largest, (inner - int(first == second)).abs_upper())
    passed = bool(largest <= tolerance.lower())
    return {"passed": passed, "largest_spin_normalization_exchange_error_upper": largest,
            "singlet_spin_antisymmetric": passed, "triplet_spin_symmetric": passed,
            "all_four_features_odd_in_d_by_exact_linear_polynomials": True,
            "odd_feature_projection_uses_monomial_parity_not_a_numerical_threshold": True,
            "cos_preimage_character_is_not_crystal_translation_or_E_half_period_character": True,
            "full_response_zero_blocks_checked_by_raw_shell_and_polynomial_control": True}


def source_plan(configuration, ground_order, ground_panels):
    """Concrete integer planning counts, never a feasibility certificate."""
    configuration.validate()
    d = configuration.scalar_rank
    # Largest exponent uses an R,T,R triangle and its phi1 augmentation:
    # (6d)^2 entries.  Simultaneously live named matrices are counted,
    # and sixteen such matrices are provisioned for backend scratch.
    kernel_nodes = 2 * d
    dense_entries = (4 * kernel_nodes**2 + 2 * d * kernel_nodes + 9 * d**2
                     + 7 * (2 * d)**2 + 2 * (3 * d)**2 + 3 * (6 * d)**2
                     + 16 * (6 * d)**2)
    # Each exact-weight source is a formula DAG.  Every compact update
    # stores at most 2d coefficients; no extra polynomial directions.
    function_coefficients = 2 * d * (16 + 8 * (configuration.neumann_degree + 1))
    imported_coefficients = 2 * ground_order * ground_panels
    rough_matrix_multiplies = 256  # Planning provision, not a backend speed/complexity theorem.
    rough_ball_multiplications = rough_matrix_multiplies * (6 * d)**3
    return {"scalar_basis_count": d, "actual_physical_rank": 2 * d,
            "full_physical_module_channels": list(CHANNELS),
            "response_channels_after_exact_NK_reduction": list(ACTIVE_CHANNELS),
            "largest_augmented_matrix_shape": [6 * d, 6 * d],
            "simultaneous_dense_entries_including_scratch_provision": dense_entries,
            "formula_coefficients_provision": function_coefficients + imported_coefficients,
            "rough_ball_multiplication_work_provision": rough_ball_multiplications,
            "matrix_work_provision_is_a_proved_upper_bound": False,
            "runtime_seconds_established": False, "feasibility_established": False,
            "free_weighted_sources_allocate_new_polynomial_space": False,
            "all_dense_arrays_Arb_host_not_device": True,
            "device_planning_limit_bytes": 40 * 1024**3, "host_planning_limit_bytes": 64 * 1024**3,
            "per_job_seconds_cap": 21600, "cumulative_batch_seconds_cap": 86400}


def evaluate_configuration(backend, work, configuration, ground_raw, result, checkpoint=None):
    b = backend
    b.ctx.dps, b.ctx.threads = configuration.digits, 1
    ground = ImportedGround.prepare(b, ground_raw)
    plan = source_plan(configuration, ground.order, len(ground.panels))
    plan["storage"] = work.storage_estimate(plan["simultaneous_dense_entries_including_scratch_provision"],
                                           plan["formula_coefficients_provision"])
    attempt = {"configuration": configuration.saved(), "planning": plan,
               "arithmetic_complete": False, "enclosure_target_pass": False,
               "controls_pass": False, "unresolved": True,
               "parameters": {"alpha_exact": "1/1000000", "Lambda_over_m_exact": "1", "gB_exact": "0"},
               "exact_inputs": {"mu2": ground.mu2, "gamma": ground.gamma_exact,
                    "gamma_midpoint_exact": ground.gamma_midpoint, "gamma_mismatch_upper": ground.gamma_mismatch,
                    "E_g": ground.energy_exact, "energy_midpoint_exact": ground.energy_approximate,
                    "energy_mismatch_upper": ground.energy_error, "phase_aligned_ground_error_upper": ground.vector_error,
                    "original_ground_residual_upper": ground.residual,
                    "E_radial_residual_reference_separation_lower": ground.radial_separation,
                    "phase_sign_margin_lower": ground.sign_margin,
                    "E_radial_gap_used_for_triplet_response": False,
                    "imported_ground_is_exact_eigenvector_of_N_midpoint_operator": False}}
    result["attempts"].append(attempt)
    if checkpoint is not None:
        checkpoint()
    try:
        attempt["backend_entire_exponential_prerequisite"] = backend_exponential_gate(b, work)
        controls = {}
        controls["native_finite_COM_and_literal_jets"] = native_difference_controls(b, work)
        weights = NativeWeights(b, configuration.angular_degree)
        controls["raw_shell_and_actual_kernel_coefficients"] = shell_controls(weights, work)
        controls["all_valley_free_current"] = free_current_controls(b, work)
        controls["spin_normalization_exchange_parity"] = exchange_chart_controls(b)
        attempt["controls"] = controls
        if checkpoint is not None:
            checkpoint()
        independent_controls = (controls["native_finite_COM_and_literal_jets"]["both_steps_pass"],
            controls["raw_shell_and_actual_kernel_coefficients"]["amplitude_and_exact_zero_block_controls_pass"],
            controls["raw_shell_and_actual_kernel_coefficients"]["kernel_pair_controls_pass"],
            controls["all_valley_free_current"]["all_directions_components_steps_signs_pass"],
            controls["spin_normalization_exchange_parity"]["passed"])
        if not all(independent_controls):
            raise Unresolved("A prescribed pointwise identity control has not attained its enclosed allowance")
        discretization = Discretization(b, work, configuration, ground)
        controls["structural_contact_signs"] = sign_controls(discretization)
        norms, errors, rows, physical_weights, margins, tail = discretization.norm_and_error_allocations()
        discretization.norm_errors = errors
        attempt.update({"seed_block_norms_upper": norms, "seed_block_error_allocations": errors,
                        "projection_quadrature_panel_contributions_upper": rows,
                        "common_physical_panel_weight_upper_including_singlet_one": physical_weights,
                        "complete_Bernstein_ellipse_margins": margins,
                        "full_native_cutoff_and_derivative_tails": tail,
                        "physical_feature_Gram_norm_upper": discretization.gram_norm,
                        "physical_feature_Gram_ball_radius_Frobenius_upper": matrix_radius(b, discretization.gram),
                        "native_h_and_h11_compressed": False})
        if checkpoint is not None:
            checkpoint()
        model = FiniteModel(discretization)
        propagated = propagation(model, norms, errors)
        attempt["completion_propagation"] = propagated
        if checkpoint is not None:
            checkpoint()
        curvatures = curvature_evaluation(model, propagated)
        attempt["curvatures"] = curvatures
        attempt["directed_integrals"] = {"count": discretization.integrals.count,
                    "largest_returned_ball_radius_upper": discretization.integrals.maximum_radius,
                    "requested_tolerance_is_only_an_accuracy_goal": True,
                    "full_continuum_h_weights_retained_in_every_integral": True,
                    "weighted_formula_DAG_nodes": len(model.functions.nodes),
                    "all_formula_coefficients_in_declared_physical_ranges": True}
        overlap = curvatures["independent_intervals_overlap"]
        control_pass = all(independent_controls) and controls["structural_contact_signs"]["passed"] and overlap
        final = curvatures["COM_intersection"]
        target = final is not None and (final.upper() - final.lower()).upper() <= b.real(F(1, 1000)).lower()
        attempt.update({"arithmetic_complete": bool(overlap), "controls_pass": bool(control_pass),
                        "enclosure_target_pass": bool(target), "unresolved": not (control_pass and target),
                        "unresolved_reason": None if control_pass and target else
                            "Independent curvature nonoverlap: halt this batch for source/input review" if not overlap else
                            "The absolute COM interval width exceeds the exact1/1000 target"})
        # No intersection or width comparison with the number one is
        # performed; the physical window/production Ward test is unset.
        if not overlap:
            # Both routes purport to enclose the same exact observable at
            # every tuple.  Later accuracy refinement cannot erase this
            # contradiction.  Preserve both computed intervals and halt.
            return attempt, None, None
        return attempt, curvatures["refinement_driver_bounds"], rows
    except AnalyticDomain as error:
        # Complete-ellipse certification is outside the quadrature
        # callback.  It cannot silently become a point-sampling fallback
        # or an uncaught incomplete result; preserve this attempted tuple.
        attempt["unresolved_reason"] = "Complete analytic<internal>/denominator margin unresolved: " + str(error)
        return attempt, None, None
    except Unresolved as error:
        attempt["unresolved_reason"] = str(error)
        if isinstance(error, EngineeringCap):
            raise
        return attempt, None, None


def output_scope():
    return {
        "scope": "Original-sine contact-only n=q=2 finite COM branch diagnostic; full fixed-charge Fock ground is not evaluated",
        "source_specification": SOURCE_SPEC, "source_independent_review": SOURCE_REVIEW,
        "source_assembly_supplement": ASSEMBLY_SPEC, "source_assembly_independent_review": ASSEMBLY_REVIEW,
        "source_control_contract": CONTROL_SPEC, "source_control_independent_review": CONTROL_REVIEW,
        "fixed_law": "alpha=1/1000000, Lambda/m=1, gB=0, gamma=log(4Emax)/(2mu2)",
        "arithmetic_complete": False, "enclosure_target_pass": False,
        "controls_pass": False, "unresolved": True,
        "differentiable_physical_window_certified": False,
        "interacting_native_Ward_error_certified": False,
        "physical_clock_load_continuation_certified": False,
        "full_fixed_charge_Fock_ground_certified": False,
        "open_exterior_flow_certified": False, "local_momentum_row_certified": False,
        "trapping_radius_certified": False, "temperature_certified": False,
        "physical_comparison_with_one_authorized": False,
        "N1_drift_comparison_in_frozen_batch": False,
        "same_exact_coupling_rule": True,
        "componentwise_midpoint_ground_state_used": False,
        "native_free_multiplication_compressed": False,
    }


def read_backend_probe_host(path):
    with Path(path).open() as stream:
        data = json.load(stream)
    if data.get("all_checks_pass") is not True or data.get("model_evaluated") is not False:
        raise Unresolved("The separately admitted pinned directed-backend API probe has not passed")
    if "python-flint0.9.0" not in data.get("backend", ""):
        raise Unresolved("The saved backend prerequisite is not the isolated pinned0.9.0 API")
    checks = data.get("checks", {})
    if not checks or any(value is not True for value in checks.values()):
        raise Unresolved("The saved backend prerequisite has an incomplete check packet")
    return data


def write_result(path, result):
    # Only the separate scientific result is rewritten.  E input files
    # and all source/artifacts stay untouched; no hashes or manifest.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    text_path = path.with_suffix(".txt")
    lines = [result["scope"],
             "arithmetic_complete=" + str(result["arithmetic_complete"]),
             "enclosure_target_pass=" + str(result["enclosure_target_pass"]),
             "controls_pass=" + str(result["controls_pass"]),
             "unresolved=" + str(result["unresolved"])]
    if result.get("unresolved_reason"):
        lines.append("reason: " + result["unresolved_reason"])
    for index, attempt in enumerate(result.get("attempts", ()), 1):
        configuration = attempt["configuration"]
        lines.append("tuple " + str(index) + ": L=" + str(configuration["cutoff"])
                     + ", N=" + str(configuration["polynomial_counts_including_constant"])
                     + ", digits=" + str(configuration["decimal_digits"])
                     + ", J=" + str(configuration["angular_series_last_degree"])
                     + ", Neumann=" + str(configuration["Neumann_last_degree"]))
        curvatures = attempt.get("curvatures", {})
        for name in ("N76_curvature", "N79_curvature", "N76_COM", "N79_COM", "COM_intersection"):
            value = curvatures.get(name)
            if value is not None:
                lines.append(name + ": " + value.get("display_ball", "exact dyadic endpoints in JSON"))
        if attempt.get("unresolved_reason"):
            lines.append("tuple reason: " + attempt["unresolved_reason"])
    lines.append("Physical window, interacting Ward error, clock, full fixed-charge Fock and open flow: unproved.")
    text_path.write_text("\n".join(lines) + "\n")


def run(args, backend, admission, ground_raw, probe_raw):
    b = backend
    b.activate(admission, 128)
    work = Work(b, seconds=args.seconds, batch_elapsed_seconds=args.batch_elapsed_seconds,
                max_evaluations=args.max_evaluations)
    imported = ImportedGround.prepare(b, ground_raw)
    initial_cutoff = max(16, int(imported.support))
    configuration = Configuration(initial_cutoff, (16,) * len(panels_for(initial_cutoff)))
    refinements = Refinements()
    result = output_scope()
    result.update({"runtime_admission": admission, "attempts": [],
                   "E_ground_export_input": str(args.ground_export),
                   "E_ground_export_prerequisite_passed": True,
                   "separate_backend_API_probe_input": str(args.backend_probe),
                   "separate_backend_API_probe_all_checks_pass": probe_raw["all_checks_pass"],
                   "backend": "isolated python-flint0.9.0 directed Arb/Acb; no NumPy/CuPy candidate arithmetic",
                   "arithmetic_contract": {
                       "exact_rational_inputs_and_dyadic_endpoint_outputs": True,
                       "display_decimals_or_binary64_used_as_scientific_inputs": False,
                       "directed_scalar_matrix_integration_enclosures_retained": True,
                       "complex_roots_forward_analytic_branch_requirements": True,
                       "nonnormal_phi1_uses_entire_directed_augmented_exponential": True,
                       "backend_spot_checks_are_not_the_rounding_proof": True,
                       "primary_documentation": [
                           "https://python-flint.readthedocs.io/en/latest/general.html",
                           "https://python-flint.readthedocs.io/en/latest/acb.html#flint.acb.integral",
                           "https://python-flint.readthedocs.io/en/latest/acb_mat.html#flint.acb_mat.exp"]},
                   "actual_source_audits_and_runtime_released_by_parent": True,
                   "job_seconds_cap": args.seconds, "prior_batch_seconds_engineering_only": args.batch_elapsed_seconds,
                   "own_job_thread_cap": 4, "tuple_cap": min(args.max_tuples, 12),
                   "target_absolute_COM_width_exact": "1/1000",
                   "mid_run_foreign_context_is_not_an_abort_or_exclusivity_claim": True})
    driver = "initial declared preparation targets; no physical coefficient refit"
    try:
        for _ in range(min(args.max_tuples, 12)):
            work.check()
            refinements.admit(configuration, driver)
            attempt, drivers, panel_rows = evaluate_configuration(b, work, configuration, ground_raw, result,
                checkpoint=lambda: write_result(args.output, encode_tree(b, result)))
            result = encode_tree(b, result)
            write_result(args.output, result)
            print("N tuple completed: " + json.dumps(configuration.saved()) +
                  "; arithmetic=" + str(attempt["arithmetic_complete"]) +
                  "; target=" + str(attempt["enclosure_target_pass"]) +
                  "; controls=" + str(attempt["controls_pass"]), flush=True)
            if attempt["enclosure_target_pass"] and attempt["controls_pass"]:
                result.update({"arithmetic_complete": True, "enclosure_target_pass": True,
                               "controls_pass": True, "unresolved": False, "unresolved_reason": None,
                               "achieved_COM_interval": result["attempts"][-1]["curvatures"]["COM_intersection"]})
                break
            result.update({"arithmetic_complete": attempt["arithmetic_complete"],
                           "enclosure_target_pass": attempt["enclosure_target_pass"],
                           "controls_pass": attempt["controls_pass"], "unresolved": True,
                           "unresolved_reason": attempt.get("unresolved_reason")})
            if drivers is None:
                # An unclosed prerequisite/control is sent to author/code
                # review, not attributed to a numerical model failure.
                break
            category = max(drivers, key=lambda name: drivers[name].upper())
            driver = category
            result["last_limiting_certified_contribution"] = {"category": category,
                                                           "bound": b.encode(drivers[category])}
            panel_index = None
            if category in ("projection", "quadrature"):
                panel_index = max(range(len(configuration.counts)),
                                  key=lambda index: max(values[index].upper() for values in panel_rows.values()))
            if len(refinements.attempted) >= min(args.max_tuples, 12):
                result["unresolved_reason"] = "The declared finite joint-tuple cap was reached before target/control completion"
                break
            configuration = Refinements.advance(configuration, category, panel_index)
            # Release prior tuple function/matrix objects before new
            # precision.  Encoded endpoints preserve valid achieved data.
            del attempt, drivers, panel_rows
    except Unresolved as error:
        result.update({"unresolved": True, "unresolved_reason": str(error)})
    finally:
        result = encode_tree(b, result)
        result["own_integrand_work_counter"] = work.evaluations
        result["elapsed_seconds_engineering_only"] = time.perf_counter() - work.start
        result["cumulative_seconds_engineering_only"] = args.batch_elapsed_seconds + result["elapsed_seconds_engineering_only"]
        write_result(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-export", type=Path, required=True)
    parser.add_argument("--backend-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--actual-code-audits-released", action="store_true")
    parser.add_argument("--max-tuples", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=21600)
    parser.add_argument("--batch-elapsed-seconds", type=float, default=0.0)
    parser.add_argument("--max-evaluations", type=int, default=10**8)
    args = parser.parse_args()
    if not args.actual_code_audits_released:
        parser.error("Both N-author and independent A0 actual-source audits plus parent runtime release are required")
    if args.output.exists() or args.output.with_suffix(".txt").exists() or not args.output.parent.is_dir():
        parser.error("Scientific output must be new JSON/TXT files in the parent-admitted result directory")
    if not 1 <= args.max_tuples <= 12 or not 0 < args.seconds <= 21600:
        parser.error("Tuple/job caps exceed the reviewed finite batch")
    if not 0 <= args.batch_elapsed_seconds < 86400 or args.max_evaluations <= 0:
        parser.error("Cumulative engineering work cap is invalid")
    if os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        parser.error("Assigned GPU UUID differs from the parent-selected launch environment")
    # Host-only imports and strict flag/schema reads are complete before
    # the final same-process free query/allocation/sole-owned-PID guard.
    ground_raw = ImportedGround.read_host(args.ground_export)
    probe_raw = read_backend_probe_host(args.backend_probe)
    backend = BallBackend.load_host()
    from remote_runtime import remote_execution
    admission = remote_execution(max_threads=4)
    print("N admitted " + json.dumps(admission), flush=True)
    result = run(args, backend, admission, ground_raw, probe_raw)
    if not (result["arithmetic_complete"] and result["enclosure_target_pass"] and result["controls_pass"]):
        raise SystemExit(2)  # Bounded unresolved artifact, never a law-failure code.


if __name__ == "__main__":
    main()
