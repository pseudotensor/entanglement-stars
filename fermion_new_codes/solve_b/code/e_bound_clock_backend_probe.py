#!/usr/bin/env python3
"""Engineering-only pinned ball-backend/API probe, after Ops admission.

No clock model is evaluated and no package is installed by this program.
All imports and argument validation precede the final same-process GPU
admission.  This source itself needs author/independent read-only audit.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from remote_runtime import remote_execution


def load_backend():
    """Audited dependency subset copied here so the probe is self-contained."""
    global arb, acb, arb_mat, acb_mat, ctx
    import flint
    from flint import arb, acb, arb_mat, acb_mat, ctx
    if getattr(flint, "__version__", None) != "0.9.0":
        raise RuntimeError("The isolated pinned python-flint0.9.0 backend is required")
    for owner, name in ((arb, "legendre_p_root"), (arb, "mid_rad_10exp"),
                        (acb, "integral"), (arb_mat, "inv"), (acb_mat, "eig")):
        if not hasattr(owner, name):
            raise RuntimeError("Pinned directed backend lacks required API: " + name)
    return "python-flint" + flint.__version__ + "; Arb/Acb directed ball operations"


def a(value=0):
    if isinstance(value, F):
        return arb(value.numerator) / value.denominator
    return arb(value)


def c(value=0, imaginary=0):
    return acb(a(value), a(imaginary))


def lower(value):
    return value.lower()


def upper(value):
    return value.upper()


def encoded(value):
    if isinstance(value, acb):
        return {"real": encoded(value.real), "imag": encoded(value.imag)}
    if not value.is_finite():
        return {"status": "non-finite/unresolved"}
    def endpoint(bound):
        mantissa, exponent = bound.man_exp()
        return {"mantissa": str(mantissa), "exponent2": int(exponent)}
    return {"lower_dyadic": endpoint(lower(value)), "upper_dyadic": endpoint(upper(value)),
            "display_ball": value.str(30)}


def exact_endpoint_square_comparison(endpoint):
    """Compare the exact dyadic endpoint square with two using integers only."""
    mantissa, exponent = endpoint.man_exp()
    mantissa, exponent = int(mantissa), int(exponent)
    square_operand = mantissa * mantissa
    if exponent >= 0:
        square_operand <<= 2 * exponent
        two_operand = 2
    else:
        two_operand = 2 << (-2 * exponent)
    margin = square_operand - two_operand
    return {"endpoint_mantissa": str(mantissa), "endpoint_exponent2": exponent,
            "square_operand": str(square_operand), "two_operand": str(two_operand),
            "signed_integer_margin": str(margin),
            "square_strictly_below_two": margin < 0,
            "square_strictly_above_two": margin > 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("Output must be a new file in an existing admitted result directory")
    backend = load_backend()  # Host-only import, before final admission.
    admission = remote_execution(max_threads=4)
    ctx.dps = 128
    checks = {}
    engineering_radius = a(F(1, 10**60))
    pi, sqrt2 = arb.pi(), arb(2).sqrt()
    endpoint_square_proofs = {
        "lower": exact_endpoint_square_comparison(sqrt2.lower()),
        "upper": exact_endpoint_square_comparison(sqrt2.upper()),
    }
    constant_checks = {
        "pi_finite": pi.is_finite(),
        "sqrt2_finite": sqrt2.is_finite(),
        "pi_strict_endpoints": pi.lower() < pi.upper(),
        "sqrt2_lower_square_strictly_below_two": endpoint_square_proofs["lower"]["square_strictly_below_two"],
        "sqrt2_upper_square_strictly_above_two": endpoint_square_proofs["upper"]["square_strictly_above_two"],
        "endpoint_man_exp_pair": len(pi.lower().man_exp()) == 2,
        "mid_rad_10exp_triple": len(pi.mid_rad_10exp(30)) == 3,
    }
    checks["directed_constants_and_exact_endpoint_api"] = all(constant_checks.values())
    roots = [arb.legendre_p_root(8, k, weight=True) for k in range(8)]
    checks["Gauss_algebraic_ball_nodes_weights_decreasing"] = (
        all(node.is_finite() and weight.is_finite()
            and lower(node) > -1 and upper(node) < 1
            and lower(weight) > 0
            and upper(node.rad()) < engineering_radius
            and upper(weight.rad()) < engineering_radius for node, weight in roots)
        and all(lower(roots[k][0]) > upper(roots[k + 1][0]) for k in range(7))
        and sum((weight for _, weight in roots), a(0)).contains(2)
        and sum((weight * node**14 for node, weight in roots), a(0)).contains(a(F(2, 15))))
    analytic_calls = [0]

    def callback(z, analytic):
        analytic_calls[0] += bool(analytic)
        return z.sqrt(analytic=analytic)

    integral = acb.integral(callback, 0, 1, abs_tol=a(F(1, 10**80)),
                            rel_tol=a(F(1, 10**80)), eval_limit=10000)
    cut_test = acb(-1, arb(0, a(F(1, 10)))).sqrt(analytic=True)
    checks["rigorous_integration_and_branch_flag"] = (
        integral.is_finite() and integral.contains(c(F(2, 3)))
        and upper(integral.rad()) < engineering_radius
        and analytic_calls[0] > 0 and not cut_test.is_finite())
    real_matrix = arb_mat([[2, 1], [1, 3]])
    real_inverse = real_matrix.inv()
    complex_matrix = acb_mat([[c(2, 1), 1], [1, c(3, -1)]])
    complex_inverse = complex_matrix.inv()
    real_product, complex_product = real_matrix * real_inverse, complex_matrix * complex_inverse
    checks["directed_real_complex_inverse"] = (
        all(matrix[i, j].is_finite() and upper(matrix[i, j].rad()) < engineering_radius
            for matrix in (real_inverse, complex_inverse, real_product, complex_product)
            for i in range(2) for j in range(2))
        and all(real_product[i, j].contains(int(i == j))
                and complex_product[i, j].contains(int(i == j))
                for i in range(2) for j in range(2)))
    eigenvalues = acb_mat([[1, 0], [0, 2]]).eig(algorithm="rump")
    checks["documented_rigorous_eigen_api"] = (
        len(eigenvalues) == 2
        and all(item.is_finite() and upper(item.rad()) < engineering_radius for item in eigenvalues)
        and not eigenvalues[0].overlaps(eigenvalues[1])
        and sum(item.contains(1) for item in eigenvalues) == 1
        and sum(item.contains(2) for item in eigenvalues) == 1)
    data = {"scope": "engineering-only dependency/API probe; no model result",
            "backend": backend, "admission": admission, "precision_decimal_digits": ctx.dps,
            "model_evaluated": False, "full_evaluator_certified": False,
            "predeclared_engineering_radius_bound": encoded(engineering_radius),
            "directed_constant_predicates": constant_checks,
            "constant_bracket_acceptance_method": "exact dyadic endpoint square comparison",
            "exact_endpoint_square_proofs": endpoint_square_proofs,
            "constant_diagnostics": {"pi": encoded(pi), "sqrt2": encoded(sqrt2),
                                     "sqrt2_lower_squared": encoded(sqrt2.lower()**2),
                                     "sqrt2_upper_squared": encoded(sqrt2.upper()**2),
                                     "rounded_lower_square_strictly_below_two": sqrt2.lower()**2 < 2,
                                     "rounded_upper_square_strictly_above_two": 2 < sqrt2.upper()**2,
                                     "pi_mid_rad_10exp_arity": len(pi.mid_rad_10exp(30))},
            "checks": checks, "all_checks_pass": all(checks.values()),
            "sample_rigorous_integral": encoded(integral),
            "documented_contract": "acb.integral callbacks honor analytic=True; all radii retained"}
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    if not data["all_checks_pass"]:
        raise RuntimeError("Pinned backend engineering/API prerequisites failed")


if __name__ == "__main__":
    main()
