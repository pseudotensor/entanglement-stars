"""One bounded b111 certificate, implementing the original pair's E18.

Serves insert I6a and item 7's light-cut ratio.  This job certifies only
b111; the required b110 calculation remains separate.  No finite-y fit
or previously estimated coefficient is an input.  Run only on the box
with the lane-isolated python-flint==0.9.0 dependency.
"""

import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


OUTPUT = Path(__file__).with_name("p111_result.json")
START = time.monotonic()
REPORT = {
    "status": "running",
    "science_specification": "ETA_massive_directions_v1.tex E17/E18",
    "purpose": "insert I6a; item 7 light-cut ratio",
    "scope": "b111 only; b110 remains required and is not computed here",
    "parameters": {"direction": [1, 1, 1], "mass_derivative": "y=0 finite part",
                   "delta": "2^-40", "upper_endpoint": "3/2"},
    "resources": {"wall_seconds": 180, "address_space_bytes": 536870912,
                  "threads": 1, "nice": 10, "cuda": "disabled"},
    "quadrature": {"primary_bits": 192, "failed_piece_retry_bits": 256,
                   "piece_width_limit": "2e-5", "final_width_limit": "2e-4",
                   "absolute_tolerance": "1e-8", "relative_tolerance": "1e-8",
                   "degree_limit": 64, "evaluation_limit": 20000,
                   "queue_limit": "2*precision (384 or 512)"},
    "finite_sums": {
        "frobenius_terms": "n=0..63; E18.2 complex value/derivative tails",
        "poisson_modes": "n=1..64; E18.3 tail starts at J=65",
        "density_W_terms": "n=0..255; r^256/(1-r) complex tail",
        "direct_modes": "n=1..63; E18.6 tail starts at J=64",
        "tail_images": "each complex disk enclosed by a real/imaginary rectangle",
    },
    "attempts": [],
    "guard_samples": {},
    "results": {},
}


def save():
    REPORT["elapsed_seconds"] = time.monotonic() - START
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
    temporary.replace(OUTPUT)


def deadline(_signal, _frame):
    raise TimeoutError("the agreed 180-second wall cap was reached")


class NonanalyticImage(Exception):
    def __init__(self, reason, value, input_argument=None):
        self.reason = reason
        self.image = str(value)
        self.input_image = None if input_argument is None else str(input_argument)
        super().__init__(reason)


def main():
    with OUTPUT.open("x") as stream:
        json.dump(REPORT, stream)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    import flint
    from flint import acb, arb, ctx

    ctx.threads = 1
    REPORT["implementation"] = {
        "python": sys.version.split()[0], "python_flint": flint.__version__,
        "box": "compute-host", "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nice_actual": os.getpriority(os.PRIO_PROCESS, 0),
    }

    def finite(z, stage="nonfinite_function_image"):
        if not z.is_finite():
            raise NonanalyticImage(stage, z)
        return z

    def denominator(z):
        finite(z, "nonfinite_denominator")
        if z.contains(0):
            raise NonanalyticImage("denominator_contains_zero", z)
        return z

    def positive_real(z, stage):
        finite(z, "nonfinite_" + stage)
        if not z.real > 0:
            raise NonanalyticImage("nonpositive_real_" + stage, z)
        return z

    def positive_sqrt(z):
        positive_real(z, "sqrt_argument")
        return finite(z.sqrt(), "nonfinite_sqrt_value")

    def logarithm(z, analytic):
        finite(z, "nonfinite_logarithm_argument")
        if z.imag.contains(0) and not z.real > 0:
            raise NonanalyticImage("logarithm_cut", z)
        return finite(z.log(analytic=analytic), "nonfinite_logarithm_value")

    def elliptic_k(z):
        finite(z, "nonfinite_elliptic_parameter")
        if z.imag.contains(0) and not z.real < 1:
            raise NonanalyticImage("elliptic_cut", z)
        value = z.elliptic_k()
        if not value.is_finite():
            raise NonanalyticImage("nonfinite_elliptic_value", value, z)
        return value

    def modulus(z, limit, stage, strict=False):
        # Moduli occur only in domain guards and rigorous tail bounds.
        radius = finite(abs(z), "nonfinite_modulus_" + stage).upper()
        passes = radius < limit if strict else radius <= limit
        if not passes:
            raise NonanalyticImage("modulus_guard_" + stage, radius, z)
        return radius

    def exact_endpoint(z):
        rational = z.fmpq()
        return int(rational.p), int(rational.q)

    def decimal_endpoint(z, lower, places=40):
        numerator, divisor = exact_endpoint(z)
        scale = 10 ** places
        integer = numerator * scale // divisor if lower else -((-numerator * scale) // divisor)
        digits = str(abs(integer)).zfill(places + 1)
        return ("-" if integer < 0 else "") + digits[:-places] + "." + digits[-places:]

    def interval(z):
        finite(z)
        lo, hi = z.lower(), z.upper()
        p, q = exact_endpoint(lo)
        r, s = exact_endpoint(hi)
        return {"ball": z.str(65, more=True),
                "lower": decimal_endpoint(lo, True), "upper": decimal_endpoint(hi, False),
                "lower_exact": {"numerator": str(p), "denominator": str(q)},
                "upper_exact": {"numerator": str(r), "denominator": str(s)},
                "width_upper": decimal_endpoint((hi - lo).upper(), False)}

    def acceptable(value, width):
        return (value.is_finite() and value.imag.contains(0)
                and value.real.upper() - value.real.lower() <= arb(width))

    tail_maxima = {}

    def rectangle(radius, label):
        upper = finite(radius, "nonfinite_tail_" + label).upper()
        if not upper >= 0:
            raise NonanalyticImage("negative_tail_" + label, upper)
        previous = tail_maxima.get(label)
        if previous is None or upper > previous:
            tail_maxima[label] = upper
            REPORT["active_piece"]["tail_radius_maxima"][label] = interval(upper)
        return acb(arb(0, upper), arb(0, upper))

    coefficient_cache = {}

    def coefficients(precision):
        if precision not in coefficient_cache:
            a, b = [arb(1)], [arb(0)]
            for n in range(256):
                a.append(a[-1] * (2 * n + 1) ** 2 / (4 * (n + 1) ** 2))
                if n < 63:
                    b.append(b[-1] + arb(1) / ((2 * n + 1) * (n + 1)))
            coefficient_cache[precision] = a, [a[n] * b[n] for n in range(64)]
        return coefficient_cache[precision]

    def polynomial_and_derivative(items, z):
        value, derivative = acb(items[-1]), acb(0)
        for coefficient in reversed(items[:-1]):
            derivative = derivative * z + value
            value = value * z + coefficient
        return value, derivative

    def node(q, precision, analytic, pi):
        u2 = denominator(3 - q)
        x, x_y = q / u2, (3 - 2 * q) / (u2 * u2)
        rx = modulus(x, arb(1) / 4, "frobenius_x")
        a, ab = coefficients(precision)
        zeta, zeta_prime = polynomial_and_derivative(a[:64], x)
        numerator, numerator_prime = polynomial_and_derivative(ab, x)
        ztail = rx ** 64 / (1 - rx)
        dztail = rx ** 63 * (64 / (1 - rx) + rx / (1 - rx) ** 2)
        zeta += rectangle(ztail, "node_Z")
        zeta_prime += rectangle(dztail, "node_Zprime")
        numerator += rectangle(2 * ztail, "node_N")
        numerator_prime += rectangle(2 * dztail, "node_Nprime")
        denominator(zeta)
        dvalue = numerator / zeta
        dprime = (numerator_prime * zeta - numerator * zeta_prime) / (zeta * zeta)
        elliptic_a = pi * zeta / 2
        tvalue = (2 / pi) * (acb(4).log() - logarithm(x, analytic) / 2 - dvalue)
        finite(tvalue, "nonfinite_T")
        if not tvalue.real > arb(1) / 2:
            raise NonanalyticImage("T_real_guard", tvalue)
        exp2d = finite((2 * dvalue).exp(), "nonfinite_exp_2D")
        w = x * exp2d / 16
        rw = modulus(w, arb(1) / 8, "poisson_w")
        inverse_t = 1 / denominator(tvalue)
        w2, power, even_power = w * w, acb(1), w * w
        sum_bc, sum_effective, sum_j = acb(0), acb(0), acb(0)
        for n in range(1, 65):
            inverse = 1 / denominator(1 - even_power)
            bn, cn = 2 * power * inverse, (1 + even_power) * inverse
            sum_bc += bn * cn
            sum_effective += n * (bn + 2 * w2 * bn ** 3)
            sum_j += n * n * bn * (cn - inverse_t / (pi * n))
            power *= w
            even_power *= w2
        qbar = 4 * pi * sum_bc - 2 * pi * pi * tvalue * sum_effective
        jbar = sum_j
        # The first omitted mode is J=65; powers begin at w^(J-1).
        j, remainder = 65, rw ** 130
        one_minus = 1 - rw
        disk_denominator = 1 - remainder
        l0 = rw ** (j - 1) / one_minus
        l1 = rw ** (j - 1) * (j / one_minus + rw / one_minus ** 2)
        l2 = rw ** (j - 1) * (j * j / one_minus + 2 * j * rw / one_minus ** 2
                              + rw * (1 + rw) / one_minus ** 3)
        c0 = 2 * (1 + remainder) / disk_denominator ** 2
        c1 = 2 / disk_denominator + 16 * remainder / disk_denominator ** 3
        qtail = 4 * arb.pi() * c0 * l0 + 2 * arb.pi() ** 2 * abs(tvalue).upper() * c1 * l1
        jtail = c0 * l2 + 2 * abs(inverse_t).upper() * l1 / (arb.pi() * disk_denominator)
        qbar += rectangle(qtail, "node_Qbar")
        jbar += rectangle(jtail, "node_Jbar")
        gref = 1 / (3 * u2) + x_y * (-2 * dprime / 3
                    - exp2d * (1 + 2 * x * dprime) * qbar / (16 * pi))
        ry = -pi ** 3 * tvalue * exp2d * jbar / denominator(16 * u2 * elliptic_a ** 2)

        s = positive_sqrt(1 - 2 * q / 3)
        nu_denominator = denominator(3 - 2 * q)
        nu_over_q = (2 - q) ** 3 / nu_denominator
        nu = q * nu_over_q
        rnu = modulus(nu, arb(9) / 10, "density_nu")
        w_density = acb(a[256])
        for coefficient in reversed(a[1:256]):
            w_density = w_density * nu + coefficient
        w_density += rectangle(rnu ** 256 / (1 - rnu), "density_W")
        z_density = 1 + nu * w_density
        rho = s * z_density / pi
        rho_difference_over_q = (-2 * z_density / denominator(3 * (1 + s))
                                  + nu_over_q * w_density) / pi
        return finite(rho * (gref + ry) - rho_difference_over_q / 3)

    def off_node(q, d, piece, pi):
        u, v = positive_sqrt(3 - q), positive_sqrt(q)
        product = denominator(u * v)
        total = denominator(u + v)
        alpha = total / 2
        k_numerator = 2 * d if piece == 3 else 3 - 2 * q
        k = k_numerator / (total * total)
        p, m = k * k, 1 - k * k
        for value, label in [(p, "p"), (1 - p, "one_minus_p"),
                             (m, "m"), (1 - m, "one_minus_m")]:
            positive_real(value, label)
        kp, km = elliptic_k(p), elliptic_k(m)
        epsilon = pi * km / denominator(kp)
        finite(epsilon, "nonfinite_epsilon")
        if not epsilon.real > 2:
            raise NonanalyticImage("epsilon_real_guard", epsilon)
        epsilon_y = pi * pi / denominator(2 * m * kp * kp * product)
        curvature = alpha * alpha * pi * pi / denominator(product * product * kp * kp)
        z = finite((-epsilon).exp(), "nonfinite_direct_nome")
        rz = modulus(z, arb(-2).exp(), "direct_nome", strict=True)
        power, sum_f, sum_h = z, acb(0), acb(0)
        for n in range(1, 64):
            inverse = 1 / denominator(1 + power)
            scale = power * inverse * inverse
            sum_f += n * n * scale
            sum_h += (n * epsilon * (1 - power) * inverse - 1) * scale
            power *= z
        f_epsilon, h2 = -4 * epsilon * sum_f, -acb(1) / 4 + 2 * sum_h
        # The first omitted direct term is J=64.
        j, remainder = 64, rz ** 64
        one_minus, disk_denominator = 1 - rz, 1 - remainder
        l0 = remainder / one_minus
        l1 = remainder * (j / one_minus + rz / one_minus ** 2)
        l2 = remainder * (j * j / one_minus + 2 * j * rz / one_minus ** 2
                           + rz * (1 + rz) / one_minus ** 3)
        epsilon_bound = abs(epsilon).upper()
        f_tail = 4 * epsilon_bound * l2 / disk_denominator ** 2
        h_tail = 2 * (epsilon_bound * (1 + remainder) * l1 / disk_denominator ** 3
                       + l0 / disk_denominator ** 2)
        f_epsilon += rectangle(f_tail, "off_Fepsilon")
        h2 += rectangle(h_tail, "off_H2")
        sy = epsilon_y * f_epsilon + curvature * h2
        if piece == 1:
            c = d ** 3 * (2 + d) / denominator(1 + 2 * d)
            positive_real(c, "density_complement")
            rho = 2 * positive_sqrt(1 + 2 * d) * elliptic_k(1 - c) / (acb(3).sqrt() * pi ** 2)
        elif piece == 2:
            radicand = (1 - d) ** 3 * (1 + d)
            c = d ** 3 * (2 - d) / denominator(radicand)
            positive_real(c, "density_complement")
            rho = 2 * (1 - 2 * d) * elliptic_k(1 - c) / denominator(
                acb(3).sqrt() * pi ** 2 * positive_sqrt(radicand))
        else:
            d3 = (1 + 2 * d) ** 3 * (3 - 2 * d)
            v3 = 32 * d / denominator(d3)
            positive_real(1 - v3, "density_one_minus_v3")
            rho = 16 * d * elliptic_k(v3) / denominator(acb(3).sqrt() * pi ** 2 * positive_sqrt(d3))
        return finite(rho * sy + 1 / denominator(3 * pi * q))

    ctx.prec = 256
    delta = arb(2) ** -40
    bdelta = 2 - delta.log()
    omission_node = 200 * delta * (bdelta ** 2 + 2 * bdelta + 2) + 2 * delta / 3
    omission_middle = 4 * 25001 * delta * (3 - delta.log()) + 2 * delta
    omission_outer = 25001 * delta
    omission_total = (omission_node + omission_middle + omission_outer).upper()
    REPORT["omitted_integral_bounds"] = {
        "node": interval(omission_node), "middle": interval(omission_middle),
        "outer": interval(omission_outer), "signed_absolute_total": interval(omission_total)}
    save()
    pieces = []
    for piece in range(4):
        accepted = False
        for precision in (192, 256):
            ctx.prec = precision
            pi, log2 = acb.pi(), acb(2).log()
            calls = {"total": 0, "refused_images": 0, "refusals_by_reason": {}}
            tail_maxima = {}
            REPORT["active_piece"] = {"piece": piece, "bits": precision, "calls": calls,
                                       "tail_radius_maxima": {}}
            save()

            def integrand(z, analytic):
                calls["total"] += 1
                if calls["total"] % 512 == 0:
                    save()
                try:
                    d = finite((-z * log2).exp(), "nonfinite_d")
                    q = d if piece == 0 else 1 - d if piece == 1 else 1 + d if piece == 2 else acb(3) / 2 - d
                    value = node(q, precision, analytic, pi) if piece == 0 else off_node(q, d, piece, pi)
                    return finite(value * log2 * d)
                except NonanalyticImage as error:
                    calls["refused_images"] += 1
                    counts = calls["refusals_by_reason"]
                    counts[error.reason] = counts.get(error.reason, 0) + 1
                    if error.reason not in REPORT["guard_samples"]:
                        REPORT["guard_samples"][error.reason] = {
                            "piece": piece, "bits": precision, "analytic": bool(analytic),
                            "argument": str(z), "tested_image": error.image,
                            "tested_input": error.input_image}
                        save()
                    return acb("nan")

            start = 1 if piece < 2 else 2
            value = acb.integral(integrand, acb(start), acb(40), rel_tol=arb("1e-8"),
                                 abs_tol=arb("1e-8"), deg_limit=64, eval_limit=20000,
                                 depth_limit=2 * precision, use_heap=True)
            record = {"piece": piece, "bits": precision, "variable": "u",
                      "endpoints": [str(start), "40"], "queue_limit": 2 * precision,
                      "calls": calls, "tail_radius_maxima": REPORT["active_piece"]["tail_radius_maxima"],
                      "accepted": bool(acceptable(value, "2e-5"))}
            if value.is_finite():
                record["real"], record["imaginary"] = interval(value.real), interval(value.imag)
            else:
                record["nonfinite"] = True
            REPORT["attempts"].append(record)
            save()
            if record["accepted"]:
                pieces.append(value.real)
                accepted = True
                break
        if not accepted:
            raise ArithmeticError("a b111 panel failed the agreed finite/width test")

    ctx.prec = 256
    coefficient = sum(pieces, arb(0)) + arb(0, omission_total) - (arb(3).log() - arb(2).log() + 1) / (3 * arb.pi())
    if not (coefficient.is_finite() and coefficient.upper() - coefficient.lower() <= arb("2e-4")):
        raise ArithmeticError("the final b111 interval failed the agreed width test")
    REPORT["results"]["b111"] = interval(coefficient)
    REPORT["status"] = "complete"
    save()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Refusing to replace an existing result or repeat this check")
    try:
        main()
    except Exception as error:
        REPORT["status"] = "incomplete"
        REPORT["error"] = type(error).__name__ + ": " + str(error)
        save()
    finally:
        signal.alarm(0)
    print(json.dumps({"status": REPORT["status"], "elapsed_seconds": REPORT.get("elapsed_seconds"),
                      "completed_pieces": sum(bool(a["accepted"]) for a in REPORT["attempts"])}))
    sys.exit(0 if REPORT["status"] == "complete" else 1)
