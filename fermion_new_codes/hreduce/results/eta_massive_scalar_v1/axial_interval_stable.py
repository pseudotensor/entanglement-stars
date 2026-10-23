"""Necessary stable-integrand repair of the same axial check, under E6/E8/E9.

Serves insert I6a and item 7's light-cut ratio; it does not calculate a
tilted coefficient or certify the reused massless directional inputs.
Run on the box with the lane-isolated python-flint==0.9.0 dependency.
All quadrature pieces and outward endpoints are saved before reporting.
"""

import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


OUTPUT = Path(__file__).with_name("stable_result.json")
START = time.monotonic()
REPORT = {
    "status": "running",
    "science_specification": "ETA_massive_directions_v1.tex E6/E8/E9",
    "prior_failed_results": ["result.json", "queue_repair_result.json"],
    "prior_records": "both producers and results preserved unchanged",
    "integrand_repair": {"identity": "E9.1", "D_series_terms": 160,
                         "complex_q_modulus_limit": "3/4",
                         "D_tail_absolute": "pi*(3/4)^160; rectangular complex enclosure"},
    "quantity_order": ["eta100_y1_25", "b100_integral"],
    "certified_components": {"eta100_y1_25": False, "b100": False},
    "results": {},
    "purpose": "insert I6a; item 7 light-cut ratio",
    "parameters": {"ma": "1/5", "y": "1/25", "delta": "2^-40"},
    "resources": {"wall_seconds": 60, "address_space_bytes": 268435456,
                  "threads": 1, "nice": 10, "cuda": "disabled"},
    "quadrature": {"primary_bits": 192, "failed_piece_retry_bits": 256,
                   "piece_width_limit": "2e-9", "final_width_limit": "2e-8",
                   "absolute_tolerance": "1e-12", "relative_tolerance": "1e-12",
                   "degree_limit": 64, "evaluation_limit": 20000,
                   "queue_limit": "2*precision (384 or 512)"},
    "massless_input": {"lower": "0.91905664456747", "upper": "0.91929747790884",
                       "premise": "conditional on the earlier supplied directional errors"},
    "attempts": [],
    "guard_samples": {},
    "tilted_coefficients": "not calculated; native-error and finite-mass remainder certificates missing",
}


def save():
    REPORT["elapsed_seconds"] = time.monotonic() - START
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
    temporary.replace(OUTPUT)


def deadline(_signal, _frame):
    raise TimeoutError("the agreed 60-second wall cap was reached")


class NonanalyticImage(Exception):
    def __init__(self, reason, value, input_argument=None):
        self.reason = reason
        self.image = str(value)
        self.input_image = None if input_argument is None else str(input_argument)
        super().__init__(reason)


def main():
    # Exclusive initial creation prevents an accidental repeat of this job.
    with OUTPUT.open("x") as stream:
        json.dump(REPORT, stream)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(60)
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    import flint
    from flint import acb, arb, ctx

    ctx.threads = 1
    REPORT["implementation"] = {
        "python": sys.version.split()[0], "python_flint": flint.__version__,
        "box": "compute-host", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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

    def elliptic(z, kind):
        finite(z, "nonfinite_elliptic_parameter_" + kind)
        if z.imag.contains(0) and not z.real < 1:
            raise NonanalyticImage("elliptic_cut_" + kind, z)
        value = z.elliptic_k() if kind == "K" else z.elliptic_e()
        if not value.is_finite():
            raise NonanalyticImage("nonfinite_elliptic_value_" + kind, value, z)
        return value

    def logarithm(z, analytic):
        finite(z, "nonfinite_logarithm_argument")
        if z.imag.contains(0) and not z.real > 0:
            raise NonanalyticImage("logarithm_cut", z)
        return finite(z.log(analytic=analytic), "nonfinite_logarithm_value")

    def exact_endpoint(z):
        rational = z.fmpq()
        numerator, divisor = int(rational.p), int(rational.q)
        return numerator, divisor

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
        return {
            "ball": z.str(65, more=True),
            "lower": decimal_endpoint(lo, True), "upper": decimal_endpoint(hi, False),
            "lower_exact": {"numerator": str(p), "denominator": str(q)},
            "upper_exact": {"numerator": str(r), "denominator": str(s)},
            "width_upper": decimal_endpoint((hi - lo).upper(), False),
        }

    def acceptable(value, width):
        return (value.is_finite() and value.imag.contains(0)
                and (value.real.upper() - value.real.lower()) <= arb(width))

    series_cache = {}

    def d_series(q, precision):
        modulus = finite(abs(q), "nonfinite_D_series_modulus")
        if not modulus <= arb(3) / 4:
            raise NonanalyticImage("D_series_modulus_exceeds_three_quarters", modulus, q)
        if precision not in series_cache:
            coefficients = [arb(1)]
            for n in range(159):
                coefficients.append(coefficients[-1] * (2 * n + 1) * (2 * n + 3)
                                    / (4 * (n + 2) * (n + 1)))
            tail = (arb.pi() * (arb(3) / 4) ** 160).upper()
            series_cache[precision] = coefficients, tail
        coefficients, tail = series_cache[precision]
        polynomial = acb(coefficients[-1])
        for coefficient in reversed(coefficients[:-1]):
            polynomial = polynomial * q + coefficient
        # A rectangle contains the complete complex remainder disk.
        remainder = acb(arb(0, tail), arb(0, tail))
        return finite((acb.pi() / 4) * polynomial + remainder, "nonfinite_D_series")

    def finish_quantity(quantity, pieces):
        ctx.prec = 256
        delta = arb(2) ** -40
        tail_base = delta * (2 - delta.log())
        b_tail, a_tail = (18 * tail_base).upper(), (7 * tail_base).upper()
        REPORT["omitted_piece_bounds"] = {"b_signed_absolute": interval(b_tail),
                                          "eta_one_sided_upper": interval(a_tail)}
        if quantity == "eta100_y1_25":
            # A one-sided [0,T] enclosure using exact outward T.
            a = sum(pieces, arb(0)) + arb(a_tail / 2, a_tail / 2)
            if not (a.is_finite() and a.lower() > 0
                    and a.upper() - a.lower() <= arb("2e-8")):
                raise ArithmeticError("the final A interval failed positivity/width test")
            m1 = 3 * arb(2).sqrt() * (1 / arb(2).sqrt()).atan() / arb.pi()
            ratio_lo = (arb("0.91905664456747") / (3 * a.upper() / 2)).lower()
            ratio_hi = (arb("0.91929747790884") / (m1 * a.lower())).upper()
            REPORT["results"].update({
                "eta100_y1_25": interval(a), "m1": interval(m1),
                "sphere_ratio_conditional_axis_only": {
                    "lower": decimal_endpoint(ratio_lo, True),
                    "upper": decimal_endpoint(ratio_hi, False),
                    "premise": REPORT["massless_input"]["premise"],
                },
            })
            REPORT["certified_components"]["eta100_y1_25"] = True
        else:
            b = sum(pieces, arb(0)) - (1 + arb(8).log()) / (3 * arb.pi()) + arb(0, b_tail)
            if not (b.is_finite() and b.upper() - b.lower() <= arb("2e-8")):
                raise ArithmeticError("the final b interval failed the width test")
            REPORT["results"]["b100"] = interval(b)
            from decimal import Decimal, localcontext
            with localcontext() as context:
                context.prec = 100
                REPORT["results"]["b_endpoints_round_to_minus_0_4175"] = all(
                    Decimal(REPORT["results"]["b100"][key]).quantize(Decimal("0.0001"))
                    == Decimal("-0.4175") for key in ("lower", "upper"))
            REPORT["certified_components"]["b100"] = True
        # Accepted A and its conditional ratio survive a later b failure/cap.
        save()

    pieces = {}
    for quantity in ("eta100_y1_25", "b100_integral"):
        pieces[quantity] = []
        for piece in range(4):
            accepted = False
            for precision in (192, 256):
                ctx.prec = precision
                pi, log2 = acb.pi(), acb(2).log()
                calls = {"total": 0, "refused_images": 0, "refusals_by_reason": {}}
                REPORT["active_piece"] = {"quantity": quantity, "piece": piece,
                                          "bits": precision, "calls": calls}
                save()

                def integrand(z, analytic):
                    calls["total"] += 1
                    if calls["total"] % 512 == 0:
                        save()
                    try:
                        if piece < 3:
                            x = finite((-z * log2).exp())
                            s = x if piece == 0 else 1 - x if piece == 1 else 1 + x
                            weight = log2 * x
                            rho_parameter = 1 - x * x if piece in (1, 2) else s * (2 - s)
                        else:
                            s, weight = z, acb(1)
                            rho_parameter = s * (2 - s)
                        rho = 2 * elliptic(rho_parameter, "K") / (pi * pi)
                        if quantity == "b100_integral":
                            one_plus_s = denominator(1 + s)
                            t, q = 1 / one_plus_s, s / one_plus_s
                            expression = (2 * t / 3) * (
                                1 - elliptic(t, "K") * (elliptic(q, "K")
                                + (1 - 2 * q) * d_series(q, precision)) / pi)
                        else:
                            w = s + acb(1) / 25
                            one_plus_w = denominator(1 + w)
                            expression = (acb(16).log() + 2 * logarithm(one_plus_w, analytic)
                                          - logarithm(w, analytic)
                                          + (4 / pi) * elliptic(1 / one_plus_w, "K")
                                          * elliptic(w / one_plus_w, "K")
                                          * (1 - w) / one_plus_w) / 6
                        return finite(weight * rho * expression)
                    except NonanalyticImage as error:
                        calls["refused_images"] += 1
                        reasons = calls["refusals_by_reason"]
                        reasons[error.reason] = reasons.get(error.reason, 0) + 1
                        if error.reason not in REPORT["guard_samples"]:
                            REPORT["guard_samples"][error.reason] = {
                                "quantity": quantity, "piece": piece, "bits": precision,
                                "analytic": bool(analytic), "argument": str(z),
                                "tested_image": error.image, "tested_input": error.input_image,
                            }
                            save()
                        return acb("nan")

                start, end = (acb(1), acb(40)) if piece < 3 else (acb(3) / 2, acb(2))
                value = acb.integral(integrand, start, end,
                                     rel_tol=arb("1e-12"), abs_tol=arb("1e-12"),
                                     deg_limit=64, eval_limit=20000, depth_limit=2 * precision,
                                     use_heap=True)
                record = {"quantity": quantity, "piece": piece, "bits": precision,
                          "variable": "u" if piece < 3 else "s",
                          "queue_limit": 2 * precision,
                          "endpoints": ["1", "40"] if piece < 3 else ["3/2", "2"],
                          "calls": calls, "accepted": bool(acceptable(value, "2e-9"))}
                if value.is_finite():
                    record["real"] = interval(value.real)
                    record["imaginary"] = interval(value.imag)
                else:
                    record["nonfinite"] = True
                REPORT["attempts"].append(record)
                save()
                if record["accepted"]:
                    pieces[quantity].append(value.real)
                    accepted = True
                    break
            if not accepted:
                raise ArithmeticError("a scalar piece failed the agreed finite/width test")

        finish_quantity(quantity, pieces[quantity])

    if not all(REPORT["certified_components"].values()):
        raise ArithmeticError("overall completion requires both final A and b certificates")
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
