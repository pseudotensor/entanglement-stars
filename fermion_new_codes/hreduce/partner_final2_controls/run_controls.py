"""Execute partner_final2_code_spec.tex; run on the authorized remote qi2 host.

Interval endpoints are serialized by directed Decimal division of their exact
binary mpf tuples. mpmath's usual interval string is not used as a certificate.
"""

from __future__ import annotations

import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import mpmath as mp


SERIES_LAST = 96
PRECISIONS = (70, 100)
CELL_DIGITS = 40
TARGET = Fraction(119, 100)
MIN_WIDTH = Fraction(1, 2**24)


def endpoint_decimal(binary, digits: int, rounding: str) -> str:
    """Outward decimal conversion of the exact dyadic interval endpoint."""
    sign, mantissa, exponent, _ = binary
    if mantissa == 0:
        return "0"
    signed = -int(mantissa) if sign else int(mantissa)
    context = Context(prec=digits, rounding=rounding)
    if exponent >= 0:
        result = context.plus(Decimal(signed << exponent))
    else:
        result = context.divide(Decimal(signed), Decimal(1 << (-exponent)))
    return str(result)


def enclosure(value, digits: int) -> dict:
    lower, upper = value._mpi_
    return {
        "lower": endpoint_decimal(lower, digits, ROUND_FLOOR),
        "upper": endpoint_decimal(upper, digits, ROUND_CEILING),
    }


def rational_text(value: Fraction) -> str:
    return str(value.numerator) + "/" + str(value.denominator)


class SeriesEvaluator:
    def __init__(self, digits: int):
        mp.iv.dps = digits
        self.iv = mp.iv
        self.digits = digits
        self.cache = {}

    def number(self, value: Fraction):
        return self.iv.mpf(value.numerator) / value.denominator

    def uv(self, value: Fraction):
        if value in self.cache:
            return self.cache[value]
        t = self.number(value)
        x = t * t / 16
        term0 = term1 = self.iv.mpf(1)
        sum0 = sum1 = self.iv.mpf(1)
        for n in range(1, SERIES_LAST + 1):
            term0 *= x / (n * n)
            term1 *= x / (n * (n + 1))
            sum0 += term0
            sum1 += term1
        first0 = term0 * x / ((SERIES_LAST + 1) ** 2)
        first1 = term1 * x / ((SERIES_LAST + 1) * (SERIES_LAST + 2))
        ratio0 = x / ((SERIES_LAST + 2) ** 2)
        ratio1 = x / ((SERIES_LAST + 2) * (SERIES_LAST + 3))
        if not (ratio0.b < 1 and ratio1.b < 1):
            raise ArithmeticError("Positive-series tail ratio is not below one")
        tail0 = first0 / (1 - ratio0)
        tail1 = first1 / (1 - ratio1)
        sum0 += self.iv.mpf([0, tail0.b])
        sum1 += self.iv.mpf([0, tail1.b])
        scaled = self.iv.exp(-t / 2)
        i0 = sum0
        i1 = (t / 4) * sum1
        result = (scaled * i0, scaled * (i0 + i1) / 2)
        self.cache[value] = result
        return result

    def cell(self, left: Fraction, right: Fraction) -> dict:
        U, V = self.uv(left)
        bound = self.iv.sqrt(self.iv.pi * self.number(right)) ** 3 * V * U**2
        upper = endpoint_decimal(bound._mpi_[1], CELL_DIGITS, ROUND_CEILING)
        # Test both the computed interval endpoint and its outward serialization.
        target = self.number(TARGET)
        passed = bool(bound.b < target.a and self.iv.mpf(upper).b < target.a)
        return {"left": rational_text(left), "right": rational_text(right),
                "upper": upper, "passed": passed}


def covers_domain(cells: list[dict]) -> bool:
    if not cells:
        return False
    ordered = sorted(cells, key=lambda c: Fraction(c["left"]))
    previous = Fraction(0)
    for cell in ordered:
        if Fraction(cell["left"]) != previous:
            return False
        previous = Fraction(cell["right"])
    return previous == 64


def phase_a(digits: int, previous: dict | None = None) -> dict:
    start = time.monotonic()
    evaluator = SeriesEvaluator(digits)
    cells, failed, untested = [], [], []
    if previous is None:
        stack = [(Fraction(i, 8), Fraction(i + 1, 8)) for i in reversed(range(512))]
        while stack:
            left, right = stack.pop()
            cell = evaluator.cell(left, right)
            if cell["passed"]:
                cells.append(cell)
            elif right - left <= MIN_WIDTH:
                failed.append(cell)
                untested = [[rational_text(l), rational_text(u)] for l, u in stack]
                break
            else:
                middle = (left + right) / 2
                stack.extend([(middle, right), (left, middle)])
    else:
        for old in previous["accepted_cells"] + previous["failed_cells"]:
            cell = evaluator.cell(Fraction(old["left"]), Fraction(old["right"]))
            (cells if cell["passed"] else failed).append(cell)
        untested = previous["untested_intervals"]
    full = covers_domain(cells)
    max_cell = max(cells, key=lambda c: Decimal(c["upper"])) if cells else None
    result = {
        "arithmetic_dps": digits, "series_last_n": SERIES_LAST,
        "cell_bound_decimal_digits": CELL_DIGITS,
        "accepted_cells": cells, "failed_cells": failed,
        "untested_intervals": untested, "accepted_count": len(cells),
        "coverage_0_to_64": full, "maximum_upper_cell": max_cell,
        "same_prior_cells_pass": None if previous is None else
            bool(len(cells) == previous["accepted_count"] and not failed),
        "passed": bool(full and not failed and not untested),
        "elapsed_seconds": time.monotonic() - start,
    }
    print(f"A {digits} digits: {len(cells)} accepted cells, "
          f"{len(failed)} failed, pass={result['passed']}", flush=True)
    return result


def phase_b(digits: int) -> dict:
    start = time.monotonic()
    evaluator = SeriesEvaluator(digits)
    iv = evaluator.iv
    midpoint = iv.mpf(0)
    serialized_midpoint = iv.mpf(0)
    rows = []
    for i in range(2048):
        t = Fraction(2 * i + 1, 64)
        U, _ = evaluator.uv(t)
        term = U**3 / 128
        midpoint += term
        lower = endpoint_decimal(term._mpi_[0], CELL_DIGITS, ROUND_FLOOR)
        serialized_midpoint += iv.mpf(lower)
        rows.append({"midpoint": rational_text(t), "lower": lower})
        if (i + 1) % 512 == 0:
            print(f"B {digits} digits: {i + 1}/2048 midpoint terms", flush=True)
    tail = (1 - iv.exp(-64))**3 / (16 * iv.pi * iv.sqrt(iv.pi))
    expression = midpoint + tail
    product = expression * iv.pi**2 / (3 * iv.log(2))
    serialized_product = (serialized_midpoint + tail) * iv.pi**2 / (3 * iv.log(2))
    target = evaluator.number(TARGET)
    passed = bool(product.a > target.b and serialized_product.a > target.b)
    result = {
        "arithmetic_dps": digits, "series_last_n": SERIES_LAST,
        "uniform_width": "1/32", "interval_count": len(rows),
        "midpoint_term_decimal_digits": CELL_DIGITS,
        "midpoint_terms": rows,
        "midpoint_sum_interval": enclosure(midpoint, digits),
        "tail_lower_expression_interval": enclosure(tail, digits),
        "zeta_lower_expression_interval": enclosure(expression, digits),
        "product_lower_expression_interval": enclosure(product, digits),
        "product_div_1_19_interval": enclosure(product / target, digits),
        "serialized_midpoint_product_interval": enclosure(serialized_product, digits),
        "passed": passed, "elapsed_seconds": time.monotonic() - start,
    }
    print(f"B {digits} digits: lower product test pass={passed}", flush=True)
    return result


def phase_c(digits: int, envelope_passed: bool) -> dict:
    start = time.monotonic()
    mp.mp.dps = digits
    boundaries = [Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 2),
                  Fraction(3, 4), Fraction(7, 8), Fraction(15, 16),
                  Fraction(31, 32), Fraction(1)]
    points = [mp.mpf(x.numerator) / x.denominator for x in boundaries]

    def integrand(z):
        if z == 0:
            return mp.mpf(0)
        if z == 1:
            return 1 / (2 * mp.pi**mp.mpf("1.5"))
        t = (z / (1 - z))**2
        U = mp.exp(-t / 2) * mp.besseli(0, t / 2)
        return z * U**3 / (2 * (1 - z)**3)

    zeta, error = mp.quad(integrand, points, method="tanh-sinh", error=True)
    c0 = mp.sqrt(mp.pi / (6 * mp.log(2)))
    tau0 = 2 * mp.pi * zeta * c0 / mp.mpf("0.005")
    C0 = zeta * mp.pi**2 / (3 * mp.log(2))
    values = {"zeta": zeta, "c0": c0, "tau0": tau0, "C0": C0}
    if envelope_passed:
        env = mp.mpf(119) / 100
        values.update(c_env=c0 / mp.sqrt(env), tau_env=tau0 / mp.sqrt(env),
                      C_env=C0 / env)
    result = {
        "arithmetic_dps": digits, "status": "decimal_diagnostic",
        "method": "tanh-sinh", "subdivision": [rational_text(p) for p in boundaries],
        "scaled_U": "exp(-t/2)*besseli(0,t/2), arbitrary-exponent mp.mpf",
        "values": {k: mp.nstr(v, digits) for k, v in values.items()},
        "quadrature_error_estimate": mp.nstr(error, digits),
        "error_is_rigorous": False, "interval_certificate": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    print(f"C {digits} digits: zeta={mp.nstr(zeta, 22)}, "
          f"tau0={mp.nstr(tau0, 18)}", flush=True)
    return result


def write_outputs(output: Path, data: dict) -> None:
    # Actual scientific phase data are checkpointed, including failures.
    temp = output / "results.json.tmp"
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(output / "results.json")
    lines = ["# Partner final2 coefficient and point-source controls", "",
             "Scope: sine vacuum coefficient and a declared installed cubic Poisson row.",
             "This job supplies no default internal source, star, instrument or production simulation.",
             "Interval arithmetic: 70 and 100 decimal digits; positive Bessel series through n=96.",
             "Cell and midpoint witnesses are rounded outward to 40 significant decimal digits.",
             "Full aggregate interval endpoints retain the computation precision.", "",
             "| Test | Digits | Result |", "|---|---:|---|"]
    for phase in ("A", "B"):
        for digits, record in data.get(phase, {}).items():
            lines.append(f"| {phase} | {digits} | {'PASS' if record['passed'] else 'FAIL'} |")
    for digits, record in data.get("C", {}).items():
        if "values" in record:
            lines.extend(["", f"## C decimal labels ({digits} digits; not rigorous intervals)", ""])
            lines.extend(f"- {name}: {value}" for name, value in record["values"].items())
            lines.append("- quadrature error estimate: " + record["quadrature_error_estimate"])
        else:
            lines.append(f"C {digits}: {record.get('error', 'not completed')}")
    if "differences" in data:
        lines.extend(["", "## Absolute differences between C precisions", ""])
        lines.extend(f"- {k}: {v}" for k, v in data["differences"].items())
    if "certificate_passed" in data:
        lines.extend(["", "Combined A/B certificate: " +
                      ("PASS" if data["certificate_passed"] else "FAIL"), ""])
    lines.append("The exact rational infinite-tail check and all interval cells/terms are in results.json.")
    lines.append("C labels are numerical controls; physical identification remains outside this calculation.")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tail = (1 + Fraction(1, 128) + Fraction(1, 2**28))**2
    assert tail < TARGET
    data = {
        "spec": "partner_final2_code_spec.tex", "implementation_version": 1,
        "settings": {"precisions": list(PRECISIONS), "series_last_n": SERIES_LAST,
                     "initial_A_width": "1/8", "minimum_A_width": rational_text(MIN_WIDTH),
                     "target": "119/100", "B_width": "1/32", "alpha_label": "0.005"},
        "runtime": {"host": socket.gethostname(), "remote_pid": os.getpid(),
                    "python": sys.executable, "python_version": platform.python_version(),
                    "mpmath": mp.__version__, "nice": os.getpriority(os.PRIO_PROCESS, 0),
                    "threads": {k: os.environ.get(k) for k in
                                ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                 "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES"]}},
        "analytic_tail": {"rational_upper": rational_text(tail), "strict_below_119_100": True},
        "A": {}, "B": {}, "C": {},
    }
    write_outputs(args.output, data)
    previous = None
    for digits in PRECISIONS:
        record = phase_a(digits, previous)
        data["A"][str(digits)] = record
        previous = record
        write_outputs(args.output, data)
    for digits in PRECISIONS:
        data["B"][str(digits)] = phase_b(digits)
        write_outputs(args.output, data)
    data["certificate_passed"] = all(r["passed"] for phase in ("A", "B")
                                     for r in data[phase].values())
    envelope_passed = all(r["passed"] for r in data["A"].values())
    write_outputs(args.output, data)
    for digits in PRECISIONS:
        try:
            data["C"][str(digits)] = phase_c(digits, envelope_passed)
        except Exception as error:
            data["C"][str(digits)] = {"status": "failed", "error": repr(error)}
        write_outputs(args.output, data)
    if all("values" in r for r in data["C"].values()):
        mp.mp.dps = 110
        low = data["C"]["70"]["values"]
        high = data["C"]["100"]["values"]
        data["differences"] = {k: mp.nstr(abs(mp.mpf(high[k]) - mp.mpf(low[k])), 100)
                               for k in low.keys() & high.keys()}
    write_outputs(args.output, data)
    success = data["certificate_passed"] and all("values" in r for r in data["C"].values())
    print(json.dumps({"certificate_passed": data["certificate_passed"],
                      "completed": success, "results": str(args.output / "results.json")}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
