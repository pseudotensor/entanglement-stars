#!/usr/bin/env python3
"""Engineering-only callback regression; no model evaluation or installation."""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import e_bound_clock as base
from remote_runtime import remote_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("Output must be a new file in an existing result directory")
    pools = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in pools):
        parser.error("All six worker pools must be predeclared as one")
    backend = base.load_backend()
    base.ctx.dps = 128
    if hasattr(base.ctx, "threads"):
        base.ctx.threads = 1
    admission = remote_execution(max_threads=4)
    started = time.perf_counter()
    records, works = [], []
    caps = SimpleNamespace(seconds=30, max_evaluations=4096,
                           integral_evaluations=512, memory_mb=4096)
    tolerance, radius_limit = F(1, 10**64), base.a(F(1, 10**60))

    def check_budget():
        if time.perf_counter() - started > caps.seconds:
            raise base.EngineeringCap("Whole callback probe time cap reached")
        if sum(work.evaluations for work in works) > caps.max_evaluations:
            raise base.EngineeringCap("Whole callback probe work cap reached")

    def fresh():
        check_budget()
        remaining = caps.max_evaluations - sum(work.evaluations for work in works)
        if remaining <= 0:
            raise base.EngineeringCap("No remaining whole-probe evaluation budget")
        work_args = SimpleNamespace(seconds=caps.seconds, max_evaluations=remaining,
                                    integral_evaluations=caps.integral_evaluations,
                                    memory_mb=caps.memory_mb)
        work = base.Work(work_args)
        work.start = started
        works.append(work)
        return work, base.Integrals(work, tolerance)

    def polynomial(integrals):
        result = integrals.integrate(lambda z, analytic: z * z,
                                     base.c(0), base.c(1), "polynomial engineering control")
        passed = (result.is_finite() and result.contains(base.c(F(1, 3)))
                  and result.imag.contains(0)
                  and base.upper(result.rad()) < radius_limit)
        return passed, base.encoded(result)

    work, integrals = fresh()
    passed, enclosure = polynomial(integrals)
    records.append({"name": "ordinary_polynomial", "passed": passed,
                    "enclosure": enclosure, "work_evaluations": work.evaluations})
    check_budget()

    for nested, recoverable in ((False, False), (True, False),
                                (False, True), (True, True)):
        work, integrals = fresh()
        sentinel = base.Unresolved("precreated recoverable callback sentinel")
        child_exception, child_state, seen_integrands = [], [], [0]
        returned, caught = False, None

        def fail(z, analytic):
            seen_integrands[0] += 1
            if recoverable:
                raise sentinel
            return z * z  # Expired Work.check must run before this.

        def outer(z, analytic):
            if not recoverable:
                work.start = time.perf_counter() - (caps.seconds + 1)
            try:
                return integrals.integrate(fail, base.c(0), base.c(1), "nested fatal control")
            except BaseException as error:
                child_exception.append(error)
                child_state.append({"integration_depth": integrals.integration_depth,
                                    "fatal_latch_is_original_exception": integrals.fatal_callback_error is error})
                raise

        if not nested and not recoverable:
            work.start = time.perf_counter() - (caps.seconds + 1)
        try:
            integrals.integrate(outer if nested else fail,
                                base.c(0), base.c(1), "fatal callback control")
            returned = True
        except BaseException as error:
            caught = error
        expected = base.Unresolved if recoverable else base.EngineeringCap
        identity_ok = caught is sentinel if recoverable else isinstance(caught, expected)
        if nested:
            identity_ok = identity_ok and len(child_exception) == 1 and caught is child_exception[0]
            identity_ok = identity_ok and child_state == [{"integration_depth": 1,
                                                          "fatal_latch_is_original_exception": True}]
        reset_ok = integrals.integration_depth == 0 and integrals.fatal_callback_error is None
        result = {"name": ("nested_" if nested else "ordinary_") + ("recoverable" if recoverable else "time_cap"),
                  "exception_type": None if caught is None else type(caught).__name__,
                  "exception_message": None if caught is None else str(caught),
                  "original_exception_identity_preserved": identity_ok,
                  "fatal_integral_returned": returned,
                  "integration_depth_after_catch": integrals.integration_depth,
                  "fatal_latch_cleared": integrals.fatal_callback_error is None,
                  "failing_integrand_call_count": seen_integrands[0],
                  "nested_child_unwind_states": child_state,
                  "passed": not returned and isinstance(caught, expected) and identity_ok and reset_ok}
        if not recoverable:
            result["passed"] = result["passed"] and seen_integrands[0] == 0
        else:
            recovery_ok, recovery_enclosure = polynomial(integrals)
            result.update({"same_integrals_recovery_passed": recovery_ok,
                           "recovery_enclosure": recovery_enclosure})
            result["passed"] = result["passed"] and seen_integrands[0] == 1 and recovery_ok
        result["work_evaluations"] = work.evaluations
        records.append(result)
        check_budget()

    data = {"scope": "engineering-only real-backend callback controls; no model result",
            "backend": backend, "admission": admission, "precision_decimal_digits": base.ctx.dps,
            "predeclared_caps": vars(caps), "predeclared_radius_bound": base.encoded(radius_limit),
            "model_evaluation_attempted": False, "model_evaluated": False,
            "ground_preparation_certified": False, "ramsey_certified": False,
            "full_clock_certified": False, "COM_response_or_triplet_gap_certified": False,
            "checks": records, "all_checks_pass": all(record["passed"] for record in records),
            "evaluation_count": sum(work.evaluations for work in works),
            "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    if not data["all_checks_pass"]:
        raise RuntimeError("Actual callback regression controls failed")


if __name__ == "__main__":
    main()
