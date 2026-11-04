#!/usr/bin/env python3
"""Engineering-only regression of N's real nested Acb callback boundary.

The exact contract is strand_N_callback_probe.tex and its independent
review.  This program reads no ground artifact and invokes no model
constructor.  Its source and the optional production per-call cap need
N-author and independent A0 audits before a parent-admitted remote run.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
import os
from pathlib import Path
import time

import n_pair_com as base
from remote_runtime import remote_execution


SPEC = "solve/strand_N_callback_probe.tex"
REVIEW = "solve/review_N_callback_probe_spec.tex"
POOL_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


class ProbeUnresolved(RuntimeError):
    """An independent regression/budget failure, distinct from injected errors."""


class WholeBudget:
    """Independent 30-second clock and sum of every actual Work counter."""

    def __init__(self, backend):
        backend.require()
        self.backend, self.started = backend, time.perf_counter()
        self.seconds, self.checks, self.per_call = 30, 4096, 512
        self.works, self.observations = [], []

    def reconcile(self, label):
        elapsed = time.perf_counter() - self.started
        counts = [work.evaluations for work in self.works]
        counted = sum(counts)
        valid = elapsed <= self.seconds and 0 <= counted <= self.checks
        row = {"checkpoint": label, "independent_elapsed_seconds": elapsed,
               "actual_work_counts": counts, "sum_actual_work_counts": counted,
               "remaining_seconds": self.seconds - elapsed,
               "remaining_counted_checks": self.checks - counted,
               "independent_whole_probe_budgets_valid": valid}
        self.observations.append(row)
        if not valid:
            raise ProbeUnresolved("Independent whole-probe 30s/4096 budget exhausted at " + label)
        return row

    def fresh(self, label):
        row = self.reconcile("before constructing work: " + label)
        remaining_seconds = row["remaining_seconds"]
        remaining_checks = row["remaining_counted_checks"]
        if remaining_seconds <= 0 or remaining_checks <= 1:
            raise ProbeUnresolved("No positive remaining time or rejecting-increment reserve for " + label)
        assigned_checks = remaining_checks - 1
        # Work.check(1) increments before rejecting.  That one rejection
        # remains within the independent global counted budget.
        work = base.Work(self.backend, seconds=remaining_seconds,
                         host_bytes=4 * 1024**3, max_evaluations=assigned_checks)
        self.works.append(work)
        integrals = base.Integrals(self.backend, work, 128, eval_limit=self.per_call)
        self.reconcile("after constructing work: " + label)
        return work, integrals, {"assigned_seconds": remaining_seconds,
                                 "assigned_counted_check_limit": assigned_checks,
                                 "one_rejecting_increment_reserved": True,
                                 "per_call_Acb_evaluation_limit": integrals.eval_limit}


def polynomial(backend, budget, work, integrals, panel, label):
    row = {"name": label, "passed": False, "initial_depth": work.integration_depth,
           "initial_latch_clear": work.integration_exception is None,
           "work_identity": id(work), "integrals_identity": id(integrals)}
    try:
        row["budget_before"] = budget.reconcile("before polynomial/recovery: " + label)
        value = integrals.integrate(lambda z, analytic: z**2, panel, label)
        target = backend.complex(F(1, 3))
        radius = backend.absolute_upper(value - value.mid()) if value.is_finite() else None
        threshold = backend.real(F(1, 10**60)).lower()
        flags = {"finite": value.is_finite(),
                 "contains_directed_exact_rational_target_ball": value.is_finite() and value.contains(target),
                 "imaginary_part_contains_zero": value.is_finite() and value.imag.contains(0),
                 "actual_radius_upper_below_exact_threshold": radius is not None and radius < threshold,
                 "initial_shared_depth_zero": row["initial_depth"] == 0,
                 "initial_shared_latch_clear": row["initial_latch_clear"],
                 "final_shared_depth_zero": work.integration_depth == 0,
                 "final_shared_latch_clear": work.integration_exception is None}
        row.update({"checks": flags, "enclosure": backend.encode(value),
                    "target_enclosure": backend.encode(target),
                    "radius_upper": None if radius is None else backend.encode(radius),
                    "radius_threshold_exact": str(F(1, 10**60)),
                    "final_depth": work.integration_depth, "final_latch_clear": work.integration_exception is None,
                    "requested_tolerance_goal_only": backend.encode(integrals.goal)})
        row["budget_after"] = budget.reconcile("after polynomial/recovery: " + label)
        row["passed"] = all(flags.values())
        if not row["passed"]:
            row["unresolved_reason"] = "Polynomial enclosure or shared-state control failed"
    except BaseException as error:
        row.update({"passed": False, "unexpected_exception_type": type(error).__name__,
                    "unresolved_reason": str(error), "final_depth": work.integration_depth,
                    "final_latch_clear": work.integration_exception is None})
        try:
            row["budget_after"] = budget.reconcile("after failed polynomial/recovery: " + label)
        except ProbeUnresolved as budget_error:
            row["independent_budget_failure"] = str(budget_error)
    row["work_evaluations"] = work.evaluations
    return row


def failure_case(backend, budget, panel, nested, recoverable):
    label = ("nested_" if nested else "ordinary_") + ("recoverable" if recoverable else "time_cap")
    before = budget.reconcile("before failure case: " + label)
    work, outer, allocation = budget.fresh(label)
    child = base.Integrals(backend, work, 128, eval_limit=512) if nested else None
    sentinel = base.Unresolved("N callback probe precreated recoverable sentinel: " + label) if recoverable else None
    expected = [sentinel]
    record = {"name": label, "passed": False, "allocation": allocation,
              "budget_before": before, "initial_depth": work.integration_depth,
              "initial_latch_clear": work.integration_exception is None,
              "failing_function_entries": 0, "scientific_polynomial_body_entries": 0,
              "failing_function_depths": [], "outer_function_entries": 0,
              "outer_function_depths": [], "outer_integral_returned_any_result": False,
              "outer_integral_returned_finite_result": False,
              "outer_integral_returned_nonfinite_partial_result": False,
              "inner_python_exceptions": [], "observer_errors": [], "post_inner_observations": [],
              "post_inner_flags": {"exactly_one_observation": not nested,
                                    "inner_returned_invalid": not nested,
                                    "depth_still_positive": not nested,
                                    "depth_is_outer_callback_one": not nested,
                                    "original_shared_latch_retained": not nested,
                                    "no_inner_python_exception": True,
                                    "observers_completed_without_error": True},
              "same_work_for_outer_and_child": not nested or outer.work is child.work is work,
              "different_actual_nested_Integrals_instances": not nested or outer is not child,
              "work_identity": id(work), "outer_integrals_identity": id(outer),
              "child_integrals_identity": None if child is None else id(child)}
    original_work, original_outer = work, outer
    caught = None

    def fail(z, analytic):
        record["failing_function_entries"] += 1
        record["failing_function_depths"].append(work.integration_depth)
        if recoverable:
            raise sentinel
        # The actual callback has already entered C at the observed
        # depth.  Only this test object's clock is deliberately expired.
        work.start = time.perf_counter() - work.seconds - 1
        record["actual_work_clock_expired_inside_failing_callback"] = True
        try:
            work.check()  # Unchanged production checker, increment zero.
        except base.EngineeringCap as error:
            expected[0] = error
            record["actual_time_cap_type_at_throw"] = type(error).__name__
            record["actual_time_cap_message_at_throw"] = str(error)
            record["actual_time_cap_identity_at_throw"] = id(error)
            raise
        record["scientific_polynomial_body_entries"] += 1
        return z**2  # This body must never be reached.

    def nested_outer(z, analytic):
        record["outer_function_entries"] += 1
        record["outer_function_depths"].append(work.integration_depth)
        try:
            inner_value = child.integrate(fail, panel, label + " actual inner C integration")
        except BaseException as error:
            record["inner_python_exceptions"].append({"type": type(error).__name__, "message": str(error),
                                                       "identity": id(error)})
            record["post_inner_flags"]["no_inner_python_exception"] = False
            return outer.invalid
        # Every observation is an independent flag.  An assertion raised
        # here could be masked by the correct first-error preservation.
        try:
            latched = work.integration_exception
            observation = {"inner_returned_invalid": not inner_value.is_finite(),
                           "integration_depth_after_inner": work.integration_depth,
                           "original_shared_exception_is_still_latched":
                               latched is not None and expected[0] is not None and latched[0] is expected[0]}
            record["post_inner_observations"].append(observation)
            flags = record["post_inner_flags"]
            flags["exactly_one_observation"] = len(record["post_inner_observations"]) == 1
            flags["inner_returned_invalid"] = observation["inner_returned_invalid"]
            flags["depth_still_positive"] = observation["integration_depth_after_inner"] > 0
            flags["depth_is_outer_callback_one"] = observation["integration_depth_after_inner"] == 1
            flags["original_shared_latch_retained"] = observation["original_shared_exception_is_still_latched"]
        except BaseException as error:
            record["observer_errors"].append({"type": type(error).__name__, "message": str(error)})
            record["post_inner_flags"]["observers_completed_without_error"] = False
        return inner_value

    try:
        value = outer.integrate(nested_outer if nested else fail, panel, label + " actual outer C integration")
        record["outer_integral_returned_any_result"] = True
        record["outer_integral_returned_finite_result"] = value.is_finite()
        record["outer_integral_returned_nonfinite_partial_result"] = not value.is_finite()
    except BaseException as error:
        caught = error
    record.update({"original_exception_type": None if expected[0] is None else type(expected[0]).__name__,
                   "original_exception_message": None if expected[0] is None else str(expected[0]),
                   "original_exception_identity": None if expected[0] is None else id(expected[0]),
                   "caught_exception_type": None if caught is None else type(caught).__name__,
                   "caught_exception_message": None if caught is None else str(caught),
                   "caught_exception_identity": None if caught is None else id(caught),
                   "original_exception_identity_preserved": expected[0] is not None and caught is expected[0],
                   "depth_after_outer_catch": work.integration_depth,
                   "shared_latch_clear_after_outer_catch": work.integration_exception is None,
                   "work_evaluations_after_failure": work.evaluations})
    expected_type = base.Unresolved if recoverable else base.EngineeringCap
    type_exact = caught is not None and type(caught) is expected_type
    depths_ok = (record["failing_function_depths"] == [1] if not nested
                 else len(record["failing_function_depths"]) == 1 and record["failing_function_depths"][0] >= 2)
    ordinary_flags = {
        "initial_shared_depth_zero": record["initial_depth"] == 0,
        "initial_shared_latch_clear": record["initial_latch_clear"],
        "expected_exception_exact_type": type_exact,
        "original_exception_identity_preserved": record["original_exception_identity_preserved"],
        "failing_function_entered_exactly_once": record["failing_function_entries"] == 1,
        "failure_in_actual_required_C_depth": depths_ok,
        "later_polynomial_body_unvisited": record["scientific_polynomial_body_entries"] == 0,
        "outer_never_returned_finite_or_partial_result": not record["outer_integral_returned_any_result"],
        "final_shared_depth_zero": work.integration_depth == 0,
        "final_shared_latch_clear": work.integration_exception is None,
        "nested_outer_entered_once_at_C_depth_one": not nested or
            record["outer_function_entries"] == 1 and record["outer_function_depths"] == [1],
        "same_Work_two_Integrals_if_nested": record["same_work_for_outer_and_child"],
        "actual_different_child_instance_if_nested": record["different_actual_nested_Integrals_instances"],
    }
    record["checks"] = ordinary_flags
    try:
        record["budget_after_failure"] = budget.reconcile("after failure case: " + label)
    except ProbeUnresolved as error:
        record["independent_budget_failure"] = str(error)
        record["unresolved_reason"] = str(error)
        return record
    record["passed"] = all(ordinary_flags.values()) and all(record["post_inner_flags"].values())
    if not record["passed"]:
        record["unresolved_reason"] = "Exception identity, actual-depth, independent observer or clean-state control failed"
        return record
    if recoverable:
        recovery = polynomial(backend, budget, work, outer, panel, label + " same-object recovery")
        same_objects = work is original_work and outer is original_outer and outer.work is work
        record.update({"recovery": recovery, "same_actual_Work_and_outer_Integrals_reused_for_recovery": same_objects})
        record["passed"] = record["passed"] and recovery["passed"] and same_objects
        if not record["passed"]:
            record["unresolved_reason"] = "Same-object polynomial recovery failed"
    else:
        record["expired_work_not_reset_or_used_for_recovery"] = True
    record["work_evaluations_after_case_and_recovery"] = work.evaluations
    try:
        record["budget_after_case_and_recovery"] = budget.reconcile("after complete case/recovery: " + label)
    except ProbeUnresolved as error:
        record.update({"passed": False, "independent_budget_failure": str(error), "unresolved_reason": str(error)})
    return record


def run(backend, admission):
    backend.activate(admission, 128)
    budget = WholeBudget(backend)
    panel = base.Panel(F(0), F(1))
    data = {"scope": "engineering-only N callback/error-latch regression; no model or physical result",
            "specification": SPEC, "independent_specification_review": REVIEW,
            "backend": "isolated python-flint0.9.0 directed Arb/Acb", "admission": admission,
            "precision_decimal_digits": 128,
            "fixed_caps": {"whole_probe_seconds_cooperative": 30, "whole_probe_counted_work_checks": 4096,
                           "per_call_Acb_evaluation_limit": 512, "host_limit_bytes_per_Work": 4 * 1024**3,
                           "own_threads_at_most": 4, "arithmetic_workers": 1,
                           "one_rejecting_counted_increment_reserved_for_each_Work": True},
            "production_default_per_call_Acb_limit_unchanged": 10**6,
            "production_checker_callback_backend_and_latch_monkeypatched": False,
            "model_evaluation_attempted": False, "model_evaluated": False,
            "ground_artifact_read": False, "model_constructor_invoked": False,
            "ground_preparation_certified": False, "COM_enclosure_certified": False,
            "arithmetic_complete_for_N_model": False, "enclosure_target_pass": False,
            "COM_response_or_triplet_gap_certified": False,
            "interacting_native_Ward_error_certified": False, "differentiable_physical_window_certified": False,
            "ramsey_certified": False, "full_clock_certified": False,
            "physical_clock_load_continuation_certified": False,
            "full_fixed_charge_Fock_ground_certified": False, "open_exterior_flow_certified": False,
            "local_momentum_row_certified": False, "trapping_radius_certified": False, "temperature_certified": False,
            "all_checks_pass": False, "status": "unresolved", "unresolved_reason": None,
            "checks": []}
    try:
        budget.reconcile("before ordinary polynomial case")
        work, integrals, allocation = budget.fresh("ordinary polynomial")
        ordinary = polynomial(backend, budget, work, integrals, panel, "ordinary polynomial")
        ordinary["allocation"] = allocation
        data["checks"].append(ordinary)
        budget.reconcile("after ordinary polynomial case")
        if not ordinary["passed"]:
            raise ProbeUnresolved(ordinary.get("unresolved_reason", "Ordinary polynomial case failed"))
        for nested, recoverable in ((False, False), (True, False), (False, True), (True, True)):
            record = failure_case(backend, budget, panel, nested, recoverable)
            data["checks"].append(record)
            if not record["passed"]:
                raise ProbeUnresolved(record.get("unresolved_reason", "Expected callback case failed"))
        data["final_budget"] = budget.reconcile("before final acceptance")
        all_observers = all(row["independent_whole_probe_budgets_valid"] for row in budget.observations)
        complete_cases = len(data["checks"]) == 5 and all(row["passed"] for row in data["checks"])
        clean_works = all(work.integration_depth == 0 and work.integration_exception is None for work in budget.works)
        data["final_acceptance_checks"] = {"ordinary_and_all_four_failure_cases_pass": complete_cases,
                                            "all_independent_budget_observers_pass": all_observers,
                                            "all_Work_latches_depths_clean": clean_works}
        data["all_checks_pass"] = complete_cases and all_observers and clean_works
        data["status"] = "engineering_callback_controls_pass" if data["all_checks_pass"] else "unresolved"
        if not data["all_checks_pass"]:
            data["unresolved_reason"] = "One or more independent final acceptance checks failed"
    except BaseException as error:
        data.update({"all_checks_pass": False, "status": "unresolved",
                     "unresolved_reason": type(error).__name__ + ": " + str(error)})
    finally:
        try:
            data["budget_after_all_cases_and_final_acceptance"] = budget.reconcile("after all cases/recoveries/final acceptance")
        except ProbeUnresolved as error:
            data.update({"all_checks_pass": False, "status": "unresolved", "unresolved_reason": str(error)})
        data["independent_budget_observations"] = budget.observations
        data["actual_work_evaluation_counts"] = [work.evaluations for work in budget.works]
        data["whole_probe_counted_work_checks"] = sum(data["actual_work_evaluation_counts"])
        data["elapsed_engineering_seconds"] = budget.observations[-1]["independent_elapsed_seconds"]
    return data


def write_result(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")
    lines = [data["scope"], "all_checks_pass=" + str(data["all_checks_pass"]),
             "status=" + data["status"],
             "counted_work_checks=" + str(data["whole_probe_counted_work_checks"]),
             "elapsed_engineering_seconds=" + str(data["elapsed_engineering_seconds"])]
    for row in data["checks"]:
        lines.append(row["name"] + ": passed=" + str(row["passed"]))
    if data.get("unresolved_reason"):
        lines.append("reason: " + data["unresolved_reason"])
    lines.append("Model, ground, COM, Ward, clock, full Fock, open flow, trapping and temperature claims: false.")
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--actual-code-audits-released", action="store_true")
    args = parser.parse_args()
    if not args.actual_code_audits_released:
        parser.error("N-author/A0 actual-source audits and parent runtime admission are required")
    if args.output.suffix != ".json":
        parser.error("Output must have a .json suffix with a separate .txt companion")
    if args.output.resolve() == args.output.with_suffix(".txt").resolve():
        parser.error("JSON and TXT output destinations must be distinct")
    if not args.output.parent.is_dir() or args.output.exists() or args.output.with_suffix(".txt").exists():
        parser.error("Output must be new JSON/TXT files in an existing parent-admitted result directory")
    if any(os.environ.get(name) != "1" for name in POOL_NAMES):
        parser.error("All six numerical worker pools must be predeclared as one")
    if os.environ.get("QI2_GPU_UUID") != args.gpu_uuid:
        parser.error("Probe GPU UUID must equal the parent-selected launch environment")
    backend = base.BallBackend.load_host()  # Final host-only import before admission.
    admission = remote_execution(max_threads=4)
    print("N callback probe admitted " + json.dumps(admission), flush=True)
    data = run(backend, admission)
    write_result(args.output, data)
    if not data["all_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
