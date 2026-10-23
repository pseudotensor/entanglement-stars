"""Ground-only GT1--GT7 input checks for a reviewed direct compression.

No ground evaluator is imported here.  The separately audited standalone
codec, generalized-root and interval comparison paths are reused.  A
review acknowledgement is required in the caller, and never replaces the
actual schema, controls, error, exact-endpoint and consuming-precision
checks below.  No current projected-compression diagnostic is supplied.
"""

from __future__ import annotations

import json

import e_bound_clock_compression as direct


class InputUnresolved(RuntimeError):
    """The mathematical direct input cannot be admitted by this interface."""


def require(condition, reason):
    if not condition:
        raise InputUnresolved(reason)


def walk_endpoints(value, backend, work):
    """Check every retained endpoint object; display strings have no role."""
    if isinstance(value, dict):
        if "lower_dyadic" in value or "upper_dyadic" in value:
            work.check(1)
            backend.decode(value)
        else:
            for key, item in value.items():
                if key != "display_ball":
                    walk_endpoints(item, backend, work)
    elif isinstance(value, list):
        for item in value:
            walk_endpoints(item, backend, work)


def comparison_pass(record, backend, work, required_names=None):
    require(isinstance(record, dict) and record.get("status") == "all_full_enclosures_overlap"
            and record.get("all_same_exact_quantities_overlap") is True,
            "A required standalone full-enclosure comparison is incomplete")
    rows = record.get("comparisons")
    require(isinstance(rows, list) and rows, "A passed comparison lacks its endpoint rows")
    names = set()
    for row in rows:
        require(isinstance(row, dict) and row.get("overlap") is True,
                "A required standalone comparison contains a failed or missing overlap")
        names.add(row.get("quantity"))
        work.check(1)
        left, right = backend.decode(row.get("left_full_enclosure")), backend.decode(row.get("right_full_enclosure"))
        retained = backend.decode(row.get("intersection"))
        require(left.overlaps(right) and retained.overlaps(left) and retained.overlaps(right),
                "A standalone retained comparison contradicts its exact endpoint intervals")
    if required_names is not None:
        require(required_names <= names, "A standalone comparison omits a required same exact quantity")


def configuration_result(record, backend, work, expected):
    b = backend
    require(isinstance(record, dict) and record.get("configuration") == expected
            and record.get("status") == "this_configuration_full_integrals_and_root_gates_enclosed",
            "A direct configuration did not complete its full integral/root stage")
    require(record.get("method") == "exact_scalar_moments_product_Gauss"
            and record.get("trial_identifier") == direct.TRIAL_IDENTIFIER
            and record.get("measure_identifier") == direct.MEASURE_IDENTIFIER,
            "A direct configuration uses a different method/trial/measure")
    gates = record.get("positivity_and_root_gates")
    require(isinstance(gates, dict) and gates and all(value is True for value in gates.values()),
            "A direct configuration lacks passing strict positivity/root gates")
    require(record.get("same_exact_coupling_rule_across_all_refinements") is True,
            "A direct configuration does not enclose the same frozen coupling")
    for progress_name, completed_flag in (("moment_rule_accumulation_progress", "finite_region_node_sum_completed"),
                                          ("double_rule_accumulation_progress", "finite_square_double_sum_completed")):
        progress = record.get(progress_name)
        require(isinstance(progress, dict) and progress.get(completed_flag) is True
                and not any(value is True for key, value in progress.items() if key.startswith("incomplete_prefix")),
                "An incomplete quadrature prefix was labelled as a full integral")
    node_count = len(expected["panels"]) * expected["positive_scalar_Gauss_order_per_panel"]
    moment_progress = record["moment_rule_accumulation_progress"]
    double_progress = record["double_rule_accumulation_progress"]
    require(moment_progress.get("required_nodes") == node_count and moment_progress.get("moment_nodes_completed") == node_count
            and double_progress.get("required_symmetric_node_pairs") == node_count * (node_count + 1) // 2
            and double_progress.get("processed_symmetric_node_pairs") == double_progress["required_symmetric_node_pairs"],
            "A direct full integral lacks its completed finite-rule counts")
    rule = record.get("certified_algebraic_Gauss_rule")
    require(isinstance(rule, dict) and rule.get("all_real_nodes_inside_panel_and_all_weights_positive") is True
            and rule.get("root_solver_tolerance_used_as_certificate") is False and isinstance(rule.get("panels"), list),
            "The direct input lacks its certified positive algebraic Gauss-rule records")
    require(len(rule["panels"]) == len(expected["panels"]) and all(
            panel.get("completed") is True and len(panel.get("nodes_and_weights", [])) == expected["positive_scalar_Gauss_order_per_panel"]
            for panel in rule["panels"]), "A direct Gauss-rule panel is incomplete")
    cells = record.get("complete_ellipse_branch_and_majorant_records")
    require(isinstance(cells, list) and len(cells) == len(expected["panels"]),
            "The direct input lacks its complete analytic-ellipse records")
    for cell in cells:
        require(isinstance(cell, dict) and isinstance(cell.get("gates"), dict)
                and cell["gates"] and all(value is True for value in cell["gates"].values()),
                "A direct analytic ellipse lacks a proved branch margin")
    require(isinstance(record.get("analytic_quadrature_error_allocation"), dict)
            and record["analytic_quadrature_error_allocation"].get("exact_rule_positive_and_full_tensor_error_used") is True,
            "The direct input lacks its full analytic quadrature-error allocation")
    tails = record.get("positive_full_lattice_tails")
    require(isinstance(tails, dict)
            and tails.get("uses_full_measure_including_restricted_angles_and_van_Hove_region") is True
            and tails.get("full_sphere_density_continued_to_outer_region") is False,
            "The direct input does not retain its complete full-lattice outer-region bound")
    bound_names = ("polynomial_only_arithmetic_radius_upper_diagnostic",
                   "full_rule_joint_arithmetic_and_weight_series_radius_upper",
                   "full_minus_polynomial_rule_absolute_upper_decorrelation_diagnostic",
                   "analytic_quadrature_absolute_error_upper", "positive_full_outer_region_bound_upper")
    for container_name, quantities in (("moments", ("U0", "U1", "U2", "V0")),
                                       ("diagonal_compression_entries", ("K00", "K11"))):
        container = record.get(container_name)
        require(isinstance(container, dict), "A direct full-integral error record is absent")
        for name in quantities:
            scalar = container.get(name)
            require(isinstance(scalar, dict) and scalar.get("full_W_remainder_already_propagated_in_rule_ball") is True
                    and scalar.get("polynomial_and_decorrelation_diagnostics_added_as_extra_errors") is False,
                    "A direct scalar loses or double-counts its full-W uncertainty")
            for field in bound_names:
                work.check(1)
                require(b.decode(scalar.get(field)).lower() >= 0, "A direct scalar error bound is absent or negative")
            for field in ("finite_region_polynomial_weight_rule", "finite_region_full_W_Gauss_value",
                          "finite_exact_integral_after_Gauss_error", "full_exact_integral"):
                work.check(1)
                b.decode(scalar.get(field))
    prior_comparisons = record.get("all_completed_prior_matrix_comparisons")
    require(isinstance(prior_comparisons, list) and len(prior_comparisons) >= 3,
            "A direct configuration omits a completed prior full matrix comparison")
    for comparison in prior_comparisons:
        comparison_pass(comparison, b, work, {"G00", "G01", "G10", "G11", "K00", "K01", "K10", "K11"})
    gram = direct.decode_matrix(b, work, record.get("Gram"), "Direct configuration full Gram")
    matrix = direct.decode_matrix(b, work, record.get("compression"), "Direct configuration full compression")
    result = {"G" + str(i) + str(j): gram[i][j] for i in range(2) for j in range(2)}
    result.update({"K" + str(i) + str(j): matrix[i][j] for i in range(2) for j in range(2)})
    result.update({"mu2": b.decode(record.get("mu2")), "gamma": b.decode(record.get("gamma"))})
    return result


class DirectInput:
    def __init__(self, path, flint_module, admission, work, saved):
        self.b = direct.BallBackend(flint_module)
        self.b.activate(admission)
        self.saved, self.path = saved, path
        saved.update({"artifact": str(path), "status": "unresolved",
                      "transfer_source": "solve/strand_E_compression_transfer.tex",
                      "transfer_review": "solve/review_E_compression_transfer.tex",
                      "direct_compression_used_as_input": False, "direct_compression_is_ground_certificate": False,
                      "artifact_review_acknowledgement_is_not_schema_validation": True,
                      "consumer_rechecks": []})
        work.check()
        require(path.stat().st_size <= work.args.memory_mb * 1024**2,
                "Direct input text alone exceeds the declared resident-memory allowance")
        self.data = json.loads(path.read_text())
        work.check()
        self.validate_schema(work)
        self.mu2, self.gamma = self.recheck(work, saved["consumer_rechecks"])
        saved.update({"status": "validated_same_rule_mathematical_input",
                      "direct_compression_used_as_input": True,
                      "Cartesian_measure_control_status": "deferred_to_ground_stage",
                      "ground_adoption_allowed_from_direct_artifact": False})

    def validate_schema(self, work):
        b, data = self.b, self.data
        require(isinstance(data, dict), "The direct compression input is not a JSON object")
        for name, expected in (("stage", "direct_trial_compression"), ("source", direct.SPEC),
                               ("independent_source_review", direct.REVIEW), ("status", "compression_enclosed"),
                               ("fixed_parameters", direct.FIXED_PARAMETERS), ("trial_identifier", direct.TRIAL_IDENTIFIER),
                               ("measure_identifier", direct.MEASURE_IDENTIFIER),
                               ("backend", "isolated python-flint0.9.0; Arb directed ball arithmetic"),
                               ("Cartesian_measure_control_status", "deferred_to_ground_stage")):
            require(data.get(name) == expected, "Direct input has a different frozen definition or incomplete stage: " + name)
        for flag in ("ground_adoption_allowed", "model_evaluation_attempted", "model_evaluated", "ground_point_certified",
                     "ground_preparation_certified", "ramsey_certified", "full_clock_certified", "COM_enclosure_certified"):
            require(data.get(flag) is False, "Direct input is improperly promoted to a model or physical certificate: " + flag)
        admission = data.get("admission")
        require(isinstance(admission, dict) and all(admission.get(flag) is True for flag in
                ("same_process_fresh_device_query_before_allocation", "cuda_context_in_this_single_job_process",
                 "post_allocation_single_owned_compute_pid_verified")), "Direct input lacks its historical launch-admission records")
        require(type(admission.get("single_process_pid")) is int and admission["single_process_pid"] > 0
                and admission.get("niceness") == 10 and admission.get("maximum_threads") == 4
                and isinstance(admission.get("assigned_gpu_uuid"), str) and admission["assigned_gpu_uuid"].startswith("GPU-"),
                "Direct input historical process/backend scope is invalid")
        stage = data.get("direct_trial_compression")
        require(isinstance(stage, dict) and stage.get("status") == "compression_enclosed"
                and stage.get("method") == "exact_scalar_moments_product_Gauss"
                and stage.get("all_required_completion_controls_pass") is True
                and stage.get("same_exact_coupling_rule_across_all_refinements") is True
                and stage.get("trial_identifier") == direct.TRIAL_IDENTIFIER and stage.get("measure_identifier") == direct.MEASURE_IDENTIFIER
                and stage.get("Cartesian_measure_control_status") == "deferred_to_ground_stage"
                and stage.get("ground_adoption_allowed") is False,
                "The direct stage is incomplete or differs from the frozen exact compression")
        prior = stage.get("prior_projected_artifact")
        require(isinstance(prior, dict) and prior.get("status") == "all_declared_completed_prior_rows_validated"
                and prior.get("header_validated") is True and prior.get("raw_projected_compression_substituted") is False
                and prior.get("top_level_mu2_gamma_used") is False
                and prior.get("required_retry3_completed_rows_and_incomplete_row_present") is True,
                "The direct input does not retain its required full projected-prior control")
        completed, skipped = prior.get("completed_phase_rows"), prior.get("skipped_incomplete_phase_rows")
        require(isinstance(completed, list) and isinstance(skipped, list), "The direct prior row records are absent")
        orders = {row.get("configuration", {}).get("Legendre_order_per_panel") for row in completed if isinstance(row, dict)}
        skipped_orders = {row.get("configuration", {}).get("Legendre_order_per_panel") for row in skipped if isinstance(row, dict)}
        require({16, 32, 64} <= orders and 128 in skipped_orders,
                "The completed N16/N32/N64 or explicitly skipped N128 prior record is missing")
        for row in completed:
            require(row.get("status") == "full_widened_matrix_stage_accepted_for_comparison_only"
                    and row.get("phase") == "compression_and_determinants_enclosed_before_positivity_gates"
                    and isinstance(row.get("analytical_error_records"), dict), "A completed prior row lacks full widened error data")
            direct.decode_matrix(b, work, row.get("Gram"), "Retained prior full Gram")
            direct.decode_matrix(b, work, row.get("compression"), "Retained prior FULL widened compression")
            direct.decode_matrix(b, work, row.get("Gram_exterior_tail"), "Retained prior full Gram exterior tail")
            errors = row["analytical_error_records"]
            for name in ("compression_absolute_error_bounds", "operator_tail_projection", "operator_quadrature", "trial_projection_and_exterior"):
                direct.decode_matrix(b, work, errors.get(name), "Retained prior " + name, True)
            for name in ("operator_tail_projection_error_upper", "operator_quadrature_error_upper"):
                require(b.decode(errors.get(name)).lower() >= 0, "A retained prior operator error bound is absent or negative")
        require(all(isinstance(row.get("reason"), str) and row["reason"] for row in skipped),
                "An incomplete prior row lacks its explicit skip reason")
        attempts = stage.get("baseline_attempts")
        require(isinstance(attempts, list) and attempts, "The direct input has no baseline/control records")
        accepted = [row for row in attempts if row.get("status") == "baseline_and_both_separate_controls_completed"]
        require(len(accepted) == 1, "The direct input lacks its unique completed baseline/control stage")
        row = accepted[0]
        cfg = row.get("configuration")
        require(cfg == stage.get("successful_baseline_configuration"), "Successful direct configuration identifiers disagree")
        for name, ladder in (("cutoff", direct.CUTOFFS), ("positive_scalar_Gauss_order_per_panel", direct.QUADRATURES),
                             ("precision_decimal_digits", direct.PRECISIONS), ("weight_series_last_power", (8, 16, 32, 64, 128))):
            require(type(cfg.get(name)) is int and cfg[name] in ladder, "Invalid declared direct numerical choice: " + name)
        baseline_cfg = direct.Configuration(cfg["cutoff"], cfg["positive_scalar_Gauss_order_per_panel"],
                                            cfg["precision_decimal_digits"], cfg["weight_series_last_power"])
        require(cfg == baseline_cfg.saved(), "The direct baseline changes the frozen panel partition or chart")
        require(row.get("separate_baseline_relative_controls_both_completed") is True,
                "The two direct baseline-relative controls did not both complete")
        self.configuration_quantities = [configuration_result(row.get("baseline"), b, work, cfg)]
        next_q, next_precision = direct.next_configurations(baseline_cfg)
        require(next_q is not None and next_precision is not None, "The direct control ladders were exhausted")
        for name, next_cfg, changed in (("next_quadrature_control", next_q, "Gauss order"),
                                        ("next_precision_control", next_precision, "arithmetic precision")):
            control = row.get(name)
            require(isinstance(control, dict) and control.get("status") == "separate_baseline_relative_control_pass"
                    and control.get("baseline_configuration") == cfg and control.get("only_changed_numerical_choice") == changed
                    and control.get("baseline_weight_series_order_retained") is True,
                    "A separate baseline-relative direct control is incomplete or changes other choices")
            comparison_pass(control.get("baseline_relative_full_enclosure_overlap"), b, work)
            self.configuration_quantities.append(configuration_result(control.get("configuration_result"), b, work, next_cfg.saved()))
        walk_endpoints(stage, b, work)
        self.saved["retained_direct_control_and_error_records"] = stage
        self.saved["validated_direct_control_statuses"] = {
            "baseline_and_separate_next_quadrature_and_next_precision": True,
            "all_completed_prior_full_matrices_including_failed_positivity_N16": True,
            "incomplete_N128_skipped": True, "all_retained_endpoint_and_error_records": True,
            "Cartesian_control_still_mandatory_in_ground_stage": True}

    def recheck(self, work, records):
        """Repeat strict roots/overlaps at CURRENT consuming precision."""
        b = self.b
        record = {"consuming_precision_decimal_digits": int(b.ctx.dps), "status": "unresolved"}
        records.append(record)
        reference = self.data["direct_trial_compression"].get("tightened_common_reference")
        require(isinstance(reference, dict), "The direct common-intersection reference is missing")
        gram = direct.decode_matrix(b, work, reference.get("Gram"), "Imported common full Gram")
        matrix = direct.decode_matrix(b, work, reference.get("compression"), "Imported common full compression")
        for label, values in (("G", gram), ("K", matrix)):
            require(values[0][1].lower() == values[1][0].lower() and values[0][1].upper() == values[1][0].upper(),
                    "Imported full " + label + " is not a symmetric interval matrix")
        imported_mu2, imported_gamma = b.decode(reference.get("mu2")), b.decode(reference.get("gamma"))
        record["imported_full_Gram"] = [[b.encode(value) for value in row] for row in gram]
        record["imported_full_compression"] = [[b.encode(value) for value in row] for row in matrix]
        record["imported_mu2"], record["imported_gamma"] = b.encode(imported_mu2), b.encode(imported_gamma)
        values = {"U0": gram[0][0], "U1": gram[0][1], "U2": gram[1][1],
                  "V0": b.decode(reference.get("moments", {}).get("V0")),
                  "K00": matrix[0][0], "K01": matrix[0][1], "K11": matrix[1][1]}
        record["literal_Gram_moment_identity_intersections"] = []
        for name in ("U0", "U1", "U2"):
            comparison = {}
            record["literal_Gram_moment_identity_intersections"].append(comparison)
            intersection = direct.intersect_pair(b, values[name], b.decode(reference.get("moments", {}).get(name)), name, comparison)
            if intersection is None:
                raise direct.InconsistentEnclosures("Imported full Gram contradicts its same exact moment record")
            values[name] = intersection
        record["literal_K01_full_moment_identity_intersection"] = {}
        identity = (values["U0"]**2 + values["V0"]**2) / 2
        intersection = direct.intersect_pair(b, values["K01"], identity, "K01 exact full moment identity",
                                             record["literal_K01_full_moment_identity_intersection"])
        if intersection is None:
            raise direct.InconsistentEnclosures("Imported K01 contradicts its exact full moment identity")
        values["K01"] = intersection
        recomputed = direct.generalized_root(b, values, record)
        record["same_exact_scalar_import_intersections"] = {}
        scalar = direct.compare_enclosures(b, recomputed, {"mu2": imported_mu2, "gamma": imported_gamma},
                                           record["same_exact_scalar_import_intersections"], {"mu2", "gamma"})
        record["top_level_and_common_scalar_intersections"] = {}
        scalar = direct.compare_enclosures(b, scalar, {name: b.decode(self.data.get(name)) for name in ("mu2", "gamma")},
                                           record["top_level_and_common_scalar_intersections"], {"mu2", "gamma"})
        record["repeated_baseline_and_both_control_overlaps"] = []
        for quantities in self.configuration_quantities:
            comparison = {}
            record["repeated_baseline_and_both_control_overlaps"].append(comparison)
            direct.compare_enclosures(b, recomputed, quantities, comparison, set(quantities))
        record["repeated_all_completed_prior_full_matrix_overlaps"] = []
        for row in self.data["direct_trial_compression"]["prior_projected_artifact"]["completed_phase_rows"]:
            g = direct.decode_matrix(b, work, row["Gram"], "Prior comparison Gram")
            k = direct.decode_matrix(b, work, row["compression"], "Prior comparison compression")
            quantities = {"G" + str(i) + str(j): g[i][j] for i in range(2) for j in range(2)}
            quantities.update({"K" + str(i) + str(j): k[i][j] for i in range(2) for j in range(2)})
            for name in ("mu2", "gamma"):
                if row.get(name) is not None:
                    quantities[name] = b.decode(row[name])
            comparison = {"prior_attempt_index": row["attempt_index"]}
            record["repeated_all_completed_prior_full_matrix_overlaps"].append(comparison)
            direct.compare_enclosures(b, recomputed, quantities, comparison, set(quantities))
        record.update({"status": "strict_roots_and_full_prior_control_overlaps_rechecked",
                       "rechecked_mu2": b.encode(scalar["mu2"]), "rechecked_gamma": b.encode(scalar["gamma"])})
        return scalar["mu2"], scalar["gamma"]
