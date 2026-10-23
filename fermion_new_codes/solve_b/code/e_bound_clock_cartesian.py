"""MC1--MC17: the SAME finite-cube Cartesian control, scalar representation.

Only the separate TG entry calls this helper.  It uses the caller's already
admitted directed backend and Work object; importing it activates no backend
and does not import the TG caller.  E author and independent B invoked-source
audits precede any separately parent-admitted execution.
"""

from __future__ import annotations

from fractions import Fraction as F
import math

import e_bound_clock as base
import e_bound_clock_ground_input as input_api


SPEC = "solve/strand_E_cartesian_control.tex"
REVIEW = "solve/review_E_cartesian_control.tex"
METHOD = "positive_Laplace_cube_Gaussian_moments_coordinate_Jacobian_polynomial"
JACOBIAN_DEGREE = 8
PARAMETER_CUTOFF = 8
ELLIPSE_PARAMETER = 2
PARAMETER_ORDERS = (64, 128)
PANEL_ENDPOINTS = (("0", "1/2"), ("1/2", "1"), ("1", "2"), ("2", "4"), ("4", "8"))
MOMENT_POWERS = (-2, -1, 0, 1, 2)
PARAMETER_POWERS = (-2, -1, 1)


def _finite(value, label):
    if not value.is_finite():
        raise base.Unresolved(label + " is not a finite directed enclosure")
    return value


def _nonnegative(value, label):
    _finite(value, label)
    if value.lower() < 0:
        raise base.Unresolved(label + " is not certified nonnegative")
    return value


def _endpoint(value):
    if not value.is_finite() or not value.is_exact():
        raise ArithmeticError("MC Cartesian intersection endpoint is not a finite exact dyadic")
    return base.dyadic(value)


def _operation(work, record, charge, label, function):
    """Preserve the original rejecting-request semantics and partial evidence."""
    record.update({"operation": label, "proposed_charge": charge, "preceding_counter": work.evaluations,
                   "guard_admitted": False, "backend_result_returned": False, "completed_under_caps": False})
    try:
        work.check(charge)
        record["guard_admitted"] = True
        result = function()
        record["backend_result_returned"] = True
        work.check()
        record["completed_under_caps"] = True
        return result
    except BaseException as error:
        record["failure"] = {"class": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["counter_after"] = work.evaluations


def _reserve(work, progress, tokens, entries, label):
    row = {"label": label, "proposed_entries": entries, "creation_binary_precision": int(base.ctx.prec),
           "guard_admitted": False, "currently_reserved": False,
           "price_selection": "unchanged TG maximum of current and all active creation precisions"}
    progress["helper_memory_reservations"].append(row)
    try:
        token = work.reserve(entries, label)
        tokens.append(token)
        row.update({"token": token, "guard_admitted": True, "currently_reserved": True})
    except BaseException as error:
        row["failure"] = {"class": type(error).__name__, "message": str(error)}
        raise


def _moments(argument, sqrt_pi, work, record=None):
    """MC5 complete Gaussian moments G0,...,G9 on the real positive axis."""
    work.check()
    argument = base.positive(argument, "MC complete Gaussian moment root argument")
    values = [sqrt_pi / (2 * argument.sqrt())]
    if record is not None:
        record.update({"status": "incomplete_bounded_moment_recurrence", "argument": base.encoded(argument),
                       "last_complete_moment_sequence": [base.encoded(values[0])]})
    for n in range(9):
        work.check()
        values.append((2 * n + 1) * values[-1] / (2 * argument))
        if record is not None:
            record["last_complete_moment_sequence"].append(base.encoded(values[-1]))
    for value in values:
        base.positive(value, "MC complete Gaussian moment")
    work.check()
    if record is not None:
        record["status"] = "all_ten_complete_Gaussian_moments_enclosed"
    return values


def _legs(moments, coefficients):
    return [sum((coefficient * moments[n + j] for n, coefficient in enumerate(coefficients)), base.a(0))
            for j in (0, 1)]


def _setup(d, progress):
    work = d.work
    progress["phase"] = "bounded coefficient, coordinate-tail and Jacobian setup"
    alpha, cutoff = F(base.ALPHA), F(d.cutoff)
    if alpha != F(1, 1_000_000) or d.cutoff not in (16, 24, 32, 48, 64):
        raise base.Unresolved("MC Cartesian helper requires the unchanged exact reference alpha and cutoff ladder")
    rho = alpha**2 * cutoff**2
    if not 0 < rho < 1:
        raise base.Unresolved("MC actual coordinate Jacobian ratio does not lie strictly between zero and one")
    sqrt_pi = base.positive(base.arb.pi(), "MC directed pi").sqrt()
    linear_prefactor = 1 / (4 * base.arb.pi()**3)
    coordinate = {}
    progress["coordinate_Jacobian_and_cutoff_bounds_MC4_MC9"] = coordinate
    coordinate.update({"rho_exact_rational": str(rho), "rho_strictly_below_one": True,
                       "coefficient_c_n_exact_rationals": [], "coefficient_c_n_alpha_to_2n_balls": [],
                       "coordinate_tail_B_n_upper_MC6": [], "Gaussian_moment_recurrence_at2": {}})
    exact_coefficients, coefficients = [], []
    for n in range(JACOBIAN_DEGREE + 1):
        work.check()
        coefficient = F(math.comb(2 * n, n), 4**n)
        if not 0 < coefficient <= 1:
            raise ArithmeticError("MC exact coordinate-Jacobian coefficient violates its proved bound")
        exact_coefficients.append(coefficient)
        coefficients.append(base.a(coefficient * alpha**(2 * n)))
        coordinate["coefficient_c_n_exact_rationals"].append(str(coefficient))
        coordinate["coefficient_c_n_alpha_to_2n_balls"].append(base.encoded(coefficients[-1]))
    eta = base.upper(base.a(rho**(JACOBIAN_DEGREE + 1) / (1 - rho)))
    coordinate["Jacobian_eta_upper_MC4"] = base.encoded(eta)
    gaussian = _moments(base.a(2), sqrt_pi, work, coordinate["Gaussian_moment_recurrence_at2"])
    coordinate.update({"coefficient_c_n_exact_rationals": [str(value) for value in exact_coefficients],
                       "coefficient_c_n_alpha_to_2n_balls": [base.encoded(value) for value in coefficients],
                       "rho_exact_rational": str(rho), "rho_strictly_below_one": True,
                       "Jacobian_eta_upper_MC4": base.encoded(eta),
                       "Gaussian_moments_at2_MC5": [base.encoded(value) for value in gaussian]})
    exp_coordinate = (-2 * base.a(cutoff)**2).exp()
    tails = [base.upper(exp_coordinate / (4 * base.a(cutoff)))]
    coordinate["coordinate_tail_B_n_upper_MC6"].append(base.encoded(tails[0]))
    for n in range(1, 10):
        work.check()
        tails.append(base.upper(base.a(cutoff)**(2 * n - 1) * exp_coordinate / 4 + base.a(F(2 * n - 1, 4)) * tails[-1]))
        coordinate["coordinate_tail_B_n_upper_MC6"].append(base.encoded(tails[-1]))
    p = _legs(gaussian, coefficients)
    e = [base.upper(sum((coefficient * tails[n + j] for n, coefficient in enumerate(coefficients)), base.a(0)))
         for j in (0, 1)]
    dleg = [base.upper(e[j] + eta * gaussian[j]) for j in (0, 1)]
    u = [base.upper(p[j] + dleg[j]) for j in (0, 1)]
    b0 = base.upper(3 * u[0]**2 * dleg[0])
    b2 = base.upper(b0 + 3 * (dleg[1] * u[0]**2 + 2 * u[1] * u[0] * dleg[0]))
    for value in tails + e + dleg + u + [eta, b0, b2]:
        _nonnegative(value, "MC positive coordinate/product error bound")
    coordinate.update({"coordinate_tail_B_n_upper_MC6": [base.encoded(value) for value in tails],
                       "polynomial_coordinate_tail_E_j_upper_MC7": [base.encoded(value) for value in e],
                       "Jacobian_leg_error_upper": [base.encoded(base.upper(eta * gaussian[j])) for j in (0, 1)],
                       "complete_leg_D_j_upper_MC7": [base.encoded(value) for value in dleg],
                       "P_j_at0_balls": [base.encoded(value) for value in p],
                       "both_real_leg_U_j_upper": [base.encoded(value) for value in u],
                       "complete_product_b0_upper_MC8": base.encoded(b0), "complete_product_b2_upper_MC8": base.encoded(b2),
                       "all_cross_terms_and_coordinate_tail_products_retained": True,
                       "exact_Jacobian_continued_beyond_physical_endpoint": False})
    progress["phase"] = "positive omitted parameter-tail setup"
    parameter_exp = base.a(-PARAMETER_CUTOFF**2).exp()
    parameter_tails = {-2: base.upper(p[0]**3 * parameter_exp),
                       -1: base.upper(p[0]**3 * parameter_exp / (sqrt_pi * PARAMETER_CUTOFF)),
                       1: base.upper((p[0]**3 + 3 * p[1] * p[0]**2) * parameter_exp / (sqrt_pi * PARAMETER_CUTOFF))}
    progress["positive_parameter_tail_upper_MC14"] = {
        str(power): base.encoded(_nonnegative(value, "MC positive omitted parameter tail"))
        for power, value in parameter_tails.items()}
    panels = []
    progress["complete_parameter_ellipse_certificates_MC11_MC13"] = []
    for left_text, right_text in PANEL_ENDPOINTS:
        work.check()
        left, right = F(left_text), F(right_text)
        center, half = (left + right) / 2, (right - left) / 2
        low_real, high_real = center - F(5, 4) * half, center + F(5, 4) * half
        distance = F(0) if low_real <= 0 <= high_real else min(abs(low_real), abs(high_real))
        delta = 2 + distance**2 - F(9, 16) * half**2
        certificate = {"left_exact_rational": str(left), "right_exact_rational": str(right),
                       "center_exact_rational": str(center), "halfwidth_exact_rational": str(half),
                       "real_rectangle_endpoints_exact_rationals": [str(low_real), str(high_real)],
                       "imaginary_semiaxis_exact_rational": str(F(3, 4) * half),
                       "x_star_exact_rational": str(distance), "delta_exact_rational": str(delta),
                       "delta_strictly_positive": delta > 0, "status": "unresolved"}
        progress["complete_parameter_ellipse_certificates_MC11_MC13"].append(certificate)
        if delta <= 0:
            raise base.Unresolved("MC complete parameter ellipse lacks a strict right-half-plane branch margin")
        certificate["majorant_Gaussian_moment_recurrence"] = {}
        stars = _legs(_moments(base.a(delta), sqrt_pi, work, certificate["majorant_Gaussian_moment_recurrence"]), coefficients)
        radius = center + 2 * half
        exponential = base.a(F(9, 16) * half**2 - distance**2).exp()
        majorants = {-2: base.upper(2 * base.a(radius) * exponential * stars[0]**3),
                     -1: base.upper(2 * exponential / sqrt_pi * stars[0]**3),
                     1: base.upper(2 * exponential / sqrt_pi * (stars[0]**3 + 3 * stars[1] * stars[0]**2))}
        errors = {order: {power: base.upper(8 * base.a(right - left) * majorants[power]
                                                   * base.arb((1, -2 * order))) for power in PARAMETER_POWERS}
                  for order in PARAMETER_ORDERS}
        for value in list(majorants.values()) + [value for order_errors in errors.values() for value in order_errors.values()]:
            _nonnegative(value, "MC ellipse majorant or complete Gauss-error bound")
        certificate.update({"root_branch": "principal sqrt on complete Re(2+s^2)>0 ellipse",
                            "P_j_star_balls_MC11": [base.encoded(value) for value in stars],
                            "R_exact_rational": str(radius), "exponential_A_ball_MC11": base.encoded(exponential),
                            "integrand_M_a_upper_MC12": {str(power): base.encoded(value) for power, value in majorants.items()},
                            "Gauss_absolute_error_upper_MC13_by_order": {
                                str(order): {str(power): base.encoded(value) for power, value in order_errors.items()}
                                for order, order_errors in errors.items()},
                            "complete_ellipse_not_just_center_certified": True, "status": "complete_analytic_ellipse_certified"})
        panels.append({"left": left, "right": right, "center": center, "half": half, "errors": errors,
                       "certificate": certificate})
    work.check()
    return {"coefficients": coefficients, "sqrt_pi": sqrt_pi, "p": p, "product_errors": {-2: b0, -1: b0, 0: b0, 1: b2, 2: b2},
            "parameter_tails": parameter_tails, "panels": panels, "linear_prefactor": linear_prefactor}


def _parameter_order(d, setup, order, progress, record):
    work = d.work
    record.update({"parameter_Gauss_order": order, "status": "unresolved", "panels": [],
                   "raw_finite_parameter_rule_sums": None, "fully_widened_moments": {}, "full_Cartesian_cubes": [],
                   "original_MC15_tests": [], "completed_parameter_nodes": 0, "all_five_panels_completed": False})
    progress["phase"] = "inherited positive Gauss rule Q=" + str(order)
    record["inherited_rule_operation"] = {}
    expected_rule_charge = 0 if (base.ctx.prec, order) in work.rules else order
    record["inherited_rule_operation"]["uncached_inherited_charge_if_needed"] = expected_rule_charge
    rule = _operation(work, record["inherited_rule_operation"], 0, "MC inherited positive Gauss rule Q=" + str(order), lambda: work.rule(order))
    if len(rule) != order:
        raise ArithmeticError("MC inherited positive Gauss rule has the wrong node count")
    totals = {power: base.a(0) for power in PARAMETER_POWERS}
    error_totals = {power: base.a(0) for power in PARAMETER_POWERS}
    for panel_index, panel in enumerate(setup["panels"]):
        cell = {"panel_index": panel_index, "left_exact_rational": str(panel["left"]),
                "right_exact_rational": str(panel["right"]), "parameter_Gauss_order": order,
                "ellipse_certificate": panel["certificate"], "status": "incomplete_rule_prefix",
                "completed_node_count": 0, "required_node_count": order, "node_operations": [],
                "last_complete_three_integrand_rule_prefix_sums": {str(power): base.encoded(base.a(0)) for power in PARAMETER_POWERS},
                "incomplete_prefix_not_a_full_rule_quadrature_enclosure": True,
                "Gauss_absolute_error_upper_MC13": {str(power): base.encoded(value) for power, value in panel["errors"][order].items()}}
        record["panels"].append(cell)
        sums = {power: base.a(0) for power in PARAMETER_POWERS}
        for node_index, (node, canonical_weight) in enumerate(rule):
            progress["phase"] = "Q=" + str(order) + ", parameter panel=" + str(panel_index) + ", node=" + str(node_index)
            operation = {"node_index": node_index}
            cell["node_operations"].append(operation)
            def evaluate_node():
                parameter = base.a(panel["center"]) + base.a(panel["half"]) * node
                weight = base.positive(base.a(panel["half"]) * canonical_weight, "MC actual physical positive Gauss weight")
                if not parameter.lower() > base.a(panel["left"]).upper() or not parameter.upper() < base.a(panel["right"]).lower():
                    raise base.Unresolved("MC actual parameter-node enclosure leaves its declared panel")
                p0, p1 = _legs(_moments(2 + parameter**2, setup["sqrt_pi"], work), setup["coefficients"])
                exponential = (-parameter**2).exp()
                integrands = {-2: 2 * parameter * exponential * p0**3,
                              -1: 2 / setup["sqrt_pi"] * exponential * p0**3,
                              1: 2 / setup["sqrt_pi"] * exponential * (p0**3 + 3 * p1 * p0**2)}
                following = {power: _finite(sums[power] + weight * integrands[power], "MC complete rule-prefix sum")
                             for power in PARAMETER_POWERS}
                serialized = {str(power): base.encoded(value) for power, value in following.items()}
                return following, serialized
            sums, serialized = _operation(work, operation, 32, "MC complete shared Gaussian-moment/integrand node", evaluate_node)
            cell["completed_node_count"] = node_index + 1
            cell["last_complete_three_integrand_rule_prefix_sums"] = serialized
            record["completed_parameter_nodes"] += 1
        work.check()
        cell.update({"status": "complete_finite_positive_rule", "incomplete_prefix_not_a_full_rule_quadrature_enclosure": False})
        totals = {power: totals[power] + sums[power] for power in PARAMETER_POWERS}
        error_totals = {power: base.upper(error_totals[power] + panel["errors"][order][power]) for power in PARAMETER_POWERS}
        record["raw_finite_parameter_rule_sums"] = {str(power): base.encoded(value) for power, value in totals.items()}
        record["completed_panel_count"] = panel_index + 1
    record["all_five_panels_completed"] = True
    progress["phase"] = "Q=" + str(order) + ", full moment widening and Cartesian linear combinations"
    raw = dict(totals)
    raw[0], raw[2] = setup["p"][0]**3, setup["p"][0]**3 + 3 * setup["p"][1] * setup["p"][0]**2
    full = {}
    for power in MOMENT_POWERS:
        work.check()
        row = {"epsilon_power": power, "raw_surrogate_ball": base.encoded(raw[power]),
               "raw_surrogate_arithmetic_radius": base.encoded(raw[power].rad()), "status": "unresolved"}
        record["fully_widened_moments"][str(power)] = row
        current = raw[power]
        if power in PARAMETER_POWERS:
            current = base.widen(current, error_totals[power])
            positive_tail = base.interval_hull(base.a(0), setup["parameter_tails"][power])
            current += positive_tail
            row.update({"summed_complete_ellipse_Gauss_error_upper_MC13": base.encoded(error_totals[power]),
                        "positive_parameter_tail_interval_outward_MC14": base.encoded(positive_tail)})
        full[power] = _finite(base.widen(current, setup["product_errors"][power]), "MC full finite-cube moment")
        row.update({"surrogate_after_ellipse_and_parameter_tail": base.encoded(current),
                    "complete_coordinate_Jacobian_product_error_upper_MC9": base.encoded(setup["product_errors"][power]),
                    "full_finite_cube_moment": base.encoded(full[power]), "status": "full_finite_cube_moment_enclosed",
                    "diagnostic_error_categories_already_added_once": True})
    cubes = []
    record["raw_finite_parameter_surrogate_cubes_diagnostic_only"] = []
    for k in (0, 1, 2):
        work.check()
        record["raw_finite_parameter_surrogate_cubes_diagnostic_only"].append(
            base.encoded(setup["linear_prefactor"] * (raw[k] + 2 * raw[k - 1] + raw[k - 2])))
        cube = _finite(setup["linear_prefactor"] * (full[k] + 2 * full[k - 1] + full[k - 2]), "MC full Cartesian cube")
        cubes.append(cube)
        record["full_Cartesian_cubes"].append({"epsilon_power": k, "full_Cartesian_cube": base.encoded(cube),
                                               "joint_full_ball_radius_includes_all_errors_once": True})
    work.check()
    record["status"] = "full_Cartesian_enclosures_formed_original_MC15_tests_pending"
    return cubes


def _original_gate(cartesian, radial, tail, record):
    """Literal retained comparator, exterior allowance and radius gate MC15."""
    radius = base.upper(cartesian.rad() + radial.rad())
    discrepancy = cartesian - radial
    allowed = base.interval_hull(-radius, tail + radius)
    scale = max(base.a(1), base.abs_upper(radial))
    threshold = base.a(F(1, 10**12)) * scale
    agreement, accurate = discrepancy.overlaps(allowed), radius < threshold
    record.update({"Cartesian_cube": base.encoded(cartesian), "radial_sphere": base.encoded(radial),
                   "cube_minus_sphere": base.encoded(discrepancy), "positive_Gaussian_tail_upper": base.encoded(tail),
                   "directed_integration_radius_sum": base.encoded(radius), "literal_positive_tail_allowed_ball": base.encoded(allowed),
                   "UNCHANGED_original_radius_threshold": base.encoded(threshold),
                   "positive_tail_agreement_passed": agreement, "original_radius_gate_passed": accurate,
                   "joint_ball_radii_used_once_no_error_category_double_count": True, "status": "unresolved"})
    if not agreement:
        raise base.Unresolved("Cartesian/radial eight-preimage Gram control fails positive-tail agreement")
    if not accurate:
        raise base.Unresolved("Cartesian normalization control arithmetic radius remains unresolved")
    record["status"] = "both_literal_original_MC15_gates_pass"


def measure_controls(d, progress):
    """Return the caller-owned record only after both fixed orders and all gates."""
    if not isinstance(progress, dict):
        raise TypeError("MC measure_controls requires the caller-owned progress dictionary")
    work, tokens = d.work, []
    progress.update({"method": METHOD, "source": SPEC, "independent_source_review": REVIEW,
                     "point": {"source_parameter_exact": "0", "alpha_exact_rational": str(base.ALPHA)},
                     "cutoff": d.cutoff, "coordinate_Jacobian_degree": JACOBIAN_DEGREE,
                     "parameter_cutoff": PARAMETER_CUTOFF, "ellipse_parameter": ELLIPSE_PARAMETER,
                     "fixed_separate_parameter_Gauss_orders": list(PARAMETER_ORDERS), "status": "unresolved",
                     "phase": "initial shared-work and memory admission", "source_review_required": False,
                     "separate_parameter_order_controls": [], "radial_comparator_rows": [], "rows": [],
                     "full_two_order_overlap_tests_before_intersections": [],
                     "exact_endpoint_intersections_and_outward_conversions": [],
                     "coordinate_Jacobian_and_cutoff_bounds_MC4_MC9": {}, "positive_parameter_tail_upper_MC14": {},
                     "complete_parameter_ellipse_certificates_MC11_MC13": [], "helper_memory_reservations": [],
                     "cached_rule_reservations_remain_owned_by_shared_Work": True,
                     "preceding_requested_counter_at_helper_entry": work.evaluations,
                     "angular_first_three_exact": [str(value) for value in d.coefficients[:3]],
                     "Cartesian_factor": "8/(2*pi)^3; octant symmetry after exact physical eight-preimage change of variables",
                     "core_proxy_contract": {"fixed_setup": 1024, "per_complete_shared_parameter_node": 32,
                         "required_parameter_nodes": 960, "at_most_uncached_inherited_rule_units": 192,
                         "declared_complete_core_upper": 31936, "other_original_radial_native_work_is_additional": True,
                         "no_total_cost_or_success_prediction": True, "completed": False}})
    try:
        work.check()
        _reserve(work, progress, tokens, 512, "MC simultaneously live coefficient/tail/ellipse/final-combination setup arrays")
        _reserve(work, progress, tokens, 128, "MC bounded streaming Gaussian moment and parameter-node scratch")
        progress["fixed_setup_operation"] = {}
        core_before = work.evaluations
        setup = _operation(work, progress["fixed_setup_operation"], 1024, "MC fixed bounded coefficient/moment/tail/ellipse/combinations setup",
                           lambda: _setup(d, progress))
        cubes_by_order = []
        for order in PARAMETER_ORDERS:
            record = {"parameter_Gauss_order": order, "status": "unresolved"}
            progress["separate_parameter_order_controls"].append(record)
            _reserve(work, progress, tokens, 256, "MC retained complete order/panel sums/errors/moment/cube result arrays Q=" + str(order))
            cubes_by_order.append(_parameter_order(d, setup, order, progress, record))
        progress["core_proxy_contract"].update({"actual_requested_core_units_before_original_radial_work": work.evaluations - core_before,
                                               "both_fixed_orders_and_all_960_nodes_completed": True,
                                               "fixed_final_combination_work_included_in_setup_charge": True, "completed": True})
        # A contradiction between purported full enclosures must not be hidden
        # by a later ordinary comparator failure.  No intersection is formed
        # until both orders have passed every literal original MC15 gate.
        progress["phase"] = "same exact full cubes, two-order overlap before original comparator tests"
        for power in (0, 1, 2):
            work.check()
            left, right = cubes_by_order[0][power], cubes_by_order[1][power]
            overlap = left.overlaps(right)
            progress["full_two_order_overlap_tests_before_intersections"].append({
                "epsilon_power": power, "order64_full_cube": base.encoded(left), "order128_full_cube": base.encoded(right),
                "same_exact_full_cube_enclosures_overlap": overlap, "intersection_formed_before_original_gates": False})
            if not overlap:
                progress["source_review_required"] = True
                raise input_api.direct.InconsistentEnclosures("MC Cartesian method requires source review: disjoint full Q64/Q128 cube enclosures")
        _reserve(work, progress, tokens, 64, "MC retained unchanged radial comparator and positive-tail balls")
        radial, tails = [], []
        for power in (0, 1, 2):
            row = {"epsilon_power": power, "status": "original_adaptive_radial_comparator_pending"}
            progress["radial_comparator_rows"].append(row)
            progress["phase"] = "original adaptive radial comparator, epsilon power=" + str(power)
            work.check()
            value = _finite(d.inner(d.trial(0), d.trial(power)), "MC original radial comparator")
            row["radial_sphere"] = base.encoded(value)
            tail = _nonnegative(d.trial_tail(power), "MC unchanged positive exterior allowance")
            row.update({"positive_Gaussian_tail_upper": base.encoded(tail), "status": "original_radial_and_tail_enclosed"})
            radial.append(value)
            tails.append(tail)
            work.check()
        for cubes, record in zip(cubes_by_order, progress["separate_parameter_order_controls"]):
            progress["phase"] = "individual original MC15 gates Q=" + str(record["parameter_Gauss_order"])
            for power in (0, 1, 2):
                work.check()
                row = {"epsilon_power": power, "status": "unresolved"}
                record["original_MC15_tests"].append(row)
                _original_gate(cubes[power], radial[power], tails[power], row)
                work.check()
            record["status"] = "this_order_full_cube_and_all_original_MC15_gates_pass"
        _reserve(work, progress, tokens, 128, "MC simultaneous exact intersections/outward conversions/final returned-ball tests")
        for power in (0, 1, 2):
            progress["phase"] = "exact two-order intersection and returned-ball original gates, epsilon power=" + str(power)
            work.check()
            left, right = cubes_by_order[0][power], cubes_by_order[1][power]
            lo, hi = max(left.lower(), right.lower()), min(left.upper(), right.upper())
            overlap = lo <= hi
            row = {"epsilon_power": power, "status": "unresolved", "same_exact_full_cube_enclosures_overlap": overlap,
                   "order64_full_cube": base.encoded(left), "order128_full_cube": base.encoded(right),
                   "exact_outward_endpoint_intersection": None}
            progress["exact_endpoint_intersections_and_outward_conversions"].append(row)
            if not overlap:
                progress["source_review_required"] = True
                raise input_api.direct.InconsistentEnclosures("MC Cartesian method requires source review: disjoint full Q64/Q128 cube enclosures")
            row["exact_outward_endpoint_intersection"] = {"lower_dyadic": _endpoint(lo), "upper_dyadic": _endpoint(hi)}
            returned = _finite(base.interval_hull(lo, hi), "MC actual returned outward intersection ball")
            if not returned.lower() <= lo or not returned.upper() >= hi:
                raise ArithmeticError("MC outward conversion does not contain both exact intersection endpoints")
            left_extra = base.upper(lo - returned.lower())
            right_extra = base.upper(returned.upper() - hi)
            row.update({"actual_returned_outward_ball": base.encoded(returned),
                        "actual_returned_ball_radius": base.encoded(returned.rad()),
                        "conversion_extra_left_endpoint_width_upper": base.encoded(left_extra),
                        "conversion_extra_right_endpoint_width_upper": base.encoded(right_extra),
                        "conversion_rounding_retained_in_returned_ball_not_added_twice": True})
            legacy = {"epsilon_power": power, "status": "unresolved"}
            progress["rows"].append(legacy)
            row["final_returned_ball_original_MC15_test"] = legacy
            _original_gate(returned, radial[power], tails[power], legacy)
            row["status"] = "exact_intersection_outward_conversion_and_repeated_original_gates_pass"
            work.check()
        progress.update({"phase": "complete", "status": "both_separate_orders_and_all_original_Cartesian_radial_gates_pass",
                         "all_returned_rows_use_actual_outward_intersection_ball": True,
                         "native_kernel_controls_still_required_after_return": True})
        work.check()
        return progress
    except BaseException as error:
        progress.update({"status": "unresolved", "failure": {"class": type(error).__name__, "module": type(error).__module__,
                                                                  "message": str(error), "phase": progress["phase"]}})
        raise
    finally:
        progress["requested_counter_at_helper_exit"] = work.evaluations
        progress["actual_requested_helper_units_including_original_radial_and_rejections"] = (
            work.evaluations - progress["preceding_requested_counter_at_helper_entry"])
        for token in reversed(tokens):
            work.release(token)
        for row in progress["helper_memory_reservations"]:
            row["currently_reserved"] = False
