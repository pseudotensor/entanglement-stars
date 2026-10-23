#!/usr/bin/env python3
"""HOLD 18.19 recovered-source endpoint norm, sparse energy and cut current.

No eigensolver, Gram product, propagation, repair or native-channel rerun.
Both code review and a pinned, accepted complete source pass precede execution.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy as np

from hold_error_bounds import (downwards, gamma_rational, pairwise_sum, rational,
                               sqrt_upper, upwards)
from hold_exact import log, write_json
from hold_recovered_input import load_recovered
from hold_source_action import require_complex_abi, require_real_abi
from hold_source_preflight import OFFSETS, identity
from hold_source_runtime import environment_check, resources
from hold_source_seed_control import energy_allowance, sqrt_interval


VERSION = "recovered-original-A-k3-origin-control-1"
SUPPORT = tuple(range(2040, 2056))  # zero-based spinor rows, sites 1021..1028
CUT_ROW = 2048                     # chi=1 at original sites 1025..8192


def rational_record(value):
    value = rational(value)
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def require_finite(*arrays):
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ArithmeticError("A recovered-origin input/contraction is nonfinite")


def operator_rows(bands):
    """Validate exact antisymmetry, boundaries, spinor ordering and five terms."""
    require_real_abi(bands)
    if bands.shape != (6, 16384):
        raise ValueError("The cached endpoint has another band layout")
    require_finite(bands)
    rows = []
    for i in range(16384):
        row = {}
        for band, offset in enumerate(OFFSETS):
            j = i+offset
            value = float(bands[band, i])
            if not 0 <= j < 16384:
                if value != 0:
                    raise ValueError("A cached endpoint violates its open boundary")
                continue
            if value:
                if abs(i//2-j//2) > 2:
                    raise ValueError("An endpoint link exceeds the range-two site stencil")
                row[j] = value
        if len(row) > 5:
            raise ValueError("The five-product endpoint arithmetic model is inapplicable")
        rows.append(row)
    row_bound = Fraction(0)
    for i, row in enumerate(rows):
        for j, value in row.items():
            if rows[j].get(i, 0.) != -value:
                raise ValueError("Cached h=iB is not exactly Hermitian in the given ordering")
        row_bound = max(row_bound, sum((abs(rational(v)) for v in row.values()), Fraction(0)))
    return rows, row_bound


def exact_current_pairs(rows):
    """Exact binary-coefficient short paths for J=i[h^2,chi]/2=i*j."""
    exact = {}
    # Examine every structural h^2 path; only cut-crossing entries survive.
    for i, row in enumerate(rows):
        for middle, left in row.items():
            for j, right in rows[middle].items():
                if j <= i or (j >= CUT_ROW) == (i >= CUT_ROW):
                    continue
                key = (i, j)
                sign = int(j >= CUT_ROW)-int(i >= CUT_ROW)
                exact[key] = exact.get(key, Fraction(0))-sign*rational(left)*rational(right)/2
    exact = {key: value for key, value in exact.items() if value}
    if not exact or len(exact) > 120 or any(i not in SUPPORT or j not in SUPPORT for i, j in exact):
        raise ValueError("The actual cut-current support exceeds the specified 16 rows")
    pairs = []
    norm_rows = {i: Fraction(0) for i in SUPPORT}
    for (i, j), value in sorted(exact.items()):
        coefficient = float(value)
        if not math.isfinite(coefficient) or coefficient == 0:
            raise ArithmeticError("A nonzero exact current coefficient is not representable")
        delta = abs(value-rational(coefficient))
        magnitude = abs(rational(coefficient))+delta
        norm_rows[i] += magnitude
        norm_rows[j] += magnitude
        pairs.append(dict(rows=[i, j], coefficient=coefficient,
            paired_coefficient=-coefficient, exact_coefficient=rational_record(value),
            coefficient_error_exact=rational_record(delta), coefficient_error_upper=upwards(delta)))
    return pairs, max(norm_rows.values())


def pairwise_row_sum(values):
    """The first eight levels of the existing C-order depth-22 tree."""
    if values.shape != (16384, 256) or values.dtype != np.float64:
        raise ValueError("The spatial energy tree requires the fixed real product block")
    for _ in range(8):
        values = values[:, 0::2]+values[:, 1::2]
    return values[:, 0].copy()


def save_raw(path, squares, real, imag, spatial_real, spatial_imag, spatial_binding, current=None):
    values = dict(block_indices=np.arange(len(squares)),
        block_squared_norm=np.array(squares), block_energy_real=np.array(real),
        block_energy_imaginary=np.array(imag),
        recovered_source_energy_row_block_real=spatial_real,
        recovered_source_energy_row_block_imaginary=spatial_imag,
        recovered_spatial_completed_block_count=np.array(len(squares)),
        recovered_spatial_completed_block_mask=np.arange(32) < len(squares),
        recovered_spatial_spinor_row_indices=np.arange(16384),
        recovered_spatial_column_ranges=np.array([[256*b, 256*(b+1)] for b in range(32)]),
        recovered_spatial_binding_json=json.dumps(spatial_binding, sort_keys=True))
    if current is not None:
        values.update(current)
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **values)
    temporary.replace(path)


def save_operator(source, root, bands):
    metadata = source.metadata
    record = json.loads(Path(metadata["record"]).read_text())
    matches = [row for row in record["records"] if row.get("E") is not None
               and row["T"] == metadata["time_start"]]
    if len(matches) != 1 or matches[0]["E"] != metadata["charge"]:
        raise ValueError("The actual endpoint record has another charge/clock")
    args = record["args"]
    expected = dict(g=.25, rs=64., tau=8., ell=4, nsites=2048, half_offset=0,
                    site_fields="averaged", m=.2, reading="A", R0_over_rs=4.)
    if any(args[key] != value for key, value in expected.items()):
        raise ValueError("The endpoint field-history declaration differs")
    with np.load(metadata["fields"], allow_pickle=False) as fields:
        k = metadata["k"]
        times = fields["T"]
        positions = np.flatnonzero(times == metadata["time_start"])
        radius, clock, flow = fields["R"], fields["N_applied"][k], fields["v_applied"][k]
        if (positions.tolist() != [k] or not np.array_equal(radius, np.arange(1, 2049, dtype=float))
                or clock.shape != (2048,) or flow.shape != (2048,)
                or clock.dtype != np.float64 or flow.dtype != np.float64
                or not clock.dtype.isnative or not flow.dtype.isnative or np.any(clock <= 0)):
            raise ValueError("The endpoint applied-field grid/clock/layout differs")
        require_finite(radius, clock, flow)
        target = root/"held_operator_bands.npz"
        np.savez(target, bands=bands, offsets=OFFSETS, inner_radius=radius,
            inner_clock_applied=clock, inner_flow_applied=flow)
    source.check()
    return dict(artifact=identity(target), endpoint_operator=source.operator,
        fields_key="applied", site_order="2*(original site-1)+spin, original sites 1..8192",
        operator_convention="h=iB, exact placement from accepted unscaled endpoint cache; no reassembly",
        exterior_convention="N=1, v=-sqrt(2*g*charge/R) outside engine sites 1..2048",
        history_args=expected, unique_endpoint_record=True, unique_endpoint_field_index=metadata["k"])


def energy_and_norm(source, bands, report, report_path, raw_path):
    squares, real_parts, imag_parts = [], [], []
    spatial_real = np.full((16384, 32), np.nan)
    spatial_imag = np.full_like(spatial_real, np.nan)
    spatial_binding = dict(method=VERSION, source_kind=source.metadata["source_kind"],
        source_array=source.metadata["source_array"],
        source_construction=source.metadata["source_construction"],
        endpoint=source.metadata["k"], time=source.metadata["time_start"],
        endpoint_operator=source.operator, operator_artifact=report["operator"]["artifact"],
        control_output_path=str(report_path),
        consumer_implementations={name: source.inputs.current[str(Path(__file__).with_name(name).resolve())]
            for name in ("hold_recovered_input.py", "hold_recovered_origin.py")},
        identity_scope="prospective inputs/code/path binding; accepted final origin JSON also binds raw archive content")
    report["spatial_energy"] = dict(binding=spatial_binding, shape=[16384, 32],
        completed_block_count=0, completed_block_mask=[False]*32,
        layout="zero-based spinor row then 256-column source-block index",
        incomplete_entries="NaN until that block is complete; incomplete archive cannot be consumed",
        row_tree_depth=8, whole_block_row_tree_depth=14, whole_outer_block_tree_depth=5,
        later_sharp_mask_reduction_depth_upper=19, total_arithmetic_gamma=29,
        allowance="same global energy B_E per component for every sharp 0/1 site mask",
        imaginary_meaning="unsymmetrized chi-h energy; generally nonzero, not a Hermiticity residual",
        full_mask_reassociation="later row/block association may differ from the main block/outer-block total")
    lower2_total, upper2_total = Fraction(0), Fraction(0)
    zeros_x = np.zeros((16384, 256), dtype=np.complex128)
    zeros_z2 = np.zeros_like(zeros_x)
    applied = np.empty_like(zeros_x)
    rho = rational(source.method.rho)
    for block in range(32):
        resources(report_path.parent)
        source.check()
        start, stop = 256*block, 256*(block+1)
        x = np.array(source.occupied[:, start:stop], copy=True, order="C")
        arrays = (zeros_x, x, zeros_z2, applied)
        for array in arrays:
            require_complex_abi(array, writable=True)
        if (any(np.shares_memory(arrays[i], arrays[j]) for i in range(4) for j in range(i))
                or any(np.shares_memory(bands, array) for array in arrays)):
            raise ValueError("The raw endpoint product buffers overlap")
        require_finite(x)
        status = source.method.library.hold_source_stage(16384, 256, bands,
            zeros_x, x, zeros_z2, applied, 1, 0., 0.)
        require_finite(applied)
        if status:
            raise ArithmeticError("The raw endpoint C product failed")
        squared = pairwise_sum(x.real*x.real+x.imag*x.imag)
        real_rows = pairwise_row_sum(x.real*applied.real+x.imag*applied.imag)
        imag_rows = pairwise_row_sum(x.real*applied.imag-x.imag*applied.real)
        require_finite(real_rows, imag_rows)
        # The depth8 row trees followed by this depth14 tree are exactly the
        # original flattened C-order tree, with no repeated product/reduction.
        real = pairwise_sum(real_rows)
        imag = pairwise_sum(imag_rows)
        require_finite(squared, real, imag)
        if squared <= 0:
            raise ArithmeticError("An endpoint block norm is not positive")
        lower2 = rational(squared)/(1+gamma_rational(24))
        upper2 = rational(squared)/(1-gamma_rational(24))
        lower, upper = sqrt_interval(lower2, upper2)
        report["blocks"].append(dict(index=block, columns=[start, stop], entry_count=2**22,
            reduction_tree_depth=22, squared_norm_measured=squared,
            squared_norm_lower=downwards(lower2), squared_norm_upper=upwards(upper2),
            norm_lower=lower, norm_upper=upper, energy_real=real, energy_imaginary=imag,
            energy_error_upper_per_component=upwards(energy_allowance(rho, upper2, 24))))
        lower2_total += lower2
        upper2_total += upper2
        squares.append(squared)
        real_parts.append(real)
        imag_parts.append(imag)
        spatial_real[:, block] = real_rows
        spatial_imag[:, block] = imag_rows
        report["spatial_energy"].update(completed_block_count=block+1,
            completed_block_mask=[b <= block for b in range(32)])
        save_raw(raw_path, squares, real_parts, imag_parts, spatial_real, spatial_imag, spatial_binding)
        write_json(report_path, report)
    lower, upper = sqrt_interval(lower2_total, upper2_total)
    real, imag = pairwise_sum(real_parts), pairwise_sum(imag_parts)
    allowance = energy_allowance(rho, upper2_total, 29)
    require_finite(spatial_real, spatial_imag)
    report["spatial_energy"]["fixed_array_error_upper_per_component_for_any_sharp_mask"] = upwards(allowance)
    imaginary_ok = abs(rational(imag)) <= allowance
    report["energy"] = dict(real=real, imaginary=imag, fixed_array_error_upper_per_component=upwards(allowance),
        fixed_array_real_interval=[downwards(rational(real)-allowance), upwards(rational(real)+allowance)],
        whole_Hermitian_imaginary_control_passed=bool(imaginary_ok), energy_reduction_gamma=29,
        arithmetic_model="raw h_s five-product/four-add C product gamma9; explicit real products and depth22+5 trees")
    if not imaginary_ok:
        raise ArithmeticError("The whole Hermitian endpoint energy fails its imaginary control")
    consistent = {}
    for kind, entry in source.source_error["totals"].items():
        error = rational(entry["error_upper"])
        seed_low, seed_high = rational(source.seed_control["norm_lower"]), rational(source.seed_control["norm_upper"])
        target_low, target_high = max(Fraction(0), seed_low-error), seed_high+error
        consistent[kind] = dict(enlarged_initial_norm_interval=[downwards(target_low), upwards(target_high)],
            endpoint_interval_intersects=bool(rational(upper) >= target_low and rational(lower) <= target_high))
    report["norm"] = dict(lower=lower, upper=upper,
        squared_lower=downwards(lower2_total), squared_upper=upwards(upper2_total),
        trace_measured=pairwise_sum(squares), trace_deviation_from_8192=pairwise_sum(squares)-8192.,
        trace_deviation_interval=[downwards(lower2_total-8192), upwards(upper2_total-8192)],
        source_norm_consistency=consistent, block_gamma=24, outer_tree_depth=5,
        meaning="actual stored recovered covariance; no isometry assumption or repair")
    if not all(item["endpoint_interval_intersects"] for item in consistent.values()):
        raise ArithmeticError("Recovered norm is inconsistent with the accepted unitary-source bound")
    source.check()
    return squares, real_parts, imag_parts, spatial_real, spatial_imag, spatial_binding, upper2_total, allowance


def current_control(source, pairs, current_norm_bound):
    phi = np.array(source.occupied[list(SUPPORT), :], copy=True, order="C")
    require_finite(phi)
    q_measured, q_lower, q_upper = [], [], []
    row_records = []
    for local, row in enumerate(SUPPORT):
        values = phi[local]
        measured = pairwise_sum(values.real*values.real+values.imag*values.imag)
        require_finite(measured)
        if measured < 0:
            raise ArithmeticError("A current-support row norm is negative")
        lower = rational(measured)/(1+gamma_rational(15))
        upper = rational(measured)/(1-gamma_rational(15))
        q_measured.append(measured)
        q_lower.append(lower)
        q_upper.append(upper)
        row_records.append(dict(row=row, original_site=row//2+1, spin=row%2,
            squared_norm_measured=measured, squared_norm_lower=downwards(lower), squared_norm_upper=upwards(upper)))
    components = dict(pair_contractions=Fraction(0), final_products_and_tree=Fraction(0),
                      coefficient_rounding=Fraction(0))
    contributions, t_values = [], []
    product_scale = Fraction(0)
    for pair in pairs:
        i, j = pair["rows"]
        left, right = phi[i-SUPPORT[0]], phi[j-SUPPORT[0]]
        measured = pairwise_sum(left.imag*right.real-left.real*right.imag)
        coefficient = pair["coefficient"]
        contribution = 2.*(coefficient*measured)
        require_finite(measured, contribution)
        norm_product = sqrt_upper(q_upper[i-SUPPORT[0]]*q_upper[j-SUPPORT[0]])
        t_error = gamma_rational(15)*norm_product
        exact_delta = pair["coefficient_error_exact"]
        delta = Fraction(int(exact_delta["numerator"]), int(exact_delta["denominator"]))
        components["pair_contractions"] += 2*abs(rational(coefficient))*t_error
        product_scale += 2*abs(rational(coefficient))*abs(rational(measured))
        components["coefficient_rounding"] += 2*delta*norm_product
        pair.update(t_hat=measured, t_error_upper=upwards(t_error), measured_pair_contribution=contribution)
        t_values.append(measured)
        contributions.append(contribution)
    depth = (max(1, len(pairs))-1).bit_length()
    components["final_products_and_tree"] = gamma_rational(1+depth)*product_scale
    allowance = sum(components.values(), Fraction(0))
    measured = pairwise_sum(contributions)
    require_finite(measured)
    result = dict(real=measured, fixed_array_error_upper=upwards(allowance),
        fixed_array_real_interval=[downwards(rational(measured)-allowance), upwards(rational(measured)+allowance)],
        component_error_upper={key: upwards(value) for key, value in components.items()},
        verified_support_spinor_rows=list(SUPPORT), original_support_sites=[1021, 1028],
        actual_nonzero_support_rows=sorted({row for pair in pairs for row in pair["rows"]}),
        cut_radius=1024., exterior="original sites R>1024a", supported_row_norms=row_records,
        supported_squared_norm_lower=downwards(sum(q_lower, Fraction(0))),
        supported_squared_norm_upper=upwards(sum(q_upper, Fraction(0))),
        exact_stored_operator_norm_upper=upwards(current_norm_bound),
        operator_norm_model="max paired row sum of abs(jhat)+exact coefficient error",
        pairs=pairs, independent_pair_count=len(pairs), final_tree_depth=depth,
        contraction_gamma=15, contraction_tree_depth=13,
        imaginary_control="real by exact Hermitian pairing identity; unsymmetrized imaginary contraction not performed",
        convention="C-only J=i[h_s^2,chi_out]/2 at original-history endpoint start; no vacuum subtraction")
    raw = dict(current_support_rows=np.array(SUPPORT), current_row_squared_norm=np.array(q_measured),
        current_pair_rows=np.array([pair["rows"] for pair in pairs], dtype=np.int64),
        current_pair_coefficients=np.array([pair["coefficient"] for pair in pairs]),
        current_pair_t=np.array(t_values), current_pair_contributions=np.array(contributions))
    source.check()
    return result, raw, sum(q_upper, Fraction(0)), allowance


def observable_allowances(source, q_upper, stored_norm, assembly_norm, fixed_array):
    f_upper = sqrt_upper(q_upper)
    result = dict(observable_stored_norm_upper=upwards(stored_norm),
        observable_assembly_norm_upper=upwards(assembly_norm),
        computed_squared_norm_upper=upwards(q_upper), computed_norm_upper=upwards(f_upper),
        fixed_array_arithmetic_upper=upwards(fixed_array),
        endpoint_assembly_upper=upwards(assembly_norm*q_upper))
    totals = {}
    for kind, entry in source.source_error["totals"].items():
        epsilon = rational(entry["error_upper"])
        growth = 2*f_upper*epsilon+epsilon**2
        a = stored_norm+(assembly_norm if kind == "model_operator" else 0)
        allowance = a*growth
        total = fixed_array+allowance+(assembly_norm*q_upper if kind == "model_operator" else 0)
        result[kind] = dict(source_F_error_upper=upwards(epsilon),
            covariance_trace_norm_allowance_upper=upwards(growth), source_observable_error_upper=upwards(allowance),
            complete_endpoint_error_upper=upwards(total),
            composition=("fixed+(a_stored+eta_A)*(2 Fhat epsilon_model+epsilon_model^2)+eta_A Qhat"
                if kind == "model_operator" else "fixed+a_stored*(2 Fhat epsilon_stored+epsilon_stored^2)"))
        totals[kind] = total
    result["producer_seed_version_difference"] = "unavailable and excluded from every computational allowance"
    result["stored_and_model_totals"] = "separate alternatives; never added together"
    return result, totals


def cell_comparison(measured, target, precision, allowances):
    render = f".{precision}e"
    text = format(target, render)
    exponent = int(text.split("e")[1])
    half_step = Fraction(10)**(exponent-precision)/2
    lo, hi = Fraction(text)-half_step, Fraction(text)+half_step
    intervals = {}
    for role, error in allowances.items():
        lower, upper = rational(measured)-error, rational(measured)+error
        intervals[role] = dict(error_upper=upwards(error),
            interval=[downwards(lower), upwards(upper)],
            cell_status=("inside" if lo < lower and upper < hi else
                         "excludes" if upper <= lo or lower >= hi else "unresolved"))
    return dict(own=measured, external_target=target, full_difference=measured-target,
        declared_format="%"+render, own_declared=format(measured, render), external_declared=text,
        central_cells_match=format(measured, render) == text,
        exact_open_target_cell=[str(lo), str(hi)], computational_comparisons=intervals,
        physical_comparison="pending actual judgment; unknown producer-seed difference remains unavailable")


def run(args):
    started = time.monotonic()
    environment_check()
    # Acceptance pin is checked before any endpoint contraction/output begins.
    source = load_recovered(args.descriptor, args.build, args.comparison)
    root = Path(args.out).resolve()
    root.relative_to(Path("<internal>/second_route_bulk"))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("A recovered-origin control cannot overwrite or implicitly repeat output")
    resources(root)
    report_path, raw_path = root/"recovered_origin_checks.json", root/"recovered_origin_arrays.npz"
    report = dict(method=VERSION, completed=False, input=source.metadata,
        descriptor=source.descriptor, external_origin_targets=source.external_origin_targets,
        source_error=source.source_error, blocks=[], physical_rows_released=False,
        eigensolver_release="pending actual author and independent judgment",
        missing_held_comparators=dict(spectral_minimum=None, spectral_maximum=None,
            first_held_current=None, first_held_flow=None),
        arithmetic_assumptions="accepted C RN/binary64/gradual-underflow probes; no material underflow/overflow",
        scope="elapsed-zero recovered physical covariance control; no Gram/eigh/propagation/repair",
        release="withheld; complete origin numerical judgments and later held gates remain required")
    write_json(report_path, report)
    try:
        for name in ("hold_recovered_input.py", "hold_recovered_origin.py", "hold_source_seed_control.py"):
            source.inputs.add(Path(__file__).with_name(name))
        bands = source.method.bands(source.metadata["k"], scaled=False)
        rows, h_norm = operator_rows(bands)
        if h_norm > rational(source.operator["absolute_row_sum_upper"]):
            raise ArithmeticError("Actual endpoint row sum exceeds its accepted upper bound")
        report["operator"] = save_operator(source, root, bands)
        pairs, j_norm = exact_current_pairs(rows)
        (squares, real, imag, spatial_real, spatial_imag, spatial_binding,
            q_upper, energy_error) = energy_and_norm(source, bands, report, report_path, raw_path)
        current, current_raw, support_q, current_error = current_control(source, pairs, j_norm)
        report["current"] = current
        eta_h = rational(source.operator["assembly_error_upper"])
        h_bound = rational(source.operator["absolute_row_sum_upper"])
        eta_j = (2*rational(source.method.rho)+eta_h)*eta_h
        report["energy"]["allowances"], energy_totals = observable_allowances(source, q_upper, h_bound, eta_h, energy_error)
        report["current"]["allowances"], current_totals = observable_allowances(source, support_q, j_norm, eta_j, current_error)
        external = source.external_origin_targets["selected"]
        energy_target = external["energy_row"]["per_channel_3"]["E_hC"]
        current_target = external["native_current_record"]["per_channel_3_L_at_t"]
        report["external_comparisons"] = dict(
            E_hC=cell_comparison(report["energy"]["real"], energy_target, 6,
                dict(fixed_array=energy_error, **energy_totals)),
            origin_start_current=cell_comparison(current["real"], current_target, 4,
                dict(fixed_array=current_error, **current_totals)))
        # This explicitly named diagnostic cannot populate an independent E_inv.
        report["own_E_hC_minus_external_E_ref_diagnostic"] = (
            report["energy"]["real"]-external["energy_row"]["per_channel_3"]["E_ref"])
        report["own_E_ref"] = None
        report["own_E_inv"] = None
        coarse = abs(rational(report["energy"]["real"])-rational(energy_target)) <= Fraction(1, 10**7)
        report["coarse_energy_input_sanity_gate"] = dict(passed=bool(coarse), absolute_tolerance=1e-7,
            meaning="input sanity only; no printed-digit, physical, eigensolver or held-row acceptance")
        save_raw(raw_path, squares, real, imag, spatial_real, spatial_imag, spatial_binding, current_raw)
        source.check()
        report.update(completed=True, raw_scalar_arrays=identity(raw_path),
            verified_inputs=source.inputs.current, elapsed_seconds=time.monotonic()-started)
        write_json(report_path, report)
        log("recovered_origin_control_complete", endpoint=source.metadata["k"], coarse_input_gate=bool(coarse),
            E_hC=report["energy"]["real"], current=current["real"], eigensolver_release=report["eigensolver_release"])
        if not coarse:
            raise SystemExit(2)
    except BaseException as error:
        report.update(stop_reason=type(error).__name__+": "+str(error), elapsed_seconds=time.monotonic()-started)
        # A fully saved coarse-gate failure remains a completed diagnostic,
        # with an explicit failure and no eigensolver release.
        if not isinstance(error, SystemExit):
            report["completed"] = False
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--out", required=True)
    run(parser.parse_args())
