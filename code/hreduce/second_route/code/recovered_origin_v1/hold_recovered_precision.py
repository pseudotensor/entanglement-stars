"""Scalar assessment of the once-only saved recovered HOLD products, §18.22.

No eigensolver, Gram product, sparse source action or matrix transform here.
"""
from fractions import Fraction
import numpy as np

from hold_error_bounds import (exact_saved_eigensum, gamma_rational, inflated_gram,
    inflated_residuals, rational, sqrt_upper, upwards)
from hold_recovered_origin import cell_comparison, rational_record


def exact_dot(left, right):
    exact = sum((rational(float(a))*rational(float(b)) for a, b in zip(left, right, strict=True)), Fraction(0))
    central = float(exact)
    return dict(value=central, exact=rational_record(exact),
        conversion_error_upper=upwards(abs(rational(central)-exact)),
        scope="exact products/sum of saved binary scalar inputs; construction errors remain separate")


def compare(own, target, bounds):
    known = {key: value for key, value in bounds.items() if value is not None}
    result = cell_comparison(own, target, 6, known)
    for key, value in bounds.items():
        if value is None:
            result["computational_comparisons"][key] = dict(cell_status="unresolved",
                error_upper=None, interval=None, reason="required inertia/arithmetic certificate unresolved")
    return result


def assess(stages, binding, operator):
    source, origin, root = stages.source, stages.origin, stages.root
    with np.load(stages.output("spectrum", "spectrum"), allow_pickle=False) as raw:
        e = raw["eigenvalues"]
    with np.load(stages.output("deviation", "diagonals"), allow_pickle=False) as raw:
        if not np.array_equal(raw["eigenvalues"], e):
            raise ValueError("Saved transform diagonals have another spectral ordering")
        pre, post = raw["G_hat_diagonal"], raw["Dhat_diagonal"]
    if (pre.shape != (16384,) or post.shape != (16384,) or pre.dtype != np.complex128
            or post.dtype != np.complex128 or not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post))):
        raise ValueError("The complete pre/post transform diagonals are unavailable/nonfinite")
    residual = stages.report["stages"]["residual"]["values"]
    phi_gram = stages.report["stages"]["source_gram"]["values"]
    vector_gram = stages.report["stages"]["vector_gram"]["values"]
    negative = e < 0
    n_minus = int(np.count_nonzero(negative))
    gap = rational(float(np.min(np.abs(e))))
    eta_h = rational(source.operator["assembly_error_upper"])
    precision = dict(method="recovered-k3-saved-product-precision-1", source_kind=source.metadata["source_kind"],
        input=source.metadata, input_binding=binding, physical_rows_released=False,
        arithmetic_model="reviewed complex BLAS gamma_(8n+16), binary64 RN, no material underflow/overflow; exact/outward scalar bounds",
        origin_sparse_energy_reused=origin["energy"], origin_current_reused=origin["current"],
        origin_control=stages.binding["origin_control"],
        producer_seed_version_difference="unavailable; excluded from every computational allowance",
        release="withheld; actual judgments and record-specific missing held comparisons remain")
    eta_phi = inflated_gram(16384, 8192, phi_gram["state_gram_F"])
    spectral = inflated_residuals(16384, operator["row_nonzeros"], operator["h_absolute_row_sum_upper"],
        float(np.max(np.abs(e))), residual["eigen_residual_F"], vector_gram["eigenvector_gram_F"])
    delta = rational(spectral["gram_F_upper"])
    spectral_error = rational(spectral["polar_spectral_error_upper"]) if "polar_spectral_error_upper" in spectral else None
    inertia = spectral_error is not None and delta < 1 and gap > spectral_error
    model_inertia = bool(inertia and gap > spectral_error+eta_h)
    spectral.update(inertia_resolved=bool(inertia), model_endpoint_inertia_resolved=model_inertia,
        minimum_saved_abs_eigenvalue=float(gap), source_gram_dot_length=16384, source_gram_width=8192,
        state_gram_F_upper=upwards(eta_phi), source_covariance_operator_upper=upwards(1+eta_phi),
        source_covariance_F_upper=upwards(sqrt_upper(8192)*(1+eta_phi)),
        source_covariance_purity_F_lower=0., source_covariance_purity_F_upper=upwards((1+eta_phi)*eta_phi),
        purity_scope="actual nonorthogonal recovered array; source reconstruction and producer differences are separate")
    if inertia:
        spectral.update(reference_projector_operator_error_upper=upwards(spectral_error/(2*gap-spectral_error)),
            reference_projector_F_error_upper=upwards(sqrt_upper(2)*spectral_error/(2*gap-spectral_error)),
            reference_projector_scope="true stored-h projector versus polar-unitary saved-eigenvalue projector; raw-V Gram term separate")
    precision["spectral_inflation"] = spectral
    precision["source_covariance_errors"] = {}
    for kind, entry in source.source_error["totals"].items():
        epsilon = rational(entry["error_upper"])
        precision["source_covariance_errors"][kind] = dict(source_F_error_upper=upwards(epsilon),
            covariance_F_error_upper=upwards(2*sqrt_upper(1+eta_phi)*epsilon+epsilon**2),
            scope="same exact held unitary only; different held operators/projectors/phases require separate allowances")
    reference, sum_error = exact_saved_eigensum(e[negative])
    anchor = origin["energy"]["real"]
    exact_inv = rational(anchor)-rational(reference)
    invariant = float(exact_inv)
    subtraction = abs(rational(invariant)-exact_inv)
    energy_error = rational(origin["energy"]["fixed_array_error_upper_per_component"])
    reference_spectral = sqrt_upper(n_minus)*spectral_error if inertia else None
    fixed_inv = energy_error+rational(sum_error)+reference_spectral+subtraction if inertia else None
    fixed_reference = rational(sum_error)+reference_spectral if inertia else None
    energy_allowances = origin["energy"]["allowances"]
    stored_source = rational(energy_allowances["stored_operator"]["source_observable_error_upper"])
    model_source = rational(energy_allowances["model_operator"]["source_observable_error_upper"])
    final_assembly = rational(energy_allowances["endpoint_assembly_upper"])
    model_reference = n_minus*eta_h if model_inertia else None
    inv_bounds = dict(fixed_saved_source_stored_h=fixed_inv,
        stored_history=(fixed_inv+stored_source if inertia else None),
        model_history_endpoint=(fixed_inv+model_source+final_assembly+model_reference if model_inertia else None))
    ref_bounds = dict(saved_binary_sum_only=rational(sum_error), fixed_stored_h=fixed_reference,
        model_endpoint=(fixed_reference+model_reference if model_inertia else None))
    target = origin["external_origin_targets"]["selected"]["energy_row"]["per_channel_3"]
    energies = dict(own_E_hC=anchor, own_E_ref=reference, own_E_inv=invariant,
        negative_count=n_minus, external_negative_count=target["n_neg"], external_n0=target["n0"],
        external_negative_count_matches=n_minus == target["n_neg"],
        E_ref_allocation_certificate="resolved for stored h" if inertia else "unresolved strict-negative diagnostic",
        reference_binary_sum_conversion_error_upper=sum_error,
        reference_spectral_error_upper=upwards(reference_spectral) if inertia else None,
        invariant_subtraction_conversion_error_upper=upwards(subtraction),
        invariant_fixed_error_upper=upwards(fixed_inv) if inertia else None,
        stored_history_source_energy_upper=upwards(stored_source),
        model_history_source_energy_upper=upwards(model_source),
        endpoint_assembly_energy_upper=upwards(final_assembly),
        model_reference_assembly_upper=upwards(model_reference) if model_inertia else None,
        E_ref_comparison=compare(reference, target["E_ref"], ref_bounds),
        E_inv_comparison=compare(invariant, target["E_inv"], inv_bounds),
        coordinate_gross_energy_real=exact_dot(e, pre.real),
        coordinate_gross_energy_imaginary=exact_dot(e, pre.imag),
        deviation_energy_real=exact_dot(e, post.real),
        deviation_energy_imaginary=exact_dot(e, post.imag),
        independent_reference_scope="one own held spectrum; shared negative-eigenvalue sums are not independent reference estimates")
    energies["coordinate_energy_minus_sparse_anchor"] = energies["coordinate_gross_energy_real"]["value"]-anchor
    energies["deviation_energy_minus_sparse_anchor_invariant"] = energies["deviation_energy_real"]["value"]-invariant
    precision["initial_energies"] = energies
    diagonal_squared = Fraction(0)
    for index in range(16384):
        real_error = rational(float(post[index].real))-rational(float(pre[index].real))+int(negative[index])
        imag_error = rational(float(post[index].imag))-rational(float(pre[index].imag))
        diagonal_squared += real_error**2+imag_error**2
    beta_p = sqrt_upper(diagonal_squared)
    # These are scalar forecasts on the already saved products, not new GEMMs.
    f_upper = rational(origin["norm"]["upper"])
    g_d, g_r = gamma_rational(8*16384+16), gamma_rational(8*8192+16)
    beta_b = g_d*sqrt_upper(16384*(1+delta))*f_upper
    m_b = sqrt_upper(1+delta)*f_upper+beta_b
    beta_g = g_r*m_b**2
    beta_d = 2*m_b*beta_b+beta_b**2+beta_g+beta_p
    covariance_f = sqrt_upper(8192)*(1+eta_phi)
    reconstructed = (2*delta+delta**2)*covariance_f+(1+delta)*beta_d
    precision["saved_transform_allowance"] = dict(status="available under declared BLAS arithmetic model",
        dimension=16384, occupied_width=8192, gamma_d_upper=upwards(g_d), gamma_r_upper=upwards(g_r),
        endpoint_source_F_upper=upwards(f_upper), true_V_gram_upper=upwards(delta),
        beta_B_upper=upwards(beta_b), computed_B_F_upper=upwards(m_b), beta_G_upper=upwards(beta_g),
        diagonal_subtraction_error_squared_exact=rational_record(diagonal_squared), beta_P_upper=upwards(beta_p),
        beta_D_upper=upwards(beta_d), source_covariance_F_upper=upwards(covariance_f),
        beta_reconstruction_upper=upwards(reconstructed),
        beta_D_target="||Dhat_saved-(V^dagger C_hat V-P)||F, P=diag(1[epsilon<0])",
        reconstruction_target="||V Dhat V^dagger-(C_hat-V P V^dagger)||F at elapsed zero",
        meaning="global arithmetic model, not measured BLAS error; true-h projector and later propagation/phase/readout terms separate")
    checks = dict(method="recovered-k3-saved-spectral-checks-1", source_kind=source.metadata["source_kind"],
        input=source.metadata, input_binding=binding,
        eigen_residual_F=residual["eigen_residual_F"],
        eigen_residual_column_max=residual["eigen_residual_column_max"],
        state_gram_F=phi_gram["state_gram_F"], state_gram_entry_max=phi_gram["state_gram_entry_max"],
        eigenvector_gram_F=vector_gram["eigenvector_gram_F"],
        eigenvector_gram_entry_max=vector_gram["eigenvector_gram_entry_max"],
        state_gram_antihermitian=phi_gram["antihermitian"],
        deviation_antihermitian=stages.report["stages"]["deviation_control"]["values"],
        eigenvalue_min=float(e[0]), eigenvalue_max=float(e[-1]), negative_levels=n_minus,
        paired_eigenvalue_error_max=residual["paired_eigenvalue_error_max"],
        zero_rule="strict epsilon<0 without clipping or repair", guard_weight=48,
        guard_weight_scope="separate reporting weight; never applied to h/Phi/Dhat or per-channel controls",
        origin_current_reference=stages.binding["origin_control"],
        held_extrema_comparison=dict(status="unavailable", minimum=None, maximum=None),
        first_held_current_comparison=dict(status="unavailable", value=None),
        first_held_flow_comparison=dict(status="unavailable", value=None),
        physical_rows_released=False, release="withheld; missing matching held operator and first-interval gates")
    initial = dict(input=source.metadata, source_kind=source.metadata["source_kind"],
        own_measured=dict(E_hC=anchor, E_ref=reference, E_inv=invariant,
            origin_C_current=origin["current"]["real"], trace=origin["norm"]["trace_measured"]),
        external_origin_targets=origin["external_origin_targets"], initial_energy_controls=energies,
        input_binding=binding, source_error=source.source_error,
        release="withheld; actual spectral judgment and unavailable matching held gates")
    outputs = {}
    for key, name, value in (("precision", "precision_checks.json", precision),
            ("checks", "spectral_checks.json", checks), ("initial", "initial_gate.json", initial)):
        path = root/name
        outputs[key] = stages.save_product_json(path, value)
    return dict(stored_h_inertia_resolved=bool(inertia), model_endpoint_inertia_resolved=model_inertia,
        own_E_ref=reference, own_E_inv=invariant, physical_rows_released=False), outputs
