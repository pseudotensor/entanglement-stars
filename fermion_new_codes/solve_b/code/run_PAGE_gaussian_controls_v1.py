"""Known answers and a factored identity on one actual saved heavy state.

Implements section 2 of solve/PAGE_secondroute_spec_v1.md.
"""

import argparse
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analytic-only",action="store_true")
    args = parser.parse_args()
    runtime = require_remote_cpu(args.output)
    import numpy as np

    def norm(matrix):
        return {"frobenius": float(np.linalg.norm(matrix)),
                "operator": float(np.linalg.svd(matrix, compute_uv=False).max(initial=0))}

    def entropy(values):
        if np.min(values) < -1e-12 or np.max(values) > 1+1e-12:
            raise ValueError("Invalid known-answer covariance spectrum")
        values = np.clip(values, 0, 1)
        x = values[(values > 0) & (values < 1)]
        return float(np.sum(-x*np.log(x)-(1-x)*np.log1p(-x)))

    column = np.array([[1], [1]], dtype=float)/np.sqrt(2)
    projector = np.full((2, 2), 0.5)
    controls = []
    for occupation in (1.0, 0.37, 0.0):
        c = occupation*(column@column.T)
        a = c[1:, 1:]
        k = c[1:, :1]
        impurity = (c-c@c)[1:, 1:]
        expected = occupation/2-occupation**2/4
        actual = a-a@a
        controls.append({"occupation": occupation,
                         "identity_residual": norm(actual-k@k.T-impurity),
                         "known_scalar_error": float(abs(actual[0, 0]-expected)),
                         "entropy": entropy(np.linalg.eigvalsh(a)),
                         "expected_entropy": entropy(np.array([occupation/2]))})
    state_entropy = entropy(np.linalg.eigvalsh((column@column.T)[1:, 1:]))
    vacuum_entropy = entropy(np.linalg.eigvalsh(projector[1:, 1:]))
    known = {"two_mode_rows": controls, "vacuum_excess": state_entropy-vacuum_entropy,
             "vacuum_log2_error": abs(vacuum_entropy-np.log(2))}
    product = np.diag([0.37,0.61])
    product_a,product_k = product[1:,1:],product[1:,:1]
    product_impurity = (product-product@product)[1:,1:]
    known["mixed_product_control"] = {
        "occupations":[0.37,0.61],"cross_block_frobenius":float(np.linalg.norm(product_k)),
        "local_entropy":entropy(np.linalg.eigvalsh(product_a)),
        "expected_entropy":entropy(np.array([0.61])),
        "identity_residual":norm(product_a-product_a@product_a-product_k@product_k.T-product_impurity),
        "physical_impurity":norm(product_impurity)}
    (args.output/"analytic_controls.json").write_text(json.dumps(known, indent=2)+"\n")
    if (abs(known["vacuum_excess"]) > 1e-12 or any(
            row["identity_residual"]["operator"] > 1e-12 for row in controls)
            or known["mixed_product_control"]["cross_block_frobenius"] != 0
            or known["mixed_product_control"]["local_entropy"] <= 0
            or known["mixed_product_control"]["identity_residual"]["operator"] > 1e-12):
        raise RuntimeError("Known-answer controls failed")
    if args.analytic_only:
        print(json.dumps({"status":"saved","known_answer_pass":True,
                          "output":str(args.output/"analytic_controls.json")}))
        return

    with np.load(args.state, allow_pickle=False) as data:
        sea = data["sea_1"]
        dust = data["dust_1"]
        occupations = data["f_1"]
        radii = data["R"]
        rs = float(data["rs"])
        clock = float(data["T"])/rs
    if not all(np.isfinite(x).all() for x in (sea, dust, occupations, radii)):
        raise ValueError("Nonfinite stored state")
    if np.any(occupations < 0) or np.any(occupations > 1):
        raise ValueError("Invalid stored dust occupation")
    if sea.shape[0] != 2*len(radii) or dust.shape[0] != sea.shape[0]:
        raise ValueError("Stored component/site layout mismatch")
    gram_sea = sea.conj().T@sea
    gram_dust = dust.conj().T@dust
    cross = sea.conj().T@dust
    orthogonality = {
        "sea_gram": norm(gram_sea-np.eye(sea.shape[1])),
        "dust_gram": norm(gram_dust-np.eye(dust.shape[1])),
        "sea_dust_frobenius": float(np.linalg.norm(cross)),
        "sea_dust_operator": float(np.linalg.svd(cross, compute_uv=False).max(initial=0))}
    z = np.column_stack((sea, dust*np.sqrt(occupations)[None, :]))
    inside_sites = radii < 2*rs
    inside = np.repeat(inside_sites, 2)
    zx, zy = z[inside], z[~inside]
    a = zx@zx.conj().T
    np.save(args.output/"stored_A.npy", a)
    k = zx@zy.conj().T
    np.save(args.output/"stored_K.npy", k)
    kk = k@k.conj().T
    row = zx@z.conj().T
    impurity = a-row@row.conj().T
    wx = dust[inside]
    physical = (wx*(occupations*(1-occupations))[None, :])@wx.conj().T
    lhs = a-a@a
    residual = lhs-kk-impurity
    np.savez_compressed(args.output/"stored_identity.npz", A=a, KK=kk,
                        impurity=impurity, physical_impurity=physical,
                        residual=residual, occupations=occupations,
                        selected_sites=radii[inside_sites], gram_sea=gram_sea,
                        gram_dust=gram_dust, gram_cross=cross)
    scales = {"lhs": norm(lhs), "KK": norm(kk), "impurity": norm(impurity),
              "physical_impurity": norm(physical)}
    residual_norm = norm(residual)
    report = {"scope": "Algebraic identity on the actual saved heavy covariance; "
                       "not a contraction of a light production state.",
              "runtime": runtime, "source": str(args.state.resolve()),
              "channel": 1, "clock_rs": clock, "cut_rs": 2,
              "cut_convention": "X consists of stored R < 2 r_s; both components",
              "selected_site_count": int(inside_sites.sum()),
              "occupation_min": float(occupations.min()),
              "occupation_max": float(occupations.max()),
              "orthogonality": orthogonality, "term_scales": scales,
              "identity_residual": residual_norm,
              "relative_operator_residual": residual_norm["operator"]/max(
                  1.0, *(v["operator"] for v in scales.values())),
              "general_minus_physical_impurity": norm(impurity-physical),
              "entropy_interpretation": "requires separate covariance physicality control"}
    (args.output/"stored_identity.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "saved", "known_answer_pass": True,
                      "output": str(args.output/"stored_identity.json")}))


if __name__ == "__main__":
    main()
