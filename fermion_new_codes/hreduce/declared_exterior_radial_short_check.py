#!/usr/bin/env python3
"""Bounded implementation of the reviewer B exterior radial-response specification.

Scientific specification: declared_exterior.tex, BOUNDED SHORT-CHECK
SPECIFICATION. This computes its continuum radial response, not a native
Cartesian spacing scan. Run only on the research box after paired review.
No state vectors, raw arrays or bulk logs are saved.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preset", choices=("main", "fine"), default="main")
    parser.add_argument("--baseline", type=Path,
                        help="original five-row result, required for fine preset")
    args = parser.parse_args()
    started = time.monotonic()
    result = {
        "scope": "specified continuum full-matrix radial detector response",
        "specification": "declared_exterior.tex: BOUNDED SHORT-CHECK SPECIFICATION",
        "scientific_acceptance": "pending original author and RED review",
        "preset": args.preset,
        "parameters": {"rs": 1, "kappa": -20, "Rstar": 1.5,
                       "pstar": 80 / 3, "Delta": 8, "T": 0.5,
                       "detector_width": 0.25, "core_radius": 0.5},
        "limits": {"wall_seconds": 180, "memory_bytes": 512 * 1024**2,
                   "cpu_threads": 1, "gpu": False},
        "rows": [], "checks": {}, "status": "running",
    }

    def save():
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    def expired(signum, frame):
        raise TimeoutError("180-second total wall budget exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    save()
    try:
        assert os.getpriority(os.PRIO_PROCESS, 0) == 10, "run with nice 10"
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(key) == "1", f"{key} must be 1"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be disabled"
        import numpy as np
        from scipy import sparse
        from scipy.special import betainc, spherical_jn
        from scipy.sparse.linalg import expm_multiply

        def real(value):
            return float(np.real(value))

        def hermiticity(matrix):
            difference = matrix - matrix.getH()
            numerator = float(np.max(np.abs(difference.data), initial=0))
            denominator = max(1.0, float(np.max(np.abs(matrix.data), initial=0)))
            return numerator / denominator

        def one_grid(denominator, rmax, quadrature):
            row_started = time.monotonic()
            dr = 1 / denominator
            r = np.arange(1, round(rmax / dr)) * dr
            n = r.size
            kappa = -20
            identity = sparse.eye(n, format="csr")
            derivative = sparse.diags(
                [np.ones(n - 2), -8 * np.ones(n - 1),
                 8 * np.ones(n - 1), -np.ones(n - 2)],
                [-2, -1, 1, 2], shape=(n, n), format="csr") / (12 * dr)
            momentum = -1j * derivative
            sx = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
            sy = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
            spin_identity = sparse.eye(2, format="csr")
            dmat = (sparse.kron(sy, momentum, format="csr")
                    + sparse.kron(sx, sparse.diags(kappa / r), format="csr"))
            fraction = betainc(1.5, 12, np.minimum((r / 0.5)**2, 1))
            v = -np.sqrt(fraction / r)
            vmat = sparse.diags(v, format="csr")
            hc = (dmat + sparse.kron(spin_identity,
                  (vmat @ momentum + momentum @ vmat) / 2, format="csr"))
            laplacian = dmat @ dmat
            amat = sparse.kron(spin_identity,
                               vmat @ momentum + 1j * sparse.diags(v / r),
                               format="csr")
            h2 = (-dmat @ laplacian / 10
                  - (amat @ laplacian + laplacian @ amat.getH()) / 5).tocsr()
            errors = {"hc": hermiticity(hc), "h2": hermiticity(h2)}
            assert max(errors.values()) <= 1e-12, errors

            z, wz = np.polynomial.legendre.leggauss(quadrature)
            delta, pstar, radius, kval = 8.0, 80 / 3, 1.5, 20
            p = pstar + delta * z
            wp = delta * wz
            envelope = np.exp(-1 / (1 - z*z))
            constant = 1 / np.sqrt(np.dot(wp, envelope**2))
            action = np.sqrt((p * radius)**2 - kval**2) - kval * np.arccos(
                kval / (p * radius))
            weighted = wp * constant * envelope * np.exp(-1j * action) / np.sqrt(np.pi)
            multiplier = 2 * z / (delta * (1 - z*z)**2)
            argument = p[:, None] * r[None, :]
            upper = argument * spherical_jn(kval - 1, argument)
            lower = -argument * spherical_jn(kval, argument)
            u = np.concatenate((weighted @ upper, weighted @ lower)) * np.sqrt(dr)
            up = np.concatenate(((weighted * multiplier) @ upper,
                                 (weighted * multiplier) @ lower)) * np.sqrt(dr)
            del argument, upper, lower
            raw_norm = float(np.linalg.norm(u))
            u /= raw_norm
            up /= raw_norm
            up -= u * real(np.vdot(u, up))
            omega0 = real(np.vdot(u, hc @ u))
            eprime = 2 * real(np.vdot(up, hc @ u))
            e2 = real(np.vdot(u, h2 @ u))
            assert abs(eprime) >= 1e-3, "fixed-frequency derivative below threshold"
            dpstar = -e2 / eprime
            u2 = dpstar * up
            scale = dr**2 / (1 + kappa**2)
            generator = -1j * sparse.bmat(
                [[hc, None], [scale * h2, hc]], format="csr")
            times = np.linspace(0, 0.5, 5)
            propagated = expm_multiply(generator, np.concatenate((u, scale * u2)),
                                       start=0, stop=0.5, num=5, endpoint=True,
                                       traceA=0.0)
            weights = np.tile(np.exp(-(r - radius)**2 / (2 * 0.25**2)), 2)
            radial = np.tile(r, 2)
            samples = []
            for t, state in zip(times, propagated):
                ut, response = state[:2*n], state[2*n:] / scale
                hc_response, h2_u = hc @ response, h2 @ ut
                norm_error = abs(real(np.vdot(ut, ut)) - 1)
                norm_response = 2 * real(np.vdot(ut, response))
                energy_response = (2 * real(np.vdot(ut, hc_response))
                                   + real(np.vdot(ut, h2_u)))
                norm_magnitude = max(1.0, 2 * float(np.sum(np.abs(ut.conj() * response))))
                energy_magnitude = max(1.0,
                    2 * float(np.sum(np.abs(ut.conj() * hc_response)))
                    + float(np.sum(np.abs(ut.conj() * h2_u))))
                probability = np.abs(ut.reshape(2, n))**2
                samples.append({
                    "t": float(t), "norm_error": norm_error,
                    "norm_response": norm_response,
                    "norm_response_natural_magnitude": norm_magnitude,
                    "norm_response_relative": abs(norm_response) / norm_magnitude,
                    "energy_response": energy_response,
                    "energy_response_natural_magnitude": energy_magnitude,
                    "energy_response_relative": abs(energy_response) / energy_magnitude,
                    "mean_r": real(np.vdot(ut, radial * ut)),
                    "P0F": real(np.vdot(ut, weights * ut)),
                    "c_F": 2 * real(np.vdot(ut, weights * response)),
                    "c_r": 2 * real(np.vdot(ut, radial * response)),
                    "probability_r_lt_0_25": float(probability[:, r < 0.25].sum()),
                    "probability_r_lt_0_5": float(probability[:, r < 0.5].sum()),
                    "probability_outer_unit": float(probability[:, r > rmax - 1].sum()),
                })
                if args.preset == "fine":
                    radial_response = (2 * np.real(ut.conj() * radial * response)).reshape(2, n)
                    parts = [float(radial_response[:, mask].sum()) for mask in
                             (r < 0.25, (r >= 0.25) & (r < 0.5), r >= 0.5)]
                    samples[-1]["c_r_partitions"] = dict(zip(
                        ("r_lt_0_25", "r_0_25_to_0_5", "r_ge_0_5"), parts))
                    samples[-1]["c_r_partition_sum_error"] = abs(
                        sum(parts) - samples[-1]["c_r"])
                    samples[-1]["c_r_partition_check_passed"] = (
                        samples[-1]["c_r_partition_sum_error"]
                        < 1e-10 * max(1.0, sum(abs(x) for x in parts)))
            row = {"dr": dr, "rmax": rmax, "sites": int(n),
                   "quadrature": quadrature, "hermiticity_relative": errors,
                   "raw_preparation_norm": raw_norm, "omega0": omega0,
                   "E0prime": eprime, "E2": e2, "delta_pstar": dpstar,
                   "samples": samples, "elapsed_seconds": time.monotonic() - row_started}
            row["propagation_checks_passed"] = all(
                q["norm_error"] < 1e-9 and q["norm_response_relative"] < 1e-8
                and q["energy_response_relative"] < 1e-8 for q in samples)
            result["rows"].append(row)
            save()
            return row, u

        if args.preset == "fine":
            assert args.baseline is not None, "fine preset requires the preserved baseline result"
            baseline = json.loads(args.baseline.read_text())
            assert baseline["parameters"] == result["parameters"], "baseline preparation changed"
            baseline_rows = [row for row in baseline["rows"]
                             if row["dr"] == 1 / 256 and row["rmax"] == 8
                             and row["quadrature"] == 512]
            assert len(baseline_rows) == 1
            result["baseline_result"] = str(args.baseline)
            fine_rows = [baseline_rows[0]]
            for denominator in (512, 1024):
                row, _ = one_grid(denominator, 8, 512)
                fine_rows.append(row)
            checks = result["checks"]
            checks["all_propagation_passed"] = all(
                row["propagation_checks_passed"] for row in fine_rows)
            checks["all_partition_sums_passed"] = all(
                sample["c_r_partition_check_passed"]
                for row in result["rows"] for sample in row["samples"])
            for coefficient in ("c_F", "c_r"):
                values = [row["samples"][-1][coefficient] for row in fine_rows]
                coarse = abs(values[1] - values[0])
                fine = abs(values[2] - values[1])
                scale = max(1.0, abs(values[-1]))
                ratio = coarse / fine if fine else None
                order_supported = ratio is not None and 8 <= ratio <= 24
                both_negligible = coarse < 1e-6 * scale and fine < 1e-6 * scale
                checks[coefficient] = {
                    "values_dr_256_512_1024": values,
                    "coarse_difference": coarse, "fine_difference": fine,
                    "coarse_to_fine_difference_ratio": ratio,
                    "fourth_order_ratio_supported": order_supported,
                    "both_differences_negligible": both_negligible,
                    "estimated_dr4_error": fine / 15 if order_supported else None,
                    "dr4_error_is_asymptotic_estimate_only": True,
                    "refinement_passed": fine < 3e-3 * scale
                        and (order_supported or both_negligible)
                        and (fine / 15 < 1e-3 * scale if order_supported else both_negligible),
                }
            checks["E0prime_values_dr_256_512_1024"] = [row["E0prime"] for row in fine_rows]
            checks["E0prime_positive"] = all(row["E0prime"] >= 1e-3 for row in fine_rows)
            checks["E0prime_fine_difference"] = abs(
                fine_rows[-1]["E0prime"] - fine_rows[-2]["E0prime"])
            checks["E0prime_magnitude_passed"] = (
                checks["E0prime_fine_difference"] < 1e-3 * abs(fine_rows[-1]["E0prime"]))
            result["status"] = "completed; numerical and scientific acceptance require paired review"
            return 0

        main_rows = []
        middle_preparation = None
        for denominator in (64, 128, 256):
            row, preparation = one_grid(denominator, 8, 512)
            main_rows.append(row)
            if denominator == 128:
                middle_preparation = preparation
        quadrature_row, quadrature_preparation = one_grid(128, 8, 1024)
        domain_row, _ = one_grid(128, 12, 512)
        checks = result["checks"]
        checks["preparation_quadrature_norm_difference"] = float(
            np.linalg.norm(quadrature_preparation - middle_preparation))
        checks["preparation_quadrature_passed"] = (
            checks["preparation_quadrature_norm_difference"] < 1e-8)
        checks["all_propagation_passed"] = all(
            row["propagation_checks_passed"] for row in result["rows"])
        for coefficient in ("c_F", "c_r"):
            values = [row["samples"][-1][coefficient] for row in main_rows]
            coarse, fine = abs(values[1] - values[0]), abs(values[2] - values[1])
            scale = max(1.0, abs(values[2]))
            quad_difference = abs(quadrature_row["samples"][-1][coefficient] - values[1])
            domain_difference = abs(domain_row["samples"][-1][coefficient] - values[1])
            checks[coefficient] = {
                "main_values": values, "coarse_difference": coarse,
                "fine_difference": fine, "estimated_dr4_error": fine / 15,
                "coarse_to_fine_difference_ratio": coarse / fine if fine else None,
                "dr4_error_is_asymptotic_estimate_only": True,
                "refinement_passed": fine < 3e-3 * scale and fine < coarse
                    and fine / 15 < 1e-3 * scale,
                "quadrature_difference": quad_difference,
                "quadrature_passed": quad_difference < 1e-4 * max(1, abs(values[1])),
                "domain_difference": domain_difference,
                "domain_passed": domain_difference < 1e-3 * max(1, abs(values[1])),
            }
        checks["E0prime_main_values"] = [row["E0prime"] for row in main_rows]
        checks["E0prime_same_sign"] = bool(all(
            row["E0prime"] * main_rows[-1]["E0prime"] > 0 for row in main_rows))
        checks["E0prime_fine_difference"] = abs(
            main_rows[-1]["E0prime"] - main_rows[-2]["E0prime"])
        checks["E0prime_magnitude_passed"] = (
            checks["E0prime_fine_difference"] < 1e-3 * abs(main_rows[-1]["E0prime"]))
        result["status"] = "completed; numerical and scientific acceptance require paired review"
    except (Exception, MemoryError) as exc:
        result["status"] = "unaccepted partial result"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        save()
    return 0 if result["status"].startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
