"""Remote controls specified by partner_admissibility_code_spec.tex.

Run with the qi2 Python on compute-host, nice 10, and one CPU thread.
These are matrix/scattering identity controls, not an astrophysical model.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy import linalg


SIGMA = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def fro(matrix):
    return float(linalg.norm(matrix, "fro"))


def analytic_probability(bi, bf, d, tau):
    bi, bf, d, tau = map(lambda x: mp.mpf(str(x)), (bi, bf, d, tau))
    if d == 0 or bi == bf:
        return mp.mpf(0)
    wi, wf = mp.sqrt(d * d + bi * bi), mp.sqrt(d * d + bf * bf)
    change, difference = abs(bf - bi), abs(wf - wi)
    return (
        mp.sinh(mp.pi * tau * (change + difference) / 2)
        * mp.sinh(mp.pi * tau * (change - difference) / 2)
        / (mp.sinh(mp.pi * tau * wi) * mp.sinh(mp.pi * tau * wf))
    )


def probability_bound(z):
    decay = mp.exp(-2 * mp.pi * mp.mpf(str(z)))
    return min(mp.mpf(1), decay / (1 - decay) ** 2)


def integrate_tanh(bi, bf, d, tau, endpoint, rtol=1e-12, atol=1e-13):
    _, vi = linalg.eigh(d * SIGMA[0] + bi * SIGMA[2])
    _, vf = linalg.eigh(d * SIGMA[0] + bf * SIGMA[2])

    def rhs(u, chi):
        b = bi + 0.5 * (bf - bi) * (1 + np.tanh(u))
        return -1j * tau * (d * SIGMA[0] + b * SIGMA[2]) @ chi

    solution = solve_ivp(
        rhs, (-endpoint, endpoint), vi[:, 0].astype(complex),
        method="DOP853", rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    norms = np.sum(np.abs(solution.y) ** 2, axis=0)
    probability = float(abs(np.vdot(vf[:, 1], solution.y[:, -1])) ** 2)
    return {
        "endpoint": endpoint, "rtol": rtol, "atol": atol,
        "probability": probability,
        "maximum_norm_defect": float(np.max(np.abs(norms - 1))),
        "accepted_steps": len(solution.t) - 1,
        "rhs_evaluations": solution.nfev,
        "tail_probability_bound": float(2 * tau * abs(bf - bi) * np.exp(-2 * endpoint)),
    }


def control_a():
    cases = []
    for bf in (0.5, 2.0):
        bmin = min(1.0, bf)
        for ratio in (0, 0.25, 1, 4, 10):
            for z in (0.05, 0.5, 2.0):
                cases.append(("grid", 1.0, bf, ratio * bmin, z / bmin))
    for tau in (0.05, 2.0):
        cases.append(("no-change", 1.0, 1.0, 1.0, tau))
    for tau in (0.001, 0.0005):
        cases.append(("sudden", 1.0, 2.0, 1.0, tau))

    results = []
    maximum_violation = mp.mpf(0)
    for label, bi, bf, d, tau in cases:
        z = tau * min(bi, bf)
        exact = analytic_probability(bi, bf, d, tau)
        bound = probability_bound(z)
        maximum_violation = max(maximum_violation, -exact, exact - bound)
        runs = [integrate_tanh(bi, bf, d, tau, t) for t in (14, 18)]
        for run in runs:
            run["absolute_analytic_discrepancy"] = abs(run["probability"] - float(exact))
        row = {
            "kind": label, "bi": bi, "bf": bf, "d": d, "tau": tau, "z": z,
            "analytic_probability_80_digit": mp.nstr(exact, 80),
            "analytic_probability": float(exact),
            "bound_80_digit": mp.nstr(bound, 80), "bound": float(bound),
            "runs": runs,
            "endpoint_refinement": abs(runs[1]["probability"] - runs[0]["probability"]),
        }
        if label == "sudden":
            wi, wf = np.hypot(d, bi), np.hypot(d, bf)
            sudden = float(0.5 * (1 - (d * d + bi * bf) / (wi * wf)))
            row["ideal_sudden_probability"] = sudden
            row["analytic_minus_sudden"] = float(exact) - sudden
            row["direct_minus_sudden"] = runs[1]["probability"] - sudden
        results.append(row)

    worst = max(results, key=lambda row: row["runs"][1]["absolute_analytic_discrepancy"])
    tighter = integrate_tanh(worst["bi"], worst["bf"], worst["d"], worst["tau"], 18, 2e-13, 2e-14)
    tighter["absolute_analytic_discrepancy"] = abs(tighter["probability"] - worst["analytic_probability"])
    tighter["change_from_original"] = abs(tighter["probability"] - worst["runs"][1]["probability"])
    worst["tighter_tolerance_run"] = tighter
    max_error = max(row["runs"][1]["absolute_analytic_discrepancy"] for row in results)
    return {
        "cases": results, "arithmetic_digits": mp.mp.dps,
        "largest_absolute_ode_discrepancy_T18": max_error,
        "largest_positive_bound_violation_80_digit": mp.nstr(maximum_violation, 80),
        "bound_rounding_allowance": "1e-70",
        "specified_formula_criterion": bool(max_error < 1e-9 and maximum_violation <= mp.mpf("1e-70")),
        "worst_case_parameters": {key: worst[key] for key in ("bi", "bf", "d", "tau")},
    }


def sine_lattice(length):
    coordinates = np.indices((length, length, length)).reshape(3, -1).T
    nsites = length ** 3
    derivative = np.zeros((2 * nsites, 2 * nsites), dtype=complex)
    for axis in range(3):
        translation = np.zeros((nsites, nsites), dtype=complex)
        for source, coordinate in enumerate(coordinates):
            destination = coordinate.copy()
            destination[axis] = (destination[axis] + 1) % length
            target = np.ravel_multi_index(tuple(destination), (length,) * 3)
            translation[target, source] = 1
        derivative += np.kron((translation - translation.conj().T) / (2j), SIGMA[axis])
    return coordinates, derivative


def polar_law(mass, psi, lapse, derivative, orientation):
    a = np.diag(mass / psi) + 1j * np.diag(psi ** -3) @ derivative
    if orientation == -1:
        a = a.conj().T
    square = a.conj().T @ a
    eigenvalues, vectors = linalg.eigh((square + square.conj().T) / 2)
    if eigenvalues[0] <= 0:
        raise RuntimeError("Positive spectral square root lost in numerical control")
    energy = (vectors * np.sqrt(eigenvalues)) @ vectors.conj().T
    root = (vectors * eigenvalues ** 0.25) @ vectors.conj().T
    # Right multiplication by E^{-1}, evaluated by a Hermitian linear solve.
    polar = linalg.solve(energy, a.conj().T, assume_a="her").conj().T
    clock = root @ np.diag(lapse * psi) @ root
    identity = np.eye(len(psi), dtype=complex)
    plus = np.vstack((identity, polar)) / np.sqrt(2)
    minus = np.vstack((identity, -polar)) / np.sqrt(2)
    hamiltonian = plus @ clock @ plus.conj().T - minus @ clock @ minus.conj().T
    return a, energy, polar, clock, hamiltonian, plus, minus


def control_b():
    coordinates, derivative = sine_lattice(4)
    x, y, z = coordinates.T
    psi = np.repeat(1 + 0.1 * np.cos(np.pi * x / 2) + 0.07 * np.sin(np.pi * y / 2) + 0.05 * np.cos(np.pi * z / 2), 2)
    lapse = np.repeat(1 + 0.08 * np.sin(np.pi * (x + y) / 2) + 0.05 * np.cos(np.pi * z / 2), 2)
    contrast = np.repeat(np.cos(np.pi * x / 2) + 0.37 * np.sin(np.pi * (y + z) / 2), 2)
    results = []
    for mass in (0.1, 1.0, 3.0):
        for orientation in (1, -1):
            a, energy, polar, clock, h, plus, minus = polar_law(mass, psi, lapse, derivative, orientation)
            delta_a = np.diag(-mass * contrast / psi) - 3j * np.diag(psi ** -3 * contrast) @ derivative
            if orientation == -1:
                delta_a = delta_a.conj().T
            rhs = polar.conj().T @ delta_a - delta_a.conj().T @ polar
            k = linalg.solve_sylvester(energy, energy, rhs)
            vertex = (clock @ k + k @ clock) / 2
            normalization = max(1.0, fro(vertex))
            steps = []
            for step in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
                hp = polar_law(mass, psi * np.exp(step * contrast), lapse, derivative, orientation)[4]
                hm = polar_law(mass, psi * np.exp(-step * contrast), lapse, derivative, orientation)[4]
                ruler_block = plus.conj().T @ ((hp - hm) / (2 * step)) @ minus
                np_h = polar_law(mass, psi, lapse * np.exp(step * contrast), derivative, orientation)[4]
                nm_h = polar_law(mass, psi, lapse * np.exp(-step * contrast), derivative, orientation)[4]
                lapse_block = plus.conj().T @ ((np_h - nm_h) / (2 * step)) @ minus
                rounding = float(np.finfo(float).eps * (fro(np_h) + fro(nm_h)) / (2 * step))
                steps.append({
                    "epsilon": step,
                    "normalized_ruler_vertex_error": fro(ruler_block - vertex) / normalization,
                    "lapse_interband_norm": fro(lapse_block),
                    "lapse_roundoff_proxy": rounding,
                    "lapse_norm_over_roundoff_proxy": fro(lapse_block) / rounding,
                })
            results.append({
                "mass": mass, "orientation": orientation,
                "minimum_singular_value_A": float(linalg.svdvals(a)[-1]),
                "minimum_eigenvalue_E": float(linalg.eigvalsh(energy)[0]),
                "minimum_eigenvalue_C": float(linalg.eigvalsh(clock)[0]),
                "polar_unitarity_defect": fro(polar.conj().T @ polar - np.eye(len(psi))),
                "hamiltonian_hermiticity_defect": fro(h - h.conj().T),
                "K_antihermiticity_defect": fro(k + k.conj().T),
                "Sylvester_residual": fro(energy @ k + k @ energy - rhs),
                "vertex_Frobenius_norm": fro(vertex),
                "steps": steps,
                "best_normalized_ruler_error": min(row["normalized_ruler_vertex_error"] for row in steps),
            })
    return {
        "a": 1, "length": 4, "site_order": "x, y, z; z fastest; spin fastest within site",
        "translation_convention": "T_i sends a site basis ket to the next periodic site",
        "spectral_root": "Hermitian eigendecomposition of A-dagger A; no entrywise roots or negative-eigenvalue clipping",
        "linear_solves": "scipy.linalg.solve(assume_a=her) for polar factor; solve_sylvester for K",
        "roundoff_proxy": "eps_machine*(||h_Nplus||_F+||h_Nminus||_F)/(2*epsilon), descriptive, not a certified enclosure",
        "D_hermiticity_defect": fro(derivative - derivative.conj().T),
        "minimum_psi": float(min(psi)), "minimum_N": float(min(lapse)),
        "cases": results,
        "specified_ruler_vertex_criterion": all(row["best_normalized_ruler_error"] < 1e-7 for row in results),
    }


def control_c():
    p0, p1 = 1.0, np.sqrt(2.0)
    h0, h1 = SIGMA[2] + p0 ** -2 * SIGMA[0], SIGMA[2] + p1 ** -2 * SIGMA[0]
    e0, e1 = np.sqrt(1 + p0 ** -4), np.sqrt(1 + p1 ** -4)
    _, v0 = linalg.eigh(h0)
    _, v1 = linalg.eigh(h1)
    vacuum = v0[:, 0]
    projector = np.outer(vacuum, vacuum.conj())
    complement = np.eye(2) - projector
    hamiltonian = e1 * np.eye(2) + h1
    w = float(abs(np.vdot(v1[:, 0], vacuum)) ** 2)
    gap = float(2 * e1)
    gate = fro(complement @ hamiltonian @ projector) ** 2
    mean = float(np.vdot(vacuum, hamiltonian @ vacuum).real)
    variance = float(np.vdot(vacuum, hamiltonian @ hamiltonian @ vacuum).real - mean ** 2)
    expected = w * (1 - w) * gap ** 2
    times = []
    for time in (0.0, 0.1, 1.0, np.pi / gap):
        propagated = linalg.expm(-1j * hamiltonian * time) @ vacuum
        direct = float(linalg.norm(complement @ propagated) ** 2)
        analytic = float(4 * w * (1 - w) * np.sin(gap * time / 2) ** 2)
        times.append({"T": float(time), "direct_probability": direct, "analytic_probability": analytic, "absolute_discrepancy": abs(direct - analytic)})
    return {
        "P0": p0, "P1": float(p1), "w": w, "gap": gap,
        "initial_vacuum_energy_residual": float(abs(np.vdot(vacuum, (e0 * np.eye(2) + h0) @ vacuum))),
        "gate_Frobenius_squared": gate, "variance": variance,
        "w_one_minus_w_gap_squared": expected,
        "gate_minus_variance": gate - variance, "gate_minus_overlap_identity": gate - expected,
        "times": times,
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def report(data):
    a, b, c = data["A"], data["B"], data["C"]
    lines = [
        "# Partner analytic-identity controls", "",
        "Executed on compute-host with qi2 Python, nice 10 and one CPU thread; no GPU used.", "",
        "These controls test the author-specified tanh formula, actual one-sine polar derivative, and neutral spectral gate. They supply no realistic collapse history, native trapping evolution, observational exclusion, or whole-paper validation.", "",
        "## A: direct first-order tanh scattering", "",
        f"Largest absolute T=18 ODE discrepancy: {a['largest_absolute_ode_discrepancy_T18']:.12g}; largest positive high-precision bound violation: {a['largest_positive_bound_violation_80_digit']}.", "",
        "The 80-digit analytic evaluation and 1e-70 rounding allowance are recorded in JSON. Direct integration uses DOP853 (rtol 1e-12, atol 1e-13); the worst case is also rerun with tighter tolerances and preserved separately. Small relative probabilities below the absolute ODE floor are not certified.", "",
        "| Kind | b_f | d | tau | Direct T18 | Analytic | Abs. error | Norm defect | Endpoint change | B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a["cases"]:
        run = row["runs"][1]
        lines.append(f"| {row['kind']} | {row['bf']:g} | {row['d']:g} | {row['tau']:g} | {run['probability']:.12g} | {row['analytic_probability']:.12g} | {run['absolute_analytic_discrepancy']:.4g} | {run['maximum_norm_defect']:.4g} | {row['endpoint_refinement']:.4g} | {row['bound']:.12g} |")
    for row in a["cases"]:
        if "tighter_tolerance_run" in row:
            run = row["tighter_tolerance_run"]
            lines.extend(["", f"Worst-case refinement {a['worst_case_parameters']}: absolute discrepancy {run['absolute_analytic_discrepancy']:.12g}; probability change {run['change_from_original']:.12g}; norm defect {run['maximum_norm_defect']:.12g}."])
        if row["kind"] == "sudden":
            lines.extend(["", f"Sudden control tau={row['tau']:g}: ideal sudden probability {row['ideal_sudden_probability']:.12g}; analytic-minus-sudden {row['analytic_minus_sudden']:.12g}; direct-minus-sudden {row['direct_minus_sudden']:.12g}."])
    lines.extend(["", "## B: actual one-sine polar vertex", "", "The periodic 4^3 matrix uses site/spin order and translation conventions recorded in JSON. Square roots are spectral; the right polar inverse and Sylvester equation use documented linear solves.", "", "| m | Orientation | min singular A | min eigen C | Polar defect | K+K† defect | Hermiticity defect | Sylvester residual | Vertex norm | Best normalized error |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in b["cases"]:
        lines.append(f"| {row['mass']:g} | {row['orientation']} | {row['minimum_singular_value_A']:.8g} | {row['minimum_eigenvalue_C']:.8g} | {row['polar_unitarity_defect']:.4g} | {row['K_antihermiticity_defect']:.4g} | {row['hamiltonian_hermiticity_defect']:.4g} | {row['Sylvester_residual']:.4g} | {row['vertex_Frobenius_norm']:.8g} | {row['best_normalized_ruler_error']:.4g} |")
    lines.extend(["", "| m | Orientation | epsilon | Normalized ruler error | Lapse interband norm | Lapse roundoff proxy | Ratio to proxy |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in b["cases"]:
        for step in row["steps"]:
            lines.append(f"| {row['mass']:g} | {row['orientation']} | {step['epsilon']:g} | {step['normalized_ruler_vertex_error']:.8g} | {step['lapse_interband_norm']:.8g} | {step['lapse_roundoff_proxy']:.8g} | {step['lapse_norm_over_roundoff_proxy']:.5g} |")
    lines.extend(["", "The lapse roundoff proxy is descriptive, not a certified numerical enclosure. These profiles are prescribed conditioning controls and do not solve H2.", "", "## C: neutral spectral gate", "", f"w={c['w']:.15g}, gap={c['gap']:.15g}; gate squared={c['gate_Frobenius_squared']:.15g}, variance={c['variance']:.15g}, overlap identity={c['w_one_minus_w_gap_squared']:.15g}.", "", f"Gate-minus-variance={c['gate_minus_variance']:.12g}; gate-minus-overlap={c['gate_minus_overlap_identity']:.12g}.", "", "| T | Direct leakage | Analytic leakage | Absolute discrepancy |", "|---:|---:|---:|---:|"])
    for row in c["times"]:
        lines.append(f"| {row['T']:.12g} | {row['direct_probability']:.12g} | {row['analytic_probability']:.12g} | {row['absolute_discrepancy']:.4g} |")
    lines.extend(["", "The gate uses the actual free polar neutral block. It does not identify the P21 positive right-factor promotion with the current default internal H2 lift.", "", "The scientific author must read the full controls before importing numerical evidence into the manuscript.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if not Path('<internal>/python').samefile(os.sys.executable):
        raise RuntimeError("Use the authorized remote qi2 interpreter")
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError("Run with nice -n 10")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"Expected {key}=1")
    mp.mp.dps = 80
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = {"metadata": {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mp.__version__,
        "nice": os.getpriority(os.PRIO_PROCESS, 0), "threads": 1, "gpu_used": False,
        "specification": "research/paper_package_20261013/partner_admissibility_code_spec.tex",
    }}
    for name, operation in (("A", control_a), ("B", control_b), ("C", control_c)):
        data[name] = operation()
        write_json(output / f"{name}_results.json", data[name])
        print(json.dumps({"completed_control": name, "result": str(output / f"{name}_results.json")}), flush=True)
    data["metadata"]["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output / "results.json", data)
    (output / "RESULTS.md").write_text(report(data))
    print(json.dumps({"A_formula_criterion": data['A']['specified_formula_criterion'], "A_maximum_discrepancy": data['A']['largest_absolute_ode_discrepancy_T18'], "B_ruler_criterion": data['B']['specified_ruler_vertex_criterion'], "results": str(output / 'results.json'), "report": str(output / 'RESULTS.md')}), flush=True)


if __name__ == "__main__":
    main()
