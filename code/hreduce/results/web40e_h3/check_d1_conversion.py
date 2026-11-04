"""One bounded arithmetic check for the owner-approved D1 scope sentence.

The scientific sign/identification is supplied by both original mechanics
workers. This checks their shared literal Decimal specification only.
Run remotely in qi2 with one thread and no GPU.
"""

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**2, 128 * 1024**2))
    started = time.perf_counter()
    with localcontext() as context:
        context.prec = 80
        planck_mass = Decimal("1.220890e19")
        lower = Decimal("-3.4e-45")
        upper = Decimal("3.4e-42")
        coefficient = (Decimal(2) / 5) / planck_mass**2
        ratio = coefficient / upper
        eta = -2 * planck_mass**2 * coefficient
        checks = {
            "coefficient_positive": coefficient > 0,
            "coefficient_range": Decimal("2.683e-39") < coefficient < Decimal("2.685e-39"),
            "ratio_range": Decimal(788) < ratio < Decimal(790),
            "eta_exact_to_decimal_roundoff": abs(eta + Decimal("0.8")) < Decimal("1e-75"),
        }
        assert all(checks.values()), checks
        result = {
            "serves": "WEB40E H3 / approved D1 elementary-proton comparison scope sentence",
            "purpose": "Literal bounded arithmetic; no new physics fit or observational analysis",
            "decimal_precision": context.prec,
            "inputs": {
                "Planck_mass_energy_GeV": str(planck_mass),
                "a_in_Planck_lengths": "2",
                "SME_proton_interval_GeV_minus_2": [str(lower), str(upper)],
                "mapping_from_original_pair": "delta_E=-c*k^3 maps to positive SME c; use positive upper endpoint",
            },
            "isotropic_c_GeV_minus_2": str(coefficient),
            "ratio_to_positive_upper_endpoint": str(ratio),
            "eta_p": str(eta),
            "checks": checks,
            "scope": "Direct elementary-proton identification under cited reaction assumptions; no composite bound-state prediction",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time(),
            "max_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    Path("d1_conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
