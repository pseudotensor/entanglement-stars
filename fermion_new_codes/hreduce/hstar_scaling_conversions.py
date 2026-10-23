"""Physical-unit arithmetic specified in hstar_scaling_short.tex.

Use CODATA 2022 central values and the IAU 2015 B3 nominal solar mass
parameter. The output is a unit conversion, not a stellar calculation
or an uncertainty enclosure. Run remotely in qi2 at nice 10.
"""

import json
import os

import mpmath as mp


def main():
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10
    mp.mp.dps = 50
    constants = {
        "h_J_s": "6.62607015e-34",
        "c_m_s": "299792458",
        "G_m3_kg_s2": "6.67430e-11",
        "nominal_GM_sun_m3_s2": "1.3271244e20",
        "e_C": "1.602176634e-19",
    }
    h = mp.mpf(constants["h_J_s"])
    c = mp.mpf(constants["c_m_s"])
    grav = mp.mpf(constants["G_m3_kg_s2"])
    gm_sun = mp.mpf(constants["nominal_GM_sun_m3_s2"])
    hbar = h / (2 * mp.pi)
    length_p = mp.sqrt(hbar * grav / c**3)
    mass_p = mp.sqrt(hbar * c / grav)
    nominal_mass_sun = gm_sun / grav
    planck_energy = mass_p * c**2
    gev_in_joules = mp.mpf(constants["e_C"]) * mp.mpf("1e9")
    values = {
        "hbar_J_s": hbar,
        "planck_length_m": length_p,
        "planck_mass_kg": mass_p,
        "planck_energy_J": planck_energy,
        "spacing_0p0055_lP_m": mp.mpf("0.0055") * length_p,
        "spacing_0p0004_lP_m": mp.mpf("0.0004") * length_p,
        "nominal_solar_mass_kg": nominal_mass_sun,
        "nominal_solar_mass_over_planck_mass": nominal_mass_sun / mass_p,
        "schwarzschild_radius_per_nominal_solar_mass_m": 2 * gm_sun / c**2,
        "constituent_mass_energy_prefactor_GeV": (
            mp.sqrt(mass_p**3 / nominal_mass_sun) * c**2 / gev_in_joules
        ),
    }
    ratio_relative_difference = abs(
        (nominal_mass_sun / mass_p) / (gm_sun / (c**2 * length_p)) - 1
    )
    radius_relative_difference = abs(
        (2 * grav * nominal_mass_sun / c**2)
        / (2 * length_p * (nominal_mass_sun / mass_p)) - 1
    )
    all_positive_finite = all(mp.isfinite(value) and value > 0 for value in values.values())
    assert all_positive_finite
    assert ratio_relative_difference < mp.mpf("1e-25")
    assert radius_relative_difference < mp.mpf("1e-25")
    print(json.dumps({
        "scope": "central-value unit conversions; no stellar solution or uncertainty enclosure",
        "constants": constants,
        "sources": [
            "https://physics.nist.gov/cuu/pdf/wall_2022.pdf",
            "https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf",
        ],
        "formulas": {
            "hbar": "h/(2*pi)",
            "lP": "sqrt(hbar*G/c^3)",
            "mP": "sqrt(hbar*c/G)",
            "nominal_M_sun": "nominal_GM_sun/G",
            "radius_per_nominal_solar_mass": "2*nominal_GM_sun/c^2",
            "planck_energy": "mP*c^2",
            "constituent_mass_energy_prefactor_GeV": "sqrt(mP^3/nominal_M_sun)*c^2/(e*1e9)",
        },
        "checks": {
            "all_positive_finite": all_positive_finite,
            "solar_planck_ratio_relative_difference": mp.nstr(ratio_relative_difference, 40),
            "radius_identity_relative_difference": mp.nstr(radius_relative_difference, 40),
            "relative_difference_limit": "1e-25",
        },
        "arithmetic_decimal_places": mp.mp.dps,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "values": {name: mp.nstr(value, 40) for name, value in values.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
