"""Explicit free ingredients for reviewed prescribed polar constructions.

uv_r=None is the original sine control; uv_r=2 and uv_r=3 are the
separately reviewed T10 ingredients. Each run names its law and inputs.
The order-six builder path has not yet been numerically validated.
The supported Dirac inputs are the reviewed Cartesian parents, whose
square is the scalar-spin sum of directional squares. The A/C callers
retain their displayed diagnostic masses and do not select the T10
entropy mass rectangle through this flag.
Run remotely.
"""

from __future__ import annotations

import numpy as np

from matrix_jet import Jet


def diagonal_field(field: Jet) -> Jet:
    return field.map(lambda x: np.diag(np.repeat(x, 2)))


def polar_jet(ruler: Jet, dirac: np.ndarray, mass: float,
              uv_r: int | None = None, omega: float = 1.0) -> dict:
    inverse = diagonal_field(ruler ** -1)
    inverse_cubed = diagonal_field(ruler ** -3)
    if uv_r is None:
        matrix_a = mass * inverse + 1j * inverse_cubed @ dirac
    elif uv_r in (2, 3):
        # Supported Cartesian-parent assumption: D_a^2 equals the
        # scalar-spin sum_i D_i^2. This is not a generic-Dirac API claim.
        inverse_squared = diagonal_field(ruler ** -2)
        tpsi = inverse_squared @ (dirac @ dirac) @ inverse_squared
        real_factor = (mass ** 2 * np.eye(len(dirac))
                       + tpsi.matrix_power(2 * uv_r) / omega ** (4 * uv_r - 2)).positive_root()
        psi = diagonal_field(ruler)
        matrix_a = inverse_cubed @ (psi @ real_factor @ psi + 1j * dirac)
    else:
        raise ValueError("Select the original law or reviewed uv_r=2,3 explicitly.")
    energy0 = (matrix_a.dagger() @ matrix_a).hermitian().positive_root()
    energy1 = (matrix_a @ matrix_a.dagger()).hermitian().positive_root()
    unitary = matrix_a @ energy0.inverse()
    return {"A": matrix_a, "E0": energy0, "E1": energy1, "U": unitary,
            "rootE0": energy0.positive_root()}


def polar_packet(ruler, dirac, mass, uv_r=None, omega=1.0):
    return {name: matrix.value for name, matrix in polar_jet(Jet(ruler), dirac, mass, uv_r, omega).items()}
