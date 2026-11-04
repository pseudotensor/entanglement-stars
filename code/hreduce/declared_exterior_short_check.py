"""Exact transcription of the reviewer B exterior and periodic-ray check SPEC.

Run only on the authorized qi2 compute box.  This checks the supplied
symbolic identities; their physical interpretation belongs to the authors.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import sympy as sp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks = []

    def check(name: str, actual: sp.Expr, expected: sp.Expr, *, trig=False):
        residual = sp.cancel(sp.together(actual - expected))
        if trig:
            residual = sp.trigsimp(sp.expand_trig(residual), method="fu")
        residual = sp.factor(sp.cancel(sp.together(residual)))
        item = {"name": name, "passed": residual == sp.S.Zero}
        if residual != sp.S.Zero:
            item["exact_residual"] = sp.sstr(residual)
        checks.append(item)

    q = sp.Rational
    x, y, z = sp.symbols("x y z", real=True)
    amplitude = sp.symbols("C", positive=True)
    coords = (x, y, z)
    radius = sp.sqrt(x*x + y*y + z*z)
    normal = sp.Matrix([coord / radius for coord in coords])
    a4 = sum(component**4 for component in normal)
    phi0 = -2 * amplitude * sp.sqrt(radius)
    phi2 = amplitude * radius**q(-3, 2) * (91*a4 - 45) / 32
    stencil_hessian = sp.Matrix(3, 3, lambda i, j: (
        sp.diff(phi0, coords[i], 3, coords[j])
        + sp.diff(phi0, coords[i], coords[j], 3)
    ) / 6)
    check("stencil_hessian_trace", sp.trace(stencil_hessian),
          amplitude * radius**q(-7, 2) * (-6 + q(77, 8)*a4))
    check("stencil_hessian_radial_contraction",
          (normal.T * stencil_hessian * normal)[0],
          amplitude * radius**q(-7, 2) * (-q(15, 4) + q(35, 8)*a4))
    phi2_hessian = sp.hessian(phi2, coords)
    check("particular_scalar_equation",
          sp.trace(phi2_hessian) + 3*(normal.T * phi2_hessian * normal)[0],
          amplitude * radius**q(-7, 2) * (69 - 91*a4) / 4)
    velocity0 = sp.Matrix([sp.diff(phi0, coord) for coord in coords])
    velocity2 = sp.Matrix([
        sp.diff(phi2, coord) + sp.diff(phi0, coord, 3)/6
        for coord in coords
    ])
    for i in range(3):
        check(f"velocity2_component_{i}", velocity2[i],
              amplitude * radius**q(-5, 2)
              * (672*normal[i]**3 + (183-1001*a4)*normal[i]) / 64)
    check("velocity2_radial", (normal.T * velocity2)[0],
          amplitude * radius**q(-5, 2) * (183-329*a4)/64)
    check("squared_speed_coefficient",
          2*(velocity0.T*velocity2)[0],
          amplitude**2 * radius**-3 * (329*a4-183)/32)

    r, pr, angular_momentum = sp.symbols("r pr L", positive=True)
    angle = sp.symbols("phi", real=True)
    tangent_momentum = angular_momentum/r
    magnitude = sp.sqrt(pr**2 + tangent_momentum**2)
    flow_magnitude = sp.sqrt(q(2, 3)/r)
    b = (255 + 329*sp.cos(4*angle))/256
    tangential = q(21, 8)*sp.sin(4*angle)
    h_geo = -flow_magnitude/r**2 * (b*pr + tangential*tangent_momentum)
    nx, ny = sp.cos(angle), sp.sin(angle)
    px = pr*nx - tangent_momentum*ny
    py = pr*ny + tangent_momentum*nx
    h_disp = -(px**4 + py**4)/(6*magnitude) + q(2, 3)*flow_magnitude*(nx*px**3 + ny*py**3)
    base = {r: sp.S.One, pr: sp.sqrt(2), angular_momentum: sp.S.One}

    def forcing(hamiltonian: sp.Expr) -> sp.Expr:
        result = sp.sqrt(3) * (
            -sp.diff(hamiltonian, r)/3
            + 5*sp.sqrt(2)*sp.diff(hamiltonian, pr)/6
            + sp.diff(hamiltonian, angle, pr)
            + sp.sqrt(2)*sp.diff(hamiltonian, angle)/3
        )
        return result.subs(base, simultaneous=True)

    forcing_geo = -q(425, 128) - q(4333, 384)*sp.cos(4*angle) + q(1057, 192)*sp.sqrt(2)*sp.sin(4*angle)
    forcing_disp = q(19, 4) - q(77, 12)*sp.cos(4*angle) - q(11, 3)*sp.sqrt(2)*sp.sin(4*angle)
    response_geo = q(425, 128) + q(4333, 6528)*sp.cos(4*angle) - q(1057, 3264)*sp.sqrt(2)*sp.sin(4*angle)
    response_disp = -q(19, 4) + q(77, 204)*sp.cos(4*angle) + q(11, 51)*sp.sqrt(2)*sp.sin(4*angle)
    check("periodic_ray_geometry_forcing", forcing(h_geo), forcing_geo, trig=True)
    check("periodic_ray_dispersion_forcing", forcing(h_disp), forcing_disp, trig=True)
    check("periodic_ray_geometry_ode", sp.diff(response_geo, angle, 2)-response_geo, forcing_geo, trig=True)
    check("periodic_ray_dispersion_ode", sp.diff(response_disp, angle, 2)-response_disp, forcing_disp, trig=True)

    passed = all(item["passed"] for item in checks)
    data = {
        "check": "reviewer B declared-exterior symbolic SPEC",
        "sympy_version": sp.__version__,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "elapsed_seconds": time.monotonic()-started,
        "arithmetic": "exact rational symbolic differentiation; zero residual required",
        "domain": "r>0, C>0; periodic xy orbit at rs=2/3, r=L=1, pr=sqrt(2)",
        "checks": checks,
        "all_passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks": len(checks), "all_passed": passed, "elapsed_seconds": data["elapsed_seconds"], "nice": data["nice"]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
