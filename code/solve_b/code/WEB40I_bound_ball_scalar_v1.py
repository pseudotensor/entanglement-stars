"""Unrun implementation of N/A's fixed X4 bound-rest scalar contract.

Imported only by the guarded own scalar producer on the box. This module
defines functions; it neither imports mpmath nor launches a calculation.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

SPEC_PATH = "<internal>/WEB40I_bound_ball_ray_map_v1.tex"
SPEC_SHA256 = "f7f7a248ce3d1b20ccbcd9771141ef8ac8e0db5cf3571e5814ea755a496403bc"
MARGINAL_SHA256 = "1bf75ab20dfbe657e81d671b2fc03edbdc3e45455b065e98f8119145c5fc1ebb"


def inherited_comparison(path):
    """Copy labelled saved values; never call an X1 scientific routine."""
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("named marginal JSON exceeds the small-input ceiling")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MARGINAL_SHA256:
        raise ValueError("saved marginal-comparator identity differs")
    source = json.loads(raw)
    if source["complete"] is not True or source["new_ode_solves"] != 0:
        raise ValueError("unexpected saved scalar-comparison record")
    high = source["scalar_80"]
    onsets = [{key: row[key] for key in ("criterion", "gap", "lapse", "constant")}
              for row in high["onsets"]]
    keys = ("criterion", "convention", "size", "frequency_index", "q", "b",
            "empty", "D", "N_H", "N_peel", "oscillation_cycles", "A_endpoint",
            "delta_T_over_rs")
    segments = [{key: row[key] for key in keys}
                for row in high["segments"]
                if row["size"] == 64 and row["frequency_index"] in (1, 6)]
    if len(onsets) != 3 or len(segments) != 12:
        raise ValueError("missing fixed-frequency inherited comparison rows")
    return {"path": str(path.resolve()), "sha256": MARGINAL_SHA256,
            "label": "marginally bound comparator; inherited, not recomputed",
            "onsets": onsets, "segments": segments}


def evaluate_precision(mp, digits, checkpoint):
    partial = {"status": "started", "decimal_working_precision": digits}
    try:
        return _evaluate_precision(mp, digits, checkpoint, partial)
    except BaseException:
        partial["status"] = "interrupted_or_failed"
        checkpoint(partial)
        raise


def _evaluate_precision(mp, digits, checkpoint, partial):
    """B8--B24, exactly alpha=4, Sg=64 and the prescribed two frequencies."""
    with mp.workdps(digits):
        m = mp.mpf
        zero, one, third = m(0), m(1), m(1)/3
        sqrt3, log3, e = mp.sqrt(3), mp.log(3), mp.sqrt(3)/2
        tolerance = m("1e-30")
        root_tolerance = m("1e-40" if digits == 50 else "1e-65")
        calls = Counter()
        finite = {}
        identity_residuals = []
        onsets, endpoints, segments, quadrature_nodes, grid, augmented = [], [], [], [], [], []
        partial.update(onsets=onsets, endpoints=endpoints, segments=segments,
                       quadrature_nodes=quadrature_nodes, map_grid=grid, map_augmented=augmented,
                       finite_values_for_cross_precision=finite, function_evaluation_counts=calls)

        def fmt(value):
            return mp.nstr(value, n=digits, strip_zeros=False)

        def encoded(value):
            if isinstance(value, dict):
                return {key: encoded(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encoded(item) for item in value]
            if value is None or isinstance(value, (str, bool, int)):
                return value
            return fmt(value)

        def retain(prefix, values):
            for key, value in values.items():
                if value is not None and not isinstance(value, (str, bool, int, dict, list)):
                    if not mp.isfinite(value):
                        raise ArithmeticError("nonfinite scalar: " + prefix + "." + key)
                    finite[prefix + "." + key] = fmt(value)
            return encoded(values)

        def B(x):
            calls["B"] += 1
            return (-2*mp.log1p(-x) + mp.log1p(3*x*x)
                    + (12*x-4)/(1+3*x*x) + 6*sqrt3*mp.atan(sqrt3*x))

        def dB(x):
            calls["dB"] += 1
            return 32/((1+3*x*x)**2*(1-x))

        def logP(x):
            calls["logP"] += 1
            return mp.log(m(9)/8)+mp.log1p(-x)+3*mp.log1p(x)-mp.log1p(3*x)

        def K(x):
            calls["K"] += 1
            return (1+3*x*x)**2*(9*x*x-2*x+1)/(16*(1+x)*(1+3*x))

        def entry(x):
            return (x-third)/(1+x)

        def V(x):
            return -B(-x)+log3

        def clocks(x):
            X = 4/(1+3*x*x)
            eta = 2*mp.atan(sqrt3*x)
            u, v = B(x)-log3, V(x)
            T = (u+v)/2
            pg = T+2*mp.sqrt(X)+mp.log((mp.sqrt(X)-1)/(mp.sqrt(X)+1))-4+log3
            return {"u_over_rg": u, "V_over_rg": v, "T_over_rg": T,
                    "tau_over_rg": 4*(eta+mp.sin(eta)), "PG_t_over_rg": pg}

        def point(x, prefix, label):
            xi = entry(x)
            X, Xi = 4/(1+3*x*x), 4/(1+3*xi*xi)
            eta, eta_i = 2*mp.atan(sqrt3*x), 2*mp.atan(sqrt3*xi)
            P = mp.exp(logP(x))
            tau_prime = 16*sqrt3/(1+3*x*x)**2
            T_prime = (dB(x)+dB(-x))/2
            defects = {
                "primitive_derivative": mp.diff(B, x)-dB(x),
                "composed_map_derivative": mp.diff(lambda z: V(entry(z)), x)/dB(x)-P,
                "peeling_log_derivative": -2*mp.diff(logP, x)/dB(x)-K(x),
                "exterior_tortoise": V(x)-(B(x)-log3)-2*(X+mp.log(X-1)),
                "interior_diameter": eta-eta_i-mp.pi/3,
                "surface_radial_velocity": (-24*x/(1+3*x*x)**2)/tau_prime+e*x,
                "surface_geodesic": 1-1/X-e*e*(1-x*x),
                "surface_clock_matching": T_prime/tau_prime-e/(e*e*(1-x)*(1+x)),
                "outgoing_affine_matching": dB(x)/tau_prime-1/(e*(1-x)),
                "incoming_affine_matching": dB(-x)/tau_prime-1/(e*(1+x)),
                "mass_matching": m(8)*(m(1)/2)**3-1,
                "clock_ratio": dB(x)/T_prime-(1+x),
            }
            identity_residuals.extend(abs(v) for v in defects.values())
            values = {"x": x, "x_in": xi, "eta": eta, "eta_in": eta_i,
                      "X": X, "X_in": Xi, "gap": X-1, "lapse": 1-1/X,
                      "p_over_rg": V(xi), "P": P, "K": K(x), **clocks(x)}
            return {"label": label, **retain(prefix, values),
                    "identity_residuals": retain(prefix+".identity", defects)}

        onset_x = {}
        for c_string in ("0.95", "0.80", "0.50"):
            partial["active_operation"] = "onset_"+c_string
            c = m(c_string)
            lo, hi = third, one
            count = 2
            flo, fhi = K(lo)-c, K(hi)-c
            if not flo < 0 < fhi:
                raise ArithmeticError("onset root lacks its prescribed bracket")
            while hi-lo > root_tolerance:
                mid = (lo+hi)/2
                fmid = K(mid)-c
                count += 1
                if fmid <= 0:
                    lo, flo = mid, fmid
                else:
                    hi, fhi = mid, fmid
                partial["active_root"] = encoded({"criterion": c, "lower": lo, "upper": hi,
                                                  "f_lower": flo, "f_upper": fhi,
                                                  "function_evaluations": count})
            xc = (lo+hi)/2
            residual = K(xc)-c
            identity_residuals.append(abs(residual))
            count += 1
            onset_x[c_string] = xc
            row = point(xc, "onset."+c_string, "onset_"+c_string)
            row.update(retain("onset."+c_string+".root", {
                "criterion": c, "constant": 2*log3+2+2*mp.pi*sqrt3-B(xc),
                "lower": lo, "upper": hi, "f_lower": flo, "f_upper": fhi,
                "residual": residual, "width": hi-lo,
                "stopping_width": root_tolerance}))
            row["root_function_evaluations"] = count
            row["root_bisections"] = count-3
            onsets.append(row)
            checkpoint(partial)

        endpoint_x = []

        def quad(name, function, lo, hi):
            nodes = []
            quadrature_nodes.append({"name": name, "nodes": nodes})
            def observed(x):
                value = function(x)
                nodes.append({"x": fmt(x), "value": fmt(value)})
                return value
            value, error = mp.quad(observed, [lo, hi], error=True)
            quadrature_nodes[-1]["function_evaluations"] = len(nodes)
            return value, error

        for frequency, q in (("hawking", one/(4*mp.pi)), ("unit", one)):
            b = (2*q/64)**(m(2)/3)
            for convention in ("gap", "lapse"):
                xd = (mp.sqrt((3-b)/(3*(1+b))) if convention == "gap"
                      else mp.sqrt(1-4*b/3))
                if not third <= xd < 1:
                    raise ArithmeticError("prescribed endpoint outside the post-release domain")
                one_minus_x = (4*b/(3*(1+b)*(1+xd)) if convention == "gap"
                               else 4*b/(3*(1+xd)))
                identity_residuals.append(abs(one_minus_x-(1-xd)))
                endpoint_x.append((frequency+"_"+convention, xd))
                end = point(xd, "endpoint."+frequency+"."+convention,
                            frequency+"_"+convention)
                end.update(retain("endpoint."+frequency+"."+convention+".parameters",
                                  {"q": q, "S": m(64), "b": b,
                                   "one_minus_x_stable": one_minus_x,
                                   "one_minus_x_identity": one_minus_x-(1-xd)}))
                end.update(frequency=frequency, convention=convention, valid=True)
                endpoints.append(end)
                for c_string, xc in onset_x.items():
                    c = m(c_string)
                    label = frequency+"."+convention+"."+c_string
                    partial["active_operation"] = "segment_"+label
                    raw_D = B(xd)-B(xc)
                    raw_peel = logP(xc)-logP(xd)
                    accepted = xd > xc
                    expected = frequency == "hawking" or c_string == "0.50"
                    if accepted != expected:
                        raise ArithmeticError("empty-segment census differs from the proved inequalities")
                    start_clocks, end_clocks = clocks(xc), clocks(xd)
                    raw_clocks = {"delta_"+key: end_clocks[key]-start_clocks[key]
                                  for key in start_clocks}
                    accepted_values = {"D": raw_D if accepted else zero,
                                       "N_H": raw_D/2 if accepted else zero,
                                       "N_peel": raw_peel if accepted else zero,
                                       "mode_cycles": q*raw_D/(2*mp.pi) if accepted else zero,
                                       **{key: value if accepted else zero
                                          for key, value in raw_clocks.items()}}
                    constant = 2*log3+2+2*mp.pi*sqrt3-B(xc)
                    leading = m(4)/3*mp.log(64/(2*q))+constant
                    raw_values = {"raw_D": raw_D, "raw_N_peel": raw_peel,
                                  "leading_log_plus_constant": leading,
                                  "exact_minus_log_plus_constant": raw_D-leading,
                                  "clock_identity_residual": raw_D-2*raw_clocks["delta_T_over_rg"]
                                  -(B(-xd)-B(-xc)),
                                  **{"raw_"+key: value for key, value in raw_clocks.items()}}
                    identity_residuals.append(abs(raw_values["clock_identity_residual"]))
                    integral = None
                    if accepted:
                        duration_quad, duration_error = quad(label+".duration", dB, xc, xd)
                        peel_quad, peel_error = quad(label+".peeling", lambda x: K(x)*dB(x)/2, xc, xd)
                        lower, upper = raw_peel-c*raw_D/2, raw_D/2-raw_peel
                        if min(lower, upper) < -tolerance:
                            raise ArithmeticError("integrated peeling inequality failed")
                        differences = {"duration_quad_minus_primitive": duration_quad-raw_D,
                                       "peeling_quad_minus_log_ratio": peel_quad-raw_peel}
                        identity_residuals.extend(abs(value) for value in differences.values())
                        integral = retain("segment."+label+".integral", {
                            "duration": duration_quad, "peeling": peel_quad,
                            "duration_error_estimate": duration_error,
                            "peeling_error_estimate": peel_error,
                            "lower_peeling_slack": lower, "upper_peeling_slack": upper,
                            **differences})
                    segments.append({"frequency": frequency, "convention": convention,
                                     "criterion": c_string, "status": "accepted" if accepted else "empty",
                                     "valid": True,
                                     **retain("segment."+label+".accepted", accepted_values),
                                     **retain("segment."+label+".raw", raw_values),
                                     "accepted_integrals": integral})
                checkpoint(partial)

        for j in range(257):
            partial["active_operation"] = "map_grid_"+str(j)
            grid.append(point(1-m(2)/3*mp.power(2, -m(j)/8), "map."+str(j), "grid_"+str(j)))
            if j % 64 == 0:
                checkpoint(partial)
        augmented.extend(point(x, "augmented.onset."+label, "onset_"+label)
                         for label, x in onset_x.items())
        augmented += [point(x, "augmented.endpoint."+label, label) for label, x in endpoint_x]
        maximum_identity = max(identity_residuals, default=zero)
        if maximum_identity >= tolerance:
            raise ArithmeticError("fixed scalar identity consistency threshold failed")
        return {"decimal_working_precision": digits,
                "specification": {"path": SPEC_PATH, "sha256": SPEC_SHA256,
                                  "equations": "B8--B24; section 7 agreed finite scalar contract"},
                "case": {"alpha": "4", "rg_over_a": "64", "frequencies": ["1/(4*pi)", "1"],
                         "production_preparation_identity": False},
                "onsets": onsets, "endpoints": endpoints, "segments": segments,
                "map_grid": grid, "map_augmented": augmented,
                "quadrature_nodes": quadrature_nodes,
                "analytic_limits": encoded({"P_at_start": m(8)/9, "K_at_start": one/18,
                                            "u_at_start_over_rg": mp.pi*sqrt3,
                                            "p_H_over_rg": 6+mp.pi*sqrt3+mp.log(4)}),
                "consistency": {"maximum_absolute_identity_defect": fmt(maximum_identity),
                                "required_absolute_limit": "1e-30", "passed": True,
                                "interval_certificate": False},
                "function_evaluation_counts": dict(calls),
                "identity_rows": len(onsets)+len(endpoints)+len(grid)+len(augmented),
                "quadrature_function_evaluations": sum(len(z["nodes"]) for z in quadrature_nodes),
                "finite_values_for_cross_precision": finite}


def compare_precisions(mp, low, high):
    left, right = low["finite_values_for_cross_precision"], high["finite_values_for_cross_precision"]
    if left.keys() != right.keys():
        raise ValueError("fixed scalar rows differ across precisions")
    with mp.workdps(80):
        differences = {}
        maximum = mp.mpf(0)
        for key in left:
            a, b = mp.mpf(left[key]), mp.mpf(right[key])
            absolute = abs(a-b)
            maximum = max(maximum, absolute)
            scale = max(abs(a), abs(b))
            differences[key] = {"absolute": mp.nstr(absolute, 80),
                                "scaled_relative": mp.nstr(absolute/scale, 80) if scale else None}
        return {"passed": maximum < mp.mpf("1e-30"),
                "maximum_absolute_difference": mp.nstr(maximum, 80),
                "required_absolute_limit": "1e-30", "finite_entries": len(differences),
                "differences": differences, "interval_certificate": False}
