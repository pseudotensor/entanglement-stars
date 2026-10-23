"""V2.9 stage diagnostic on four saved leaves and their real centers.

Serves the failed fixed-mass certificate for insert I6a and item 7.
No entropy sum, integration, refinement or directional value is computed.
The original evaluator is stopped immediately before its first hF call.
"""
from fractions import Fraction
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

ROOT = Path(__file__).parent
OUTPUT = ROOT / "fixed_mass_image_diagnostic_result.json"
START = time.monotonic()
REPORT = {"status": "running", "purpose": "insert I6a and item7 fixed-mass enclosure failure",
          "specification": "ETA_massive_directions_v2.tex V2.9",
          "scope": "four stored110 rectangles and their real centers; phase stages only",
          "not_an_eta_certificate": True,
          "resources": {"bits": 192, "threads": 1, "nice": 10, "cuda": "disabled",
                        "wall_seconds": 30, "internal_seconds": 25,
                        "address_space_bytes": 268435456}, "cases": []}
CURRENT = None
NAMES = {"x", "theta", "phi", "u", "v", "u2", "v2", "Y", "M", "U", "V", "total",
         "numerator", "C", "D", "k", "p", "m", "kp", "km", "epsilon", "hchi",
         "chi", "one_chi", "one_mchi", "rf", "delta"}


class PhaseReady(Exception):
    pass


def stop_before_entropy(*_args):
    raise PhaseReady("phase reached; no entropy term evaluated")


def deadline(_signal, _frame):
    raise TimeoutError("diagnostic persistence deadline")


def save():
    REPORT["elapsed_seconds"] = time.monotonic() - START
    REPORT["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
    temporary.replace(OUTPUT)


def encode(value):
    if isinstance(value, acb):
        result = {"finite": bool(value.is_finite()), "ball": value.str(65, more=True)}
        if value.is_finite():
            result.update(real=mod.interval(value.real), imaginary=mod.interval(value.imag))
        return result
    return {"finite": bool(value.is_finite()), "real": mod.interval(value)} if value.is_finite() else {"finite": False, "ball": str(value)}


def trace(frame, event, argument):
    if CURRENT is None:
        return trace
    name = frame.f_code.co_name
    if event == "call" and name == "denominator" and "z" in frame.f_locals:
        caller = frame.f_back
        CURRENT["last_denominator"] = {
            "caller": caller.f_code.co_name, "source": caller.f_code.co_filename,
            "line": caller.f_lineno, "argument": encode(frame.f_locals["z"])}
    if event == "line" and name in ("evaluate", "entropy", "candidate_phase"):
        for key in NAMES & frame.f_locals.keys():
            value = frame.f_locals[key]
            if isinstance(value, (arb, acb)):
                CURRENT["stage_images"][key] = encode(value)
    if event == "exception" and isinstance(argument[1], mod.NonanalyticImage):
        CURRENT.setdefault("first_refusal_site", {
            "function": name, "source": frame.f_code.co_filename, "line": frame.f_lineno,
            "stage": frame.f_locals.get("stage"), "reason": argument[1].reason})
    return trace


def rectangle(radius):
    radius = mod.finite(radius).upper()
    if not radius >= 0:
        raise mod.NonanalyticImage("negative_midpoint_error", radius)
    return acb(arb(0, radius), arb(0, radius))


def midpoint_and_radius(z):
    center = acb(z.real.mid(), z.imag.mid())
    radius = (z.real.rad()**2 + z.imag.rad()**2).sqrt().upper()
    return center, radius


def local_k(c, label):
    mod.positive_real(c, "Kc_complement")
    c0, rho = midpoint_and_radius(c)
    a = c.real.lower()
    raw = (1-c).elliptic_k()
    record = {"complement": encode(c), "midpoint": encode(c0), "rho": encode(rho),
              "a": encode(a), "raw": encode(raw)}
    CURRENT["special_images"][label] = record
    point_value = (1-c0).elliptic_k()
    record["point"] = encode(point_value)
    mod.finite(point_value, "nonfinite_midpoint_K")
    error = (rho/(2*a)).upper()
    value = mod.finite(point_value + rectangle(error))
    record.update(error=encode(error), alternative=encode(value))
    return value


def local_rf(x, y):
    mod.positive_real(x, "RF_x")
    mod.positive_real(y, "RF_y")
    a = min(arb(1), x.real.lower(), y.real.lower())
    if not a > 0:
        raise mod.NonanalyticImage("RF_midpoint_domain", a)
    x0, rx = midpoint_and_radius(x)
    y0, ry = midpoint_and_radius(y)
    raw = acb.elliptic_rf(x, y, acb(1))
    record = {"x": encode(x), "y": encode(y), "x0": encode(x0), "y0": encode(y0),
              "rho_x": encode(rx), "rho_y": encode(ry), "a": encode(a), "raw": encode(raw)}
    CURRENT["special_images"]["RF"] = record
    point_value = acb.elliptic_rf(x0, y0, acb(1))
    record["point"] = encode(point_value)
    mod.finite(point_value, "nonfinite_midpoint_RF")
    error = ((rx+ry)/(6*a*a.sqrt())).upper()
    value = mod.finite(point_value + rectangle(error))
    record.update(error=encode(error), alternative=encode(value))
    return value


def candidate_phase(z, t):
    x = mod.finite((-z*mod.LOG2).exp())
    theta, phi = mod.PI/2*(1-x), mod.PI/2*t
    co, si = (theta/2).cos(), (theta/2).sin()
    u, v = mod.SQRT2*co, mod.SQRT2*si
    Y = mod.positive_real(acb(1)/25 + phi.sin()**2, "mass_squared")
    M = mod.positive_sqrt(Y)
    C, D = 1+Y, (mod.PI/2*x).sin()
    u2, v2 = 1+D, 1-D
    U, V = mod.positive_sqrt(C+D), mod.positive_sqrt(C-D)
    total = mod.denominator(U+V)
    k = D/mod.denominator(C+U*V)
    p = k*k
    m = 1-p
    mod.positive_real(p, "p")
    mod.positive_real(m, "m")
    kp, km = mod.denominator(local_k(m, "Kp")), local_k(p, "Km")
    epsilon = mod.finite(mod.PI*km/kp)
    if not epsilon.real > arb(3)/2:
        raise mod.NonanalyticImage("epsilon_real_guard", epsilon)
    hchi = (1/mod.denominator(U+u) + 1/mod.denominator(V+v))*(U+V+u+v)/mod.denominator(4*U*V)
    mod.positive_real(hchi, "Hchi")
    chi = Y*hchi
    one_chi = mod.positive_real(1-chi, "one_minus_chi")
    one_mchi = mod.positive_real(1-m*chi, "one_minus_mchi")
    rf = local_rf(one_chi, one_mchi)
    delta = mod.positive_real(mod.PI*M*mod.positive_sqrt(hchi)*rf/kp, "delta")
    mod.positive_real(epsilon-delta, "epsilon_minus_delta")
    return True


def run_case(leaf, mode, route):
    global CURRENT
    bounds = tuple(Fraction(x) for x in leaf["bounds_z_t"])
    a, b, c, d = bounds
    cz, ct, hz, ht = (a+b)/2, (c+d)/2, (b-a)/2, (d-c)/2
    if mode == "joint_box":
        z = acb(arb(mod.point(cz), 4*mod.point(hz)), arb(0, 4*mod.point(hz)))
        t = acb(arb(mod.point(ct), 4*mod.point(ht)), arb(0, 4*mod.point(ht)))
    else:
        z, t = acb(mod.point(cz)), acb(mod.point(ct))
    case = {"leaf_id": leaf["id"], "mode": mode, "route": route,
            "saved_bounds_z_t": leaf["bounds_z_t"], "saved_depths": leaf["depths"],
            "saved_guard_reason": leaf.get("guard_reason"), "z": encode(z), "t": encode(t),
            "stage_images": {}, "special_images": {}, "status": "running"}
    REPORT["cases"].append(case)
    CURRENT = case
    sys.settrace(trace)
    try:
        if route == "original":
            mod.evaluate(z, t, {"direction": "110"}, "saved_leaf_diagnostic", True)
            raise RuntimeError("original diagnostic unexpectedly evaluated an entropy")
        candidate_phase(z, t)
        case["status"] = "phase_reached_no_entropy"
    except PhaseReady:
        case["status"] = "phase_reached_no_entropy"
    except mod.NonanalyticImage as error:
        case.update(status="guard_refused", first_guard=error.reason,
                    tested_image=error.image, tested_input=error.input_image,
                    traceback=[{"source": x.filename, "line": x.lineno,
                                "function": x.name, "statement": x.line}
                               for x in traceback.extract_tb(error.__traceback__)])
    finally:
        sys.settrace(None)
        CURRENT = None
        save()


def main():
    global mod, arb, acb
    with OUTPUT.open("x") as stream:
        json.dump(REPORT, stream)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(25)
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    import flint
    from flint import arb, acb, ctx
    ctx.prec, ctx.threads = 192, 1
    code = ROOT / "fixed_mass_interval_repair.py"
    data = ROOT / "fixed_mass_repair_result.json"
    spec = importlib.util.spec_from_file_location("fixed_mass_repair_diagnostic_source", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.arb, mod.acb = arb, acb
    mod.constants()
    mod.hfermi = stop_before_entropy
    full = json.loads(data.read_text())
    wanted = (9274, 9300, 9558, 9998)
    leaves = {x["id"]: x for x in full["active_panel"]["leaves"] if x["id"] in wanted}
    assert set(leaves) == set(wanted)
    del full
    gc.collect()
    REPORT["inputs"] = {"producer": str(code), "producer_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                        "terminal_result": str(data), "terminal_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                        "saved_leaf_ids": list(wanted)}
    REPORT["implementation"] = {"python": sys.version.split()[0], "python_flint": flint.__version__,
                                "box": "compute-host", "nice_actual": os.getpriority(os.PRIO_PROCESS, 0),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    save()
    for leaf_id in wanted:
        for mode in ("joint_box", "real_center"):
            for route in ("original", "candidate_CD_midpoint"):
                run_case(leaves[leaf_id], mode, route)
    REPORT["status"] = "complete_diagnostic_only"
    save()


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Refusing to overwrite or repeat this diagnostic")
    try:
        main()
    except Exception as error:
        REPORT["status"] = "incomplete_diagnostic"
        REPORT["error"] = type(error).__name__ + ": " + str(error)
        save()
    finally:
        sys.settrace(None)
        signal.alarm(0)
    print(json.dumps({"status": REPORT["status"], "case_count": len(REPORT["cases"]),
                      "elapsed_seconds": REPORT.get("elapsed_seconds")}))
    sys.exit(0 if REPORT["status"] == "complete_diagnostic_only" else 1)
