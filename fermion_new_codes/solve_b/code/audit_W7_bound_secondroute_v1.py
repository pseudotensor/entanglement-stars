#!/usr/bin/env python3
"""Own implementation of W7_bound_secondroute_spec_v1.md; CPU box only.

Controls, physical contraction, and post-freeze comparison are separate modes.
No other lane's implementation or derived readout is imported or executed.
"""
import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["QI2_REMOTE_CPU"] = "1"

import argparse
import hashlib
import json
import platform
import socket
import sys
import zipfile
from pathlib import Path

import numpy as np
from mpmath.ctx_iv import MPIntervalContext
from mpmath.libmp import mpf_cmp
from threadpoolctl import threadpool_info, threadpool_limits

IV = MPIntervalContext()
IV.prec = 256


class AdmissionError(Exception):
    def __init__(self, message, status="failed"):
        super().__init__(message)
        self.status = status


def require(ok, message):
    if not ok:
        raise AdmissionError(message)


def point(x):
    """Exact integer/rational/binary64 input, never decimalized floats."""
    if isinstance(x, (float, np.floating)):
        require(np.isfinite(x), "nonfinite scalar")
        n, d = float(x).as_integer_ratio()
        return IV.mpf(n) / IV.mpf(d)
    return IV.mpf(x)


def lo(x):
    return IV.make_mpf((x._mpi_[0], x._mpi_[0]))


def hi(x):
    return IV.make_mpf((x._mpi_[1], x._mpi_[1]))


def span(a, b):
    require(mpf_cmp(a._mpi_[0], b._mpi_[1]) <= 0, "empty interval")
    return IV.make_mpf((a._mpi_[0], b._mpi_[1]))


ZERO, ONE = point(0), point(1)
U = ONE / point(2**53)
NU = ONE / point(2**900)
LIMIT = point(2**20)
SOURCE_PATHS = [
    "<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T0015.00.npz",
    "<internal>/orbitals_c19_rs64_A_R2_tau8_ell4_avg_ext100_T0056.00.npz"]


def le(a, b):
    return mpf_cmp(a._mpi_[1], b._mpi_[0]) <= 0


def lt(a, b):
    return mpf_cmp(a._mpi_[1], b._mpi_[0]) < 0


def contains(a, x):
    return (mpf_cmp(a._mpi_[0], x._mpi_[0]) <= 0 and
            mpf_cmp(a._mpi_[1], x._mpi_[1]) >= 0)


def maximum(a, b):
    return span(a if le(lo(b), lo(a)) else b,
                a if le(hi(b), hi(a)) else b)


def intersection(a, b):
    lower = lo(a) if le(lo(b), lo(a)) else lo(b)
    upper = hi(a) if le(hi(a), hi(b)) else hi(b)
    return span(lower, upper)


def enclose_json(a):
    return {"outward_interval": str(a), "exact_dyadic_endpoints":
            [[int(v) for v in t] for t in a._mpi_]}


def unpack(a):
    return IV.make_mpf(tuple(tuple(int(v) for v in t)
                             for t in a["exact_dyadic_endpoints"]))


def gamma(n):
    v = point(int(n)) * U
    require(lt(v, ONE), "gamma outside domain")
    return v / (ONE - v)


def beta(k):
    return point(2) * gamma(2 * int(k) + 2)


def upper(a):
    return hi(a)


def norm_squared(a):
    a = np.asarray(a, dtype=np.complex128)
    require(all(v <= 4096 for v in a.shape), "array dimension exceeds domain")
    require(np.isfinite(a).all(), "nonfinite norm operand")
    require(np.all(np.abs(a.real) <= 2**20) and
            np.all(np.abs(a.imag) <= 2**20), "component exceeds domain")
    shat = float(np.sum(a.real * a.real + a.imag * a.imag, dtype=np.float64))
    require(np.isfinite(shat), "nonfinite norm reduction")
    g = gamma(4 * a.size + 4)
    lower = maximum(ZERO, (point(shat) - NU) / (ONE + g))
    upper_bound = (point(shat) + NU) / (ONE - g)
    result = span(lo(lower), hi(upper_bound))
    require(le(hi(IV.sqrt(result)), LIMIT), "Frobenius norm exceeds domain")
    return result


def norm(a):
    return IV.sqrt(norm_squared(a))


def product(a, b):
    norm(a); norm(b)
    return a @ b


def scaled(c, a):
    require(le(abs(point(c)), LIMIT), "coefficient exceeds domain")
    norm(a)
    return np.float64(c) * a


def added(a, b):
    norm(a); norm(b)
    return a + b


def digest_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def matrix_record(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sha256_shape_dtype_data": digest_array(a),
            "norm_squared": enclose_json(norm_squared(a))}


def admission_le(a, b, name):
    if le(a, b):
        return "pass"
    if lt(b, a):
        raise AdmissionError(name + ": disjoint violation")
    raise AdmissionError(name + ": interval overlap", "unresolved")


def recipe_admission(a):
    # Failure to establish this sufficient condition says nothing about
    # singularity or the existence of the exact polar frame.
    if not le(a, ONE/point(2)):
        raise AdmissionError("fixed near-identity recipe not admitted", "unresolved")
    return "pass"


def order_diagnostic(a, b):
    if le(a, b):
        return "ordered"
    if lt(b, a):
        raise AdmissionError("incompatible exact-domain order")
    return "compatible_overlap"


def joint_selected(frame, d, save_dir=None, tag="frame"):
    """Fixed degree-two selected joint polar action and all error terms."""
    frame = np.asarray(frame, dtype=np.complex128)
    require(frame.ndim == 2, "frame must be a matrix")
    m, q = frame.shape
    require(0 < d <= q <= m <= 4096, "invalid joint frame dimensions")
    nf = hi(norm(frame))
    ghat = product(frame.conj().T, frame)
    ng = hi(norm(ghat))
    # Two real floating operations per entry. Explicit triangle/conjugate
    # construction makes the represented result exactly Hermitian.
    gs0 = scaled(0.5, added(ghat, ghat.conj().T))
    gs = np.triu(gs0, 1)
    gs = gs + gs.conj().T
    idx = np.diag_indices(q)
    gs[idx] = gs0.diagonal().real
    ns = hi(norm(gs))
    hmat = gs.copy()
    hmat[idx] = added(gs.diagonal().real, -np.ones(q))
    nh = hi(norm(hmat))
    delta = (beta(m) * nf**2 + NU + gamma(2) * ng + NU +
             gamma(1) * (ns + IV.sqrt(point(q))) + NU)
    recipe_admission(nh + delta)
    f2 = IV.sqrt(ONE + nh + delta)
    tail = point(5) * nh**3 / (point(16) * (ONE - nh))
    tau = tail + delta / (point(2) * (ONE - nh - delta)**(point(3)/point(2)))
    selector = np.zeros((q, d), dtype=np.complex128)
    selector[np.arange(q-d, q), np.arange(d)] = 1
    amat = hmat[:, q-d:].copy()
    cmat = product(hmat, amat)
    lmat = scaled(-0.5, amat)
    kmat = scaled(0.375, cmat)
    r0 = added(selector, lmat)
    rmat = added(r0, kmat)
    ymat = product(frame, rmat)
    na, nc, nl, nk, nr0, nr = [hi(norm(a)) for a in
                              (amat, cmat, lmat, kmat, r0, rmat)]
    ec = beta(q) * nh * na + NU
    el = gamma(1) * na / point(2) + NU
    ek = gamma(1) * (point(3)/point(8)) * nc + NU
    e0 = gamma(1) * (IV.sqrt(point(d)) + nl) + NU
    e1 = gamma(1) * (nr0 + nk) + NU
    er = (point(3)/point(8))*ec + el + ek + e0 + e1
    ey = beta(q) * nf * nr + NU
    eta = ey + f2 * (er + tau)
    eblock = ey + f2 * (er + IV.sqrt(point(d))*tau)
    jmat = product(ymat.conj().T, ymat)
    jresid = jmat.copy()
    jresid[np.diag_indices(d)] = added(jmat.diagonal(), -np.ones(d))
    residual = norm(jresid)
    dot_error = beta(m) * hi(norm(ymat))**2 + NU
    subtract_error = gamma(1) * (hi(norm(jmat)) + IV.sqrt(point(d))) + NU
    allowed = point(2)*eblock + eblock**2 + dot_error + subtract_error
    admission_le(residual, allowed, "selected dust Gram consistency")
    arrays = {"F": frame, "Ghat": ghat, "Gs": gs, "H": hmat,
              "A": amat, "C": cmat, "L": lmat, "K": kmat,
              "R0": r0, "R": rmat, "Y": ymat, "J": jmat}
    rec = {"arrays": {k: matrix_record(v) for k, v in arrays.items()},
           "gram_skew_represented": matrix_record(ghat-ghat.conj().T),
           "bounds": {k: enclose_json(v) for k, v in
                      {"h": nh, "delta": delta, "f2": f2, "tail": tail,
                       "tau": tau, "eC": ec, "eL": el, "eK": ek,
                       "e0": e0, "e1": e1, "eR": er, "eY": ey,
                       "eta_per_column": eta, "e_selected_block": eblock,
                       "lambda_lower": ONE-nh-delta,
                       "lambda_upper": ONE+nh+delta,
                       "dust_gram_measured": residual,
                       "dust_gram_dot_error": dot_error,
                       "dust_gram_subtraction_error": subtract_error,
                       "dust_gram_allowed": allowed}.items()},
           "conditioning": "pass", "dust_gram_consistency": "pass"}
    if save_dir is not None:
        path = Path(save_dir) / (tag + "_selected.npz")
        np.savez(path, **{k: v for k, v in arrays.items()
                         if k in ("A", "C", "L", "K", "R0", "R", "Y", "J")})
        rec["selected_arrays"] = str(path)
    return ymat, eta, rec


def projected_probability(y, eta):
    s = norm_squared(y)
    amp = span(lo(maximum(ZERO, IV.sqrt(lo(s))-eta)),
               hi(IV.sqrt(hi(s))+eta))
    canonical = intersection(amp, span(ZERO, ONE))**2
    return canonical, {"represented_squared_norm": enclose_json(s),
                       "canonical_probability": enclose_json(canonical)}


def partitions(y, eta, mask):
    require(mask.dtype == bool and mask.shape == (y.size,), "bad component mask")
    inside, rin = projected_probability(y[mask], eta)
    outside, rout = projected_probability(y[~mask], eta)
    full, rfull = projected_probability(y, eta)
    require(contains(full, ONE), "canonical full norm excludes one")
    require(contains(inside+outside, ONE), "canonical partition excludes one")
    old_in, old_out = inside, outside
    inside = intersection(old_in, ONE-old_out)
    outside = intersection(old_out, ONE-old_in)
    return inside, outside, {"selected": rin, "complement": rout, "full": rfull,
                             "direct_partition_sum": enclose_json(old_in+old_out),
                             "selected_tightened": enclose_json(inside),
                             "complement_tightened": enclose_json(outside)}


def bound_function(p, eps):
    # Monotone endpoint construction; no dependency-sensitive subtraction.
    low = maximum(ZERO, IV.sqrt(lo(p))-IV.sqrt(hi(eps)))**2
    high = maximum(ZERO, IV.sqrt(hi(p))-IV.sqrt(lo(eps)))**2
    return span(lo(low), hi(high))


def labels_match(early, late):
    require(len(set(early)) == len(early), "duplicate early identifier")
    require(len(set(late)) == len(late), "duplicate late identifier")
    require(early == late, "inconsistent identifier/order map")


def weighted_totals(rows):
    labels = [tuple(r["id"]) for r in rows]
    labels_match(labels, labels)
    btotal, ntotal, nalt = ZERO, ZERO, ZERO
    for row in rows:
        k, j = row["id"]
        require(isinstance(k, int) and k > 0 and isinstance(j, int) and j >= 0,
                "invalid dust identifier")
        require(row["weight"] == 16*k, "incorrect channel weight")
        b, n, eps = [unpack(row[k]) for k in ("b", "n", "epsilon")]
        order_diagnostic(b, n)
        btotal += point(16*k)*b
        ntotal += point(16*k)*n
        nalt += point(16*k)*(ONE-eps)
    require(contains(ntotal-nalt, ZERO), "independent inside partition conflict")
    return btotal, ntotal, nalt


def compute_channel(early, late, d, pmask, qmask, k, save_dir=None, prefix=""):
    ye, ee, re = joint_selected(early, d, save_dir, prefix+"early")
    yl, el, rl = joint_selected(late, d, save_dir, prefix+"late")
    labels = [(int(k), int(j)) for j in range(d)]
    labels_match(labels, labels)
    rows = []
    for j in range(d):
        n, eps, pe = partitions(ye[:, j], ee, pmask)
        p, _, pl = partitions(yl[:, j], el, qmask)
        b = bound_function(p, eps)
        ordering = order_diagnostic(b, n)
        # Native-float centers are explicitly non-enclosing diagnostics only.
        ec = float(np.sum(np.abs(ye[~pmask, j])**2))
        pc = float(np.sum(np.abs(yl[qmask, j])**2))
        nc = float(np.sum(np.abs(ye[pmask, j])**2))
        bc = max(0., np.sqrt(pc)-np.sqrt(ec))**2
        rows.append({"id": [int(k), j], "weight": 16*int(k),
                     "p": enclose_json(p), "epsilon": enclose_json(eps),
                     "n": enclose_json(n), "b": enclose_json(b),
                     "b_le_inside": ordering, "early_partition": pe,
                     "late_partition": pl,
                     "center_diagnostic_only": {"p": pc, "epsilon": ec,
                                                "n": nc, "b": float(bc)}})
    return rows, {"early": re, "late": rl}, (ye, yl)


def masks(grid, inside_cut, far_cut):
    grid = np.asarray(grid)
    require(grid.ndim == 1 and np.isfinite(grid).all(), "bad radial grid")
    require(np.array_equal(grid, np.arange(1, grid.size+1)), "changed/nonmonotone grid")
    return (np.repeat(grid <= inside_cut, 2),
            np.repeat(grid > far_cut, 2))


def validate_payload(payload, nsite, kappas, dust_counts, clock, rs, dtype, sea_cols):
    """Same schema checks for small fixtures and actual read-only NPZ payloads."""
    keys = set(payload.keys())
    required = {"T", "rs", "R", "kappas", "weights"}
    required |= {f"{s}_{k}" for k in kappas for s in ("sea", "dust", "f")}
    require(required <= keys, "missing required archive key")
    framed = {v for v in keys if v.startswith(("sea_", "dust_", "f_"))}
    require(framed == required-{"T", "rs", "R", "kappas", "weights"},
            "unexpected frame/occupation channel")
    require(np.asarray(payload["T"]).shape == () and
            np.asarray(payload["rs"]).shape == (), "clock and rs must be scalar arrays")
    require(float(payload["T"]) == clock and float(payload["rs"]) == rs,
            "incorrect stored clock or rs")
    require(np.array_equal(payload["kappas"], np.array(kappas)), "channel roster conflict")
    require(np.array_equal(payload["weights"], 16*np.array(kappas)), "incorrect weights")
    require(np.asarray(payload["R"]).shape == (nsite,), "incorrect radial grid shape")
    masks(payload["R"], 0, 0)
    for k, d in zip(kappas, dust_counts):
        sea, dust, f = [np.asarray(payload[f"{s}_{k}"]) for s in ("sea", "dust", "f")]
        require(sea.shape == (2*nsite, sea_cols), "sea shape conflict")
        require(dust.shape == (2*nsite, d) and f.shape == (d,), "dust/occupation shape conflict")
        require(sea.dtype == dtype and dust.dtype == dtype, "frame dtype conflict")
        require(np.isfinite(sea).all() and np.isfinite(dust).all(), "nonfinite frame")
        require(np.array_equal(f, np.ones(d)), "nonunit dust occupations")
    return {"clock": clock, "rs": rs, "grid_sites": nsite,
            "kappas": list(kappas), "dust_counts": list(dust_counts), "status": "pass"}


def validate_roles(slices):
    require([v["path"] for v in slices] == SOURCE_PATHS,
            "exactly two ordered early/late source roles required")


def hashes(path):
    hs = {"sha256": hashlib.sha256(), "md5": hashlib.md5()}
    with open(path, "rb") as f:
        while data := f.read(8*1024*1024):
            for h in hs.values():
                h.update(data)
    return {k: h.hexdigest() for k, h in hs.items()}


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def run_identity(args):
    return {"code_sha256": hashes(__file__)["sha256"],
            "spec_sha256": hashes(args.spec)["sha256"],
            "inventory_sha256": hashes(args.inventory)["sha256"]}


def backend():
    return {"host": socket.gethostname(), "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__,
            "mpmath_interval_precision": IV.prec,
            "threads": threadpool_info(), "environment":
            {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "QI2_REMOTE_CPU")},
            "rounding_model": "ordinary CPU binary64 round-to-nearest; no reduced precision",
            "underflow_allowance": enclose_json(NU)}


def rounding_comparison(value, target):
    # target is a decimal string, used only in explicit post-freeze comparison.
    parts = target.split(".")
    places = len(parts[1]) if len(parts) == 2 else 0
    center = point(int(target.replace(".", ""))) / point(10**places)
    half = ONE / point(2*10**places)
    a, b = center-half, center+half
    if lt(a, value) and lt(value, b):
        status = "agreement_at_printed_precision"
    elif lt(value, a) or lt(b, value):
        status = "disagreement_at_printed_precision"
    else:
        status = "precision_unresolved"
    return {"target": target, "open_rounding_bin": enclose_json(span(lo(a), hi(b))),
            "status": status}


def lower_statement(value, threshold):
    if le(threshold, value):
        return "supported"
    if lt(value, threshold):
        return "not_supported"
    return "unresolved"


def run_controls(args):
    out = Path(args.output)
    checks, data = [], {}

    def known(name, value, expected):
        require(contains(value, expected), name+": exact answer not enclosed")
        checks.append(name)

    def rejects(name, fn, expected=None):
        try:
            fn()
        except AdmissionError as exc:
            if expected:
                require(exc.status == expected, name+": wrong rejection class")
            checks.append(name)
            data[name] = {"status": exc.status, "reason": str(exc)}
        else:
            raise AdmissionError(name+": deliberately bad case was accepted")

    def fixture(name, fe, fl, p, q, d=1, k=1):
        rows, diag, ys = compute_channel(fe, fl, d, np.array(p, bool),
                                         np.array(q, bool), k, out, name+"_")
        data[name] = {"rows": rows, "construction": diag}
        return rows, ys, diag

    e1 = np.array([[1.], [0.]], dtype=np.complex128)
    e2 = e1[::-1].copy()
    rzero, _, _ = fixture("zero", e1, e1, [1, 0], [0, 1])
    runit, _, _ = fixture("unit", e1, e2, [1, 0], [0, 1])
    for name, row, vals in [("zero", rzero[0], (0, 0, 0, 1)),
                             ("unit", runit[0], (0, 1, 1, 1))]:
        for key, value in zip(("epsilon", "p", "b", "n"), vals):
            known(name+"_"+key, unpack(row[key]), point(value))
        bt, nt, _ = weighted_totals([row])
        known(name+"_weighted_B", bt, point(16*vals[2]))
        known(name+"_weighted_N", nt, point(16))
    h4 = np.array([[1,1,1,1], [1,1,-1,-1], [1,-1,1,-1], [1,-1,-1,1]],
                  dtype=np.complex128)*0.5
    f = np.ones((4, 1), dtype=np.complex128)*0.5
    uf = h4 @ f
    rdy, _, _ = fixture("dyadic", f, uf, [1,1,1,0], [1,0,0,0], k=2)
    for key, value in zip(("epsilon", "p", "b", "n"),
                          (ONE/point(4), ONE, ONE/point(4), point(3)/point(4))):
        known("dyadic_"+key, unpack(rdy[0][key]), value)
    true_response = norm_squared((h4 @ (f*np.array([[1],[1],[1],[0]])))[[0]])
    known("dyadic_projected_response", true_response, point(9)/point(16))
    rsmall, _, _ = fixture("dyadic_smallP", f, uf, [1,0,0,0], [1,0,0,0])
    known("dyadic_smallP_b", unpack(rsmall[0]["b"]), point(7)/point(4)-IV.sqrt(point(3)))
    known("dyadic_smallP_n", unpack(rsmall[0]["n"]), ONE/point(4))
    combined = runit+rdy
    bt, nt, _ = weighted_totals(combined)
    known("weighted_B24", bt, point(24)); known("weighted_N40", nt, point(40))
    known("percopy_B", unpack(runit[0]["b"])+unpack(rdy[0]["b"]), point(5)/point(4))
    known("percopy_N", unpack(runit[0]["n"])+unpack(rdy[0]["n"]), point(7)/point(4))
    explicit_b, explicit_n = ZERO, ZERO
    for row in combined:
        for _ in range(row["weight"]):
            explicit_b += unpack(row["b"]); explicit_n += unpack(row["n"])
    known("explicit_copies_B", explicit_b, point(24)); known("explicit_copies_N", explicit_n, point(40))
    permuted = list(reversed(combined))
    labels_match([tuple(r["id"]) for r in permuted], [tuple(r["id"]) for r in permuted])
    known("common_permutation_B", weighted_totals(permuted)[0], point(24))
    rejects("inconsistent_labels", lambda: labels_match([(1,0),(2,0)], [(2,0),(1,0)]))
    rejects("duplicate_labels", lambda: weighted_totals(runit+runit))
    rejects("wrong_row_weight", lambda: weighted_totals([{**runit[0], "weight": 32}]))
    ri, _, _ = fixture("interference", f, uf, [1,1,0,0], [0,1,0,0])
    for key, value in zip(("p", "epsilon", "b", "n"),
                          (ZERO, ONE/point(2), ZERO, ONE/point(2))):
        known("interference_"+key, unpack(ri[0][key]), value)
    pf = f*np.array([[1],[1],[0],[0]])
    a, b = (h4 @ pf)[1,0], (h4 @ (f-pf))[1,0]
    known("inside_component", norm_squared(np.array([a])), ONE/point(4))
    known("outside_component", norm_squared(np.array([b])), ONE/point(4))
    known("cross_term", point(float(2*np.real(a*np.conj(b)))), -ONE/point(2))
    require(not contains(norm_squared(np.array([a])), ZERO), "wrong-order fixture failed to distinguish")
    require(not contains(unpack(ri[0]["p"]), ONE/point(2)), "additive ancestry fixture not distinguished")
    checks.extend(["wrong_order_distinguished", "additive_ancestry_distinguished"])
    # Q P U f is exactly zero; Q U P f above has weight 1/4.
    known("reversed_order_zero", norm_squared((uf*np.array([[1],[1],[0],[0]]))[[1]]), ZERO)
    for complex_k in (False, True):
        km = np.array([[0, 1j], [-1j, 0]] if complex_k else [[0,1],[1,0]], np.complex128)
        sm = np.eye(2, dtype=np.complex128)+km/256
        for phase in (False, True):
            polar = np.diag([1, 1j]) if phase else np.eye(2)
            frame = polar @ sm
            y, eta, diag = joint_selected(frame, 1, out, f"lowdin_{complex_k}_{phase}")
            exact_column = polar[:, 1:]
            diff = norm(y-exact_column)
            subtraction_error = gamma(1)*(hi(norm(y))+hi(norm(exact_column)))+NU
            admission_le(diff+subtraction_error, eta, "exact joint polar column enclosure")
            prob, _ = projected_probability(y[[0],0], eta)
            known(f"lowdin_probability_{complex_k}_{phase}", prob, ZERO)
            s = ONE/point(256)
            dustonly_first_magnitude = s/IV.sqrt(ONE+s**2)
            require(lt(eta, dustonly_first_magnitude), "dust-only normalization not distinguished")
            checks.append(f"dustonly_rejected_{complex_k}_{phase}")
            data[f"lowdin_{complex_k}_{phase}"] = {
                "construction": diag,
                "column_difference_represented": enclose_json(diff),
                "column_difference_subtraction_error": enclose_json(subtraction_error),
                "dustonly_exact_first_magnitude": enclose_json(dustonly_first_magnitude)}
    rejects("large_lowdin", lambda: joint_selected(np.array([[1,.5],[.5,1]],np.complex128),1), "unresolved")
    ident = np.eye(2, dtype=np.complex128)
    rb, _, _ = fixture("basis_original", ident, ident, [1,0], [1,0], d=2)
    c = np.float64(0.7071067811865476)  # exact represented scalar; polar is exact Hadamard
    had = c*np.array([[1,1],[1,-1]],np.complex128)
    rr, _, _ = fixture("basis_rotated", had, had, [1,0], [1,0], d=2)
    known("basis_original_B", sum((unpack(r["b"]) for r in rb), ZERO), ONE)
    known("basis_rotated_B", sum((unpack(r["b"]) for r in rr), ZERO), ZERO)
    for name, rows in (("original", rb), ("rotated", rr)):
        known("basis_"+name+"_N", sum((unpack(r["n"]) for r in rows), ZERO), ONE)
    for name, p, eps, expected in [("equal", ONE/point(4), ONE/point(4), ZERO),
                                   ("below", ONE/point(16), ONE/point(4), ZERO),
                                   ("eps_zero", ONE/point(4), ZERO, ONE/point(4)),
                                   ("eps_one", ONE/point(4), ONE, ZERO)]:
        known("branch_"+name, bound_function(p,eps), expected)
    routside, _, _ = fixture("outside_early", e2, e2, [1,0], [0,1])
    known("eps_one_inside", unpack(routside[0]["n"]), ZERO)
    known("eps_one_full_outside", unpack(routside[0]["epsilon"]), ONE)
    for cut in (2, 2.5):
        p, q = masks(np.arange(1,5), cut, 3)
        require(p.tolist() == [True]*4+[False]*4 and q.tolist() == [False]*6+[True]*2,
                "mask boundary mismatch")
        checks.append("mask_"+str(cut))
    # Same production schema helper with scaled fixture dimensions.
    payload = {"T": np.array(2.), "rs": np.array(1.), "R": np.arange(1,3),
               "kappas": np.array([1,2]), "weights": np.array([16,32]),
               "sea_1": np.eye(4,dtype=np.complex128)[:,:1],
               "sea_2": np.eye(4,dtype=np.complex128)[:,:1],
               "dust_1": np.eye(4,dtype=np.complex128)[:,1:2],
               "dust_2": np.empty((4,0),np.complex128),
               "f_1": np.ones(1), "f_2": np.ones(0)}
    def validate(v):
        return validate_payload(v,2,[1,2],[1,0],2.,1.,np.dtype('complex128'),1)
    validate(payload); checks.append("schema_valid")
    changes = {"missing_key": ("sea_1", None),
               "missing_column": ("dust_1", np.empty((4,0),np.complex128)),
               "nonfinite": ("dust_1", np.full((4,1),np.nan+0j)),
               "clock": ("T", np.array(3.)), "rs": ("rs", np.array(2.)),
               "clock_scalar_shape": ("T", np.array([2.])),
               "rs_scalar_shape": ("rs", np.array([1.])),
               "grid_order": ("R", np.array([2,1])),
               "grid_changed": ("R", np.array([1,3])),
               "dtype": ("dust_1", np.ones((4,1),np.complex64)),
               "occupation": ("f_1", np.array([.5])),
               "weight": ("weights", np.array([16,16])),
               "unexpected_occupied": ("dust_2", np.ones((4,1),np.complex128)),
               "unexpected_channel": ("dust_3", np.ones((4,1),np.complex128))}
    for name, (key, value) in changes.items():
        altered = dict(payload)
        if value is None:
            del altered[key]
        else:
            altered[key] = value
        rejects("schema_"+name, lambda v=altered: validate(v))
    rejects("zero_frame", lambda: joint_selected(np.zeros((2,1),np.complex128),1), "unresolved")
    rejects("rank_deficient", lambda: joint_selected(np.ones((2,2),np.complex128),1), "unresolved")
    rejects("recipe_boundary", lambda: recipe_admission(span(ZERO,ONE)), "unresolved")
    rejects("admission_boundary", lambda: admission_le(span(point(0),ONE), ONE/point(2),"test"), "unresolved")
    require(rounding_comparison(span(point(1),point(2)),"1")["status"] == "precision_unresolved",
            "rounding boundary promoted")
    require(rounding_comparison(point(3)/point(2),"1")["status"] == "precision_unresolved",
            "rounding endpoint promoted")
    require(rounding_comparison(ONE,"1")["status"] == "agreement_at_printed_precision", "rounding interior")
    require(rounding_comparison(point(3),"1")["status"] == "disagreement_at_printed_precision", "rounding disjoint")
    checks.append("rounding_three_branches_and_endpoint_touch")
    for name, value, expected in [("above",point(4),"supported"),
                                  ("equality",point(3),"supported"),
                                  ("below",point(2),"not_supported"),
                                  ("touch_from_below",span(point(2),point(3)),"unresolved"),
                                  ("overlap",span(point(2),point(4)),"unresolved")]:
        require(lower_statement(value,point(3)) == expected, "lower statement "+name)
        checks.append("lower_statement_"+name)
    roles = [{"path": path} for path in SOURCE_PATHS]
    validate_roles(roles); checks.append("two_source_roles")
    rejects("missing_source_role", lambda: validate_roles(roles[:1]))
    rejects("extra_source_role", lambda: validate_roles(roles+roles[:1]))
    rejects("reversed_source_roles", lambda: validate_roles(roles[::-1]))
    require(order_diagnostic(ZERO,ONE) == "ordered", "ordered diagnostic")
    require(order_diagnostic(span(ZERO,point(2)),ONE) == "compatible_overlap", "overlap diagnostic")
    rejects("incompatible_order", lambda: order_diagnostic(point(2),ONE))
    require(lt(ZERO, NU) and lt(ZERO, beta(1)*ZERO+NU) and
            lt(ZERO, gamma(1)*ZERO+NU), "underflow allowance omitted")
    checks.append("positive_underflow_zero_operations")
    result = {"status": "pass", "identity": run_identity(args), "backend": backend(),
              "checks": checks, "fixtures": data,
              "scope": "Manufactured controls only; no physical input contracted."}
    save_json(out/"controls.json", result)
    print(json.dumps({"status": "pass", "checks": len(checks), "output": str(out/"controls.json")}))


def run_physical(args):
    out = Path(args.output)
    controls = json.loads((out/"controls.json").read_text())
    require(controls["status"] == "pass" and controls["identity"] == run_identity(args),
            "missing or changed-code/spec/inventory known-answer pass")
    inventory = json.loads(Path(args.inventory).read_text())
    validate_roles(inventory["slices"])
    verification = {"identity": run_identity(args), "inputs": [], "backend": backend()}
    for item in inventory["slices"]:
        actual = hashes(item["path"])
        require(actual == item["hashes"], "raw input hash conflict")
        verification["inputs"].append({"path": item["path"], "hashes": actual})
    root_input = inventory["dense_root_input"]
    require(hashes(root_input["path"]) == root_input["hashes"], "dense-root hash conflict")
    root = json.loads(Path(root_input["path"]).read_text())["records"][120]
    require(root["T"] == 960 and root["T_over_rs"] == 15 and
            root["R_t"] == 51.76040530856484 and
            root["trapping_zeros"] == [7.04798671590984,51.76040530856484],
            "dense root record conflict")
    archives = []
    for item, clock, dtype in zip(inventory["slices"], (960.,3584.),
                                  (np.dtype("complex64"),np.dtype("complex128"))):
        with zipfile.ZipFile(item["path"]) as z:
            require(len(z.namelist()) == len(set(z.namelist())), "duplicate NPZ key")
        archive = np.load(item["path"], allow_pickle=False)
        validate_payload(archive,2048,list(range(1,11)),[5,4,3,2,2,0,0,0,0,0],
                         clock,64.,dtype,2048)
        archives.append(archive)
    pmask, qmask = masks(archives[0]["R"], root["R_t"], 16*64)
    require(int(pmask.sum()) == 102 and int(qmask.sum()) == 2048, "production mask counts")
    np.savez(out/"masks.npz", R=archives[0]["R"], early_inside=pmask, late_far=qmask)
    verification.update({"status": "pass", "root_record": root,
                         "mask_component_counts": [int(pmask.sum()),int(qmask.sum())],
                         "cut_neighbors": {"inside": [51,52], "far": [1024,1025]}})
    save_json(out/"raw_structure.json", verification)
    rows, channels = [], {}
    for k, d in enumerate((5,4,3,2,2),1):
        frames = [np.concatenate((a[f"sea_{k}"].astype(np.complex128),
                                  a[f"dust_{k}"].astype(np.complex128)),axis=1)
                  for a in archives]
        r, diag, _ = compute_channel(*frames,d,pmask,qmask,k,out,f"k{k}_")
        rows.extend(r)
        b, n, na = weighted_totals(r)
        channels[str(k)] = {"rows": r, "construction": diag,
                            "B": enclose_json(b), "N": enclose_json(n),
                            "N_from_complement": enclose_json(na)}
        save_json(out/f"channel_{k}.json", channels[str(k)])
        print(json.dumps({"channel_complete": k}), flush=True)
        del frames
    for a in archives:
        a.close()
    b, n, na = weighted_totals(rows)
    order = order_diagnostic(b,n)
    population_order = order_diagnostic(n,point(640))
    result = {"identity": run_identity(args), "rows": rows,
              "channels": {k: {x: v[x] for x in ("B","N","N_from_complement")}
                           for k,v in channels.items()},
              "B": enclose_json(b), "N": enclose_json(n),
              "N_from_complement": enclose_json(na),
              "B_le_N": order, "N_le_640": population_order,
              "raw_structure": "pass", "stored_functional_arithmetic": "pass",
              "physical_history_association": {"status": "conditional",
                  "unresolved": ["literal common-unitary history/column mixing association",
                                 "key-level sea physical-role identification",
                                 "ext100 interleaved exporter lineage",
                                 "serialization/evolution error relative to unrounded history"]},
              "printed_comparison": "not_performed; freeze own rows first",
              "center_diagnostic_only": {
                  key: float(sum(r["weight"]*r["center_diagnostic_only"][key] for r in rows))
                  for key in ("b","n")}}
    save_json(out/"own_rows_frozen.json", result)
    print(json.dumps({"stored_functional_arithmetic": "pass", "own_rows_frozen": True}))


def run_comparison(args):
    out = Path(args.output)
    path = out/"own_rows_frozen.json"
    frozen = json.loads(path.read_text())
    require(frozen["identity"] == run_identity(args), "changed frozen-run identity")
    require(frozen["stored_functional_arithmetic"] == "pass", "unadmitted own rows")
    b,n = unpack(frozen["B"]), unpack(frozen["N"])
    result = {"frozen_own_rows_sha256": hashes(path)["sha256"],
              "identity": run_identity(args),
              "B_conservative_3p0": lower_statement(b,point(3)),
              "N_display_611": rounding_comparison(n,"611"),
              "B_source_detailed": rounding_comparison(b,"3.0054591"),
              "N_source_detailed": rounding_comparison(n,"610.82"),
              "B": frozen["B"], "N": frozen["N"],
              "scope": "Stored functional only; physical-history association remains conditional."
             }
    save_json(out/"printed_comparison.json", result)
    print(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("controls","physical","compare"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--spec",required=True)
    parser.add_argument("--inventory",required=True)
    args = parser.parse_args()
    require(os.environ.get("QI2_REMOTE_CPU") == "1", "CPU environment required")
    require(socket.gethostname() == "gpu" and
            Path(__file__).resolve().is_relative_to(Path("<internal>/solve_b_bulk")),
            "execution restricted to authorized box bulk tree")
    Path(args.output).mkdir(parents=True,exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            {"controls":run_controls,"physical":run_physical,"compare":run_comparison}[args.mode](args)
    except AdmissionError as exc:
        save_json(Path(args.output)/(args.mode+"_failure.json"),
                  {"status":exc.status,"reason":str(exc),"identity":run_identity(args)})
        raise


if __name__ == "__main__":
    main()
