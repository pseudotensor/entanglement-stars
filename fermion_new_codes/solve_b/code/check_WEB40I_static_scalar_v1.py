#!/usr/bin/env python3
"""One predeclared X7 scalar attempt, SC21--27; no target values as inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

np = DOP853 = brentq = None
D = 16.0
PI = math.pi
HC_GRID = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.8)
LEVELS = (
    dict(name="A", radial=32, polar=12, azimuth=24, check_polar=8,
         r0=1e-5, rtol=1e-9, atol=1e-11, max_step=.02, peak_xtol=1e-8),
    dict(name="B", radial=48, polar=18, azimuth=36, check_polar=12,
         r0=5e-6, rtol=1e-11, atol=1e-13, max_step=.01, peak_xtol=3e-10),
)


class BudgetStop(RuntimeError):
    pass


class MemberStop(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


class EOS:
    def __init__(self, level):
        self.level = level
        x, w = np.polynomial.legendre.leggauss(level["radial"])
        self.s, self.w = (x + 1) / 2, w / 2
        t, wt = np.polynomial.legendre.leggauss(level["polar"])
        phi = 2 * PI * np.arange(level["azimuth"]) / level["azimuth"]
        st = np.sqrt(1 - t * t)[:, None]
        self.direction_sq = np.stack(np.broadcast_arrays(
            st ** 2 * np.cos(phi)[None, :] ** 2,
            st ** 2 * np.sin(phi)[None, :] ** 2,
            t[:, None] ** 2), axis=-1).reshape(-1, 3)
        self.angular_w = np.repeat(wt / (2 * level["azimuth"]), level["azimuth"])
        self.check_t, self.check_w = np.polynomial.legendre.leggauss(level["check_polar"])
        self.calls, self.max_zeta_K = 0, 0.0
        self.cache = {}

    def angular(self, k, zeta):
        if zeta == 0:
            return np.ones_like(np.asarray(k, dtype=float))
        a = np.asarray(k, dtype=float).reshape(-1)
        factors = 1 - (zeta * a[:, None, None]) ** 2 * self.direction_sq[None, :, :]
        if np.min(factors) <= 0:
            raise MemberStop("occupied sine-momentum domain failed")
        out = np.prod(factors, axis=2) ** -.5 @ self.angular_w
        return out.reshape(np.asarray(k).shape)

    def values(self, H, zeta):
        self.calls += 1
        if not math.isfinite(H):
            raise MemberStop("nonfinite EOS input")
        if H <= 0:
            return (0.0,) * 9
        U = math.exp(H)
        K = math.sqrt(math.expm1(2 * H))
        self.max_zeta_K = max(self.max_zeta_K, zeta * K)
        if zeta * K >= 1:
            raise MemberStop("zeta K >= 1")
        key = (H, zeta)
        if key in self.cache:
            return self.cache[key]
        k, wk = K * self.s, K * self.w
        e = np.sqrt(1 + k * k)
        delta = (K * K - k * k) / (U + e)
        A = self.angular(k, zeta)
        measure = D / (2 * PI * PI) * wk * k * k * A
        n, energy, pressure = float(measure.sum()), float(measure @ e), float(measure @ delta)
        nH = D / (2 * PI * PI) * K * U * U * float(self.angular(np.array(K), zeta))
        n1 = D * K ** 5 / (20 * PI * PI)
        e1 = D / (4 * PI * PI) * float((wk * k ** 4) @ e)
        p1 = D / (4 * PI * PI) * float((wk * k ** 4) @ delta)
        out = (n, energy, pressure, nH, U * nH, U * n, n1, e1, p1)
        if len(self.cache) >= 2048:
            self.cache.clear()
        self.cache[key] = out
        return out

    def native(self, H, F):
        if H <= 0:
            return (0.0,) * 4
        U, beta = math.exp(H), 1 - F
        K = math.sqrt(math.expm1(2 * H))
        k, wk = K * self.s, K * self.w
        ek2, k2 = 1 + k * k, k * k
        A = (1 - 4 * beta) * ek2 ** 2 - (2 * beta + 4 * beta ** 2) * ek2 * k2 - .6 * beta ** 2 * k2 ** 2
        DD = (-2 + 4 * beta) * ek2 + (2 / 3) * beta * k2
        Ab = -4 * ek2 ** 2 - (2 + 8 * beta) * ek2 * k2 - 1.2 * beta * k2 ** 2
        Db = 4 * ek2 + (2 / 3) * k2
        P = A / F ** 2 + DD / F + 1
        Pb = 2 * A / F ** 3 + Ab / F ** 2 + DD / F ** 2 + Db / F
        C = D / (20 * PI * PI)
        weights = wk * k2 / np.sqrt(ek2)
        psi = C * float(weights @ P)
        psi_beta = C * float(weights @ Pb)
        edge_k2, edge_e2 = K * K, U * U
        edge_A = (1 - 4 * beta) * edge_e2 ** 2 - (2 * beta + 4 * beta ** 2) * edge_e2 * edge_k2 - .6 * beta ** 2 * edge_k2 ** 2
        edge_D = (-2 + 4 * beta) * edge_e2 + (2 / 3) * beta * edge_k2
        psi_U = C * K * (edge_A / F ** 2 + edge_D / F + 1)
        return psi_U, U * psi_U - psi, psi + 2 * F * psi_beta, psi

    def direct_boost(self, K, b):
        k, wk = (K * self.s)[:, None], K * self.w
        t, wt = self.check_t[None, :], self.check_w / 2
        ek = np.sqrt(1 + k * k)
        gamma = 1 / math.sqrt(1 - b * b)
        energy = gamma * (ek + b * k * t)
        pr = gamma * (k * t + b * ek)
        p2 = pr * pr + k * k * (1 - t * t)
        chi1 = p2 * p2 / (10 * energy) - .4 * b * pr * p2
        angular = ((energy / ek) * chi1) @ wt
        return D / (2 * PI * PI) * float((wk * (K * self.s) ** 2) @ angular)


class Runner:
    def __init__(self, out, started, source, source_sha):
        self.out, self.started = out, started
        self.raw = out / "raw"
        self.raw.mkdir(exist_ok=False)
        self.record = dict(status="started", scope="SC21--27 empirical scalar control only",
                           source_sha256=source_sha, code_sha256=digest(__file__),
                           source_path=str(source), levels=LEVELS, central_grid=HC_GRID,
                           limits=dict(integrations=160, total_seconds=600, soft_seconds=580,
                                       threads=4, address_space_bytes=4 * 1024 ** 3),
                           units=dict(d=16, scalar_zeta=.2, finite_energy_factor=200,
                                      finite_count_factor=1000),
                           source_checks=[], members=[], peaks={}, responses={},
                           criteria={}, incomplete=[], raw_directory=str(self.raw))
        self.count = 0
        self.member = None
        self.trace_r, self.trace_y, self.trace_profiles = [], [], []
        self.cache = {}
        self.checkpoint()

    def guard(self):
        if time.monotonic() - self.started >= 580:
            raise BudgetStop("580-second graceful deadline")

    def checkpoint(self):
        self.record["integrations_started"] = self.count
        self.record["elapsed_seconds"] = time.monotonic() - self.started
        self.record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(self.out / "summary.json", self.record)

    def flush_trace(self):
        if self.member is not None and self.trace_r:
            write_npz(self.raw / (self.member["id"] + ".npz"),
                      r=np.array(self.trace_r), state=np.array(self.trace_y),
                      state_columns=np.array(self.member["state_columns"]),
                      profiles=np.array(self.trace_profiles),
                      profile_columns=np.array(["F", "n", "e", "p", "n_H", "e_H", "p_H",
                          "scalar_n_eta", "scalar_e_eta", "scalar_p_eta",
                          "native_n_eta", "native_e_eta", "native_p_effective_eta", "native_Psi"]))

    def sources(self, eos_a, eos_b):
        for eos in (eos_a, eos_b):
            write_npz(self.raw / ("quadrature_" + eos.level["name"] + ".npz"),
                      radial_unit_nodes=eos.s, radial_unit_weights=eos.w,
                      direction_sq=eos.direction_sq, angular_weights=eos.angular_w,
                      boost_polar_nodes=eos.check_t, boost_polar_weights=eos.check_w)
        for zeta in (0., .2):
            for H in (.05, .2, .4, .8):
                self.guard()
                a, b = eos_a.values(H, zeta)[:6], eos_b.values(H, zeta)[:6]
                diffs = [abs(x - y) for x, y in zip(a, b)]
                flags = [q <= 1e-9 * (1 + abs(y)) for q, y in zip(diffs, b)]
                self.record["source_checks"].append(dict(kind="SC21", zeta=zeta, H=H,
                    fields=["n", "e", "p", "n_H", "e_H", "p_H"], A=a, B=b,
                    differences=diffs, flags=flags, passed=all(flags)))
        for b in (0., .25, .5):
            for K in (.25, 1.):
                for eos in (eos_a, eos_b):
                    self.guard()
                    H = .5 * math.log1p(K * K)
                    formula = eos.native(H, 1 - b * b)[3]
                    direct = eos.direct_boost(K, b)
                    difference = abs(formula - direct)
                    self.record["source_checks"].append(dict(kind="SC10_SC8", level=eos.level["name"],
                        b=b, K=K, formula=formula, direct=direct, difference=difference,
                        passed=difference <= 1e-9 * (1 + abs(formula))))
        self.record["criteria"]["source_checks"] = all(x["passed"] for x in self.record["source_checks"])
        self.checkpoint()
        if not self.record["criteria"]["source_checks"]:
            raise MemberStop("pre-outcome source criterion failed; no stellar integration")

    def shoot(self, eos, Hc, zeta, response=False):
        key = (eos.level["name"], float(Hc), zeta, response)
        if key in self.cache:
            return self.cache[key]
        self.guard()
        if self.count >= 160:
            raise BudgetStop("160 stellar integration cap")
        self.count += 1
        label = f'{self.count:03d}_{eos.level["name"]}_{"response" if response else "scalar"}'
        columns = ["M", "H", "B", "M_Hc", "H_Hc", "B_Hc"]
        if response:
            columns += ["scalar_M_eta", "scalar_H_eta", "scalar_B_eta",
                        "native_M_eta", "native_H_eta", "native_B_eta",
                        "scalar_envelope_integral", "native_envelope_integral"]
        member = dict(id=label, level=eos.level["name"], Hc=Hc, zeta=zeta,
                      response=response, status="running", state_columns=columns,
                      F_min=1., max_zeta_K=0., rhs_calls=0,
                      raw_npz="raw/" + label + ".npz")
        self.record["members"].append(member)
        self.member, self.trace_r, self.trace_y, self.trace_profiles = member, [], [], []
        start = time.monotonic()
        r0 = eos.level["r0"]
        n, e, p, nH, eH, pH, ns, es, ps = eos.values(Hc, zeta)
        a, c = 4 * PI / 3, 2 * PI / 3
        y = [a * e * r0 ** 3, Hc - c * (e + 3 * p) * r0 ** 2, a * n * r0 ** 3,
             a * eH * r0 ** 3, 1 - c * (eH + 3 * pH) * r0 ** 2, a * nH * r0 ** 3]
        if response:
            nn, en, pn, psi = eos.native(Hc, 1.)
            y += [a * es * r0 ** 3, -c * (es + 3 * ps) * r0 ** 2, a * ns * r0 ** 3,
                  a * en * r0 ** 3, -c * (en + 3 * pn) * r0 ** 2, a * nn * r0 ** 3,
                  a * ps / math.exp(Hc) * r0 ** 3, a * psi / math.exp(Hc) * r0 ** 3]
        y = np.array(y, dtype=float)
        member["initial_state"] = y.tolist()

        def rhs(r, state):
            self.guard()
            member["rhs_calls"] += 1
            M, H, _ = state[:3]
            F = 1 - 2 * M / r
            member["F_min"] = min(member["F_min"], float(F))
            if F <= .05 or not np.all(np.isfinite(state)):
                raise MemberStop("F <= .05 or nonfinite ODE state")
            values = eos.values(float(H), zeta)
            n, energy, pressure, nH, eH, pH, ns, es, ps = values
            member["max_zeta_K"] = max(member["max_zeta_K"],
                zeta * math.sqrt(math.expm1(2 * max(0., float(H)))))
            A = M + 4 * PI * r ** 3 * pressure
            J = np.array([[0., 4 * PI * r * r * eH, 0.],
                          [-1 / (r * r * F) - 2 * A / (r ** 3 * F * F),
                           -4 * PI * r * pH / F, 0.],
                          [4 * PI * r * n / F ** 1.5,
                           4 * PI * r * r * nH / math.sqrt(F), 0.]])
            out = np.zeros_like(state)
            out[:3] = (4 * PI * r * r * energy, -A / (r * r * F),
                       4 * PI * r * r * n / math.sqrt(F))
            out[3:6] = J @ state[3:6]
            if response:
                nn, en, pn, psi = eos.native(float(H), float(F))
                for offset, ne, ee, pe in ((6, ns, es, ps), (9, nn, en, pn)):
                    out[offset:offset + 3] = J @ state[offset:offset + 3] + np.array(
                        [4 * PI * r * r * ee, -4 * PI * r * pe / F,
                         4 * PI * r * r * ne / math.sqrt(F)])
                U = math.exp(max(0., float(H)))
                out[12:14] = (4 * PI * r * r * ps / (U * math.sqrt(F)),
                              4 * PI * r * r * psi / (U * math.sqrt(F)))
            if not np.all(np.isfinite(out)):
                raise MemberStop("nonfinite ODE derivative")
            member["last_rhs_r"], member["last_rhs_state"] = r, state.tolist()
            return out

        def append_trace(r, state):
            F = 1 - 2 * float(state[0]) / r
            vals = eos.values(float(state[1]), zeta)
            native = eos.native(float(state[1]), F) if response else (0.,) * 4
            self.trace_r.append(r)
            self.trace_y.append(state.copy())
            self.trace_profiles.append((F, *vals, *native))

        append_trace(r0, y)
        self.flush_trace()
        self.checkpoint()
        last_flush = time.monotonic()
        try:
            solver = DOP853(rhs, r0, y, 16., rtol=eos.level["rtol"],
                            atol=eos.level["atol"], max_step=eos.level["max_step"])
            surface = None
            while solver.status == "running":
                self.guard()
                solver.step()
                if solver.status == "failed":
                    raise MemberStop("DOP853 failure")
                if solver.y[1] <= 0:
                    dense = solver.dense_output()
                    R = brentq(lambda r: float(dense(r)[1]), solver.t_old, solver.t,
                               xtol=1e-14, rtol=4 * np.finfo(float).eps)
                    surface = dense(R)
                    append_trace(R, surface)
                    member["event_bracket"] = [solver.t_old, solver.t]
                    break
                append_trace(solver.t, solver.y)
                if time.monotonic() - last_flush >= 1:
                    self.flush_trace()
                    self.checkpoint()
                    last_flush = time.monotonic()
            if surface is None:
                raise MemberStop("first H=0 event absent by r=16")
            M, H, B = map(float, surface[:3])
            F = 1 - 2 * M / R
            member.update(status="complete", R=R, M=M, B=B, H_surface=H,
                          mu_infinity=math.sqrt(F), F_surface=F,
                          M_Hc=float(surface[3]), B_Hc=float(surface[5]),
                          steps=len(self.trace_r) - 1, end_state=surface.tolist())
            if response:
                Hprime = -M / (R * R * F)
                responses = {}
                for name, offset, iq in (("scalar", 6, 12), ("native", 9, 13)):
                    ZM, ZH, ZB = map(float, surface[offset:offset + 3])
                    envelope = ZM - math.sqrt(F) * ZB
                    integral = float(surface[iq])
                    rdiag = {"scalar" if name == "scalar" else "mean":
                             -(ZH - (0. if name == "scalar" else .3) * (1 - F) ** 2 / F ** 2) / Hprime}
                    if name == "native":
                        rdiag["union"] = -(ZH - (1 / 6) * (1 - F) ** 2 / F ** 2) / Hprime
                    responses[name] = dict(M_eta=ZM, H_eta=ZH, B_eta=ZB,
                        fixed_count_envelope=envelope, integral=integral,
                        envelope_residual=envelope + math.sqrt(F) * integral,
                        fixed_label_radius_eta=rdiag,
                        continued_peak_mass_count_only=True)
                member["responses"] = responses
            self.cache[key] = member
            return member
        except BaseException as exc:
            member.update(status="incomplete", reason=f"{type(exc).__name__}: {exc}")
            self.record["incomplete"].append(dict(id=label, reason=member["reason"]))
            raise
        finally:
            member["wall_seconds"] = time.monotonic() - start
            member["eos_calls_cumulative"] = eos.calls
            self.flush_trace()
            self.checkpoint()
            self.member = None

    def locate(self, eos, zeta):
        samples = []
        for Hc in HC_GRID:
            try:
                m = self.shoot(eos, Hc, zeta)
                samples.append(dict(Hc=Hc, id=m["id"], derivative=m["M_Hc"]))
            except MemberStop as exc:
                samples.append(dict(Hc=Hc, incomplete=str(exc)))
        result = dict(samples=samples, status="incomplete")
        key = eos.level["name"] + ("_cone" if zeta == 0 else "_sine")
        self.record["peaks"][key] = result
        self.checkpoint()
        tol = 1e-8 if eos.level["name"] == "A" else 1e-10
        brackets = [(i, samples[i]["Hc"], samples[i + 1]["Hc"])
                    for i in range(len(samples) - 1)
                    if samples[i].get("derivative", 0) > tol
                    and samples[i + 1].get("derivative", 0) < -tol]
        result["resolved_brackets"] = brackets
        if not brackets:
            result["reason"] = "no resolved positive-to-negative bracket in fixed list"
            return result
        idx, lo, hi = brackets[0]
        if any(s.get("derivative", 0) <= tol for s in samples[:idx + 1]):
            result["reason"] = "unresolved or nonpositive preceding derivative"
            return result
        root_history = []

        def derivative(Hc):
            m = self.shoot(eos, Hc, zeta)
            root_history.append(dict(Hc=Hc, derivative=m["M_Hc"], id=m["id"]))
            return m["M_Hc"]

        try:
            hc = brentq(derivative, lo, hi, xtol=eos.level["peak_xtol"],
                        rtol=4 * np.finfo(float).eps, maxiter=40)
            peak = self.shoot(eos, hc, zeta)
            curvatures = []
            for shift in (.001, .0005):
                minus = self.shoot(eos, hc - shift, zeta)
                plus = self.shoot(eos, hc + shift, zeta)
                curvatures.append((plus["M_Hc"] - minus["M_Hc"]) / (2 * shift))
            flags = dict(mass_derivative=abs(peak["M_Hc"]) <= tol,
                count_derivative=abs(peak["B_Hc"]) <= 1e-8 * (1 + abs(peak["B"])),
                curvature_negative=all(x < 0 for x in curvatures),
                curvature_agreement=abs(curvatures[0] - curvatures[1]) < .1 * min(map(abs, curvatures)))
            result.update(status="complete", label="first resolved/located peak, not certified first/global",
                Hc=hc, member_id=peak["id"], root_history=root_history,
                M=peak["M"], B=peak["B"], R=peak["R"], mu_infinity=peak["mu_infinity"],
                M_Hc=peak["M_Hc"], B_Hc=peak["B_Hc"], curvature_estimates=curvatures,
                flags=flags, all_local_criteria=all(flags.values()))
            if zeta == .2:
                result["Ea"], result["N_B"] = 200 * peak["M"], 1000 * peak["B"]
        except MemberStop as exc:
            result.update(reason=str(exc), root_history=root_history)
        self.checkpoint()
        return result

    def run(self):
        eos_a, eos_b = [EOS(level) for level in LEVELS]
        self.sources(eos_a, eos_b)
        for eos in (eos_a, eos_b):
            for zeta in (0., .2):
                self.locate(eos, zeta)
            cone = self.record["peaks"][eos.level["name"] + "_cone"]
            if cone["status"] == "complete":
                try:
                    m = self.shoot(eos, cone["Hc"], 0., response=True)
                    self.record["responses"][eos.level["name"]] = dict(member_id=m["id"],
                        Hc=cone["Hc"], values=m["responses"], base_M=m["M"], base_B=m["B"],
                        M_Hc=m["M_Hc"], B_Hc=m["B_Hc"],
                        base_flags=dict(mass_derivative=abs(m["M_Hc"]) <=
                            (1e-8 if eos.level["name"] == "A" else 1e-10),
                            count_derivative=abs(m["B_Hc"]) <= 1e-8 * (1 + abs(m["B"]))))
                except MemberStop as exc:
                    self.record["responses"][eos.level["name"]] = dict(incomplete=str(exc))
                self.checkpoint()
        self.compare()

    def compare(self):
        comparisons = {}
        complete = all(p["status"] == "complete" for p in self.record["peaks"].values())
        self.record["criteria"]["all_four_peaks_complete"] = complete
        self.record["criteria"]["all_local_peak_criteria"] = complete and all(
            p["all_local_criteria"] for p in self.record["peaks"].values())
        for curve in ("cone", "sine"):
            a, b = [self.record["peaks"].get(level + "_" + curve, {}) for level in ("A", "B")]
            if a.get("status") == "complete" and b.get("status") == "complete":
                delta = {k: abs(a[k] - b[k]) for k in ("Hc", "M", "B", "R", "mu_infinity")}
                comparisons[curve] = dict(differences=delta, passed=all(v < 1e-4 for v in delta.values()))
            else:
                comparisons[curve] = dict(passed=False, incomplete=True)
        a, b = [self.record["peaks"].get(level + "_sine", {}) for level in ("A", "B")]
        if a.get("status") == "complete" and b.get("status") == "complete":
            comparisons["finite_pair"] = dict(Ea_difference=abs(a["Ea"] - b["Ea"]),
                N_B_difference=abs(a["N_B"] - b["N_B"]),
                passed=abs(a["Ea"] - b["Ea"]) < .001 and abs(a["N_B"] - b["N_B"]) < .01)
        ra, rb = [self.record["responses"].get(level, {}) for level in ("A", "B")]
        responses_complete = "values" in ra and "values" in rb
        self.record["criteria"]["responses_complete"] = responses_complete
        if responses_complete:
            for mode in ("scalar", "native"):
                a, b = ra["values"][mode], rb["values"][mode]
                fields = ("M_eta", "H_eta", "B_eta", "fixed_count_envelope", "integral")
                delta = {k: abs(a[k] - b[k]) for k in fields}
                signs = {k: bool(a[k] * b[k] > 0 and min(abs(a[k]), abs(b[k])) > 10 * delta[k])
                         for k in ("M_eta", "B_eta", "fixed_count_envelope")}
                radius_delta = {k: abs(a["fixed_label_radius_eta"][k] - b["fixed_label_radius_eta"][k])
                                for k in a["fixed_label_radius_eta"]}
                comparisons[mode + "_response"] = dict(differences=delta,
                    fixed_label_radius_differences=radius_delta,
                    refinement_passed=all(v < 1e-4 for v in (*delta.values(), *radius_delta.values())),
                    empirical_sign_flags=signs,
                    augmented_base_flags={"A": ra["base_flags"], "B": rb["base_flags"]},
                    envelope_flags=[abs(x["envelope_residual"]) <= 1e-6 for x in (a, b)])
        self.record["comparisons"] = comparisons
        gates = self.record["criteria"]
        def curve_local(curve):
            peaks = [self.record["peaks"].get(level + "_" + curve, {}) for level in ("A", "B")]
            return bool(gates["source_checks"] and all(
                p.get("status") == "complete" and p.get("all_local_criteria") for p in peaks))

        self.record["claim_criteria"] = dict(
            finite_scalar_located_peak=bool(curve_local("sine") and comparisons.get("sine", {}).get("passed")
                                            and comparisons.get("finite_pair", {}).get("passed")))
        for mode in ("scalar", "native"):
            c = comparisons.get(mode + "_response", {})
            self.record["claim_criteria"][mode + "_continued_peak_response"] = bool(
                curve_local("cone") and comparisons.get("cone", {}).get("passed")
                and c.get("refinement_passed") and all(c.get("envelope_flags", [False]))
                and all(all(c.get("augmented_base_flags", {}).get(level, {"missing": False}).values())
                        for level in ("A", "B"))
                and all(c.get("empirical_sign_flags", {"missing": False}).values()))
        self.record["status"] = "finite_attempt_complete"
        self.record["numerical_scope"] = "Empirical checks only; no interval, finite-native149, nonlinear3.9 or orbital maximum"
        self.checkpoint()


def main():
    global np, DOP853, brentq
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--outer-start", type=float, required=True)
    args = ap.parse_args()
    if digest(args.source) != args.source_sha256:
        raise SystemExit("source identity mismatch")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or len(os.sched_getaffinity(0)) > 4:
        raise SystemExit("nice/CPU-affinity enforcement missing")
    address_limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    if address_limit < 0 or address_limit > 4 * 1024 ** 3:
        raise SystemExit("4 GiB address-space enforcement missing")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("CUDA must be disabled")
    out = args.output_dir
    if not out.is_dir() or (out / "summary.json").exists():
        raise SystemExit("fresh launcher-owned output required")

    def stop(signum, frame):
        raise BudgetStop(f"graceful signal {signum}")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    remaining = 580 - (time.monotonic() - args.outer_start)
    if remaining <= 0:
        raise SystemExit("outer budget expired")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    runner = None
    try:
        import numpy as np
        from scipy.integrate import DOP853
        from scipy.optimize import brentq
        runner = Runner(out, args.outer_start, args.source, args.source_sha256)
        runner.record["library_versions"] = dict(numpy=np.__version__)
        runner.run()
        return 0
    except BaseException as exc:
        if runner is not None:
            runner.record["status"] = "incomplete"
            runner.record["reason"] = f"{type(exc).__name__}: {exc}"
            if runner.member is not None and runner.member.get("status") == "running":
                runner.member.update(status="incomplete", reason=runner.record["reason"])
        else:
            write_json(out / "summary.json", dict(status="startup_incomplete", reason=f"{type(exc).__name__}: {exc}"))
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        if runner is not None:
            runner.flush_trace()
            runner.checkpoint()


if __name__ == "__main__":
    sys.exit(main())
