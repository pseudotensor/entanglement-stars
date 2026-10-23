#!/usr/bin/env python3
"""Own dense implementation of the agreed, fixed Computation C fixture.

The original N/A pair must inspect this actual source before execution.
Scientific execution is restricted to the named box CPU and one bounded attempt.
No foreign numerical source or result is imported. Raw data remain on the box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

CHANNELS = (("H", .2, 1, 16), ("L1", 0., 1, 16),
            ("L2", 0., 2, 32), ("L3", 0., 3, 48))
DIM, NSITES, RANK, THREADS = 512, 256, 256, 8
SOURCE_TOL, READOUT_TOL, CLIP_TOL = 1e-12, 1e-10, 1e-12
DEFINITION_SELECTION_SHA = "6a3813bbe75a6414973a165defbdaa0c0c213c75b44ca5a13dd3da43a9e3f908"
NP = LA = None
STOP = False


class Incomplete(RuntimeError):
    pass


def stop_requested(signum, frame):
    global STOP
    STOP = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if NP is not None and isinstance(obj, NP.generic):
        return json_safe(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Runtime:
    def __init__(self, args):
        self.args = args
        self.run = args.output_dir
        self.raw = self.run / "raw"
        self.raw.mkdir()
        self.summary = dict(status="incomplete", complete=False,
            purpose="Computation C dense prescribed two-field fixture control",
            dimensions=dict(sites=256, components=2, records=41,
                            propagations=40, entropy_checkpoints=11),
            definition_selection_sha256=DEFINITION_SELECTION_SHA,
            known_common_input_numbers=True, foreign_terminal_comparison=False,
            dtype="complex128", thresholds=dict(source=SOURCE_TOL,
                energy_current_entropy=READOUT_TOL, clipping=CLIP_TOL,
                matrix_scaled_Frobenius=1e-12, state_Gram=1e-10),
            channels={}, weighted_rows=[], weighted_entropy=[],
            active_operation="initialization", all_controls_passed=False,
            arrays_box_only=True, thread_guard_samples=[],
            maximum_sampled_process_threads=0)
        self.current_checks = {}
        self.failed = []
        self.active_channel = None
        self.active_record = None

    def persist(self):
        self.summary.update(elapsed_seconds=time.monotonic() - self.args.outer_start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            active_channel=self.active_channel, active_record=self.active_record)
        atomic_json(self.run / "summary.json", self.summary)

    def guard(self, operation, dense=False):
        self.summary["active_operation"] = operation
        threads = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                           if line.startswith("Threads:")))
        self.summary["thread_guard_samples"].append(dict(
            elapsed_seconds=time.monotonic()-self.args.outer_start,
            operation=operation, process_threads=threads))
        self.summary["maximum_sampled_process_threads"] = max(
            self.summary["maximum_sampled_process_threads"], threads)
        self.persist()
        if threads > 16:
            raise Incomplete("Aggregate process thread count exceeds 16 at a guard sample")
        elapsed = time.monotonic() - self.args.outer_start
        if STOP or elapsed >= 1740 or (dense and elapsed >= 1680):
            raise Incomplete("Earlier atomic-save deadline; no further dense operation")
        memory = dict(s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())
        if int(memory["MemAvailable"].split()[0]) * 1024 < 64 * 2**30:
            raise Incomplete("MemAvailable reserve below 64 GiB")
        if shutil.disk_usage("/").free < 100 * 2**30:
            raise Incomplete("Root-disk reserve below 100 GiB")
        if shutil.disk_usage("/data").free < 2 * 2**30:
            raise Incomplete("Insufficient raw-result disk reserve")

    def metric(self, name, value, limit, strict=False):
        value = float(value)
        good = math.isfinite(value) and (value < limit if strict else value <= limit)
        self.current_checks[name] = dict(value=value, limit=limit, passed=good,
                                         strict=bool(strict))
        if not good:
            self.failed.append(dict(channel=self.active_channel,
                                    record=self.active_record, check=name,
                                    value=value, limit=limit))
        return good

    def condition(self, name, good):
        return self.metric(name, 0 if good else 1, 0)

    def abort_failed(self):
        if self.failed:
            self.summary["failed_controls"] = self.failed
            raise Incomplete("A predeclared control failed; preserve data and do not retry")

    def save_arrays(self, relative, **arrays):
        target = self.raw / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        temporary = target.with_suffix(".npz.tmp")
        for name, value in arrays.items():
            value = NP.asarray(value)
            if value.dtype.hasobject:
                raise TypeError("Object arrays are not permitted: " + name)
            if value.dtype.kind in "fc" and not NP.isfinite(value).all():
                self.condition("finite_" + name, False)
        with temporary.open("xb") as stream:
            NP.savez(stream, **arrays)
        temporary.replace(target)
        return str(target.relative_to(self.run))

    def eigh(self, matrix, label):
        self.guard(label, dense=True)
        return LA.eigh(matrix, driver="evd", overwrite_a=False, check_finite=True)

    def eigvalsh(self, matrix, label):
        self.guard(label, dense=True)
        return LA.eigh(matrix, eigvals_only=True, driver="evd", check_finite=True)

    def singular_values(self, matrix, label):
        self.guard(label, dense=True)
        return LA.svd(matrix, compute_uv=False, overwrite_a=False,
                      check_finite=True, lapack_driver="gesdd")


def maxabs(a):
    return float(NP.max(NP.abs(a), initial=0))


def herm(a):
    return (a + a.conj().T) * .5


def gram_defect(a):
    return maxabs(a.conj().T @ a - NP.eye(a.shape[1]))


def eigensystem_checks(rt, matrix, values, vectors, label, raw_matrix=None):
    failures_before = len(rt.failed)
    scale = max(1., float(LA.norm(matrix, "fro")))
    rt.metric(label + "_eigen_residual",
        float(LA.norm(matrix @ vectors-vectors*values[None, :], "fro")), 1e-12*scale)
    rt.metric(label + "_eigenbasis_Gram", gram_defect(vectors), 1e-12, strict=True)
    original = matrix if raw_matrix is None else raw_matrix
    rt.metric(label + "_Hermitian_defect",
        float(LA.norm(original-original.conj().T, "fro")), 1e-12*scale)
    if len(rt.failed) > failures_before:
        # This also retains a failed basis in an unadopted wider candidate.
        path = rt.save_arrays(
            f"{rt.active_channel}/failed_eigensystem_{rt.active_record:04d}_{label}.npz",
            matrix=matrix, original_matrix=original, eigenvalues=values, eigenvectors=vectors)
        for failure in rt.failed[failures_before:]:
            failure["raw_file"] = path


def comparison_checks(rt, label, left, right, absolute_tolerance):
    difference = maxabs(NP.asarray(left)-NP.asarray(right))
    scale = max(1., maxabs(NP.asarray(left)), maxabs(NP.asarray(right)))
    rt.metric(label, difference, absolute_tolerance)
    rt.metric(label + "_relative", difference/scale, absolute_tolerance/scale)
    rt.current_checks[label + "_relative"]["normalization_scale"] = scale
    return dict(absolute_difference=difference, normalization_scale=scale,
                scale_relative_difference=difference/scale)


def fields(k):
    radius = NP.arange(1, 257, dtype=float)
    boundary = 80. - 1.2 * k
    lapse = 1. - .25 * NP.maximum(1. - (radius / boundary) ** 2, 0.)
    flow = NP.where(radius < boundary,
                    -NP.sqrt(48. / boundary ** 3) * radius,
                    -NP.sqrt(48. / radius))
    return lapse, flow, boundary


def operators(mass, kappa):
    sx = NP.array([[0, 1], [1, 0]], dtype=complex)
    sy = NP.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = NP.diag([1., -1.])
    base = NP.zeros((DIM, DIM), dtype=complex)
    current = NP.zeros_like(base)
    symmetry = NP.zeros_like(base)
    for n in range(NSITES):
        ix = slice(2*n, 2*n+2)
        base[ix, ix] = mass * sz + (kappa / (n+1)) * sy
        symmetry[ix, ix] = (-1)**n * sy
        if n < NSITES - 1:
            iy = slice(2*n+2, 2*n+4)
            base[ix, iy] = -.5j * sx
            base[iy, ix] = .5j * sx
        if n < NSITES - 2:
            iy = slice(2*n+4, 2*n+6)
            current[ix, iy] = -.25j * NP.eye(2)
            current[iy, ix] = .25j * NP.eye(2)
    return base, current, symmetry


def clocked(base, current, lapse, flow):
    nn, vv = NP.repeat(lapse, 2), NP.repeat(flow, 2)
    return .5 * (nn[:, None] + nn[None, :]) * base + \
           .5 * (vv[:, None] + vv[None, :]) * current


def stencil_action(columns, mass, kappa, lapse, flow):
    """Independent component-level finite link action; no dense builder call."""
    x = columns.reshape(NSITES, 2, -1)
    out = NP.zeros_like(x)
    radius = NP.arange(1, 257, dtype=float)
    out[:, 0] = lapse[:, None] * (mass*x[:, 0] -
                                  1j*(kappa/radius)[:, None]*x[:, 1])
    out[:, 1] = lapse[:, None] * (-mass*x[:, 1] +
                                  1j*(kappa/radius)[:, None]*x[:, 0])
    nearest = .25 * (lapse[:-1] + lapse[1:])
    out[:-1, 0] += -1j*nearest[:, None]*x[1:, 1]
    out[:-1, 1] += -1j*nearest[:, None]*x[1:, 0]
    out[1:, 0] += 1j*nearest[:, None]*x[:-1, 1]
    out[1:, 1] += 1j*nearest[:, None]*x[:-1, 0]
    next_link = .125 * (flow[:-2] + flow[2:])
    out[:-2] += -1j*next_link[:, None, None]*x[2:]
    out[2:] += 1j*next_link[:, None, None]*x[:-2]
    return out.reshape(DIM, -1)


def energy_blocks(energies, tolerance):
    blocks, first, scale = [], 0, maxabs(energies)
    while first < DIM:
        end = first + 1
        while end < DIM and energies[end] - energies[first] < tolerance * scale:
            end += 1
        blocks.append((first, end))
        first = end
    return blocks


def reference_candidate(rt, energies, vectors, previous, tolerance):
    overlap = previous.conj().T @ vectors
    weights = NP.sum(NP.abs(overlap) ** 2, axis=0)
    workspace = vectors.copy()
    blocks = energy_blocks(energies, tolerance)
    rotations = {}
    for first, end in blocks:
        if end-first > 1:
            block_gram = overlap[:, first:end].conj().T @ overlap[:, first:end]
            values, rotation = rt.eigh(herm(block_gram), "reference block Gram")
            eigensystem_checks(rt, herm(block_gram), values, rotation,
                               f"reference_tol_{tolerance:g}_block_{first}_{end}",
                               raw_matrix=block_gram)
            workspace[:, first:end] = vectors[:, first:end] @ rotation
            weights[first:end] = values
            rotations[f"rotation_{first}_{end}"] = rotation
    intermediate = int(NP.count_nonzero((weights > .1) & (weights < .9)))
    return workspace, weights, blocks, rotations, intermediate


def reference_diagnostics(energies, selected, blocks):
    pockets, inversions, partial = 0, 0, []
    first_empty_block = next((b for b, (i, j) in enumerate(blocks)
                              if not selected[i:j].all()), len(blocks))
    gaps = []
    for b, (i, j) in enumerate(blocks):
        part = selected[i:j]
        if part.any() and not part.all():
            partial.append(b)
        if not (energies[i] <= 0 <= energies[j-1]):
            pockets += int(NP.count_nonzero(part & (energies[i:j] > 0)))
            pockets += int(NP.count_nonzero(~part & (energies[i:j] < 0)))
        if b > first_empty_block:
            inversions += int(part.sum())
        if b+1 < len(blocks):
            ni, nj = blocks[b+1]
            other = selected[ni:nj]
            if (part.any() and (~other).any()) or ((~part).any() and other.any()):
                gaps.append(float(energies[ni] - energies[j-1]))
    gap = min(gaps) if gaps else None
    return dict(pockets=pockets, inversions=inversions, partial_blocks=partial,
                gap_between_blocks=gap, weyl_gap=0. if partial else gap)


def continue_reference(rt, energies, vectors, previous, previous_mask):
    candidate = reference_candidate(rt, energies, vectors, previous, 1e-8)
    tolerance = 1e-8
    attempted_counts = [candidate[-1]]
    if candidate[-1]:
        alternative = reference_candidate(rt, energies, vectors, previous, 1e-6)
        attempted_counts.append(alternative[-1])
        if alternative[-1] < candidate[-1]:
            candidate, tolerance = alternative, 1e-6
    workspace, weights, blocks, rotations, intermediate = candidate
    index = NP.arange(DIM)
    order = NP.lexsort((index, energies, -weights))
    selected = NP.zeros(DIM, dtype=bool)
    selected[order[:RANK]] = True
    boundary = weights[order[RANK-1]]
    tied = NP.flatnonzero(NP.abs(weights-boundary) <= 1e-12)
    ambiguous, needed = False, 0
    chosen = NP.array([], dtype=int)
    if NP.any(~selected[tied]):
        above = weights > boundary + 1e-12
        needed = RANK - int(above.sum())
        old = tied[previous_mask[tied]]
        if len(old) == needed:
            chosen = old
        else:
            chosen = tied[NP.lexsort((tied, energies[tied]))[:needed]]
            ambiguous = True
        selected = above.copy()
        selected[chosen] = True
    result = workspace[:, selected].copy()
    diagnostic = reference_diagnostics(energies, selected, blocks)
    diagnostic.update(tolerance=tolerance, intermediate_weights=intermediate,
        attempted_intermediate_counts=attempted_counts, ambiguous=ambiguous,
        tied_set=tied.tolist(), tied_needed=needed, tied_chosen=chosen.tolist(),
        boundary_weight=float(boundary), rank=int(selected.sum()))
    raw = dict(mask=selected, reference_weights=weights,
        reference_blocks=NP.asarray(blocks, dtype=int), previous_mask=previous_mask,
        tied_set=tied, tied_chosen=chosen, **rotations)
    return result, selected, diagnostic, raw


def light_anchor(rt, energies, vectors, symmetry):
    workspace = vectors.copy()
    rotations = {}
    for first, end in energy_blocks(energies, 1e-8):
        if end-first > 1:
            block = vectors[:, first:end]
            raw_sector_matrix = block.conj().T @ symmetry @ block
            sector_matrix = herm(raw_sector_matrix)
            sector_values, rotation = rt.eigh(sector_matrix, "initial light sector block")
            eigensystem_checks(rt, sector_matrix, sector_values, rotation,
                               f"initial_sector_block_{first}_{end}",
                               raw_matrix=raw_sector_matrix)
            workspace[:, first:end] = block @ rotation
            rotations[f"anchor_rotation_{first}_{end}"] = rotation
    labels = NP.real(NP.sum(workspace.conj() * (symmetry @ workspace), axis=0))
    plus = NP.flatnonzero(labels > 0)
    n_plus = int(NP.count_nonzero(energies[plus] < 0))
    if not 0 < n_plus < len(plus):
        raise Incomplete("Unresolved initial light sector anchor")
    mu = .5 * (energies[plus[n_plus-1]] + energies[plus[n_plus]])
    selection = energies - mu*NP.sign(labels) < 0
    rt.condition("initial_light_n_plus", n_plus == 128)
    rt.condition("initial_light_anchor_rank", int(selection.sum()) == 256)
    rt.condition("initial_light_anchor_negative_set", NP.array_equal(selection, energies < 0))
    rt.metric("initial_light_sector_impurity", maxabs(NP.abs(labels)-1), 1e-12)
    return dict(n_plus=n_plus, mu=float(mu),
                selector_gap=float(NP.min(NP.abs(energies-mu*NP.sign(labels)))),
                sector_impurity=maxabs(NP.abs(labels)-1)), \
           dict(anchor_labels=labels, anchor_selection=selection,
                anchor_vectors=workspace, **rotations)


def scalar_entropy(value):
    if value <= 0 or value >= 1:
        return 0.
    return -value*math.log(value) - (1-value)*math.log1p(-value)


def entropy(rt, values, label):
    raw = NP.asarray(values, dtype=float)
    excursion = NP.maximum(NP.maximum(-raw, raw-1.), 0.)
    rt.metric(label + "_range_excursion", maxabs(excursion), CLIP_TOL)
    clipped = NP.clip(raw, 0., 1.)
    lower = clipped < 1e-300
    upper = (1.-clipped) <= NP.finfo(float).eps
    keep = ~(lower | upper)
    terms = NP.zeros_like(clipped)
    x = clipped[keep]
    terms[keep] = -x*NP.log(x) - (1.-x)*NP.log1p(-x)
    distances = NP.concatenate((clipped[lower], (1.-clipped)[upper & ~lower]))
    discarded = math.fsum(scalar_entropy(float(x)) for x in distances)
    nonzero_excursions = excursion[excursion > 0]
    diagnostic = dict(entropy=float(NP.sum(terms)), raw_min=float(NP.min(raw)),
        raw_max=float(NP.max(raw)), clipped_count=len(nonzero_excursions),
        clipped_max=maxabs(excursion), clipped_total=float(NP.sum(excursion)),
        lower_endpoint_count=int(lower.sum()), upper_endpoint_count=int(upper.sum()),
        endpoint_discard_allowance=discarded,
        clipping_continuity_sensitivity=len(raw)*scalar_entropy(min(.5, maxabs(excursion))))
    return diagnostic["entropy"], diagnostic


def spectrum(rt, matrix, label, raw):
    symmetric = herm(matrix)
    values, vectors = rt.eigh(symmetric, label + " spectrum")
    scale = max(1., float(LA.norm(symmetric, "fro")))
    eigensystem_checks(rt, symmetric, values, vectors, label)
    rt.metric(label + "_raw_Hermitian_defect",
              float(LA.norm(matrix-matrix.conj().T, "fro")), 1e-12*scale)
    raw[label + "_matrix"] = matrix
    raw[label + "_Hermitian"] = symmetric
    raw[label + "_eigenvalues"] = values
    raw[label + "_eigenvectors"] = vectors
    return values, vectors


def checkpoint(rt, k, state, reference, covariance, archive, complement, rank):
    rt.guard("entropy checkpoint")
    raw = {}
    m = archive.conj().T @ covariance @ archive
    retained = archive.conj().T @ state
    cc = complement.conj().T @ covariance @ complement
    ref_columns = archive.conj().T @ reference
    ref_compression = ref_columns @ ref_columns.conj().T
    raw.update(retained_columns=retained, reference_columns=ref_columns,
               reference_compression=ref_compression,
               reference_compression_Hermitian=herm(ref_compression))
    archive_values, _ = spectrum(rt, m, "archive", raw)
    complement_values, _ = spectrum(rt, cc, "complement", raw)
    singular_values = rt.singular_values(retained, "retained-channel singular values")
    reference_singular_values = rt.singular_values(ref_columns, "reference singular values")
    raw.update(retained_singular_values=singular_values,
               reference_singular_values=reference_singular_values)
    s_archive, d_archive = entropy(rt, archive_values, "archive")
    s_retained, d_retained = entropy(rt, singular_values**2, "retained")
    s_complement, d_complement = entropy(rt, complement_values, "complement")
    s_reference, d_reference = entropy(rt, reference_singular_values**2, "reference")
    rt.metric("archive_sorted_spectrum_difference",
              maxabs(NP.sort(archive_values)-NP.sort(singular_values**2)), READOUT_TOL)
    rt.metric("archive_entropy_difference", abs(s_archive-s_retained), READOUT_TOL)
    purity_values, purity_vectors = spectrum(rt, covariance, "global_covariance", raw)
    products = NP.abs(purity_values*(purity_values-1.))
    delta = float(NP.sum(products))
    rt.metric("global_covariance_range_excursion",
        maxabs(NP.maximum(NP.maximum(-purity_values, purity_values-1), 0)), CLIP_TOL)
    rt.metric("purity_delta", delta, 1e-8)
    floor_argument = (delta + 4096*NP.finfo(float).eps) / DIM
    rt.condition("purity_floor_argument_domain", 0 < floor_argument < .5)
    floor = DIM*scalar_entropy(floor_argument)
    z = abs(s_archive-s_complement)
    rt.metric("own_purity_floor_ratio", z/floor if floor else math.inf, 1.)
    projected_columns = purity_vectors[:, purity_values >= .5]
    rt.condition("projected_covariance_rank", projected_columns.shape[1] == rank)
    projected_covariance = projected_columns @ projected_columns.conj().T
    raw["projected_occupied_columns"] = projected_columns
    projected_m = archive.conj().T @ projected_covariance @ archive
    projected_cc = complement.conj().T @ projected_covariance @ complement
    projected_values, _ = spectrum(rt, projected_m, "projected_archive", raw)
    projected_complement_values, _ = spectrum(rt, projected_cc, "projected_complement", raw)
    s0, d0 = entropy(rt, projected_values, "projected_archive")
    sc0, dc0 = entropy(rt, projected_complement_values, "projected_complement")
    rt.metric("projected_complementary_entropy_defect", abs(s0-sc0), READOUT_TOL)
    candidate = archive.copy()
    if k:
        for _ in range(2):
            candidate -= archive @ (archive.conj().T @ candidate)
        append_singular = rt.singular_values(candidate, "static archive append residual")
        squared = append_singular**2
        drop_threshold = max(1e-12, 1e-14*maxabs(squared))
        rt.condition("static_archive_no_new_columns", not NP.any(squared > drop_threshold))
        raw["archive_candidate_twice_projected"] = candidate
        raw["archive_candidate_singular_values"] = append_singular
    else:
        squared, drop_threshold = NP.ones(96), 1e-12
    diagnostic = dict(k=k, t=2*k, S_dense=s_archive, S_retained=s_retained,
        S_complement=s_complement, S_vac=s_reference, increment=s_archive-s_reference,
        Z=z, purity_delta=delta, max_purity_product=maxabs(products),
        machine_epsilon=float(NP.finfo(float).eps), purity_floor=floor,
        purity_floor_ratio=z/floor if floor else None,
        projected_rank=projected_columns.shape[1], projected_Z=abs(s0-sc0),
        projected_entropy_shift=s0-s_archive,
        projected_distance_Frobenius=float(LA.norm(projected_covariance-covariance, "fro")),
        archive_columns=archive.shape[1], appended_columns=96 if k == 0 else 0,
        candidate_squared_singular_max=maxabs(squared), candidate_drop_threshold=drop_threshold,
        entropy_diagnostics=dict(archive=d_archive, retained=d_retained,
            complement=d_complement, reference=d_reference,
            projected_archive=d0, projected_complement=dc0))
    return diagnostic, raw


def sources(rt, base, current, state, reference, difference, mass, kappa):
    epsilon_dense = NP.real(NP.einsum("ij,ji->i", base, difference)).reshape(256, 2).sum(axis=1)
    current_dense = NP.real(NP.einsum("ij,ji->i", current, difference)).reshape(256, 2).sum(axis=1)
    epsilon_action = NP.zeros(NSITES)
    current_action = NP.zeros(NSITES)
    for columns, sign in ((state, 1), (reference, -1)):
        bare = stencil_action(columns, mass, kappa, NP.ones(256), NP.zeros(256))
        flow = stencil_action(columns, mass, kappa, NP.zeros(256), NP.ones(256))
        epsilon_action += sign*NP.real(NP.sum(columns.conj()*bare, axis=1)).reshape(256, 2).sum(axis=1)
        current_action += sign*NP.real(NP.sum(columns.conj()*flow, axis=1)).reshape(256, 2).sum(axis=1)
    comparison_checks(rt, "epsilon_source_difference", epsilon_dense, epsilon_action, SOURCE_TOL)
    comparison_checks(rt, "current_source_difference", current_dense, current_action, SOURCE_TOL)
    return dict(epsilon=epsilon_dense, current=current_dense,
                epsilon_action=epsilon_action, current_action=current_action)


def currents(rt, hamiltonian, difference):
    square = hamiltonian @ hamiltonian
    readouts = {}
    raw = {}
    for cut in (80, 160):
        split = 2*cut
        indicator = NP.repeat(NP.arange(1, 257) > cut, 2).astype(float)
        commutator_factor = indicator[None, :] - indicator[:, None]
        number_operator = 1j*hamiltonian*commutator_factor
        energy_operator = .5j*square*commutator_factor
        number = float(NP.real(NP.einsum("ij,ji->", difference, number_operator)))
        energy = float(NP.real(NP.einsum("ij,ji->", difference, energy_operator)))
        number_cross = float(-2*NP.imag(NP.einsum("ij,ji->",
            hamiltonian[:split, split:], difference[split:, :split])))
        energy_cross = float(-NP.imag(NP.einsum("ij,ji->",
            square[:split, split:], difference[split:, :split])))
        number_comparison = comparison_checks(rt, f"number_current_{cut}_difference",
                                               number, number_cross, READOUT_TOL)
        energy_comparison = comparison_checks(rt, f"energy_current_{cut}_difference",
                                               energy, energy_cross, READOUT_TOL)
        readouts[str(cut)] = dict(Phi_N=number, Phi_E=energy,
            Phi_N_cross=number_cross, Phi_E_cross=energy_cross,
            number_comparison=number_comparison, energy_comparison=energy_comparison)
        for label, operator in (("number", number_operator), ("energy", energy_operator)):
            indices = NP.nonzero(operator)
            raw[f"{label}_current_indices_{cut}"] = NP.asarray(indices)
            raw[f"{label}_current_entries_{cut}"] = operator[indices]
    return readouts, raw


def guarded_ratio(numerator, denominator, numerator_budget, denominator_budget):
    conditioned = abs(denominator) > denominator_budget
    quotient = numerator/denominator if denominator != 0 else None
    sensitivity = None
    if conditioned:
        sensitivity = numerator_budget/(abs(denominator)-denominator_budget) + \
            abs(numerator)*denominator_budget/(abs(denominator)*(abs(denominator)-denominator_budget))
    return dict(numerator=numerator, denominator=denominator, ratio=quotient,
        conditioning="resolved" if conditioned else "unresolved denominator",
        numerator_absolute_diagnostic_budget=numerator_budget,
        denominator_absolute_diagnostic_budget=denominator_budget,
        quotient_absolute_sensitivity=sensitivity, certified_error_bound=False)


def weighted_readouts(rt):
    channels = rt.summary["channels"]
    for k in range(41):
        rows = {name: channels[name]["rows"][k] for name, *_ in CHANNELS}
        row = dict(k=k, t=2*k, E_heavy=16*rows["H"]["E_inv"],
            E_light=16*rows["L1"]["E_inv"]+32*rows["L2"]["E_inv"],
            E_guard=48*rows["L3"]["E_inv"], collars={})
        row["guard_over_sourcing"] = guarded_ratio(row["E_guard"], row["E_light"],
                                                   48*READOUT_TOL, 48*READOUT_TOL)
        for cut in ("80", "160"):
            cutrow = {}
            for quantity in ("Phi_N", "Phi_E"):
                heavy = 16*rows["H"]["collars"][cut][quantity]
                light = 16*rows["L1"]["collars"][cut][quantity] + \
                        32*rows["L2"]["collars"][cut][quantity]
                guard = 48*rows["L3"]["collars"][cut][quantity]
                cutrow[quantity] = dict(heavy=heavy, sourcing_light=light, guard=guard,
                    guard_over_sourcing=guarded_ratio(guard, light, 48*READOUT_TOL, 48*READOUT_TOL))
            row["collars"][cut] = cutrow
        arrays = {}
        for name, *_ in CHANNELS:
            with NP.load(rt.run / rows[name]["raw_file"], allow_pickle=False) as data:
                arrays[name] = (data["epsilon"].copy(), data["current"].copy())
        eps = 16*arrays["H"][0]+16*arrays["L1"][0]+32*arrays["L2"][0]
        cur = 16*arrays["H"][1]+16*arrays["L1"][1]+32*arrays["L2"][1]
        row["weighted_source_file"] = rt.save_arrays(f"weighted/record_{k:04d}.npz",
            epsilon_sourcing=eps, current_sourcing=cur,
            epsilon_guard=48*arrays["L3"][0], current_guard=48*arrays["L3"][1])
        rt.summary["weighted_rows"].append(row)
    for checkpoint_index, k in enumerate(range(0, 41, 4)):
        entropies = {name: channels[name]["checkpoints"][checkpoint_index]["S_dense"]
                     for name, *_ in CHANNELS}
        total = sum(weight*entropies[name] for name, _, _, weight in CHANNELS)
        guard = 48*entropies["L3"]
        rt.summary["weighted_entropy"].append(dict(k=k, t=2*k, S_total=total,
            guard_share=guarded_ratio(guard, total, 48*READOUT_TOL,
                                     sum(c[3] for c in CHANNELS)*READOUT_TOL),
            channel_S_dense=entropies))


def channel(rt, name, mass, kappa, weight):
    rt.active_channel, rt.active_record = name, 0
    rt.current_checks = {}
    information = dict(mass=mass, kappa=kappa, weight=weight,
        sourcing=name != "L3", rows=[], checkpoints=[], propagations=0)
    rt.summary["channels"][name] = information
    base, current, symmetry = operators(mass, kappa)
    initial_lapse, initial_flow, _ = fields(0)
    initial_h = clocked(base, current, initial_lapse, initial_flow)
    ext_h = initial_h[320:, 320:].copy()
    ee, eu = rt.eigh(ext_h, "static exterior eigendecomposition")
    ext_scale = max(1., float(LA.norm(ext_h, "fro")))
    rt.metric("exterior_eigen_residual",
        float(LA.norm(ext_h @ eu-eu*ee[None, :], "fro")), 1e-12*ext_scale)
    rt.metric("exterior_eigenbasis_Gram", gram_defect(eu), 1e-12, strict=True)
    positive, negative = ee > 0, ee < 0
    rt.condition("archive_positive_count", int(positive.sum()) == 96)
    rt.condition("archive_negative_count", int(negative.sum()) == 96)
    rt.condition("archive_half_gap", float(NP.min(NP.abs(ee))) >= 1e-6)
    if rt.failed:
        rt.save_arrays(f"{name}/failed_exterior.npz", exterior=ext_h,
                       exterior_eigenvalues=ee, exterior_eigenvectors=eu)
        information["cache_checks"] = rt.current_checks.copy()
        rt.abort_failed()
    archive = NP.zeros((DIM, 96), dtype=complex)
    archive[320:] = eu[:, positive]
    complement = NP.zeros((DIM, 416), dtype=complex)
    complement[:320, :320] = NP.eye(320)
    complement[320:, 320:] = eu[:, negative]
    complete_basis = NP.column_stack((archive, complement))
    rt.metric("archive_Gram", gram_defect(archive), 1e-12, strict=True)
    rt.metric("complement_Gram", gram_defect(complement), 1e-12, strict=True)
    rt.metric("archive_complement_overlap", maxabs(archive.conj().T @ complement), 1e-12)
    rt.metric("complete_basis_Gram", gram_defect(complete_basis), 1e-12)
    rt.metric("complete_basis_resolution", maxabs(complete_basis @ complete_basis.conj().T-NP.eye(DIM)), 1e-12)
    information.update(archive_positive_count=int(positive.sum()),
        exterior_half_gap=float(NP.min(NP.abs(ee))),
        exterior_positive_min=float(ee[positive].min()),
        exterior_positive_max=float(ee[positive].max()),
        archive_indices=NP.flatnonzero(positive).tolist())
    information["cache_file"] = rt.save_arrays(f"{name}/cache.npz", h0=base,
        Q=current, S=symmetry, archive=archive, complement=complement,
        exterior_h=ext_h, exterior_eigenvalues=ee, exterior_eigenvectors=eu,
        positive_exterior_indices=NP.flatnonzero(positive))
    information["cache_checks"] = rt.current_checks.copy()
    rt.persist()
    rt.abort_failed()
    state = reference = mask = None
    old_h = old_lapse = old_flow = None
    previous_reference_diagnostic = None
    for k in range(41):
        rt.active_record, rt.current_checks = k, {}
        rt.guard("record fields and operator")
        lapse, flow, boundary = fields(k)
        hamiltonian = clocked(base, current, lapse, flow)
        scale = max(1., float(LA.norm(hamiltonian, "fro")))
        rt.metric("Hamiltonian_Hermitian", float(LA.norm(hamiltonian-hamiltonian.conj().T, "fro")), 1e-12*scale)
        action_matrix = stencil_action(NP.eye(DIM, dtype=complex), mass, kappa, lapse, flow)
        rt.metric("independent_operator_action", float(LA.norm(action_matrix-hamiltonian, "fro")), 1e-12*scale)
        rt.metric("antiunitary_symmetry", float(LA.norm(symmetry @ hamiltonian.conj() @ symmetry + hamiltonian, "fro")), 1e-12*scale)
        if mass == 0:
            rt.metric("massless_sector_commutator", float(LA.norm(symmetry @ hamiltonian-hamiltonian @ symmetry, "fro")), 1e-12*scale)
            rt.metric("massless_real_part", float(LA.norm(hamiltonian.real, "fro")), 1e-12*scale)
        rt.metric("static_exterior_equality", maxabs(hamiltonian[320:, 320:]-ext_h), 0.)
        energies, vectors = rt.eigh(hamiltonian, "full Hamiltonian eigendecomposition")
        residual = float(LA.norm(hamiltonian @ vectors-vectors*energies[None, :], "fro"))
        rt.metric("Hamiltonian_eigen_residual", residual, 1e-12*scale)
        rt.metric("Hamiltonian_eigenbasis_Gram", gram_defect(vectors), 1e-12, strict=True)
        rt.metric("Hamiltonian_spectral_pairing", maxabs(energies+energies[::-1]), 1e-12*scale)
        energies.setflags(write=False)
        vectors.setflags(write=False)
        phases = NP.exp(-2j*energies)
        spectral_file = rt.save_arrays(f"{name}/spectrum_{k:04d}.npz",
            lapse=lapse, flow=flow, eigenvalues=energies,
            eigenvectors=vectors, propagation_phases=phases)
        raw_reference, anchor_information = {}, None
        g, tau_lz = 0., None
        if k == 0:
            occupied = NP.flatnonzero(energies < 0)
            empty = NP.flatnonzero(energies >= 0)
            rt.condition("initial_sea_rank", len(occupied) == RANK)
            rt.condition("initial_resolved_zero_gap", float(NP.min(NP.abs(energies))) > residual)
            rt.condition("initial_dust_available", mass == 0 or len(empty) >= 6)
            if rt.failed:
                rt.save_arrays(f"{name}/failed_initial.npz", energies=energies, vectors=vectors,
                               lapse=lapse, flow=flow)
                information["failed_input_checks"] = rt.current_checks.copy()
                rt.abort_failed()
            reference = vectors[:, occupied].copy()
            state = NP.column_stack((reference, vectors[:, empty[:6]])) if mass else reference.copy()
            mask = energies < 0
            raw_reference = dict(mask=mask, reference_weights=mask.astype(float),
                reference_blocks=NP.asarray(energy_blocks(energies, 1e-8), dtype=int))
            reference_diagnostic = reference_diagnostics(energies, mask, energy_blocks(energies, 1e-8))
            reference_diagnostic.update(tolerance=1e-8, intermediate_weights=0,
                attempted_intermediate_counts=[0], ambiguous=False, rank=256,
                tied_set=[], tied_chosen=[], tied_needed=0)
            if mass == 0:
                anchor_information, anchor_raw = light_anchor(rt, energies, vectors, symmetry)
                raw_reference.update(anchor_raw)
            information.update(initial_sea_rank=len(occupied),
                initial_state_rank=state.shape[1], initial_gap=float(NP.min(NP.abs(energies))),
                initial_dust_energies=energies[empty[:6]].tolist() if mass else [],
                initial_anchor=anchor_information)
            rt.save_arrays(f"{name}/state_{k:04d}.npz", X=state,
                           sea_column_count=NP.array(256), dust_column_count=NP.array(6 if mass else 0))
        else:
            delta = hamiltonian-old_h
            g = float(NP.max(NP.sum(NP.abs(delta), axis=1)))
            delta_independent = clocked(base, current, lapse-old_lapse, flow-old_flow)
            rt.metric("delta_operator_linearity", float(LA.norm(delta-delta_independent, "fro")), 1e-12*scale)
            rt.metric("one_panel_bound", g, 93/800+1e-12)
            tau_lz = math.sqrt(2/g) if g else None
            rt.condition("one_panel_rate_branch", g == 0 or tau_lz > 2)
            reference, mask, reference_diagnostic, raw_reference = continue_reference(
                rt, energies, vectors, reference, mask)
        expected_rank = 262 if mass else 256
        rt.condition("state_column_count", state.shape == (DIM, expected_rank))
        rt.condition("reference_column_count", reference.shape == (DIM, 256))
        rt.metric("state_Gram", gram_defect(state), 1e-10)
        rt.metric("reference_Gram", gram_defect(reference), 1e-12, strict=True)
        if rt.failed:
            rt.save_arrays(f"{name}/failed_input_{k:04d}.npz", energies=energies,
                vectors=vectors, reference=reference, lapse=lapse, flow=flow,
                **raw_reference)
            information["failed_input_checks"] = rt.current_checks.copy()
            rt.abort_failed()
        rt.guard("same-record source energy and current contractions")
        covariance = state @ state.conj().T
        projector = reference @ reference.conj().T
        difference = covariance-projector
        trace = float(NP.real(NP.trace(difference)))
        rt.metric("subtracted_trace_identity", abs(trace-(6 if mass else 0)), READOUT_TOL)
        source_arrays = sources(rt, base, current, state, reference, difference, mass, kappa)
        energy = float(NP.real(NP.einsum("ij,ji->", hamiltonian, difference)))
        source_energy = float(NP.dot(lapse, source_arrays["epsilon"]) + NP.dot(flow, source_arrays["current"]))
        coefficient_state = vectors.conj().T @ state
        coefficient_reference = vectors.conj().T @ reference
        occupations = NP.sum(NP.abs(coefficient_state)**2, axis=1)
        reference_occupations = NP.sum(NP.abs(coefficient_reference)**2, axis=1)
        coefficient_energy = float(NP.dot(energies, occupations-reference_occupations))
        mask_energy = float(NP.dot(energies, occupations-mask))
        rt.metric("energy_source_identity", abs(energy-source_energy), READOUT_TOL)
        rt.metric("energy_spectral_identity", abs(energy-coefficient_energy), READOUT_TOL)
        collar_values, collar_raw = currents(rt, hamiltonian, difference)
        reference_diagnostic["Hamiltonian_commutator_Frobenius"] = float(LA.norm(hamiltonian @ projector-projector @ hamiltonian, "fro"))
        reference_diagnostic["sector_commutator_Frobenius"] = float(LA.norm(symmetry @ projector-projector @ symmetry, "fro"))
        previous_gap = previous_reference_diagnostic["weyl_gap"] if previous_reference_diagnostic else None
        reference_diagnostic["rate_verified_against_previous_gap"] = bool(previous_gap is not None and 2*g < previous_gap)
        trapping = lapse-NP.abs(flow)
        minimum_site = int(NP.argmin(trapping))
        row = dict(k=k, t=2*k, boundary=boundary, E_inv=energy,
            E_source=source_energy, E_coefficients=coefficient_energy,
            mask_shortcut=mask_energy, mask_shortcut_defect=mask_energy-energy,
            energy_identity_scale_relative=max(abs(energy-source_energy), abs(energy-coefficient_energy))/max(1., abs(energy)),
            trace_difference=trace, spectral_min=float(energies[0]), spectral_max=float(energies[-1]),
            trapping_min=float(trapping[minimum_site]), trapping_site=minimum_site+1,
            strict_trapping=bool(trapping[minimum_site] < 0), rate_G=g,
            tau_LZ=tau_lz, tau_LZ_infinite=bool(k > 0 and g == 0),
            reference=reference_diagnostic, collars=collar_values,
            state_file=f"raw/{name}/state_{k:04d}.npz", spectral_file=spectral_file)
        raw = dict(reference=reference,
            **source_arrays, **raw_reference, **collar_raw)
        row["raw_file"] = rt.save_arrays(f"{name}/record_{k:04d}.npz", **raw)
        row["controls"] = rt.current_checks.copy()
        information["rows"].append(row)
        rt.persist()
        if k % 4 == 0:
            entropy_row, entropy_raw = checkpoint(rt, k, state, reference,
                covariance, archive, complement, expected_rank)
            entropy_row["raw_file"] = rt.save_arrays(f"{name}/entropy_{k:04d}.npz", **entropy_raw)
            information["checkpoints"].append(entropy_row)
        row["controls"] = rt.current_checks.copy()
        rt.persist()
        rt.abort_failed()
        if k < 40:
            rt.guard("exact immutable-eigenpair propagation", dense=True)
            next_state = vectors @ (phases[:, None]*coefficient_state)
            held_before = float(NP.real(NP.sum(state.conj()*(hamiltonian @ state))))
            held_after = float(NP.real(NP.sum(next_state.conj()*(hamiltonian @ next_state))))
            rt.metric("held_step_energy_difference", abs(held_after-held_before), READOUT_TOL)
            rt.save_arrays(f"{name}/state_{k+1:04d}.npz", X=next_state,
                sea_column_count=NP.array(256), dust_column_count=NP.array(6 if mass else 0))
            state = next_state
            information["propagations"] += 1
            row["held_step_energy_before"] = held_before
            row["held_step_energy_after"] = held_after
            row["controls"] = rt.current_checks.copy()
            rt.persist()
            rt.abort_failed()
        old_h, old_lapse, old_flow = hamiltonian, lapse, flow
        previous_reference_diagnostic = reference_diagnostic
    information["complete"] = True
    rt.persist()


def raw_locator(rt):
    lines = ["path\tsize_bytes\tsha256\tbox_location"]
    for path in sorted(rt.raw.rglob("*")):
        if path.is_file():
            lines.append(f"{path.relative_to(rt.run)}\t{path.stat().st_size}\t{sha(path)}\tcompute-host:{path}")
    # The launcher adds the log only after the child has exited and the log closes.
    temporary = rt.run / "LARGE_FILES.tsv.tmp"
    temporary.write_text("\n".join(lines)+"\n")
    temporary.replace(rt.run / "LARGE_FILES.tsv")


def main():
    global NP, LA
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-start", type=float, required=True)
    args = parser.parse_args()
    expected = Path("<internal>/C_fixture_second_route_v1")
    if Path(__file__).resolve().parent != expected / "code" or args.output_dir != expected / "run":
        raise SystemExit("Only the own named box location is allowed")
    if resource.getrlimit(resource.RLIMIT_AS) != (4*2**30, 4*2**30):
        raise SystemExit("The launcher must set the 4 GiB address-space bound before imports")
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or len(os.sched_getaffinity(0)) != THREADS:
        raise SystemExit("Actual nice/CPU-affinity guard failed")
    signal.signal(signal.SIGTERM, stop_requested)
    import numpy as numpy_module
    import scipy.linalg as scipy_linalg
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    NP, LA = numpy_module, scipy_linalg
    rt = Runtime(args)
    try:
        with threadpool_limits(limits=THREADS):
            pools = threadpool_info()
            rt.summary.update(numpy_version=NP.__version__, scipy_version=scipy.__version__,
                library_threadpools=pools, actual_nice=os.getpriority(os.PRIO_PROCESS, 0),
                actual_affinity=sorted(os.sched_getaffinity(0)),
                address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                source_sha256=sha(__file__))
            rt.condition("actual_BLAS_thread_counts", all(p["num_threads"] <= THREADS for p in pools))
            rt.abort_failed()
            for args_channel in CHANNELS:
                channel(rt, *args_channel)
            rt.guard("weighted saved-row combinations")
            weighted_readouts(rt)
            exact_census = all(len(rt.summary["channels"][name]["rows"]) == 41 and
                len(rt.summary["channels"][name]["checkpoints"]) == 11 and
                rt.summary["channels"][name]["propagations"] == 40 for name, *_ in CHANNELS)
            rt.condition("full_fixture_census", exact_census)
            rt.abort_failed()
            rt.summary.update(status="complete", complete=True, all_controls_passed=True,
                              active_operation="terminal own result; foreign comparison pending")
    except BaseException as exc:
        rt.summary.update(status="incomplete", complete=False, all_controls_passed=False,
            exception_type=type(exc).__name__, exception=str(exc),
            traceback=traceback.format_exc(), failed_controls=rt.failed)
    finally:
        rt.persist()
        raw_locator(rt)
        rt.summary["raw_locator_sha256"] = sha(rt.run / "LARGE_FILES.tsv")
        rt.persist()
    print(json.dumps({key: rt.summary[key] for key in
        ("status", "complete", "elapsed_seconds", "peak_RSS_KiB")}))
    return 0 if rt.summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
