#!/usr/bin/env python3
"""HOLD 18.22: once-only recovered kappa-3 spectrum and saved full products.

New input branch only. Actual source/origin acceptance and static review must
precede execution. A resume loads completed stages; uncertain stages stop.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from hold_error_bounds import pairwise_sum
from hold_exact import log, register_bulk, resource_check, write_json
from hold_recovered_input import content_equal, load_recovered
from hold_recovered_origin import operator_rows, require_finite
from hold_source_preflight import OFFSETS, identity
from hold_spectral_checks import gram_error


VERSION = "recovered-original-A-k3-spectral-stages-1"
ACCEPTED_ORIGIN_SHA256 = {120: "ebe81d4a28e35b803547d042381adf60c502c4646d788cb6e6e08b32bfad35a6", 240: "3f91f7d7192dbffb1f6d45eebffa4a9257137f866000a0562bd1d58d33fcad50"}
STAGE_OUTPUTS = {
    "operator": {"held_fields.npz"},
    "spectrum": {"spectrum.npz"},
    "coordinates": {"occupied_eigenbasis.npy", "occupied_eigenbasis_binding.json"},
    "deviation": {"deviation_eigenbasis.npz", "spectral_transform_diagonals.npz"},
    "residual": {"eigen_residuals.npz"},
    "source_gram": {"state_gram.npz", "state_gram_arrays.npz"},
    "vector_gram": {"vector_gram_arrays.npz"},
    "deviation_control": {"deviation_checks.json"},
    "assessment": {"precision_checks.json", "spectral_checks.json", "initial_gate.json"},
}


def logical(value):
    if isinstance(value, dict):
        if {"path", "size", "sha256"} <= set(value):
            return {key: value[key] for key in ("path", "size", "sha256")}
        return {key: logical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [logical(item) for item in value]
    return value


def binding_hash(value):
    return hashlib.sha256(json.dumps(logical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment_check():
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if (any(os.environ.get(key) != "16" for key in keys)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.getpriority(os.PRIO_PROCESS, 0) < 10):
        raise RuntimeError("Recovered dense stages require CPU-only nice10 and all thread limits16")


def finite_matrix(array, shape, dtype):
    if array.shape != shape or array.dtype != np.dtype(dtype) or not array.dtype.isnative:
        raise ValueError("A recovered spectral product has another shape/precision")
    for start in range(0, shape[0], 128):
        require_finite(array[start:start+128])


def antihermitian(matrix):
    partials, maximum = [], 0.
    for start in range(0, len(matrix), 128):
        stop = min(start+128, len(matrix))
        difference = matrix[start:stop]-matrix[:, start:stop].conj().T
        require_finite(difference)
        partials.append(float(np.linalg.norm(difference))**2)
        maximum = max(maximum, float(np.max(np.abs(difference))))
    return dict(F=float(np.sqrt(pairwise_sum(partials))), entry_max=maximum,
        meaning="measured anti-Hermitian defect; no symmetrization or independent construction certificate")


def promote(temporary, path):
    # link creates the final name atomically and refuses an existing name.
    os.link(temporary, path)
    temporary.unlink()


def save_product_json(path, value):
    temporary = path.with_suffix(".partial.json")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+"\n")
    promote(temporary, path)
    return identity(path)


def save_npz(path, binding, **arrays):
    temporary = path.with_suffix(".partial.npz")
    with temporary.open("xb") as stream:
        np.savez(stream, input_binding_json=json.dumps(binding, sort_keys=True), **arrays)
    with np.load(temporary, allow_pickle=False) as saved:
        for key, original in arrays.items():
            original = np.asarray(original)
            restored = saved[key]
            if restored.shape != original.shape or restored.dtype != original.dtype or not restored.dtype.isnative:
                raise ValueError("A serialized recovered product changed layout/precision")
            if restored.ndim == 2:
                finite_matrix(restored, original.shape, original.dtype)
            else:
                require_finite(restored)
    promote(temporary, path)
    if path.stat().st_size > 20*2**20:
        register_bulk(path)
    return identity(path)


def save_npy(path, array):
    temporary = path.with_suffix(".partial.npy")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    restored = np.load(temporary, mmap_mode="r", allow_pickle=False)
    finite_matrix(restored, array.shape, array.dtype)
    del restored
    promote(temporary, path)
    register_bulk(path)
    return identity(path)


def load_origin(source, origin_path):
    saved = source.inputs.add(origin_path)
    endpoint = source.metadata["k"]
    pin = ACCEPTED_ORIGIN_SHA256[endpoint]
    if pin is None or saved["sha256"] != pin:
        raise RuntimeError("The actual endpoint origin judgment is not accepted and pinned")
    origin = json.loads(Path(saved["path"]).read_text())
    if (origin["method"] != "recovered-original-A-k3-origin-control-1" or origin["completed"] is not True
            or origin["physical_rows_released"] is not False
            or origin["coarse_energy_input_sanity_gate"]["passed"] is not True
            or origin["energy"]["whole_Hermitian_imaginary_control_passed"] is not True
            or origin["spatial_energy"]["completed_block_count"] != 32
            or origin["spatial_energy"]["completed_block_mask"] != [True]*32
            or len(origin["blocks"]) != 32):
        raise ValueError("The accepted origin control is incomplete or failed")
    metadata = origin["input"]
    for key in ("source_kind", "source_endpoint", "fields", "record", "k", "kappa", "nsites", "mass",
                "rs_nominal", "hold_rs", "charge", "g", "time_start", "source_error"):
        if metadata[key] != source.metadata[key]:
            raise ValueError("The origin/source physical tuple differs: " + key)
    for key in ("source_array", "source_construction", "source_seed"):
        if not content_equal(metadata[key], source.metadata[key]):
            raise ValueError("The origin binds another recovered input: " + key)
    if (not content_equal(origin["descriptor"], source.descriptor)
            or origin["operator"]["endpoint_operator"] != source.operator
            or not all(item["endpoint_interval_intersects"]
                       for item in origin["norm"]["source_norm_consistency"].values())):
        raise ValueError("Origin descriptor/operator/norm controls differ")
    for artifact in origin["verified_inputs"].values():
        source.inputs.verify(artifact)
    source.inputs.verify(origin["raw_scalar_arrays"])
    source.inputs.verify(origin["operator"]["artifact"])
    with np.load(origin["raw_scalar_arrays"]["path"], allow_pickle=False) as raw:
        if (raw["recovered_spatial_completed_block_count"].item() != 32
                or raw["recovered_spatial_completed_block_mask"].tolist() != [True]*32
                or raw["block_indices"].tolist() != list(range(32))
                or raw["recovered_spatial_spinor_row_indices"].tolist() != list(range(16384))
                or raw["recovered_spatial_column_ranges"].tolist() != [[256*b, 256*(b+1)] for b in range(32)]
                or logical(json.loads(raw["recovered_spatial_binding_json"].item()))
                   != logical(origin["spatial_energy"]["binding"])):
            raise ValueError("The accepted spatial raw archive has another completion/order/binding")
        for key in ("recovered_source_energy_row_block_real", "recovered_source_energy_row_block_imaginary"):
            finite_matrix(raw[key], (16384, 32), np.float64)
    source.check()
    return saved, origin


class Stages:
    def __init__(self, source, origin_id, origin, root, resume):
        self.source, self.origin, self.root = source, origin, root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root/"recovered_spectral_progress.json"
        self.started = time.monotonic()
        for name in ("hold_recovered_spectrum.py", "hold_recovered_precision.py", "hold_recovered_input.py",
                     "hold_recovered_origin.py", "hold_spectral_checks.py", "hold_error_bounds.py", "hold_exact.py"):
            source.inputs.add(Path(__file__).with_name(name))
        binding = dict(method=VERSION, input=source.metadata, origin_control=origin_id,
            origin_raw_arrays=origin["raw_scalar_arrays"], origin_operator=origin["operator"]["artifact"],
            accepted_origin_limits=origin["external_comparisons"], verified_inputs=dict(source.inputs.current))
        self.binding = binding
        self.digest = binding_hash(binding)
        if resume:
            if not self.path.exists():
                raise ValueError("No saved recovered spectral stages exist for resume")
            self.report = json.loads(self.path.read_text())
            if (self.report["method"] != VERSION or self.report["input_binding_hash"] != self.digest
                    or self.report["completed"] is True):
                raise ValueError("The recovered stage is complete or its accepted input/code tuple changed")
            for stage in self.report["stages"].values():
                if stage["status"] != "completed":
                    raise RuntimeError("An uncertain/failed numerical stage requires actual assessment, not automatic repetition")
                for saved in stage["outputs"].values():
                    source.inputs.verify(saved)
            self.validate_inventory()
        else:
            if any(root.iterdir()):
                raise FileExistsError("Existing recovered spectral outputs require explicit compatible resume")
            self.report = dict(method=VERSION, input=source.metadata, input_binding=binding,
                input_binding_hash=self.digest, completed=False, stages={}, physical_rows_released=False,
                source_kind=source.metadata["source_kind"],
                release="withheld; actual spectral judgment and unavailable held comparisons remain")
            write_json(self.path, self.report)

    def validate_inventory(self):
        names = list(self.report["stages"])
        if names != list(STAGE_OUTPUTS)[:len(names)]:
            raise RuntimeError("The saved numerical stages are not the required completed prefix")
        allowed = {self.path.name}
        for name, stage in self.report["stages"].items():
            if stage["status"] != "completed":
                raise RuntimeError("A started/failed stage cannot automatically resume")
            outputs = stage["outputs"]
            if {Path(saved["path"]).name for saved in outputs.values()} != STAGE_OUTPUTS[name]:
                raise RuntimeError("A completed stage has an incompatible output set")
            for saved in outputs.values():
                if Path(saved["path"]).parent != self.root:
                    raise RuntimeError("A completed stage output belongs to another directory")
                allowed.add(Path(saved["path"]).name)
        if {path.name for path in self.root.iterdir()} != allowed:
            raise RuntimeError("Orphan final/partial spectral artifacts require actual assessment; no numerical repetition")

    def save_product_json(self, path, value):
        return save_product_json(path, value)

    def execute(self, name, operation):
        if name in self.report["stages"]:
            stage = self.report["stages"][name]
            if stage["status"] != "completed":
                raise RuntimeError("A numerical stage cannot be implicitly repeated")
            return stage["values"]
        self.source.check()
        self.validate_inventory()
        resource_check(16384, dense=True)
        if shutil.disk_usage(self.root).free < 32*2**30:
            raise RuntimeError("Recovered dense stages lack their 32GiB bulk-storage margin")
        dependencies = {key: value["outputs"] for key, value in self.report["stages"].items()}
        binding = dict(common=self.binding, stage=name, prior_completed_products=dependencies)
        stage = dict(status="started", prospective_binding=binding)
        self.report["stages"][name] = stage
        write_json(self.path, self.report)
        before = time.monotonic()
        log("recovered_spectral_stage_started", stage=name, endpoint=self.source.metadata["k"])
        try:
            values, outputs = operation(binding)
            self.source.check()
            for saved in outputs.values():
                self.source.inputs.verify(saved)
            stage.update(status="completed", values=values, outputs=outputs,
                elapsed_seconds=time.monotonic()-before)
            write_json(self.path, self.report)
            log("recovered_spectral_stage_completed", stage=name, seconds=stage["elapsed_seconds"])
            return values
        except BaseException as error:
            stage.update(status="failed_or_uncertain", stop_reason=type(error).__name__+": "+str(error),
                elapsed_seconds=time.monotonic()-before)
            self.report["completed"] = False
            write_json(self.path, self.report)
            raise

    def output(self, stage, key):
        saved = self.report["stages"][stage]["outputs"][key]
        self.source.inputs.verify(saved)
        return Path(saved["path"])


def run(args):
    environment_check()
    if any(pin is None for pin in ACCEPTED_ORIGIN_SHA256.values()):
        raise RuntimeError("Both actual recovered-origin judgments must be pinned before either dense stage")
    source = load_recovered(args.descriptor, args.build, args.comparison)
    origin_id, origin = load_origin(source, args.origin)
    root = Path(args.out).resolve()
    root.relative_to(Path("<internal>/second_route_bulk"))
    stages = Stages(source, origin_id, origin, root, args.resume)

    def operator(binding):
        with np.load(origin["operator"]["artifact"]["path"], allow_pickle=False) as raw:
            bands = raw["bands"]
            if raw["offsets"].tolist() != list(OFFSETS):
                raise ValueError("Origin operator band ordering changed")
            inner = {key: raw[key] for key in ("inner_radius", "inner_clock_applied", "inner_flow_applied")}
        if hashlib.sha256(bands.tobytes(order="C")).hexdigest() != source.operator["band_content_sha256"]:
            raise ValueError("The held operator is not the accepted unscaled endpoint")
        _, row_bound = operator_rows(bands)
        row_indices, column_indices, coefficients = [], [], []
        for band, offset in enumerate(OFFSETS):
            selected = np.flatnonzero(bands[band])
            row_indices.extend(selected.tolist())
            column_indices.extend((selected+offset).tolist())
            coefficients.extend(bands[band, selected].tolist())
        data = np.zeros(len(coefficients), dtype=np.complex128)
        data.imag = coefficients  # exact component placement, h=iB
        h = sp.coo_matrix((data, (row_indices, column_indices)), shape=(16384, 16384)).tocsr()
        if (h-h.getH()).nnz or np.any(h.data.real != 0) or np.max(np.diff(h.indptr)) > 5:
            raise ValueError("The stored endpoint CSR violates its exact Hermitian/five-term convention")
        path = root/"held_fields.npz"
        saved = save_npz(path, binding, bands=bands, offsets=OFFSETS,
            h_data=h.data, h_indices=h.indices, h_indptr=h.indptr,
            radius=np.arange(1, 8193, dtype=float), **inner)
        from hold_error_bounds import upwards
        return dict(h_absolute_row_sum_upper=upwards(row_bound), row_nonzeros=int(np.max(np.diff(h.indptr))),
            dimension=16384, field_scope="accepted inner samples/point-tail declaration; h reused exactly from cache"), dict(held_fields=saved)

    operator_values = stages.execute("operator", operator)

    def load_h():
        with np.load(stages.output("operator", "held_fields"), allow_pickle=False) as raw:
            return sp.csr_matrix((raw["h_data"], raw["h_indices"], raw["h_indptr"]), shape=(16384, 16384))

    def spectrum(binding):
        h = load_h()
        dense = np.array(h.toarray(), order="F")
        e, v = la.eigh(dense, overwrite_a=True, check_finite=False, driver="evd")
        del dense
        finite_matrix(v, (16384, 16384), np.complex128)
        require_finite(e)
        if e.shape != (16384,) or e.dtype != np.float64 or np.any(e[1:] < e[:-1]):
            raise ValueError("The full held spectrum has another ordering/layout")
        saved = save_npz(root/"spectrum.npz", binding, eigenvalues=e, eigenvectors=v)
        return dict(driver="evd", eigenvalue_min=float(e[0]), eigenvalue_max=float(e[-1]),
            minimum_abs_eigenvalue=float(np.min(np.abs(e))), negative_levels=int(np.count_nonzero(e < 0)),
            zero_rule="strict epsilon<0; no clipping, pairing enforcement or occupation repair"), dict(spectrum=saved)

    stages.execute("spectrum", spectrum)

    def load_spectrum():
        with np.load(stages.output("spectrum", "spectrum"), allow_pickle=False) as raw:
            return raw["eigenvalues"], raw["eigenvectors"]

    def coordinates(binding):
        _, v = load_spectrum()
        coordinates = v.conj().T @ source.occupied
        finite_matrix(coordinates, (16384, 8192), np.complex128)
        saved = save_npy(root/"occupied_eigenbasis.npy", coordinates)
        companion = root/"occupied_eigenbasis_binding.json"
        companion_id = save_product_json(companion, dict(input_binding=binding, coordinates=saved, completed=True))
        return dict(shape=[16384, 8192], operation="one full V^dagger Phi"), dict(
            coordinates=saved, binding=companion_id)

    stages.execute("coordinates", coordinates)

    def deviation(binding):
        e, v = load_spectrum()
        del v
        coordinates = np.load(stages.output("coordinates", "coordinates"), mmap_mode="r", allow_pickle=False)
        finite_matrix(coordinates, (16384, 8192), np.complex128)
        deviation = coordinates @ coordinates.conj().T
        finite_matrix(deviation, (16384, 16384), np.complex128)
        pre = deviation.diagonal().copy()
        deviation[np.diag_indices_from(deviation)] -= (e < 0)
        post = deviation.diagonal().copy()
        finite_matrix(deviation, (16384, 16384), np.complex128)
        saved = save_npz(root/"deviation_eigenbasis.npz", binding, deviation=deviation, eigenvalues=e)
        diagonals = save_npz(root/"spectral_transform_diagonals.npz", binding,
            G_hat_diagonal=pre, Dhat_diagonal=post, eigenvalues=e)
        return dict(shape=[16384, 16384], operation="one full B B^dagger then saved strict-negative diagonal subtraction"), dict(
            deviation=saved, diagonals=diagonals)

    stages.execute("deviation", deviation)

    def residual(binding):
        e, v = load_spectrum()
        h = load_h()
        columns = np.empty(16384)
        for start in range(0, 16384, 512):
            source.check()
            part = v[:, start:start+512]
            difference = h @ part-part*e[None, start:start+512]
            require_finite(difference)
            columns[start:start+512] = np.linalg.norm(difference, axis=0)
        saved = save_npz(root/"eigen_residuals.npz", binding, residual_columns=columns,
            paired_eigenvalue_errors=e+e[::-1], near_zero_eigenvalues=e[np.argsort(np.abs(e))[:32]])
        return dict(eigen_residual_F=float(np.linalg.norm(columns)), eigen_residual_column_max=float(np.max(columns)),
            paired_eigenvalue_error_max=float(np.max(np.abs(e+e[::-1])))), dict(residual=saved)

    stages.execute("residual", residual)

    def source_gram(binding):
        norm, maximum, columns, matrix = gram_error(source.occupied, keep_matrix=True)
        finite_matrix(matrix, (8192, 8192), np.complex128)
        anti = antihermitian(matrix)
        trace = np.trace(matrix)
        saved = save_npz(root/"state_gram.npz", binding, gram_minus_identity=matrix)
        small = save_npz(root/"state_gram_arrays.npz", binding,
            column_squared_errors=columns, gram_defect_diagonal=matrix.diagonal().copy())
        return dict(state_gram_F=norm, state_gram_entry_max=maximum,
            gram_defect_trace_real=float(trace.real), gram_defect_trace_imaginary=float(trace.imag),
            gram_scope="Phi^dagger Phi-I; saved trace/diagonal are the actual Gram defect",
            antihermitian=anti, dot_length=16384, width=8192,
            conditional_purity_F_lower=float(max(0., 1-norm)*norm),
            conditional_purity_F_upper=float((1+norm)*norm)), dict(gram=saved, arrays=small)

    stages.execute("source_gram", source_gram)

    def vector_gram(binding):
        _, v = load_spectrum()
        norm, maximum, columns = gram_error(v)
        require_finite(norm, maximum, columns)
        saved = save_npz(root/"vector_gram_arrays.npz", binding, column_squared_errors=columns)
        return dict(eigenvector_gram_F=norm, eigenvector_gram_entry_max=maximum,
            dot_length=16384, width=16384), dict(arrays=saved)

    stages.execute("vector_gram", vector_gram)

    def deviation_control(binding):
        with np.load(stages.output("deviation", "deviation"), allow_pickle=False) as raw:
            deviation = raw["deviation"]
        finite_matrix(deviation, (16384, 16384), np.complex128)
        values = antihermitian(deviation)
        saved = root/"deviation_checks.json"
        saved_id = save_product_json(saved, dict(input_binding=binding, antihermitian=values, completed=True))
        return values, dict(checks=saved_id)

    stages.execute("deviation_control", deviation_control)

    def assessment(binding):
        from hold_recovered_precision import assess
        return assess(stages, binding, operator_values)

    stages.execute("assessment", assessment)
    source.check()
    stages.report.update(completed=True, elapsed_seconds_this_invocation=time.monotonic()-stages.started,
        release="withheld; actual numerical judgments, matching held extrema and first held interval comparisons remain")
    write_json(stages.path, stages.report)
    log("recovered_spectral_products_complete", endpoint=source.metadata["k"], physical_rows_released=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("descriptor", "comparison", "build", "origin", "out"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
