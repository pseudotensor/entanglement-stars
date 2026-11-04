"""Explicit, read-only recovered kappa-3 input; HOLD specification 18.19.

This module does not change native checkpoint loaders. The source acceptance
pin stays disabled until both scientists judge the complete source pass.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from hold_error_bounds import rational, sqrt_upper, upwards
from hold_source_action import Method, content_equal
from hold_source_preflight import identity, unchanged


KIND = "recovered-original-A-physical-Phi"
METHOD = "original-A-k3-source-four-subactions-1"
ACCEPTED_SOURCE_SHA256 = "19be36cdf080b34aac7cfcf0f454a3c5caeb9374a6a3aada182579e628d87a52"
EXTRACT_SHA256 = "1e3abb5390b6e255b893360946c3f39b4971b55864c9889967bb438d6c6262e6"
ORIGINAL_SUMMARY_SHA256 = "6c7fcd9a0d0e918e39c88c2b3993bcdbde42f4239a55b7a6d25becd272001d78"
PINS = dict(
    controller="eeb5bc52df3c0466bd9702a6d914b2f61ed84f0da2a7da4b2e81488cba3e04d5",
    fixed_fixture="dbf550ca5fe64ff387a86b291d14b427420fb9199d7385a2c9a2abd5b5a93408",
    full_seed_control="10451e22f675e3a5f073490d2d2fd949d7236ea3d6c6084d382f0f8a73300f75",
    reviewed_v2_result="87821f28f4ffb39ddebd9030e4a80332666492980ed1f425d52028e4c80df9c5",
    completed_v1_result="be401dc72cca9c0b769e14a44ce99309854e27f1d62ddd053c7919b4554043fe",
    operator_cache="a1986fe37f19417c492daef437a48d4db0dda318b7aa6f3cd3da4e8e76123604",
    build_result="c5bb31d6ea22ef6356b044630f7f6c2badb43b1a25d7a629d9b753ba1cd531ef")
ENDPOINT_BAND_SHA256 = {
    120: "928db8d4485b5b63ed8e5b2b116b1aef8b3d0ef4944e35b04fa95b08ec1a963e",
    240: "10471aa7f9c1c3a6cd7a990fd54d59ce62024db9e564d26a8dc7d199c59c911e"}


class ContentInputs:
    """Verify content once per canonical path; monitor fresh within-call stats."""
    def __init__(self):
        self.current = {}

    def verify(self, saved):
        path = str(Path(saved["path"]).resolve(strict=True))
        if saved["path"] != path:
            raise ValueError("A content-bound path is not canonical")
        if path not in self.current:
            self.current[path] = identity(path)
        if not content_equal(saved, self.current[path]):
            raise ValueError("A recovered-input content identity differs: " + path)
        self.check()
        return self.current[path]

    def add(self, path):
        saved = identity(path)
        return self.verify(saved)

    def check(self):
        unchanged(self.current)


def input_digest(artifacts):
    values = {key: {field: value[field] for field in ("path", "size", "sha256")}
              for key, value in artifacts.items()}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def positive(value):
    value = rational(value)
    if value <= 0:
        raise ValueError("A required source allowance/norm is not positive")
    return value


def validate_blocks(report, inputs, seed):
    if (report["completed_blocks"] != list(range(32))
            or set(report["block_result_identities"]) != {str(b) for b in range(32)}):
        raise ValueError("The recovered source does not cover each of the 32 blocks exactly once")
    if len(seed["blocks"]) != 32 or len(report["initial_norm_scaled_block_forecasts"]) != 32:
        raise ValueError("The saved seed/forecast does not cover all source columns")
    digest = input_digest(report["artifacts"])
    blocks = []
    for b in range(32):
        saved = inputs.verify(report["block_result_identities"][str(b)])
        block = json.loads(Path(saved["path"]).read_text())
        columns = [256*b, 256*(b+1)]
        if ((block["method"], block["index"], block["columns"], block["completed"],
             block["source_pass_input_sha256"]) != (METHOD, b, columns, True, digest)
                or set(block["endpoints"]) != {"120", "240"}
                or seed["blocks"][b]["index"] != b
                or seed["blocks"][b]["columns"] != columns
                or report["initial_norm_scaled_block_forecasts"][b]["index"] != b):
            raise ValueError("A completed source block has another input/range")
        for endpoint in (120, 240):
            item = block["endpoints"][str(endpoint)]
            logical = item["source_columns"]
            if ((item["time"], item["after_operator_index"], item["before_operator_index"])
                    != (8*endpoint, endpoint-1, endpoint)
                    or (logical["columns"], logical["shape"], logical["dtype"], logical["byte_order"])
                    != (columns, [16384, 256], "complex128", "native")
                    or len(logical["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in logical["sha256"])
                    or item["initial_norm_control"] != seed["blocks"][b]
                    or item["error"] != report["initial_norm_scaled_block_forecasts"][b]["endpoints"][str(endpoint)]["error"]):
                raise ValueError("A source endpoint clock/column/error binding differs")
        blocks.append(block)
    denominator = positive(seed["norm_lower"])
    for endpoint in ("120", "240"):
        for kind in ("stored_operator", "model_operator"):
            recombined = sqrt_upper(sum((positive(b["endpoints"][endpoint]["error"][kind]["error_upper"])**2
                                        for b in blocks), Fraction(0)))
            saved = report["source_error_controls"][endpoint][kind]
            absolute = positive(saved["error_upper"])
            relative = positive(saved["relative_error_upper"])
            if (absolute < recombined or relative < recombined/denominator
                    or relative > Fraction(1, 10**10) or saved["full_source_ceiling_passed"] is not True):
                raise ArithmeticError("A complete source-error aggregate/ceiling is invalid")
    return blocks


@dataclass
class RecoveredSource:
    metadata: dict
    occupied: np.ndarray
    method: Method
    operator: dict
    seed_control: dict
    external_origin_targets: dict
    source_error: dict
    inputs: ContentInputs
    descriptor: dict

    def check(self):
        self.inputs.check()
        self.method.check_runtime()


def load_recovered(descriptor_path, build_folder, comparison_path):
    # No permissive command-line digest override. Later pin-only change requires
    # the actual complete-source judgments and the same static review.
    if ACCEPTED_SOURCE_SHA256 is None:
        raise RuntimeError("Complete recovered source is not yet accepted and pinned")
    inputs = ContentInputs()
    descriptor_id = inputs.add(descriptor_path)
    descriptor = json.loads(Path(descriptor_id["path"]).read_text())
    if descriptor["source_kind"] != KIND or descriptor["method"] != METHOD:
        raise ValueError("Only the explicit recovered physical-source kind is accepted")
    endpoint = descriptor["source_endpoint"]
    if type(endpoint) is not int or endpoint not in (120, 240):
        raise ValueError("The recovered endpoint must be exactly 120 or 240")
    source_id = inputs.verify(descriptor["source_construction"])
    if source_id["sha256"] != ACCEPTED_SOURCE_SHA256:
        raise ValueError("The construction result is not the accepted complete snapshot")
    report = json.loads(Path(source_id["path"]).read_text())
    if ((report["source_kind"], report["method"], report["completed"], report["active_segment"],
         report["physical_rows_released"], report["first_cohort_projection_passed"])
            != (KIND, METHOD, True, None, False, True)
            or set(report["raw_sources"]) != {"120", "240"}
            or not report["execution_segments"]
            or not 0 < report["elapsed_seconds_consumed"] <= 8*3600
            or not 0 < report["first_cohort_seconds"] < 2*3600):
        raise ValueError("The source pass is incomplete, active or outside its fixed budget")
    for key, digest in PINS.items():
        if report["artifacts"][key]["sha256"] != digest:
            raise ValueError("A reviewed source prerequisite pin differs: " + key)
    for saved in report["artifacts"].values():
        inputs.verify(saved)
    method = Method(build_folder)
    for key, saved in method.artifacts.items():
        if not content_equal(saved, report["artifacts"][key]):
            raise ValueError("The current method differs from the accepted source method")
        inputs.verify(saved)
    seed = json.loads(Path(report["artifacts"]["full_seed_control"]["path"]).read_text())
    if (seed["completed"], seed["storage_control_passed"], seed["elapsed_seconds_consumed"]) != (
            True, True, report["seed_setup_seconds"]):
        raise ValueError("The source's actual seed control differs")
    blocks = validate_blocks(report, inputs, seed)
    for key, saved in report["raw_sources"].items():
        if (Path(saved["path"]).name != f"k3_source_Phi_T{8*int(key):04d}.npy"
                or Path(saved["path"]).parent != Path(source_id["path"]).parent):
            raise ValueError("A recovered source is partial or has another endpoint path")
        inputs.verify(saved)
    raw_id = report["raw_sources"][str(endpoint)]
    if not content_equal(raw_id, descriptor["source_array"]):
        raise ValueError("The descriptor selects another raw recovered array")
    occupied = np.load(raw_id["path"], mmap_mode="r", allow_pickle=False)
    if (occupied.shape != (16384, 8192) or occupied.dtype != np.complex128
            or not occupied.dtype.isnative or not occupied.flags.c_contiguous
            or occupied.flags.writeable):
        raise ValueError("The recovered physical array has another precision/layout")
    v1 = json.loads(Path(report["artifacts"]["completed_v1_result"]["path"]).read_text())
    operator = v1["operators"][endpoint]
    if ((operator["index"], operator["time"]) != (endpoint, 8*endpoint)
            or operator["band_content_sha256"] != ENDPOINT_BAND_SHA256[endpoint]
            or operator["absolute_row_sum_upper"] > method.rho):
        raise ValueError("The endpoint operator metadata differ")
    selected = method.bands(endpoint, scaled=False)
    if (hashlib.sha256(selected.tobytes(order="C")).hexdigest() != operator["band_content_sha256"]
            or not np.array_equal(selected, method.cache[endpoint])):
        raise ValueError("The unscaled endpoint operator differs from its accepted cache")
    extract_id = inputs.add(comparison_path)
    if extract_id["sha256"] != EXTRACT_SHA256:
        raise ValueError("The comparator extract differs from its reviewed snapshot")
    extract = json.loads(Path(extract_id["path"]).read_text())
    if (extract["source"]["sha256"] != ORIGINAL_SUMMARY_SHA256 or extract["final"] is not True
            or extract["method"] != "original-A-k3-external-origin-target-extract-1"):
        raise ValueError("The external comparator is not its reviewed metadata snapshot")
    matches = [row for row in extract["origins"] if row["k"] == endpoint and row["time"] == 8*endpoint]
    if len(matches) != 1 or matches[0]["energy_row"]["charge"] != operator["charge"]:
        raise ValueError("The endpoint external comparator has another charge/clock")
    info = report["input"]
    expected = dict(kappa=3, mass=0., g=.25, rs_nominal=64., reading="A", nsites=8192,
                    occupied_columns=8192, source_intervals=240, duration=8.,
                    hold_indices=[120, 240], hold_clocks=[960., 1920.])
    if any(info[key] != value for key, value in expected.items()):
        raise ValueError("The recovered source has another physical input tuple")
    source_error = dict(endpoint=endpoint, totals=report["source_error_controls"][str(endpoint)],
        initial_norm_lower=seed["norm_lower"], initial_norm_upper=seed["norm_upper"],
        relative_denominator="same accepted initial full-source norm lower bound",
        block_controls=[b["endpoints"][str(endpoint)] for b in blocks],
        producer_seed_version_difference=report["producer_seed_version_difference"],
        named_component_F_upper={})
    for kind in ("stored_operator", "model_operator"):
        names = blocks[0]["endpoints"][str(endpoint)]["error"][kind]["components"]
        if any(set(b["endpoints"][str(endpoint)]["error"][kind]["components"]) != set(names) for b in blocks):
            raise ValueError("Source blocks have inconsistent named error components")
        source_error["named_component_F_upper"][kind] = {name: upwards(sqrt_upper(sum(
            (rational(b["endpoints"][str(endpoint)]["error"][kind]["components"][name])**2
             for b in blocks), Fraction(0)))) for name in names}
    metadata = dict(source_kind=KIND, source_array=raw_id, source_construction=source_id,
        source_endpoint=endpoint, source_error=source_error,
        fields=report["artifacts"]["v1_fields"]["path"], record=report["artifacts"]["v1_record"]["path"],
        k=endpoint, kappa=3, nsites=8192, mass=0., rs_nominal=64., hold_rs=endpoint/8.,
        charge=operator["charge"], g=.25, time_start=8.*endpoint,
        source_seed=report["artifacts"]["v1_seed"],
        original_seed_declaration=v1["seed_declaration"])
    inputs.check()
    return RecoveredSource(metadata, occupied, method, operator, seed,
        dict(comparator_extract=inputs.current[str(Path(comparison_path).resolve())],
             original_summary=extract["source"], selected=matches[0],
             scope=extract["scope"], producer_seed_version_difference=extract["producer_seed_version_difference"]),
        source_error, inputs, descriptor_id)
