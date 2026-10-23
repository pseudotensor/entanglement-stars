#!/usr/bin/env python3
"""Compare immutable saved C readouts; no producer or scientific imports."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import struct
import sys
import time
import zipfile


TOL = 1e-10
CHANNELS = {"H": ("heavy", 16), "L1": ("light1", 16),
            "L2": ("light2", 32), "L3": ("light3", 48)}
EXPECTED = {
    "summary.json": "fd8058df9cde4b18a7da5485853a50111a90211f0874040662715d7800f88073",
    "execution.json": "1ab045176848e0e6f92d455f049b8d1183ac5a628b6e72aa99c15a84e28e2f9a",
    "LARGE_FILES.tsv": "84e5dbe62f7d5bd539ea6f7dc4a92b2f2f8704ef11b9c3124f88255330b3a8e7",
    "foreign_source": "20b8694bbc9779cacc29ca5b5585fc090134b8b949d851684f327f750db33c1d",
    "foreign_result": "03300a11f410b8f41a57bf9aec39c39f45bfb48f54de810ed832647e9bd13e85",
    "foreign_table": "ff8f4a2af468245c72be5170bb6715dd8709252917e71b4a05fcc21a9aac2999",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(rows, fields, expected):
    out = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        assert key not in out, ("duplicate", fields, key)
        out[key] = row
    assert set(out) == expected, ("census", fields, set(out) ^ expected)
    return out


def numeric(value):
    assert not isinstance(value, bool) and isinstance(value, (int, float))
    assert math.isfinite(value)
    return value


def saved_vector(path, member):
    """Decode only one stored numeric 1D NPY member, without NumPy or pickle."""
    with zipfile.ZipFile(path) as archive:
        with archive.open(member + ".npy") as stream:
            assert stream.read(6) == b"\x93NUMPY"
            version = tuple(stream.read(2))
            assert version in ((1, 0), (2, 0), (3, 0)), version
            size_format = "<H" if version == (1, 0) else "<I"
            width = struct.calcsize(size_format)
            header_size = struct.unpack(size_format, stream.read(width))[0]
            assert header_size < 1000000
            header = ast.literal_eval(stream.read(header_size).decode(
                "utf-8" if version == (3, 0) else "latin1"))
            assert len(header["shape"]) == 1 and header["shape"][0] <= 512
            dtype = header["descr"]
            assert dtype in ("<f8", ">f8", "<i8", ">i8", "<i4", ">i4")
            code = {"f8": "d", "i8": "q", "i4": "i"}[dtype[1:]]
            count = header["shape"][0]
            data = stream.read()
            assert len(data) == count * struct.calcsize(dtype[0] + code)
            values = list(struct.unpack(dtype[0] + str(count) + code, data))
            assert all(math.isfinite(x) for x in values)
            return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    root = args.run.resolve()
    bundle = json.load(sys.stdin)
    bindings = []
    own_bytes = {}
    for name in ("summary.json", "execution.json", "LARGE_FILES.tsv"):
        data = (root / name).read_bytes()
        assert digest(data) == EXPECTED[name], name
        own_bytes[name] = data
        bindings.append(dict(path=str(root / name), bytes=len(data), sha256=digest(data)))
    foreign_bytes = {}
    for key in ("foreign_source", "foreign_result", "foreign_table"):
        entry = bundle[key]
        data = entry["text"].encode("utf-8")
        assert digest(data) == EXPECTED[key], key
        foreign_bytes[key] = data
        bindings.append(dict(path=entry["path"], bytes=len(data), sha256=digest(data),
                             transport="stdin only; no foreign file copied"))
    own = json.loads(own_bytes["summary.json"])
    receipt = json.loads(own_bytes["execution.json"])
    foreign = json.loads(foreign_bytes["foreign_result"])
    assert own["complete"] and receipt["execution_complete"]
    assert receipt["child_returncode"] == 0
    for name in ("summary.json", "LARGE_FILES.tsv"):
        assert receipt[name]["sha256"] == digest(own_bytes[name])
        assert receipt[name]["bytes"] == len(own_bytes[name])
    assert set(own["channels"]) == set(CHANNELS)
    expected_rows = {(label, k) for label, _ in CHANNELS.values() for k in range(41)}
    expected_checkpoints = {(label, k) for label, _ in CHANNELS.values()
                            for k in range(0, 41, 4)}
    fr = unique(foreign["E_inv_rows"], ("label", "step"), expected_rows)
    fc = unique(foreign["S_esc"], ("label", "step"), expected_checkpoints)
    fw = unique(foreign["weighted_rows"], ("step",), {(k,) for k in range(41)})
    manifest = {}
    for row in csv.DictReader(io.StringIO(own_bytes["LARGE_FILES.tsv"].decode()), delimiter="\t"):
        assert row["path"] not in manifest
        manifest[row["path"]] = row
    pairs, ratios, raw_bindings = [], [], []

    def compare(category, channel, k, field, a, b, primary=False, tolerance=TOL):
        a, b = numeric(a), numeric(b)
        difference = a - b
        pairs.append(dict(category=category, channel=channel, k=k, t=2*k,
                          field=field, own=a, foreign=b, difference_own_minus_foreign=difference,
                          absolute_difference=abs(difference), tolerance=tolerance,
                          passed=abs(difference) <= tolerance, primary=primary))

    def raw_file(relative):
        path = (root / relative).resolve()
        assert path.is_relative_to(root / "raw")
        row = manifest[relative]
        before = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                h.update(block)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert before.st_size == int(row["size_bytes"]) and h.hexdigest() == row["sha256"]
        raw_bindings.append(dict(path=relative, bytes=before.st_size,
                                 sha256=h.hexdigest(), box_location=row["box_location"]))
        return path

    own_rows, own_checkpoints = {}, {}
    for name, (label, weight) in CHANNELS.items():
        channel = own["channels"][name]
        assert channel["weight"] == weight and channel["propagations"] == 40
        assert channel["sourcing"] == (name != "L3") and channel["archive_positive_count"] == 96
        rows = unique(channel["rows"], ("k",), {(k,) for k in range(41)})
        checkpoints = unique(channel["checkpoints"], ("k",), {(k,) for k in range(0, 41, 4)})
        own_rows[name], own_checkpoints[name] = rows, checkpoints
        cache = raw_file(channel["cache_file"])
        exterior = saved_vector(cache, "exterior_eigenvalues")
        indices = saved_vector(cache, "positive_exterior_indices")
        assert len(exterior) == 192 and len(indices) == 96
        assert indices == channel["archive_indices"] == fc[(label, 0)]["archive_mode_indices"]
        assert all(exterior[i] > 0 for i in indices)
        for index, (a, b) in enumerate(zip([exterior[i] for i in indices],
                                         fc[(label, 0)]["archive_eigenvalues"], strict=True)):
            compare("exterior_positive_spectrum", name, 0, str(index), a, b)
        for k in range(41):
            a, b = rows[(k,)], fr[(label, k)]
            assert a["t"] == b["t"] == 2*k and b["w"] == weight
            assert b["sourcing"] == channel["sourcing"]
            assert set(a["collars"]) == {"80", "160"}
            assert set(b["collar"]) == {"R80", "R160"}
            compare("record", name, k, "E_inv", a["E_inv"], b["E_inv"])
            for cut in ("80", "160"):
                for quantity in ("Phi_E", "Phi_N"):
                    compare("record", name, k, quantity+"_R"+cut,
                            a["collars"][cut][quantity], b["collar"]["R"+cut][quantity])
        for k in range(0, 41, 4):
            a, b = checkpoints[(k,)], fc[(label, k)]
            assert a["t"] == b["t"] == 2*k and b["w"] == weight
            assert a["archive_columns"] == b["archive_columns"] == b["positive_levels"] == 96
            assert b["exterior_dimension"] == 192
            compare("half_gap", name, k, "exterior_half_gap", channel["exterior_half_gap"], b["half_gap"])
            for ka, kb in (("S_dense", "S_dense"), ("S_retained", "S_retained_channel"),
                           ("S_complement", "S_complement"), ("S_vac", "S_vac_archive"),
                           ("increment", "S_increment")):
                compare("entropy", name, k, ka, a[ka], b[kb], primary=(ka == "S_dense"))
            spectrum = saved_vector(raw_file(a["raw_file"]), "archive_eigenvalues")
            assert len(spectrum) == len(b["spectrum_AHCA"]) == 96
            for index, (x, y) in enumerate(zip(sorted(spectrum), sorted(b["spectrum_AHCA"]), strict=True)):
                compare("archive_covariance_spectrum", name, k, str(index), x, y)

    ow = unique(own["weighted_rows"], ("k",), {(k,) for k in range(41)})
    oe = unique(own["weighted_entropy"], ("k",), {(k,) for k in range(0, 41, 4)})
    foreign_totals = {}
    foreign_guard = {}
    for k in range(41):
        a, b = ow[(k,)], fw[(k,)]
        assert a["t"] == b["t"] == 2*k
        mappings = [("E_inv", {"heavy_w": a["E_heavy"], "light_sourcing_w": a["E_light"],
                                "light_guard_w": a["E_guard"]}, a["guard_over_sourcing"])]
        for cut in ("80", "160"):
            for quantity in ("Phi_E", "Phi_N"):
                value = a["collars"][cut][quantity]
                mappings.append((quantity+"_R"+cut, {"heavy_w": value["heavy"],
                    "light_sourcing_w": value["sourcing_light"], "light_guard_w": value["guard"]},
                    value["guard_over_sourcing"]))
        for field, groups, ratio in mappings:
            for group, value in groups.items():
                compare("weighted_record", "weighted", k, field+"."+group, value, b[field][group])
            ratios.append(dict(category="weighted_record", k=k, field=field, own=ratio,
                foreign_numerator=b[field]["light_guard_w"],
                foreign_denominator=b[field]["light_sourcing_w"],
                foreign_reported_ratio=b[field]["guard_over_sourcing"],
                acceptance="diagnostic only; no quotient pass or failure"))
        if k % 4 == 0:
            rows = [fc[(label, k)] for label, _ in CHANNELS.values()]
            for row in rows:
                assert row["weighted"] == row["w"] * row["S_dense"]
            total = sum(row["weighted"] for row in rows)
            guard = fc[("light3", k)]["weighted"]
            foreign_totals[k], foreign_guard[k] = total, guard/total
            compare("weighted_entropy", "weighted", k, "S_total", oe[(k,)]["S_total"], total)
            ratio = oe[(k,)]["guard_share"]
            compare("weighted_entropy", "weighted", k, "guard_numerator", ratio["numerator"], guard)
            compare("weighted_entropy", "weighted", k, "guard_denominator", ratio["denominator"], total)
            ratios.append(dict(category="weighted_entropy", k=k, field="guard_share", own=ratio,
                foreign_numerator=guard, foreign_denominator=total, foreign_reported_ratio=guard/total,
                acceptance="diagnostic only; no quotient pass or failure"))

    text = foreign_bytes["foreign_table"].decode()
    header = next(line[len("# columns: "):] for line in text.splitlines() if line.startswith("# columns: "))
    table_rows = list(csv.DictReader([header]+[line for line in text.splitlines()
                                              if line and not line.startswith("#")], delimiter="\t"))
    table = unique(table_rows, ("key", "step"), {(label, str(k)) for label, k in expected_checkpoints})
    for (label, step), row in table.items():
        k = int(step)
        target = fc[(label, k)]
        for field, value in row.items():
            if field == "key":
                assert value == target["label"]
                continue
            compare("foreign_table_roundtrip", label, k, field, float(value), target[field], tolerance=0)
    comments = {}
    pattern = re.compile(r"^# weighted step (\d+): S_esc total (\S+) guard share (\S+); E_inv heavy (\S+) light sourcing (\S+) guard (\S+); Phi_E R160 heavy (\S+) light sourcing (\S+) guard (\S+); Phi_E R80 heavy (\S+) light sourcing (\S+) guard (\S+)$")
    for line in text.splitlines():
        if not line.startswith("# weighted step "):
            continue
        match = pattern.fullmatch(line)
        assert match is not None
        k = int(match[1])
        assert k not in comments
        comments[k] = [float(x) for x in match.groups()[1:]]
    assert set(comments) == set(range(0, 41, 4))
    for k, values in comments.items():
        target = [foreign_totals[k], foreign_guard[k]]
        for field in ("E_inv", "Phi_E_R160", "Phi_E_R80"):
            target += [fw[(k,)][field][x] for x in ("heavy_w", "light_sourcing_w", "light_guard_w")]
        for index, (a, b) in enumerate(zip(values, target, strict=True)):
            compare("foreign_weighted_comment_roundtrip", "weighted", k, str(index), a, b, tolerance=0)

    maxima = {}
    for row in pairs:
        key = row["category"] + ":" + (row["field"] if "spectrum" not in row["category"] else "all_indices")
        if key not in maxima or row["absolute_difference"] > maxima[key]["absolute_difference"]:
            maxima[key] = row
    primary = [row for row in pairs if row["primary"]]
    assert len(primary) == 44
    result = dict(purpose="Saved-data comparison of the prescribed Computation C fixture only",
        input_bindings=bindings, script_sha256=digest(Path(__file__).read_bytes()),
        uniqueness_census_clocks_weights_cuts_passed=True,
        primary_definition="44 matched per-channel S_dense values at absolute 1e-10",
        primary_pass=all(row["passed"] for row in primary),
        primary_count=len(primary), pair_count=len(pairs), pairs=pairs,
        maxima=maxima, failures=[row for row in pairs if not row["passed"]],
        ratios=ratios, box_only_raw_files_verified=raw_bindings,
        route_controls=dict(own_all_controls_passed=own["all_controls_passed"],
                            foreign_pass=foreign["pass"], foreign_fails=foreign["fails"]),
        unavailable_external_comparisons=["site-source arrays", "reference columns and masks"],
        scope="No solver, eigensolve, operator reconstruction, propagation, averaging, tuning or physical-history clearance",
        execution=dict(host=socket.gethostname(), pid=os.getpid(), nice=os.getpriority(os.PRIO_PROCESS, 0),
                       wall_seconds=time.monotonic()-started,
                       peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Saved C-fixture comparison", "", result["primary_definition"]+".",
        f"Primary pass: {result['primary_pass']}; pairs: {len(pairs)}; separately located failures: {len(result['failures'])}.",
        "", "| Group and field | Maximum absolute difference | Channel | k |",
        "|---|---:|---|---:|"]
    for key, row in sorted(maxima.items()):
        lines.append(f"| {key} | {row['absolute_difference']:.17g} | {row['channel']} | {row['k']} |")
    lines += ["", "All signed pairs, route-specific controls, ratio conditioning and verified raw-file identities are in comparison.json.",
              "Raw arrays remain on the box. Quotient diagnostics confer no flux or entropy acceptance.",
              result["scope"]+"."]
    (args.output_dir / "comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"primary_pass": result["primary_pass"], "pairs": len(pairs),
                      "failures": len(result["failures"]), "output_bytes": output.stat().st_size,
                      "sha256": digest(output.read_bytes()), "execution": result["execution"]}))
    return 0 if result["primary_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
