"""Format frozen evolved-cut fields; no contraction or scientific arithmetic."""
import argparse
import json
from pathlib import Path


def display(value):
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return format(value, ".12g")
    return str(value)


def table(headers, records):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
                "| " + " | ".join(display(v) for v in row) + " |"
                for row in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-root", required=True)
    args = parser.parse_args()
    results = json.loads((args.data / "results.json").read_text())["rows"]
    aggregates = json.loads((args.data / "tables.json").read_text())["rows"]
    clocks = [12, 15, 16, 20, 24, 25, 28, 30, 35, 40]
    assert len(results) == 100 and len(aggregates) == 30
    root = args.link_root.rstrip("/")
    lines = ["# Evolved-cut primary-channel values", "",
             "Scope [N]: Primary-cut values of the declared-array contraction, explicit polar control and independently constructed held/flat references. The T12 section is a charge-fallback control. Weighted columns use 16κ once. Unavailable entries retain the frozen covariance gates. Digits display saved arithmetic, not physical uncertainty. This finite-channel table supplies no angular-tail, physical-history, continued-reference or three-dimensional entropy determination.", "",
             f"SOURCES: {root}/results.json; {root}/tables.json.", "",
             f"[Complete frozen results]({root}/results.json); [complete frozen reductions]({root}/tables.json).", ""]
    for clock in clocks:
        records = sorted((r for r in results if r["clock_rs"] == clock),
                         key=lambda r: r["kappa"])
        assert [r["kappa"] for r in records] == list(range(1, 11))
        matches = [r for r in aggregates if r["clock_rs"] == clock and r["site_shift"] == 0]
        assert len(matches) == 1
        aggregate = matches[0]
        terms = sorted(aggregate["terms"], key=lambda r: r["kappa"])
        assert [r["kappa"] for r in terms] == list(range(1, 11))
        cuts = []
        for record in records:
            primary = [c for c in record["cuts"] if c["site_shift"] == 0]
            assert len(primary) == 1
            cuts.append(primary[0])
        for field in ["source_dtype", "cut_label", "radius_over_a", "area_over_a2", "eta_area"]:
            assert all(r[field] == records[0][field] for r in records)
        assert all(c["interior_sites"] == cuts[0]["interior_sites"] for c in cuts)
        lines += [f"## T/r0 = {clock}" + (" — charge_fallback control" if clock == 12 else ""), "",
                  "Metadata: " + "; ".join(f"{field} = {display(value)}" for field, value in [
                      ("cut_label", aggregate["cut_label"]),
                      ("source dtype", records[0]["source_dtype"]),
                      ("radius_over_a", aggregate["radius_over_a"]),
                      ("interior_sites", cuts[0]["interior_sites"]),
                      ("area_over_a2", aggregate["area_over_a2"]),
                      ("eta_area", aggregate["eta_area"])]) + ".", "",
                  "Table A: saved channel entropies (nats).", ""]
        lines += table(["κ", "Stored weight", "raw_global.valid", "Raw physical exterior", "Polar physical exterior", "Held", "Flat"],
                       [[r["kappa"], r["weight"], r["raw_global"]["valid"],
                         c["raw"]["physical_exterior_entropy"],
                         c["orthonormal_control"]["physical_exterior_entropy"],
                         c["held"]["entropy"], c["flat"]["entropy"]]
                        for r, c in zip(records, cuts)])
        lines += ["", "Table B: saved weighted terms (nats).", ""]
        fields = ["kappa", "raw", "orthonormal_control", "held", "flat", "raw_excess", "orthonormal_control_excess", "held_minus_flat"]
        lines += table(["κ"] + fields[1:], [[t[f] for f in fields] for t in terms])
        lines += ["", "Per-channel saved records: " + "; ".join(
            f"[κ{r['kappa']}]({root}/T{clock}_k{r['kappa']}/result.json)" for r in records) + ".", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
