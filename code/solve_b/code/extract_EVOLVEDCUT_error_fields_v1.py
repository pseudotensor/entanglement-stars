"""Extract saved matched-partition entropy diagnostics; no entropy recomputation."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.input.read_text())
    selection = [(12, 0, 58), (15, 1, 51), (20, 1, 38),
                 (28, 1, 37), (35, 1, 37), (40, 1, 36)]
    rows, summaries = [], []
    for clock, shift, sites in selection:
        records = sorted((r for r in saved['rows'] if r['clock_rs'] == clock),
                         key=lambda r: r['kappa'])
        assert [r['kappa'] for r in records] == list(range(1, 11))
        assert sum(r['weight'] for r in records) == 880
        group = []
        for record in records:
            matches = [c for c in record['cuts'] if c['site_shift'] == shift]
            assert len(matches) == 1
            cut = matches[0]
            assert cut['interior_sites'] == sites
            polar = cut['orthonormal_control']
            ext = polar['physical_exterior_entropy']
            interior = polar['physical_interior_entropy']
            assert polar['exterior']['valid'] and polar['interior']['valid']
            assert ext is not None and interior is not None
            assert cut['held']['valid'] and cut['flat']['valid']
            delta = ext - interior
            row = {'clock_rs': clock, 'site_shift': shift,
                   'kappa': record['kappa'], 'weight': record['weight'],
                   'interior_sites': sites, 'source_dtype': record['source_dtype'],
                   'raw_global_valid': record['raw_global']['valid'],
                   'raw_physical_exterior_entropy': cut['raw']['physical_exterior_entropy'],
                   'raw_physical_interior_entropy': cut['raw']['physical_interior_entropy'],
                   'polar_physical_exterior_entropy': ext,
                   'polar_physical_interior_entropy': interior,
                   'delta_exterior_minus_interior': delta,
                   'absolute_delta': abs(delta),
                   'larger_side_error_lower_bound_if_common_ideal_target': abs(delta)/2,
                   'held_entropy': cut['held']['entropy'],
                   'flat_entropy': cut['flat']['entropy']}
            group.append(row)
        rows.extend(group)
        weighted = lambda field: math.fsum(r['weight']*r[field] for r in group)
        ext = weighted('polar_physical_exterior_entropy')
        interior = weighted('polar_physical_interior_entropy')
        held, flat = weighted('held_entropy'), weighted('flat_entropy')
        signed_delta = weighted('delta_exterior_minus_interior')
        summaries.append({'clock_rs': clock, 'site_shift': shift,
                          'interior_sites': sites, 'channel_count': len(group),
                          'weight_sum': sum(r['weight'] for r in group),
                          'weighted_polar_exterior': ext,
                          'weighted_polar_interior': interior,
                          'weighted_held': held, 'weighted_flat': flat,
                          'weighted_signed_delta': signed_delta,
                          'weighted_absolute_delta_E2_empirical': weighted('absolute_delta'),
                          'larger_weighted_side_error_lower_bound_if_common_ideal_target': abs(signed_delta)/2,
                          'minimum_delta': min(r['delta_exterior_minus_interior'] for r in group),
                          'maximum_delta': max(r['delta_exterior_minus_interior'] for r in group),
                          'minimum_absolute_delta': min(r['absolute_delta'] for r in group),
                          'maximum_absolute_delta': max(r['absolute_delta'] for r in group),
                          'weighted_exterior_minus_held': ext-held,
                          'weighted_interior_minus_held': interior-held})
    assert len(rows) == 60
    assert rows[0]['polar_physical_exterior_entropy'] == 2.313838679224235
    assert rows[0]['polar_physical_interior_entropy'] == 2.3138386761991154
    result = {'scope': 'Saved polar-control fields and arithmetic diagnostics only. E2 is empirical, not a certified upper error. Half differences are conditional lower bounds on the larger side error, never exterior upper bounds.',
              'source': str(args.input), 'controls': {'unique_rows': 60,
              'channels_per_clock': 10, 'weight_sum_per_clock': 880,
              'site_counts_and_T12_k1_values': 'passed'},
              'rows': rows, 'summaries': summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'controls': result['controls'], 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
