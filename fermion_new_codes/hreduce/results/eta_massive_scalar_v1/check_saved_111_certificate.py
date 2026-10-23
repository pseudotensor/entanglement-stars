"""Check the stored V2.11 exact cover and sums without evaluating entropy."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path('<internal>/environment_continuation')


def endpoints(record):
    return tuple(Fraction(int(record[k]['numerator']), int(record[k]['denominator']))
                 for k in ('lower_exact', 'upper_exact'))


def main():
    source = ROOT / 'fixed_mass_scalar_result.json'
    data = source.read_bytes()
    result = json.loads(data)
    accepted = result['accepted_subcharts111']
    assert len(accepted) == 488
    seen, sums = set(), {}
    for link in accepted:
        piece, chart = link['piece'], link['subchart']
        assert (piece, chart) not in seen
        seen.add((piece, chart))
        attempt = result['attempts111'][link['attempt_index']]
        start, count = (1, 124) if piece < 2 else (2, 120)
        assert piece in range(4) and chart in range(count)
        assert attempt['piece'] == piece and attempt['subchart'] == chart
        assert attempt['accepted'] and attempt['subchart_count'] == count
        assert attempt['bits'] in (192, 256)
        left, right = map(Fraction, attempt['z'])
        assert left == start + Fraction(chart, 4) and right == left + Fraction(1, 4)
        lo, hi = endpoints(attempt['real'])
        il, ih = endpoints(attempt['imaginary'])
        assert lo <= hi and il <= 0 <= ih
        assert hi-lo <= Fraction(1, 50000 * count)
        a, b = sums.get(piece, (Fraction(0), Fraction(0)))
        sums[piece] = a+lo, b+hi
    assert seen == {(p, c) for p in range(4) for c in range(124 if p < 2 else 120)}
    assert len(result['pieces111']) == 4
    for record in result['pieces111']:
        assert record['accepted'] and endpoints(record['real']) == sums[record['piece']]
        assert sums[record['piece']][1]-sums[record['piece']][0] <= Fraction(1, 50000)
    omitted_lo, omitted_hi = endpoints(result['omitted_integral_bounds']['eta111_nonnegative'])
    assert 0 <= omitted_lo <= omitted_hi
    total = sum((x[0] for x in sums.values()), Fraction(0)), sum((x[1] for x in sums.values()), Fraction(0)) + omitted_hi
    final = result['results']['eta111_y1_25']
    assert endpoints(final) == total and total[1]-total[0] <= Fraction(1, 5000)
    assert Fraction(final['lower']) <= total[0] and Fraction(final['upper']) >= total[1]
    assert 'eta110_y1_25' not in result['results']
    assert 'eta_sphere_y1_25' not in result['results']
    report = dict(status='pass', scope='Exact saved 111 quarter-chart cover, widths, sums and one-sided omission only; no integrand or analytic-proof validation',
        accepted_subcharts=len(accepted), pieces=4,
        source_sha256=hashlib.sha256(data).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eta111_y1_25=final)
    with (ROOT / 'saved_111_check.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
