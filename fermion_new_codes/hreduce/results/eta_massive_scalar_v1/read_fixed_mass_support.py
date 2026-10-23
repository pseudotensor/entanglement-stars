"""Outward two-anchor support readout from saved scalar certificates only."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path('<internal>/environment_continuation')
AXIAL = ROOT.parent / 'stable_result.json'
TILTED = ROOT / 'fixed_mass_scalar_result.json'
AXIAL_SHA = '71c5df353b17ecde07d69872c611ce8c998fa419243d5151f71ca1713213bf52'
TILTED_SHA = '0ff742d0ad69a7ce303217de02166411a4e77c2e3fab24852de127e588aa2768'


def checked(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Changed saved input: ' + str(path))
    return json.loads(data)


def endpoints(record):
    return [arb(int(record[k]['numerator'])) / int(record[k]['denominator'])
            for k in ('lower_exact', 'upper_exact')]


def rational(x):
    q = x.fmpq()
    return Fraction(int(q.p), int(q.q))


def decimal(q, lower):
    scale = 10**30
    n = q.numerator*scale // q.denominator if lower else -((-q.numerator*scale)//q.denominator)
    digits = str(abs(n)).zfill(31)
    return ('-' if n < 0 else '') + digits[:-30] + '.' + digits[-30:]


def interval(lo, hi):
    a, b = rational(lo.lower()), rational(hi.upper())
    if a > b:
        raise ArithmeticError('Reversed support interval')
    return dict(lower=decimal(a, True), upper=decimal(b, False),
        lower_exact=dict(numerator=str(a.numerator), denominator=str(a.denominator)),
        upper_exact=dict(numerator=str(b.numerator), denominator=str(b.denominator)))


def inner(a, t):
    c = t/SQRT3
    s = a/c-1
    z = (s*s+2).sqrt()
    return 3*c/PI*(PI/2-2*s.atan()+2*z*(s/z).atan())


def outer(a, t):
    return 3*a/2+(SQRT3*t-3*a)*M3


def main():
    global arb, PI, SQRT3, M3
    import flint
    from flint import arb, ctx
    ctx.prec, ctx.threads = 192, 1
    PI, SQRT3 = arb.pi(), arb(3).sqrt()
    M3 = arb(3)/2-3*arb(2).sqrt()/PI*(1/SQRT3).acos()
    axial = checked(AXIAL, AXIAL_SHA)['results']['eta100_y1_25']
    tilted_record = checked(TILTED, TILTED_SHA)
    tilted = tilted_record['results']['eta111_y1_25']
    a0, a1 = endpoints(axial)
    t0, t1 = endpoints(tilted)
    for a in (a0, a1):
        for t in (t0, t1):
            s = a/(t/SQRT3)-1
            if not (s >= 0 and s <= 1 and SQRT3*t >= 2*a and SQRT3*t <= 3*a):
                raise ArithmeticError('Two-anchor body is outside the specified geometric regime')
    cube_inner, cube_outer = inner(arb(1), SQRT3), outer(arb(1), SQRT3)
    if not cube_inner.contains(arb(3)/2) or not cube_outer.contains(arb(3)/2):
        raise ArithmeticError('Cube mean-width normalization check failed')
    inner_ball, outer_ball = inner(a0, t0), outer(a1, t1)
    for value in (inner_ball, outer_ball):
        if rational(value.upper())-rational(value.lower()) >= Fraction(1, 10**40):
            raise ArithmeticError('Support-expression arithmetic width exceeds its budget')
    lower, upper = inner_ball.lower(), outer_ball.upper()
    if not 0 < lower < upper:
        raise ArithmeticError('Nonpositive or reversed sphere bounds')
    # Conservative outward certified anchor-only massless enclosure; no
    # empirical 205-direction numerator is used in this ratio.
    massless_lo, massless_hi = arb('0.8164'), arb('1.0203')
    output = dict(status='complete', scope='Certified two-anchor fixed-y support readout; not the requested 561/1326-direction production',
        y='1/25', implementation=dict(python_flint=flint.__version__, bits=ctx.prec),
        inputs=dict(axial_path=str(AXIAL), axial_sha256=AXIAL_SHA,
                    tilted_path=str(TILTED), tilted_sha256=TILTED_SHA,
                    eta100=axial, eta111=tilted,
                    massless_sphere=['0.8164', '1.0203']),
        support_inner_expression=interval(inner_ball, inner_ball),
        support_outer_expression=interval(outer_ball, outer_ball),
        eta_sphere_y1_25=interval(lower, upper),
        eta0_over_eta_y1_25=interval(massless_lo/upper, massless_hi/lower),
        formal_fixed_y_spacing_over_lP=interval(2*lower.sqrt(), 2*upper.sqrt()),
        massless_spacing_over_lP=interval(2*massless_lo.sqrt(), 2*massless_hi.sqrt()),
        cube_inner=interval(cube_inner, cube_inner), cube_outer=interval(cube_outer, cube_outer),
        missing=['eta110 certificate', 'b110 certificate', '561/1326 certified direction sets',
                 'fixed physical mass implicit match', 'evolved history coefficient'])
    output['producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with (ROOT / 'two_anchor_support_result.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
