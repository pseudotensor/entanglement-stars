"""Independent ladder arithmetic; no foreign computation modules are imported.

The residual contract is the original reviewer B author's supplied recipe and the
sole review in redteam_LADDERS_v1p3_arithmetic_spec_v1.md. This first stage
preserves the per-record residuals, independent pass and both prefix masks.
"""
import argparse
import json
import math
from pathlib import Path


LIMIT = 1e-3
BAND = 1e-4


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def norm(value):
    return finite(value) and value >= 0


def prefix_step(state, local):
    if state is False or local is False:
        return False
    if state is None or local is None:
        return None
    return True


def field_change(entry):
    if not isinstance(entry, list) or len(entry) < 3 or not all(norm(x) for x in entry[1:3]):
        raise ValueError('missing finite dN/dv in history entry')
    return max(entry[1:3])


def residual(record, expect_newton=False):
    newton = record.get('solve_v32')
    if isinstance(newton, dict):
        kept, confirming = newton.get('kept_residual'), newton.get('confirming_residual')
        if not norm(kept) or not norm(confirming):
            raise ValueError('Newton record lacks explicit kept/confirming residual')
        return dict(kept=kept, confirming=confirming, contract='Newton explicit', kept_index=None)
    if expect_newton:
        raise ValueError('declared Newton record lacks explicit solve_v32')
    history = record.get('iteration_history')
    if not isinstance(history, list) or not history:
        raise ValueError('missing plain iteration history')
    if record.get('converged') is True:
        return dict(kept=field_change(history[-1]), confirming=None,
                    contract='plain converged final entry', kept_index=len(history)-1)
    raw = record.get('unconverged')
    if not isinstance(raw, dict):
        raise ValueError('unconverged plain record lacks kept identity')
    index, kept = raw.get('iteration'), raw.get('residual')
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(history) or not norm(kept):
        raise ValueError('invalid plain kept identity')
    check = field_change(history[index])
    # The saved scalar and its raw history entry must describe the same value.
    allowance = 32 * math.ulp(max(abs(kept), abs(check)))
    if abs(kept-check) > allowance:
        raise ValueError('plain kept residual/history identity mismatch')
    following = field_change(history[index+1]) if index+1 < len(history) else None
    return dict(kept=kept, confirming=following, kept_index=index,
                contract='plain following entry' if following is not None else 'plain kept last entry')


def classify(record, expect_newton=False):
    result = residual(record, expect_newton)
    kept, confirming = result['kept'], result['confirming']
    passes = kept <= LIMIT and (confirming is None or confirming <= LIMIT)
    maximum = max(kept, confirming) if confirming is not None else kept
    result.update(counts=passes, kept_pass=kept <= LIMIT,
                  banded=passes and BAND < maximum <= LIMIT,
                  reasons=([name for name, value in [('kept', kept), ('confirming', confirming)]
                            if value is not None and value > LIMIT]))
    return result


def known_answers():
    fixtures = [
        ('plain converged', dict(converged=True, iteration_history=[[None, 0, 0], [0, 1e-5, 2e-5]]),
         dict(kept=2e-5, confirming=None, counts=True, banded=False)),
        ('following fails', dict(converged=False, unconverged=dict(iteration=1, residual=5e-5),
          iteration_history=[[None, .1, .2], [0, 4e-5, 5e-5], [0, .002, .001]]),
         dict(kept=5e-5, confirming=.002, counts=False, banded=False)),
        ('Newton precedence', dict(converged=False, solve_v32=dict(kept_residual=2e-4, confirming_residual=3e-4),
          iteration_history=[[None, .2, .3]]),
         dict(kept=2e-4, confirming=3e-4, counts=True, banded=True)),
        ('inclusive upper', dict(converged=True, iteration_history=[[None, .001, 0]]),
         dict(kept=.001, confirming=None, counts=True, banded=True)),
        ('strict lower band', dict(converged=True, iteration_history=[[None, .0001, 0]]),
         dict(kept=.0001, confirming=None, counts=True, banded=False)),
        ('plain last kept', dict(converged=False, unconverged=dict(iteration=0, residual=.0003),
          iteration_history=[[None, .0003, .0002]]),
         dict(kept=.0003, confirming=None, counts=True, banded=True)),
    ]
    answers = []
    for name, record, expected in fixtures:
        actual = classify(record)
        okay = all(actual[key] == value for key, value in expected.items())
        answers.append(dict(name=name, expected=expected, actual=actual, passed=okay))
    malformed = dict(converged=False, unconverged=dict(iteration=0, residual=.0002),
                     iteration_history=[[None, .0003, .0001]])
    try:
        classify(malformed)
        rejected = False
    except ValueError:
        rejected = True
    answers.append(dict(name='inconsistent kept identity rejected', passed=rejected))
    for name, record, expected_newton in [
        ('negative norm rejected', dict(converged=True, iteration_history=[[None, -1, 0]]), False),
        ('missing Newton metadata rejected', dict(converged=True, iteration_history=[[None, 0, 0]]), True),
    ]:
        try:
            classify(record, expected_newton)
            rejected = False
        except ValueError:
            rejected = True
        answers.append(dict(name=name, passed=rejected))
    prefix, cumulative = [], True
    for value in [True, False, True]:
        cumulative = cumulative and value
        prefix.append(cumulative)
    answers.append(dict(name='later pass does not repair prefix', actual=prefix,
                        expected=[True, False, False], passed=prefix == [True, False, False]))
    unknown_chain=[]
    state=True
    for local in [True,None,True,False,True]:
        state=prefix_step(state,local)
        unknown_chain.append(state)
    answers.append(dict(name='unknown ancestry remains unknown until definite failure',
                        actual=unknown_chain,expected=[True,None,None,False,False],
                        passed=unknown_chain==[True,None,None,False,False]))
    next_above=math.nextafter(LIMIT,math.inf)
    boundary=classify(dict(converged=True,iteration_history=[[None,next_above,0]]))
    answers.append(dict(name='next binary64 above upper threshold fails',passed=boundary['counts'] is False))
    local_flags=[]; certified_flags=[]; kept_flags=[]; bands=[]; cstate=kstate=True
    for kept,confirming in zip([1e-4,1e-3,2e-4,2e-3,1e-5],[0,1e-3,2e-3,0,0]):
        classified=classify(dict(solve_v32=dict(kept_residual=kept,confirming_residual=confirming)))
        cstate=prefix_step(cstate,classified['counts']); kstate=prefix_step(kstate,classified['kept_pass'])
        local_flags.append(classified['counts']); certified_flags.append(cstate); kept_flags.append(kstate)
        bands.append(classified['banded'])
    answers.append(dict(name='specified five-record local and both-prefix control',
                        actual=dict(local=local_flags,certified=certified_flags,kept=kept_flags,banded=bands),
                        passed=local_flags==[True,True,False,False,True]
                        and certified_flags==[True,True,False,False,False]
                        and kept_flags==[True,True,True,False,False]
                        and bands==[False,True,False,False,False]))
    if not all(item['passed'] for item in answers):
        raise RuntimeError('known-answer failure')
    return answers


def member_series(path):
    source = json.loads(Path(path).read_text())
    raw_records = source['records']
    rows, dropped, ancestry = [], [], []
    rs, tau = source['args']['rs'], source['args']['tau']
    if not finite(rs) or not finite(tau) or rs <= 0 or tau <= 0:
        raise ValueError('nominal rs and tau must be finite and positive')
    head_rs = source.get('head', {}).get('rs_rest_label')
    if head_rs is not None and (not finite(head_rs) or head_rs != rs):
        raise ValueError('head.rs_rest_label differs from args.rs')
    expect_newton = source['args'].get('mix') == 'newton'
    tolerance = 256 * math.ulp(max(1.0, rs, tau,
                                *(abs(record['T']) for record in raw_records if finite(record.get('T')))))
    clock_errors = []
    previous_T = None
    for index, record in enumerate(raw_records):
        T, t = record.get('T'), record.get('T_over_rs')
        if not finite(T) or not finite(t):
            clock_errors.append(dict(index=index, reason='missing/nonfinite clock'))
            continue
        if abs(t-T/rs) > tolerance/rs:
            clock_errors.append(dict(index=index, reason='T_over_rs differs from T/nominal rs'))
        if index == 0 and abs(T) > tolerance:
            clock_errors.append(dict(index=index, reason='initial clock is not zero'))
        if previous_T is not None and abs(T-previous_T-tau) > tolerance:
            clock_errors.append(dict(index=index, reason='cadence differs from args.tau'))
        previous_T = T
    certified, kept_prefix = True, True
    for index, record in enumerate(raw_records):
        row = dict(source_index=index, T=record.get('T'), T_over_rs=record.get('T_over_rs'),
                   converged=record.get('converged'))
        try:
            row.update(classify(record, expect_newton))
        except ValueError as error:
            row.update(unresolved=str(error), counts=None, kept_pass=None, banded=None)
        certified = prefix_step(certified, row['counts'])
        kept_prefix = prefix_step(kept_prefix, row['kept_pass'])
        row.update(certified=certified if not clock_errors else None,
                   kept_prefix=kept_prefix if not clock_errors else None)
        ancestry.append(row)
        if not finite(record.get('E')):
            dropped.append(index)
            continue
        rows.append(row)
    times = [row['T_over_rs'] for row in rows]
    monotone = all(finite(t) for t in times) and all(a < b for a, b in zip(times, times[1:]))
    if not monotone:
        raise ValueError('nonfinite, duplicate or nonmonotone clocks: ' + path)
    first_failure = next((row for row in ancestry if row['counts'] is not True), None)
    last_certified = next((row['T_over_rs'] for row in reversed(rows) if row['certified']), None)
    last_kept = next((row['T_over_rs'] for row in reversed(rows) if row['kept_prefix']), None)
    summary = dict(records=len(rows), dropped_without_E=dropped, first=times[0], last=times[-1],
                   counting=sum(row['counts'] is True for row in rows),
                   unresolved=sum('unresolved' in row for row in rows),
                   stored_converged=sum(row['converged'] is True for row in rows),
                   last_certified=last_certified, last_kept_prefix=last_kept,
                   first_failure=first_failure,
                   banded_inside=[row['T_over_rs'] for row in rows if row['banded'] and row['certified']],
                   banded_outside=[row['T_over_rs'] for row in rows if row['banded'] and row['certified'] is False])
    return dict(source=path, clock_contract=dict(nominal_rs=rs, tau=tau, tolerance_T=tolerance,
                errors=clock_errors), summary=summary, rows=rows, ancestry=ancestry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    answers = known_answers()
    (output/'known_answers.json').write_text(json.dumps(answers, indent=2)+'\n')
    if args.controls_only:
        return
    paths = json.loads(Path(args.inputs).read_text())
    members = {}
    for name, path in paths.items():
        members[name] = member_series(path)
        (output/'residual_series.json').write_text(json.dumps(members, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(member=name, **members[name]['summary'])), flush=True)


if __name__ == '__main__':
    main()
