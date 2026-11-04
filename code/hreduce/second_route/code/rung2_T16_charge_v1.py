"""One composite-record stored-charge normalization; no field or gate solve."""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time

ROOT = Path('<internal>/rung2_T16_charge_v1')
OWN = Path('<internal>/member_06.json')
ORIGINAL = Path('<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json')
PINS = {OWN: ('5d5e8ca183ec492d7580b732c09cb7bb606b02385b391c4482cae977a5810c38', 3553788), ORIGINAL: ('a8faf679cbd196be868b76a2cb039d06b55a19747ad6216b8b6cc38dd2094a23', 7014125)}
TAG = 'c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d'
SIGNALS = {signal.SIGALRM, signal.SIGXCPU}
SCALE = 10**30
OUTPUTS = ('primary_charges.json', 'rung2_T16_charge_result.json')


class Number(str):
    """An unmodified JSON number token, distinct from a JSON string."""


def encode(value):
    if isinstance(value, Number):
        # Input constants are rejected by the parser; tokens remain valid JSON.
        return str(value)
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(str(k)) + ':' + encode(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(encode(v) for v in value) + ']'
    return json.dumps(value, allow_nan=False)


def reject_constant(token):
    raise ValueError('Nonfinite JSON constant: ' + token)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def rational(token):
    if not isinstance(token, Number):
        raise ValueError('Required primary input is not a JSON number token')
    return Fraction(str(token))


def identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def load_pinned(path, sources):
    with path.open('rb') as stream:
        before = identity(os.fstat(stream.fileno()))
        raw = stream.read()
        after = identity(os.fstat(stream.fileno()))
    if before != after or identity(path.stat()) != after:
        raise ValueError('Input identity changed: ' + str(path))
    sha = hashlib.sha256(raw).hexdigest()
    if (sha, len(raw)) != PINS[path]:
        raise ValueError('Input pin mismatch: ' + str(path))
    sources.append(dict(path=str(path), sha256=sha, **before))
    return json.loads(raw, parse_int=Number, parse_float=Number,
                      parse_constant=reject_constant, object_pairs_hook=no_duplicates)


def atomic(name, value):
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    try:
        path = ROOT / name
        temporary = ROOT / (name + '.tmp')
        with temporary.open('x') as stream:
            stream.write(encode(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


@contextmanager
def protected_transition():
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def directed(value):
    n, d = value.numerator, value.denominator
    lower = (SCALE * n) // d
    upper = -((-SCALE * n) // d)
    result = dict(numerator=str(n), denominator=str(d),
                  lower_integer=str(lower), upper_integer=str(upper), scale=str(SCALE))
    valid = d > 0 and math.gcd(n, d) == 1 and lower*d <= SCALE*n <= upper*d and 0 <= upper-lower <= 1
    return result, valid


def arithmetic_controls():
    rows = []
    for x, y, expected, deficit in ((3, 2, Fraction(3, 2), Fraction(-1, 2)),
                                  (0, 2, Fraction(0), Fraction(1)),
                                  (-1, 2, Fraction(-1, 2), Fraction(3, 2)),
                                  (7, 7, Fraction(1), Fraction(0))):
        value = Fraction(x, y)
        complement = 1 - value
        a, av = directed(value)
        b, bv = directed(complement)
        if value != expected or complement != deficit or value*y != x or not av or not bv:
            raise AssertionError('Fixed scalar arithmetic control failed')
        rows.append(dict(x=x, y=y, ratio=a, deficit=b, validated=True))
    for token in ('NaN', 'Infinity', '-Infinity'):
        try:
            json.loads(token, parse_constant=reject_constant)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite input was admitted')
    for y in (Fraction(0), Fraction(-2)):
        try:
            positive_denominator(y)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonpositive denominator was admitted')
    return dict(fixed_cases=rows, nonfinite_rejected=True, nonpositive_denominator_rejected=True)


def positive_denominator(value):
    if value <= 0:
        raise ValueError('Preparation charge must be positive')


def join(records, own=False):
    matches = [(i, row) for i, row in enumerate(records)
               if rational(row['T']) == 1024 and rational(row['T_over_rs']) == 16]
    if len(records) != 321 or len(matches) != 1 or matches[0][0] != 128:
        raise ValueError('Record count or unique exact clock/index join failed')
    row = matches[0][1]
    if own and rational(row['index']) != 128:
        raise ValueError('Own saved record index failed')
    return row


def main():
    if Path(__file__).resolve().parent != ROOT:
        raise RuntimeError('Run only in the named box output root')
    if any((ROOT / name).exists() or (ROOT / (name + '.tmp')).exists() for name in OUTPUTS):
        raise FileExistsError('Existing product requires assessment; no retry')
    if shutil.disk_usage('/').free < 100_000_000_000 or shutil.disk_usage('/data').free < 100_000_000_000:
        raise RuntimeError('Required disk reserve unavailable')
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [cpus[0]])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 25))
    def deadline(signum, frame):
        raise TimeoutError('Fixed scalar normalization allocation exhausted')
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGXCPU, deadline)
    signal.alarm(25)
    start = time.monotonic()
    report = dict(status='incomplete', phase='inputs', sources=[], saved=[], validated=[],
                  scope='Stored-primary decimal arithmetic only; no independent charge solve or history certificate',
                  remaining_work=['Independent native Gauss-row and confirming-map extents',
                                  'Energy-centroid definition', 'Separately bound remaining rung and averaging controls'])
    def save():
        products = {'primary_charges': 'primary_charges.json',
                    'ratio': 'rung2_T16_charge_result.json:ratio',
                    'deficit': 'rung2_T16_charge_result.json:deficit'}
        report['missing_scalar_outputs'] = [path for key, path in products.items() if key not in report['saved']]
        report['missing_scalar_validations'] = [key for key in ('primary_binding', 'fixed_arithmetic_controls', 'ratio', 'deficit') if key not in report['validated']]
        report.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic(OUTPUTS[1], report)
    try:
        own = load_pinned(OWN, report['sources'])
        original = load_pinned(ORIGINAL, report['sources'])
        row, first = join(own['records'], True), join(original['records'])
        x_token, y_token = row['record_E'], original['head']['E0']
        inherited = {k: first[k] for k in ('record_label', 'converged', 'unconverged',
                    'continuation', 'gauge', 'indicator', 'solve_v32', 'solver')}
        with protected_transition():
            primary = dict(sources=report['sources'], own_member=own['member'],
                own_source=own['source'], own_json_sha256=own['json_sha256'],
                record_index=128, T=row['T'], T_over_rs=row['T_over_rs'],
                numerator_decimal_token=str(x_token), original_numerator_decimal_token=str(first['E']),
                denominator_decimal_token=str(y_token),
                numerator_source='member_06.records[128].record_E', denominator_source='original.head.E0',
                args={k: original['args'][k] for k in ('m','g','rs','tau','ell','prep_ell','nsites',
                    'kappa_sea','reference','site_fields','mix','tag','resume_from','resume_T_over_rs')},
                preparation={k: original['head'][k] for k in ('E0','label','prep_ell','evolution_ell','resume_slice_v34')},
                container_script=original['script'], record_engine=first['engine'], record_label=first['label'],
                inherited_controls=inherited, own_raw_gates=row['raw_gates'])
            # Preserve primary tokens/metadata before division or later binding assertions.
            atomic(OUTPUTS[0], primary)
            report['saved'].append('primary_charges')
            report['primary'] = primary
            report['phase'] = 'binding'
            save()
        if own['source']['path'] != str(ORIGINAL) or own['json_sha256'] != PINS[ORIGINAL][0] or own['member'] != 'P64\\_s2 composite (321; 40)' or rational(own['record_count']) != 321:
            raise ValueError('Own source/member linkage failed')
        expected = dict(m=Fraction(1,5),g=Fraction(1,4),rs=64,tau=8,ell=2,prep_ell=4,nsites=2048,kappa_sea=10)
        if any(rational(original['args'][k]) != v for k,v in expected.items()):
            raise ValueError('Original numerical argument binding failed')
        if any(original['args'][k] != v for k,v in dict(reference='R2',site_fields='averaged',mix='newton',tag=TAG).items()):
            raise ValueError('Original text argument binding failed')
        restart = original['head']['resume_slice_v34']
        if original['script'] != 'engine_collapse_v34.py' or first['engine'] != 'engine_collapse_v25' or rational(restart['T_over_rs']) != 17 or rational(restart['record_index']) != 136 or restart['slice'] != '<internal>/orbitals_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_T0017.00.npz':
            raise ValueError('Composite inherited-prefix binding failed')
        x, y = rational(x_token), rational(y_token)
        if x != rational(first['E']):
            raise ValueError('Saved numerator differs from original decimal primary')
        positive_denominator(y)
        report['validated'].append('primary_binding')
        report['arithmetic_controls'] = arithmetic_controls()
        report['validated'].append('fixed_arithmetic_controls')
        report['phase'] = 'ratio'
        with protected_transition():
            value = x/y
            ratio, ratio_valid = directed(value)
            report['ratio'] = ratio
            report['saved'].append('ratio')
            save()
        if not ratio_valid or value*y != x:
            raise AssertionError('Actual ratio reducedness, cross multiplication or directed interval failed')
        report['validated'].append('ratio')
        report['phase'] = 'deficit'
        with protected_transition():
            complement = 1-value
            deficit, deficit_valid = directed(complement)
            report['deficit'] = deficit
            report['saved'].append('deficit')
            save()
        if not deficit_valid or complement+value != 1:
            raise AssertionError('Actual signed complement control failed')
        report['validated'].append('deficit')
        report.update(status='completed_rung2_T16_charge_normalization_only', phase='complete')
    except Exception as error:
        report.update(status='incomplete', error=type(error).__name__+': '+str(error))
    finally:
        signal.alarm(0)
        save()
    print(json.dumps(dict(status=report['status'], saved=report['saved'], validated=report['validated'])))
    if report['status'] != 'completed_rung2_T16_charge_normalization_only':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
