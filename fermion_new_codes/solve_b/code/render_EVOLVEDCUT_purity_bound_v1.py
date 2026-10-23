"""Readable companion from saved upper-bound JSON; no bound recomputation."""
import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu

READ=Context(prec=14,rounding=ROUND_HALF_EVEN)
HIGH=Context(prec=10,rounding=ROUND_CEILING)
LOW=Context(prec=10,rounding=ROUND_FLOOR)


def upper(value):return str(HIGH.create_decimal(Decimal(value)))
def lower(value):return str(LOW.create_decimal(Decimal(value)))
def reading(value):return str(READ.divide(Decimal(value['numerator']),Decimal(value['denominator'])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require_remote_cpu(args.output.parent);source=args.input.read_bytes();data=json.loads(source)
    if len(data['rows'])!=60 or len(data['summaries'])!=6:raise ValueError('incomplete companion population')
    lines=['# Saved-complement purity bound: complete channel companion','',
           'Scope [N]: All sixty selected channel readings and six weighted comparisons from the frozen saved-data arithmetic. Every U is conditional on the explicitly stated original-operation model. U_C bounds the state-entropy error for the ideal range projector of the represented source columns, with exterior entropy read on its pure-state complement; U_H bounds the held negative-projector reference-entropy error. Their sum bounds the held-subtracted excess error. T12 is the charge-fallback control, not an applied trapping-root result. Upper bounds are rounded upward to ten significant digits for this table; displayed readings use fourteen significant digits. Exact rational readings, eighty-digit outward intervals, inputs and all intermediate terms remain in the [complete results JSON](<internal>/results.json). No eigensolver, covariance or entropy-trace rerun is performed by this renderer.','',
           '| T/r0 | κ | Shift / interior sites | Complement minus held | U_C | U_H | U_C + U_H | 16κ(U_C + U_H) |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['rows']:
        prefix=f"| {row['clock_rs']} | {row['kappa']} | {row['site_shift']:+d} / {row['interior_sites']} |"
        if row['availability']!='available_under_stated_model':lines.append(prefix+' unavailable | unavailable | unavailable | unavailable | unavailable |');continue
        if row['certified_from_saved_record_upper'] is not None:raise ValueError('unexpected unconditional certificate')
        lines.append(prefix+f" {reading(row['held_subtracted_reading_exact'])} | {upper(row['state_small']['conditional_model_upper']['upper'])} | {upper(row['held_small']['conditional_model_upper']['upper'])} | {upper(row['conditional_model_upper'])} | {upper(row['weighted_conditional_model_upper'])} |")
    lines+=['','All entries are nats. The complete weighted upper is a triangle sum; no cancellation or statistical scaling is used.','',
            '| T/r0 | Weighted complement minus held | Weighted U_C | Weighted U_H | Weighted upper | Gap interval including printed rounding | Conditional criterion | Unconditional certificate |',
            '|---:|---:|---:|---:|---:|---:|---|---|']
    for item in data['summaries']:
        if item['availability']!='available_under_stated_model':lines.append(f"| {item['clock_rs']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |");continue
        comparison=item['comparison'];gap=comparison['absolute_gap_including_display']
        lines.append(f"| {item['clock_rs']} | {reading(item['weighted_held_subtracted_exact'])} | {upper(item['weighted_state_upper']['upper'])} | {upper(item['weighted_held_upper']['upper'])} | {upper(item['conditional_model_upper'])} | [{lower(gap['lower'])}, {upper(gap['upper'])}] | {comparison['conditional_model_and_supplied_allowance_verdict']} | unavailable |")
    lines+=['','The conditional criterion compares the full gap interval with the supplied first-route 1.85e-7 allowance plus this route’s conditional model upper. It does not independently certify that supplied allowance, the original BLAS/libm hypotheses, physical history/reference roles, unmatched cuts or omitted channels. The earlier exterior-reading diagnostic failure remains unchanged.','',
            'SOURCES: [complete results](<internal>/results.json) (SHA256 {source_sha256}); [matched saved metadata](<internal>/matched_input_metadata.json); [frozen specification](<internal>/EVOLVEDCUT_purity_bound_spec_v1.md). The source JSON retains the specification and implementation hashes. Bulk provenance: `<internal>/results.json` and `<internal>/matched_input_metadata.json`.'.replace('{source_sha256}',hashlib.sha256(source).hexdigest()),'',
            'Renderer: '+str(Path(__file__))+' (SHA256 '+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+').','']
    args.output.write_text('\n'.join(lines));print(json.dumps({'channel_rows':60,'weighted_rows':6}),flush=True)


if __name__=='__main__':main()
