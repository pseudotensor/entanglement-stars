#!/usr/bin/env python3
"""Separate typed serialization repair; never change the historical consumers.

The c15 ledger16 numeric payload is reused byte for byte. Only its five lost
kernel-statistic records are regenerated. Other operations are the remaining
prescribed original stages, once only. No physical row is released.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

BASE = Path('<internal>/second_route_bulk')
OLD = BASE / 'holds/k3_source/hreduce_readouts_v1'
NEW = BASE / 'holds/k3_source/hreduce_readouts_repair_v1'
CODE = BASE / 'hreduce_code/recovered_readouts_v1'
sys.path.insert(0, str(CODE))

import numpy as np
import hold_recovered_diagnostics as diag
from hold_recovered_readout_input import ReadoutStages, load_readout
from hold_recovered_spectrum import binding_hash, finite_matrix, promote
from hold_recovered_origin import require_finite
from hold_source_preflight import identity
from hold_exact import register_bulk, write_json
from hold_error_bounds import gamma_rational, rational, sqrt_upper, upwards
from hold_origin_diagnostics import derived, value_record
from hold_observables import (current_from_energy_kernel, interval_kernel,
    kernel_statistics, phase_arithmetic_model)

VERSION = 'recovered-selected-readouts-explicit-serialization-repair-1'
PINS = {
    'c15/prepare/prepare_progress.json':
        'eb702db976998c1bfceb9983688f3db9eccde5b0a35e11077710e70c39a4f718',
    'c15/diagnostics/diagnostics_progress.json':
        '63f2266373f9b095673193dd3da77310df08f1db20cec6fa79a31d27ccc5eb45',
    'c15/diagnostics/ledger_16_arrays.partial.npz':
        '428ff35783f873743d53af9b0f8a5f23ed98afe870c6e9a3b3f09b8c941ababd',
    'c15/diagnostics/ledger_16_arrays.partial.metadata.json':
        'd6df9f86bc979b7cc529afb89e879b3c67c287b76f4376c023f9561bc526be04',
    'execution/c15_prepare_exit.json':
        '05306f72931125c900227a2b2800be9183f7a411a0392c7333ba8bc31a0677ee',
    'execution/c15_diagnostics_exit.json':
        'a71a4dcc8c9933f9798b51fbaab75ddf62fc03ad85b5470a9461f3c6909cbd77',
}
LEGACY_NAMES = tuple(diag.INVENTORY)[:7]
PARTIAL_NAME = 'ledger_16_arrays.partial.npz'
PARENT_SUFFIX = '_origin_primary_block_partials_parent'


def exact_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def metadata_paths(context, preparation, cut):
    return {kind + PARENT_SUFFIX: diag.prepared_path(context, preparation,
        'origin', f'origin_cut{cut}_{kind}.npz').as_posix()
        for kind in diag.KINDS}


def check_text(key, value, expected):
    if value.shape != () or value.dtype.kind != 'U' or value.item() != expected:
        raise ValueError('Unexpected/non-scalar text metadata: ' + key)


def typed_writer(context, preparation):
    """Explicit replacement for future NPZ saves; all numeric checks remain."""
    def save(path, binding, **arrays):
        stage = binding['stage']
        text = metadata_paths(context, preparation, int(stage[7:])) if stage in (
            'ledger_16', 'ledger_30') else {}
        if set(arrays) & set(text) != set(text):
            raise ValueError('A ledger is missing its three declared parent paths')
        temporary = path.with_suffix('.partial.npz')
        with temporary.open('xb') as stream:
            np.savez(stream, input_binding_json=json.dumps(binding, sort_keys=True), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as saved:
            if len(saved.files) != len(arrays) + 1 or set(saved.files) != set(arrays) | {'input_binding_json'}:
                raise ValueError('Unexpected or repeated serialized member')
            check_text('input_binding_json', saved['input_binding_json'], json.dumps(binding, sort_keys=True))
            for key, original in arrays.items():
                original, restored = np.asarray(original), saved[key]
                if (restored.shape != original.shape or restored.dtype != original.dtype
                        or not restored.dtype.isnative):
                    raise ValueError('A typed product changed native layout/precision')
                if key in text:
                    check_text(key, original, text[key])
                    check_text(key, restored, text[key])
                else:
                    if restored.dtype.kind not in 'biufc':
                        raise ValueError('Only the declared parent paths may be text')
                    if restored.ndim == 2:
                        finite_matrix(restored, original.shape, original.dtype)
                        for start in range(0, len(restored), 128):
                            if not exact_bits(original[start:start+128], restored[start:start+128]):
                                raise ValueError('A numeric round trip changed stored bits')
                    else:
                        require_finite(restored)
                        if not exact_bits(original, restored):
                            raise ValueError('A numeric round trip changed stored bits')
        promote(temporary, path)
        if path.stat().st_size > 20 * 2**20:
            register_bulk(path)
        return identity(path)
    return save


def legacy_parent(context, preparation_id):
    parents = {}
    for name, pin in PINS.items():
        saved = context.source.inputs.add(OLD / name)
        if saved['sha256'] != pin:
            raise ValueError('A fixed historical recovery parent changed: ' + name)
        parents[name] = saved
    report = json.loads((OLD / 'c15/diagnostics/diagnostics_progress.json').read_text())
    expected_binding = dict(common=context.binding, consumer='diagnostics',
                            extra=dict(accepted_preparation=preparation_id))
    if (report['method'] != diag.READOUT_VERSION or report['consumer'] != 'diagnostics'
            or report['completed'] is not False or report['physical_rows_released'] is not False
            or report['input_binding'] != expected_binding
            or report['input_binding_hash'] != binding_hash(expected_binding)
            or list(report['stages']) != [*LEGACY_NAMES, 'ledger_16']):
        raise ValueError('The immutable failed parent has another exact association')
    allowed = {'diagnostics_progress.json', PARTIAL_NAME, 'ledger_16_arrays.partial.metadata.json'}
    prior = {}
    for name in LEGACY_NAMES:
        stage = report['stages'][name]
        expected = dict(common=expected_binding, stage=name, prior_completed_products=copy.deepcopy(prior))
        if (stage['status'] != 'completed' or set(stage['outputs']) != set(diag.INVENTORY[name])
                or stage['prospective_binding'] != expected):
            raise ValueError('The historical completed prefix is not exact')
        for filename, saved in stage['outputs'].items():
            if Path(saved['path']) != OLD / 'c15/diagnostics' / filename:
                raise ValueError('A historical prefix output has another role/path')
            context.source.inputs.verify(saved)
            allowed.add(filename)
        prior[name] = stage['outputs']
    failed = report['stages']['ledger_16']
    if (failed['status'] != 'failed_or_uncertain' or failed['exception_type'] != 'TypeError'
            or 'isfinite' not in failed['stop_reason']
            or failed['prospective_binding'] != dict(common=expected_binding,
                stage='ledger_16', prior_completed_products=prior)):
        raise ValueError('The fixed serialization failure changed')
    if {p.name for p in (OLD / 'c15/diagnostics').iterdir()} != allowed:
        raise FileExistsError('An unassessed legacy product appeared')
    return report, parents


class RepairStages(ReadoutStages):
    """New journal; only the seven specifically bound legacy outputs may be external."""
    def __init__(self, context, root, resume, extra, legacy=None):
        self.legacy = legacy
        super().__init__(context, root, diag.INVENTORY, 'diagnostics', resume, extra)
        # A process may stop after the superclass publishes its empty journal.
        # Resume may seed that still-empty journal, but must never replace any
        # nonempty completed prefix or uncertain operation.
        if legacy is not None and not self.report['stages']:
            self.report['stages'] = {name: copy.deepcopy(legacy['stages'][name]) for name in LEGACY_NAMES}
            self.report['legacy_stage_names'] = list(LEGACY_NAMES)
            write_json(self.path, self.report)
        self.validate()

    def validate(self):
        stages = self.report['stages']
        if list(stages) != list(self.inventory)[:len(stages)]:
            raise ValueError('Continuation stages are not an exact completed prefix')
        allowed, prior = {self.path.name}, {}
        for name, stage in stages.items():
            imported = self.legacy is not None and name in LEGACY_NAMES
            if imported:
                if stage != self.legacy['stages'][name]:
                    raise ValueError('An imported legacy stage was edited')
            elif stage.get('prospective_binding') != dict(common=self.binding, stage=name,
                                                          prior_completed_products=prior):
                raise ValueError('A new stage lacks the explicit adapter association')
            if stage['status'] != 'completed' or set(stage['outputs']) != set(self.inventory[name]):
                raise RuntimeError('An uncertain continuation requires assessment, never repetition')
            for filename, saved in stage['outputs'].items():
                parent = OLD / 'c15/diagnostics' if imported else self.root
                if Path(saved['path']) != parent / filename:
                    raise ValueError('A continuation output has another parent/role')
                self.context.source.inputs.verify(saved)
                if not imported:
                    allowed.add(filename)
            prior[name] = stage['outputs']
        if {p.name for p in self.root.iterdir()} != allowed:
            raise FileExistsError('An orphan continuation final/partial/statistic remains')


class SavedLedger:
    """Small saved numeric payload plus five missing statistics; no observable sums."""
    def __init__(self, context, preparation, legacy, stages, e):
        self.context, self.preparation, self.stages = context, preparation, stages
        self.original = OLD / 'c15/diagnostics' / PARTIAL_NAME
        self.parent_binding = legacy['stages']['ledger_16']['prospective_binding']
        paths = metadata_paths(context, preparation, 16)
        schema = {'eigenvalues': ('<f8', (16384,)), 'elapsed': ('<f8', (2,)),
            'native_current_integral': ('<c16', ()),
            'positive_inventory_matching_integral': ('<c16', ())}
        for name in ('signed', 'particle', 'hole', 'native_current', 'positive_inventory', 'localization_offset'):
            schema[name + '_endpoints'] = ('<c16', (2,))
        for prefix in ('particle_crossing', 'hole_crossing', 'native_current'):
            for suffix, dtype, shape in (('_primary_block_partials','<f8',(128,2)),
                    ('_paired_row_partials','<c16',(16384,)), ('_primary','<c16',()), ('_paired','<c16',())):
                schema[prefix + suffix] = (dtype, shape)
        text = {'input_binding_json': json.dumps(self.parent_binding, sort_keys=True), **paths}
        with np.load(self.original, allow_pickle=False) as saved:
            if len(saved.files) != 26 or set(saved.files) != set(schema) | set(text):
                raise ValueError('The saved ledger does not have its exact26-member inventory')
            self.arrays = {key: saved[key] for key in saved.files}
        for key, expected in text.items():
            array = self.arrays[key]
            check_text(key, array, expected)
            if array.dtype != np.dtype('<U' + str(len(expected))) or not array.dtype.isnative:
                raise ValueError('The exact saved Unicode width/byte order differs')
        for key, (dtype, shape) in schema.items():
            array = self.arrays[key]
            if array.dtype != np.dtype(dtype) or array.shape != shape or not array.dtype.isnative:
                raise ValueError('The saved ledger numeric schema differs: ' + key)
            require_finite(array)
        if not exact_bits(self.arrays['eigenvalues'], e) or not exact_bits(self.arrays['elapsed'], np.array([0.,128.])):
            raise ValueError('The saved exact eigenvalue order/elapsed interval differs')
        self.statistics = {}
        self.scratch = stages.root / 'ledger_16_statistics.partial.json'
        self.metadata = {}

    def prepared_kernel(self, context, preparation, stage, name, e):
        if stage == 'cut16_signed':
            # No signed matrix is needed: its endpoints/statistics were already saved.
            with np.load(diag.prepared_path(context, preparation, stage, name), allow_pickle=False) as raw:
                order = raw['eigenvalues']
                metadata = json.loads(raw['input_binding_json'].item())['kernel_metadata']
            if not exact_bits(order, e):
                raise ValueError('The signed metadata has another exact eigenvalue order')
            return None, metadata
        return diag.prepared_kernel(context, preparation, stage, name, e)

    def statistic(self, matrix, name):
        if name in self.statistics:
            raise RuntimeError('A missing statistic must be regenerated only once')
        value = kernel_statistics(matrix)
        self.statistics[name] = value
        write_json(self.scratch, dict(method=VERSION, parent_binding=self.parent_binding,
            statistics=self.statistics, physical_rows_released=False))
        return value

    def integral(self, kernel, e, duration, prefix, arrays):
        integrated = interval_kernel(kernel, e, duration)
        statistics = self.statistic(integrated, prefix + '_integral')
        del integrated
        primary = complex(self.arrays[prefix + '_primary'].item())
        paired = complex(self.arrays[prefix + '_paired'].item())
        scale = rational(statistics['kernel_abs_sum_arithmetic_model_upper'])
        error, paired_error = gamma_rational(28)*scale, gamma_rational(29)*scale
        for suffix in ('_primary_block_partials', '_paired_row_partials', '_primary', '_paired'):
            arrays[prefix + suffix] = self.arrays[prefix + suffix]
        return primary, error, dict(primary=value_record(primary,error),paired=value_record(paired,paired_error),
            paired_minus_primary=diag.complex_record(paired-primary),integral_kernel_statistics=statistics,
            definition='K_ij dt exp(+i (e_i-e_j)dt/2) sinc((e_i-e_j)dt/(2pi)); start elapsed zero',
            sinc_phase_construction_error_upper=None,
            arithmetic_scope='fixed rounded integral matrix only; derivative/sinc/phase construction remains unresolved')

    def save(self, path, binding, **arrays):
        if set(arrays) != set(self.arrays) - {'input_binding_json'}:
            raise ValueError('Recovery did not account for every original member')
        for key, value in arrays.items():
            if not exact_bits(value, self.arrays[key]):
                raise ValueError('Scalar recovery disagrees with a saved central value: ' + key)
        if set(self.statistics) != {'particle_derivative', 'hole_derivative',
                'particle_crossing_integral', 'hole_crossing_integral', 'native_current_integral'}:
            raise ValueError('The five missing-statistic passes are not exactly complete')
        temporary = path.with_suffix('.partial.npz')
        with self.original.open('rb') as source, temporary.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024*1024)
            target.flush()
            os.fsync(target.fileno())
        saved = identity(temporary)
        if saved['sha256'] != PINS['c15/diagnostics/' + PARTIAL_NAME]:
            raise ValueError('Recovery must preserve the original numeric payload bytes')
        promote(temporary, path)
        self.context.check()
        return identity(path)


# The ledger below preserves the pinned original scalar/error/report algebra.
# Its only substitutions are saved endpoints/scalars, statistics-only helpers,
# byte-identical payload copying, and explicit external recovery provenance.
def recovered_ledger(context,preparation,stages,binding,cut,duration,e,recovery):
    if cut != 16 or duration != 128.:
        raise ValueError('Only the stopped ledger16 interval may be recovered')
    origin_path = diag.prepared_path(context,preparation,'origin','origin_energy_checks.json')
    origin = json.loads(origin_path.read_text())
    arrays = dict(eigenvalues=e,elapsed=np.array([0.,duration]))
    values,endpoints,errors,crossings,crossing_errors = {},{},{},{},{}
    for kind in diag.KINDS:
        label = f'cut{cut}_{kind}'
        kernel,metadata = recovery.prepared_kernel(context,preparation,label,label+'_energy.npz',e)
        start = origin['kernels'][label]['primary']
        initial = diag.record_complex(start)
        final = complex(recovery.arrays[kind+'_endpoints'][1])
        final_error = rational(float(phase_arithmetic_model(metadata['statistics'],
            float(np.max(np.abs(e))),[duration])[0]))
        endpoints[kind] = [initial,final]
        errors[kind] = [rational(start['fixed_saved_input_error_upper_per_component']),final_error]
        arrays[kind+'_endpoints'] = np.array(endpoints[kind])
        arrays[kind+'_origin_primary_block_partials_parent'] = np.array(
            diag.prepared_path(context,preparation,'origin','origin_'+label+'.npz').as_posix())
        values[kind] = dict(origin_reused=start,first_endpoint=diag.complex_record(final),
            first_endpoint_fixed_kernel_arithmetic_model_upper=upwards(final_error),kernel_metadata=metadata)
        if kind != 'signed':
            matching = current_from_energy_kernel(kernel,e)
            del kernel
            current_statistics = recovery.statistic(matching,kind+'_derivative')
            integral,error,reading = recovery.integral(matching,e,duration,kind+'_crossing',arrays)
            del matching
            crossings[kind],crossing_errors[kind] = integral,error
            values[kind]['matching_derivative_statistics'] = current_statistics
            values[kind]['matching_integral'] = reading
        else:
            del kernel
    native,metadata = recovery.prepared_kernel(context,preparation,f'cut{cut}_current',
                                      f'cut{cut}_native_current.npz',e)
    if cut == 16:
        native_endpoints = np.array([diag.record_complex(stages.report['stages'][f'union_probe_{i}']
            ['values']['values']['current16']['D_kernel']) for i in range(2)])
    else:
        raise RuntimeError('A different endpoint contraction is forbidden in recovery')
    flow,flow_error,native_integral = recovery.integral(native,e,duration,'native_current',arrays)
    del native
    arrays['native_current_endpoints'] = native_endpoints
    arrays['native_current_integral'] = np.array(flow)
    residuals = {}
    residuals['native_energy'] = derived([*endpoints['signed'][::-1],flow],
        [*errors['signed'][::-1],flow_error],[1,-1,-1])[2]
    for kind in ('particle','hole'):
        residuals[kind+'_inventory'] = derived([*endpoints[kind][::-1],crossings[kind]],
            [*errors[kind][::-1],crossing_errors[kind]],[1,-1,-1])[2]
    q,k,q_error,k_error = [],[],[],[]
    for index in range(2):
        total,total_error,_ = derived([endpoints[x][index] for x in ('particle','hole')],
            [errors[x][index] for x in ('particle','hole')],[1,1])
        offset,offset_error,_ = derived([total,endpoints['signed'][index]],
            [total_error,errors['signed'][index]],[1,-1])
        q.append(total); q_error.append(total_error)
        k.append(offset); k_error.append(offset_error)
    inventory_integral,inventory_integral_error,inventory_integral_record = derived(
        [crossings[x] for x in ('particle','hole')],
        [crossing_errors[x] for x in ('particle','hole')],[1,1])
    residuals['summed_inventory'] = derived([q[1],q[0],inventory_integral],
        [q_error[1],q_error[0],inventory_integral_error],[1,-1,-1])[2]
    values['positive_inventory'] = dict(
        origin=value_record(q[0],q_error[0]),first_endpoint=value_record(q[1],q_error[1]),
        matching_integral=inventory_integral_record,
        scope='scalar sum of saved particle/hole endpoints and independently integrated crossings; all construction/library terms remain separate')
    residuals['two_term_detector'] = derived([q[1],q[0],flow,k[1],k[0]],
        [q_error[1],q_error[0],flow_error,k_error[1],k_error[0]],[1,-1,-1,-1,1])[2]
    arrays['positive_inventory_endpoints'] = np.array(q)
    arrays['positive_inventory_matching_integral'] = np.array(inventory_integral)
    arrays['localization_offset_endpoints'] = np.array(k)
    rho = rational(context.source.method.rho)
    energy_coefficient,native_coefficient,inventory_coefficient = sqrt_upper(16384)*rho,4*rho**2,2*sqrt_upper(16384)*rho**2
    source = dict(energy_endpoint=diag.source_allowances(context,energy_coefficient),
        positive_inventory_endpoint=diag.source_allowances(context,2*energy_coefficient),
        native_integral=diag.source_allowances(context,native_coefficient,duration),
        matching_inventory_integral=diag.source_allowances(context,inventory_coefficient,duration),
        matching_positive_inventory_integral=diag.source_allowances(context,2*inventory_coefficient,duration),
        native_residual=diag.source_allowances(context,2*energy_coefficient+abs(rational(duration))*native_coefficient),
        inventory_residual=diag.source_allowances(context,2*energy_coefficient+abs(rational(duration))*inventory_coefficient),
        summed_inventory_residual=diag.source_allowances(context,4*energy_coefficient+2*abs(rational(duration))*inventory_coefficient),
        two_term_residual=diag.source_allowances(context,10*energy_coefficient+abs(rational(duration))*native_coefficient),
        combination_scope='triangle bounds: two endpoints per allocation; Q has two terms, K=Q-E has three; no observed-spread substitution')
    array_name,report_name = f'ledger_{cut}_arrays.npz',f'ledger_{cut}_checks.json'
    array_id = recovery.save(stages.root/array_name,binding,**arrays)
    report = dict(input_binding=binding,cut_radius=cut*64.,duration=duration,energy_readings=values,
        native_current=dict(endpoints=[diag.complex_record(x) for x in native_endpoints],
            integral=native_integral,kernel_metadata=metadata),continuity_residuals=residuals,
        source_additions=source,uncertainties=diag.unresolved_terms(context),arrays=array_id,
        equations=['Delta E_out - integral J_native','Delta Q_pm - integral J_Qpm',
                   'Delta(Q_particle+Q_hole) - integral(J_Qparticle+J_Qhole)',
                   'Delta Q - integral J_native - Delta(Q-E_out)'],physical_rows_released=False)
    report['recovery'] = dict(method=VERSION, original_production_binding=recovery.parent_binding,
        saved_payload=context.source.inputs.current[str(recovery.original)],
        regenerated_statistics_only=list(recovery.statistics),
        original_numeric_payload_recontracted=False, original_failed_parent_preserved=True,
        statistic_bits_not_previously_saved=True)
    report_id = diag.save_json(stages.root/report_name,report)
    recovery.scratch.unlink()
    return dict(cut_radius=cut*64.,duration=duration,continuity_residuals=residuals,
        native_current_integral=diag.complex_record(flow)),{array_name:array_id,report_name:report_id}


def run(args):
    expected = NEW / args.case / 'diagnostics'
    if Path(args.out).resolve() != expected:
        raise ValueError('The explicit continuation has a fixed separate output root')
    # The historical preparation validator sees its original common binding.
    context = load_readout(args)
    preparation_id, preparation = diag.load_preparation(context, args.preparation_progress)
    legacy, parents = legacy_parent(context, preparation_id) if args.case == 'c15' else (None, {})
    adapter_ids = {p.name: context.source.inputs.add(p) for p in (
        Path(__file__), Path(__file__).parent.parent / 'run_saved_recovered_readouts_repair.py')}
    extra = dict(accepted_preparation=preparation_id, explicit_repair_version=VERSION,
        adapter_code=adapter_ids, historical_recovery_parents=parents,
        serializer_override='typed_writer: only three exact scalar Unicode parent-role paths',
        legacy_numeric_payload_keeps_original_embedded_binding=args.case == 'c15',
        physical_rows_released=False)
    stages = RepairStages(context, expected, args.resume, extra, legacy)
    with np.load(diag.prepared_path(context, preparation, 'fields', 'clock_arrays.npz'), allow_pickle=False) as saved:
        times = saved['diagnostic_elapsed']
        if (times.dtype != np.float64 or times.tolist() != [0.,128.,14464.]
                or saved['interval_durations'][0] != times[1]
                or saved['interval_starts'][0] != context.source.metadata['time_start']):
            raise ValueError('The unchanged selected-clock contract differs')
    # Future ledgers use the explicitly bound typed serializer, not a changed
    # historical module or a blanket exception around numeric validation.
    diag.save_arrays = typed_writer(context, preparation)
    if args.case == 'c15':
        with np.load(context.output('spectrum', 'spectrum'), allow_pickle=False) as raw:
            e = raw['eigenvalues']
        if e.shape != (16384,) or e.dtype != np.float64:
            raise ValueError('The accepted full eigenvalue layout differs')
        require_finite(e)
        # No V, Dhat, phase/probe contraction or completed preparation is loaded.
        if 'ledger_16' not in stages.report['stages']:
            recovery = SavedLedger(context, preparation, legacy, stages, e)
            stages.execute('ledger_16', lambda binding: recovered_ledger(
                context, preparation, stages, binding, 16, float(times[1]), e, recovery))
    else:
        e, v, d = context.load_spectrum_deviation()
        stages.execute('union_setup', lambda binding: diag.union_setup(context, preparation, stages, binding, e, v))
        for index, time in enumerate(times):
            stages.execute(f'union_probe_{index}', lambda binding: diag.union_probe(
                context, preparation, stages, binding, index, float(time), e, v, d))
        d = None
        stages.execute('inside_setup', lambda binding: diag.inside_setup(context, preparation, stages, binding))
        for index, time in enumerate(times[:2]):
            stages.execute(f'inside_probe_{index}', lambda binding: diag.inside_probe(
                context, stages, binding, index, float(time), e, v))
        v = None
        stages.execute('ledger_16', lambda binding: diag.ledger(
            context, preparation, stages, binding, 16, float(times[1]), e))
    stages.execute('ledger_30', lambda binding: diag.ledger(
        context, preparation, stages, binding, 30, float(times[1]), e))
    stages.execute('summary', lambda binding: diag.diagnostic_summary(context, stages, binding, times))
    stages.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('descriptor', 'build', 'comparison', 'origin', 'spectral-progress', 'preparation-progress', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--case', choices=('c15', 'c30'), required=True)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
