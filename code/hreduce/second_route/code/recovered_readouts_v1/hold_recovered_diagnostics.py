#!/usr/bin/env python3
"""HOLD 18.24: selected recovered-source diagnostics; physical rows stay locked."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from hold_error_bounds import downwards, gamma_rational, pairwise_sum, rational, sqrt_upper, upwards
from hold_interval_checks import expectation, probe_covariance
from hold_origin_diagnostics import derived, kernel_reductions, value_record
from hold_observables import (current_from_energy_kernel, interval_kernel,
    kernel_statistics, paired_phase_values, phase_arithmetic_model, phase_values)
from hold_recovered_prepare import BRANCHES, INVENTORY as PREPARE_INVENTORY, KINDS, WINDOWS
from hold_recovered_readout_input import (READOUT_VERSION, ReadoutStages, binding_hash,
    load_readout, save_arrays, save_json)


INVENTORY = {'union_setup':{'union_reference.npz','union_operators.npz','union_setup.json'}}
for index in range(3):
    INVENTORY[f'union_probe_{index}'] = {f'union_probe_{index}.npz',f'union_probe_{index}_checks.json'}
INVENTORY['inside_setup'] = {'inside_operator.npz','inside_anchor.json'}
for index in range(2):
    INVENTORY[f'inside_probe_{index}'] = {f'inside_probe_{index}.npz',f'inside_probe_{index}_checks.json'}
for cut in (16,30):
    INVENTORY[f'ledger_{cut}'] = {f'ledger_{cut}_arrays.npz',f'ledger_{cut}_checks.json'}
INVENTORY['summary'] = {'diagnostics_checks.json'}


def load_preparation(context,path):
    saved_id = context.source.inputs.add(path)
    saved = json.loads(Path(saved_id['path']).read_text())
    if ((saved['method'],saved['consumer'],saved['completed'],saved['physical_rows_released'])
            != (READOUT_VERSION,'prepare',True,False)
            or list(saved['stages']) != list(PREPARE_INVENTORY)
            or binding_hash(saved['input_binding']) != saved['input_binding_hash']
            or binding_hash(saved['input_binding']['common']) != binding_hash(context.binding)):
        raise ValueError('The recovered preparation has another incomplete input/code association')
    parent = Path(saved_id['path']).parent
    allowed = {Path(saved_id['path']).name}
    for name,stage in saved['stages'].items():
        if stage['status'] != 'completed' or set(stage['outputs']) != set(PREPARE_INVENTORY[name]):
            raise ValueError('The recovered preparation has an uncertain numerical stage')
        for filename,artifact in stage['outputs'].items():
            context.source.inputs.verify(artifact)
            if Path(artifact['path']) != parent/filename:
                raise ValueError('A prepared product has another parent/role')
            allowed.add(filename)
    if {p.name for p in parent.iterdir()} != allowed:
        raise FileExistsError('Orphan preparation products/partials remain')
    context.check()
    return saved_id,saved


def prepared_path(context,preparation,stage,filename):
    saved = preparation['stages'][stage]['outputs'][filename]
    context.source.inputs.verify(saved)
    return Path(saved['path'])


def prepared_kernel(context,preparation,stage,name,e):
    with np.load(prepared_path(context,preparation,stage,name),allow_pickle=False) as raw:
        order,kernel = raw['eigenvalues'],raw['kernel']
        binding = json.loads(raw['input_binding_json'].item())
    if (order.dtype != e.dtype or not np.array_equal(order.view(np.uint64),e.view(np.uint64))
            or kernel.shape != (16384,16384) or kernel.dtype != np.complex128):
        raise ValueError('The prepared full kernel has another ordering/layout')
    return kernel,binding['kernel_metadata']


def complex_record(value):
    return dict(real=float(value.real),imaginary=float(value.imag))


def union_setup(context,preparation,stages,binding,e,v):
    definitions = preparation['stages']['local_operators']['values']
    support = np.unique(np.concatenate((
        np.array(definitions['current16']['support']),
        *(np.array(definitions[f'window{width}']['support']) for width in WINDOWS))))
    negative = v[support][:,e < 0]
    reference = negative @ negative.conj().T
    physical = context.source.occupied[support]
    direct = physical @ physical.conj().T
    operators = {}
    source_definitions = {}
    for label,filename in (('current16','current16.npz'),*((f'window{width}',f'window{width}.npz') for width in WINDOWS)):
        with np.load(prepared_path(context,preparation,'local_operators',filename),allow_pickle=False) as raw:
            local_support = raw['support']
            selected = np.searchsorted(support,local_support)
            if not np.array_equal(support[selected],local_support):
                raise ValueError('A required local observable is outside the shared probe support')
            names = ('matrix',) if label == 'current16' else BRANCHES
            for name in names:
                key = 'current16' if name == 'matrix' else label+'_'+name
                full = np.zeros((len(support),len(support)),dtype=np.complex128)
                full[np.ix_(selected,selected)] = raw[name]
                operators[key] = full
                source_definitions[key] = definitions[label]
    references = {key:complex_record(expectation(matrix,reference)) for key,matrix in operators.items()}
    direct_windows = {key:dict(C_only=complex_record(expectation(matrix,direct)),
        minus_raw_V_reference=complex_record(expectation(matrix,direct-reference)))
        for key,matrix in operators.items() if key != 'current16'}
    reference_id = save_arrays(stages.root/'union_reference.npz',binding,
        support=support,raw_V_negative_projector_block=reference,direct_coordinate_source_covariance=direct)
    operators_id = save_arrays(stages.root/'union_operators.npz',binding,support=support,**operators)
    values = dict(support=support.tolist(),vacuum_readings=references,direct_origin_window_readings=direct_windows,
        direct_origin_C_current_reused=context.origin['current'],
        direct_current_scope='accepted paired origin contraction reused; not recontracted on the union block',
        local_definitions=source_definitions)
    report_id = save_json(stages.root/'union_setup.json',dict(input_binding=binding,**values,physical_rows_released=False))
    return values,{'union_reference.npz':reference_id,'union_operators.npz':operators_id,'union_setup.json':report_id}


def union_probe(context,preparation,stages,binding,index,time,e,v,d):
    with np.load(stages.output('union_setup','union_reference.npz'),allow_pickle=False) as saved:
        support,reference = saved['support'],saved['raw_V_negative_projector_block']
    with np.load(stages.output('union_setup','union_operators.npz'),allow_pickle=False) as saved:
        operators = {key:saved[key] for key in saved.files if key not in ('support','input_binding_json')}
    covariance,projections = probe_covariance(v,e,context.source.occupied,support,time)
    rotated = v[support]*np.exp(-1j*e*time)[None,:]
    local_d = (rotated @ d) @ rotated.conj().T
    values = {}
    for key,matrix in operators.items():
        c_only,vacuum,d_local = (expectation(matrix,x) for x in (covariance,reference,local_d))
        stage = 'cut16_current' if key == 'current16' else key
        filename = 'cut16_native_current.npz' if key == 'current16' else key+'.npz'
        kernel,metadata = prepared_kernel(context,preparation,stage,filename,e)
        primary = phase_values(kernel,e,[time])[0]
        alternate = paired_phase_values(kernel,e,[time])[0] if index else None
        arithmetic = phase_arithmetic_model(metadata['statistics'],float(np.max(np.abs(e))),[time])[0]
        values[key] = dict(C_only=complex_record(c_only),P_raw_V=complex_record(vacuum),D_local=complex_record(d_local),
            D_kernel=complex_record(primary),C_minus_D_kernel_minus_P=complex_record(c_only-primary-vacuum),
            D_local_minus_D_kernel=complex_record(d_local-primary),
            paired_kernel=complex_record(alternate) if alternate is not None else None,
            paired_minus_primary=complex_record(alternate-primary) if alternate is not None else None,
            fixed_kernel_phase_accumulation_model_upper=float(arithmetic),
            source_additions=metadata['source_additions'],
            phase_library_scope='assumed exp arithmetic model; shared spectral/operator/transform errors remain separate')
        del kernel
    arrays_id = save_arrays(stages.root/f'union_probe_{index}.npz',binding,support=support,
        elapsed=np.array(time),Y=projections,C_source_probe=covariance,Dhat_local_covariance=local_d)
    report_id = save_json(stages.root/f'union_probe_{index}_checks.json',dict(input_binding=binding,
        elapsed=time,values=values,arrays=arrays_id,physical_rows_released=False,
        probe_definition='W=V exp(+i e tau)V_S^dagger; Y=Phi^dagger W; C_S=Y^dagger Y; W omitted but definition bound'))
    return dict(elapsed=time,values=values),{f'union_probe_{index}.npz':arrays_id,
                                          f'union_probe_{index}_checks.json':report_id}


def unresolved_terms(context):
    return dict(saved_transform_allowance=context.precision['saved_transform_allowance'],
        inflated_spectral_controls=context.precision['spectral_inflation'],
        source_covariance_errors=context.precision['source_covariance_errors'],
        probe_contraction_error_upper=None,cut_local_operator_construction_error_upper=None,
        true_projector_replacement_error_upper=None,spectral_propagation_error_upper=None,
        validated_phase_sinc_library_error_upper=None,producer_seed_version_error_upper=None,
        different_model_held_evolution_error_upper=None,total_physical_error_upper=None,
        scope='null terms are unresolved, not zero; displayed fixed-matrix arithmetic is not a physical certificate')


def source_allowances(context,coefficient,duration=1):
    factor = abs(rational(duration))*rational(coefficient)
    return {kind:dict(observable_F_upper=upwards(rational(coefficient)),
        duration_factor=abs(float(duration)),covariance_F_error_upper=entry['covariance_F_error_upper'],
        source_observable_error_upper=upwards(factor*rational(entry['covariance_F_error_upper'])),
        scope='same exact stored held unitary; stored/model source-history alternatives are not added')
        for kind,entry in context.precision['source_covariance_errors'].items()}


def inside_setup(context,preparation,stages,binding):
    with np.load(prepared_path(context,preparation,'fields','readout_fields.npz'),allow_pickle=False) as saved:
        inside = np.repeat(saved['radius'] <= 1024.,2)
    h = context.load_h()
    links = h.tocoo()
    data = .5*links.data*(inside[links.row].astype(int)+inside[links.col].astype(int))
    nonzero = data != 0
    support = np.unique(np.r_[links.row[nonzero],links.col[nonzero]])
    if (inside.shape != (16384,) or np.flatnonzero(inside).tolist() != list(range(2048))
            or not np.array_equal(support,np.arange(2052))):
        raise ValueError('The inside allocation has another cut or range-two support')
    mask = inside[support].astype(int)
    operator = h[support][:,support].toarray()*(mask[:,None]+mask[None,:])*.5
    if not np.array_equal(operator,operator.conj().T):
        raise ArithmeticError('The cached-CSR inside allocation is not Hermitian')
    raw_id = context.origin['raw_scalar_arrays']
    context.source.inputs.verify(raw_id)
    with np.load(raw_id['path'],allow_pickle=False) as raw:
        real = raw['recovered_source_energy_row_block_real']
        imaginary = raw['recovered_source_energy_row_block_imaginary']
        raw_binding = json.loads(raw['recovered_spatial_binding_json'].item())
        if (real.shape != (16384,32) or imaginary.shape != real.shape
                or real.dtype != np.float64 or imaginary.dtype != np.float64
                or not np.all(np.isfinite(real)) or not np.all(np.isfinite(imaginary))
                or raw['recovered_spatial_completed_block_count'].item() != 32
                or not np.array_equal(raw['recovered_spatial_completed_block_mask'],np.ones(32,dtype=bool))
                or not np.array_equal(raw['recovered_spatial_spinor_row_indices'],np.arange(16384))
                or not np.array_equal(raw['recovered_spatial_column_ranges'],
                                      np.arange(33,dtype=np.int64).repeat(2)[1:-1].reshape(32,2)*256)
                or binding_hash(raw_binding) != binding_hash(context.origin['spatial_energy']['binding'])):
            raise ValueError('The accepted spatial source partials are incomplete or differently bound')
        anchors = {name:dict(real=pairwise_sum(real[selected]),
            unsymmetrized_imaginary=pairwise_sum(imaginary[selected]))
            for name,selected in (('inside',inside),('outside',~inside))}
    allowance = context.origin['spatial_energy']['fixed_array_error_upper_per_component_for_any_sharp_mask']
    for anchor in anchors.values():
        anchor['fixed_array_real_interval'] = [downwards(rational(anchor['real'])-rational(allowance)),
                                               upwards(rational(anchor['real'])+rational(allowance))]
    partition_comparison = derived(
        [complex(anchors['inside']['real']),complex(anchors['outside']['real']),
         complex(context.origin['energy']['real'])],
        [allowance,allowance,context.origin['energy']['fixed_array_error_upper_per_component']],
        [1,1,-1])[2]
    arrays_id = save_arrays(stages.root/'inside_operator.npz',binding,
        support=support,inside_spinor_mask=inside,matrix=operator)
    values = dict(cut_radius=1024.,support=support.tolist(),direct_source_anchors=anchors,
        origin_spatial_arrays=raw_id,global_B_E_per_component=allowance,
        global_B_E_scope='single inherited gamma29 allowance includes sharp-mask reduction; no second mask allowance added',
        direct_source_whole_main_association=context.origin['energy'],
        partition_sum_reassociation_minus_main=partition_comparison,
        partition_comparison_scope='two separately reduced sharp masks minus the saved main association; 2 B_E plus B_main and scalar rounding, each mask retaining its single B_E',
        imaginary_scope='saved chi h partials are unsymmetrized, not a Hermiticity residual',
        source_covariance_errors=context.precision['source_covariance_errors'])
    report_id = save_json(stages.root/'inside_anchor.json',dict(input_binding=binding,
        **values,physical_rows_released=False))
    return values,{'inside_operator.npz':arrays_id,'inside_anchor.json':report_id}


def inside_probe(context,stages,binding,index,time,e,v):
    with np.load(stages.output('inside_setup','inside_operator.npz'),allow_pickle=False) as saved:
        support,operator = saved['support'],saved['matrix']
    covariance,projections = probe_covariance(v,e,context.source.occupied,support,time)
    value = expectation(operator,covariance)
    values = dict(elapsed=float(time),C_inside=complex_record(value),
        probe_definition='W=V exp(+i e tau)V_S^dagger; Y=Phi^dagger W; C_S=Y^dagger Y',
        uncertainties=unresolved_terms(context))
    array_name,report_name = f'inside_probe_{index}.npz',f'inside_probe_{index}_checks.json'
    array_id = save_arrays(stages.root/array_name,binding,support=support,
        elapsed=np.array(time),Y=projections,C_source_probe=covariance,C_inside=np.array(value))
    report_id = save_json(stages.root/report_name,dict(input_binding=binding,**values,
        arrays=array_id,physical_rows_released=False))
    return values,{array_name:array_id,report_name:report_id}


def record_complex(record):
    return complex(record['real'],record['imaginary'])


def integral_readout(kernel,e,duration,prefix,arrays):
    integrated = interval_kernel(kernel,e,duration)
    statistics = kernel_statistics(integrated)
    primary,paired,blocks,rows = kernel_reductions(integrated)
    del integrated
    scale = rational(statistics['kernel_abs_sum_arithmetic_model_upper'])
    error = gamma_rational(28)*scale
    paired_error = gamma_rational(29)*scale
    arrays[prefix+'_primary_block_partials'] = blocks
    arrays[prefix+'_paired_row_partials'] = rows
    arrays[prefix+'_primary'] = np.array(primary)
    arrays[prefix+'_paired'] = np.array(paired)
    return primary,error,dict(primary=value_record(primary,error),paired=value_record(paired,paired_error),
        paired_minus_primary=complex_record(paired-primary),integral_kernel_statistics=statistics,
        definition='K_ij dt exp(+i (e_i-e_j)dt/2) sinc((e_i-e_j)dt/(2pi)); start elapsed zero',
        sinc_phase_construction_error_upper=None,
        arithmetic_scope='fixed rounded integral matrix only; derivative/sinc/phase construction remains unresolved')


def ledger(context,preparation,stages,binding,cut,duration,e):
    origin_path = prepared_path(context,preparation,'origin','origin_energy_checks.json')
    origin = json.loads(origin_path.read_text())
    arrays = dict(eigenvalues=e,elapsed=np.array([0.,duration]))
    values,endpoints,errors,crossings,crossing_errors = {},{},{},{},{}
    for kind in KINDS:
        label = f'cut{cut}_{kind}'
        kernel,metadata = prepared_kernel(context,preparation,label,label+'_energy.npz',e)
        start = origin['kernels'][label]['primary']
        initial = record_complex(start)
        final = phase_values(kernel,e,[duration])[0]
        final_error = rational(float(phase_arithmetic_model(metadata['statistics'],
            float(np.max(np.abs(e))),[duration])[0]))
        endpoints[kind] = [initial,final]
        errors[kind] = [rational(start['fixed_saved_input_error_upper_per_component']),final_error]
        arrays[kind+'_endpoints'] = np.array(endpoints[kind])
        arrays[kind+'_origin_primary_block_partials_parent'] = np.array(
            prepared_path(context,preparation,'origin','origin_'+label+'.npz').as_posix())
        values[kind] = dict(origin_reused=start,first_endpoint=complex_record(final),
            first_endpoint_fixed_kernel_arithmetic_model_upper=upwards(final_error),kernel_metadata=metadata)
        if kind != 'signed':
            matching = current_from_energy_kernel(kernel,e)
            del kernel
            current_statistics = kernel_statistics(matching)
            integral,error,reading = integral_readout(matching,e,duration,kind+'_crossing',arrays)
            del matching
            crossings[kind],crossing_errors[kind] = integral,error
            values[kind]['matching_derivative_statistics'] = current_statistics
            values[kind]['matching_integral'] = reading
        else:
            del kernel
    native,metadata = prepared_kernel(context,preparation,f'cut{cut}_current',
                                      f'cut{cut}_native_current.npz',e)
    if cut == 16:
        native_endpoints = np.array([record_complex(stages.report['stages'][f'union_probe_{i}']
            ['values']['values']['current16']['D_kernel']) for i in range(2)])
    else:
        native_endpoints = phase_values(native,e,[0.,duration])
    flow,flow_error,native_integral = integral_readout(native,e,duration,'native_current',arrays)
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
    source = dict(energy_endpoint=source_allowances(context,energy_coefficient),
        positive_inventory_endpoint=source_allowances(context,2*energy_coefficient),
        native_integral=source_allowances(context,native_coefficient,duration),
        matching_inventory_integral=source_allowances(context,inventory_coefficient,duration),
        matching_positive_inventory_integral=source_allowances(context,2*inventory_coefficient,duration),
        native_residual=source_allowances(context,2*energy_coefficient+abs(rational(duration))*native_coefficient),
        inventory_residual=source_allowances(context,2*energy_coefficient+abs(rational(duration))*inventory_coefficient),
        summed_inventory_residual=source_allowances(context,4*energy_coefficient+2*abs(rational(duration))*inventory_coefficient),
        two_term_residual=source_allowances(context,10*energy_coefficient+abs(rational(duration))*native_coefficient),
        combination_scope='triangle bounds: two endpoints per allocation; Q has two terms, K=Q-E has three; no observed-spread substitution')
    array_name,report_name = f'ledger_{cut}_arrays.npz',f'ledger_{cut}_checks.json'
    array_id = save_arrays(stages.root/array_name,binding,**arrays)
    report = dict(input_binding=binding,cut_radius=cut*64.,duration=duration,energy_readings=values,
        native_current=dict(endpoints=[complex_record(x) for x in native_endpoints],
            integral=native_integral,kernel_metadata=metadata),continuity_residuals=residuals,
        source_additions=source,uncertainties=unresolved_terms(context),arrays=array_id,
        equations=['Delta E_out - integral J_native','Delta Q_pm - integral J_Qpm',
                   'Delta(Q_particle+Q_hole) - integral(J_Qparticle+J_Qhole)',
                   'Delta Q - integral J_native - Delta(Q-E_out)'],physical_rows_released=False)
    report_id = save_json(stages.root/report_name,report)
    return dict(cut_radius=cut*64.,duration=duration,continuity_residuals=residuals,
        native_current_integral=complex_record(flow)),{array_name:array_id,report_name:report_id}


def diagnostic_summary(context,stages,binding,times):
    anchor = stages.report['stages']['inside_setup']['values']
    direct = complex(anchor['direct_source_anchors']['inside']['real'])
    raw = [record_complex(stages.report['stages'][f'inside_probe_{i}']['values']['C_inside']) for i in range(2)]
    current = [stages.report['stages'][f'union_probe_{i}']['values']['values']['current16'] for i in range(3)]
    vacuum = record_complex(stages.report['stages']['union_setup']['values']['vacuum_readings']['current16'])
    flow_d = record_complex(stages.report['stages']['ledger_16']['values']['native_current_integral'])
    duration = rational(float(times[1]))
    spectral = context.precision['spectral_inflation']
    delta,residual,covariance = (rational(spectral[name]) for name in
        ('gram_F_upper','residual_F_upper','source_covariance_F_upper'))
    if not 0 <= delta < 1 or residual < 0 or covariance < 0:
        raise ValueError('The accepted raw-V total-change controls are invalid')
    b0 = delta
    b1 = delta+abs(duration)*residual*sqrt_upper(1+delta)
    total_change = rational(context.source.method.rho)*(2+delta)*covariance*(b0+b1)
    values = dict(method=READOUT_VERSION,input=context.source.metadata,input_binding=binding,
        completed=True,selected_elapsed_clocks=times.tolist(),physical_rows_released=False,
        source_kind=context.source.metadata['source_kind'],kappa=3,guard_reporting_weight=48,
        completion_map={name:dict(status=stage['status'],outputs=stage['outputs'])
            for name,stage in stages.report['stages'].items() if name != 'summary'},
        source_initial_current_reused=context.origin['current'],
        raw_V_current_source_seam=complex_record(record_complex(current[0]['C_only'])
                                                -complex(context.origin['current']['real'])),
        measured_raw_V_vacuum_current=complex_record(vacuum),
        raw_V_inside_initial_seam=complex_record(raw[0]-direct),
        raw_inside_loss=complex_record(raw[0]-raw[1]),
        source_anchored_inside_loss=complex_record(direct-raw[1]),
        D_analytic_current_integral=complex_record(flow_d),
        C_analytic_current_integral=complex_record(flow_d+float(duration)*vacuum),
        raw_inside_loss_minus_C_analytic_integral=complex_record(raw[0]-raw[1]-flow_d-float(duration)*vacuum),
        source_anchored_inside_loss_minus_C_analytic_integral=complex_record(direct-raw[1]-flow_d-float(duration)*vacuum),
        raw_V_total_energy_change_upper=upwards(total_change),
        total_change_scope='stored h and exact stored source under raw-V model; probe contraction/phase/source-history errors separate',
        endpoint_relation='F_out = F_inside_loss + Delta E_total; no exterior flow was independently measured here',
        union_current_and_signed_branches={f'clock_{i}':stages.report['stages'][f'union_probe_{i}']['values'] for i in range(3)},
        inside_anchor=anchor,uncertainties=unresolved_terms(context),
        missing_held_comparators=dict(extrema=None,first_current=None,first_flow=None,rows=None),
        release='withheld; selected mandatory diagnostics do not enable all-row or physical evaluation')
    report_id = save_json(stages.root/'diagnostics_checks.json',values)
    return dict(completed_selected_diagnostics=True,physical_rows_released=False),{'diagnostics_checks.json':report_id}


def run(args):
    context = load_readout(args)
    preparation_id,preparation = load_preparation(context,args.preparation_progress)
    stages = ReadoutStages(context,args.out,INVENTORY,'diagnostics',args.resume,
                           extra_binding=dict(accepted_preparation=preparation_id))
    with np.load(prepared_path(context,preparation,'fields','clock_arrays.npz'),allow_pickle=False) as saved:
        times = saved['diagnostic_elapsed']
        if (times.dtype != np.float64 or times.tolist() != [0.,128.,14464.]
                or saved['interval_durations'][0] != times[1]
                or saved['interval_starts'][0] != context.source.metadata['time_start']):
            raise ValueError('The completed preparation has another selected-clock contract')
    e,v,d = context.load_spectrum_deviation()
    stages.execute('union_setup',lambda binding:union_setup(context,preparation,stages,binding,e,v))
    for index,time in enumerate(times):
        stages.execute(f'union_probe_{index}',lambda binding:union_probe(
            context,preparation,stages,binding,index,float(time),e,v,d))
    d = None
    stages.execute('inside_setup',lambda binding:inside_setup(context,preparation,stages,binding))
    for index,time in enumerate(times[:2]):
        stages.execute(f'inside_probe_{index}',lambda binding:inside_probe(
            context,stages,binding,index,float(time),e,v))
    v = None
    for cut in (16,30):
        stages.execute(f'ledger_{cut}',lambda binding:ledger(
            context,preparation,stages,binding,cut,float(times[1]),e))
    stages.execute('summary',lambda binding:diagnostic_summary(context,stages,binding,times))
    stages.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('descriptor','build','comparison','origin','spectral-progress','preparation-progress','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
