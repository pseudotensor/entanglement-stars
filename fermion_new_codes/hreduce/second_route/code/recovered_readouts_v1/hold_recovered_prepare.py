#!/usr/bin/env python3
"""HOLD 18.24: prepare required recovered-only observables, once per product."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from hold_error_bounds import downwards, gamma_rational, rational, sqrt_upper, upwards
from hold_geometry import geometry_diagnostics
from hold_observables import (cut_energy_kernel, kernel_statistics, local_kernel,
                             sharp_outside_in_eigenbasis, signed_window_observables)
from hold_recovered_readout_input import (ReadoutStages, load_readout, save_arrays,
                                         save_json)


KINDS = ('signed','particle','hole')
CUTS = ('cut16','cut30','root_inner','root_outer')
BRANCHES = ('E_flux_plus_out','E_flux_plus_in','E_flux_minus_out','E_flux_minus_in')
WINDOWS = (32,16,64)
INVENTORY = {
    'fields': {'readout_fields.npz','geometry.json','geometry_arrays.npz','clock_template.json','clock_arrays.npz'},
    'local_operators': {'current16.npz','current30.npz',*(f'window{w}.npz' for w in WINDOWS),'local_operators.json'},
    'tails': {'tail_arrays.npz','tail_checks.json'}}
for cut in CUTS:
    INVENTORY[cut+'_chi'] = {cut+'_chi.npz'}
    for kind in KINDS:
        INVENTORY[cut+'_'+kind] = {cut+'_'+kind+'_energy.npz'}
    if cut in ('cut16','cut30'):
        INVENTORY[cut+'_current'] = {cut+'_native_current.npz'}
for window in WINDOWS:
    for branch in BRANCHES:
        INVENTORY[f'window{window}_{branch}'] = {f'window{window}_{branch}.npz'}
INVENTORY['origin'] = {'origin_energy_arrays.npz','origin_energy_checks.json',
                      *(f'origin_{cut}_{kind}.npz' for cut in CUTS for kind in KINDS)}


def fields_and_clocks(context,stages,binding,template_path):
    meta = context.source.metadata
    with np.load(context.output('operator','held_fields'),allow_pickle=False) as saved:
        radius = saved['radius']
        inner = {key:saved[key] for key in ('inner_radius','inner_clock_applied','inner_flow_applied')}
    if (radius.dtype != np.float64 or not np.array_equal(radius,np.arange(1.,8193.))
            or not np.array_equal(inner['inner_radius'],radius[:2048])):
        raise ValueError('The accepted original physical radius/inner sample order differs')
    clock = np.ones(8192,dtype=np.float64)
    flow = -np.sqrt(2*meta['g']*meta['charge']/radius)
    clock[:2048],flow[:2048] = inner['inner_clock_applied'],inner['inner_flow_applied']
    if not np.all(np.isfinite(clock)) or not np.all(np.isfinite(flow)):
        raise ArithmeticError('The sampled readout fields are nonfinite')
    geometry,geometry_arrays = geometry_diagnostics(radius,clock,flow,meta)
    template_id = context.source.inputs.add(template_path)
    template = json.loads(Path(template_id['path']).read_text())
    settings,history = template['settings'],template['history']
    expected = dict(nsites=8192,fields_key='applied',freeze_at_rs=meta['hold_rs'],
                    R_cut_rs=24.,R_collar_rs=16.,collar_split=True,collar_split_windows='32,16,64')
    if any(settings[key] != value for key,value in expected.items()):
        raise ValueError('The clock template has another original hold/chain convention')
    for key in ('fields','record'):
        if Path(settings[key]).name != Path(meta[key]).name:
            raise ValueError('The clock template has another original physical history')
    for key,value in dict(g=.25,rs_nominal=64.,tau=8.,ell=4.,engine_sites=2048,T_start=meta['time_start']).items():
        if history[key] != value:
            raise ValueError('The clock-template history tuple differs')
    clocks = np.array([row['t'] for row in template['rows']],dtype=np.float64)
    row_k = np.array([row['k'] for row in template['rows']],dtype=np.int64)
    starts = np.array([row['t'] for row in template['records']],dtype=np.float64)
    ends = np.array([row['t_end'] for row in template['records']],dtype=np.float64)
    if (not len(clocks) or not len(starts) or not all(np.all(np.isfinite(x)) for x in (clocks,starts,ends))
            or np.any(clocks < meta['time_start']) or np.any(clocks[1:] < clocks[:-1])
            or np.any(ends <= starts) or starts[0] != meta['time_start']
            or np.any(starts[1:] != ends[:-1])):
        raise ValueError('The original clock template has invalid row/interval clocks')
    return_marker = min(geometry['applied_return_absolute_clock'],
        geometry['reciprocal_linear_return_absolute_clock'],
        meta['time_start']+float(template['reflection_clock']['T_refl']))
    eligible = clocks[clocks < return_marker]
    if not len(eligible):
        raise ValueError('No template checkpoint precedes the required return markers')
    times = np.array([0.,ends[0]-starts[0],float(eligible[-1])-meta['time_start']])
    clock_metadata = dict(template_identity=template_id,role='clock template only; no other-channel physical targets consumed',
        original_template_settings=settings,original_template_history=history,
        original_reflection_clock=template['reflection_clock'],return_marker_absolute=return_marker,
        diagnostic_elapsed_clocks=times.tolist(),missing_kappa3_targets=dict(extrema=None,current=None,flow=None,rows=None),
        expected_first_elapsed=128.,expected_longest_elapsed=14464.)
    outputs = {}
    outputs['readout_fields.npz'] = save_arrays(stages.root/'readout_fields.npz',dict(
        **binding,scope='sampled N/v for geometry/local symbol only; cached CSR is unchanged'),
        radius=radius,clock=clock,flow=flow,**inner)
    outputs['geometry.json'] = save_json(stages.root/'geometry.json',dict(
        input_binding=binding,input=meta,geometry=geometry,physical_rows_released=False))
    outputs['geometry_arrays.npz'] = save_arrays(stages.root/'geometry_arrays.npz',binding,**geometry_arrays)
    outputs['clock_template.json'] = save_json(stages.root/'clock_template.json',dict(
        input_binding=binding,input=meta,**clock_metadata,physical_rows_released=False))
    outputs['clock_arrays.npz'] = save_arrays(stages.root/'clock_arrays.npz',binding,
        absolute_clocks=clocks,elapsed_clocks=clocks-meta['time_start'],original_row_index=np.arange(len(clocks)),
        original_row_record_index=row_k,duplicate_origin=np.r_[False,clocks[1:] == clocks[:-1]],
        interval_starts=starts,interval_ends=ends,interval_durations=ends-starts,
        diagnostic_elapsed=times,diagnostic_absolute=times+meta['time_start'])
    if not geometry['root_partition_valid'] or not geometry['operational_return_clock_valid']:
        raise RuntimeError('Saved geometry does not validate the prescribed roots/operational marker')
    if times.tolist() != [0.,128.,14464.]:
        raise RuntimeError('A different validated template clock was saved for actual review before further diagnostics')
    return dict(geometry=geometry,clock_template=clock_metadata),outputs


def exact_record(value):
    value = Fraction(value)
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def cut30_pairs(h):
    split = 3840
    support = list(range(split-8,split+8))
    rows = [{int(j):rational(float(b.imag)) for j,b in zip(
        h.indices[h.indptr[i]:h.indptr[i+1]],h.data[h.indptr[i]:h.indptr[i+1]],strict=True)}
        for i in range(16384)]
    exact = {}
    for i,row in enumerate(rows):
        for middle,left in row.items():
            for j,right in rows[middle].items():
                if j <= i or (j >= split) == (i >= split):
                    continue
                sign = int(j >= split)-int(i >= split)
                exact[i,j] = exact.get((i,j),Fraction(0))-sign*left*right/2
    exact = {key:value for key,value in exact.items() if value}
    if not exact or len(exact) > 120 or any(i not in support or j not in support for i,j in exact):
        raise ValueError('The actual second-cut current violates the required sixteen-row support')
    pairs,row_bounds = [],dict.fromkeys(support,Fraction(0))
    for (i,j),value in sorted(exact.items()):
        coefficient = float(value)
        if not np.isfinite(coefficient) or coefficient == 0:
            raise ArithmeticError('A nonzero second-cut coefficient is not representable')
        error = abs(value-rational(coefficient))
        row_bounds[i] += abs(rational(coefficient))+error
        row_bounds[j] += abs(rational(coefficient))+error
        pairs.append(dict(rows=[i,j],coefficient=coefficient,paired_coefficient=-coefficient,
            exact_coefficient=exact_record(value),coefficient_error_exact=exact_record(error),
            coefficient_error_upper=upwards(error)))
    return support,pairs,upwards(max(row_bounds.values()))


def paired_matrix(support,pairs):
    matrix = np.zeros((len(support),len(support)),dtype=np.complex128)
    positions = {row:index for index,row in enumerate(support)}
    for pair in pairs:
        i,j = [positions[row] for row in pair['rows']]
        if pair['paired_coefficient'] != -pair['coefficient']:
            raise ValueError('A current coefficient lacks its exact negative pair')
        matrix.imag[i,j] = pair['coefficient']
        matrix.imag[j,i] = pair['paired_coefficient']
    if np.any(matrix != matrix.conj().T):
        raise ValueError('The rounded paired current is not Hermitian')
    return matrix


def window_norm_bound(matrix):
    rows,columns = [Fraction(0)]*len(matrix),[Fraction(0)]*len(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = abs(rational(float(matrix[i,j].real)))+abs(rational(float(matrix[i,j].imag)))
            rows[i] += value
            columns[j] += value
    return dict(absolute_row_sum_upper=upwards(max(rows)),absolute_column_sum_upper=upwards(max(columns)),
                stored_window_F_upper=upwards(sqrt_upper(len(matrix)*max(rows)*max(columns))))


def local_operators(context,stages,binding):
    with np.load(stages.output('fields','readout_fields.npz'),allow_pickle=False) as saved:
        radius,clock,flow = (saved[key] for key in ('radius','clock','flow'))
    h = context.load_h()
    origin = context.origin['current']
    keys = ('rows','coefficient','paired_coefficient','exact_coefficient','coefficient_error_exact','coefficient_error_upper')
    support16 = origin['verified_support_spinor_rows']
    pairs16 = [{key:pair[key] for key in keys} for pair in origin['pairs']]
    if support16 != list(range(2040,2056)) or origin['cut_radius'] != 1024.:
        raise ValueError('The accepted origin current is not the original cut16 operator')
    support30,pairs30,norm30 = cut30_pairs(h)
    outputs,definitions = {},{}
    for cut,support,pairs,norm in ((16,support16,pairs16,origin['exact_stored_operator_norm_upper']),
                                    (30,support30,pairs30,norm30)):
        name = f'current{cut}.npz'
        definition = dict(kind='native C-only i[h^2,chi]/2',cut_radius=cut*64.,
            support=support,pairs=pairs,exact_stored_operator_norm_upper=norm,
            reused_origin_coefficients=(cut == 16),coefficient_rounding_scope='separate from local kernel construction')
        outputs[name] = save_arrays(stages.root/name,dict(**binding,definition=definition),
                                   support=np.array(support),matrix=paired_matrix(support,pairs))
        definitions[f'current{cut}'] = definition
    for width in WINDOWS:
        support,operators,window = signed_window_observables(radius,clock,flow,3,1024.,width)
        definition = dict(window=window,support=support.tolist(),norm_bounds={
            name:window_norm_bound(operators[name]) for name in BRANCHES},
            scope='stored local-symbol matrices; ideal-symbol construction/eigenpair error remains separate')
        name = f'window{width}.npz'
        outputs[name] = save_arrays(stages.root/name,dict(**binding,definition=definition),support=support,**operators)
        definitions[f'window{width}'] = definition
    outputs['local_operators.json'] = save_json(stages.root/'local_operators.json',dict(
        input_binding=binding,definitions=definitions,physical_rows_released=False))
    return definitions,outputs


def tail_stage(context,stages,binding,e,v,d):
    from hold_tail_checks import tail_diagnostics
    spectral = context.precision['spectral_inflation']
    helper_controls = dict(eigenvector_gram_F=context.checks['eigenvector_gram_F'],
        source_covariance_F=spectral['source_covariance_F_upper'],
        source_covariance_purity_F_lower=spectral['source_covariance_purity_F_lower'],
        source_covariance_purity_F_upper=spectral['source_covariance_purity_F_upper'],
        arithmetic_precision=spectral)
    legacy,arrays = tail_diagnostics(v,e,d,context.source.occupied,context.load_h(),helper_controls)
    rows = []
    for old in legacy['tails']:
        row = dict(sites=old['sites'],spinor_rows=old['spinor_rows'],
            direct_computed_source_readings={key:value for key,value in old.items()
                if key.startswith(('source_','direct_','directly_'))},
            legacy_conditional_estimates=old,source_history_additions={},
            reconstructed_row_multiplication_error=None,
            reconstructed_row_multiplication_status='unresolved; separate from mathematical beta_reconstruction',
            true_h_projector_replacement='separate raw-V Gram/polar/true-projector allowance; not included in direct rows')
        for kind,source_error in context.precision['source_covariance_errors'].items():
            b = rational(source_error['covariance_F_error_upper'])
            lower = max(Fraction(0),rational(old['direct_source_row_F_lower'])-b)
            upper = rational(old['direct_source_row_F_upper'])+b
            row['source_history_additions'][kind] = dict(covariance_F_upper=upwards(b),
                same_reference_row_F_interval=[downwards(lower),upwards(upper)],
                signed_trace_source_addition_upper=upwards(sqrt_upper(old['spinor_rows'])*b),
                scope='same raw-V reference and common exact stored held unitary; tail scalar/projector/construction terms separate')
        rows.append(row)
    report = dict(input_binding=binding,input=context.source.metadata,tails=rows,
        saved_transform_allowance=context.precision['saved_transform_allowance'],
        inflated_spectral_controls=spectral,legacy_helper_report=legacy,
        physical_rows_released=False,
        scope='direct source-row arithmetic controls plus separately scoped transform/history terms; no combined physical tail certificate')
    output_arrays = save_arrays(stages.root/'tail_arrays.npz',binding,**arrays)
    report['arrays'] = output_arrays
    output_report = save_json(stages.root/'tail_checks.json',report)
    return dict(tail_sites=[2,64,512],physical_rows_released=False),{
        'tail_arrays.npz':output_arrays,'tail_checks.json':output_report}


def kernel_save(context,stages,binding,name,kernel,e,definition):
    statistics = kernel_statistics(kernel)
    spectral = context.precision['spectral_inflation']
    rho = rational(context.source.method.rho)
    matrix_bound = rational(context.dhat_F_upper())
    gram_allowance = rho*rational(spectral['gram_F_upper'])*matrix_bound
    if definition['observable'] == 'energy':
        source_coefficient = sqrt_upper(16384)*rho
        source_scope = 'exact defined stored-h energy allocation; rounded cut/kernel/eigenpair construction separate'
    elif definition['observable'] == 'current':
        source_coefficient = 4*rho**2
        source_scope = 'exact defined stored-h native current; rounded coefficients/kernel construction separate'
    else:
        source_coefficient = rational(definition['local_definition']['norm_bounds'][definition['branch']]['stored_window_F_upper'])
        source_scope = 'stored local window matrix; ideal-symbol difference separate'
    source_additions = {kind:dict(observable_F_upper=upwards(source_coefficient),
        covariance_F_error_upper=entry['covariance_F_error_upper'],
        source_observable_error_upper=upwards(source_coefficient*rational(entry['covariance_F_error_upper'])),
        scope='same exact stored held unitary only; differing model held operator/reference/evolution unresolved')
        for kind,entry in context.precision['source_covariance_errors'].items()}
    metadata = dict(definition=definition,statistics=statistics,Dhat_F_upper=upwards(matrix_bound),
        outside_shortcut_allowance_upper=upwards(gram_allowance) if definition['observable'] == 'energy' else None,
        outside_shortcut_scope='finite-V I-minus-inside correction only; cut product/kernel/source construction separate',
        saved_transform_allowance=context.precision['saved_transform_allowance'],physical_rows_released=False)
    metadata.update(source_additions=source_additions,source_coefficient_scope=source_scope)
    saved = save_arrays(stages.root/name,dict(**binding,kernel_metadata=metadata),kernel=kernel,eigenvalues=e)
    return metadata,{name:saved}


def kernel_load(stages,stage,name,e):
    with np.load(stages.output(stage,name),allow_pickle=False) as raw:
        ordering,kernel = raw['eigenvalues'],raw['kernel']
        binding = json.loads(raw['input_binding_json'].item())
    if (ordering.dtype != e.dtype or not np.array_equal(ordering.view(np.uint64),e.view(np.uint64))
            or kernel.dtype != np.complex128 or kernel.shape != (16384,16384)):
        raise ValueError('A saved recovered kernel has another full dimension/ordering')
    return kernel,binding['kernel_metadata']


def origin_stage(context,stages,binding,e,diagonal):
    from hold_origin_diagnostics import derived, kernel_reductions, value_record
    from hold_recovered_precision import exact_dot
    from hold_recovered_origin import cell_comparison
    saved_energy = context.precision['initial_energies']
    whole,errors,whole_records = {},{},{}
    for kind in KINDS:
        if kind == 'signed':
            real,imag = saved_energy['deviation_energy_real'],saved_energy['deviation_energy_imaginary']
        else:
            selected = e > 0 if kind == 'particle' else e < 0
            real,imag = exact_dot(e[selected],diagonal.real[selected]),exact_dot(e[selected],diagonal.imag[selected])
        whole[kind] = complex(real['value'],imag['value'])
        errors[kind] = max(rational(real['conversion_error_upper']),rational(imag['conversion_error_upper']))
        whole_records[kind] = dict(value_record(whole[kind],errors[kind]),
            real_exact_scalar=real,imaginary_exact_scalar=imag,
            reused_saved_signed_sum=(kind == 'signed'))
    outside,outside_errors,kernel_records,outputs,raw_blocks,raw_pairs = {},{},{},{},[],[]
    for cut in CUTS:
        outside[cut],outside_errors[cut] = {},{}
        for kind in KINDS:
            stage,name = cut+'_'+kind,cut+'_'+kind+'_energy.npz'
            kernel,metadata = kernel_load(stages,stage,name,e)
            primary,paired,blocks,paired_rows = kernel_reductions(kernel)
            del kernel
            scale = rational(metadata['statistics']['kernel_abs_sum_arithmetic_model_upper'])
            primary_error,paired_error = gamma_rational(28)*scale,gamma_rational(29)*scale
            outside[cut][kind],outside_errors[cut][kind] = primary,primary_error
            key = cut+'_'+kind
            kernel_records[key] = dict(primary=value_record(primary,primary_error),paired=value_record(paired,paired_error),
                paired_minus_primary=[float((paired-primary).real),float((paired-primary).imag)],
                kernel_metadata=metadata,primary_depth=28,paired_depth=29)
            filename = 'origin_'+key+'.npz'
            outputs[filename] = save_arrays(stages.root/filename,dict(**binding,parent_kernel=stages.report['stages'][stage]['outputs'][name]),
                primary=np.array(primary),paired=np.array(paired),primary_block_partials=blocks,paired_row_partials=paired_rows)
            raw_blocks.append(blocks)
            raw_pairs.append(paired_rows)
    cuts = {}
    for cut in CUTS:
        values,allowances = outside[cut],outside_errors[cut]
        cuts[cut] = dict(outside={kind:kernel_records[cut+'_'+kind]['primary'] for kind in KINDS},
            native_signed_inside=derived([whole['signed'],values['signed']],[errors['signed'],allowances['signed']],[1,-1])[2],
            positive_outside_sum=derived([values['particle'],values['hole']],[allowances['particle'],allowances['hole']],[1,1])[2],
            localization_difference=derived([values['particle'],values['hole'],values['signed']],
                [allowances['particle'],allowances['hole'],allowances['signed']],[1,1,-1])[2])
    regions,region_values,region_errors = {},{},{}
    for kind in KINDS:
        values = [whole[kind],outside['root_inner'][kind],outside['root_outer'][kind],outside['cut16'][kind]]
        bounds = [errors[kind],outside_errors['root_inner'][kind],outside_errors['root_outer'][kind],outside_errors['cut16'][kind]]
        result = {}
        region_values[kind],region_errors[kind] = [],[]
        for name,weights in (('core',[1,-1,0,0]),('between_roots',[0,1,-1,0]),
                              ('outer_root_to_collar',[0,0,1,-1]),('outside_collar',[0,0,0,1])):
            value,error,record = derived(values,bounds,weights)
            result[name] = record
            region_values[kind].append(value)
            region_errors[kind].append(error)
        result['telescoping_residual'] = derived(region_values[kind]+[whole[kind]],
            region_errors[kind]+[errors[kind]],[1,1,1,1,-1])[2]
        result['scope'] = 'algebraic difference check only; no independent continuity certificate'
        regions[kind] = result
    external = context.source.external_origin_targets['selected']['energy_row']['per_channel_3']
    counterpart = {'E_inv':whole_records['signed'],'E_out':cuts['cut16']['outside']['signed'],
                   'E_inv_inside':cuts['cut16']['native_signed_inside']}
    comparisons = {name:dict(own=record,external_original_history_value=external[name],
        fixed_kernel_comparison=cell_comparison(record['real'],external[name],6,
            dict(fixed_saved_kernel=record['fixed_saved_input_error_upper_per_component'])),
        scope='original-history source counterpart only; unknown producer seed/version and held-row comparators remain separate')
        for name,record in counterpart.items()}
    numerator,numerator_error,numerator_record = derived([region_values[k][0] for k in ('particle','hole')],
        [region_errors[k][0] for k in ('particle','hole')],[1,1])
    denominator,denominator_error,denominator_record = derived([whole[k] for k in ('particle','hole')],
        [errors[k] for k in ('particle','hole')],[1,1])
    denominator_clear = abs(rational(denominator.real)) > denominator_error
    raw_id = save_arrays(stages.root/'origin_energy_arrays.npz',binding,eigenvalues=e,
        saved_deviation_diagonal=diagonal,primary_block_partials=np.array(raw_blocks),paired_row_partials=np.array(raw_pairs))
    outputs['origin_energy_arrays.npz'] = raw_id
    report = dict(input_binding=binding,input=context.source.metadata,completed=True,elapsed=0.,physical_rows_released=False,
        whole_totals=whole_records,cuts=cuts,root_regions=regions,kernels=kernel_records,
        inner_positive_inventory_fraction=dict(nominal=numerator.real/denominator.real if denominator_clear else None,
            numerator=numerator_record,denominator=denominator_record,
            fixed_saved_scalar_denominator_excludes_zero=denominator_clear,
            status='construction/Pauli/effect uncertainty unresolved; origin-location diagnostic only'),
        whole_positive_minus_signed=derived([whole[k] for k in ('particle','hole','signed')],
            [errors[k] for k in ('particle','hole','signed')],[1,1,-1])[2],
        original_history_comparisons=comparisons,arrays=raw_id,
        source_covariance_errors=context.precision['source_covariance_errors'],
        saved_transform_allowance=context.precision['saved_transform_allowance'],
        existing_origin_current_control=context.binding['accepted_origin_control'],
        construction_status='unresolved kernel/cut/eigenpair/projector/phase and producer association; scalar intervals are not physical certificates')
    outputs['origin_energy_checks.json'] = save_json(stages.root/'origin_energy_checks.json',report)
    return dict(energy_kernel_count=12,physical_rows_released=False),outputs


def run(args):
    context = load_readout(args)
    template_id = context.source.inputs.add(args.clock_template)
    stages = ReadoutStages(context,args.out,INVENTORY,'prepare',args.resume,
                           extra_binding=dict(clock_template=template_id))
    fields = stages.execute('fields',lambda binding:fields_and_clocks(
        context,stages,binding,template_id['path']))
    definitions = stages.execute('local_operators',lambda binding:local_operators(context,stages,binding))
    e,v,d = context.load_spectrum_deviation()
    stages.execute('tails',lambda binding:tail_stage(context,stages,binding,e,v,d))
    with np.load(stages.output('fields','readout_fields.npz'),allow_pickle=False) as saved:
        radius = saved['radius']
    geometry = fields['geometry']
    cuts = dict(cut16=1024.,cut30=1920.,root_inner=geometry['applied_roots'][0],root_outer=geometry['applied_roots'][1])
    for cut,position in cuts.items():
        def make_chi(binding):
            chi = sharp_outside_in_eigenbasis(v,radius,position)
            name = cut+'_chi.npz'
            saved = save_arrays(stages.root/name,dict(**binding,cut=cut,cut_radius=position,
                convention='I-minus-inside; product rounding and finite-V correction separate'),
                chi=chi,inside_spinor_mask=np.repeat(radius <= position,2),eigenvalues=e)
            return dict(cut_radius=position,operation='one V_inside^dagger V_inside then I-minus'),{name:saved}
        stages.execute(cut+'_chi',make_chi)
        with np.load(stages.output(cut+'_chi',cut+'_chi.npz'),allow_pickle=False) as saved:
            chi = saved['chi']
            if not np.array_equal(saved['eigenvalues'].view(np.uint64),e.view(np.uint64)):
                raise ValueError('The saved cut product has another eigenvalue order')
        for kind in KINDS:
            def make_energy(binding):
                kernel = cut_energy_kernel(chi,e,d,kind)
                return kernel_save(context,stages,binding,cut+'_'+kind+'_energy.npz',kernel,e,
                    dict(observable='energy',kind=kind,cut=cut,cut_radius=position,units='1/a',
                         cut_product=stages.report['stages'][cut+'_chi']['outputs'][cut+'_chi.npz']))
            stages.execute(cut+'_'+kind,make_energy)
        chi = None
        if cut in ('cut16','cut30'):
            def make_current(binding):
                number = cut[3:]
                with np.load(stages.output('local_operators','current'+number+'.npz'),allow_pickle=False) as saved:
                    support,matrix = saved['support'],saved['matrix']
                kernel = local_kernel(v,d,support,matrix)
                return kernel_save(context,stages,binding,cut+'_native_current.npz',kernel,e,
                    dict(observable='current',kind='native',cut_radius=position,units='1/a^2',
                         local_definition=definitions['current'+number]))
            stages.execute(cut+'_current',make_current)
    for width in WINDOWS:
        with np.load(stages.output('local_operators',f'window{width}.npz'),allow_pickle=False) as saved:
            support = saved['support']
            operators = {name:saved[name] for name in BRANCHES}
        for branch in BRANCHES:
            name = f'window{width}_{branch}'
            def make_window(binding):
                kernel = local_kernel(v,d,support,operators[branch])
                return kernel_save(context,stages,binding,name+'.npz',kernel,e,
                    dict(observable='window',branch=branch,width=width,units='1/a^2',
                         local_definition=definitions[f'window{width}']))
            stages.execute(name,make_window)
    diagonal = d.diagonal().copy()
    v = d = None
    stages.execute('origin',lambda binding:origin_stage(context,stages,binding,e,diagonal))
    stages.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('descriptor','build','comparison','origin','spectral-progress','clock-template','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
