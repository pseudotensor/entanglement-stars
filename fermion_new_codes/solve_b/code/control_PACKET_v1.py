"""Manufactured packet controls only; no physical archive intake."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse import csr_matrix
from cpu_runtime import require_remote_cpu
from packet_core_v1 import (S1, S2, S3, operators, lower_band, bare_packet, projected_packet,
    exponential_action, tail_degree, pure_distance, row_constant, root_partition,
    applied_roots, validate_history, variance_value, validate_roster, contraction_status,
    midpoint_fractions, fourier_diagnostics, velocity_parts, source_snapshot, verified_controls, propagation_step, held_ledger, spectral_admission,
    comparison_association, readout_admission, admit_record_observables, validate_completed_states,
    packet_labels, validate_step_association, pure_distance_details, validate_observable_schema, OBSERVABLE_SCALARS)

from summarize_PACKET_secondroute_v1 import extreme, located_differences, reversals


def json_scalar(value):
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args()
    runtime = require_remote_cpu(args.output)
    checks = []
    def require_equal(actual,expected=0.,scale=1.):
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)),initial=0))
        if not np.isfinite(error) or error > 5e-12*max(1.,scale):
            raise ValueError('manufactured equality rejected')
    def equal(name, actual, expected=0., scale=1.):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)), initial=0))
        tol = 5e-12*max(1., scale)
        try:
            require_equal(actual,expected,scale)
            passed=True
        except ValueError:
            passed=False
        checks.append({'name': name, 'error': error, 'tolerance': tol, 'passed': passed})
    def reject(name, fn):
        try:
            fn()
        except (ValueError, KeyError, IndexError):
            checks.append({'name': name, 'passed': True, 'guard_rejected': True})
        else:
            checks.append({'name': name, 'passed': False, 'guard_rejected': False})
    rng = np.random.default_rng(721)
    for size in (1, 2, 3, 7):
        r = np.arange(1., size+1)
        n = 1+.03*r
        v = -.6+.07*r
        op = operators(r, n, v, 2)
        dense = np.zeros((2*size, 2*size), complex)
        for i in range(size):
            dense[2*i:2*i+2, 2*i:2*i+2] = n[i]*(.2*S3+2/r[i]*S2)
            for step, field, pauli, den in ((1, n, S1, 4j), (2, v, np.eye(2), 8j)):
                j = i+step
                if j < size:
                    block = (field[i]+field[j])/den*pauli
                    dense[2*i:2*i+2, 2*j:2*j+2] = block
                    dense[2*j:2*j+2, 2*i:2*i+2] = block.conj().T
        equal(f'block_{size}', op['h'].toarray(), dense)
        equal(f'hermitian_{size}', dense, dense.conj().T)
        psi = rng.normal(size=(2*size, 3))+1j*rng.normal(size=(2*size, 3))
        equal(f'action_{size}', op['h']@psi, dense@psi)
        reconstructed = np.diag(lower_band(op['h'])[0])
        for off in range(1, 5):
            if off < 2*size:
                d = lower_band(op['h'])[off, :2*size-off]
                reconstructed += np.diag(d, -off)+np.diag(d.conj(), off)
        equal(f'band_{size}', reconstructed, dense)
        position = np.diag(np.repeat(r, 2))
        for part, distance in [('hn', 1), ('hv', 2)]:
            exact = velocity_parts(op, r)[0 if part == 'hn' else 1].toarray()
            direct = np.zeros_like(exact)
            field = n if part == 'hn' else v
            pauli = S1 if part == 'hn' else np.eye(2)
            for i in range(size-distance):
                j = i+distance
                block = (field[i]+field[j])/4*pauli
                direct[2*i:2*i+2, 2*j:2*j+2] = block
                direct[2*j:2*j+2, 2*i:2*i+2] = block
            equal(f'velocity_{part}_{size}', exact, direct)
        constant, _, _ = row_constant(r, 2, v)
        for i in range(size):
            row = (op['h0']+v[i]*op['q']).toarray()[2*i:2*i+2]
            equal(f'row_gram_{size}_{i}', row@row.conj().T, np.eye(2)*(constant[i]/2)**2)
            equal(f'row_trace_{size}_{i}', np.linalg.svd(row, compute_uv=False).sum(), constant[i])
        e, u = eigh(op['h0'].toarray())
        checks.append({'name': f'spectrum_admission_{size}', 'passed': spectral_admission(op['h0'],e,u,.2)['gap_consistent_with_mass']})
        A = op['h0'].toarray()[::2, 1::2]
        paired = np.sqrt(.2**2+np.linalg.svd(A, compute_uv=False)**2)
        equal(f'paired_{size}', e, np.sort(np.r_[-paired, paired]))
        for centre in (1., 2.3):
            raw, _ = bare_packet(r, centre, -.3)
            projected, q = projected_packet(raw, u[:, e > 0])
            equal(f'projected_norm_{size}_{centre}', np.vdot(projected, projected), 1.)
            equal(f'negative_projection_{size}_{centre}', u[:, e < 0].conj().T@projected, 0.)
            equal(f'preparation_phase_{size}_{centre}', pure_distance(projected, projected_packet(raw*1j, u[:, e > 0])[0]), 0.)
        psi /= np.linalg.norm(psi, axis=0)
        h0psi, qpsi = op['h0']@psi, op['q']@psi
        local = psi.reshape(size, 2, -1)
        source = np.real(np.sum(local.conj()*(h0psi+np.repeat(v, 2)[:, None]*qpsi).reshape(size, 2, -1), axis=1))
        cauchy = np.sqrt(np.sum(abs(local)**2, axis=1))*constant[:, None]/2
        equal(f'rankone_cauchy_{size}', np.maximum(abs(source)-cauchy, 0), 0.)
    up = np.array([[1.], [0.]])
    equal('half_projection_retained', projected_packet(np.ones(2)/np.sqrt(2), up)[1], .5)
    reject('null_projection', lambda: projected_packet(np.array([0., 1.]), up))
    equal('pure_phase', pure_distance(np.array([1., 0.]), np.array([1j, 0.])), 0.)
    equal('pure_orthogonal', pure_distance(np.array([1., 0.]), np.array([0., 1.])), 1.)
    centrifugal = np.hypot(.2, 5.)
    checks.append({'name': 'centrifugal_above_homogeneous', 'passed': bool(centrifugal > np.hypot(.2, 1.)), 'energy': centrifugal})
    for label, h in [('zero', np.zeros((2, 2))), ('s3', S3), ('2s3', 2*S3), ('s1', S1)]:
        psi = rng.normal(size=(2, 3))+1j*rng.normal(size=(2, 3))
        for t in (-1.7, 0., .3, 2., 30.):
            scale = 2*float(np.linalg.norm(h, 2))
            result, meta = exponential_action(csr_matrix(h), psi, t, scale)
            frequency = 0. if label == 'zero' else 2. if label == '2s3' else 1.
            exact_matrix = np.eye(2) if frequency == 0 else np.cos(t*frequency)*np.eye(2)-1j*np.sin(t*frequency)*(h/frequency)
            expected = exact_matrix@psi
            equal(f'exponential_{label}_{t}', result, expected)
            equal(f'expm_additional_{label}_{t}', expm(-1j*t*h)@psi, expected)
            equal(f'norm_{label}_{t}', np.sum(abs(result)**2, axis=0), np.sum(abs(psi)**2, axis=0))
            back, _ = exponential_action(csr_matrix(h), result, -t, scale)
            equal(f'reverse_{label}_{t}', back, psi)
            checks.append({'name': f'tail_{label}_{t}', 'passed': meta['tail'] <= 1e-14, **meta})
            for j in range(3):
                alone, _ = exponential_action(csr_matrix(h), psi[:, j], t, scale)
                equal(f'column_{label}_{t}_{j}', alone, result[:, j])
    for z in (0., .1, 3., 20., 120.):
        degree, bound = tail_degree(z)
        checks.append({'name': f'tail_degree_{z}', 'passed': bound <= 1e-14 and degree <= 4096, 'degree': degree, 'bound': bound})
    state = np.array([np.sqrt(3)/2, .5], complex)
    out, _ = exponential_action(csr_matrix(S3), state, .7, 2.)
    energy = lambda x, h: float(np.vdot(x, h@x).real/np.vdot(x, x).real)
    equal('held_energy', energy(out, S3), .5)
    equal('next_energy', energy(out, 2*S3), 1.)
    equal('held_work', energy(out, 2*S3)-energy(state, S3), energy(out, S3))
    equal('raw_work', np.vdot(out, 2*S3@out)-np.vdot(state, S3@state), np.vdot(out, S3@out))
    commuting, _ = propagation_step(csr_matrix(S3), csr_matrix(2*S3), state, 1., 16, lambda f:4., held=False)
    equal('commuting_midpoint', commuting, expm(-1.5j*S3)@state)
    errors = []
    reverse_errors = []
    exact = expm(-.5j*S3)@expm(-1j*(S1-.5*S3))@state
    for intervals in (8, 16, 32):
        hs = [np.cos(t)*S1+np.sin(t)*S2 for t in np.linspace(0, 1, intervals+1)]
        x = state.copy()
        rev = state.copy()
        for j in range(intervals):
            x, _ = propagation_step(csr_matrix(hs[j]), csr_matrix(hs[j+1]), x, 1/intervals, 1, lambda f:3.)
            k = intervals-1-j
            rev, _ = propagation_step(csr_matrix(hs[k]), csr_matrix(hs[k+1]), rev, 1/intervals, 1, lambda f:3.)
        errors.append(float(np.linalg.norm(x-exact)))
        reverse_errors.append(float(np.linalg.norm(rev-exact)))
    checks.append({'name': 'noncommuting_refinement', 'errors': errors, 'passed': all(errors[j] >= 1.5*errors[j+1] for j in range(2))})
    checks.append({'name': 'reversed_order_detected', 'errors': reverse_errors, 'passed': reverse_errors[-1] > 10*errors[-1]})
    # Manufactured periodic controls use independent periodic shift only here.
    size = 12
    shift = np.roll(np.eye(size), 1, axis=1)
    for k in (np.pi/6, np.pi/3, np.pi/2, 5*np.pi/6, -np.pi/6, -np.pi/3, -np.pi/2, -5*np.pi/6):
        spin = bare_packet(np.arange(1., 13.), 6., k, sigma=1e20)[0].reshape(size, 2)
        # Infinite-width bare packet has exactly a single periodic Fourier mode.
        psi = spin.reshape(-1)
        p = (shift-shift.T)/(2j)
        q = (shift@shift-shift.T@shift.T)/(4j)
        h = np.kron(np.eye(size), .2*S3)+np.kron(p, S1)-.7*np.kron(q, np.eye(2))
        kinetic = np.kron((shift+shift.T)/2, S1)
        flow = -.7*np.kron((shift@shift+shift.T@shift.T)/2, np.eye(2))
        eps = np.hypot(.2, np.sin(k))
        equal(f'periodic_energy_{k}', energy(psi, h), eps-.7*np.sin(2*k)/2)
        equal(f'periodic_current_{k}', energy(psi, np.kron(q, np.eye(2))), np.sin(2*k)/2)
        equal(f'periodic_kinetic_{k}', energy(psi, kinetic), np.sin(k)*np.cos(k)/eps)
        equal(f'periodic_flow_{k}', energy(psi, flow), -.7*np.cos(2*k))
        grid, bins, probability, masks = fourier_diagnostics(spin[:, :, None])
        probability = probability[:, 0]
        equal(f'parseval_{k}', probability.sum(), 1.)
        equal(f'onebin_{k}', probability.max(), 1.)
    r = np.arange(1., 8.)
    mask = root_partition(r, [2., 5.])
    equal('boundary_inclusion', mask[:, 1], [1, 0, 0])
    equal('partition_sum', mask.sum(axis=0), np.ones(7))
    checks.append({'name': 'missing_partition_unavailable', 'passed': root_partition(r, None) is None})
    equal('moving_root_stationary', float(5 > 4)-float(5 > 6), 1.)
    for label, f, status in [('pair', [1, 1, -1, -1, 1, 1, 1], 'resolved_pair'), ('absent', [1]*7, 'absent_or_unpaired'), ('plateau', [1, 0, 0, -1, 1, 1, 1], 'ambiguous'), ('multiple', [1, -1, 1, -1, 1, 1, 1], 'ambiguous')]:
        roots = applied_roots(r, np.ones(7)*2, -(2-np.asarray(f)))
        checks.append({'name': f'roots_{label}', 'passed': roots['status'] == status, 'status': roots['status']})
    for label, lapse in [('complex', np.ones(3)+1j), ('zero', [1, 0, 1]), ('negative', [1, -1, 1]), ('nonfinite', [1, np.nan, 1])]:
        reject('lapse_'+label, lambda lapse=lapse: operators(np.arange(1., 4.), lapse, np.zeros(3), 1))
    reject('changed_grid', lambda: operators(np.array([1., 2., 4.]), np.ones(3), np.zeros(3), 1))
    reject('nonfinite_state', lambda: exponential_action(csr_matrix(S3), np.array([np.nan, 1.]), 1., 2.))
    reject('zero_norm', lambda: projected_packet(np.zeros(2), up))
    reject('unknown_width', lambda: bare_packet(r, 2., .3, convention='unknown'))
    reject('degree_cap', lambda: tail_degree(1e8))
    # Guard fixtures use a manufactured complete structural history, not input archives.
    history = [{'T': j*8., 'T_over_rs': j/8.} for j in range(321)]
    history[97].update({'trapping_zeros': [2., 5.], 'r_in': 2., 'R_t': 5.})
    fields = {'T': np.arange(321)*8., 'R': np.arange(1., 2049.), 'N_applied': np.ones((321, 2048)), 'v_applied': np.zeros((321, 2048)), 'eps_tot': np.zeros((321, 2048)), 'J_tot': np.zeros((321, 2048)), 'g': np.array(.25), 'ell': np.array(4), 'tau': np.array(8.)}
    checks.append({'name': 'association_good', 'passed': validate_history(history, fields)})
    for label in ('duplicate_clock', 'missing_clock', 'record_association', 'missing_source', 'grid', 'scalar'):
        hh = [dict(row) for row in history]
        ff = dict(fields)
        if label == 'duplicate_clock':
            ff['T'] = fields['T'].copy(); ff['T'][3] = ff['T'][2]
        elif label == 'missing_clock': ff['T'] = fields['T'][:-1]
        elif label == 'record_association': hh[2]['T'] = 99.
        elif label == 'missing_source': del ff['eps_tot']
        elif label == 'grid': ff['R'] = fields['R']+.5
        else: ff['tau'] = np.array([8.])
        reject('association_'+label, lambda hh=hh, ff=ff: validate_history(hh, ff))
    reject('wrong_roster', lambda: validate_roster([('c', .3)], [('c', -.3)]))
    reject('midpoint_count', lambda: midpoint_fractions(0))
    checks.append({'name': 'noncontraction_rejected', 'passed': contraction_status(2*np.eye(2)) == 'noncontraction'})
    raw, displayed, clipped = variance_value(np.array([1., 1., 1.]), np.array([1., np.sqrt(1+1e-13), 2.]))
    checks.append({'name': 'variance_cancellation_policy', 'passed': bool(displayed[0] == 0 and displayed[1] == 0 and np.isnan(displayed[2]) and clipped[1])})
    probe = np.zeros((12, 2, 1), complex); probe[:, 0, 0] = 1/np.sqrt(12)
    grid, bins, probability, masks = fourier_diagnostics(probe)
    equal('nyquist_positive', grid[bins == 6], np.pi)
    checks.append({'name': 'analytic_zero_masks', 'passed': bool(not masks['negative_cosine'][bins == 0][0] and not masks['opposed_current'][bins == 6][0] and not masks['opposed_current'][bins == 3][0])})
    # Exact integer Fourier boundary masks on an eight-site fixture.
    for bin_number in (-4, -3, -2, -1, 0, 1, 2, 3):
        eight = np.zeros((8, 2, 1), complex)
        eight[:, 0, 0] = np.exp(2j*np.pi*bin_number*np.arange(8)/8)/np.sqrt(8)
        grid, bins, probability, masks = fourier_diagnostics(eight)
        peak = int(np.argmax(probability[:, 0]))
        expected_bin = 4 if bin_number == -4 else bin_number
        equal(f'eight_bin_{bin_number}', bins[peak], expected_bin)
        expected_flow = abs(expected_bin) in (2,)
        expected_current = abs(expected_bin) in (3,)
        checks.append({'name':f'exact_Q_zero_{bin_number}','passed':bool(masks['Q_symbol'][peak]==0.) if (4*expected_bin)%8==0 else True})
        checks.append({'name':f'exact_cos_zero_{bin_number}','passed':bool(masks['cosine_symbol'][peak]==0.) if abs(expected_bin) in (1,3) else True})
        checks.append({'name': f'eight_masks_{bin_number}', 'passed': bool(masks['negative_cosine'][peak] == expected_flow and masks['opposed_current'][peak] == expected_current)})
    # Weighted source sum has exactly one multiplicity per channel.
    r = np.arange(1., 4.)
    hand = sum(16*k*2*np.sqrt(.2**2+(k/r)**2+np.array([1,2,1])/4+np.array([1,0,1])*.3**2/16) for k in range(1,11))/(4*np.pi*r*r)
    actual = sum(16*k*row_constant(r, k, np.full(3,.3))[0] for k in range(1,11))/(4*np.pi*r*r)
    equal('ten_channel_weights', actual, hand, scale=float(np.max(hand)))
    held, _ = propagation_step(csr_matrix(S3), csr_matrix(2*S3), state, .7, 1, lambda f:2., held=True)
    ledger = held_ledger(state[:,None], held[:,None], csr_matrix(S3), csr_matrix(2*S3))
    equal('production_held_before', ledger['frozen_energy_before'], .5)
    equal('production_held_next', ledger['next_energy'], 1.)
    equal('production_held_work', ledger['field_work'], .5)
    equal('production_raw_work', ledger['work_raw_defect_real'], 0.)
    checks.append({'name':'step_association_good','passed':validate_step_association(97,98,97,98)})
    reject('shifted_record_H',lambda:validate_step_association(97,98,98,99))
    reject('wrong_free_gap',lambda:spectral_admission(csr_matrix(.2*S3),np.array([-.2,.2]),np.array([[0.,1.],[1.,0.]]),1.))
    for label,f in [('two_tangencies',[1,0,1,0,1]),('outer_only',[-1,-1,1,1,1]),('wall',[1,-1,-1,-1,-1])]:
        z=applied_roots(np.arange(1.,6.),np.ones(5)*2,-(2-np.asarray(f)))
        expected=label=='outer_only'
        checks.append({'name':'root_geometry_'+label,'passed':z['outer_available']==expected,'geometry':z})
    z=applied_roots(np.arange(1.,6.),np.ones(5)*2,np.array([3.,3.,1.,1.,1.]))
    checks.append({'name':'root_outward_unavailable','passed':not z['outer_available']})
    numerator=sum(16*k*row_constant(np.array([1.]),k,np.array([0.]),mass=0.)[0][0] for k in range(1,11))
    equal('one_site_source_numerator',numerator,12320.,scale=12320.)
    equal('one_site_source_shell',numerator/(4*np.pi),3080/np.pi,scale=3080/np.pi)
    o=operators(np.arange(1.,6.),np.ones(5),np.ones(5)*2,0,mass=0.)
    rows=(o['h0']+2*o['q']).toarray()[4:6]
    equal('five_site_gram_I',rows@rows.conj().T,np.eye(2))
    equal('five_site_trace_two',row_constant(np.arange(1.,6.),0,np.ones(5)*2,mass=0.)[0][2],2.)
    reject('distance_nonfinite',lambda:pure_distance_details(np.array([np.nan,1.]),np.array([1.,0.])))
    reject('distance_zero_norm',lambda:pure_distance_details(np.zeros(2),np.array([1.,0.])))
    pp=np.full((3,2),1/3)
    vals={'dense_shares':np.array([[1.,0.],[0.,1.],[0.,0.]]),'applied_shares':np.full((3,2),np.nan),'position_variance':np.array([1.,2.])}
    admission=admit_record_observables(vals,pp,pp,True,False,False)
    checks.append({'name':'readout_missing_root_separate','passed':all(admission['admitted_columns'])})
    broken=pp.copy();broken[0,0]=np.nan
    checks.append({'name':'readout_nonfinite_probability','passed':not all(admit_record_observables(vals,broken,pp,True,False,False)['admitted_columns'])})
    broken=pp.copy();broken[0,0]=2.
    checks.append({'name':'readout_probability_range','passed':not all(admit_record_observables(vals,broken,pp,True,False,False)['admitted_columns'])})
    checks.append({'name':'derived_nan_rejected','passed':not readout_admission(np.array([np.nan]))['admitted']})
    def schema_fixture(count):
        obj={key:np.zeros((224,count)) for key in OBSERVABLE_SCALARS}
        obj.update({'dense_shares':np.zeros((224,3,count)),'applied_shares':np.zeros((224,3,count)),
            'record_indices':np.arange(97,321),'T':np.arange(97,321)*8.,'momentum_grid':np.zeros(2048),'momentum_bins':np.zeros(2048)})
        return obj
    d=schema_fixture(32)
    d16=schema_fixture(16)
    labs=packet_labels(1,'E');primary=packet_labels(1,'I32')
    equal('comparison_primary_count',comparison_association(d,d16,labs,primary,True),16)
    reject('comparison_unknown_label',lambda:comparison_association(d,d,labs,labs[:-1]+[['amplitude','unknown',.45]],False))
    reject('comparison_missing_clock',lambda:comparison_association(d,{**d,'T':d['T']+1.},labs,labs))
    complete={'records':224,'intervals':223,'start_record':97,'last_record':320,'T_first':776.,'T_last':2560.,'columns':16,'labels':primary}
    checks.append({'name':'state_association_good','passed':validate_completed_states(complete,(224,4096,16),primary)})
    wrong=dict(complete);wrong['last_record']=319
    reject('state_missing_final',lambda:validate_completed_states(wrong,(224,4096,16),primary))
    binding_path=args.output/'binding_fixture.json'
    binding_path.write_text(json.dumps({'passed':True,'source_bindings':source_snapshot(args.spec)}))
    checks.append({'name':'control_binding_valid','passed':bool(verified_controls(binding_path,args.spec))})
    binding_path.write_text(json.dumps({'passed':True,'source_bindings':{}}))
    reject('control_binding_tampered',lambda:verified_controls(binding_path,args.spec))
    binding_path.unlink()
    badrootvals=dict(vals);badrootvals['dense_shares']=np.array([[1.1,0.],[0.,1.],[0.,0.]])
    checks.append({'name':'partition_closure_failure','passed':not all(admit_record_observables(badrootvals,pp,pp,True,False,False)['admitted_columns'])})
    badvar=dict(vals);badvar['position_variance']=np.array([np.nan,2.])
    checks.append({'name':'position_variance_failure','passed':not all(admit_record_observables(badvar,pp,pp,True,False,False)['admitted_columns'])})
    equal('raw_overlap_phase_radicand',pure_distance_details(np.array([1.,0.]),np.array([1j,0.]))['raw_radicand'],0.)
    primary=packet_labels(5,'E')
    reject('comparison_unequal_cohorts',lambda:comparison_association(d16,d16,primary,primary[:-1]))
    reject('unknown_packet_mode',lambda:packet_labels(5,'unknown'))
    reject('complex_scalar',lambda:validate_history(history,{**fields,'tau':np.array(8.+0j)}))
    hh=[dict(row) for row in history];hh[1]['T_over_rs']=99.
    reject('mismatched_display_clock',lambda:validate_history(hh,fields))
    tiny=applied_roots(np.arange(1.,4.),np.ones(3)*1e-300,np.array([-.5,-2.,-.5])*1e-300)
    checks.append({'name':'represented_tiny_sign_crossings','passed':tiny['outer_available'] and len(tiny['crossings'])==2})
    for label,rr in [('empty',np.array([])),('zero',np.array([0.,1.,2.])),('reversed',np.array([3.,2.,1.])),('duplicate',np.array([1.,1.,2.]))]:
        reject('root_radius_'+label,lambda rr=rr:applied_roots(rr,np.ones(len(rr)),np.zeros(len(rr))))
    reject('operator_nonfinite_kappa',lambda:operators(np.arange(1.,4.),np.ones(3),np.zeros(3),np.nan))
    one=operators(np.array([1.]),np.ones(1),np.zeros(1),5)
    oe,ou=eigh(one['h0'].toarray())
    actual_energy=energy(ou[:,-1],one['h0'])
    equal('one_site_positive_eigenstate',actual_energy,np.hypot(.2,5.))
    checks.append({'name':'one_site_above_homogeneous','passed':actual_energy>np.hypot(.2,1.)})
    open_op=operators(np.arange(1.,8.),np.ones(7),np.zeros(7),1)['h0']
    wrapped=open_op.toarray();wrapped[0,-1]=1.;wrapped[-1,0]=1.
    reject('deliberate_wraparound',lambda:lower_band(csr_matrix(wrapped)))
    wrong=open_op.toarray();wrong[0,3]*=-1;wrong[3,0]*=-1
    reject('deliberate_hop_sign',lambda:require_equal(wrong,open_op.toarray()))
    stationary=np.zeros((7,1));stationary[4]=1.
    initial=root_partition(np.arange(1.,8.),[1.,6.])@stationary
    final=root_partition(np.arange(1.,8.),[1.,4.])@stationary
    equal('production_moving_cut_stationary',final[2]-initial[2],1.)
    localized=np.zeros(14,complex);localized[8]=1.
    veln,velv=velocity_parts(operators(np.arange(1.,8.),np.ones(7),np.zeros(7),1),np.arange(1.,8.))
    equal('stationary_cut_current',energy(localized,veln+velv),0.)
    rr=np.array([97,98,99,100])
    ex=extreme(np.array([-2.,3.,3.,np.nan]),rr)
    checks.append({'name':'summary_signed_tied_partial','passed':ex['value']==3. and ex['record_indices']==[98,99] and ex['unavailable_count']==1})
    ex=extreme(np.full(4,np.nan),rr)
    checks.append({'name':'summary_all_missing','passed':ex['value'] is None and ex['unavailable_count']==4})
    lv=np.array([[[-2.],[2.]],[[2.],[np.nan]],[[0.],[1.]],[[np.nan],[0.]]])
    loc=located_differences(lv,rr)[0]
    checks.append({'name':'summary_component_ties','passed':loc['maximum_absolute']==2. and loc['locators']==[[97,0],[97,1],[98,0]] and loc['unavailable_count']==2})
    gap=reversals(np.array([-1.,np.nan,1.,1.]),rr)
    checks.append({'name':'reversal_gap_unresolved','passed':gap['brackets']==[] and gap['gap_spanning_unresolved']==[[97,99]]})
    zero=reversals(np.array([-1.,0.,1.,1.]),rr)
    checks.append({'name':'reversal_zero_bracket','passed':zero['brackets']==[[97,99]] and zero['zero_unresolved_records']==[98]})
    interval=reversals(np.array([-1.,1.]),np.array([97,98]),intervals=True)
    checks.append({'name':'centroid_interval_endpoints','passed':interval['brackets']==[[[97,98],[98,99]]]})
    big=pure_distance_details(np.array([1e100,0.]),np.array([1e100,0.]))
    equal('large_finite_overlap',big['overlap_real'],1.)
    equal('large_finite_distance',big['stable_distance'],0.)
    missing=dict(d16);del missing['energy_0']
    reject('missing_observable_key',lambda:validate_observable_schema(missing,16))
    wrong=dict(d16);wrong['dense_shares']=np.zeros((224,3,15))
    reject('wrong_observable_columns',lambda:validate_observable_schema(wrong,16))
    wrongtime=dict(d);wrongtime['T']=d['T'].copy();wrongtime['T'][2]=99.
    reject('comparison_wrong_clock_value',lambda:comparison_association(d,wrongtime,labs,labs))
    wronglabels=[list(x) for x in labs];wronglabels[-1]=['amplitude','unknown',.45]
    reject('comparison_unknown_complete_label',lambda:comparison_association(d,d,labs,wronglabels))
    permuted=[list(x) for x in labs];permuted[0],permuted[1]=permuted[1],permuted[0]
    reject('comparison_permuted_same_count',lambda:comparison_association(d,d,labs,permuted))
    huge=applied_roots(np.arange(1.,4.),np.array([1e308,1e308,1e308]),np.array([0.,-1.5e308,0.]))
    checks.append({'name':'scaled_root_interpolation','passed':huge['outer_available'] and 2.<=huge['outer_root']<=3.})
    result = {'source_bindings': source_snapshot(args.spec), 'runtime': runtime, 'checks': checks, 'passed': all(c['passed'] for c in checks), 'count': len(checks)}
    (args.output/'known_answers.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=json_scalar)+'\n')
    print(json.dumps({'passed': result['passed'], 'count': len(checks), 'failed': [c for c in checks if not c['passed']]}, default=json_scalar))
    if not result['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
