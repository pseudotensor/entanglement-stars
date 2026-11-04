"""HOLD 18.24: accepted recovered products, never a native-checkpoint fallback."""
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import time

import numpy as np
import scipy.sparse as sp

from hold_error_bounds import rational, sqrt_upper, upwards
from hold_recovered_input import KIND, content_equal, load_recovered
from hold_recovered_spectrum import (STAGE_OUTPUTS, VERSION, binding_hash,
    environment_check, finite_matrix, load_origin, logical, save_npz, save_product_json)
from hold_exact import resource_check, write_json


# Both actual completed spectral judgments and a joint pin review precede use.
ACCEPTED_SPECTRAL_SHA256 = {120: 'f6b895848c3cb398d916eb0fa0b4a79ffbc1a24d0fda90e9b8690729cb4a0a15', 240: 'bcf5a3fc3c10bbcf3ceb83bc7fd855c1d7bcdf2c4999a82a23fa4aea008cb917'}
READOUT_VERSION = 'recovered-original-A-k3-required-readouts-1'
CONSUMER_FILES = ('hold_recovered_readout_input.py','hold_recovered_prepare.py',
    'hold_recovered_diagnostics.py','hold_recovered_spectrum.py','hold_recovered_precision.py',
    'hold_recovered_input.py','hold_recovered_origin.py','hold_exact_readouts.py',
    'hold_origin_diagnostics.py','hold_observables.py','hold_geometry.py','hold_tail_checks.py',
    'hold_interval_checks.py','hold_error_bounds.py','hold_exact.py')


@dataclass
class ReadoutInput:
    source: object
    origin: dict
    progress: dict
    progress_identity: dict
    precision: dict
    checks: dict
    initial: dict
    binding: dict

    def output(self,stage,role):
        saved = self.progress['stages'][stage]['outputs'][role]
        self.source.inputs.verify(saved)
        return Path(saved['path'])

    def check(self):
        self.source.check()

    def load_h(self):
        with np.load(self.output('operator','held_fields'),allow_pickle=False) as saved:
            h = sp.csr_matrix((saved['h_data'],saved['h_indices'],saved['h_indptr']),shape=(16384,16384))
        if (h.dtype != np.complex128 or not h.has_canonical_format or (h-h.getH()).nnz
                or np.any(h.data.real != 0) or not np.all(np.isfinite(h.data))
                or np.max(np.diff(h.indptr)) > 5):
            raise ValueError('The accepted saved CSR violates its original endpoint convention')
        self.check()
        return h

    def load_spectrum_deviation(self):
        with np.load(self.output('spectrum','spectrum'),allow_pickle=False) as saved:
            e,v = saved['eigenvalues'],saved['eigenvectors']
        with np.load(self.output('deviation','deviation'),allow_pickle=False) as saved:
            other,d = saved['eigenvalues'],saved['deviation']
        finite_matrix(v,(16384,16384),np.complex128)
        finite_matrix(d,(16384,16384),np.complex128)
        if (e.dtype != np.float64 or e.shape != (16384,) or not np.all(np.isfinite(e))
                or np.any(e[1:] < e[:-1]) or other.dtype != e.dtype
                or not np.array_equal(e.view(np.uint64),other.view(np.uint64))):
            raise ValueError('The accepted full spectrum/deviation ordering differs')
        self.check()
        return e,v,d

    def dhat_F_upper(self):
        spectral = self.precision['spectral_inflation']
        delta = rational(spectral['gram_F_upper'])
        covariance = rational(spectral['source_covariance_F_upper'])
        beta = rational(self.precision['saved_transform_allowance']['beta_D_upper'])
        count = self.progress['stages']['spectrum']['values']['negative_levels']
        if (not 0 <= delta < 1 or covariance < 0 or beta < 0
                or type(count) is not int or not 0 <= count <= 16384):
            raise ValueError('Accepted saved-matrix bound prerequisites differ')
        return upwards((1+delta)*covariance+sqrt_upper(count)+beta)


def load_readout(args):
    environment_check()
    if any(value is None for value in ACCEPTED_SPECTRAL_SHA256.values()):
        raise RuntimeError('Both actual complete spectral judgments must first be jointly pinned and reviewed')
    source = load_recovered(args.descriptor,args.build,args.comparison)
    origin_id,origin = load_origin(source,args.origin)
    for name in CONSUMER_FILES:
        source.inputs.add(Path(__file__).with_name(name))
    progress_id = source.inputs.add(args.spectral_progress)
    if progress_id['sha256'] != ACCEPTED_SPECTRAL_SHA256[source.metadata['k']]:
        raise ValueError('The recovered spectral result is not the accepted completed snapshot')
    progress = json.loads(Path(progress_id['path']).read_text())
    if ((progress['method'],progress['completed'],progress['physical_rows_released'],progress['source_kind'])
            != (VERSION,True,False,KIND) or list(progress['stages']) != list(STAGE_OUTPUTS)
            or logical(progress['input']) != logical(source.metadata)
            or binding_hash(progress['input_binding']) != progress['input_binding_hash']
            or not content_equal(progress['input_binding']['origin_control'],origin_id)):
        raise ValueError('The completed spectral source/origin/stage tuple differs')
    for saved in progress['input_binding']['verified_inputs'].values():
        source.inputs.verify(saved)
    parent = Path(progress_id['path']).parent
    allowed = {Path(progress_id['path']).name}
    for name,stage in progress['stages'].items():
        if (stage['status'] != 'completed'
                or {Path(s['path']).name for s in stage['outputs'].values()} != STAGE_OUTPUTS[name]
                or stage['prospective_binding']['stage'] != name
                or binding_hash(stage['prospective_binding']['common']) != progress['input_binding_hash']):
            raise ValueError('An accepted spectral stage is incomplete or has another binding')
        for saved in stage['outputs'].values():
            source.inputs.verify(saved)
            if Path(saved['path']).parent != parent:
                raise ValueError('An accepted spectral output has another parent')
            allowed.add(Path(saved['path']).name)
    if {p.name for p in parent.iterdir()} != allowed:
        raise FileExistsError('An orphan spectral product or partial remains')
    controls = {}
    for name,role in (('precision','precision'),('checks','checks'),('initial','initial')):
        saved = progress['stages']['assessment']['outputs'][role]
        controls[name] = json.loads(Path(saved['path']).read_text())
        if logical(controls[name]['input']) != logical(source.metadata):
            raise ValueError('A recovered precision/check tuple differs')
    if not controls['precision']['spectral_inflation']['inertia_resolved']:
        raise RuntimeError('The accepted stored-h inertia is unresolved; no readout allocation permitted')
    binding = dict(method=READOUT_VERSION,source_kind=KIND,input=source.metadata,
        accepted_spectral_progress=progress_id,accepted_origin_control=origin_id,
        spectral_output_roles={name:stage['outputs'] for name,stage in progress['stages'].items()},
        origin_raw_arrays=origin['raw_scalar_arrays'],verified_inputs=dict(source.inputs.current),
        physical_rows_released=False,guard_reporting_weight=48,
        missing_held_comparators=dict(extrema=None,first_current=None,first_flow=None))
    source.check()
    return ReadoutInput(source,origin,progress,progress_id,controls['precision'],
                        controls['checks'],controls['initial'],binding)


class ReadoutStages:
    """Closed per-consumer products; an uncertain operation is never repeated."""
    def __init__(self,context,root,inventory,kind,resume,extra_binding=None):
        self.context,self.root,self.inventory = context,Path(root).resolve(),inventory
        self.root.relative_to(Path('<internal>/second_route_bulk'))
        spectral_root = Path(context.progress_identity['path']).parent
        if self.root == spectral_root or spectral_root in self.root.parents:
            raise ValueError('Readout products must be outside the closed spectral directory')
        self.root.mkdir(parents=True,exist_ok=True)
        self.path = self.root/(kind+'_progress.json')
        self.binding = dict(common=context.binding,consumer=kind,extra=extra_binding)
        self.digest = binding_hash(self.binding)
        if resume:
            self.report = json.loads(self.path.read_text())
            if (self.report['input_binding_hash'] != self.digest or self.report['completed']
                    or self.report['consumer'] != kind):
                raise ValueError('The saved readout result is complete or has another input/code identity')
            self.validate()
        else:
            if any(self.root.iterdir()):
                raise FileExistsError('Existing readout products require explicit compatible resume')
            self.report = dict(method=READOUT_VERSION,consumer=kind,input=context.source.metadata,
                input_binding=self.binding,input_binding_hash=self.digest,completed=False,
                stages={},physical_rows_released=False)
            write_json(self.path,self.report)

    def validate(self):
        stages = self.report['stages']
        if list(stages) != list(self.inventory)[:len(stages)]:
            raise ValueError('Readout stages are not an exact completed prefix')
        allowed = {self.path.name}
        for name,stage in stages.items():
            if stage['status'] != 'completed' or set(stage['outputs']) != set(self.inventory[name]):
                raise RuntimeError('A failed, uncertain or incompatible readout stage needs assessment')
            for filename,saved in stage['outputs'].items():
                if Path(saved['path']) != self.root/filename:
                    raise ValueError('A readout output belongs to another directory')
                self.context.source.inputs.verify(saved)
                allowed.add(filename)
        if {p.name for p in self.root.iterdir()} != allowed:
            raise FileExistsError('An orphan readout final, partial or temporary product remains')

    def execute(self,name,operation):
        if name in self.report['stages']:
            self.validate()
            return self.report['stages'][name]['values']
        self.validate()
        self.context.check()
        resource_check(16384,dense=True)
        if shutil.disk_usage(self.root).free < 32*2**30:
            raise RuntimeError('Insufficient bulk margin for the next required product')
        binding = dict(common=self.binding,stage=name,prior_completed_products={
            key:value['outputs'] for key,value in self.report['stages'].items()})
        stage = dict(status='started',prospective_binding=binding)
        self.report['stages'][name] = stage
        write_json(self.path,self.report)
        start = time.monotonic()
        try:
            values,outputs = operation(binding)
            if set(outputs) != set(self.inventory[name]):
                raise ValueError('The required stage output roles differ')
            self.context.check()
            for saved in outputs.values():
                self.context.source.inputs.verify(saved)
            stage.update(status='completed',values=values,outputs=outputs,elapsed_seconds=time.monotonic()-start)
            write_json(self.path,self.report)
            return values
        except BaseException as error:
            stage.update(status='failed_or_uncertain',exception_type=type(error).__name__,
                         stop_reason=str(error),elapsed_seconds=time.monotonic()-start)
            write_json(self.path,self.report)
            raise

    def output(self,stage,filename):
        saved = self.report['stages'][stage]['outputs'][filename]
        self.context.source.inputs.verify(saved)
        return Path(saved['path'])

    def finish(self):
        self.validate()
        if list(self.report['stages']) != list(self.inventory):
            raise ValueError('The required readout stages are incomplete')
        self.context.check()
        self.report['completed'] = True
        write_json(self.path,self.report)


def save_arrays(path,binding,**arrays):
    return save_npz(Path(path),binding,**arrays)


def save_json(path,value):
    return save_product_json(Path(path),value)
