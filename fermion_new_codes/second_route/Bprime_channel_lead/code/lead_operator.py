"""Completed native-ball/collar/explicit-lead Hamiltonian, including its complement."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from lead_channels import FoldedLead, chain_labels, valley_permutation, antiunitary_permutation
from lead_native import NativeLayout, canonical_identity


def cross_gram(left, right=None):
    left = np.asarray(left).reshape(len(left), -1)
    right = left if right is None else np.asarray(right).reshape(len(right), -1)
    return left.conj()@right.T


def norm_upper(matrix):
    """Two inexpensive conservative upper bounds on the induced norm."""
    matrix = np.asarray(matrix)
    return float(min(np.linalg.norm(matrix), np.sqrt(np.linalg.norm(matrix, 1)*np.linalg.norm(matrix, np.inf))))


def shell_projection(native_flat, dofs, E):
    # Avoid materializing a many-GB complex conjugate of E.
    return (np.asarray(native_flat[:, dofs]).conj()@E).conj()


class HybridLayout(NativeLayout):
    def __init__(self, n, chains, cells, first_odd=41):
        super().__init__(n)
        self.chains, self.cells, self.first_odd = int(chains), int(cells), int(first_odd)
        self.lead_size = 4*self.chains*self.cells
        self.total_dofs = self.native_size+self.lead_size

    def native_view(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.total_dofs:
            raise ValueError('Wrong hybrid state dimensions')
        return states[:, :self.native_size].reshape((len(states),)+self.cube_shape)

    def lead_view(self, states):
        states = np.asarray(states)
        if states.ndim != 2 or states.shape[1] != self.total_dofs:
            raise ValueError('Wrong hybrid state dimensions')
        return states[:, self.native_size:].reshape(len(states), self.chains, self.cells, 4)

    def pack(self, native, lead=None):
        native = np.asarray(native, dtype=np.complex128)
        if native.ndim != 5 or native.shape[1:] != self.cube_shape:
            raise ValueError('Wrong hybrid native cube input')
        result = np.zeros((len(native), self.total_dofs), dtype=np.complex128)
        result[:, :self.native_size] = native.reshape(len(native), -1)
        if lead is not None:
            lead = np.asarray(lead, dtype=np.complex128)
            if lead.shape != (len(native), self.chains, self.cells, 4):
                raise ValueError('Wrong folded lead input')
            result[:, self.native_size:] = lead.reshape(len(native), -1)
        return result

    def metadata(self):
        return {**super().metadata(), 'lead_shape': [self.chains, self.cells, 4],
                'first_odd_radius': self.first_odd,
                'folded_order': ['plus_spin0', 'plus_spin1', 'minus_spin0', 'minus_spin1']}


class HybridOperator:
    """Internal apply=-iH; all native shell P/Q degrees of freedom propagate."""
    def __init__(self, native, setup_artifact, radial_N, radial_v, end_radius):
        if isinstance(setup_artifact, (str, Path)):
            from lead_setup import load_setup
            setup_artifact = load_setup(setup_artifact)
        self.native, self.setup = native, setup_artifact
        expected_inputs = setup_artifact.get('report', {}).get('operator_input_fingerprints')
        if expected_inputs is not None:
            actual_inputs = {'native_N': canonical_identity(native.lapse), 'native_v': canonical_identity(native.flow),
                'native_mask': canonical_identity(native.mask), 'radial_collar_N': canonical_identity(np.asarray(radial_N)[:42]),
                'radial_collar_v': canonical_identity(np.asarray(radial_v)[:42]), 'mass': native.mass, 'spacing': native.spacing}
            if actual_inputs != expected_inputs:
                raise ValueError('Native or radial collar fields differ from the constructed field-dependent blocks')
        self.K = int(setup_artifact['K'])
        self.E, self.shell_dofs = tuple(setup_artifact['E']), tuple(setup_artifact['shell_dofs'])
        self.corrections = setup_artifact['corrections']
        self.Dg = setup_artifact['Dg']
        self.port = np.asarray(setup_artifact['port'])
        self.lead = FoldedLead(self.K, radial_N, radial_v, end=end_radius)
        self.labels = chain_labels(self.K)
        width = 4*len(self.labels)
        if len(self.E) != 4 or len(self.shell_dofs) != 4 or self.port.shape != (width, width):
            raise ValueError('Hybrid requires exactly four compatible shell frames and the complete port')
        joined = np.concatenate(self.shell_dofs)
        if len(np.unique(joined)) != len(joined):
            raise ValueError('Native shell ownership is not disjoint')
        for frame, dofs in zip(self.E, self.shell_dofs):
            if frame.dtype != np.complex128 or frame.shape != (len(dofs), width) or not np.all(np.isfinite(frame)):
                raise ValueError('Malformed completed shell frame')
        self.layout = HybridLayout(native.shape[0], len(self.labels), self.lead.cells)
        self.end_collar_indices = self.layout.native_size+(np.arange(len(self.labels))[:, None]*self.lead.cells*4+(self.lead.cells-1)*4+np.arange(4)[None]).reshape(-1)
        self.is_unitary = True
        self.port_blocks = np.asarray([self.port[4*j:4*j+4, 4*j:4*j+4] for j in range(len(self.labels))])
        residue = self.port.copy()
        for j in range(len(self.labels)):
            residue[4*j:4*j+4, 4*j:4*j+4] = 0
        if np.any(residue != 0):
            raise ValueError('Prescribed ideal radial port unexpectedly mixes chain labels')
        e_bounds = np.asarray(setup_artifact['E_norm_bounds'], float)
        if e_bounds.shape != (4,) or np.any(e_bounds < 1) or not np.all(np.isfinite(e_bounds)):
            raise ValueError('Missing conservative frame norm bounds')
        correction_rows = np.zeros(4)
        self.block_bound_records = {}
        for (c, d), delta in self.corrections.items():
            if not (0 <= c <= d < 4 and d-c <= 1) or delta.shape != (width, width) or not np.all(np.isfinite(delta)):
                raise ValueError('Invalid collar correction block')
            bound = e_bounds[c]*norm_upper(delta)*e_bounds[d]
            correction_rows[c] += bound
            if c != d:
                correction_rows[d] += bound
            self.block_bound_records[str(c)+'_'+str(d)] = bound
        ball_bound = native.B+float(np.max(correction_rows))
        port_bound = e_bounds[3]*max(float(np.linalg.norm(block, 2)) for block in self.port_blocks)
        self.B = float((ball_bound+self.lead.B+np.hypot(ball_bound-self.lead.B, 2*port_bound))/2)
        self.bound_record = {'native': native.B, 'corrected_ball': ball_bound, 'lead': self.lead.B,
                             'port': port_bound, 'completed_operator': self.B,
                             'collar_block_upper_bounds': self.block_bound_records}
        self._core_dofs = np.setdiff1d(np.flatnonzero(np.tile(native.mask.ravel(), 4)), joined)

    def apply(self, states, out=None, timings=None):
        states = np.asarray(states)
        if states.dtype != np.complex128 or states.ndim != 2 or states.shape[1] != self.layout.total_dofs or not states.flags.c_contiguous:
            raise ValueError('Hybrid generator needs contiguous flat complex128 columns')
        out = np.empty_like(states) if out is None else out
        if out.shape != states.shape or out.dtype != np.complex128 or not out.flags.c_contiguous or np.shares_memory(states, out):
            raise ValueError('Hybrid output must be a separate contiguous complex128 block')
        size = self.layout.native_size
        native, native_out = states[:, :size], out[:, :size]
        lead, lead_out = self.layout.lead_view(states), self.layout.lead_view(out)
        if timings is not None:
            started = time.perf_counter()
        self.native.apply(native, out=native_out)
        if timings is not None:
            timings['native_generator'] = time.perf_counter()-started
            started = time.perf_counter()
        self.lead.apply(lead, out=lead_out)
        if timings is not None:
            timings['lead_generator'] = time.perf_counter()-started
            started = time.perf_counter()
        coordinates = [shell_projection(native, dofs, E) for dofs, E in zip(self.shell_dofs, self.E)]
        if timings is not None:
            timings['shell_projection'] = time.perf_counter()-started
            started = time.perf_counter()
        forces = [np.zeros_like(value) for value in coordinates]
        for (c, d), delta in self.corrections.items():
            forces[c] += coordinates[d]@delta.T
            if c != d:
                forces[d] += (coordinates[c].conj()@delta).conj()
        if timings is not None:
            timings['collar_multiply'] = time.perf_counter()-started
            started = time.perf_counter()
        incoming = np.einsum('nij,cnj->cni', self.port_blocks, lead[:, :, 0])
        forces[3] += incoming.reshape(len(states), -1)
        lead_out[:, :, 0] += -1j*np.einsum('nji,cnj->cni', self.port_blocks.conj(), coordinates[3].reshape(len(states), -1, 4))
        if timings is not None:
            timings['port'] = time.perf_counter()-started
            started = time.perf_counter()
        for c, (dofs, E) in enumerate(zip(self.shell_dofs, self.E)):
            native_out[:, dofs] += -1j*(forces[c]@E.T)
        if timings is not None:
            timings['shell_reconstruction'] = time.perf_counter()-started
        return out

    def h_apply(self, states):
        return 1j*self.apply(states)

    def diagnostics(self, states, hstates=None):
        """Saved-clock amplitudes/current; these projections add no extra inventory."""
        size = self.layout.native_size
        native = states[:, :size]
        hstates = self.h_apply(states) if hstates is None else hstates
        coords, q, qh = [], [], []
        selected_native = np.zeros_like(native)
        selected_native[:, self._core_dofs] = native[:, self._core_dofs]
        for dofs, E in zip(self.shell_dofs, self.E):
            value = shell_projection(native, dofs, E)
            selected = value@E.T
            coords.append(value)
            q.append(native[:, dofs]-selected)
            selected_native[:, dofs] = selected
            h_selected = shell_projection(hstates[:, :size], dofs, E)@E.T
            qh.append(hstates[:, dofs]-h_selected)
        native_force = self.native.h_apply(selected_native)
        perpendicular = []
        for dofs, E in zip(self.shell_dofs, self.E):
            perpendicular.append(native_force[:, dofs]-shell_projection(native_force, dofs, E)@E.T)
        q3_current = 1j*(cross_gram(qh[3], q[3])-cross_gram(q[3], qh[3]))
        lead = self.layout.lead_view(states)
        total_lead = lead.reshape(len(states), -1)
        port_force = np.einsum('nij,cnj->cni', self.port_blocks, lead[:, :, 0]).reshape(len(states), -1)
        # Positive is inventory entering the native ball from the explicit lead.
        ball_current = -1j*(cross_gram(coords[3], port_force)-cross_gram(port_force, coords[3]))
        return {'shell_coordinates': np.asarray(coords), 'complement_amplitudes': q,
                'selected_to_complement_native_force': perpendicular,
                'outer_complement_Gram': cross_gram(q[3]), 'outer_complement_current': q3_current,
                'native_Gram': cross_gram(native), 'lead_Gram': cross_gram(total_lead),
                'core_Gram': cross_gram(native[:, self._core_dofs]),
                'shell_Grams': np.asarray([cross_gram(native[:, dofs]) for dofs in self.shell_dofs]),
                'signed_port_current_into_ball': ball_current,
                'interpretation': 'Outer Q3 current is exchange with the retained unled complement, not an inferred reflection coefficient.'}

    def valley_image(self, states, mu):
        result = np.empty_like(states)
        size = self.layout.native_size
        result[:, :size] = self.native.valley_image(states[:, :size], mu)
        target, phases = valley_permutation(self.K, mu)
        raw, image = self.layout.lead_view(states), self.layout.lead_view(result)
        # R acts identically on every folded cell; target includes sign/spin.
        for source, dest in enumerate(target):
            image[:, dest//4, :, dest % 4] = phases[source]*raw[:, source//4, :, source % 4]
        return result

    def antiunitary_image(self, states):
        """Candidate induced folded map; its use requires the saved full audit."""
        result = np.empty_like(states)
        size = self.layout.native_size
        result[:, :size] = self.native.antiunitary_image(states[:, :size])
        target, phases = antiunitary_permutation(self.K)
        raw, image = self.layout.lead_view(states), self.layout.lead_view(result)
        for source, dest in enumerate(target):
            image[:, dest//4, :, dest % 4] = phases[source]*raw[:, source//4, :, source % 4].conj()
        return result


def projector_current(operator, states, mask, energy=False):
    """i[H,P] or i[H^2,P]/2, retaining all crossing-link contributions."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (states.shape[1],):
        raise ValueError('Orthogonal spatial projector needs a Boolean coordinate mask')
    selected = states*mask[None]
    applied = operator.h_apply(states)
    if energy:
        applied = operator.h_apply(applied)
    return (0.5 if energy else 1.)*1j*(cross_gram(applied, selected)-cross_gram(selected, applied))
