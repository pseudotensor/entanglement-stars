#!/usr/bin/env python3
"""XN1--XN16: bounded original-A zero-coupling native Gaussian control.

The four complete covariance matrices evolve; no moving-pair truncation,
sea reset, vacuum subtraction, parameter fit, or full-Fock matrix is used.
Binary64 residuals and fixed-step comparisons are numerical diagnostics,
not a root or trajectory enclosure.  Run only in the remote qi2 process.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

SPEC = "solve/strand_X_nonlinear_response.tex XN1--XN16"
REVIEW = "solve/review_X_nonlinear_response.tex"
MASS2 = (Fraction(1,12), Fraction(1,6), Fraction(1,6), Fraction(1,4))
ETA = (1,-1,-1,1)
LOADS = (Fraction(1,256), Fraction(1,16), Fraction(1))
STEPS = (Fraction(1,8), Fraction(1,16), Fraction(1,32))
POOLS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class Unresolved(RuntimeError):
    pass


class BranchStop(Unresolved):
    pass


class Cap(Unresolved):
    pass


def host(value):
    return cp.asnumpy(value) if cp is not None and isinstance(value, cp.ndarray) else np.asarray(value)


def dagger(value):
    return value.swapaxes(-1,-2).conj()


def hermitian(value):
    return (value+dagger(value))/2


def norm(value):
    return float(xp.linalg.norm(value).item())


def real(value):
    return float(value.real.item())


def site_diagonal(matrix):
    return xp.diagonal(matrix, axis1=-2, axis2=-1).real.reshape(matrix.shape[:-2]+(-1,2)).sum(axis=-1)


def spectral_diagonal(vectors, matrix):
    return xp.sum((vectors @ matrix)*vectors.conj(), axis=-1).real


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def number(value):
    if isinstance(value, dict):
        return {str(k):number(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [number(v) for v in value]
    if isinstance(value, np.ndarray) or (cp is not None and isinstance(value, cp.ndarray)):
        if xp.iscomplexobj(value):
            return {"real":host(value.real).tolist(), "imag":host(value.imag).tolist()}
        return host(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Work:
    def __init__(self, output, prior=None, pause_seconds=0):
        self.start = time.perf_counter()-(prior['resources']['elapsed_seconds']+pause_seconds if prior else 0)
        self.output = output
        self.fields = prior['resources']['requested_field_source_evaluations'] if prior else 0
        self.accepted_steps = prior['resources']['accepted_time_steps'] if prior else 0
        self.scalar_evaluations = prior['resources']['scalar_integrand_evaluations'] if prior else 0
        self.peak_host = prior['resources']['peak_host_RSS_bytes'] if prior else 0
        self.peak_device = prior['resources']['peak_own_device_pool_bytes'] if prior else 0
        self.operations = list(prior['field_work_records']) if prior else []
        self.phase = "initialization"

    def check(self):
        if time.perf_counter()-self.start >= 6*3600:
            raise Cap("The whole-batch six-hour wall cap was reached")
        if len(os.listdir('/proc/self/task')) > 16 or os.nice(0) < 10:
            raise Cap("The own process left its sixteen-thread/nice-10 contract")
        with open('/proc/self/status') as stream:
            rss = next(int(v.split()[1])*1024 for v in stream if v.startswith('VmRSS:'))
        self.peak_host = max(self.peak_host,rss)
        if rss > 32*2**30:
            raise Cap("The whole-batch 32-GiB host memory cap was reached")
        if cp is not None and xp is cp:
            used = cp.get_default_memory_pool().total_bytes()
            self.peak_device = max(self.peak_device,used)
            if used > 24*2**30:
                raise Cap("The whole-batch 24-GiB own device-pool cap was reached")

    def begin(self, label):
        self.fields += 1
        record = {"requested_field_evaluation":self.fields, "phase":label,
                  "elapsed_start":time.perf_counter()-self.start, "status":"failed_or_incomplete"}
        self.operations.append(record)
        self.phase = label
        self.check()
        if self.fields > 16384:
            raise Cap("The 16384 complete field/source evaluation cap was reached")
        return record

    def finish(self, record):
        self.check()
        record.update(status="completed", elapsed_end=time.perf_counter()-self.start)

    def resources(self):
        return {"elapsed_seconds":time.perf_counter()-self.start,
                "requested_field_source_evaluations":self.fields,
                "completed_field_source_evaluations":sum(v['status']=='completed' for v in self.operations),
                "failed_or_incomplete_field_source_evaluations":sum(v['status']!='completed' for v in self.operations),
                "scalar_integrand_evaluations":self.scalar_evaluations,
                "accepted_time_steps":self.accepted_steps, "peak_host_RSS_bytes":self.peak_host,
                "peak_own_device_pool_bytes":self.peak_device, "own_threads":len(os.listdir('/proc/self/task'))}


class Parent:
    def __init__(self, length, work):
        self.L,self.work = length,work
        self.coords = np.array(list(itertools.product(range(-(length//2),length//2+1),repeat=3)),float)
        self.n = len(self.coords)
        self.d = 2*self.n
        self.interior = np.flatnonzero(np.max(np.abs(self.coords),axis=1)<length//2)
        self.collar = np.flatnonzero(np.max(np.abs(self.coords),axis=1)==length//2)
        pauli = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],complex)
        idx = {tuple(x):i for i,x in enumerate(self.coords)}
        self.edges = []
        dd = np.zeros((self.d,self.d),complex)
        lap = np.zeros((self.n,self.n))
        for row,pos in enumerate(self.coords):
            for axis in range(3):
                point = pos.copy(); point[axis] += 1
                col = idx.get(tuple(point))
                if col is None:
                    continue
                self.edges.append((row,col))
                block = pauli[axis]/(2j)
                dd[2*row:2*row+2,2*col:2*col+2] = block
                dd[2*col:2*col+2,2*row:2*row+2] = block.conj().T
                lap[row,row] += 1; lap[col,col] += 1
                lap[row,col] -= 1; lap[col,row] -= 1
        self.D,self.lap = xp.asarray(dd),lap
        self.eye = xp.eye(self.d,dtype=complex)
        self.r2 = np.sum(self.coords**2,axis=1)
        self.r2_carrier = xp.asarray(np.tile(np.repeat(self.r2,2),2))
        self.sigma0 = None
        self.flat_covariances = self.pg = None

    def guards(self, psi, lapse=None, sigma=None):
        if not np.all(np.isfinite(psi)) or np.min(psi[self.interior]) <= .5 or np.max(psi[self.interior]) >= 4:
            raise BranchStop("The ruler iterate left the declared open (1/2,4) chart")
        if not np.array_equal(psi[self.collar],np.ones(len(self.collar))):
            raise RuntimeError("The fixed collar ruler was changed")
        if lapse is not None:
            if not np.all(np.isfinite(lapse)):
                raise BranchStop("A solved lapse is nonfinite")
            if np.min(lapse[self.interior]) <= .125:
                raise BranchStop("The predeclared lower diagnostic lapse guard min N_I <=1/8 was triggered")
            if np.max(lapse[self.interior]) >= 8:
                raise BranchStop("The predeclared upper diagnostic lapse guard max N_I >=8 was triggered")
            if not np.array_equal(lapse[self.collar],np.ones(len(self.collar))):
                raise RuntimeError("The fixed collar lapse was changed")
        if sigma is not None and self.sigma0 is not None and sigma/self.sigma0 <= 1e-4:
            raise BranchStop("The clock Jacobian singular-value ratio reached 1e-4")

    def factors(self, psi):
        factors = {}
        scalar = xp.asarray(np.repeat(psi,2))
        for squared in dict.fromkeys(MASS2):
            self.work.check()
            mass = float(np.sqrt(float(squared)))
            aa = mass*xp.diag(scalar**-1)+1j*scalar[:,None]**-3*self.D
            p,s,qh = la.svd(host(aa),full_matrices=False,check_finite=True,lapack_driver='gesvd')
            p,q,s = xp.asarray(p[:,::-1]),xp.asarray(qh.conj().T[:,::-1]),xp.asarray(s[::-1])
            if float(s[0].item()) <= 0 or not bool(xp.all(xp.isfinite(s))):
                raise BranchStop("A full native polar factor lost its positive gap")
            tt = -mass*xp.diag(scalar**-2)-3j*scalar[:,None]**-4*self.D
            factors[squared] = {"P":p,"Q":q,"s":s,"sqrt_s":xp.sqrt(s),"TQ":tt@q,
                                "A":aa,"mass":mass,"scalar":scalar}
        return factors

    def derivative_batches(self, fac):
        p,q,s = fac['P'],fac['Q'],fac['s']
        denom = s[:,None]+s[None,:]
        root_denom = fac['sqrt_s'][:,None]+fac['sqrt_s'][None,:]
        for begin in range(0,len(self.interior),8):
            self.work.check()
            sites = self.interior[begin:begin+8]
            rows = (2*sites[:,None]+np.arange(2)[None,:]).reshape(-1)
            pr = p[rows,:].reshape(len(sites),2,self.d)
            tr = fac['TQ'][rows,:].reshape(len(sites),2,self.d)
            ff = xp.einsum('tai,taj->tij',pr.conj(),tr)
            d0 = (s[:,None]*ff+s[None,:]*dagger(ff))/denom
            d1 = (s[None,:]*ff+s[:,None]*dagger(ff))/denom
            omega = (ff-dagger(ff))/denom
            yield begin,sites,d0,d1,omega,root_denom

    def packet(self, fac, orientation, covariance, psi, derivatives=True):
        p,q,s,root = fac['P'],fac['Q'],fac['s'],fac['sqrt_s']
        rr,ll = (q,p) if orientation == 1 else (p,q)
        vv = ll@dagger(rr)
        xx = (rr*root)@dagger(rr)
        ee = (rr*s)@dagger(rr)
        hh = dagger(rr)@covariance[:self.d,self.d:]@ll
        gg = self.eye+hh+dagger(hh)
        ss = (root[:,None]*gg)*root[None,:]
        base = spectral_diagonal(rr,ss).reshape(self.n,2).sum(axis=1)
        e = xp.asarray(psi)*base
        je = xp.zeros((self.n,len(self.interior)))
        if derivatives:
            for begin,sites,d0,d1,om,rd in self.derivative_batches(fac):
                de = d0 if orientation == 1 else d1
                oo = om if orientation == 1 else -om
                dx = de/rd
                temp = (dx@gg)*root[None,None,:]+(root[None,:,None]*(hh@oo))*root[None,None,:]
                ds = spectral_diagonal(rr,temp+dagger(temp)).reshape(len(sites),self.n,2).sum(axis=2)
                je[:,begin:begin+len(sites)] = (ds*xp.asarray(psi)[None,:]).T
                je[sites,np.arange(begin,begin+len(sites))] += base[sites]
        return {"fac":fac,"orientation":orientation,"R":rr,"L":ll,"V":vv,"X":xx,"E":ee,
                "clock":e,"J":je,"positive_band_source_matrix":gg}

    def assemble(self, covariances, psi, label):
        record = self.work.begin(label)
        try:
            record.update(minimum_ruler=float(np.min(psi)) if np.all(np.isfinite(psi)) else None,
                          maximum_ruler=float(np.max(psi)) if np.all(np.isfinite(psi)) else None)
            self.guards(psi)
            facs = self.factors(psi)
            vacuum_base = xp.zeros(self.n)
            jvac = xp.zeros((self.n,len(self.interior)))
            packets = []
            for squared,orientation,covariance in zip(MASS2,ETA,covariances):
                fac = facs[squared]
                e0 = (fac['Q']*fac['s'])@dagger(fac['Q'])
                e1 = (fac['P']*fac['s'])@dagger(fac['P'])
                vb = orientation*site_diagonal(e1-e0)
                vacuum_base += vb
                for begin,sites,d0,d1,om,rd in self.derivative_batches(fac):
                    variation = (spectral_diagonal(fac['P'],d1)-spectral_diagonal(fac['Q'],d0))
                    variation = variation.reshape(len(sites),self.n,2).sum(axis=2)
                    jvac[:,begin:begin+len(sites)] += orientation*(variation*xp.asarray(psi)[None,:]).T
                    jvac[sites,np.arange(begin,begin+len(sites))] += vb[sites]
                packets.append(self.packet(fac,orientation,covariance,psi))
            vacuum = xp.asarray(psi)*vacuum_base
            source = vacuum+sum(v['clock'] for v in packets)
            jac = jvac+sum(v['J'] for v in packets)
            ci,jh = host(source),host(jac)
            ji = jh[self.interior,:]
            singular = la.svdvals(ji,check_finite=True)
            sigma = float(singular[-1])
            record.update(sigma_min=sigma,clock_residual=float(np.max(np.abs(ci[self.interior]))))
            if sigma <= 0 or not np.isfinite(sigma):
                raise BranchStop("The actual clock Jacobian is singular/nonfinite")
            self.guards(psi,sigma=sigma)
            lapse = np.ones(self.n)
            lapse[self.interior] = la.solve(ji.T,-jh[self.collar,:].sum(axis=0),check_finite=True)
            r1=jh.sum(axis=0)
            transpose_identity=ji.T@(lapse[self.interior]-1)+r1
            beta=float(np.linalg.norm(r1))/sigma
            record.update(existing_J_transpose_identity_residual_2=float(np.linalg.norm(transpose_identity)),
                          existing_J_beta_numerical=beta,
                          beta_margin_to_seven_eighths=.875-beta,
                          lapse_deviation_inf=float(np.max(np.abs(lapse[self.interior]-1))),
                          beta_is_not_a_directed_inverse_or_arithmetic_certificate=True)
            record.update(minimum_interior_lapse=float(np.min(lapse[self.interior])),
                          maximum_interior_lapse=float(np.max(lapse[self.interior])),
                          minimum_interior_lapse_site=int(self.interior[np.argmin(lapse[self.interior])]),
                          maximum_interior_lapse_site=int(self.interior[np.argmax(lapse[self.interior])]),
                          lower_lapse_guard_sites=self.interior[lapse[self.interior]<=.125].tolist(),
                          upper_lapse_guard_sites=self.interior[lapse[self.interior]>=8].tolist(),
                          guard_scope="prescribed numerical/regularity chart; no physical exclusion")
            self.guards(psi,lapse,sigma)
            for packet in packets:
                xx,vv = packet['X'],packet['V']
                bn = hermitian((xx*xp.asarray(np.repeat(lapse*psi,2))[None,:])@xx)
                packet['B'] = bn
                packet['h'] = xp.block([[xp.zeros_like(bn),bn@dagger(vv)],[vv@bn,xp.zeros_like(bn)]])
            answer = {"psi":psi.copy(),"N":lapse,"C":ci,"J":jh,"vacuum":host(vacuum),
                      "vacuum_J":host(jvac),"packets":packets,"sigma_min":sigma,
                      "inverse_norm_2":1/sigma,"clock_residual":float(np.max(np.abs(ci[self.interior]))),
                      "ruler_residual":float(np.max(np.abs(jh.T@lapse))),"covariances":covariances}
            if not np.all(np.isfinite(ci)) or not np.all(np.isfinite(jh)):
                raise BranchStop("A complete native source or derivative became nonfinite")
            record.update(clock_residual=answer['clock_residual'],ruler_residual=answer['ruler_residual'],
                          sigma_min=sigma,minimum_ruler=float(np.min(psi)),minimum_lapse=float(np.min(lapse)))
            self.work.finish(record)
            return answer
        except BaseException as error:
            record['failure'] = {"class":type(error).__name__,"message":str(error)}
            raise

    def solve(self, covariances, seed, label, budget=None):
        psi = seed.copy()
        budget = IterationBudget(12,label+" inner field solve") if budget is None else budget
        budget.tick()
        current = self.assemble(covariances,psi,label+" initial field trial")
        for iteration in range(12):
            # The request is tighter than the unchanged 1e-10 acceptance gate.
            if current['clock_residual'] <= 1e-12 and current['ruler_residual'] <= 1e-10:
                current['inner_field_trials'] = budget.count
                return current
            direction = la.solve(current['J'][self.interior],-current['C'][self.interior])
            if not np.all(np.isfinite(direction)):
                raise BranchStop("A field Newton direction is nonfinite")
            accepted = False
            for backtrack in range(8):
                budget.tick()
                candidate = psi.copy()
                candidate[self.interior] += 2.0**(-backtrack)*direction
                try:
                    trial = self.assemble(covariances,candidate,label+" backtrack "+str(backtrack))
                except BranchStop:
                    continue
                if trial['clock_residual'] < current['clock_residual']:
                    psi,current,accepted = candidate,trial,True
                    if trial['clock_residual'] <= 1e-12 and trial['ruler_residual'] <= 1e-10:
                        trial['inner_field_trials'] = budget.count
                        return trial
                    break
            if not accepted:
                raise BranchStop("The field corrector found no guarded residual-decreasing Newton iterate")
        raise Unresolved("The bounded inner field corrector exhausted its twelve-trial allowance")

    def preparation(self):
        psi = np.ones(self.n)
        facs = self.factors(psi)
        covariances = []
        for squared,orientation in zip(MASS2,ETA):
            fac = facs[squared]
            rr,ll = (fac['Q'],fac['P']) if orientation == 1 else (fac['P'],fac['Q'])
            vv = ll@dagger(rr)
            covariances.append(xp.block([[self.eye/2,-dagger(vv)/2],[-vv/2,self.eye/2]]))
        r = np.sqrt(self.r2)
        envelope = np.zeros(self.n,complex)
        mask = r < 7/4
        envelope[mask] = np.exp(-1/(1-r[mask]**2/(7/4)**2)-2*(r[mask]-1)**2-.25j*np.sqrt(r[mask]**2+.25))
        envelope /= np.linalg.norm(envelope)
        g = xp.asarray(np.kron(envelope[:,None],np.eye(2)))
        flat = facs[MASS2[0]]
        vv = flat['P']@dagger(flat['Q'])
        columns = xp.concatenate([g,vv@g],axis=0)/np.sqrt(2)
        self.pg = columns@dagger(columns)
        self.flat_covariances = covariances
        return covariances

    def initial_covariances(self, load):
        return [v+float(load)*self.pg if index == 0 else v.copy()
                for index,v in enumerate(self.flat_covariances)]

    def direction(self, packet, psi, profile, second=False):
        fac = packet['fac']
        pp,qq,s = fac['P'],fac['Q'],fac['s']
        f = xp.asarray(np.repeat(profile,2))
        ff = (dagger(pp)*f[None,:])@fac['TQ']
        den = s[:,None]+s[None,:]
        d0 = (s[:,None]*ff+s[None,:]*dagger(ff))/den
        d1 = (s[None,:]*ff+s[:,None]*dagger(ff))/den
        om = (ff-dagger(ff))/den
        orient = packet['orientation']
        de,oo = (d0,om) if orient == 1 else (d1,-om)
        dx = de/(fac['sqrt_s'][:,None]+fac['sqrt_s'][None,:])
        rr,ll = packet['R'],packet['L']
        dv = ll@oo@dagger(rr)
        out = {"dE":rr@de@dagger(rr),"dX":rr@dx@dagger(rr),"dV":dv,
               "K":rr@oo@dagger(rr),"d0":d0,"d1":d1}
        if second:
            scalar = fac['scalar']
            aa2 = f[:,None]**2*(2*fac['mass']*xp.diag(scalar**-3)+12j*scalar[:,None]**-5*self.D)
            a2b = dagger(pp)@aa2@qq
            d02 = (s[:,None]*a2b+s[None,:]*dagger(a2b)+2*dagger(ff)@ff-2*d0@d0)/den
            d12 = (s[None,:]*a2b+s[:,None]*dagger(a2b)+2*ff@dagger(ff)-2*d1@d1)/den
            out.update(d02=d02,d12=d12)
        return out


class IterationBudget:
    def __init__(self, limit, label):
        self.limit,self.label,self.count = limit,label,0

    def tick(self):
        if self.count >= self.limit:
            raise Unresolved(self.label+" exhausted its declared nonlinear corrector limit")
        self.count += 1


def covariance_guards(covariances, target_spectra=None):
    records = []
    for index,c in enumerate(covariances):
        absolute = norm(c-dagger(c))
        defect = absolute/max(1,norm(c))
        eigenvalues = la.eigvalsh(host(hermitian(c)),check_finite=True)
        spectral = max(0,float(-eigenvalues[0]),float(eigenvalues[-1]-1))
        retained = 0 if target_spectra is None else float(np.max(np.abs(eigenvalues-target_spectra[index])))
        row = {"hermiticity_absolute_F":absolute,"hermiticity_relative_F":defect,"spectral_0_to_1_defect":spectral,
               "spectrum_conservation_defect":retained}
        records.append(row)
        if absolute > 1e-10 or spectral > 1e-10 or retained > 1e-10:
            raise Unresolved("A complete covariance Hermiticity/spectral guard failed")
    return records


def propagate(packet, covariance, marker, dt):
    b,g = la.eigh(host(packet['B']),check_finite=True)
    if b[0] <= 0:
        raise BranchStop("The positive B_N spectrum failed")
    gg = xp.asarray(g)
    u = (gg*xp.asarray(np.exp(-1j*dt*b)))@dagger(gg)
    cosine,sine = (u+dagger(u))/2,(dagger(u)-u)/(2j)
    vv = packet['V']
    unitary = xp.block([[cosine,-1j*sine@dagger(vv)],[-1j*vv@sine,vv@cosine@dagger(vv)]])
    next_c = hermitian(unitary@covariance@dagger(unitary))
    next_m = None if marker is None else hermitian(unitary@marker@dagger(unitary))
    return next_c,next_m


def flat_controls(parent, state):
    field = parent.assemble(state,np.ones(parent.n),"L="+str(parent.L)+" flat algebraic control")
    parent.sigma0 = field['sigma_min']
    dd,z = la.eigh(host(parent.D))
    weights = sum(sign/(np.sqrt(float(m)+dd[:,None]**2)+np.sqrt(float(m)+dd[None,:]**2))
                  for m,sign in zip(MASS2,ETA))
    profiles = []
    for label,values in (("constant interior",np.ones(len(parent.interior))),
                         ("central site",(np.arange(len(parent.interior))==len(parent.interior)//2).astype(float)),
                         ("alternating interior",(-1.)**np.arange(len(parent.interior)))):
        phi = np.zeros(parent.n); phi[parent.interior] = values/np.linalg.norm(values)
        diagphi = np.repeat(phi,2)
        comm = host(parent.D)*diagphi[None,:]-diagphi[:,None]*host(parent.D)
        common = z.conj().T@comm@z
        rhs = 3*float(np.sum(weights*np.abs(common)**2))
        lhs = float(-phi@field['vacuum_J']@phi[parent.interior])
        profiles.append({"profile":label,"minus_phi_J0_phi":lhs,"spectral_commutator_form":rhs,
                         "relative_difference":abs(lhs-rhs)/max(1,abs(lhs),abs(rhs))})
        if lhs <= 0 or abs(lhs-rhs)/max(1,abs(lhs),abs(rhs)) > 1e-10:
            raise Unresolved("The finite flat Jacobian quadratic-form control failed")
    column = float(np.max(np.abs(field['J'].sum(axis=0))))
    sea_sources = [float(np.max(np.abs(host(p['clock'])))) for p in field['packets']]
    derivative_seas = [float(np.max(np.abs(host(p['J'])))) for p in field['packets']]
    if max(column,*sea_sources,*derivative_seas,field['clock_residual'],field['ruler_residual']) > 1e-10:
        raise Unresolved("The flat common-kernel/column-sum algebraic control failed")
    return field,{"quadratic_form":profiles,"column_sum_inf":column,
                  "sea_clock_inf":sea_sources,"sea_clock_J_inf":derivative_seas,"sigma_min_J0":parent.sigma0}


def charges(parent, field):
    excitation = np.stack([host(p['clock']) for p in field['packets']])
    collar = float(field['C'][parent.collar].sum())
    full = float(field['N']@field['C'])
    positive_sum = float(np.sum(excitation/field['psi'][None,:]))
    interior_correction = float(np.sum(field['C'][parent.interior]/field['psi'][parent.interior]))
    lapse_correction = float(field['N'][parent.interior]@field['C'][parent.interior])
    return {"collar_charge":collar,"full_functional":full,"positive_source_readout":positive_sum,
        "collar_minus_positive":collar-positive_sum,
        "minus_interior_clock_over_psi":-interior_correction,
        "positive_identity_error":collar-positive_sum+interior_correction,
        "full_minus_collar":full-collar,"interior_N_clock":lapse_correction,
        "functional_identity_error":full-collar-lapse_correction}


def density_readout(parent, marker, field, load):
    density = host(xp.diag(marker).real).reshape(2,parent.n,2).sum(axis=(0,2))
    total = float(density.sum())
    r = np.sqrt(parent.r2)
    unique = np.unique(parent.r2)
    shell_rows = []
    mass_so_far = 0.
    median = ninety = None
    for radius2 in unique:
        sites = np.flatnonzero(parent.r2==radius2)
        mass = float(density[sites].sum())
        mass_so_far += mass
        if median is None and mass_so_far >= float(load):
            median = float(np.sqrt(radius2))
        if ninety is None and mass_so_far >= 1.8*float(load):
            ninety = float(np.sqrt(radius2))
        vals = density[sites]
        mean,minv,maxv = float(vals.mean()),float(vals.min()),float(vals.max())
        orbits = {tuple(sorted(np.abs(parent.coords[s]).astype(int))) for s in sites}
        shell_rows.append({"r_squared":float(radius2),"site_count":len(sites),"marked_mass":mass,
            "mean":mean,"min":minv,"max":maxv,"absolute_spread":maxv-minv,
            "normalized_spread":(maxv-minv)/mean if mean>0 else None,
            "cubic_orbits":len(orbits)})
    rms2 = float(density@parent.r2)/(2*float(load))
    rms = float(np.sqrt(max(0,rms2)))
    hp = field['packets'][0]['h']
    moment_dot = real(1j*xp.trace(marker@(hp*parent.r2_carrier[None,:]-parent.r2_carrier[:,None]*hp)))
    octants = np.zeros(8)
    keys = list(itertools.product((-1,1),repeat=3))
    for pos,mass in zip(parent.coords,density):
        compatible = [index for index,key in enumerate(keys) if all(v==0 or v*k>0 for v,k in zip(pos,key))]
        for index in compatible:
            octants[index] += mass/len(compatible)
    axes = [float(density[np.all(np.delete(parent.coords,axis,axis=1)==0,axis=1)].sum()) for axis in range(3)]
    chi, speed = field['N']*field['psi'],field['N']*field['psi']**-2
    shell_fields = []
    for label,cut in (("median",median),("ninety_percent",ninety)):
        sites = np.flatnonzero(r<=cut) if cut is not None else np.array([],int)
        shell_fields.append({"material_quantile":label,"radius":cut,"interpretation":"native cubic ball",
            "chi_min":float(np.min(chi[sites])) if len(sites) else None,
            "chi_max":float(np.max(chi[sites])) if len(sites) else None,
            "principal_coefficient_min":float(np.min(speed[sites])) if len(sites) else None,
            "principal_coefficient_max":float(np.max(speed[sites])) if len(sites) else None})
    return {"marker_trace":total,"trace_error":total-2*float(load),"minimum_density":float(density.min()),
        "coordinate_median":median,"coordinate_90_percent":ninety,"rms_radius":rms,"rms_radius_squared":rms2,
        "moment_derivative":moment_dot,"squared_radius_derivative":moment_dot/(2*float(load)),
        "rms_radius_derivative":moment_dot/(4*float(load)*rms) if rms>0 else None,
        "collar_fraction":float(density[parent.collar].sum())/(2*float(load)),
        "shells":shell_rows,"octant_partition_totals":octants.tolist(),
        "axis_line_totals_overlapping_at_origin":axes,"shell_field_readouts":shell_fields,
        "marked_density":density.tolist(),"chi":chi.tolist(),"principal_coefficient":speed.tolist()}


def primary_readout(parent, field, coefficient):
    if coefficient.get('status') != 'enclosed':
        return {"status":"unresolved","reason":"The independently capped scalar Newton coefficient is unavailable"}
    cc = coefficient['module']
    kval = coefficient['k_ball']
    psi,lapse = field['psi'],field['N']
    geom_clock = -psi*(parent.lap@psi)
    geom_j = -np.diag(parent.lap@psi)[:,parent.interior]-psi[:,None]*parent.lap[:,parent.interior]
    geom_ruler = -(parent.lap@(lapse*psi)+lapse*(parent.lap@psi))[parent.interior]
    def intervals(values):
        return [cc.encode(kval*cc.arb(float(v))) for v in np.ravel(values)]
    midpoint = float(kval.mid())
    primary_w = midpoint*float(lapse@geom_clock)
    exact_w = float(lapse@field['vacuum'])
    rc = field['vacuum']-midpoint*geom_clock
    rr = field['vacuum_J'].T@lapse-midpoint*geom_ruler
    return {"status":"scalar_uncertainty_retained","W0":exact_w,"primary_W2_midpoint":primary_w,
        "primary_W2_ball":cc.encode(kval*cc.arb(float(lapse@geom_clock))),
        "remainder_W0_minus_W2_ball":cc.encode(cc.arb(exact_w)-kval*cc.arb(float(lapse@geom_clock))),
        "remainder_midpoint":exact_w-primary_w,"primary_clock_ball":intervals(geom_clock),
        "primary_ruler_ball":intervals(geom_ruler),"primary_J_shape":list(geom_j.shape),
        "Jacobian_coefficient_ball":cc.encode(kval),
        "Jacobian_formula":"primary_J=k_star*geometry; remainder_J=vacuum_J-primary_J; geometry stored in native state NPZ",
        "primary_J_midpoint":(midpoint*geom_j).tolist(),
        "remainder_clock_ball":[cc.encode(cc.arb(float(v))-kval*cc.arb(float(g))) for v,g in zip(field['vacuum'],geom_clock)],
        "remainder_ruler_ball":[cc.encode(cc.arb(float(v))-kval*cc.arb(float(g))) for v,g in zip(field['vacuum_J'].T@lapse,geom_ruler)],
        "remainder_clock_relative_F":float(np.linalg.norm(rc))/float(np.linalg.norm(midpoint*geom_clock)) if np.linalg.norm(geom_clock)>0 else None,
        "remainder_ruler_relative_F":float(np.linalg.norm(rr))/float(np.linalg.norm(midpoint*geom_ruler)) if np.linalg.norm(geom_ruler)>0 else None,
        "zero_primary_denominator_clock":bool(np.linalg.norm(geom_clock)==0),
        "zero_primary_denominator_ruler":bool(np.linalg.norm(geom_ruler)==0),
        "numeric_field_error_not_enclosed":True}


def response(parent, field):
    source_dot = xp.zeros(parent.n)
    cdots = []
    for packet,cov in zip(field['packets'],field['covariances']):
        cd = -1j*(packet['h']@cov-cov@packet['h'])
        cdots.append(cd)
        vv,xx = packet['V'],packet['X']
        change = dagger(vv)@cd[parent.d:,:parent.d]+cd[:parent.d,parent.d:]@vv
        source_dot += xp.asarray(field['psi'])*site_diagonal(xx@change@xx)
    velocity = np.zeros(parent.n)
    velocity[parent.interior] = la.solve(field['J'][parent.interior],-host(source_dot)[parent.interior])
    tangent_residual = field['J'][parent.interior]@velocity[parent.interior]+host(source_dot)[parent.interior]
    scale = float(np.linalg.norm(velocity))
    f = velocity/scale if scale>0 else np.zeros(parent.n)
    rows = []
    xis = []
    k_matrices = []
    spectra = []
    minimum_b = float('inf')
    for packet in field['packets']:
        b,g = la.eigh(host(packet['B']),check_finite=True)
        if b[0] <= 0:
            raise BranchStop("The reached frozen pair response lost its free gap")
        minimum_b = min(minimum_b,float(b[0]))
        spectra.append((b,xp.asarray(g)))
    delta = 2*minimum_b
    vacuum_stiffness = 0.
    for index,(packet,cd,(b,g)) in enumerate(zip(field['packets'],cdots,spectra)):
        direction = parent.direction(packet,field['psi'],velocity)
        kk = direction['K']; eta = -kk/2
        pair = xp.asarray(b[:,None]+b[None,:])
        etab = dagger(g)@eta@g
        xi = g@(etab/pair)@dagger(g)
        xis.append(xi); k_matrices.append(kk)
        normal = real(xp.trace(dagger(xi)@xi))
        lad = real(xp.trace(dagger(eta)@xi))
        response_rows = []
        for fraction in (0.,.125,.25,.5):
            frequency = fraction*delta
            value = 0. if scale==0 else real(2*xp.sum(pair*xp.abs(etab/scale)**2/(pair**2-frequency**2)))
            response_rows.append({"omega_over_global_pair_gap":fraction,"omega":frequency,"Z_f":value,
                                  "omega_squared_Z":frequency**2*value})
        vv=packet['V']; ww=xp.block([[parent.eye,parent.eye],[vv,-vv]])/np.sqrt(2)
        dv=direction['dV']; dw=xp.block([[xp.zeros_like(vv),xp.zeros_like(vv)],[dv,-dv]])/np.sqrt(2)
        cov=field['covariances'][index]
        pol=dagger(ww)@cov@ww
        connection=dagger(ww)@dw
        lhs=1j*(dagger(ww)@cd@ww+dagger(dw)@cov@ww+dagger(ww)@cov@dw)
        hh=xp.block([[packet['B'],xp.zeros_like(vv)],[xp.zeros_like(vv),-packet['B']]])-1j*connection
        rhs=hh@pol-pol@hh
        check=norm(lhs-rhs)/max(1,norm(lhs),norm(rhs))
        if check > 1e-10:
            raise Unresolved("The independent moving-polar connection representation check failed")
        excitation_number=real(xp.trace(pol[:parent.d,:parent.d]+parent.eye-pol[parent.d:,parent.d:]))
        coherence=norm(pol[:parent.d,parent.d:])
        rows.append({"sea":index+1,"orientation":ETA[index],"instantaneous_excitation_number":excitation_number,
            "particle_hole_coherence_F":coherence,"positive_clock_profile":host(packet['clock']).tolist(),
            "excitation_energy":float(field['N']@host(packet['clock'])),"L_ad":lad,"n_Xi":normal,
            "frozen_reference_response":response_rows,"connection_check_relative_F":check,
            "minimum_B_energy":float(b[0]),"vacuum_benchmark_not_in_medium_response":index==0})
        if scale>0:
            direction_f=parent.direction(packet,field['psi'],f,second=True)
            fac=packet['fac']
            first=site_diagonal(fac['P']@direction_f['d1']@dagger(fac['P'])-fac['Q']@direction_f['d0']@dagger(fac['Q']))
            second=site_diagonal(fac['P']@direction_f['d12']@dagger(fac['P'])-fac['Q']@direction_f['d02']@dagger(fac['Q']))
            vacuum_stiffness += ETA[index]*float(field['N']@host(2*xp.asarray(f)*first+xp.asarray(field['psi'])*second))
    return {"psi_velocity":velocity,"clock_tangency_residual_inf":float(np.max(np.abs(tangent_residual))),
            "source_state_derivative":host(source_dot),"profile_norm":scale,"profile":f,
            "Delta2":delta,"fixed_lapse_vacuum_second_variation":vacuum_stiffness if scale>0 else None,
            "zero_velocity_profile":scale==0,"seas":rows,"Xi":xis,"K":k_matrices}


def adiabatic_readout(parent, node, xi_derivatives, derivative_kind):
    field,resp = node['field'],node['response']
    record = parent.work.begin("L="+str(parent.L)+" normalized adiabatic trial source readout")
    result = []
    try:
        for index,(packet,cov,xi,kk,xdot) in enumerate(zip(field['packets'],field['covariances'],resp['Xi'],resp['K'],xi_derivatives)):
            nn=real(xp.trace(dagger(xi)@xi))
            cad_pol=xp.block([[xi@dagger(xi),1j*xi],[-1j*dagger(xi),(1+nn)*parent.eye-dagger(xi)@xi]])/(1+nn)
            vv=packet['V']; ww=xp.block([[parent.eye,parent.eye],[vv,-vv]])/np.sqrt(2)
            cad=hermitian(ww@cad_pol@dagger(ww))
            trial=parent.packet(packet['fac'],packet['orientation'],cad,field['psi'])
            trial_r=host(trial['J']).T@field['N']
            actual_r=host(packet['J']).T@field['N']
            dt_xi=xdot+(kk@xi-xi@kk)/2
            yy=dagger(packet['R'])@dt_xi@packet['R']
            acceleration=np.zeros(len(parent.interior))
            ruler_norm_bound=np.zeros(len(parent.interior))
            eb=host(packet['E'])
            bmax=float(la.eigvalsh(host(packet['B']),subset_by_index=[parent.d-1,parent.d-1])[0])
            for begin,sites,d0,d1,om,rd in parent.derivative_batches(packet['fac']):
                oo=om if packet['orientation']==1 else -om
                de=d0 if packet['orientation']==1 else d1
                dx=de/rd
                acceleration[begin:begin+len(sites)]=host(xp.einsum('tij,ji->t',oo,yy).real)
                xnorm=host(xp.sqrt(xp.sum(xp.abs(dx)**2,axis=(1,2))))
                vnorm=host(xp.sqrt(xp.sum(xp.abs(oo)**2,axis=(1,2))))
                direct=np.array([np.linalg.norm(eb[2*s:2*s+2,2*s:2*s+2])*abs(field['N'][s]) for s in sites])
                ruler_norm_bound[begin:begin+len(sites)]=2*xnorm*float(np.max(np.abs(field['N']*field['psi'])))*np.sqrt(float(packet['fac']['s'][-1].item()))+direct+bmax*vnorm
            trace_distance=float(np.sum(np.abs(la.eigvalsh(host(hermitian(cov-cad))))))
            clock_norm=np.array([field['psi'][s]*la.eigvalsh(eb[2*s:2*s+2,2*s:2*s+2])[-1] for s in range(parent.n)])
            trial_e=host(trial['clock'])
            actual_e=host(packet['clock'])
            result.append({"sea":index+1,"n_Xi":nn,"trial_clock":trial_e.tolist(),
                "actual_minus_trial_clock":(actual_e-trial_e).tolist(),"trial_ruler":trial_r.tolist(),
                "actual_minus_trial_ruler":(actual_r-trial_r).tolist(),
                "XS_clock_source":((1+nn)*trial_e).tolist(),
                "XS_ruler_source":((1+nn)*trial_r+acceleration).tolist(),
                "normalization_clock_correction":(nn*trial_e).tolist(),
                "normalization_ruler_correction":(nn*trial_r).tolist(),
                "acceleration_ruler_term":acceleration.tolist(),"covariance_trace_distance":trace_distance,
                "clock_difference_norm_bound_numerical":(clock_norm*trace_distance).tolist(),
                "ruler_difference_norm_bound_numerical":(ruler_norm_bound*trace_distance).tolist(),
                "covariant_pair_derivative_F":norm(dt_xi),"derivative_kind":derivative_kind,
                "XS_full_state_Duhamel_error_bound":None,"XS_force_residual_bound":None,
                "FD_and_float_source_estimates_are_not_an_enclosure":True,
                "initial_preparation_transient_retained":True})
        parent.work.finish(record)
        return result
    except BaseException as error:
        record['failure']={"class":type(error).__name__,"message":str(error)}
        raise


def archive_node(parent, node, directory, tag, coefficient):
    field,marker,resp = node['field'],node['marker'],node['response']
    path=directory/(tag+".npz")
    arrays={"coordinates":parent.coords,"interior":parent.interior,"collar":parent.collar,
            "psi":field['psi'],"N":field['N'],"clock":field['C'],"J":field['J'],
            "vacuum_clock":field['vacuum'],"vacuum_J":field['vacuum_J'],"marker":host(marker),
            "psi_velocity":resp['psi_velocity']}
    arrays['primary_J_geometry']=-np.diag(parent.lap@field['psi'])[:,parent.interior]-field['psi'][:,None]*parent.lap[:,parent.interior]
    for index,(cov,packet,xi) in enumerate(zip(field['covariances'],field['packets'],resp['Xi'])):
        arrays['covariance_'+str(index+1)]=host(cov)
        arrays['sea_clock_'+str(index+1)]=host(packet['clock'])
        arrays['sea_J_'+str(index+1)]=host(packet['J'])
        arrays['Xi_'+str(index+1)]=host(xi)
    parent.work.check()
    np.savez(path,**arrays)
    row=node['record']
    row.update(native_state_file=str(path), fields={"psi":field['psi'].tolist(),"N":field['N'].tolist(),
        "clock_residual":field['clock_residual'],"ruler_residual":field['ruler_residual'],
        "sigma_min":field['sigma_min'],"inverse_norm_2":field['inverse_norm_2'],
        "minimum_polar_gap":min(float(p['fac']['s'][0].item()) for p in field['packets']),
        "source_derivatives_file":str(path)}, primary=primary_readout(parent,field,coefficient),
        instantaneous_heavy_reference_approximation={
            "clock_row_residual":(field['C']-sum(host(p['clock']) for p in field['packets'][1:])).tolist(),
            "ruler_row_residual":(field['J'].T@field['N']-sum(host(p['J']).T@field['N'] for p in field['packets'][1:])).tolist(),
            "heavy_seas_replaced_only_in_this_residual_readout":True,"fields_not_resolved_or_substituted":True})
    return row


def make_node(parent, field, marker, load, t):
    resp=response(parent,field)
    charge=charges(parent,field)
    material=density_readout(parent,marker,field,load)
    if abs(material['trace_error']) > 1e-10 or material['minimum_density'] < -1e-10:
        raise Unresolved("The propagated marked covariance trace/positive-density guard failed")
    record={"T_exact":str(t),"T":float(t),"charges":charge,"material":material,
        "clock_tangency_residual_inf":resp['clock_tangency_residual_inf'],
        "psi_velocity":resp['psi_velocity'].tolist(),"Delta2":resp['Delta2'],
        "profile_norm":resp['profile_norm'],"fixed_lapse_vacuum_second_variation":resp['fixed_lapse_vacuum_second_variation'],
        "sea_readouts":resp['seas']}
    if resp['clock_tangency_residual_inf'] > 1e-10:
        raise Unresolved("The differentiated clock residual failed its numerical tolerance")
    stiff=resp['fixed_lapse_vacuum_second_variation']
    record['frozen_response_comparison']={"zero_static_denominator":stiff is None or stiff==0,
        "omega_squared_Z_over_actual_vacuum_magnitude":[
            sum(sea['frozen_reference_response'][j]['omega_squared_Z'] for sea in resp['seas'])/abs(stiff)
            if stiff is not None and stiff!=0 else None for j in range(4)],
        "frozen_component_not_full_reduced_response":True}
    return {"field":field,"marker":marker,"response":resp,"record":record,"T":t}


def add_fd_readout(parent, node, previous, following, dt, kind):
    if previous is None:
        derivatives=[(b-a)/dt for a,b in zip(node['response']['Xi'],following['response']['Xi'])]
        vdot=(following['response']['psi_velocity']-node['response']['psi_velocity'])/dt
    elif following is None:
        derivatives=[(b-a)/dt for a,b in zip(previous['response']['Xi'],node['response']['Xi'])]
        vdot=(node['response']['psi_velocity']-previous['response']['psi_velocity'])/dt
    else:
        derivatives=[(b-a)/(2*dt) for a,b in zip(previous['response']['Xi'],following['response']['Xi'])]
        vdot=(following['response']['psi_velocity']-previous['response']['psi_velocity'])/(2*dt)
    node['record']['turnover_FD']={"kind":kind,"fixed_difference_step":dt,
        "psi_acceleration_profile":vdot.tolist(),"scale":float(np.linalg.norm(vdot))/node['response']['profile_norm']
        if node['response']['profile_norm']>0 else None,"not_a_monochromatic_error_theorem":True}
    node['record']['adiabatic_comparison']=adiabatic_readout(parent,node,derivatives,kind)


def continue_loads(parent, flat, record, previous=None):
    record.update(status="partial_or_unresolved",corrector_attempts=list(previous['corrector_attempts']) if previous else [],reached_loads=[])
    reached={}
    value=Fraction(0)
    current=flat
    parent.last_preparation=(value,current)
    step=Fraction(previous['corrector_attempts'][-1]['load_step'])/2 if previous else Fraction(1,16)
    count=len(record['corrector_attempts'])
    for target in LOADS:
        while value<target:
            if count>=96:
                record['stop']="The 96 attempted load-corrector limit was reached"
                return reached
            trial_step=min(step,target-value,Fraction(1,16))
            if trial_step<Fraction(1,2**16):
                record['stop']="The load step fell below 2^-16"
                return reached
            attempted=value+trial_step
            count+=1
            row={"attempt":count,"from_load":str(value),"candidate_load":str(attempted),
                 "load_step":str(trial_step),"status":"failed_or_incomplete",
                 "field_counter_before":parent.work.fields}
            record['corrector_attempts'].append(row)
            cov=parent.initial_covariances(attempted)
            try:
                solved=parent.solve(cov,current['psi'],"L="+str(parent.L)+" load "+str(attempted))
                guards=covariance_guards(cov)
                value,current=attempted,solved
                parent.last_preparation=(value,current)
                path=parent.work.output/('L'+str(parent.L)+'_last_valid_preparation.npz')
                save_existing_state(parent,current,float(value)*parent.pg,path)
                record.update(last_valid_load_exact=str(value),last_valid_native_state_file=str(path))
                row.update(status="accepted",inner_field_trials=solved['inner_field_trials'],
                           covariance_guards=guards,clock_residual=solved['clock_residual'],ruler_residual=solved['ruler_residual'])
            except Cap:
                raise
            except (Unresolved,la.LinAlgError) as error:
                row['failure']={"class":type(error).__name__,"message":str(error)}
                step=trial_step/2
            finally:
                row['field_counter_after']=parent.work.fields
            print("X L="+str(parent.L)+" continuation attempt="+str(count)+" load="+str(attempted)+" "+row['status']+
                  " fields="+str(parent.work.fields)+" reason="+row.get('failure',{}).get('message','accepted'),flush=True)
            parent.checkpoint()
        if value==target:
            reached[target]=current
            record['reached_loads'].append(str(target))
    record['status']="all_prescribed_loads_reached"
    return reached


def midpoint_step(parent, node, load, dt, target_spectra, step_record):
    cov=node['field']['covariances']
    guess=[v.copy() for v in cov]
    seed=node['field']['psi'].copy()
    step_record.update(status="failed_or_incomplete",outer_attempts=[],field_counter_before=parent.work.fields)
    final_cov=final_marker=None
    for attempt in range(1,25):
        row={"outer_attempt":attempt,"status":"failed_or_incomplete","field_counter_before":parent.work.fields}
        step_record['outer_attempts'].append(row)
        midpoint=[hermitian((a+b)/2) for a,b in zip(cov,guess)]
        try:
            row['midpoint_covariance_guards']=covariance_guards(midpoint)
            mid=parent.solve(midpoint,seed,"L="+str(parent.L)+" midpoint outer "+str(attempt))
        except Cap:
            raise
        except BranchStop as error:
            row['failure']={"class":type(error).__name__,"message":str(error)}
            row['field_counter_after']=parent.work.fields
            raise
        except (Unresolved,la.LinAlgError) as error:
            row['failure']={"class":type(error).__name__,"message":str(error)}
            row['field_counter_after']=parent.work.fields
            # The retry keeps the same declared dt and consumes an outer attempt.
            continue
        proposed=[]
        for index,(packet,c) in enumerate(zip(mid['packets'],cov)):
            cnext,mnext=propagate(packet,c,node['marker'] if index==0 else None,float(dt))
            proposed.append(cnext)
            if index==0:
                final_marker=mnext
        row['candidate_covariance_guards']=covariance_guards(proposed,target_spectra)
        difference=np.sqrt(sum(norm(a-b)**2 for a,b in zip(proposed,guess)))
        denominator=max(1,np.sqrt(sum(norm(c)**2 for c in proposed)))
        update=float(difference/denominator)
        row.update(status="valid_midpoint_iterate",inner_field_trials=mid['inner_field_trials'],
            update_residual_relative_F=update,midpoint_clock_residual=mid['clock_residual'],
            midpoint_ruler_residual=mid['ruler_residual'],field_counter_after=parent.work.fields)
        guess,seed=proposed,mid['psi'].copy()
        if update<=1e-11:
            final_cov=proposed
            break
    if final_cov is None:
        raise Unresolved("The fixed-step covariance midpoint exhausted 24 outer attempts")
    try:
        endpoint=parent.solve(final_cov,seed,"L="+str(parent.L)+" endpoint reconstruction")
    except BaseException as error:
        step_record['endpoint_failure']={"class":type(error).__name__,"message":str(error)}
        raise
    if endpoint['clock_residual']>1e-10 or endpoint['ruler_residual']>1e-10:
        raise Unresolved("The reconstructed endpoint field residual failed")
    covariance_guards(final_cov,target_spectra)
    step_record.update(endpoint_field_trials=endpoint['inner_field_trials'],
        endpoint_clock_residual=endpoint['clock_residual'],endpoint_ruler_residual=endpoint['ruler_residual'],
        field_counter_after=parent.work.fields,nonlinear_update_residual=update)
    return make_node(parent,endpoint,final_marker,load,node['T']+dt)


def trajectory(parent, initial, load, dt, record, directory, coefficient):
    record.update(status="partial_or_unresolved",load_exact=str(load),dt_exact=str(dt),
                  states=[],step_attempts=[],shared_initial_fields=True,shared_initial_covariances=True)
    initial_cov=[v.copy() for v in initial['covariances']]
    fields={**initial,"covariances":initial_cov}
    node=make_node(parent,fields,float(load)*parent.pg,load,Fraction(0))
    target_spectra=[la.eigvalsh(host(c)) for c in initial_cov]
    covariance_guards(initial_cov,target_spectra)
    e0=node['record']['charges']['collar_charge']
    floor=max(e0,1e-6)
    record['initial_collar_charge']=e0
    record['energy_drift_threshold']=1e-7*floor
    record['initial_mean_added_charge']=2*float(load)
    record['initial_added_charge_variance']=2*float(load)*(1-float(load))
    record['initial_is_pure_filled_rank_two']=load==1
    tag="L"+str(parent.L)+"_load"+str(load).replace('/','_')+"_dt"+str(dt).replace('/','_')
    archive_node(parent,node,directory,tag+"_T0",coefficient)
    record['states'].append(node['record'])
    previous=None
    last=None
    try:
        for step_number in range(1,int(Fraction(2)/dt)+1):
            parent.work.check()
            if parent.work.accepted_steps>=768:
                raise Cap("The whole-batch 768 accepted-time-step cap was reached")
            trial={"step":step_number,"T_start_exact":str(node['T']),"T_end_exact":str(node['T']+dt)}
            record['step_attempts'].append(trial)
            next_node=midpoint_step(parent,node,load,dt,target_spectra,trial)
            drift=next_node['record']['charges']['collar_charge']-e0
            next_node['record']['energy_drift']=drift
            trial['candidate_energy_drift']=drift
            if abs(drift)>1e-7*floor:
                raise Unresolved("The reconstructed endpoint collar-energy drift exceeded its unchanged tolerance")
            if node['T']%Fraction(1,8)==0:
                add_fd_readout(parent,node,previous,next_node,float(dt),"one-sided initial" if previous is None else "centered fixed-step")
            parent.work.accepted_steps+=1
            trial['status']="accepted"
            last,previous,node=previous,node,next_node
            path=directory/(tag+"_last_valid.npz")
            save_existing_state(parent,node['field'],node['marker'],path)
            record['last_valid_native_state_file']=str(path)
            if node['T']%Fraction(1,8)==0:
                archive_node(parent,node,directory,tag+"_T"+str(node['T']).replace('/','_'),coefficient)
                record['states'].append(node['record'])
                print("X "+tag+" accepted T="+str(node['T'])+" fields="+str(parent.work.fields)+" drift="+format(drift,'.3e'),flush=True)
                parent.checkpoint()
        add_fd_readout(parent,node,previous,None,float(dt),"one-sided final")
        record['status']="completed_Tmax"
    except BaseException as error:
        record['failure']={"class":type(error).__name__,"message":str(error),"phase":parent.work.phase}
        record['last_valid_T_exact']=str(node['T'])
        if not isinstance(error,(Unresolved,la.LinAlgError)):
            raise
        save_existing_state(parent,node['field'],node['marker'],directory/(tag+"_last_valid.npz"))
        record['last_valid_native_state_file']=str(directory/(tag+"_last_valid.npz"))
        if isinstance(error,Cap):
            raise
    record['last_valid_T_exact']=str(node['T'])
    record['accepted_steps']=sum(r.get('status')=='accepted' for r in record['step_attempts'])
    return record


def normalized_difference(left,right,floor):
    left,right=np.asarray(left),np.asarray(right)
    return float(np.linalg.norm(left-right)/max(floor,np.linalg.norm(left),np.linalg.norm(right)))


def compare_trajectories(rows, load, work):
    completed=[r for r in rows if r['status']=='completed_Tmax']
    if len(completed)<2:
        return {"status":"unresolved","reason":"Fewer than two complete fixed-step trajectories; no joined partial history"}
    coarse,fine=completed[-2:]
    records=[]
    floor=max(coarse['initial_collar_charge'],1e-6)
    for left,right in zip(coarse['states'],fine['states']):
        if left['T_exact']!=right['T_exact']:
            raise RuntimeError("The fixed reporting grids do not share a common time")
        work.check()
        metrics={}
        for key in ('psi','N'):
            metrics[key]=normalized_difference(left['fields'][key],right['fields'][key],1)
        for key in ('collar_charge','full_functional','positive_source_readout'):
            metrics[key]=normalized_difference(left['charges'][key],right['charges'][key],floor)
        for key in ('coordinate_median','coordinate_90_percent','rms_radius','rms_radius_derivative','collar_fraction','principal_coefficient','chi','octant_partition_totals'):
            if left['material'][key] is not None and right['material'][key] is not None:
                metrics[key]=normalized_difference(left['material'][key],right['material'][key],1)
        metrics['psi_velocity']=normalized_difference(left['psi_velocity'],right['psi_velocity'],1)
        for key in ('sigma_min','inverse_norm_2','minimum_polar_gap','clock_residual','ruler_residual'):
            metrics[key]=normalized_difference(left['fields'][key],right['fields'][key],1)
        for key in ('fixed_lapse_vacuum_second_variation','clock_tangency_residual_inf','Delta2','profile_norm'):
            if left[key] is not None and right[key] is not None:
                metrics[key]=normalized_difference(left[key],right[key],1)
        metrics['marked_density_per_trace']=normalized_difference(np.asarray(left['material']['marked_density'])/(2*float(load)),np.asarray(right['material']['marked_density'])/(2*float(load)),1)
        for index,(s0,s1) in enumerate(zip(left['material']['shells'],right['material']['shells'])):
            for key in ('marked_mass','mean','min','max','absolute_spread','normalized_spread'):
                if s0[key] is not None and s1[key] is not None:
                    metrics['shell'+str(index)+'_'+key]=normalized_difference(s0[key],s1[key],1)
        for key in ('clock_row_residual','ruler_row_residual'):
            metrics['instantaneous_reference_'+key]=normalized_difference(left['instantaneous_heavy_reference_approximation'][key],right['instantaneous_heavy_reference_approximation'][key],1)
        if left['primary'].get('status')==right['primary'].get('status')=='scalar_uncertainty_retained':
            for key in ('W0','primary_W2_midpoint','remainder_midpoint','remainder_clock_relative_F','remainder_ruler_relative_F'):
                if left['primary'][key] is not None and right['primary'][key] is not None:
                    metrics['primary_'+key]=normalized_difference(left['primary'][key],right['primary'][key],floor if key in ('W0','primary_W2_midpoint','remainder_midpoint') else 1)
            metrics['primary_J']=normalized_difference(left['primary']['primary_J_midpoint'],right['primary']['primary_J_midpoint'],1)
        with np.load(left['native_state_file']) as a0,np.load(right['native_state_file']) as a1:
            metrics['clock_rows']=normalized_difference(a0['clock'],a1['clock'],1)
            metrics['Jacobian']=normalized_difference(a0['J'],a1['J'],1)
            metrics['marked_covariance_per_fixed_trace']=normalized_difference(a0['marker']/(2*float(load)),a1['marker']/(2*float(load)),1)
            for sea in range(1,5):
                metrics['covariance_'+str(sea)]=normalized_difference(a0['covariance_'+str(sea)],a1['covariance_'+str(sea)],1)
                metrics['source_'+str(sea)]=normalized_difference(a0['sea_clock_'+str(sea)],a1['sea_clock_'+str(sea)],1)
                metrics['source_J_'+str(sea)]=normalized_difference(a0['sea_J_'+str(sea)],a1['sea_J_'+str(sea)],1)
        for i,(s0,s1) in enumerate(zip(left['sea_readouts'],right['sea_readouts'])):
            for key in ('instantaneous_excitation_number','particle_hole_coherence_F','L_ad','n_Xi','minimum_B_energy'):
                metrics['sea'+str(i+1)+'_'+key]=normalized_difference(s0[key],s1[key],floor if key=='L_ad' else 1)
            metrics['sea'+str(i+1)+'_Z_f']=normalized_difference(
                [r['Z_f'] for r in s0['frozen_reference_response']],[r['Z_f'] for r in s1['frozen_reference_response']],1)
        if 'adiabatic_comparison' in left and 'adiabatic_comparison' in right:
            for i,(a0,a1) in enumerate(zip(left['adiabatic_comparison'],right['adiabatic_comparison'])):
                for key in ('XS_clock_source','XS_ruler_source','normalization_clock_correction','normalization_ruler_correction','acceleration_ruler_term'):
                    metrics['sea'+str(i+1)+'_'+key]=normalized_difference(a0[key],a1[key],1)
        if 'turnover_FD' in left and 'turnover_FD' in right:
            metrics['FD_psi_acceleration']=normalized_difference(left['turnover_FD']['psi_acceleration_profile'],right['turnover_FD']['psi_acceleration_profile'],1)
        records.append({"T_exact":left['T_exact'],"normalized_observable_differences":metrics,
                        "all_at_most_1e_minus4":all(v<=1e-4 for v in metrics.values())})
    return {"status":"time_ladder_numerically_converged" if all(r['all_at_most_1e_minus4'] for r in records) else "unresolved",
        "coarse_dt":coarse['dt_exact'],"fine_dt":fine['dt_exact'],"common_interval":["0","2"],"records":records,
        "differential_defect_bound":None,"Lipschitz_stability_bound":None,"trajectory_enclosure":False}


def save_existing_state(parent, field, marker, path):
    arrays={'coordinates':parent.coords,'interior':parent.interior,'collar':parent.collar,
        'psi':field['psi'],'N':field['N'],'clock':field['C'],'J':field['J'],
        'vacuum_clock':field['vacuum'],'vacuum_J':field['vacuum_J'],'marker':host(marker)}
    for index,(cov,packet) in enumerate(zip(field['covariances'],field['packets'])):
        arrays['covariance_'+str(index+1)]=host(cov)
        arrays['sea_clock_'+str(index+1)]=host(packet['clock'])
        arrays['sea_J_'+str(index+1)]=host(packet['J'])
    np.savez(path,**arrays)


def compare_boxes(result,work):
    if len(result['boxes'])!=2:
        return {'status':'unresolved','reason':'Both finite parents are not available'}
    a0,a1=result['boxes']
    rows=[]
    for first in a0.get('loads',[]):
        second=next((v for v in a1.get('loads',[]) if v['load_exact']==first['load_exact']),None)
        if second is None:
            continue
        runs0=[v for v in first['trajectories'] if v.get('states')]
        runs1=[v for v in second['trajectories'] if v.get('states')]
        if not runs0 or not runs1:
            continue
        # Match exactly declared fixed grids; partial runs contribute only actual common records.
        common_steps={v['dt_exact'] for v in runs0}&{v['dt_exact'] for v in runs1}
        if not common_steps:
            continue
        chosen=min(common_steps,key=Fraction)
        r0=next(v for v in runs0 if v['dt_exact']==chosen)
        r1=next(v for v in runs1 if v['dt_exact']==chosen)
        for left in r0['states']:
            right=next((v for v in r1['states'] if v['T_exact']==left['T_exact']),None)
            if right is None:
                continue
            work.check()
            with np.load(left['native_state_file']) as x0,np.load(right['native_state_file']) as x1:
                ids0=np.flatnonzero(np.max(np.abs(x0['coordinates']),axis=1)<2)
                lookup={tuple(pos):i for i,pos in enumerate(x1['coordinates'])}
                ids1=np.array([lookup[tuple(x0['coordinates'][i])] for i in ids0])
                cols0=np.array([int(np.flatnonzero(x0['interior']==i)[0]) for i in ids0])
                cols1=np.array([int(np.flatnonzero(x1['interior']==i)[0]) for i in ids1])
                metrics={key:normalized_difference(x0[key][ids0],x1[key][ids1],1) for key in ('psi','N','clock','vacuum_clock')}
                metrics['rectangular_J_common_interior']=normalized_difference(x0['J'][np.ix_(ids0,cols0)],x1['J'][np.ix_(ids1,cols1)],1)
                p0=np.ravel(2*ids0[:,None]+np.arange(2)[None,:]); p1=np.ravel(2*ids1[:,None]+np.arange(2)[None,:])
                carrier0=np.r_[p0,p0+2*len(x0['coordinates'])]
                carrier1=np.r_[p1,p1+2*len(x1['coordinates'])]
                for sea in range(1,5):
                    metrics['native_covariance_'+str(sea)+'_restriction']=normalized_difference(x0['covariance_'+str(sea)][np.ix_(carrier0,carrier0)],x1['covariance_'+str(sea)][np.ix_(carrier1,carrier1)],1)
                    metrics['clock_sea_'+str(sea)]=normalized_difference(x0['sea_clock_'+str(sea)][ids0],x1['sea_clock_'+str(sea)][ids1],1)
                rows.append({'load_exact':first['load_exact'],'dt_exact':chosen,'T_exact':left['T_exact'],
                    'common_interior_sites':len(ids0),'differences':metrics,'isolated_limit_claim':False,
                    'initial_native_polar_embedding_differences_retained':True})
    return {'status':'finite_parent_comparison' if rows else 'unresolved','records':rows,'open_limit_enclosure':False}


def scalar_coefficient(work):
    import c3_secondary_branch as cc
    import flint
    cc.arb,cc.acb,cc.ctx=flint.arb,flint.acb,flint.ctx
    cc.ctx.dps,cc.ctx.threads=80,1
    class ScalarWork:
        def __init__(self):
            self.start=work.start
            self.args=argparse.Namespace(max_evaluations=100000)
            self.integration_exception=None; self.integration_depth=0
            self.phase='X single-match scalar coefficient'
        @property
        def evaluations(self):
            return work.scalar_evaluations
        def check(self,increment=0):
            work.scalar_evaluations+=increment
            work.check()
            if work.scalar_evaluations>100000:
                raise Unresolved("The optional scalar readout reached its 100000-integrand-evaluation cap")
    saved={"heat_calls":[]}
    result={"status":"unresolved","input_mu_exact":"1/2","shape":"R_D","scalar_evaluation_cap":100000}
    try:
        value=cc.Heat(ScalarWork(),saved).evaluate('R_D',cc.a(Fraction(1,2)),(0,),80,False,'X original single Newton coefficient')[0]
        relative=cc.width(value)/cc.exact_endpoint(value.lower())
        result.update(I=cc.describe(value),relative_enclosure_width_exact=str(relative),heat_records=saved)
        if relative>Fraction(1,10**8):
            raise Unresolved("The full-zone scalar coefficient relative enclosure width exceeds1e-8")
        k=3*value
        result.update(status='enclosed',k_star=cc.encode(k),G=cc.encode(1/(2*cc.arb.pi()*k)),
                      module=cc,k_ball=k)
    except BaseException as error:
        result['failure']={"class":type(error).__name__,"message":str(error)}
        result['heat_records']=saved
        if isinstance(error,Cap):
            raise
    return result


def result_write(directory,result,work):
    result['resources']=work.resources()
    result['field_work_records']=work.operations
    stem=result.get('result_stem','x_nonlinear_gaussian')
    write_json(directory/(stem+'.json'),number(result))
    lines=['X nonlinear Gaussian zero-coupling diagnostic: '+result['status'],
           'L | reached loads | trajectories completed | time convergence']
    state_files=[]
    for box in result['boxes']:
        prep=box.get('preparation',{})
        lines.append(str(box['L'])+' | '+','.join(prep.get('reached_loads',[]))+' | preparation '+prep.get('status','pending')+' | pending')
        if prep.get('last_valid_native_state_file'):
            state_files.append(prep['last_valid_native_state_file'])
        if prep.get('corrector_attempts'):
            last=prep['corrector_attempts'][-1]
            lines.append('  corrector '+str(last['attempt'])+' from='+last['from_load']+' candidate='+last['candidate_load']+
                ' status='+last['status']+' fields='+str(last.get('field_counter_before'))+'..'+str(last.get('field_counter_after'))+
                ' reason='+last.get('failure',{}).get('message','pending'))
        for load in box.get('loads',[]):
            lines.append(str(box['L'])+' | '+load['load_exact']+' | '+
                ', '.join(r['dt_exact']+':'+r['status'] for r in load.get('trajectories',[]))+' | '+
                load.get('time_convergence',{}).get('status','pending'))
            for run in load.get('trajectories',[]):
                state_files.extend(s['native_state_file'] for s in run.get('states',[]) if s.get('native_state_file'))
                if run.get('last_valid_native_state_file'):
                    state_files.append(run['last_valid_native_state_file'])
        if 'stop' in prep:
            lines.append('L='+str(box['L'])+' preparation: '+prep['stop'])
        if 'failure' in box:
            lines.append('L='+str(box['L'])+' '+box['failure']['class']+': '+box['failure']['message'])
    saved=[v for v in set(state_files) if Path(v).exists()]
    lines.append('Actual native-state NPZ files referenced and present: '+str(len(saved))+'.')
    if result.get('prior_engineering_work',{}).get('original_run_lacked_native_state_NPZ'):
        lines.append('The interrupted original run had no NPZ checkpoint; the flat input is reconstructed numerically and saved here, with all prior work charged.')
    lines.append('Numerical residuals/time comparison are not root, trajectory, open-parent, spherical or physical enclosures.')
    if 'failure' in result:
        lines.append(result['failure']['class']+': '+result['failure']['message'])
    lines.append(json.dumps(result['resources']))
    (directory/(stem+'.txt')).write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','gpu','auto'),default='auto')
    parser.add_argument('--resume-flat-engineering-result',type=Path)
    args=parser.parse_args()
    if os.environ.get('QI2_REMOTE_RUN')!='1' or sys.executable!='<internal>/python':
        parser.error('Run only in the selected remote qi2 interpreter')
    if os.nice(0)<10:
        parser.error('The own job must run at nice 10')
    if any(os.environ.get(k) not in ('1','2','4') for k in POOLS):
        parser.error('All six own thread pools must be1,2 or4; total own threads remains capped at16')
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    prior=None
    pause_seconds=0
    stem='x_nonlinear_gaussian_continued' if args.resume_flat_engineering_result else 'x_nonlinear_gaussian'
    if (args.output_dir/(stem+'.json')).exists():
        parser.error('An existing scientific result cannot be overwritten')
    if args.resume_flat_engineering_result:
        prior=json.loads(args.resume_flat_engineering_result.read_text())
        rr=prior['resources']; pp=prior['boxes'][0]['preparation']
        valid=(prior['specification']==SPEC and prior['inputs']['mass_squared_exact']==[str(v) for v in MASS2]
            and prior['inputs']['loads_exact']==[str(v) for v in LOADS]
            and prior['inputs']['dt_exact']==[str(v) for v in STEPS] and prior['inputs']['g_M']==prior['inputs']['g_B']=='0'
            and len(prior['boxes'])==1 and prior['boxes'][0]['L']==5 and not pp['reached_loads']
            and len(pp['corrector_attempts'])==3 and all(v['from_load']=='0' and v['status']=='failed_or_incomplete' for v in pp['corrector_attempts'])
            and rr['requested_field_source_evaluations']==35 and rr['completed_field_source_evaluations']==7
            and rr['failed_or_incomplete_field_source_evaluations']==28 and rr['scalar_integrand_evaluations']==591
            and rr['accepted_time_steps']==0 and len(prior['field_work_records'])==35
            and rr['elapsed_seconds']==446.38216284289956 and prior['failure']['class']=='KeyboardInterrupt')
        if not valid:
            parser.error('The exact authorized flat-only engineering counters/state conditions were not restored')
        pause_seconds=max(0,time.time()-args.resume_flat_engineering_result.stat().st_mtime)+1
    global np,la,cp,xp
    import numpy as np
    import scipy.linalg as la
    cp=None; xp=np
    cuda=None
    if args.backend in ('gpu','auto'):
        import cupy as cp
        # Host imports finish before final free-query/allocation/sole-PID admission.
        listing=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        occupied={v.split(',')[0].strip() for v in owners.splitlines()}
        eligible=[line.split(',')[0].strip() for line in listing.splitlines()
                  if line.split(',')[0].strip() not in occupied and int(line.split(',')[1].strip())<=16]
        if eligible:
            os.environ['QI2_GPU_UUID']=eligible[0]
            os.environ['CUDA_VISIBLE_DEVICES']=eligible[0]
            from remote_runtime import remote_execution
            cuda=remote_execution(max_threads=16)
            cp.cuda.Device(0).use()
            xp=cp
        elif args.backend=='gpu':
            raise RuntimeError('No freshly eligible free GPU; no allocation or busy-device launch')
    work=Work(args.output_dir,prior,pause_seconds)
    result={'status':'unresolved','result_stem':stem,'specification':SPEC,'single_review':REVIEW,
        'runtime':{'PID':os.getpid(),'nice':os.nice(0),'interpreter':sys.executable,
                   'backend':'GPU dense products/CPU spectral factors' if xp is cp else 'CPU', 'cuda_admission':cuda},
        'inputs':{'mass_squared_exact':[str(v) for v in MASS2],'orientation':list(ETA),
                  'g_M':'0','g_B':'0','a':'1','Lambda':'sqrt(1/12)','ell':'1','boxes':[5,7],
                  'loads_exact':[str(v) for v in LOADS],'dt_exact':[str(v) for v in STEPS],'Tmax':'2'},
        'caps':{'seconds':21600,'device_bytes':24*2**30,'host_bytes':32*2**30,
                'complete_field_source_evaluations':16384,'accepted_steps':768,'own_threads':16,
                'outer_midpoint_attempts_per_step':24,'load_corrector_attempts_per_box':96,
                'inner_field_trials_per_solve':12,'endpoint_field_trials_per_solve':12},
        'boxes':[],'root_enclosure':False,'trajectory_enclosure':False,'physical_collapse_claim':False,
        'spherical_transfer':False,'open_parent_limit':False,'production_parameters_adopted':False,
        'full_Fock_matrix_built':False,'moving_pair_state_truncation':False,'sea_reset_in_evolution':False}
    if prior:
        result['prior_engineering_work']={'original_result_file':str(args.resume_flat_engineering_result),
            'original_runtime':prior['runtime'],'resources':prior['resources'],
            'failure':prior['failure'],'preparation':prior['boxes'][0]['preparation'],
            'flat_controls':prior['boxes'][0]['flat_controls'],
            'pause_seconds_charged_to_original_six_hour_wall':pause_seconds,
            'original_run_lacked_native_state_NPZ':True,
            'restored_state':'fresh numerical reconstruction of the same exact declared protected flat input; no nonflat accepted state or physical step existed',
            'counters_and_caps_reset':False}
    print('X remote numerical process PID='+str(os.getpid())+' backend='+result['runtime']['backend'],flush=True)
    coefficient={'status':'unresolved'}
    try:
        work.check()
        coefficient=scalar_coefficient(work)
        result['Newton_coefficient']={k:v for k,v in coefficient.items() if k not in ('module','k_ball')}
        for length in (5,7):
            box={'L':length,'status':'unresolved','loads':[]}
            result['boxes'].append(box)
            try:
                parent=Parent(length,work)
                state=parent.preparation()
                covariance_guards(state)
                started=time.perf_counter()
                flat,controls=flat_controls(parent,state)
                box['flat_controls']=controls
                box['feasibility']={'native_one_particle_dimension':parent.d,'carrier_covariance_dimension':2*parent.d,
                    'interior_fields':len(parent.interior),'derivative_batch_columns':8,
                    'flat_field_seconds':time.perf_counter()-started,'resources':work.resources()}
                print('X L='+str(length)+' flat algebraic/field controls passed; '+str(box['feasibility']),flush=True)
                box['preparation']={}
                path=args.output_dir/('L'+str(length)+'_last_valid_preparation.npz')
                save_existing_state(parent,flat,xp.zeros_like(parent.pg),path)
                box['preparation'].update(last_valid_load_exact='0',last_valid_native_state_file=str(path),
                    deterministic_flat_input_reconstructed_and_checkpointed=True)
                parent.checkpoint=lambda:result_write(args.output_dir,result,work)
                parent.checkpoint()
                reached=continue_loads(parent,flat,box['preparation'],prior['boxes'][0]['preparation'] if prior and length==5 else None)
                last_load,last_field=parent.last_preparation
                save_existing_state(parent,last_field,float(last_load)*parent.pg,
                    args.output_dir/('L'+str(length)+'_last_valid_preparation.npz'))
                box['preparation']['last_valid_load_exact']=str(last_load)
                box['preparation']['last_valid_native_state_file']=str(args.output_dir/('L'+str(length)+'_last_valid_preparation.npz'))
                for load,initial in reached.items():
                    row={'load_exact':str(load),'trajectories':[]}
                    box['loads'].append(row)
                    for dt in STEPS:
                        run={}
                        row['trajectories'].append(run)
                        trajectory(parent,initial,load,dt,run,args.output_dir,coefficient)
                        result_write(args.output_dir,result,work)
                    row['time_convergence']=compare_trajectories(row['trajectories'],load,work)
                box['status']='completed_bounded_box' if len(reached)==3 and all(r.get('time_convergence',{}).get('status')=='time_ladder_numerically_converged' for r in box['loads']) else 'partial_or_unresolved'
            except Cap:
                raise
            except (Unresolved,la.LinAlgError) as error:
                box['failure']={'class':type(error).__name__,'message':str(error),'phase':work.phase}
                print('X L='+str(length)+' '+type(error).__name__+': '+str(error),flush=True)
            result_write(args.output_dir,result,work)
        result['finite_parent_comparison']=compare_boxes(result,work)
        result['status']='converged_numerical_diagnostic' if all(b['status']=='completed_bounded_box' for b in result['boxes']) else 'partial_or_unresolved_numerical_diagnostic'
    except BaseException as error:
        result['failure']={'class':type(error).__name__,'message':str(error),'phase':work.phase}
        print(type(error).__name__+': '+str(error),flush=True)
        traceback.print_exc()
    finally:
        result_write(args.output_dir,result,work)
        print('X final '+result['status']+'; '+str(work.resources()),flush=True)
    return 0 if result['status']=='converged_numerical_diagnostic' else 2


if __name__=='__main__':
    raise SystemExit(main())
