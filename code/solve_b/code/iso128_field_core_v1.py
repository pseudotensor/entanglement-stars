"""Own continuous field implementation of the frozen ISO128 field recipe.

No foreign solver objects or implementations are accepted.  Numerical work is
performed only by the permitted-box entry points, after manufactured controls.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
import math
import numpy as np
from scipy.integrate import DOP853
from iso128_rebuild_core_v1 import gaussian_kernels, gaussian_centre_coefficients


class DomainUnavailable(ValueError):
    """An inspected branch or query left the declared regular domain."""


def real_vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError('Real array required: ' + name)
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or not len(value) or not np.all(np.isfinite(value)):
        raise ValueError('Finite nonempty vector required: ' + name)
    return value


class GaussianSources:
    """Full shell sum, with separately recorded numerical-zero omission bounds.

    The deliberately loose envelope bound uses the entire absolute source sum
    and the nearest omitted shell. Decimal directed arithmetic keeps positive
    bounds representable even when binary64 kernels underflow. It does not
    purport to bound the other floating-point errors of the Gaussian formula.
    """
    def __init__(self, radius, energy, current, ell=4.):
        self.radius = real_vector(radius, 'shell radius')
        self.energy = real_vector(energy, 'energy')
        self.current = real_vector(current, 'current')
        if (self.energy.shape != self.radius.shape or self.current.shape != self.radius.shape
                or np.any(self.radius <= 0) or np.any(np.diff(self.radius) <= 0)
                or not math.isfinite(ell) or ell <= 0):
            raise ValueError('Matching sources on strictly increasing positive shells required')
        self.ell = float(ell)
        self.centre = gaussian_centre_coefficients(self.energy, self.current, self.radius, ell)
        self.vacuum = bool(np.all(self.energy == 0) and np.all(self.current == 0))
        self.evaluations = []
        with localcontext() as ctx:
            ctx.prec = 80; ctx.rounding = ROUND_CEILING
            self.abs_sums = [sum((Decimal.from_float(abs(float(x))) for x in a), Decimal(0))
                             for a in (self.energy, self.current)]
            # 2*c is an explicit generous upper bound. A nextafter upward
            # conversion bounds the binary64 evaluation of this constant.
            self.envelope_constant = Decimal.from_float(float(np.nextafter(
                2 * (2 * np.pi * ell**2)**(-1.5), np.inf)))
        k0,_=gaussian_kernels(0.,self.radius,self.ell)
        factors={'rho0':(self.energy,np.ones_like(self.radius)),
                 'rho2':(self.energy,self.radius**2/(6*ell**4)-1/(2*ell**2)),
                 'J1':(self.current,self.radius/(3*ell**2)),
                 'J3':(self.current,self.radius**3/(30*ell**6)-self.radius/(6*ell**4))}
        self.centre_omissions={}
        for key,(raw,factor) in factors.items():
            effective=raw*factor
            if not np.all(np.isfinite(effective)):raise DomainUnavailable('Nonfinite effective centre source')
            with localcontext() as ctx:
                ctx.prec=80;ctx.rounding=ROUND_CEILING
                total=sum((Decimal.from_float(abs(float(x)))*Decimal.from_float(abs(float(y))) for x,y in zip(raw,factor)),Decimal(0))
            base=raw*k0
            omitted=(raw!=0)&(factor!=0)&((effective==0)|(k0==0)|(base==0)|(effective*k0==0)|(base*factor==0))
            self.centre_omissions[key]=self.omission_bound(0.,omitted,0,absolute_sum=total)

    def omission_bound(self, r, omitted, index,absolute_sum=None):
        if not np.any(omitted):
            return {'count': 0, 'decimal_upper': '0', 'float_upper': 0.}
        # Each subtraction is correctly rounded in binary64. Its lower
        # neighbor bounds the exact nonnegative distance from below, including
        # when subtraction rounded upward; taking the minimum preserves this.
        distance = float(np.min(np.abs(abs(float(r))-self.radius[omitted])))
        nearest = Decimal.from_float(float(np.nextafter(distance,0.)))
        with localcontext() as ctx:
            ctx.prec = 80; ctx.rounding = ROUND_FLOOR
            exponent_size = nearest * nearest / (2 * Decimal.from_float(self.ell)**2)
            # Decimal.exp is correctly rounded to nearest, irrespective of
            # context rounding. next_plus supplies an explicit upper neighbor.
            envelope = (-exponent_size).exp().next_plus()
            ctx.rounding = ROUND_CEILING
            bound = ((self.abs_sums[index] if absolute_sum is None else absolute_sum) * self.envelope_constant * envelope).next_plus()
            converted = float(bound)
            if math.isfinite(converted):
                converted = float(np.nextafter(converted, np.inf))
            if not math.isfinite(converted) or bound <= 0:
                raise DomainUnavailable('Numerical omission upper bound unavailable')
        return {'count': int(np.sum(omitted)), 'decimal_upper': str(bound), 'float_upper': converted}

    def __call__(self, r):
        if not math.isfinite(r) or r < 0:
            raise DomainUnavailable('Source evaluation requires finite nonnegative radius')
        k0, k1 = gaussian_kernels(float(r), self.radius, self.ell)
        products0,products1=self.energy*k0,self.current*k1
        rho, current = float(np.sum(products0)), float(np.sum(products1))
        if not math.isfinite(rho) or not math.isfinite(current):
            raise DomainUnavailable('Nonfinite direct Gaussian source')
        omitted0 = ((k0 == 0) | (products0 == 0)) & (self.energy != 0)
        omitted1 = ((k1 == 0) | (products1 == 0)) & (self.current != 0) & (r != 0)
        b0, b1 = self.omission_bound(r, omitted0, 0), self.omission_bound(r, omitted1, 1)
        self.evaluations.append({'r': float(r), 'rho': rho, 'J': current,
                                 'rho_omission': b0, 'J_omission': b1,
                                 'J_structural_zero': bool(r == 0),
                                 'rho_subnormal_products':int(np.sum((products0!=0)&(np.abs(products0)<np.finfo(float).tiny))),
                                 'J_subnormal_products':int(np.sum((products1!=0)&(np.abs(products1)<np.finfo(float).tiny))),
                                 'bound_method': 'full_absolute_source_sum_times_largest_omitted_envelope_2c_decimal80'})
        return rho, current

    def sample(self, radius):
        shape = np.shape(radius)
        return np.asarray([self(float(r)) for r in np.ravel(radius)]).reshape(shape+(2,))


def centre_coefficients(centre, g):
    values = [centre[k] for k in ('rho0', 'rho2', 'J1', 'J3')]
    if any(np.iscomplexobj(v) or np.asarray(v).shape!=() for v in values+[g]) or not all(math.isfinite(v) for v in values) or not math.isfinite(g) or g <= 0:
        raise DomainUnavailable('Nonfinite centre or invalid coupling')
    rho0, rho2, j1, j3 = values
    if rho0 <= 0:
        raise DomainUnavailable('regular_branch_unavailable: nonpositive central density')
    A = 8*np.pi*g*rho0/3
    if not math.isfinite(A) or A<=0:raise DomainUnavailable('Nonpositive/nonfinite derived central mass coefficient')
    B = 8*np.pi*g*(rho2-math.sqrt(A)*j1)/5
    C = 2*np.pi*g*j1/math.sqrt(A)
    D = np.pi*g*(j3-j1*B/(2*A))/math.sqrt(A)
    if not all(math.isfinite(x) for x in (A,B,C,D)) or A<=0:
        raise DomainUnavailable('Nonfinite derived centre coefficient')
    return dict(A=float(A), B=float(B), C=float(C), D=float(D))


class SolvedField:
    """Own accepted-step dense interpolants plus regular centre continuation."""
    def __init__(self, source, g, collar, end, ell, variant='P', breakpoints=()):
        self.field_role='own_normalized_field'
        if variant not in ('P','H','C','O'):
            raise ValueError('Unknown declared variant')
        if not all(math.isfinite(x) and x > 0 for x in (g, collar, end, ell)) or end < collar:
            raise ValueError('Positive finite field domain required')
        self.source, self.g, self.collar, self.end, self.ell = source, g, collar, end, ell
        self.variant = variant; self.steps = []; self.interpolants = []; self.trials = []
        self.failure = None; self.status = 'not_started'; self.nfev = 0
        self.vacuum = bool(source.vacuum)
        self.r0 = ell*2**(-13 if variant == 'C' else -12)
        self.rtol = 1e-13 if variant == 'H' else 1e-12
        self.max_step = ell/(8 if variant == 'H' else 4)
        self.coefficients = {} if self.vacuum else centre_coefficients(source.centre,g)
        if any(np.iscomplexobj(x) or np.asarray(x).shape!=() or not math.isfinite(x) for x in breakpoints):raise ValueError('Finite real declared breakpoints required')
        self.breakpoints = tuple(float(x) for x in breakpoints)
        if any(x <= self.r0 or x >= end for x in self.breakpoints) or tuple(sorted(set(self.breakpoints))) != self.breakpoints:
            raise ValueError('Strictly ascending interior breakpoints required')

    def centre_state(self, r):
        r = np.asarray(r, dtype=float)
        if self.vacuum:
            return np.stack((np.zeros_like(r),np.zeros_like(r)),axis=-1)
        A,B,C,D = (self.coefficients[k] for k in ('A','B','C','D'))
        return np.stack((A*r**3+B*r**5,C*r**2+D*r**4),axis=-1)

    def rhs(self, r, state):
        U,y = map(float,state)
        if not math.isfinite(r) or r <= 0 or not math.isfinite(U) or U <= 0 or not math.isfinite(y):
            safe=lambda x:float(x) if math.isfinite(float(x)) else str(float(x))
            self.trials.append({'r':safe(r),'state':[safe(x) for x in state], 'status':'regular_domain_unresolved'})
            raise DomainUnavailable('regular_domain_unresolved: nonpositive/nonfinite U trial')
        rho,J = self.source(r)
        ratio=U/r
        if not math.isfinite(ratio) or ratio<=0:
            self.trials.append({'r':float(r),'state':[U,y],'status':'nonpositive_or_nonfinite_U_over_r'})
            raise DomainUnavailable('Regular ratio U/r unavailable')
        s = math.sqrt(ratio)
        if not math.isfinite(s) or s<=0:raise DomainUnavailable('Regular inward speed unavailable')
        result = np.array([8*np.pi*self.g*r*r*(rho-s*J),4*np.pi*self.g*r*J/s])
        if not np.all(np.isfinite(result)):
            self.trials.append({'r':float(r),'state':[U,y],'status':'nonfinite_rhs'})
            raise DomainUnavailable('Nonfinite field RHS')
        return result

    def integrate(self):
        if self.status != 'not_started':
            raise ValueError('No implicit field replay or restart')
        if self.vacuum:
            self.status='complete'; self.y_collar=0.; return self
        state = self.centre_state(self.r0)
        A=self.coefficients['A']; atolU=1e-16*abs(A*self.r0**3)
        if not np.all(np.isfinite(state)) or state[0] <= 0 or not math.isfinite(atolU) or atolU <= 0:
            self.status='regular_branch_unavailable'; self.failure='Nonpositive start or underflowed tolerance'; return self
        self.atol = [atolU,1e-16]
        self.correction_sizes = {'U':abs(self.coefficients['B']*self.r0**2/A),
            'y':abs(self.coefficients['D']*self.r0**2/self.coefficients['C']) if self.coefficients['C'] else None,
            'y_zero_scale':self.coefficients['C']==0}
        self.steps.append({'r':self.r0,'state':state.tolist()})
        start=self.r0
        try:
            for stop in self.breakpoints+(self.end,):
                solver=DOP853(self.rhs,start,state,stop,rtol=self.rtol,atol=self.atol,max_step=self.max_step)
                while solver.status == 'running':
                    message=solver.step()
                    if solver.status == 'failed':
                        raise DomainUnavailable('DOP853 failure: '+str(message))
                    self.interpolants.append(solver.dense_output())
                    self.steps.append({'r':float(solver.t),'state':solver.y.tolist()})
                self.nfev += solver.nfev
                state=solver.y.copy(); start=stop
            self.status='complete'
            self.y_collar=float(self.state(self.collar)[1])
            # Inspect all accepted endpoints after the gauge is fixed.
            self.fields(np.array([x['r'] for x in self.steps]))
        except (DomainUnavailable,ValueError,FloatingPointError,OverflowError) as exc:
            self.status='regular_domain_unresolved'; self.failure=str(exc)
        return self

    def state(self, radius):
        if np.iscomplexobj(radius):raise DomainUnavailable('Real query radius required')
        rr=np.asarray(radius,dtype=float)
        if np.any(~np.isfinite(rr)) or np.any(rr<0) or np.any(rr>self.end):
            raise DomainUnavailable('Unsupported interpolation domain')
        if self.vacuum:
            return np.stack((np.zeros_like(rr),np.zeros_like(rr)),axis=-1)
        result=np.empty((rr.size,2)); flat=rr.ravel(); centre=flat<self.r0
        result[centre]=self.centre_state(flat[centre])
        endpoints=np.array([x.t for x in self.interpolants])
        if np.any(~centre):
            indexes=np.searchsorted(endpoints,flat[~centre],side='left')
            if np.any(indexes>=len(endpoints)):
                raise DomainUnavailable('Dense output not complete to query')
            target=np.flatnonzero(~centre)
            for index in np.unique(indexes):
                selected=target[indexes==index]
                result[selected]=self.interpolants[index](flat[selected]).T
        return result.reshape(rr.shape+(2,))

    def fields(self, radius, normalization=None):
        if np.iscomplexobj(radius):raise DomainUnavailable('Real field query radius required')
        if self.status != 'complete':
            raise DomainUnavailable('Incomplete field cannot be read as a complete solution')
        rr=np.asarray(radius,dtype=float); state=self.state(rr); U,y=state[...,0],state[...,1]
        if self.vacuum:
            return dict(U=U,y=y,N=np.ones_like(rr),v=np.zeros_like(rr),s=np.zeros_like(rr))
        positive=rr>0
        if np.any(U[positive]<=0):
            raise DomainUnavailable('Nonpositive interpolated U')
        gauge=self.y_collar if normalization is None else float(self.state(normalization)[1])
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            N=np.exp(y-gauge); s=np.zeros_like(rr); s[positive]=np.sqrt(U[positive]/rr[positive]); v=-N*s
        if not np.all(np.isfinite(N)) or np.any(N<=0) or not np.all(np.isfinite(v)) or np.any(v[positive]>=0):
            raise DomainUnavailable('Nonfinite or nonpositive lapse')
        return dict(U=U,y=y,N=N,v=v,s=s)

    def observables(self, radius):
        r=float(radius); field=self.fields(r)
        N,s=float(field['N']),float(field['s'])
        if self.vacuum:
            return {'f':1.,'g':-1.,'df':0.,'dg':0.,'N':1.,'dN':0.}
        if r<self.r0:
            A,B,C,D=(self.coefficients[k] for k in ('A','B','C','D'))
            sp=math.sqrt(A+B*r*r)+B*r*r/math.sqrt(A+B*r*r)
            yp=2*C*r+4*D*r**3
            gp=2*A*r+4*B*r**3
        else:
            U=float(field['U']); up,yp=self.rhs(r,[U,float(field['y'])])
            sp=(up/r-U/r**2)/(2*s); gp=up/r-U/r**2
        return {'f':N*(1-s),'g':s*s-1,'df':N*(yp*(1-s)-sp),
                'dg':gp,'N':N,'dN':N*yp}

    def applied_exterior(self,radius):
        if np.iscomplexobj(radius):raise DomainUnavailable('Real exterior query radius required')
        rr=np.asarray(radius,dtype=float)
        if np.any(~np.isfinite(rr)) or np.any(rr<0):
            raise DomainUnavailable('Invalid applied-field radius')
        N=np.empty_like(rr);v=np.empty_like(rr);inside=rr<=self.collar
        inner=self.fields(rr[inside]);N[inside]=inner['N'];v[inside]=inner['v']
        charge=float(self.state(self.collar)[0]);N[~inside]=1.;v[~inside]=-np.sqrt(charge/rr[~inside])
        return N,v
