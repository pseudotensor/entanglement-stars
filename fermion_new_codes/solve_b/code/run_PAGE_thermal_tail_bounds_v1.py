"""Directed scalar contraction of the reviewed uniform thermal-tail bounds."""
import argparse
from fractions import Fraction
import heapq
import json
import math
from pathlib import Path
from cpu_runtime import require_remote_cpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-panels',type=int,default=64)
    parser.add_argument('--max-panels',type=int,default=1024)
    parser.add_argument('--action-tolerance',default='1/64')
    parser.add_argument('--controls-only',action='store_true')
    args=parser.parse_args();runtime=require_remote_cpu(args.output)
    from flint import arb,ctx
    import flint
    ctx.threads=1
    def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def ceil_decimal(x):
        endpoint=x.upper();v=float(endpoint)
        if not math.isfinite(v):raise RuntimeError('Nonfinite interval endpoint')
        while arb(repr(v)).lower()<endpoint:v=math.nextafter(v,math.inf)
        return repr(v)
    def floor_decimal(x):
        endpoint=x.lower();v=float(endpoint)
        if not math.isfinite(v):raise RuntimeError('Nonfinite interval endpoint')
        while arb(repr(v)).upper()>endpoint:v=math.nextafter(v,-math.inf)
        return repr(v)
    def enc(x):return {'ball':x.str(65,more=True),'lower_decimal':floor_decimal(x),'upper_decimal':ceil_decimal(x)}
    def rational(f):return arb(f.numerator)/f.denominator
    def squared_interval(value):
        low,high=value.lower(),value.upper()
        minimum=arb(0) if value.contains(0) else min(abs(low),abs(high))**2
        maximum=max(abs(low),abs(high))**2
        return minimum.lower().union(maximum.upper())
    def product_interval(left,right):
        values=[a*b for a in (left.lower(),left.upper()) for b in (right.lower(),right.upper())]
        return min(v.lower() for v in values).union(max(v.upper() for v in values))
    def moments(rho,K=9):
        den=1-rho
        if not den>0:raise RuntimeError('Geometric ratio is not below one')
        n0=1/den;n1=rho/den**2;n2=rho*(1+rho)/den**3
        n3=rho*(1+4*rho+rho**2)/den**4
        power=rho**K
        return {1:power*(K*n0+n1),2:power*(K*K*n0+2*K*n1+n2),
                3:power*(K**3*n0+3*K*K*n1+3*K*n2+n3)}

    def run_precision(bits,counts):
        ctx.prec=bits
        folder=args.output/f'precision_{bits}';folder.mkdir(exist_ok=True)
        pi=arb.pi();b=4*pi;tm=2/(3*arb(3).sqrt());tau=arb(1)/2000
        alpha=arb(19)/20;K=9;C=4*((1+alpha)/(1-alpha))**2
        action_tol=arb(args.action_tolerance);unit=1/arb(3).sqrt()
        root_cache={};action_cache={};root_report={};action_report={}
        def point(frac,scaled=False):
            val=tau+(tm-tau)*rational(frac)
            return alpha*val if scaled else val
        def root(t,branch):
            lo,hi=(arb(0),unit.mid()) if branch==0 else (unit.mid(),arb(1))
            left_sign=lo*(1-lo**2)-t;right_sign=hi*(1-hi**2)-t
            if not ((left_sign<0 and right_sign>0) if branch==0 else (left_sign>0 and right_sign<0)):
                raise RuntimeError('Initial root bracket lacks strict opposite signs')
            for _ in range(bits+16):
                mid=((lo+hi)/2).mid();delta=mid*(1-mid**2)-t
                if delta>0:
                    if branch==0:hi=mid
                    else:lo=mid
                elif delta<0:
                    if branch==0:lo=mid
                    else:hi=mid
                else:break
                if hi-lo<arb(2)**(-bits+20):break
            value=lo.union(hi)
            if not (value>0 and value<1):raise RuntimeError('Root left physical interval')
            if branch==0 and not value.upper()<unit.lower():raise RuntimeError('Inner root branch unresolved')
            if branch==1 and not value.lower()>unit.upper():raise RuntimeError('Outer root branch unresolved')
            residual=value*(1-value**2)-t
            if not residual.contains(0):raise RuntimeError('Root interval lacks zero residual')
            return value,{'left_residual':enc(left_sign),'right_residual':enc(right_sign)}
        def roots(frac,scaled=False):
            key=('a' if scaled else 't')+str(frac)
            if key in root_cache:return root_cache[key]
            if frac==1 and not scaled:
                value=(unit,unit,arb(0));root_cache[key]=value
                root_report[key]={'coalesced':True,'t':enc(tm),'I':enc(arb(0))};return value
            t=point(frac,scaled);u0,bracket0=root(t,0);u1,bracket1=root(t,1)
            integral=2*(u1.atanh()-u0.atanh())
            if not integral>0:raise RuntimeError('Nonpositive I')
            root_cache[key]=(u0,u1,integral)
            root_report[key]={'t':enc(t),'inner':enc(u0),'outer':enc(u1),'I':enc(integral),
                              'initial_inner_bracket':bracket0,'initial_outer_bracket':bracket1}
            return root_cache[key]
        def theta_interval(t,u0,u1,l,r):
            theta=(pi*rational(l)/2).union(pi*rational(r)/2)
            sn=squared_interval(theta.sin());cs=squared_interval(theta.cos())
            u=u0+product_interval(u1-u0,sn)
            one_minus_u2=1-squared_interval(u)
            positive_w=product_interval(u,one_minus_u2)
            radical=product_interval(u+u0+u1,positive_w+t).sqrt()
            denominator=product_interval(u,squared_interval(one_minus_u2))
            integrand=4*(u1-u0)**2*sn*cs*radical/denominator
            integral=integrand*pi*rational(r-l)/2
            lower=max(arb(0),integral.lower());upper=integral.upper()
            if not upper>=lower:
                raise RuntimeError(f'Invalid theta enclosure at t={t}, theta fractions={l},{r}, u={u}, integrand={integrand}')
            return lower,upper,integrand
        def action(frac):
            if frac in action_cache:return action_cache[frac]
            if frac==1:action_cache[frac]=arb(0);return arb(0)
            t=point(frac);u0,u1,I=roots(frac)
            heap=[];sequence=0;low=arb(0);high=arb(0)
            def add(l,r):
                nonlocal sequence,low,high
                lower,upper,integrand=theta_interval(t,u0,u1,l,r)
                low+=lower;high+=upper;sequence+=1
                heapq.heappush(heap,(-float((upper-lower).upper()),sequence,l,r,lower,upper,integrand))
            for j in range(16):add(Fraction(j,16),Fraction(j+1,16))
            while not high.upper()-low.lower()<=(action_tol*99/100).lower():
                if len(heap)>=65536:raise RuntimeError('Action panel budget exhausted')
                _,_,l,r,lower,upper,_=heapq.heappop(heap)
                low-=lower;high-=upper;m=(l+r)/2;add(l,m);add(m,r)
            enclosure=low.lower().union(high.upper())
            if not enclosure.upper()-enclosure.lower()<=action_tol:
                raise RuntimeError('Rounded action enclosure exceeds its assigned uncertainty')
            if not (enclosure.lower()>=0 and enclosure.upper()<=I.upper()):
                raise RuntimeError('Action fails 0<=S<=I enclosure control')
            panels=[]
            for _,_,l,r,lower,upper,integrand in sorted(heap,key=lambda row:row[2]):
                panels.append({'left':str(l),'right':str(r),'integrand':enc(integrand),
                               'integral_lower':enc(lower),'integral_upper':enc(upper)})
            payload={'fraction':str(frac),'t':enc(t),'S':enc(enclosure),'I':enc(I),
                     'action_tolerance':enc(action_tol),'theta_panels':panels}
            name=f'action_{frac.numerator}_{frac.denominator}.json'
            write(folder/name,payload);action_report[str(frac)]={'file':name,'S':enc(enclosure),'theta_panels':len(panels)}
            action_cache[frac]=enclosure;return enclosure

        # Known answers, independent finite series remainder, and branch controls.
        q=arb(1);d=arb(1)/3;W=arb(2);p=(W**2-q**2).sqrt()
        diagonal=(p*d).cosh();upper=(p*d).sinh()*(W+q)/p;lower=(p*d).sinh()*(W-q)/p
        gamma=4/(2*diagonal**2+upper**2+lower**2+2)
        answer=1/(1+W**2*(p*d).sinh()**2/(W**2-q**2))
        free_gamma=4/(2*(q*d).cos()**2+2*(q*d).sin()**2+2)
        controls={'constant_barrier_difference':enc(gamma-answer),'free_gamma_minus_one':enc(free_gamma-1),
                  'geometric':{},'zero_energy':[]}
        if not (gamma-answer).contains(0) or not (free_gamma-1).contains(0):raise RuntimeError('Transfer known answer failed')
        rho=arb(1)/4;closed=moments(rho);M=32
        for power in (1,2,3):
            partial=sum((arb(k)**power*rho**k for k in range(K,M+1)),arb(0))
            ratio=rho*(arb(M+2)/(M+1))**power
            remainder=arb(M+1)**power*rho**(M+1)/(1-ratio)
            gap=closed[power]-partial
            if not (gap.lower()>0 and gap.upper()<remainder.lower()):raise RuntimeError('Geometric series known answer failed')
            controls['geometric'][str(power)]={'closed':enc(closed[power]),'partial':enc(partial),'remainder_bound':enc(remainder)}
        for xint in (2,5):
            x=arb(xint);f=1-1/x;g=x*(x-1);F=2*x-1+2*g.sqrt()
            Fx=2+(2*x-1)/g.sqrt();Fxx=-1/(2*g*g.sqrt())
            derivative2=f*f*Fxx+f*Fx/x**2
            w=f.sqrt()/x;wp=f.sqrt()*(3-2*x)/(2*x**3)
            residual=derivative2-(w*w+wp)*F
            if not residual.contains(0):raise RuntimeError('Zero-energy derivative control failed')
            controls['zero_energy'].append({'x':xint,'residual':enc(residual)})
        xL=arb(3)/2;xR=arb(5);kap=arb(3)
        wL=(1-1/xL).sqrt()/xL;wR=(1-1/xR).sqrt()/xR
        tails=kap**2*(1-1/xL)+kap*wL+kap**2/xR-kap*wR
        middle=kap**2*(1/xL-1/xR)+kap*(wR-wL)
        if not (tails+middle-kap**2).contains(0):raise RuntimeError('Integrated potential control failed')
        controls['potential_integral_difference']=enc(tails+middle-kap**2)
        for frac in (Fraction(0),Fraction(1,4),Fraction(3,4),Fraction(1)):
            sv=action(frac);iv=roots(frac)[2]
            if not (sv.lower()>=0 and sv.upper()<=iv.upper()):raise RuntimeError('Action known control failed')
        write(folder/'known_answers.json',controls)
        if args.controls_only:
            write(folder/'roots.json',root_report)
            return {'bits':bits,'controls_passed':True}

        q0=arb(1)/1000
        def low_piece(weight,occupation,power):
            N=weight/(2*pi)*occupation*q0**(power+1)/(power+1)
            return {'number':N,'energy':weight/(2*pi)*occupation*q0**(power+2)/(power+2),
                    'entropy':N*(1-occupation.log()-power*q0.log()+arb(power)/(power+1))}
        first=low_piece(arb(64),arb(57)/400,2);other=low_piece(arb(2240),arb(129)/2,4)
        retained_low={name:first[name]+other[name] for name in first}
        retained_high={'number':arb(2304)/(2*pi)*(-3*b).exp()/b,
                       'energy':arb(2304)/(2*pi)*(-3*b).exp()*(3/b+1/b**2),
                       'entropy':arb(2304)/(2*pi)*(-3*b).exp()*(3+2/b)}
        smallP=moments(arb(1).exp()**2*tau**2)
        small={'number':576*tau/(19*pi)*smallP[2],
               'energy':576*tau**2/(20*pi)*smallP[3],
               'entropy':576*tau/(19*pi)*(2*smallP[2]+2*(1/tau).log()*smallP[3])}
        largeP=moments((-b*tm).exp())
        large={'number':32/(pi*b)*largeP[1],
               'energy':32/pi*(tm*largeP[2]/b+largeP[1]/b**2),
               'entropy':32/pi*(tm*largeP[2]+2*largeP[1]/b)}
        denominators={key:arb(value) for key,value in {'number':'0.015541820002075492',
             'energy':'0.005237315471366045','entropy':'0.10787652993789794'}.items()}
        refinements=[]
        for count in counts:
            sums={name:arb(0) for name in denominators};panels=[]
            width=((tm-tau)/count).upper()
            for j in range(count):
                l=Fraction(j,count);r=Fraction(j+1,count)
                Slower=action(r).lower();Ilower=roots(r)[2].lower();Iupper=roots(l,True)[2].upper()
                Fbound=(Slower-Iupper+Ilower).lower()
                h=max(arb(0),(2*Fbound-C.log()/K).lower())
                a=(b*point(l)+h).lower()
                if not a>0:raise RuntimeError('Nonpositive exponential envelope')
                ps=moments((-a).exp());factor=32*width/pi
                contrib={'number':factor*ps[2],'energy':factor*point(r).upper()*ps[3],
                         'entropy':factor*(ps[2]+a*ps[3])}
                for name in sums:sums[name]+=contrib[name]
                panels.append({'left_fraction':str(l),'right_fraction':str(r),'left':enc(point(l)),
                    'right':enc(point(r)),'F_lower':enc(Fbound),'a_lower':enc(a),
                    'upper_sum_contributions':{name:enc(value) for name,value in contrib.items()}})
            angular={name:sums[name]+small[name]+large[name] for name in sums}
            total={name:angular[name]+retained_low[name]+retained_high[name] for name in sums}
            ratios={name:total[name]/denominators[name] for name in sums}
            rate_passes={name:bool(value.upper()<arb('1e-7').lower()) for name,value in ratios.items()}
            strong_energy_pass=bool(ratios['energy'].upper()<arb('1e-10').lower())
            passed=all(rate_passes.values()) and strong_energy_pass
            payload={'bits':bits,'t_panels':count,'numerical_normalization_targets_pass':bool(passed),
                'rate_1e7_targets_pass':rate_passes,'energy_1e10_target_pass':strong_energy_pass,
                'constants':{name:enc(value) for name,value in {'alpha':alpha,'C_alpha':C,'K':arb(K),
                    'tau':tau,'t_m':tm,'b':b,'q_0':q0,'Q':arb(3),'retained_kappa1_weight':arb(64),
                    'retained_kappa2_through8_weight':arb(2240),'retained_total_weight':arb(2304)}.items()},
                'retained_low':{k:enc(v) for k,v in retained_low.items()},'retained_high':{k:enc(v) for k,v in retained_high.items()},
                'angular_small':{k:enc(v) for k,v in small.items()},'angular_mid':{k:enc(v) for k,v in sums.items()},
                'angular_large':{k:enc(v) for k,v in large.items()},'angular_total':{k:enc(v) for k,v in angular.items()},
                'total_upper_bound_enclosures':{k:enc(v) for k,v in total.items()},
                'omitted_rate_enclosures':{k:['0',ceil_decimal(v)] for k,v in total.items()},
                'ratios_to_numerical_rates':{k:enc(v) for k,v in ratios.items()},
                'normalization_is_rigorous_lower_enclosure':False,'panels':panels}
            write(folder/f't_panels_{count}.json',payload)
            write(folder/'roots.json',root_report);write(folder/'actions.json',action_report)
            refinements.append({'t_panels':count,'upper_endpoints':{k:ceil_decimal(v) for k,v in total.items()},'targets_pass':bool(passed)})
            write(folder/'refinements.json',refinements)
            print(json.dumps({'bits':bits,'t_panels':count,'upper_bounds':refinements[-1]['upper_endpoints'],'targets_pass':bool(passed)}),flush=True)
            if passed:break
        return {'bits':bits,'controls_passed':True,'accepted_t_panels':count,
                'rate_1e7_targets_pass':rate_passes,'energy_1e10_target_pass':strong_energy_pass,
                'targets_pass':bool(passed),'upper_bounds':{k:ceil_decimal(v) for k,v in total.items()},
                'numerical_ratio_upper_bounds':{k:ceil_decimal(v) for k,v in ratios.items()}}
    counts=[];n=args.initial_panels
    while n<=args.max_panels:counts.append(n);n*=2
    if not counts:raise ValueError('Empty bounded refinement schedule')
    base=run_precision(128,counts)
    result={'scope':'Absolute omitted-rate upper certificates; ratios use numerical denominators, not admitted exact-rate lower enclosures.',
            'runtime':runtime,'python_flint_version':flint.__version__,'action_tolerance':args.action_tolerance,'base':base}
    if not args.controls_only:result['refined_precision']=run_precision(192,[base['accepted_t_panels']])
    write(args.output/'result.json',result)
    print(json.dumps({'saved':str(args.output/'result.json')}),flush=True)


if __name__=='__main__':main()
