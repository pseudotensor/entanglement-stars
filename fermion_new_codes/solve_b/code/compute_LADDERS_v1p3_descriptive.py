"""Only the descriptive models printed in the assigned ladder reports."""
import argparse
import json
from pathlib import Path
import mpmath as mp
import numpy as np


ORDINATES=['T_f','R_t_over_rs','R_t_over_2gE']
SIZE=['RS32_s4','P64_s4','S4_128_s4']
WIDTH=['P64_s2','P64_s4','P64_s8','S4_128_s4','S4_128_s8','S4_128_s16']
MASS=['M3b_s4','P64_s4','M3c_s4_nk']
RANK=['RS32_s4','RS32_s4_tau1','RS32_s2','P64_s4','P64_s2','P64_s8',
      'P64_s4_tau4','P64_s4_tau2','S4_128_s8','S4_128_s16']


def pack(value):
    if isinstance(value,mp.matrix):
        return [[pack(value[i,j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value,(mp.mpf,mp.mpc)):
        return mp.nstr(value,mp.mp.dps)
    if isinstance(value,dict): return {k:pack(v) for k,v in value.items()}
    if isinstance(value,list): return [pack(v) for v in value]
    return value


def ols(X,y,predict=None):
    X=mp.matrix(X); y=mp.matrix(y)
    coeff=mp.lu_solve(X,y) if X.rows==X.cols else mp.qr_solve(X,y)[0]
    fitted=X*coeff; residual=y-fitted
    ss=mp.fsum(v*v for v in residual)
    normal=X.T*residual
    singular=np.linalg.svd(np.array(X.tolist(),dtype=float),compute_uv=False)
    threshold=np.finfo(float).eps*max(X.rows,X.cols)*singular[0]
    rank=int(np.sum(singular>threshold))
    answer=dict(design=X,ordinate=y,coefficients=list(coeff),fitted=list(fitted),residuals=list(residual),
                rank=rank,dof=X.rows-rank,sum_squares=ss,rms=mp.sqrt(ss/X.rows),
                dof_rms=mp.sqrt(ss/(X.rows-rank)) if X.rows>rank else None,
                max_residual=max(abs(v) for v in residual),
                normal_equation_residual=max(abs(v) for v in normal),
                singular_values=singular.tolist(),rank_threshold=float(threshold),
                binary64_coefficients=np.linalg.lstsq(np.array(X.tolist(),dtype=float),
                    np.array(list(y),dtype=float),rcond=None)[0].tolist())
    if predict:
        answer['predictions']={name:mp.fsum(a*b for a,b in zip(row,coeff)) for name,row in predict.items()}
    return answer


def power(xs,ys):
    order=sorted(range(3),key=lambda i:xs[i]); x=[xs[i] for i in order]; y=[ys[i] for i in order]
    a,b=mp.log(x[1]/x[0]),mp.log(x[2]/x[1]); target=(y[1]-y[0])/(y[2]-y[1])
    def ratio(p):
        return a/b if p==0 else mp.expm1(p*a)/(mp.exp(p*a)*mp.expm1(p*b))
    def f(p): return ratio(p)-target
    brackets=[]
    for lo in range(-128,128):
        hi=lo+1
        if f(lo)==0: brackets.append((mp.mpf(lo),mp.mpf(lo)))
        elif f(lo)*f(hi)<0: brackets.append((mp.mpf(lo),mp.mpf(hi)))
    if len(brackets)!=1: raise ValueError('power exponent lacks unique bracket')
    lo,hi=brackets[0]
    for _ in range(mp.mp.dps*4+32):
        middle=(lo+hi)/2
        if f(middle)==0: lo=hi=middle; break
        if f(lo)*f(middle)<=0: hi=middle
        else: lo=middle
    p=(lo+hi)/2
    if abs(p)<mp.mpf(10)**(-mp.mp.dps//2):
        raise ValueError('degenerate zero exponent/logarithmic limit')
    A=(y[1]-y[0])/(x[1]**p-x[0]**p); Y0=y[0]-A*x[0]**p
    return dict(x=x,y=y,order=order,p=p,A=A,Y0=Y0,exponent_bracket=[lo,hi],
                reconstruction_residuals=[value-(Y0+A*base**p) for base,value in zip(x,y)])


def high_precision(members,precision):
    mp.mp.dps=precision
    def scalar(name,key): return mp.mpf(members[name]['scalars'][key])
    def ordinate(name,key):
        row=members[name]['formation']['row']; primitive=row['primitives']
        if key=='T_f': return mp.mpf(primitive['T'])/scalar(name,'rs')
        if key=='R_t_over_rs': return mp.mpf(primitive['R_t'])/scalar(name,'rs')
        return mp.mpf(primitive['R_t'])/(2*scalar(name,'g')*mp.mpf(primitive['E']))
    def iso(name):
        return ((9*mp.pi*scalar(name,'NB')/32)**(mp.mpf(1)/3)/
                (scalar(name,'m')*scalar(name,'R0')))**2
    def own(name): return scalar(name,'f_kin')/mp.mpf('0.3')
    def design(name,kind):
        rs=scalar(name,'rs')
        if kind=='size': return [mp.mpf(1),rs**(-mp.mpf(1)/3)]
        if kind=='mass_iso': return [mp.mpf(1),iso(name)]
        if kind=='mass_own': return [mp.mpf(1),own(name)]
        z,h=mp.log(1/rs),mp.log(scalar(name,'sigma'))
        return [mp.mpf(1),z,h,z*h]
    definitions=[('size_certified',SIZE[:2],'size',[]),('size_labelled',SIZE,'size',['S4_16_s4']),
                 ('width_certified',[n for n in WIDTH if n!='S4_128_s4'],'width',['S4_128_s4']),
                 ('width_labelled',WIDTH,'width',[]),('mass_isotropic',MASS,'mass_iso',['M3a_s4','M3d_s4']),
                 ('mass_own',MASS,'mass_own',['M3a_s4','M3d_s4'])]
    fits={}
    for title,names,kind,prediction_names in definitions:
        fits[title]=dict(members=names,certified=[members[n]['formation']['row']['certified'] for n in names],
                         variables=[design(n,kind)[1:] for n in names],ordinates={})
        for key in ORDINATES:
            fits[title]['ordinates'][key]=ols([design(n,kind) for n in names],[ordinate(n,key) for n in names],
                                            {n:design(n,kind) for n in prediction_names})
    powers={}
    for title,names,xs in [('size',SIZE,[1/scalar(n,'rs') for n in SIZE]),
                           ('mass_isotropic',MASS,[iso(n) for n in MASS])]:
        powers[title]={key:power(xs,[ordinate(n,key) for n in names]) for key in ORDINATES}
    rank_tables={}
    for title,names in [('certified',RANK),('every_formation_record',RANK[:-2]+['S4_128_s4']+RANK[-2:])]:
        matrix=[]
        for n in names:
            x=64/scalar(n,'rs'); z=x**(mp.mpf(4)/3)
            h=(16*scalar(n,'sigma')/scalar(n,'rs'))**2; j=8*scalar(n,'tau')/scalar(n,'rs')
            matrix.append([1,z,x*x,h,z*h,j,j*j,z*j])
        singular=mp.svd(mp.matrix(matrix),compute_uv=False)
        threshold=mp.eps*max(len(names),8)*singular[0]
        rank=sum(value>threshold for value in singular)
        rank_tables[title]=dict(members=names,design=mp.matrix(matrix),singular_values=list(singular),
                                rows=len(names),rank=rank,rank_threshold=threshold,
                                expected_algebraic_rank=8,rows_minus_rank=len(names)-rank)
    clock=fits['width_certified']['ordinates']['T_f']
    exact=[mp.mpf(245)/12,mp.mpf(71)/(48*mp.log(2)),mp.mpf(3)/(8*mp.log(2)),mp.mpf(0)]
    width_error=max(abs(a-b) for a,b in zip(clock['coefficients'],exact))
    if width_error>mp.mpf(10)**(-precision+10):
        raise RuntimeError('exact certified width-clock control failed')
    return dict(fits=fits,powers=powers,rank_tables=rank_tables,
                exact_width_clock_control=dict(expected=exact,max_error=width_error,passed=True))


def controls():
    mp.mp.dps=80
    line=ols([[1,0],[1,1],[1,2]],[1,2,5],{'three':[1,3]})
    exact=[mp.mpf(2)/3,mp.mpf(2)]
    power_answer=power([mp.mpf(1),mp.mpf(2),mp.mpf(4)],[mp.mpf(3),mp.mpf(9),mp.mpf(33)])
    tests=[dict(name='OLS independent manufactured line',passed=max(abs(a-b) for a,b in zip(line['coefficients'],exact))<mp.mpf('1e-70')),
           dict(name='OLS residual degrees of freedom',passed=line['dof']==1),
           dict(name='OLS manufactured prediction',passed=abs(line['predictions']['three']-mp.mpf(20)/3)<mp.mpf('1e-70')),
           dict(name='three point exact power',passed=max(abs(power_answer[k]-v) for k,v in [('p',2),('A',2),('Y0',1)])<mp.mpf('1e-70'))]
    if not all(t['passed'] for t in tests): raise RuntimeError('descriptive controls failed')
    return dict(tests=tests,line=line,power=power_answer)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    out=Path(args.output); members=json.loads((out/'scalar_tables.json').read_text())
    known=pack(controls()); (out/'descriptive_known_answers.json').write_text(json.dumps(known,indent=2)+'\n')
    result={}
    for precision in [80,160]:
        result[str(precision)]=pack(high_precision(members,precision))
        (out/'descriptive_arithmetic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(precision=precision,completed=True)),flush=True)


if __name__=='__main__': main()
