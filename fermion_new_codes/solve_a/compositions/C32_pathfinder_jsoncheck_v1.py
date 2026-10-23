import json,sys
R='<internal>/compC'
P={'MEM0':R+'C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json','MEMJ':R+'C32_s4c80_pod1/collapse_C32_s4_ell4_tau2_twofield_v3.json',
'CTL0':R+'C32_s4off_pod1off/collapse_C32_s4_ell4_tau2_twofield_v3_off.json','CTL50':R+'C32_s4offc50_pod1off50/collapse_C32_s4_ell4_tau2_twofield_v3_off.json',
'CTL80':'<internal>/collapse_C32_s4_ell4_tau2_twofield_v3_off.json'}
D={}
for k,p in P.items():
    d=json.load(open(p)); D[k]=d; r=d['records']
    print(k,len(r),r[0]['T_over_rs'],r[-1]['T_over_rs'],d['args'].get('light_coupling'),d.get('script'),d['args'].get('kappa_sea'),d['args'].get('m'),d['args'].get('nsites'),d['args'].get('half_offset'), [x for x in d if x!='records'])
a,b=D['CTL50']['args'],D['CTL80']['args']
for k in sorted(set(a)|set(b)):
    if a.get(k,'<absent>')!=b.get(k,'<absent>'): print('argdiff c50 vs c80',k,a.get(k,'<absent>'),b.get(k,'<absent>'))
a=D['MEMJ']['args']
for k in sorted(set(a)|set(b)):
    if a.get(k,'<absent>')!=b.get(k,'<absent>'): print('argdiff memc80 vs ctlc80',k,a.get(k,'<absent>'),b.get(k,'<absent>'))
# shared records c50 vs c80 bitwise?
r5=D['CTL50']['records'];r8=D['CTL80']['records']
n=min(len(r5),len(r8)); nd=sum(1 for i in range(n) if json.dumps(r5[i],sort_keys=True)!=json.dumps(r8[i],sort_keys=True))
print('c50 vs c80 shared',n,'differ',nd)
idx=[i for i in range(n) if json.dumps(r5[i],sort_keys=True)!=json.dumps(r8[i],sort_keys=True)][:5]
for i in idx:
    x,y=r5[i],r8[i]; print(i,x['T_over_rs'],[k for k in set(x)|set(y) if json.dumps(x.get(k),sort_keys=True)!=json.dumps(y.get(k),sort_keys=True)][:10])
print('c80 rec keys diff vs c50 at last:', set(r8[-1])^set(r5[-1]))
