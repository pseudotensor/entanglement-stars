import json,sys,os
for f in sys.argv[1:]:
    try:
        if os.path.getsize(f)>400e6: print(f,'BIG');continue
        d=json.load(open(f))
        a=d.get('args',{}) if isinstance(d,dict) else {}
        recs=d.get('records',[]) if isinstance(d,dict) else []
        T=[r.get('T_over_rs',r.get('t_over_rs')) for r in recs if isinstance(r,dict)]
        tr=[r.get('T_over_rs') for r in recs if isinstance(r,dict) and r.get('R_t') is not None]
        print(os.path.basename(f), d.get('script'), 'rs',a.get('rs'),'n',a.get('nsites'),'m',a.get('m'),'tau',a.get('tau'),'ell',a.get('ell'),'Tend_arg',a.get('T_over_rs'),'nrec',len(recs),'Tlast',T[-1] if T else None,'trapped',(tr[0],tr[-1],len(tr)) if tr else None)
    except Exception as e: print(f,'ERR',e)
