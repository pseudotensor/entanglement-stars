import json, os, collections
rows=[json.loads(l) for l in open('<internal>/census_v1.jsonl')]
print(len(rows))
g=collections.defaultdict(list)
for r in rows:
    b=os.path.basename(r['npz'])[:-len('_fields.npz')]
    g[b].append(r)
sel=[]
for b,rs in g.items():
    for r in rs:
        m=r.get('members',{})
        hasNv = 'N' in m and 'v' in m
        if r.get('root_T') is None: continue
        sel.append((b,r))
print('with root:',len(sel), 'tags', len(set(b for b,_ in sel)))
for b in sorted(set(b for b,_ in sel)):
    for r in [x for bb,x in sel if bb==b]:
        m=r.get('members',{}); a=r.get('args',{})
        print('%-95s n%s npzT %.3f-%.3f (%s) rec %s %.3f-%.3f root %.4f Rt %.1f unc %s minm %s nodes %s app %s | rs %s ns %s tau %s ell %s mix %s sf %s ds %s | %s %s' % (
            b[:95], r.get('npz_n'), r.get('npz_T0',-1), r.get('npz_T1',-1), r.get('npz_dT'), r.get('n_rec'), r.get('rec_T0',-1), r.get('rec_T1',-1),
            r['root_T'], r['root_Rt'], r.get('n_unconv'), round(r.get('min_m',9),3), m.get('N_nodes',['-'])[0], 'N_applied' in m,
            a.get('rs'), a.get('nsites'), a.get('tau'), a.get('ell'), a.get('mix'), a.get('site_fields'), a.get('dust_set'), r['size']//1000000, r['npz']))
