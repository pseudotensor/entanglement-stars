import json, os, collections
rows=[json.loads(l) for l in open('<internal>/census_v1.jsonl')]
g=collections.defaultdict(list)
for r in rows:
    b=os.path.basename(r['npz'])[:-len('_fields.npz')][len('collapse_'):]
    g[b].append(r)
out=[]
for b,rs in g.items():
    best=max(rs,key=lambda r:(r.get('npz_n') or 0, r.get('npz_T1') or 0, 'resume_mirror' not in r['npz'], 'snap' not in r['npz']))
    best['ncopies']=len(rs); best['tag']=b
    out.append(best)
out.sort(key=lambda r:(r.get('args',{}).get('rs') or 0, r['tag']))
for r in out:
    m=r.get('members',{}); a=r.get('args',{})
    full = (r.get('npz_n')==r.get('n_rec'))
    print('%s|rs%s ns%s tau%s ell%s %s %s %s|npz %s %.3f-%.3f d%s|rec %s full%s|root %s Rt %s|unc %s|minm %s|nodes%s app%s|x%d|%s' % (
        r['tag'][:80], a.get('rs'), a.get('nsites'), a.get('tau'), a.get('ell'), a.get('mix'), (a.get('site_fields') or '')[:3], a.get('dust_set'),
        r.get('npz_n'), r.get('npz_T0',-1), r.get('npz_T1',-1), r.get('npz_dT'), r.get('n_rec'), int(full),
        r.get('root_T'), None if r.get('root_Rt') is None else round(r['root_Rt'],1), r.get('n_unconv'),
        None if r.get('min_m') is None else round(r['min_m'],3), (m.get('N_nodes') or ['-'])[0], int('N_applied' in m), r['ncopies'], r['npz']))
json.dump(out, open('<internal>/census_tags_v1.json','w'), default=float)
