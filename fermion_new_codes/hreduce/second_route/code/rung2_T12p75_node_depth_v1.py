"""One stored-node N-|v| minimum; exact dyadic arithmetic, no dense-interpolant claim."""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import signal
import time
import zipfile

ROOT=Path('<internal>/rung2_T12p75_node_depth_v1')
OWN=Path('<internal>/member_06.json')
ORIGINAL=Path('<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json')
FIELDS=ORIGINAL.with_name(ORIGINAL.stem+'_fields.npz')
DECODER=OWN.with_name('audit_saved_roots_v1.py')
PINS={OWN:('5d5e8ca183ec492d7580b732c09cb7bb606b02385b391c4482cae977a5810c38',3553788),
 ORIGINAL:('a8faf679cbd196be868b76a2cb039d06b55a19747ad6216b8b6cc38dd2094a23',7014125),
 FIELDS:('da87d09a2647737cf8530ded7ac6cdd080f705a5820d42969c485666d271e531',38593841)}
DECODER_SHA='dd437a92695c8f3069421d342a1eca07d1d04914be13dac37622d349aa9d5ebc'
TAG='c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d'
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
np=None


class Number(str):
    """An unmodified JSON number token, distinct from a JSON string."""

def encode(value):
    if isinstance(value, Number):
        # Input constants are rejected by the parser; tokens remain valid JSON.
        return str(value)
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(str(k)) + ':' + encode(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(encode(v) for v in value) + ']'
    return json.dumps(value, allow_nan=False)

def reject_constant(token):
    raise ValueError('Nonfinite JSON constant: ' + token)

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result

def rational(token):
    if not isinstance(token, Number):
        raise ValueError('Required primary input is not a JSON number token')
    return Fraction(str(token))

def identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)

class NumericArrayUnpickler(pickle.Unpickler):
    """Only NumPy array construction for historical numeric object arrays."""
    def find_class(self,module,name):
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np.core.multiarray,name)
        raise pickle.UnpicklingError('Unapproved array class: '+module+'.'+name)

def nodes_member(archive,key):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported NPY version')
        if not dtype.hasobject:
            if dtype.kind not in 'fiu':
                raise ValueError('Non-numeric node dtype')
            raw=stream.read()
            if len(raw)!=math.prod(shape)*dtype.itemsize:
                raise ValueError('Numeric node payload length differs')
            return np.frombuffer(raw,dtype=dtype).reshape(shape,order='F' if order else 'C').astype(float)
        value=NumericArrayUnpickler(stream).load()
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype:
            raise ValueError('Node-array header/payload disagreement')
        for x in value.flat:
            if not isinstance(x,(int,float,np.integer,np.floating)):
                raise ValueError('Non-numeric node object')
        return np.asarray(value,dtype=float)


@contextmanager
def protected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def hash_descriptor(stream):
    h=hashlib.sha256();stream.seek(0)
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    stream.seek(0);return h.hexdigest()


def load_json(path,sources):
    with path.open('rb') as stream:
        before=identity(os.fstat(stream.fileno()));raw=stream.read();after=identity(os.fstat(stream.fileno()))
    h=hashlib.sha256(raw).hexdigest()
    if before!=after or identity(path.stat())!=after or (h,len(raw))!=PINS[path]:raise ValueError('JSON input identity/pin failed')
    sources.append(dict(path=str(path),sha256=h,**after))
    return json.loads(raw,parse_int=Number,parse_float=Number,parse_constant=reject_constant,object_pairs_hook=no_duplicates)


def historical_matches(historical,current,path):
    for key in ('device','inode','size','mtime_ns'):
        if key in historical:
            value=rational(historical[key])
            if value.denominator!=1 or value.numerator!=current[key]:
                raise ValueError('Historical input identity differs: '+str(path)+':'+key)
    if 'resolved_path' in historical and historical['resolved_path']!=str(path.resolve()):
        raise ValueError('Historical resolved input path differs: '+str(path))


def atomic_json(name,value):
    with protected():
        temp=ROOT/(name+'.tmp')
        with temp.open('x') as stream:
            stream.write(encode(value)+'\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/name)


def atomic_npz(name,arrays):
    with protected():
        temp=ROOT/(name+'.tmp')
        with temp.open('xb') as stream:
            np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/name)
        return dict(path=str(ROOT/name),size=(ROOT/name).stat().st_size,sha256=digest(ROOT/name),
            arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})


def join(records,own=False):
    matches=[(i,row) for i,row in enumerate(records) if rational(row['T'])==816 and rational(row['T_over_rs'])==Fraction(51,4)]
    if len(records)!=321 or len(matches)!=1 or matches[0][0]!=102:raise ValueError('Unique exact JSON clock/index failed')
    row=matches[0][1]
    if own and (rational(row['index'])!=102 or rational(row['field_index'])!=102):raise ValueError('Own field index failed')
    return row


def exact_vector(N,v):
    return [Fraction(*float(a).as_integer_ratio())-abs(Fraction(*float(b).as_integer_ratio())) for a,b in zip(N,v)]


def minimum(values):
    value=min(values);indices=[i for i,x in enumerate(values) if x==value]
    return value,indices


def pack_fraction(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def vector_checks(r,N,v,length=None):
    if any(x.dtype!=np.dtype('float64') or x.ndim!=1 for x in (r,N,v)) or N.shape!=r.shape or v.shape!=r.shape or len(r)<2 or length is not None and len(r)!=length:
        raise ValueError('Node vector shape/dtype mismatch')
    if not all(np.isfinite(x).all() for x in (r,N,v)) or not np.all(np.diff(r)>0):raise ValueError('Nonfinite or unordered node vectors')


def fixtures(report,save):
    report['known_answers']=[]
    cases=(((1.,1.,1.),(0.,0.,0.),Fraction(1),[0,1,2],[Fraction(1),Fraction(1),Fraction(1)]),
           ((1.,1.,1.),(-.25,1.5,-.5),Fraction(-1,2),[1],[Fraction(3,4),Fraction(-1,2),Fraction(1,2)]),
           ((2.,2.),(-1.,1.),Fraction(1),[0,1],[Fraction(1),Fraction(1)]))
    for index,(n,v,expected,ties,expected_vector) in enumerate(cases):
        with protected():
            R=np.arange(len(n),dtype='float64');N=np.array(n,dtype='float64');V=np.array(v,dtype='float64')
            vector_checks(R,N,V)
            values=exact_vector(N,V);actual,indices=minimum(values)
            row=dict(index=index,exact_vector=[pack_fraction(x) for x in values],minimum=pack_fraction(actual),minimizer_indices=indices,
                     expected_vector=[pack_fraction(x) for x in expected_vector],
                     passed=values==expected_vector and actual==expected and indices==ties)
            if index==2:
                row['PL_midpoint_depth']=float((N[0]+N[1])/2-abs((V[0]+V[1])/2))
                row['passed']=row['passed'] and row['PL_midpoint_depth']==2
            report['known_answers'].append(row);save()
            if not row['passed']:raise AssertionError('Fixed node arithmetic fixture failed')


def main():
    global np
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory only')
    names=('primary_record.json','selected_nodes.npz','node_depth.npz','rung2_T12p75_node_depth_result.json')
    if any((ROOT/n).exists() or (ROOT/(n+'.tmp')).exists() for n in names):raise FileExistsError('Existing allocation requires assessment; no retry')
    if min(shutil.disk_usage('/').free,shutil.disk_usage('/data').free)<100_000_000_000:raise RuntimeError('Disk reserve unavailable')
    resource.setrlimit(resource.RLIMIT_AS,(2**30,2**30));resource.setrlimit(resource.RLIMIT_CPU,(20,25))
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    os.sched_setaffinity(0,[min(os.sched_getaffinity(0))])
    start=time.monotonic();report=dict(status='incomplete',phase='inputs',sources=[],saved=[],validated=[],products={},
        scope='Exact stored-node depth and specified PL minimum only; no original dense-interpolant or field-error certificate',
        units=dict(T='T/a',r='r/a',N='dimensionless',v='dimensionless',depth='dimensionless'),
        remaining_physics=['Original dense interpolation/extremum and field error','Independent Gauss-row/history controls','Other named core controls'],
        resources=dict(wall_seconds=30,handled_wall_seconds=25,cpu_soft_seconds=20,cpu_hard_seconds=25,address_space_bytes=2**30,threads=1))
    def deadline(signum,frame):
        if report.get('deadline_seen'):return
        report['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
        raise TimeoutError('Fixed stored-node depth allocation')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(25)
    def save():
        required={'primary_record':'primary_record.json','nodes_r':'selected_nodes.npz:r','nodes_N':'selected_nodes.npz:N',
            'nodes_v':'selected_nodes.npz:v','exact_vector':'rung2_T12p75_node_depth_result.json:exact_vector','minimum':'rung2_T12p75_node_depth_result.json:minimum','display_npz':'node_depth.npz'}
        report['missing_outputs']=[v for k,v in required.items() if k not in report['saved']]
        report['missing_validations']=[k for k in ('primary_binding','NPZ_clock_metadata','node_vectors','known_answers','exact_vector','minimum','display_npz') if k not in report['validated']]
        report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)))
        atomic_json(names[3],report)
    selected={};field_stream=None
    try:
        own=load_json(OWN,report['sources']);original=load_json(ORIGINAL,report['sources'])
        field_stream=FIELDS.open('rb')
        before=identity(os.fstat(field_stream.fileno()));h=hash_descriptor(field_stream)
        if (h,before['size'])!=PINS[FIELDS] or identity(FIELDS.stat())!=before:raise ValueError('Fields pin/identity failed')
        report['sources'].append(dict(path=str(FIELDS),sha256=h,**before))
        row,record=join(own['records'],True),join(original['records'])
        with protected():
            primary=dict(sources=report['sources'],member=own['member'],source=own['source'],json_sha256=own['json_sha256'],
                fields_identity=own['fields_identity'],fields_sha256=own['fields_sha256'],record_index=102,field_index=102,T=row['T'],T_over_rs=row['T_over_rs'],
                arguments={k:original['args'][k] for k in ('m','g','rs','tau','ell','prep_ell','nsites','kappa_sea','reference','site_fields','mix','tag')},
                restart=original['head']['resume_slice_v34'],container_script=original['script'],record_engine=record['engine'],
                label=record['label'],record_label=record['record_label'],controls={k:record[k] for k in ('converged','unconverged','continuation','gauge','indicator','solve_v32','solver')},
                own_raw_gates=row['raw_gates'],accepted_stored_node_domain=row['roots']['stored_nodes']['domain'],units=report['units'])
            atomic_json(names[0],primary);report['primary']=primary;report['saved'].append('primary_record');save()
        if own['member']!='P64\\_s2 composite (321; 40)' or own['source']['path']!=str(ORIGINAL) or own['json_sha256']!=PINS[ORIGINAL][0] or own['fields_identity']['path']!=str(FIELDS) or own['fields_sha256']!=PINS[FIELDS][0]:raise ValueError('Own data linkage failed')
        actual_sources={item['path']:item for item in report['sources']}
        historical_matches(own['source'],actual_sources[str(ORIGINAL)],ORIGINAL)
        historical_matches(own['fields_identity'],actual_sources[str(FIELDS)],FIELDS)
        expected=dict(m=Fraction(1,5),g=Fraction(1,4),rs=64,tau=8,ell=2,prep_ell=4,nsites=2048,kappa_sea=10)
        if any(rational(original['args'][k])!=v for k,v in expected.items()) or any(original['args'][k]!=v for k,v in dict(reference='R2',site_fields='averaged',mix='newton',tag=TAG).items()):raise ValueError('Argument binding failed')
        restart=original['head']['resume_slice_v34']
        if original['script']!='engine_collapse_v34.py' or record['engine']!='engine_collapse_v25' or rational(restart['T_over_rs'])!=17 or rational(restart['record_index'])!=136 or restart['slice']!='<internal>/orbitals_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_T0017.00.npz':raise ValueError('Inherited prefix binding failed')
        if digest(DECODER)!=DECODER_SHA:raise ValueError('Owned decoder source pin failed')
        report['decoder']=dict(path=str(DECODER),sha256=DECODER_SHA,used_functions=['NumericArrayUnpickler','nodes_member'])
        report['validated'].append('primary_binding');save()
        import numpy as numpy
        np=numpy;report['numpy_version']=np.__version__
        with field_stream:
            if identity(os.fstat(field_stream.fileno()))!=before or identity(FIELDS.stat())!=before:raise ValueError('Fields changed before decoding')
            field_stream.seek(0)
            with zipfile.ZipFile(field_stream) as archive:
                def numeric(key):return np.load(io.BytesIO(archive.read(key+'.npy')),allow_pickle=False)
                clocks=numeric('T');metadata={k:numeric(k) for k in ('g','ell','tau')}
                if clocks.shape!=(321,) or not np.isfinite(clocks).all() or np.flatnonzero(clocks==816).tolist()!=[102]:raise ValueError('Unique NPZ clock failed')
                if any(a.shape!=() or not np.isfinite(a).all() or a.item()!=expected[k] for k,a in metadata.items()):raise ValueError('NPZ scalar metadata failed')
                report['npz_join']=dict(index=102,T=float(clocks[102]),clock_count=321,metadata={k:float(v) for k,v in metadata.items()})
                report['validated'].append('NPZ_clock_metadata');save()
                report['phase']='decoding_nodes'
                for key in ('r','N','v'):
                    name={'r':'r_nodes','N':'N_nodes','v':'v_nodes'}[key]
                    with archive.open(name+'.npy') as stream:
                        version=np.lib.format.read_magic(stream)
                        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                        elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                        else:raise ValueError('Unsupported node NPY version')
                    if shape!=(321,1025) or order or not dtype.hasobject:raise ValueError('Unexpected historical node header family')
                    all_rows=nodes_member(archive,name)
                    if all_rows.shape!=(321,1025) or all_rows.dtype!=np.dtype('float64'):raise ValueError('Decoded node shape/dtype failed')
                    with protected():
                        selected[key]=all_rows[102].copy();del all_rows
                        report['products'][names[1]]=atomic_npz(names[1],selected)
                        report['saved'].append('nodes_'+key);save()
            if identity(os.fstat(field_stream.fileno()))!=before or identity(FIELDS.stat())!=before:raise ValueError('Fields changed during decoding')
        R,N,V=selected['r'],selected['N'],selected['v'];vector_checks(R,N,V,1025)
        domain=primary['accepted_stored_node_domain']
        if len(domain)!=2 or R[0]!=0 or R[-1]!=2048 or [Fraction(*float(x).as_integer_ratio()) for x in (R[0],R[-1])]!=[rational(x) for x in domain]:raise ValueError('Stored-node domain endpoints failed')
        report['validated'].append('node_vectors');save()
        fixtures(report,save);report['validated'].append('known_answers')
        with protected():
            values=exact_vector(N,V)
            report['exact_vector']=[pack_fraction(x) for x in values];report['saved'].append('exact_vector');save()
        if any(x.denominator<=0 or math.gcd(x.numerator,x.denominator)!=1 for x in values):raise AssertionError('Exact vector rational control failed')
        report['validated'].append('exact_vector')
        with protected():
            depth,indices=minimum(values)
            report.update(minimum=pack_fraction(depth),minimizer_indices=indices,minimizer_radii=[float(R[i]) for i in indices],
                representative=dict(index=indices[0],radius=float(R[indices[0]])))
            report['saved'].append('minimum');save()
        if not indices or any((x==depth)!=(i in indices) or x<depth for i,x in enumerate(values)):raise AssertionError('Minimum/tie control failed')
        report['validated'].append('minimum')
        with protected():
            display=dict(depth=np.asarray([float(x) for x in values],dtype='float64'),minimum=np.asarray(float(depth),dtype='float64'),
                minimizer_indices=np.asarray(indices,dtype='int64'),minimizer_radii=R[indices].copy())
            report['products'][names[2]]=atomic_npz(names[2],display);report['saved'].append('display_npz');save()
        if not np.isfinite(display['depth']).all() or not np.isfinite(display['minimum']).all() or any(display['depth'][i]!=float(x) for i,x in enumerate(values)) or display['minimizer_indices'].tolist()!=indices or not np.array_equal(display['minimizer_radii'],R[indices]):raise AssertionError('Depth display output control failed')
        report['validated'].append('display_npz');report.update(status='completed_rung2_T12p75_stored_node_depth_only',phase='complete')
    except Exception as error:
        report.update(status='incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(25,25))
        if field_stream is not None and not field_stream.closed:field_stream.close()
        save()
    print(json.dumps(dict(status=report['status'],saved=report['saved'],validated=report['validated'])))
    if report['status']!='completed_rung2_T12p75_stored_node_depth_only':raise SystemExit(1)


if __name__=='__main__':main()
