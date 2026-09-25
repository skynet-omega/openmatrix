"""Revision of owner_write_probe: LIVE consumed buffers, value-derived four-tree guard.
No historical trace files accepted. Reuses pinned observer/AST, not old run_probe.
CUDA path untested externally; --selftest exercises provenance and guard logic CPU.
"""
import ast, hashlib, importlib.util, inspect, json, math, struct, time, traceback, uuid
from contextlib import nullcontext
from pathlib import Path
import numpy as np
DONOR_SHA='8ffaf4676c17d2c1aa8130ec1899ee6b28c2a31f7bd191e360b25c1fae145333'
GROUPS={'operator','owners','events','boundary'}

def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def exact(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def load_donor(path):
    need(sha(path)==DONOR_SHA,'Original observer changed')
    s=importlib.util.spec_from_file_location('original_owner_probe',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fingerprint(value,cp=None,stream=None):
    """Hash all supplied leaves: types, shapes, contents; GPU pointer/strides too.
    Reject opaque mutable values, do not silently stringify them or accept 4 labels.
    NaN event sentinels hashed as bytes, not rewritten to zero.
    """
    leaves={};bytes_read=0
    def walk(v,path):
        nonlocal bytes_read
        if cp is not None and isinstance(v,cp.ndarray):
            meta=('cuda',v.dtype.str,v.shape,v.strides,int(v.data.ptr))
            v=cp.asnumpy(v,stream=stream);bytes_read+=v.nbytes
        else: meta=None
        if isinstance(v,np.ndarray):
            need(not v.dtype.hasobject,'Object array: '+path)
            raw=np.ascontiguousarray(v).tobytes()
            tag=repr(meta or ('numpy',v.dtype.str,v.shape)).encode()
        elif isinstance(v,np.generic): raw=v.tobytes();tag=('scalar:'+v.dtype.str).encode()
        elif v is None or type(v) in (str,bytes,bool,int,float):
            raw=struct.pack('>d',v) if type(v) is float else repr(v).encode();tag=type(v).__name__.encode()
        elif isinstance(v,dict):
            need(all(type(k) is str for k in v),'Non-string key: '+path)
            leaves[path]={'kind':'dict','keys':sorted(v)}
            for k in sorted(v):walk(v[k],path+'/'+k.replace('~','~0').replace('/','~1'))
            return
        elif isinstance(v,(tuple,list)):
            leaves[path]={'kind':type(v).__name__,'length':len(v)}
            for i,x in enumerate(v):walk(x,path+'/'+str(i))
            return
        else: raise TypeError('Unsupported value at '+path+': '+type(v).__name__)
        leaves[path]={'sha256':hashlib.sha256(tag+b'\0'+raw).hexdigest(),'bytes':len(raw)}
    walk(value,'')
    encoded=json.dumps(leaves,sort_keys=True,separators=(',',':')).encode()
    return {'sha256':hashlib.sha256(encoded).hexdigest(),'leaves':leaves,'gpu_read_bytes':bytes_read}

class FourGuard:
    def __init__(self,reader,cp=None,stream=None):
        self.reader,self.cp,self.stream=reader,cp,stream;self.reads=0;self.gpu_bytes=0
    def read(self):
        if self.stream is not None:self.stream.synchronize()
        with self.stream if self.stream is not None else nullcontext():
            trees=self.reader();need(set(trees)==GROUPS,'Need four complete value trees')
            need(all(isinstance(v,dict) and v for v in trees.values()),'Hashes/empty trees not admitted')
            result={k:fingerprint(trees[k],self.cp,self.stream) for k in sorted(trees)}
            self.reads+=1;self.gpu_bytes+=sum(v['gpu_read_bytes'] for v in result.values())
            return result
    @staticmethod
    def equal(a,b):return set(a)==set(b)==GROUPS and all(a[k]['sha256']==b[k]['sha256'] for k in GROUPS)
    @staticmethod
    def require(a,b):need(FourGuard.equal(a,b),'Effective state/version changed')

def make_guard(adapter,layout,evolving_state):
    """Concrete reader for this pinned layout. Actual owner serializer required.
    Kernel objects are immutable resources identified by identity+source files;
    all directly-read model attributes are hashed; unsupported values BLOCK.
    Includes full W, not only lesioned positions. No disk I/O in the hot graph.
    """
    import cupy as cp
    b=adapter.brain;stream=adapter.core.stream
    source=Path(layout.__file__).read_text();tree=ast.parse(source)
    names=sorted({x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)
                  and isinstance(x.value,ast.Name) and x.value.id=='self'})
    files={str(Path(layout.__file__).resolve()):sha(layout.__file__)}
    for name,m in vars(layout).items():
        if name.startswith('_m') and hasattr(m,'__file__'): files[str(Path(m.__file__).resolve())]=sha(m.__file__)
    globals_used=sorted({(n.value.id,n.attr) for n in ast.walk(tree)
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
        and n.value.id.startswith('_m')})
    def runtime_constants():
        out={}
        for mod,key in globals_used:
            value=getattr(vars(layout)[mod],key)
            if inspect.ismodule(value) or callable(value):
                out[mod+'.'+key]={'resource_type':type(value).__name__,'identity':id(value)}
            elif hasattr(value,'__dict__'):out[mod+'.'+key]=dict(vars(value))
            else:out[mod+'.'+key]=value
        return out
    def ports_data():
        # All raw projector fields, including device arrays. RawKernel has no scientific state.
        result={}
        for k,v in vars(adapter.ports).items():
            if callable(v) or type(v).__name__ in ('RawKernel','RawModule'):
                result[k]={'resource_type':type(v).__name__,'identity':id(v)}
            else:result[k]=v
        return result
    def read():
        attrs={}
        for name in names:
            if name=='statistics':continue # diagnostic counter, restored by oracle constructor
            if name=='brain':attrs[name]={'n_neurons':b.brain.n_neurons,'node_ids':b.brain.node_ids};continue
            if name=='_online_source':continue # full state and output below
            if name=='_online_ports':attrs[name]={'pn_row':b._online_ports.pn_row};continue
            if not hasattr(b,name):attrs[name]={'absent':True};continue
            v=getattr(b,name)
            attrs[name]={'resource_type':type(v).__name__,'identity':id(v)} if callable(v) else v
        wave=adapter.events.active
        need(wave is not None,'Missing active waveform')
        return {'operator':{'source_sha256':files,'direct_read_attributes':attrs,'runtime_constants':runtime_constants()},
          'owners':{'evolving':evolving_state(b),'pn':b._online_source.state_dict(),
                    'pn_output':adapter.host_read(),'rates':b.brain.rates,
                    'brain_time_ns':b.brain.time_ns,'rng':b.brain.rng.bit_generator.state},
          'events':{'rows':adapter.events.rows,'wave':dict(vars(wave)),'projector':ports_data()},
          'boundary':{k:getattr(adapter,k) for k in ('drive','light','pn','boundary','held')}}
    return FourGuard(read,cp,stream)

class LiveWindow:
    """Single-use witness tied to the executing core and fresh device capture count."""
    def __init__(self,run_id,core_id,start_ns,phase,before):
        need(phase in ('accepted','predictor') and type(start_ns) is int,'Window identity')
        self.run_id=run_id;self.core_id=core_id;self.start_ns=start_ns;self.phase=phase
        self.before=before;self.sealed=False
    def seal(self,run_id,core_id,after):
        need(not self.sealed and (run_id,core_id)==(self.run_id,self.core_id),'Foreign/reused witness')
        FourGuard.require(self.before,after);self.sealed=True

def run_tap(adapter,reference,buffers,witness,guard,out,donor,global_limit):
    import cupy as cp
    import gpu_coefficient_layout as layout
    from effective_oracle import EffectiveOracle
    stream=adapter.core.stream;report={'status':'STARTED','new_oracle_launches':0,
       'new_warmups_completed':None,'organism_reexecuted_by_probe':False,'speed_admission':False}
    out.mkdir(parents=True,exist_ok=False)
    def deadline():need(time.monotonic()<global_limit,'120s observer budget')
    def host(v):
        with stream:return cp.asnumpy(v,stream=stream).copy()
    try:
        need(sha(layout.__file__)==donor.LAYOUT_SHA and sha(inspect.getfile(EffectiveOracle))==donor.ORACLE_SHA,'Source changed')
        tag,count,ts,starts,steps,fractions,xs,aa,rr=buffers
        stream.synchronize();m=int(host(count)[0])
        need(witness.sealed and int(host(tag)[0])==1 and 3<=m<=xs.shape[0],'Live capture incomplete')
        chosen=(2,m-1);need(chosen[0]!=chosen[1],'Need two distinct consumed queries')
        base=guard.read();cases=[]
        for k in chosen:
            case={key:host(v[k]) for key,v in zip(('z','target','rate'),(xs,aa,rr))}
            case.update(t=float(host(ts[k])),trial_start=float(host(starts[k])),h=float(host(steps[k])),fraction=float(host(fractions[k])),index=k)
            need(all(v.dtype==np.float64 and v.shape==(adapter.core.n,) and np.isfinite(v).all() for v in (case['z'],case['target'],case['rate'])),'Consumed layout/nonfinite')
            need(0<=case['t']<=125e-6,'Time outside captured block')
            cases.append(case)
            np.savez_compressed(out/f'consumed_{k}.npz',**case)
        need(reference.adapter is adapter,'Foreign original oracle')
        integrated=host(adapter.core.x) # held final integrator state, never replace with a query
        with stream:
            canary=cp.arange(32768,dtype=cp.float64);canary_host=host(canary)
        tap=donor.Tap(adapter);entry,text=donor.private_layout(layout,Path(layout.__file__).read_text(),tap)
        (out/'instrumented_layout.py').write_text(text)
        original=layout.assemble
        try:layout.assemble=entry;observed=EffectiveOracle(adapter)
        finally:layout.assemble=original
        stream.synchronize();need(tap.rounds==3,'Observer hook not captured');report['new_warmups_completed']=2
        FourGuard.require(base,guard.read());deadline()
        save(out/'WRITERS.json',{'sites':tap.labels,'kernels':tap.kernels,'rows':tap.rows.tolist(),
            'semantics':'set from Python AST; add/transform for opaque kernels are SOURCE LABELS, not inferred from differences'})
        save(out/'GUARD_AT_CAPTURE.json',witness.before);checks=[];first=None
        for occurrence,which in enumerate((0,1,0)):
            deadline();c=cases[which];vbefore=guard.read();FourGuard.require(base,vbefore)
            with stream:
                x=cp.asarray(c['z']);xcopy=x.copy();ref=reference.query(x,c['t'])
                report['new_oracle_launches']+=1;ref=[host(v) for v in ref]
                values=observed.query(x,c['t']);report['new_oracle_launches']+=1
                values=[host(v) for v in values]
            snapshots=tap.read();vafter=guard.read()
            np.savez_compressed(out/f'query_{occurrence}.npz',
                reference_z=ref[0],reference_target=ref[1],reference_rate=ref[2],
                tapped_z=values[0],tapped_target=values[1],tapped_rate=values[2],writes=snapshots)
            FourGuard.require(vbefore,vafter)
            check={'occurrence':occurrence,'captured_query':c['index'],'t_hex':c['t'].hex(),
              'reference_matches_consumed':[exact(a,c[k]) for a,k in zip(ref,('z','target','rate'))],
              'tap_matches_reference':[exact(a,b) for a,b in zip(values,ref)],
              'versions_before':vbefore,'versions_after':vafter}
            checks.append(check);save(out/'CHECKS.json',checks)
            need(all(check['reference_matches_consumed']) and all(check['tap_matches_reference']),'Identity failed; evidence saved')
            need(exact(host(x),host(xcopy)) and exact(host(canary),canary_host),'Input/canary changed')
            need(exact(host(adapter.core.x),integrated),'Observer changed integrated state')
            if occurrence==0:first=values+[snapshots]
            if occurrence==2:need(all(exact(a,b) for a,b in zip(first,values+[snapshots])),'A/B/A writes differ')
            deadline()
        report.update(status='SAME_RUN_WRITER_IDENTITY_ONLY',run_id=witness.run_id,
            epoch_start_ns=witness.start_ns,phase=witness.phase,consumed_indices=list(chosen),
            full_operator_executions=8,base_edge_visit_lower_bound=8*int(adapter.brain.cuda['indices'].size),
            specialized_edge_visits='NOT_COUNTED: no work-gate admission',
            capture_runs='No historical trace imported',quantitative_speed_gate=False)
    except BaseException:
        report.update(status='FAILED_RETAINED',error=traceback.format_exc());raise
    finally:
        report.update(code_sha256=sha(__file__),guard_reads=guard.reads,guard_D2H_bytes=guard.gpu_bytes)
        save(out/'RESULT.json',report)

def attach(adapter,reference,buffers,out,evolving_state,donor_path,invalidate):
    """Install on the fresh core AFTER reference construction, BEFORE its first advance.
    Uses buffers from probe_kc_apl_same_run.py; call in its build(drive,light).
    No lesion in this revision. Exactly one accepted block; restores advance on exit.
    """
    import cupy as cp
    import gpu_coefficient_layout as layout
    donor=load_donor(donor_path);need(sha(layout.__file__)==donor.LAYOUT_SHA,'Layout')
    guard=make_guard(adapter,layout,evolving_state);g=adapter.core;original=g.advance
    need(callable(invalidate),'Mandatory failed-session invalidator')
    identity=uuid.uuid4().hex;used=False;invalid=False
    def advance(*args,**kwargs):
        nonlocal used,invalid
        try:
            need(not invalid,'Failed observer worker cannot continue')
            g.stream.synchronize()
            with g.stream:epoch=int(cp.asnumpy(buffers[0],stream=g.stream)[0])
            if epoch!=1:return original(*args,**kwargs)
            need(not used and args and int(args[0])==125000,'Fresh 125us window required')
            frame=inspect.currentframe();phase=None
            try:
                while frame is not None:
                    if frame.f_code.co_name=='cns' and 'accepted' in frame.f_locals:
                        phase='accepted' if frame.f_locals['accepted'] else 'predictor';break
                    frame=frame.f_back
            finally:del frame
            need(phase=='accepted','Phase not established from actual caller')
            with g.stream:count=int(cp.asnumpy(buffers[1],stream=g.stream)[0])
            need(count==0,'Capture count reused from another block')
            used=True;deadline=time.monotonic()+120
            window=LiveWindow(identity,id(g),int(adapter.brain.time_ns),phase,guard.read())
            result=original(*args,**kwargs)
            # This is INSIDE g.advance: b.state/time have not yet been published.
            window.seal(identity,id(g),guard.read())
            run_tap(adapter,reference,buffers,window,guard,Path(out),donor,deadline)
            return result
        except BaseException:
            invalid=True
            invalidate()  # Must forbid snapshots/continued use of the real failed owner.
            raise
    g.advance=advance
    def restore():g.advance=original
    return restore

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true',required=True);a=p.parse_args()
    tree={k:{'array':np.array([1.,-0.]),'time':0} for k in GROUPS}
    guard=FourGuard(lambda:tree);v0=guard.read();blocked=0
    for k in GROUPS:
        tree[k]['array'][0]=2.;v1=guard.read()
        try:FourGuard.require(v0,v1)
        except ValueError:blocked+=1
        tree[k]['array'][0]=1.
    FourGuard.require(v0,guard.read())
    witness=LiveWindow('run-A',1,0,'accepted',v0)
    try:witness.seal('run-B',1,v0)
    except ValueError:blocked+=1
    witness.seal('run-A',1,v0)
    try:witness.seal('run-A',1,v0)
    except ValueError:blocked+=1
    try:FourGuard(lambda:{k:'a'*64 for k in GROUPS}).read()
    except ValueError:blocked+=1
    tree['events']['new_field']=0
    try:FourGuard.require(v0,guard.read())
    except ValueError:blocked+=1
    need(blocked==8,'Missing negative control')
    need(not exact(np.array([0.]),np.array([-0.])), 'Signed zero comparison')
    print(json.dumps({'CPU_fixture':'PASS','rejected':blocked,'CUDA_executed':False,'organism_executed':False,'code_sha256':sha(__file__)}))
