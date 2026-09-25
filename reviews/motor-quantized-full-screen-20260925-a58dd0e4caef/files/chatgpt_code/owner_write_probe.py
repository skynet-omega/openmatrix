"""Instrumentación privada del layout publicado; no integra ni publica estado.
Python solo construye el grafo: los marcadores son copias CUDA capturadas.
Comparación A/B/A contra EffectiveOracle intacto y target/rate consumidos guardados.
"""
import ast, hashlib, inspect, json, time, traceback
from pathlib import Path
import numpy as np
LAYOUT_SHA='2598415d108e5ab595351ad2f3a5a841f6212c999026981f6aa38278ca554f9f'
ORACLE_SHA='bc1ebd444d8e86288e8940ba71599daf68de7d4cd3c447c21d72f64933a1403c'
# Etiquetas del mapa publicado; el texto/antes/después conserva su alcance.
SEMANTICS={'_mi9_kernel':'declared target:add', '_gaba_kernel':'declared target:add',
 '_histamine_targets':'declared target/rate:transform', '_replace_coefficients':'declared target/rate:set',
 '_regional_current_kernel':'declared target:set', '_cvn7_kernel':'declared target/rate:set',
 '_retinal_kernel':'declared target/rate:set', '_orn_pn_kernel':'declared target:set',
 '_orn_terminal_kernel':'declared target:set', '_pvlp_kernel':'declared target:set',
 '_regional_axon_kernel':'suffix:new; no base-row write', 'kernel':'base target/rate:set'}
SNAP=r'''
extern "C" __global__ void snap(const double*z,const double*a,const double*r,
 const double*w,const int*rows,const long long*pos,int n,int m,int ts,int valid,double*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){int j=rows[i];out[i]=z[j];out[n+i]=z[ts+j];
  out[2*n+i]=valid?a[j]:0.;out[3*n+i]=valid?r[j]:0.;}
 if(i<m)out[4*n+i]=w[pos[i]];
}
'''
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def exact(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def host(cp,x,stream):return cp.asnumpy(x,stream=stream).copy()
def version(guard):
    d=guard();need(set(d)=={'operator','owners','events','boundary'},'Guard must fingerprint four complete trees')
    need(all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in d.values()),'Guard hashes required')
    return d

class Instrument(ast.NodeTransformer):
    def __init__(self,source):self.source=source;self.fn='';self.counter=0
    def tag(self,node,op):return self.fn+':'+str(node.lineno)+':'+op
    def mark(self,label,valid=True,pair=None):
        a=ast.Name(id='target',ctx=ast.Load()) if valid else ast.Constant(None)
        r=ast.Name(id='rate',ctx=ast.Load()) if valid else ast.Constant(None)
        if pair:
            a=ast.Subscript(ast.Name(pair,ast.Load()),ast.Constant(0),ast.Load())
            r=ast.Subscript(ast.Name(pair,ast.Load()),ast.Constant(1),ast.Load())
        return ast.Expr(ast.Call(ast.Attribute(ast.Name('__tap__',ast.Load()),'mark',ast.Load()),
             [ast.Constant(label),ast.Name('self',ast.Load()),ast.Name('state',ast.Load()),a,r],[]))
    def visit_FunctionDef(self,node):
        old=self.fn;self.fn=node.name;node=self.generic_visit(node)
        node.body.insert(0,self.mark(node.name+':enter',False));self.fn=old;return node
    def visit_Return(self,node):
        node=self.generic_visit(node);self.counter+=1;name='__probe_return_'+str(self.counter)
        return [ast.Assign([ast.Name(name,ast.Store())],node.value),
                self.mark(self.tag(node,'before_return'),pair=name),ast.Return(ast.Name(name,ast.Load()))]
    def assignment(self,node,op):
        target_nodes=node.targets if isinstance(node,ast.Assign) else [node.target]
        writes=sorted({x.value.id for y in target_nodes for x in ast.walk(y)
                     if isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id in ('target','rate')})
        if not writes:return self.generic_visit(node)
        label=self.tag(node,op+'/'+','.join(writes));node=self.generic_visit(node)
        return [self.mark(label+'/before'),node,self.mark(label+'/after')]
    def visit_Assign(self,node):return self.assignment(node,'set')
    def visit_AugAssign(self,node):return self.assignment(node,'aug:'+type(node.op).__name__)
    def visit_Expr(self,node):
        v=node.value
        if not isinstance(v,ast.Call) or not isinstance(v.func,ast.Attribute):return self.generic_visit(node)
        attr=v.func.attr
        if attr not in SEMANTICS:return self.generic_visit(node)
        label=self.tag(node,SEMANTICS[attr]);site=self.tag(node,attr)
        # Misma llamada y mismos argumentos. Solo cuenta durante construcción.
        v=ast.Call(ast.Attribute(ast.Name('__tap__',ast.Load()),'kernel',ast.Load()),
                   [ast.Constant(site),v.func]+v.args,v.keywords)
        nodes=[ast.Expr(v),self.mark(label+'/after')]
        if attr!='kernel':nodes.insert(0,self.mark(label+'/before'))
        return nodes

def private_layout(module,source,tap):
    ns=dict(vars(module));ns['__tap__']=tap
    tree=ast.parse(source)
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
    need(len(funcs)==17,'Unexpected layout functions')
    instrumented=ast.fix_missing_locations(Instrument(source).visit(ast.Module(body=funcs,type_ignores=[])))
    exec(compile(instrumented,'<private_instrumented_layout>','exec'),ns)
    def entry(*a,**kw):
        tap.begin();r=ns['_OrnPeripheralTerminalMixin'](*a,**kw);tap.finish();return r
    return entry,ast.unparse(instrumented)

class Tap:
    def __init__(self,adapter):
        import cupy as cp
        self.cp=cp;self.adapter=adapter;self.brain=adapter.brain;self.stream=adapter.core.stream
        self.labels=[];self.buffers=[];self.kernels=[];self.rounds=0;self.i=0;self.ki=0
        b=self.brain
        with self.stream:
            raw=host(cp,b._dynamic_gpu_rows,self.stream)
            need(raw.ndim==1 and raw.dtype.kind in 'iu' and len(raw)==4064 and len(np.unique(raw))==4064 and np.all((raw>=0)&(raw<b.brain.n_neurons)),'Wrong dynamic rows')
            self.rows=np.sort(raw).astype(np.int32)
            p=[host(cp,getattr(b,k),self.stream).reshape(-1) for k in ('_apl_gpu_positions','_general_positions') if hasattr(b,k)]
            rawp=np.concatenate(p) if p else np.empty(0,np.int64)
            need(rawp.dtype.kind in 'iu' and np.all((rawp>=0)&(rawp<b.cuda['weights'].size)),'Weight tap indices')
            self.positions=np.unique(rawp).astype(np.int64)
            self.n=len(self.rows);self.m=len(self.positions);self.ts=int(b.transmission_start)
            need(self.m<=20000,'Weight tap capacity')
            self.drows=cp.asarray(self.rows);self.dpos=cp.asarray(self.positions)
            self.snap=cp.RawKernel(SNAP,'snap',options=('--std=c++11','--fmad=false'))
            self.snap.compile()
        self.stream.synchronize()
    def begin(self):self.i=0;self.ki=0
    def finish(self):
        need(self.i==len(self.labels) and self.ki==len(self.kernels),'Captured branch/order changed')
        self.rounds+=1
    def kernel(self,site,fn,*args,**kw):
        if self.rounds==0:self.kernels.append(site)
        else:need(self.ki<len(self.kernels) and self.kernels[self.ki]==site,'Kernel order changed')
        self.ki+=1;return fn(*args,**kw)
    def mark(self,label,brain,z,a,r):
        cp=self.cp;need(brain is self.brain,'Different brain owner')
        valid=a is not None
        need(z.dtype==cp.float64 and z.ndim==1 and z.size>self.ts+int(self.rows[-1]) and z.flags.c_contiguous,'State view')
        if valid:need(all(v.dtype==cp.float64 and v.ndim==1 and v.size>int(self.rows[-1]) and v.flags.c_contiguous for v in (a,r)),'Coefficient view')
        meta={'site':label,'target_rate_valid':valid}
        if self.rounds==0:
            need((len(self.buffers)+1)*(4*self.n+self.m)*8<=256*1024**2,'Observer memory cap')
            self.labels.append(meta);self.buffers.append(cp.empty(4*self.n+self.m,dtype=cp.float64))
        else:need(self.i<len(self.labels) and self.labels[self.i]==meta,'Marker order changed')
        self.snap(((max(self.n,self.m)+255)//256,),(256,),
           (z,a if valid else z,r if valid else z,brain.cuda['weights'],self.drows,self.dpos,
            np.int32(self.n),np.int32(self.m),np.int32(self.ts),np.int32(valid),self.buffers[self.i]),stream=self.stream)
        self.i+=1
    def read(self):
        self.stream.synchronize()
        with self.stream:return host(self.cp,self.cp.stack(self.buffers),self.stream)

def run_probe(adapter,reference,cases,guard,out,kernel_edge_visits=None):
    """cases: EXACTAMENTE A/B, cada uno {z,t,target,rate,phase,global_start_ns}.
    z/target/rate son arrays NumPy FP64 de la captura completa, no la cápsula overlay.
    reference es EffectiveOracle YA construido sobre el mismo adapter.
    guard() da hashes efectivos completos; no debe incluir scratch privado de oráculos.
    kernel_edge_visits: costes auditados por etiqueta; faltantes se declaran DESCONOCIDOS.
    """
    import cupy as cp
    import gpu_coefficient_layout as layout
    from effective_oracle import EffectiveOracle
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report={'status':'STARTED','organism_executed':False,'speed_admission':False,'queries':[],
            'query_launch_attempts':0,'query_sync_completed':0,'warmup_completed':None}
    save(out/'PLAN.json',{'queries':['A','B','A'],'reference_and_tap':True,'wall_s_max':120,
        'additional_snapshot_bytes_max':256*1024**2,'layout_sha256':LAYOUT_SHA,
        'oracle_sha256':ORACLE_SHA,'validation':'exact bytes, no changed tolerances'})
    observed=None;tap=None
    def deadline():need(time.monotonic()-start<120,'120s including construction/verification')
    try:
        need(sha(layout.__file__)==LAYOUT_SHA,'Layout hash differs')
        need(sha(inspect.getfile(EffectiveOracle))==ORACLE_SHA,'Oracle source differs')
        need(reference.adapter is adapter and adapter.core is not None,'Wrong original oracle')
        need(len(cases)==2 and cases[0]['phase'] in ('predictor','accepted') and cases[0]['phase']==cases[1]['phase'] and cases[0]['global_start_ns']==cases[1]['global_start_ns'],'Same epoch/phase required')
        for c in cases:
            need(np.isfinite(c['t']) and c['t']>=0,'Local time')
            for k in ('z','target','rate'):
                need(isinstance(c[k],np.ndarray) and c[k].dtype==np.float64 and c[k].shape==(adapter.core.n,) and np.isfinite(c[k]).all(),'Capture '+k)
        stream=adapter.core.stream;stream.synchronize();v0=version(guard)
        need(adapter.brain._edge_buffer_active and not adapter.brain._general_buffer_active,'Owner weight window not installed')
        tap=Tap(adapter);entry,generated=private_layout(layout,Path(layout.__file__).read_text(),tap)
        (out/'instrumented_layout.py').write_text(generated,encoding='utf-8')
        saved=layout.assemble
        try:
            layout.assemble=entry;observed=EffectiveOracle(adapter)
        finally:layout.assemble=saved
        need(tap.rounds==3 and len(tap.labels)>0,'Layout hook not reached three times; inspect actual binding')
        stream.synchronize();report['warmup_completed']=2
        need(version(guard)==v0,'Construction changed physical/operator state')
        deadline()
        save(out/'SITES.json',{'markers':tap.labels,'kernels_per_query':tap.kernels,
            'columns':['state_q','state_transmission','target','rate','selected_weights'],
            'rows':tap.rows.tolist(),'weight_positions':tap.positions.tolist(),
            'semantic_labels':'Kernel labels from published map; copies do not independently prove kernel formulas.'})
        first=None;private_canary=None
        for occurrence,index in enumerate((0,1,0)):
            deadline();c=cases[index];before=version(guard);need(before==v0,'Version changed: rebuild, do not reuse graph')
            with stream:
                x=cp.asarray(c['z']);x_before=x.copy()
            values=[]
            for oracle in (reference,observed):
                with stream:
                    report['query_launch_attempts']+=1
                    z,a,r=oracle.query(x,float(c['t']))
                    vals=[host(cp,v,stream) for v in (z,a,r)]
                    report['query_sync_completed']+=1
                values.append(vals);deadline()
            trace=tap.read();after=version(guard)
            need(after==before,'Query changed effective owner/event/operator version')
            need(exact(host(cp,x,stream),c['z']) and bool(cp.array_equal(x,x_before)),'Input mutation')
            expected=[c[k] for k in ('z','target','rate')]
            # Guardar evidencia antes de exigir paridad.
            np.savez_compressed(out/f'query_{occurrence}.npz',
                rows=tap.rows,weight_positions=tap.positions,intermediate=trace,
                reference_z=values[0][0],reference_target=values[0][1],reference_rate=values[0][2],
                observed_z=values[1][0],observed_target=values[1][1],observed_rate=values[1][2])
            row={'occurrence':occurrence,'candidate':'AB'[index],'t_hex':float(c['t']).hex(),
                 'phase':c['phase'],'global_start_ns':int(c['global_start_ns']),
                 'input_sha256':hashlib.sha256(c['z'].tobytes()).hexdigest(),
                 'version_before':before,'version_after':after,
                 'reference_vs_consumed':[exact(a,b) for a,b in zip(values[0],expected)],
                 'observer_vs_reference':[exact(a,b) for a,b in zip(values[1],values[0])]}
            report['queries'].append(row);save(out/'RESULT.json',report)
            need(all(row['reference_vs_consumed']) and all(row['observer_vs_reference']),'Endpoint parity failure; files retained')
            if occurrence==0:
                first=[v.copy() for v in values[1]]+[trace.copy()]
                with stream:
                    private_canary=cp.arange(32768,dtype=cp.float64);canary_copy=host(cp,private_canary,stream)
            else:need(exact(host(cp,private_canary,stream),canary_copy),'Private allocation canary corrupted')
            if occurrence==2:need(all(exact(a,b) for a,b in zip(first,values[1]+[trace])),'A-B-A intermediate mismatch')
        costs=kernel_edge_visits or {};base_edges=int(adapter.brain.cuda['indices'].size)
        known=0;unknown=[]
        for site in tap.kernels:
            n=base_edges if site.endswith(':kernel') else costs.get(site)
            if n is None:unknown.append(site)
            else:need(type(n) is int and n>=0,'Invalid edge accounting');known+=n
        report.update(status='IDENTITY_TAP_ONLY',markers=len(tap.labels),
            full_operator_executions_this_probe=8,construction_executions=2,replay_executions=6,
            graph_capture_recordings_not_executions=1,original_oracle_setup_not_included=True,
            known_edge_visits_lower_bound=8*known,unknown_cost_sites=unknown,
            work_gate_qualified=False,additional_tap_bytes=sum(b.nbytes for b in tap.buffers),
            note='No integration, no all-memory safety proof, no full state-owner neutral replay.')
    except BaseException:
        report.update(status='FAILED_RETAINED_DO_NOT_REUSE_WORKER',error=traceback.format_exc())
    finally:
        report['wall_s']=time.monotonic()-start;report['module_sha256']=sha(__file__)
        save(out/'RESULT.json',report)
    return report
