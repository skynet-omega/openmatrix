"""Sonda CPU/C++ de caché temporal para un mapa lineal CSR efectivo.
No integra, no acepta pasos y no cambia eventos ni tolerancias del organismo.
Cada consulta se audita además con el producto completo; esos productos no se
cuentan como trabajo ahorrable. Los datos deben contener inputs reales de etapas.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse, ctypes as ct, hashlib, json, subprocess, time, traceback, zipfile
from pathlib import Path
import numpy as np
CPP = r'''
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <limits>
#include <algorithm>
static double up(double x){return std::nextafter(x,INFINITY);}
static double dn(double x){return std::nextafter(x,-INFINITY);}
extern "C" int env_ok(){
 volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.;
 return std::fegetround()==FE_TONEAREST && tiny*one==tiny && sizeof(double)==8;
}
extern "C" void row_norm(int n,const int64_t*p,const double*w,double*out){
 for(int i=0;i<n;i++){double a=0;for(int64_t e=p[i];e<p[i+1];e++)a=up(a+std::abs(w[e]));out[i]=a;}
}
extern "C" void product(int n,const int64_t*p,const int32_t*idx,
 const double*w,const double*x,double*y,double*lo,double*hi,int box){
 for(int i=0;i<n;i++){
  double a[32]={},l[32]={},h[32]={};
  for(int j=0;j<32;j++)for(int64_t e=p[i]+j;e<p[i+1];e+=32){
   double v=w[e]*x[idx[e]];a[j]=a[j]+v;
   if(box){l[j]=dn(l[j]+dn(v));h[j]=up(h[j]+up(v));}
  }
  for(int d=16;d;d/=2)for(int j=0;j<d;j++){
   a[j]=a[j]+a[j+d];
   if(box){l[j]=dn(l[j]+l[j+d]);h[j]=up(h[j]+h[j+d]);}
  }
  y[i]=a[0];lo[i]=box?l[0]:a[0];hi[i]=box?h[0]:a[0];
 }
}
extern "C" double predict(int n,const double*x,const double*x0,const double*x1,
 const double*y0,const double*l0,const double*h0,
 const double*y1,const double*l1,const double*h1,const double*norm,
 double alpha,double beta,double*y,double*lo,double*hi){
 double radius=0;
 for(int j=0;j<n;j++){
  double a=alpha*x0[j],b=beta*x1[j];
  double low=dn(dn(a)+dn(b)),high=up(up(a)+up(b));
  radius=std::max(radius,std::max(up(x[j]-low),up(high-x[j])));
 }
 for(int i=0;i<n;i++){
  double l=dn(dn(alpha*(alpha>=0?l0[i]:h0[i]))+
              dn(beta*(beta>=0?l1[i]:h1[i])));
  double h=up(up(alpha*(alpha>=0?h0[i]:l0[i]))+
              up(beta*(beta>=0?h1[i]:l1[i])));
  double spread=up(norm[i]*radius);
  y[i]=alpha*y0[i]+beta*y1[i];lo[i]=dn(l-spread);hi[i]=up(h+spread);
 }
 return radius;
}
'''
PLAN = {"method":"affine_input_cache_interval_v1","max_frames":128,
        "wall_budget_s":60,"minimum_full_pass_reduction":10,
        "scope":"Una versión CSR, error de producto por consulta; no error neuronal",
        "mode":"shadow_only","stage_admission":False}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def ptr(a):return a.ctypes.data_as(ct.c_void_p)
class Kernel:
    def __init__(self,folder,p,idx,w):
        source=folder/"cache_kernel.cpp";source.write_text(CPP,encoding="utf-8")
        lib=folder/"cache_kernel.so"
        cmd=["g++","-std=c++17","-O2","-ffp-contract=off","-fno-fast-math",
             "-frounding-math","-fPIC","-shared",str(source),"-o",str(lib)]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        (folder/"BUILD.txt").write_text(" ".join(cmd)+"\n"+r.stdout+r.stderr)
        need(r.returncode==0,"Compilación C++ falló")
        self.lib=ct.CDLL(str(lib.resolve()));need(self.lib.env_ok()==1,"Entorno IEEE no admitido")
        self.lib.row_norm.argtypes=[ct.c_int]+[ct.c_void_p]*3
        self.lib.product.argtypes=[ct.c_int]+[ct.c_void_p]*7+[ct.c_int]
        self.lib.predict.argtypes=[ct.c_int]+[ct.c_void_p]*10+[ct.c_double]*2+[ct.c_void_p]*3
        self.lib.predict.restype=ct.c_double
        self.n=len(p)-1;self.p,self.idx,self.w=p,idx,w;self.norm=np.empty(self.n)
        start=time.perf_counter()
        self.lib.row_norm(self.n,ptr(p),ptr(w),ptr(self.norm))
        self.setup_s=time.perf_counter()-start
        need(np.isfinite(self.norm).all(),"Norma no finita")
    def product(self,x,box=True):
        y,l,h=[np.empty(self.n) for _ in range(3)]
        self.lib.product(self.n,ptr(self.p),ptr(self.idx),ptr(self.w),ptr(x),
                         ptr(y),ptr(l),ptr(h),int(box))
        need(all(np.isfinite(a).all() for a in (y,l,h)),"Producto no finito")
        return y,l,h
    def prediction(self,x,a,b,t):
        beta=0. if b[0]==a[0] else (t-a[0])/(b[0]-a[0])
        alpha=1.-beta
        if not np.isfinite([alpha,beta]).all():return None
        y,l,h=[np.empty(self.n) for _ in range(3)]
        self.lib.predict(self.n,ptr(x),ptr(a[1]),ptr(b[1]),
            *[ptr(z) for z in a[2]],*[ptr(z) for z in b[2]],ptr(self.norm),
            alpha,beta,ptr(y),ptr(l),ptr(h))
        if not all(np.isfinite(z).all() for z in (y,l,h)):return None
        return y,l,h
class Cache:
    def __init__(self,kernel,budget):
        self.k=kernel;self.budget=budget;self.anchors=[];self.epoch=None;self.full=0
    def query(self,x,t,epoch):
        if epoch!=self.epoch:self.anchors=[];self.epoch=epoch
        estimate=None
        if self.anchors:
            a=self.anchors[0];b=self.anchors[-1]
            estimate=self.k.prediction(x,a,b,t)
        if estimate is not None:
            y,l,h=estimate
            radius=np.nextafter(np.maximum(y-l,h-y),np.inf)
            if np.all(radius<=self.budget):return estimate,True
        exact=self.k.product(x);self.full+=1
        anchor=(t,x.copy(),tuple(z.copy() for z in exact))
        # Cache is computational memoization, not an event publisher.
        self.anchors=(self.anchors[-1:]+[anchor])[-2:]
        return exact,False

def validate(d):
    p,idx,w=(d[k] for k in ("indptr","indices","weights"));n=len(p)-1
    need(p.ndim==1 and p.dtype==np.int64 and 0<n<2**26,"indptr")
    need(idx.ndim==1 and idx.dtype==np.int32 and w.dtype==np.float64 and w.shape==idx.shape and len(w)>0,"aristas")
    need(p[0]==0 and p[-1]==len(w) and np.all(p[1:]>=p[:-1]) and idx.min()>=0 and idx.max()<n,"CSR")
    x=d["rhs"];need(x.dtype==np.float64 and x.ndim==2 and x.shape[1]==n,"rhs[consulta,fila]")
    F=len(x);need(2<=F<=PLAN["max_frames"],"2..128 consultas")
    for k in ("query_s","operator_epoch"):need(d[k].shape==(F,),"Eje "+k)
    need(d["query_s"].dtype==np.float64 and d["operator_epoch"].dtype==np.int64,"Tiempo/versión")
    need(d["budget"].dtype==np.float64 and d["budget"].shape==(n,) and np.all(d["budget"]>0),"Presupuesto por receptor")
    for k in ("weights","rhs","query_s","budget"):need(np.isfinite(d[k]).all(),"No finito "+k)
    weightsha=hashlib.sha256(w.tobytes()).hexdigest()
    need(d["weights_sha256"].shape==(F,) and all(str(v)==weightsha for v in d["weights_sha256"]),"Versión W distinta: otra cápsula")
    need(d["context"].shape==() and str(d["context"].item()),"Falta procedencia")
    return n,F

def evaluate(k,d,deadline):
    cache=Cache(k,d["budget"]);rows=[];tc=tb=ta=0.
    for j,x in enumerate(d["rhs"]):
        need(time.perf_counter()<deadline,"Presupuesto agotado; prefijo conservado")
        t=time.perf_counter();reference=k.product(x,False)[0];tb+=time.perf_counter()-t
        t=time.perf_counter();(value,lo,hi),hit=cache.query(x,float(d["query_s"][j]),int(d["operator_epoch"][j]));tc+=time.perf_counter()-t
        t=time.perf_counter();exact=k.product(x);ta+=time.perf_counter()-t
        need(np.array_equal(reference.view(np.uint64),exact[0].view(np.uint64)),"Instrumentación cambió suma RN")
        # Necessary external audit, not the proof of the interval theorem.
        need(np.all(lo<=exact[2]) and np.all(hi>=exact[1]),"Envolventes disjuntas")
        row={"query":j,"time_s":float(d["query_s"][j]),"reuse":hit,
             "radius_max":float(np.max(np.maximum(value-lo,hi-value))),
             "error_vs_full_FP64":float(np.max(abs(value-reference)))}
        rows.append(row)
        save(k.folder/"PREFIX.json",rows)
    return {"queries":len(rows),"candidate_full_products":cache.full,
       "one_time_norm_passes":1,"full_pass_reduction_including_setup":len(rows)/(1+cache.full),
       "reference_RN_wall_s":tb,"candidate_wall_s":tc+k.setup_s,
       "validation_extra_wall_s":ta,"rows":rows,
       "order10_work_gate":len(rows)>=10*(1+cache.full),
       "scope":"Sonda en trayectoria del padre; no nueva trayectoria ni speedup GPU",
       "continuous_time_coverage":False,"recurrence_bound":False}

def selftest(folder):
    from decimal import Decimal as D, localcontext
    p=np.array([0,2,3],dtype=np.int64);idx=np.array([0,1,0],dtype=np.int32)
    w=np.array([2.,-1.,.5]);k=Kernel(folder,p,idx,w);k.folder=folder
    times=np.linspace(0,.001,64);rhs=np.column_stack((.2+times,.4+2*times))
    d=dict(rhs=rhs,query_s=times,operator_epoch=np.zeros(64,dtype=np.int64),budget=np.full(2,1e-6))
    report=evaluate(k,d,time.perf_counter()+20)
    need(report["order10_work_gate"],"No reutiliza el caso afín controlado")
    c=Cache(k,d["budget"]);largest=0.
    with localcontext() as ctx:
        ctx.prec=100
        for t,x in zip(times,rhs):
            (y,lo,hi),hit=c.query(x,float(t),0)
            for i in range(2):
                real=sum((D.from_float(float(w[e]))*D.from_float(float(x[idx[e]]))
                          for e in range(p[i],p[i+1])),D(0))
                need(D.from_float(float(lo[i]))<=real<=D.from_float(float(hi[i])),"Intervalo no contiene referencia Decimal")
            largest=max(largest,float(np.max(abs(y-k.product(x,False)[0]))))
    # Graduated source with no spikes: x(t)=s0+v*t, r(t)=x(t)+tau*v.
    # Late SET on q: s(h)=.5*.1*exp(-.1), not zero.
    c=Cache(k,d["budget"])
    for t in (0.,.5):c.query(np.zeros(2),t,0)
    late=np.array([.5*.1*np.exp(-.1),0.])
    out,hit=c.query(late,1.,0)
    need(not hit and abs(out[0][0])>1e-3,"Evento tardío ignorado")
    # A projection-only method without cutting at .9h misses downstream z.
    downstream=.5*np.exp(-.1)*(.1**2)/2
    need(downstream>1e-4,"Control tardío degenerado")
    _,hit=c.query(late,1.,1);need(not hit,"No invalidó cambio de propietario/versión")
    rough=Cache(k,d["budget"])
    for j,t in enumerate(times):
        x=np.array([.2+.1*np.sin(16000*np.pi*t),.3+.1*np.cos(22000*np.pi*t)])
        rough.query(x,float(t),0)
    need(rough.full>len(times)/10,"No conserva el negativo de curvatura")
    before=late.copy();a=c.query(late,.25,1);b=c.query(late,.25,1)
    need(np.array_equal(late,before),"Mutó input")
    for value in (a,b):
        need(np.isfinite(value[0][0]).all(),"Consulta no cronológica inválida")
    return {"scope":"Fixture CPU C++, no CSR real","affine":report,
       "Decimal_interval_checks":128,"max_affine_error":largest,
       "graded_no_spikes":True,"late_event_forces_refresh":True,
       "uncut_midpoint_counterexample_z":float(downstream),
       "late_cut_still_required":True,"version_invalidation":True,
       "nonchronological_queries":True,"curved_source_full_products":rough.full,
       "curved_source_queries":len(times),"CUDA_executed":False}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",type=Path)
    ap.add_argument("--sha256");ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    need(a.selftest != bool(a.input),"Elegir --input o --selftest")
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/"PLAN.json",PLAN)
    start=time.perf_counter();report={"CUDA_executed":False,"organism_executed":False}
    try:
        if a.selftest:report["test"]=selftest(a.out)
        else:
            need(a.sha256 and sha(a.input)==a.sha256,"Hash de cápsula")
            with zipfile.ZipFile(a.input) as z:
                need(sum(v.file_size for v in z.infolist())<1024**3,"Cápsula >1 GiB")
            with np.load(a.input,allow_pickle=False) as z:d={v:np.ascontiguousarray(z[v]) if z[v].ndim else z[v].copy() for v in z.files}
            n,F=validate(d)
            k=Kernel(a.out,d["indptr"],d["indices"],d["weights"]);k.folder=a.out
            report.update(input_sha256=sha(a.input),context=str(d["context"].item()),
                rows=n,edges=len(d["weights"]),result=evaluate(k,d,start+PLAN["wall_budget_s"]))
        report["status"]="COMPLETE_SHADOW_ONLY"
    except Exception:report.update(status="FAILED_RETAINED",error=traceback.format_exc())
    report.update(wall_s=time.perf_counter()-start,code_sha256=sha(Path(__file__)))
    save(a.out/"RESULT.json",report);print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if report["status"]=="COMPLETE_SHADOW_ONLY" else 2
if __name__=="__main__":raise SystemExit(main())
