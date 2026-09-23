"""B aislado: Euler/punto medio exponencial, dos coeficientes por intento.
Genera una copia de graph_core; no instala ni ejecuta CUDA/organismo.
El selftest ejecuta sus métodos aritméticos en NumPy, no el controlador CUDA.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, ast, hashlib, json, math, time, traceback, types
from pathlib import Path
GRAPH_SHA='898dbf27525887da8f127ffe8114df734a8dcc8a99072f667e263ed82838a497'
CTRL_SHA='5aa0935103ad2e29d7bf05683ad3e291949f04c7abcd6537f945cc90daa62642'
PLAN={'name':'B2_exponential_Euler_midpoint_v1','orders':[1,2],
      'norm_divisor':1,'coefficients_per_trial':2,'coefficient_fractions':[0.,.5],
      'controller':'Padre sin cambios: aceptar e<=1, doblar si e<0.1, dividir tras rechazo',
      'known_event_cuts':'Obligatorios; [] declara ausencia de fronteras',
      'global_error_bound':False,'organism_executed':False,'CUDA_executed':False,
      'CPU_tests':{'rtol':1e-6,'atol':1e-9,'true_error_limit':1e-4,'max_attempts':10000}}
TRIAL=''' def trial(self):
  self.err.fill(0);self.flag.fill(0);self.domain.fill(0)
  z,a0,b0=self.coeff(self.x,0.)
  euler=z+(-cp.expm1(-self.clock[1]*b0))*(a0-z)
  middle=z+(-cp.expm1(-.5*self.clock[1]*b0))*(a0-z)
  # Materializar ambos antes de que otro frame reutilice a0/b0.
  _,am,bm=self.coeff(middle,.5)
  upper=z+(-cp.expm1(-self.clock[1]*bm))*(am-z)
  self.full=self.project(euler,self.clock,1.) if self.project else euler
  self.fine=self.project(upper,self.clock,1.) if self.project else upper
  self.norm(self.grid,(256,),(self.full,self.fine,np.int32(self.n),np.int32(self.norm_size),np.float64(self.atol),np.float64(self.rtol),self.err,self.flag,self.domain,self.lower,self.upper));self.summary((1,),(1,),(self.err,self.flag,self.domain,self.status))
'''
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def generate(parent,control,out):
    need(sha(parent)==GRAPH_SHA and sha(control)==CTRL_SHA,'SHA del padre/control distinto')
    src=parent.read_text(encoding='utf-8');tree=ast.parse(src)
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeGraph')
    trial=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='trial')
    lines=src.splitlines(keepends=True);new=''.join(lines[:trial.lineno-1])+TRIAL+''.join(lines[trial.end_lineno:])
    old='e=fabs(a[i]-b[i])/(3*(atol+rtol*fmax(fabs(a[i]),fabs(b[i]))));'
    norm='e=fabs(a[i]-b[i])/(atol+rtol*fmax(fabs(a[i]),fabs(b[i])));'
    need(new.count(old)==1,'Ancla norma');new=new.replace(old,norm)
    new=new.replace('class NativeGraph:\n','class NativeGraph:\n evaluations_per_trial=2\n coefficient_fractions=(0.,.5)\n')
    mark='  start=time.perf_counter();self.stream='
    need(new.count(mark)==1,'Ancla constructor')
    new=new.replace(mark,'  if native_library is None:raise ValueError("B requiere biblioteca padre explicita")\n'+mark)
    mark=' def advance(self,ns,next_ns,min_ns,max_ns,budget=30,boundaries=None):\n'
    need(new.count(mark)==1,'Ancla fronteras')
    new=new.replace(mark,mark+'  if boundaries is None:raise ValueError("B requiere agenda explicita; [] si no hay eventos")\n')
    compile(new,'graph_core_B.py','exec')
    (out/'graph_core_B.py').write_text(new,encoding='utf-8')
    (out/'graph_core_parent.py').write_bytes(parent.read_bytes())
    (out/'graph_control_parent.cpp').write_bytes(control.read_bytes())
    save(out/'MANIFEST.json',{'parent_sha256':GRAPH_SHA,'controller_sha256':CTRL_SHA,
        'B_sha256':sha(out/'graph_core_B.py'),'plan':PLAN})
    return src,new

def selftest(parent,candidate,out):
    import numpy as np
    def cpu_class(src):
        c=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef))
        funcs=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ('coeff','midpoint','trial')]
        c=ast.ClassDef(name='CPU',bases=[],keywords=[],body=funcs,decorator_list=[])
        scope={'np':np,'cp':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),'<metodos_publicados_NumPy>','exec'),scope)
        return scope['CPU']
    P,B=cpu_class(parent),cpu_class(candidate)
    def setup(cls,x,model,project=None,lower=0.,upper=1.):
        g=cls();g.x=np.array(x,dtype=float);g.n=g.norm_size=len(x);g.clock=np.zeros(2)
        g.atol=1e-9;g.rtol=1e-6;g.grid=(1,);g.err=np.zeros(1);g.flag=np.zeros(1,dtype=np.int32)
        g.domain=np.zeros(1,dtype=np.int32);g.status=np.zeros(3);g.calls=0;g.projections=[]
        g.lower=np.full(g.n,lower);g.upper=np.full(g.n,upper);bufa=np.empty(g.n);bufb=np.empty(g.n)
        def projection(y,c,f):
            g.when=c[0]+f*c[1];g.projections.append(float(f))
            return np.array(y,copy=True) if project is None else project(y,g.when)
        def coefficient(z):
            g.calls+=1;a,b=model(g.when,z);bufa[:]=a;bufb[:]=b
            return bufa,bufb  # Reutiliza buffers entre etapas deliberadamente.
        def check(grid,block,args):
            if not all(np.isfinite(a).all() for a in args[:3]):g.flag[0]=1
        def norm(grid,block,args):
            a,b=args[:2]
            if not np.isfinite(a).all() or not np.isfinite(b).all():g.flag[0]=1
            g.domain[0]=int(np.any((b<g.lower)|(b>g.upper)))
            e=np.abs(a-b)/((1 if cls is B else 3)*(g.atol+g.rtol*np.maximum(abs(a),abs(b))))
            if not np.isfinite(e).all():g.flag[0]=1;e=np.where(np.isfinite(e),e,0.)
            g.err[0]=float(e.max())
        g.project=projection;g.coefficient=coefficient;g.check=check;g.norm=norm
        g.summary=lambda grid,block,args:g.status.__setitem__(slice(None),[g.err[0],g.flag[0],g.domain[0]])
        return g
    def step(g,t,h):
        before=g.x.tobytes();g.clock[:]=[t,h];g.trial()
        need(g.x.tobytes()==before,'Una propuesta mutó el estado confirmado')
        return g.status.copy()
    def integrate(cls,model,initial,T,events=(),project=None):
        g=setup(cls,initial,model,project);t=0.;h=.125;acc=rej=0;ns_min=1e-10
        cuts=sorted(set([float(T)]+list(events)));log=[];tic=time.perf_counter()
        for _ in range(10000):
            if t>=T:break
            stop=next(v for v in cuts if v>t);dt=min(h,stop-t)
            e,flag,domain=step(g,t,dt);need(flag==0,'No finito CPU')
            if e<=1:
                need(domain==0,'Dominio CPU');g.x=g.fine.copy();acc+=1
                t=stop if dt==stop-t else t+dt;h=min(.125,max(ns_min,dt*(2 if e<.1 else 1)))
            else:
                rej+=1;h=dt/2;need(h>=ns_min,'Límite de pasos CPU')
            log.append([t,dt,float(e),acc,rej])
        need(t==T,'Presupuesto CPU de intentos')
        return g,dict(accepted=acc,rejected=rej,coefficients=g.calls,wall_s=time.perf_counter()-tic),np.array(log)
    exact=lambda t:np.array([np.tanh(t+np.arctanh(.2)),.2+.1*t])
    model=lambda t,y:(np.array([1.,.2+.1*t+.1/(1+t)]),np.array([1+y[0],1+t]))
    orders=[]
    for h in (.05,.025,.0125,.00625):
        g=setup(B,exact(.3),model);step(g,.3,h);need(g.calls==2,'No son dos evaluaciones')
        need(g.projections==[0.,.5,1.,1.],'Lugares de proyección')
        orders.append([h,float(abs(g.full-exact(.3+h)).max()),float(abs(g.fine-exact(.3+h)).max())])
    observed=np.log2(np.array(orders[:-1])[:,1:]/np.array(orders[1:])[:,1:])
    need(np.all(observed[:,0]>1.7) and np.all(observed[:,1]>2.7),'Orden local inesperado')
    runs={}
    for cls,name in ((P,'parent'),(B,'B')):
        g,report,trace=integrate(cls,model,exact(0),1.)
        report['true_error']=float(abs(g.x-exact(1.)).max());need(report['true_error']<1e-4,'Error externo CPU')
        runs[name]=report;np.savez(out/(name+'_CPU.npz'),trace=trace,final=g.x,exact=exact(1.))
    # Un evento tardío con puertos exactos NO permite retirar fronteras.
    H=125e-6;te=.9*H
    def proj(y,t):
        z=y.copy();u=max(0.,(t-te)/H);z[:2]=[0.,0.] if t<te else [.5*np.exp(-u),.5*u*np.exp(-u)]
        return z
    f=lambda t,y:(np.array([y[0],y[1],y[1]]),np.array([0.,0.,1/H]))
    g=setup(B,[0,0,0],f,proj);status=step(g,0,H);truth=.5*np.exp(-.1)*.1**2/2
    need(status[0]==0 and g.fine[2]==0 and truth>1e-4,'No se conservó el falsador no-cut')
    late={'uncut_estimator':float(status[0]),'uncut_z':float(g.fine[2]),'true_z':float(truth)}
    g,r,trace=integrate(B,f,[0,0,0],H,(te,),proj);r['error_z']=float(abs(g.x[2]-truth))
    need(r['error_z']<1e-4,'Frontera tardía');late['with_cut']=r
    np.savez(out/'late_CPU.npz',trace=trace,final=g.x)
    g=setup(B,[.2],lambda t,x:(np.array([3.]),np.array([10.])))
    status=step(g,0,.4);need(status[2]==1 and g.fine[0]>1,'Clipping silencioso')
    g=setup(B,[.2],lambda t,x:(np.array([np.nan]),np.array([1.])))
    with np.errstate(invalid='ignore'):status=step(g,0,.1)
    need(status[1]!=0,'No finito aceptado')
    g=setup(B,[.5],lambda t,x:(np.array([-1.]),np.array([10.])),lower=-2.,upper=3.)
    status=step(g,0,.4);need(status[2]==0 and g.fine[0]<0,'Dominio normalizado impuesto')
    return dict(scope='Sólo métodos CPU NumPy; sin CUDA/organismo/controlador nativo',local_errors=orders,
        observed_orders=observed.tolist(),runs=runs,late_event=late,domain_and_finite_flags=True,
        input_preserved=True,borrowed_coeff_buffers=True,no_promoted_model=True)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('parent',type=Path)
    a.add_argument('controller',type=Path);a.add_argument('out',type=Path);a.add_argument('--selftest',action='store_true');q=a.parse_args()
    q.out.mkdir(parents=True,exist_ok=False);save(q.out/'PLAN.json',PLAN);r={};tic=time.perf_counter()
    try:
        parent,candidate=generate(q.parent,q.controller,q.out)
        if q.selftest:r['CPU']=selftest(parent,candidate,q.out)
        r['status']='COMPLETE'
    except Exception:r.update(status='FAILED_RETAINED',error=traceback.format_exc())
    r.update(wall_s=time.perf_counter()-tic,generator_sha256=sha(Path(__file__)),CUDA_executed=False,organism_executed=False)
    save(q.out/'RESULTADO.json',r);print(json.dumps(r,indent=2,ensure_ascii=False));return 0 if r['status']=='COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
