"""MRI-GARK ERK33a 3(2), UN bloque CPU, sin publicacion ni aceptacion automatica.
Ffast incluye eventos fechados/locales; Fslow=Ffull-Ffast, evaluado en el candidato.
Coeficientes: Sandu, SINUM57 (2019), DOI10.1137/18M1205492; SUNDIALS ERK33a.
Solo ODE reducida: el proveedor debe aplicar su masa completa sin aproximarla.
Los callbacks son puros, devuelven (derivada_FP64, aristas_visitadas). Sin anatomia.
"""
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ATOL, RTOL = 1e-7, 1e-5


def need(ok, msg):
    if not ok: raise ValueError(msg)


def normalizado(a, b):
    return float(np.max(np.abs(a-b)/(ATOL+RTOL*np.maximum(np.abs(a),np.abs(b)))))


class Ensayo:
    """Modelo: project(t,z,side), fast/full(t,z,side), version(), setup_edges.
    side es 'left' o 'right'. Solo las coordenadas prescribed pueden saltar.
    Cada llamada declara TODO su trabajo en aristas, incluso un fallback completo.
    Los inputs/outputs se copian: compatible con proveedores que retornan scratch.
    No resuelve eventos endogenos nuevos: agenda congelada solo para este replay.
    """
    def __init__(self, modelo, inicial, prescribed, bounds, t0, t1,
                 eventos, parent_edges, edge_budget):
        self.m=modelo; self.x=np.array(inicial,copy=True)
        need(self.x.dtype==np.float64 and self.x.ndim==1,'Estado FP64 vectorial')
        self.mask=np.array(prescribed,copy=True)
        need(self.mask.dtype==np.bool_ and self.mask.shape==self.x.shape,'Mascara prescrita')
        self.free=~self.mask; need(self.free.any(),'Sin coordenadas libres')
        self.lo=np.broadcast_to(np.asarray(bounds[0],float),self.x.shape).copy()
        self.hi=np.broadcast_to(np.asarray(bounds[1],float),self.x.shape).copy()
        need(not np.isnan(self.lo).any() and not np.isnan(self.hi).any() and np.all(self.lo<=self.hi),'Dominios')
        need(np.isfinite([t0,t1]).all() and 0<t1-t0<=125.0000001e-6,'Bloque <=125us')
        self.t0,self.t1=float(t0),float(t1);ev=list(map(float,eventos))
        need(ev==sorted(ev) and all(t0<=t<=t1 for t in ev),'Fechas/orden')
        self.ev=sorted(set(ev));need(type(parent_edges) is int,'parent_edges entero');self.parent_edges=parent_edges
        need(type(edge_budget) is int and edge_budget>0 and self.parent_edges>0,'Presupuesto de trabajo')
        self.limit=edge_budget;self.deadline=time.monotonic()+120
        self.token=modelo.version();self.trace=[];self.calls={'full':0,'fast':0}
        need(type(modelo.setup_edges) is int,'setup_edges entero');self.edges=modelo.setup_edges;need(self.edges>=0,'Setup negativo')
        self.macro_retries=0;self.pieces=[];self.check(self.x)
    def check(self,z):
        need(time.monotonic()<self.deadline,'Presupuesto120s')
        need(self.m.version()==self.token,'Version/propietario/eventos cambiaron')
        need(self.edges<=self.limit,'Presupuesto de aristas agotado')
        need(isinstance(z,np.ndarray) and z.dtype==np.float64 and z.shape==self.x.shape and np.isfinite(z).all(),'Layout/no finito')
    def domain(self,z):
        self.check(z);need(np.all((self.lo<=z)&(z<=self.hi)),'Dominio: no clipping')
    def projected(self,t,z,side='right'):
        self.check(z);arg=z.copy();old=arg.tobytes()
        q=self.m.project(float(t),arg,side)
        need(arg.tobytes()==old,'Project muta input');q=np.array(q,copy=True);self.domain(q)
        need(q[self.free].tobytes()==z[self.free].tobytes(),'Project cambia estado libre')
        return q
    def rhs(self,name,t,z,side='right'):
        self.check(z);arg=z.copy();old=arg.tobytes()
        self.calls[name]+=1
        if name=='full':need(self.calls[name]<=5,'Mas de cinco consultas completas')
        derivative,work=getattr(self.m,name)(float(t),arg,side)
        need(type(work) is int and work>=0,'Contabilidad de aristas')
        self.edges+=work;d=np.array(derivative,copy=True);self.check(d)
        need(arg.tobytes()==old,'RHS muta candidato')
        need(np.all(d[self.mask]==0),'RHS integra coordenadas prescritas')
        self.trace.append((name,float(t),side,work))
        return d
    def slow(self,t,z,side='right'):
        return self.rhs('full',t,z,side)-self.rhs('fast',t,z,side)
    def flow(self,left,right,y,g0,g1,sample=None):
        """DOP853 barato. Corta el subflujo, NO reinicia las etapas lentas MRI.
        No se materializa solucion densa completa; solo un punto de auditoria.
        Rtol/atol interiores fijos diez veces mas estrictos; no son cota global.
        """
        span=right-left;current=y.copy();selected=None;last=left
        cuts=sorted(set([e for e in self.ev if left<e<right]+[right]))
        for stop in cuts:
            template=current.copy()
            def local(t,u):
                z=template.copy();z[self.free]=u
                side='left' if t==stop and stop in self.ev else 'right'
                z=self.projected(t,z,side)
                v=self.rhs('fast',t,z,side)+g0+((t-left)/span)*g1
                return v[self.free]
            wanted=[stop]
            if sample is not None and last<=sample<stop:
                wanted=sorted(set([sample,stop]))
            sol=solve_ivp(local,(last,stop),current[self.free],method='DOP853',
                atol=ATOL/10,rtol=RTOL/10,t_eval=wanted,first_step=stop-last,
                max_step=stop-last)
            need(sol.success,'Subflujo DOP853 fallo')
            for k,t in enumerate(sol.t):
                z=template.copy();z[self.free]=sol.y[:,k];z=self.projected(t,z)
                if sample is not None and t==sample:selected=z.copy()
                if t==stop:current=z
            self.pieces.append({'start':last,'stop':stop,'event_boundary':stop in self.ev,'cheap_nfev':sol.nfev})
            last=stop
        return current,selected
    def run(self,reference_end):
        need(isinstance(reference_end,np.ndarray) and reference_end.dtype==np.float64 and reference_end.shape==self.x.shape,'Endpoint real requerido')
        self.domain(reference_end);refcopy=reference_end.copy();xcopy=self.x.copy()
        H=self.t1-self.t0;t=[self.t0,self.t0+H/3,self.t0+2*H/3,self.t1]
        y0=self.projected(t[0],self.x);s0=self.slow(t[0],y0);zero=np.zeros_like(y0)
        y1,_=self.flow(t[0],t[1],y0,s0,zero)
        s1=self.slow(t[1],y1);g2=-s0+2*s1
        y2,middle=self.flow(t[1],t[2],y1,g2,zero,self.t0+H/2)
        s2=self.slow(t[2],y2);g3=-2*s1+3*s2;k3=1.5*(s0-s2)
        high,_=self.flow(t[2],t[3],y2,g3,k3)
        low,_=self.flow(t[2],t[3],y2,.25*s0-s1+1.75*s2,zero)
        # Dos auditorias prospectivas, NO certificado de defecto continuo.
        defect1=self.slow(self.t0+H/2,middle)-g2
        side='left' if self.t1 in self.ev else 'right'
        zend=self.projected(self.t1,high,side)
        defect2=self.slow(self.t1,zend,side)-(g3+k3)
        scale1=ATOL+RTOL*np.maximum(np.abs(y1),np.abs(y2))
        scale2=ATOL+RTOL*np.maximum(np.abs(y2),np.abs(high))
        sampled=max(float(np.max(H*np.abs(defect1)/scale1)),float(np.max(H*np.abs(defect2)/scale2)))
        endpoint=normalizado(high,refcopy);embedded=normalizado(high,low)
        need(self.x.tobytes()==xcopy.tobytes() and reference_end.tobytes()==refcopy.tobytes(),'Inputs alterados')
        self.domain(high);self.check(high)
        reduction=self.parent_edges/max(self.edges,1)
        gates={'endpoint':endpoint<=1,'embedded':embedded<=1,'sampled_defect':sampled<=1,'work10x':reduction>=10}
        return {'status':'PASS_LIMITED_DISCRIMINATOR' if all(gates.values()) else 'FAIL_RETAINED',
            'gates':gates,'endpoint_normalized':endpoint,'embedded_3_2':embedded,
            'sampled_defect_normalized':sampled,'full_calls':self.calls['full'],
            'fast_calls':self.calls['fast'],'edge_visits':self.edges,
            'parent_edge_visits':self.parent_edges,'edge_work_ratio':reduction,
            'trace':self.trace,'pieces':self.pieces,'domain_checked':True,
            'continuous_defect_bound':False,'inner_global_error_certified':False,
            'owner_state_published':False,'CUDA_executed':False},high,low


def selftest():
    from scipy.linalg import expm
    T=125e-6;te=.9*T;mask=np.array([1,1,0,0],bool)
    M=np.array([[1.,.2],[.2,1.3]]);Mi=np.linalg.inv(M)
    W=np.array([[0.,.07],[.04,0.]])
    class Model:
        setup_edges=0
        def version(self):return 'frozen-fixture-v1'
        def project(self,t,z,side):
            out=z.copy();dt=t-te
            if dt<0 or (dt==0 and side=='left'):out[:2]=0.
            else:out[:2]=[.5*np.exp(-dt/T),.5*(dt/T)*np.exp(-dt/T)]
            return out
        def fast(self,t,z,side):
            d=np.zeros(4);d[2:]=Mi@(-z[2:]+np.array([z[1],.3*z[1]]))/T
            return d,0 # Analytic local fixture, NO event CSR traversal.
        def full(self,t,z,side):
            d=self.fast(t,z,side)[0];d[2:]+=Mi@(W@z[2:])/T
            return d,4
    K=np.zeros((4,4));K[0,0]=-1;K[1,:2]=[1,-1]
    K[2:,2:]=Mi@(-np.eye(2)+W);K[2:,1]=Mi@np.array([1.,.3]);K/=T
    exact=expm(K*(T-te))@np.array([.5,0.,0.,0.]);initial=np.zeros(4)
    e=Ensayo(Model(),initial,mask,(-np.inf,np.inf),0.,T,[te],240,24)
    r,high,_=e.run(exact)
    need(r['full_calls']==5 and r['endpoint_normalized']>1 and not r['gates']['sampled_defect'],'Recurrent late-event falsifier missing')
    need(any(p['stop']==te for p in r['pieces']),'Late event not split')
    need(np.array_equal(initial,np.zeros(4)),'Initial state mutated')
    # Error de un muestreador sin corte: sus nodos 0,H/4,H/2,3H/4 ven cero.
    missed=0.;need(exact[2]>1e-4 and missed==0,'Late negative missing')
    class Bad(Model):
        def full(self,t,z,side):return np.full_like(z,np.nan),4
    b=Ensayo(Bad(),initial,mask,(-np.inf,np.inf),0.,T,[te],240,24)
    try:b.run(exact)
    except ValueError:bad=True
    else:raise AssertionError('Nonfinite not rejected')
    class Costly(Model):
        def fast(self,t,z,side):return super().fast(t,z,side)[0],1
    b=Ensayo(Costly(),initial,mask,(-np.inf,np.inf),0.,T,[te],240,24)
    try:b.run(exact)
    except ValueError:cost=True
    else:raise AssertionError('Work cap not exercised')
    class Constant(Model):
        def full(self,t,z,side):
            d=self.fast(t,z,side)[0];d[2:]+=np.array([.02,-.01])/T
            return d,4
    L=np.zeros((5,5));L[:2,:2]=np.array([[-1.,0.],[1.,-1.]])/T
    L[2:4,2:4]=-Mi/T;L[2:4,1]=Mi@np.array([1.,.3])/T
    L[2:4,4]=np.array([.02,-.01])/T
    exact2=expm(L*te)@np.array([0.,0.,0.,0.,1.]);exact2[0]=.5
    exact2=(expm(L*(T-te))@exact2)[:4]
    positive=Ensayo(Constant(),initial,mask,(-np.inf,np.inf),0.,T,[te],240,24)
    ok,_,_=positive.run(exact2)
    need(all(ok['gates'].values()),'Constant remainder control failed')
    return {'test':'CPU_SYNTHETIC_NONDIAGONAL_MASS','candidate':r,
        'constant_remainder_control':ok,
        'max_absolute_error':float(np.max(abs(high-exact))),
        'late_uncut_negative_error':float(exact[2]),'nonfinite_rejected':bad,
        'extra_event_edge_work_rejected':cost,'initial_unchanged':True,
        'organism_executed':False,'CUDA_executed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:result=selftest()
    except BaseException:result={'status':'FAILED_RETAINED','error':traceback.format_exc()}
    result['wall_s']=time.monotonic()-start;result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(2 if 'error' in result else 0)
