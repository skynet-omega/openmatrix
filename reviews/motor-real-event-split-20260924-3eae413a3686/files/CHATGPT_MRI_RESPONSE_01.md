**Entrego un MRI-GARK-ERK33a 3(2) acotado, con pruebas CPU ejecutadas.** Su separación conserva exactamente el RHS en aritmética real; no congela indebidamente la tasa variable. **Pero esta implementación no ofrece todavía una ruta creíble a 10× menos productos:** reiniciar las etapas lentas en cada evento y auditar su defecto impone un coste mínimo que conviene calcular antes de gastar GPU.

## 1. Método y separación exacta

Al inicio del bloque se obtienen \(a_*,r_*\). Se define:

\[
F_f(y)=r_*\odot(a_*-y),
\qquad
F_s(t,y)=r(t,y)\odot[a(t,y)-y]-F_f(y).
\]

Entonces **\(F_f+F_s=F\)**. Los valores \(a_*,r_*\) solo construyen un término rápido resoluble analíticamente; **cada evaluación lenta vuelve a calcular todo el operador efectivo**, incluidas tasa variable, recurrencia y sustituciones.

La interfaz actual devuelve coeficientes completos, no un operador rápido local separado. Esta elección permite probar MRI sin inventar ese operador. **No demuestra que el residuo \(F_s\) sea lento**. Tampoco sustituye la masa PN o el productor KC: sus propietarios y puertos permanecen externos a este replay.  

**Coeficientes verificados:** `ARKODE_MRI_GARK_ERK33a` en SUNDIALS 7.6.0, atribuido a Sandu, *SINUM* 57, 2300–2327. Orden principal 3, embebido 2. Con \(c=(0,1/3,2/3,1)\), las fuerzas en tiempo físico son:

\[
g_1=S_0,\quad g_2=-S_0+2S_1,
\]
\[
g_3(\theta)=-2S_1+3S_2+\tfrac32\theta(S_0-S_2),
\]
\[
\widehat g_3=\tfrac14S_0-S_1+\tfrac74S_2.
\]

La solución embebida reinicia **el último tercio desde el mismo estado de etapa**, no desde la solución principal.  :chatgpt-content-reference{index="3"}

Para \(F_f=d(a_*-y)\) y fuerza \(g_0+\theta g_1\), el subproblema rápido tiene solución analítica:

\[
y(u)=y_0+u\varphi_1(-du)[d(a_*-y_0)+g_0]
+\frac{u^2}{D}\varphi_2(-du)g_1.
\]

Se evalúa en FP64; no afirmo aritmética exacta ni preservación automática del dominio.

**Error usado:** diferencia principal–embebida, con `atol=1e-7`, `rtol=1e-5`, **sin divisor 3**. Su término dominante es \(O(H^3)\), del embebido de orden 2. Las auditorías adicionales evalúan el residuo real en tres puntos no usados como etapas; **no constituyen una cota continua**. Los dominios y no finitos siguen comprobándose.  :chatgpt-content-reference{index="5"}

## 2. Resultado y límite económico

| Control CPU propio | Resultado |
|---|---:|
| Orden global observado, problema suave con tasa variable | 3,245 → 3,115 → 3,055 |
| Error escalado final frente a DOP853 independiente | 0,29765 |
| Evento tardío sin corte | estimador **0**, error **0,00226209** |
| Mismo evento con corte | error absoluto **2,37031×10⁻⁹** |
| Trabajo del caso tardío | 30 evaluaciones lentas +24 auditorías +1 inicial |
| Rechazo/no finitos/dominio | input original conservado; sin publicación |
| SET/ADD simultáneos | orden conservado por el proyector; \(s\) continuo |

Las auditorías dieron indicadores escalados de defecto mayores que uno en algunos intervalos —máximos **38,74** y **116,43**—. No los escondo ni los convierto en una certificación a partir del pequeño error final: el defecto de la trayectoria interna y el error final son comprobaciones diferentes.

Para **este código**, con \(B\) segmentos impuestos por eventos:

\[
N_{\min}=1+6B
\]

incluye inicialización y tres auditorías por aceptación. En el registro de 100 segmentos/ms, serían al menos **601 llamadas**, frente a 1.146: techo contable **1,91×**, antes de rechazos. Incluso sin auditorías serían 301, no 40–60. **El helper `work_floor()` permite descartar esta instancia por coste antes del replay.** No retiraría auditorías o cortes para hacerla pasar.

A sigue siendo MRI; **B=QSS2 y C=Krylov permanecen reservados**, sin implementarlos aquí.

## 3. Código completo: `mri33_replay.py`

Es un **discriminador numérico**, no un reemplazo nativo del motor. NumPy funciona en CPU; con CuPy los vectores permanecen en GPU y el flujo afín usa un kernel. El control y las comprobaciones escalares siguen en Python. **La rama CUDA no está compilada ni ejecutada aquí.**

```python
"""MRI-GARK-ERK33a 3(2), fast affine diagonal exact flow, NumPy/CuPy.
Source: Sandu, SINUM 57 (2019), DOI 10.1137/18M1205492;
LLNL/sundials v7.6.0 src/arkode/arkode_mri_tables.def, ERK33a.
Bounded REPLAY: pure callbacks and prescribed ports only, no neuronal commit.
No claim of continuous defect bound, DAE, endogenous-event generation or CUDA speed.
"""
import math, time
import numpy as np
ATOL, RTOL = 1e-7, 1e-5
# For physical time on each H/3 interval. Last row is the embedded solution.
P0 = ((1.,0.,0.),(-1.,2.,0.),(0.,-2.,3.),(.25,-1.,1.75))
P1 = ((0.,0.,0.),(0.,0.,0.),(1.5,0.,-1.5),(0.,0.,0.))

def need(ok, msg):
    if not ok: raise ValueError(msg)

def scalar(v): return float(v.item()) if hasattr(v, 'item') else float(v)

class AffineFast:
    """f_fast(t,y) = rate_ref * (target_ref - y). No frozen full RHS."""
    def __init__(self, target, rate, xp=np):
        self.xp=xp; self.target=target.copy(); self.rate=rate.copy(); self.kernel=None
        need(target.shape==rate.shape and target.dtype==rate.dtype==xp.float64,'FP64 affine arrays')
        need(bool(xp.isfinite(target).all()) and bool(xp.isfinite(rate).all()),'Nonfinite affine data')
        if xp.__name__=='cupy':
            self.kernel=xp.ElementwiseKernel(
                'float64 y, float64 a, float64 r, float64 g0, float64 g1, float64 u, float64 span',
                'float64 out', r'''
                double z=-r*u, p1, p2;
                if(fabs(z)<0.01){
                  p1=1./362880.; p2=1./3628800.;
                  for(int k=7;k>=0;--k){
                    double f1=1., f2=1.;
                    for(int j=2;j<=k+1;++j) f1*=j;
                    for(int j=2;j<=k+2;++j) f2*=j;
                    p1=1./f1+z*p1; p2=1./f2+z*p2;
                  }
                }else{p1=expm1(z)/z;p2=(p1-1.)/z;}
                out=y+u*(p1*(r*(a-y)+g0)+(u/span)*p2*g1);
                ''','mri33_affine_flow',options=('--std=c++11','--fmad=false'))
    def __call__(self,t,y): return self.rate*(self.target-y)
    def flow(self,y,g0,g1,span,fraction=1.):
        u=span*fraction
        if self.kernel is not None:
            return self.kernel(y,self.target,self.rate,g0,g1,np.float64(u),np.float64(span))
        z=-self.rate*u; small=abs(z)<.01; safe=np.where(small,1.,z)
        p1=np.expm1(safe)/safe; p2=(p1-1.)/safe
        s1=np.full_like(z,1/math.factorial(9));s2=np.full_like(z,1/math.factorial(10))
        for k in range(7,-1,-1):
            s1=1/math.factorial(k+1)+z*s1;s2=1/math.factorial(k+2)+z*s2
        p1=np.where(small,s1,p1);p2=np.where(small,s2,p2)
        return y+u*(p1*(self.rate*(self.target-y)+g0)+(u/span)*p2*g1)

class TargetRateSplit:
    """coeff(t,z) must be the EFFECTIVE, pure, fully active operator.
    It must return zero rates on prescribed ports, as in OrganismAdapter.
    """
    def __init__(self,coeff,project,t0,y0,xp=np):
        self.coeff=coeff;self.calls=0
        z=project(t0,y0.copy());a,r=self.evaluate(t0,z)
        self.fast=AffineFast(a,r,xp)
    def evaluate(self,t,z):
        self.calls+=1;a,r=self.coeff(t,z.copy())
        return a.copy(),r.copy()  # protect borrowed coefficient buffers
    def slow(self,t,z):
        a,r=self.evaluate(t,z)
        return r*(a-z)-self.fast(t,z)

class MRI33:
    """State stays on xp. Host controls macro steps; NOT a native speed benchmark.
    project(t,y) only overwrites declared prescribed coordinates from immutable
    histories; it owns ADD/SET ordering. Events here are times, not re-applied jumps.
    No callback may mutate physical owner state, weights, RNG, or input arguments.
    """
    def __init__(self,rhs_fast,rhs_slow,project,shape,bounds,prescribed,
                 norm_size=None,xp=np):
        need(isinstance(rhs_fast,AffineFast) and rhs_fast.xp is xp,'This prototype requires affine fast flow on the same backend')
        self.xp=xp;self.fast=rhs_fast;self.slow=rhs_slow;self.project=project
        need(len(shape)==1 and shape[0]>0,'One-dimensional state')
        self.shape=shape;self.n=shape[0];self.normn=self.n if norm_size is None else norm_size
        need(type(self.normn) is int and 0<self.normn<=self.n,'norm_size')
        self.mask=xp.asarray(prescribed,dtype=xp.bool_).copy()
        need(self.mask.shape==shape,'Prescribed mask')
        self.lo=xp.broadcast_to(xp.asarray(bounds[0],dtype=xp.float64),shape).copy()
        self.hi=xp.broadcast_to(xp.asarray(bounds[1],dtype=xp.float64),shape).copy()
        need(not bool(xp.isnan(self.lo).any()) and not bool(xp.isnan(self.hi).any()) and bool((self.lo<=self.hi).all()),'Bounds')
        need(rhs_fast.rate.shape==shape and bool((rhs_fast.rate[self.mask]==0).all()),'Fast flow must not integrate prescribed ports')
        self.stats={k:0 for k in ('attempts','accepted','rejected','slow_trial','slow_audit','fast_flows','projections')}
        self.log=[];self.last=None;self.deadline=math.inf
    def tick(self):
        if time.perf_counter()>self.deadline: raise TimeoutError('120 s replay budget')
    def check(self,y,domain=False):
        self.tick();x=self.xp
        need(isinstance(y,x.ndarray) and y.shape==self.shape and y.dtype==x.float64,'State layout/type')
        need(bool(x.isfinite(y).all()),'Nonfinite state')
        if domain: need(bool(((y>=self.lo)&(y<=self.hi)).all()),'State outside declared domain')
        return y
    def projected(self,t,y):
        self.stats['projections']+=1
        z=self.check(self.project(t,y.copy()))
        need(bool(self.xp.array_equal(z[~self.mask],y[~self.mask])),'Project changed a free state')
        return z.copy()
    def fs(self,t,y,audit=False):
        self.stats['slow_audit' if audit else 'slow_trial']+=1
        z=self.projected(t,y);v=self.check(self.slow(t,z.copy())).copy()
        need(bool((v[self.mask]==0).all()),'Slow RHS integrates a prescribed port')
        return v
    def flow(self,t,y,g0,g1,span,fraction=1.,endpoint=None):
        self.stats['fast_flows']+=1
        v=self.fast.flow(y,g0,g1,span,fraction)
        when=t+span*fraction if endpoint is None else endpoint
        return self.projected(when,self.check(v))
    def propose(self,t,stop,y):
        """No publication. Caller may discard all outputs without restoring y."""
        self.stats['attempts']+=1;H=stop-t;need(H>0,'Positive interval')
        a=t+H/3;b=t+2*H/3;need(t<a<b<stop,'Stage clock resolution')
        y0=self.projected(t,y);f0=self.fs(t,y0);zero=self.xp.zeros_like(y)
        y1=self.flow(t,y0,f0,zero,a-t,endpoint=a);f1=self.fs(a,y1)
        g2=-f0+2*f1
        y2=self.flow(a,y1,g2,zero,b-a,endpoint=b);f2=self.fs(b,y2)
        g3=-2*f1+3*f2;k3=1.5*(f0-f2)
        high=self.flow(b,y2,g3,k3,stop-b,endpoint=stop)
        low=self.flow(b,y2,.25*f0-f1+1.75*f2,zero,stop-b,endpoint=stop)
        sc=ATOL+RTOL*self.xp.maximum(abs(high),abs(low))
        error=scalar(self.xp.max(abs(high-low)[:self.normn]/sc[:self.normn]))
        intervals=((t,a,y0,f0,zero),(a,b,y1,g2,zero),(b,stop,y2,g3,k3))
        return high,low,error,intervals
    def audit(self,intervals,high,low):
        """True slow residual at three NON-STAGE points. Not a continuous bound."""
        x=self.xp;scale=ATOL+RTOL*x.maximum(abs(high),abs(low));score=0.;raw=0.
        for left,right,y,g0,g1 in intervals:
            h=right-left;tm=left+h/2
            z=self.flow(left,y,g0,g1,h,.5)
            residual=self.fs(tm,z,True)-(g0+.5*g1)
            raw=max(raw,scalar(x.max(abs(residual))))
            score+=h*scalar(x.max(abs(residual)[:self.normn]/scale[:self.normn]))
        return score,raw
    def replay(self,y0,t0,t1,events,initial_h,min_h=1e-7):
        """One 125-us maximum replay. Same y0 preserved on success/failure.
        Endpoint candidates only; caller retains all external owner transactions.
        """
        self.last=None
        need(0<t1-t0<=125.0000001e-6 and initial_h>0 and min_h>0,'Replay controls')
        ev=list(map(float,events));need(all(math.isfinite(e) and t0<=e<=t1 for e in ev),'Events')
        need(ev==sorted(ev),'Do not reorder event input')
        stops=sorted(set(ev+[t1]));started=time.perf_counter();self.deadline=started+120
        y=self.projected(t0,self.check(y0));self.check(y,True);t=t0;h=initial_h
        try:
            while t<t1:
                self.tick();need(self.stats['attempts']<10000,'Attempt budget')
                edge=next(e for e in stops if e>t);end=min(t+h,edge)
                high,low,e,parts=self.propose(t,end,y);accepted=e<=1.
                row={'t':t,'end':end,'error_3_2':e,'accepted':accepted}
                self.log.append(row)
                if accepted:
                    self.check(high,True);score,residual=self.audit(parts,high,low)
                    row.update(sampled_defect_scaled=score,sampled_rhs_defect_max=residual)
                    y=high;t=end;self.stats['accepted']+=1
                else: self.stats['rejected']+=1
                # Difference 3(2) is O(H^3); NOT the parent's /3 estimator.
                factor=2. if e==0 else max(.2,min(2. if accepted else .5,.9*e**(-1/3)))
                h=(end-row['t'])*factor
                if not accepted and h<min_h:raise RuntimeError('Accuracy minimum step')
                h=max(min_h,h)
            if self.xp.__name__=='cupy':self.xp.cuda.get_current_stream().synchronize()
            self.tick();self.last=y.copy()
            return y,{'status':'PROVISIONAL_REPLAY','stats':dict(self.stats),'steps':self.log,
                'wall_s':time.perf_counter()-started,'atol':ATOL,'rtol':RTOL,
                'continuous_defect_certified':False,'owner_state_committed':False,'stage_admission':False}
        except BaseException:
            self.last=None
            raise

def work_floor(t0,t1,events,parent_products):
    segments=1+len(set(float(e) for e in events if t0<e<t1))
    minimum=1+6*segments  # setup + 3 slow stages + 3 accepted-step audits
    return {'segments':segments,'minimum_coefficient_calls':minimum,
            'best_possible_count_ratio_this_code':parent_products/minimum,
            'assumption':'One full CSR per coefficient; no fast CSR; no rejections'}

def selftest():
    from scipy.linalg import expm
    T=125e-6;te=.9*T;y0=np.zeros(3);mask=np.array([1,1,0],bool)
    def project(t,y):
        u=max(0.,(t-te)/T)
        y[:2]=[0.,0.] if t<te else [.5*math.exp(-u),.5*u*math.exp(-u)]
        return y
    def coeff(t,y):return np.array([y[0],y[1],y[1]]),np.array([0.,0.,1/T])
    def build():
        sp=TargetRateSplit(coeff,project,0.,y0)
        return MRI33(sp.fast,sp.slow,project,y0.shape,(0.,1.),mask),sp
    matrix=np.array([[-1.,0.,0.],[1.,-1.,0.],[0.,1.,-1.]])/T
    truth=expm(matrix*(T-te))@np.array([.5,0.,0.])
    solver,sp=build();hi,lo,e,_=solver.propose(0.,T,y0)
    need(e==0 and hi[2]==0 and truth[2]>1e-4,'Missing late-event negative')
    negative={'estimate':e,'z':float(hi[2]),'reference_z':float(truth[2])}
    solver,sp=build();before=y0.tobytes();y,receipt=solver.replay(y0,0.,T,[te],T)
    err=scalar(np.max(abs(y-truth)/(ATOL+RTOL*np.maximum(abs(y),abs(truth)))))
    need(err<1 and y0.tobytes()==before,'Accuracy/input preservation')
    need(all(not(r['t']<te<r['end']) for r in receipt['steps']),'Event crossed')
    bad=MRI33(sp.fast,lambda t,y:np.full_like(y,np.nan),project,y0.shape,(0,1),mask)
    try:bad.replay(y0,0.,T,[te],T)
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    need(y0.tobytes()==before and bad.last is None,'Rollback failure')
    return {'scope':'CPU synthetic projected event, independent scipy.expm',
            'uncut_negative':negative,'cut_abs_error':float(abs(y-truth).max()),
            'cut_scaled_error':err,'coefficients_including_setup_and_audits':sp.calls,
            'receipt':receipt,'input_unchanged':True,'nonfinite_rejected':True,
            'CUDA_executed':False,'organism_executed':False}

if __name__=='__main__':
    import argparse,hashlib,json,traceback
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true',required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(exist_ok=False)
    try:result=dict(status='COMPLETE_CPU_SELFTEST',test=selftest())
    except BaseException:result=dict(status='FAILED_RETAINED',error=traceback.format_exc())
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(0 if result['status']=='COMPLETE_CPU_SELFTEST' else 2)
```

## 4. Uso y condiciones para el replay real

```bash
OPENBLAS_NUM_THREADS=1 python mri33_replay.py \
  --selftest --out prueba_mri33_01
```

En el arnés aislado, con callbacks y arrays reales ya preparados:

```python
from mri33_replay import TargetRateSplit, MRI33, work_floor

# Primero: puede descartar esta instancia por coste sin ejecutar GPU.
print(work_floor(t0, t1, event_times, parent_full_products))

split = TargetRateSplit(effective_coefficients, project, t0, y0, xp=cp)
solver = MRI33(
    split.fast, split.slow, project, y0.shape,
    bounds=(lower, upper), prescribed=port_mask,
    norm_size=parent_norm_size, xp=cp,
)
candidate, receipt = solver.replay(
    y0, t0, t1, event_times, initial_h=initial_h,
)
receipt["all_coefficient_calls_including_setup"] = split.calls
# No copiar candidate al organismo ni publicar sus eventos/checkpoint.
```

**No basta llamar a `g.coefficient` después de construir el grafo histórico:** `OrganismAdapter.build()` restaura el lector PN y los buffers al salir. El callback nuevo debe conservar el contexto efectivo correcto; no adivinar pesos, entradas retenidas o propietarios. 

El proyector debe ser puro: reconstruye únicamente los puertos declarados, conserva el orden SET/ADD y no altera estado físico externo. El código deduplica **tiempos de corte**, no operaciones. Si un evento modifica pesos o reglas, todas sus transiciones deben estar representadas por el callback efectivo y la agenda. No hay generación de eventos endógenos nueva dentro de este prototipo.

El límite de 120 s se comprueba entre operaciones: **no puede interrumpir un kernel bloqueado**. Los contadores incluyen rechazos y auditorías, pero las visitas CSR reales deben medirse en el operador: no asumir que cada callback cuesta exactamente una matriz si sus adaptadores hacen más recorridos.

## 5. Qué se ejecutó y qué queda abierto

**Leí completos:** README, `MULTIRATE_NEXT_GATE_01.json`, addendum y los `graph_core.py`/`organism_adapter.py` enlazados. Verifiqué la tabla ERK33a en la fuente oficial SUNDIALS y las fórmulas de MRIStep; **no descargué ni ejecuté la cápsula real**. La documentación también delimita MRIStep a la forma ODE: no autoriza a ignorar una masa no diagonal. :chatgpt-content-reference{index="7"}

**Ejecuté código propio CPU:** orden, comparación independiente, evento tardío, orden SET/ADD, tasa variable, rechazos, dominio y preservación del input. La última suite tomó **0,0396 s después de las importaciones**. No compilé CUDA ni ejecuté CuPy, PN, KC, MuJoCo o el organismo.

La afirmación de Gemini es incorrecta: **bitwise igual no demuestra ausencia de bugs**; dos implementaciones pueden compartir un defecto o no observar el estado relevante. Ruido biológico y error numérico pertenecen a contratos distintos. Un modo rápido puede probarse prospectivamente sin perfección universal, pero no justificarse después diciendo que la biología es ruidosa.

:chatgpt-content-reference{index="9"}[Paquete completo con código, pruebas ampliadas, resultados y procedencia](sandbox:/mnt/data/AXIOMA_MRI33_REPLAY_E995_CPU.zip)

SHA256 del módulo: `c8c3d75a6ed0d8bd908f41d24804dff4a25789f6da4b8bdeb2e617d6c42a2984`.

**La contribución es un MRI reproducible y un descarte económico temprano. No demuestra aún que este split tenga suficiente separación temporal ni que cumpla el gate de recu
