**B tiene fundamento como método local de segundo orden en tramos suaves, pero no hay evidencia de que ahorre trabajo en el organismo.** Entrego una variante aislada con dos evaluaciones y proyecciones explícitas. El control CPU dio un negativo importante: **2.716 evaluaciones frente a 240 del padre**, con las mismas tolerancias; B sobre-resolvió el ejemplo. No ajusté parámetros para rescatarlo.

## A/B/C y fundamento de B

**A** conserva la recuperación nominal ya entregada. **B** reduce evaluaciones por intento. **C** reduce trabajo espacial mediante una cota del efecto de la aproximación. No combino A con B en este prototipo.

En las fuentes, la tasa puede depender del estado: por ejemplo, la membrana visual usa `total/SENSOR_MEMBRANE_TAU_S`. Por eso B **reevalúa tanto \(a_M\) como \(b_M\)**. No congela globalmente una \(\lambda\) ni reemplaza los propietarios de membrana/masa por ecuaciones de tasa. 

Con \(f=b\odot(a-y)\) y \(D=\partial_t+f\cdot\nabla_y\), la expansión de la propuesta es:

\[
y_2=y+hf+\frac{h^2}{2}
\left[b\odot Da+(a-y)\odot Db-b\odot f\right]+O(h^3),
\]

que coincide con la expansión de la solución suave. Mientras tanto,

\[
y_2-y_E=\frac{h^2}{2}
\left[b\odot Da+(a-y)\odot Db\right]+O(h^3).
\]

**La diferencia estima el defecto dominante de Euler, no el error local del método superior ni una cota de la solución.** Puede cancelarse. Rectificaciones, cambios de régimen y eventos limitan la aplicación de esa expansión.

La norma propuesta es:

\[
e=\max_{i<\mathrm{norm\_size}}
\frac{|y_{2,i}-y_{E,i}|}
{\mathrm{atol}+\mathrm{rtol}\max(|y_{2,i}|,|y_{E,i}|)}.
\]

**Sin divisor 3.** Conservo el controlador binario padre: aceptar \(e\le1\), dividir tras rechazo y duplicar si \(e<0,1\). No utiliza exponente \(1/3\); para un defecto cuadrático, duplicar predice aproximadamente \(4e\), pero el siguiente paso se evalúa siempre.

### Proyecciones, dominio y costes

Se proyecta al inicio, antes del coeficiente de mitad y en **ambos endpoints** \(y_E,y_2\). Todas las fronteras conocidas permanecen en el controlador. El código exige agenda explícita —`[]` cuando no hay eventos—. Los arrays de Euler y mitad se calculan antes de que la segunda llamada reutilice buffers de coeficientes.

Si \(b\ge0\) y los targets de ambas etapas pertenecen al intervalo declarado, las actualizaciones son combinaciones convexas y preservan ese intervalo **algebraicamente**. No se demuestra que todos los targets reales satisfagan esas condiciones ni se certifica el redondeo. Se conserva el rechazo por dominio/no finitos, **sin clipping añadido**.

El coste de coeficientes es:

\[
C_P=6N_P,\qquad C_B=2N_B,
\]

contando aceptaciones **y rechazos**. Si \(N_B\ge3N_P\), desaparece el ahorro de evaluaciones. A igual número de intentos, 3× es solamente el máximo en esa parte del trabajo, no en el motor completo. 

## Resultado ejecutado

Un control analítico de dos coordenadas, con tasas dependientes del estado y del tiempo, \(T=1\) s, `rtol=1e-6`, `atol=1e-9`:

| CPU NumPy | Aceptados | Rechazados | Coeficientes | Error final verdadero |
|---|---:|---:|---:|---:|
| Padre | 37 | 3 | 240 | \(2,48937\times10^{-6}\) |
| B | 1.350 | 8 | 2.716 | \(9,62216\times10^{-9}\) |

Es comparación **a iguales tolerancias, no a igual error observado**. El defecto de orden inferior obligó a B a trabajar mucho más.

También se conservó el negativo temporal: evento a \(0,9h\), sin corte, **estimador 0 y error real 0,00226209**. Con la frontera, B obtuvo error \(2,23\times10^{-13}\), pero necesitó **4.120 evaluaciones** en ese control. No se elimina el evento para abaratarlo.

## Código completo: `generar_b2.py`

Genera `graph_core_B.py`, una copia del padre y del C++ **sin modificar el controlador**. El selftest ejecuta por AST los métodos aritméticos generados con NumPy; no importa CuPy ni ejecuta CUDA.

```python
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
```

### Uso y aislamiento

```bash
python generar_b2.py \
  /ruta/source/graph_core.py \
  /ruta/probe/generated/graph_control_parent.cpp \
  B2_01 --selftest
```

En el **arnés hijo**, importar `NativeGraph` desde `graph_core_B` y pasar explícitamente la biblioteca del **controlador padre**, no la candidata nominal A. No sobrescribir archivos operativos.

Dos dependencias de instrumentación deben corregirse explícitamente en ese arnés:

- `organism_adapter.py` contabiliza seis evaluaciones: usar `g.evaluations_per_trial` para B, sin modificar cómputo biológico.
- La construcción hace dos calentamientos y una captura: **6 llamadas de coeficientes**, no 18. La última etapa observada es **0,5**, no 0,75. No desactivar el validador del tap; declarar el contrato correcto.  

La generación mantiene el rollback de `advance`, el controlador C++ y la semántica del proyector. Su neutralidad con los propietarios reales y CUDA Graph **aún debe comprobarse**; los tests NumPy no certifican aliasing o sincronización GPU.

## Falsador de una única carga real futura

Después de las referencias intactas: **un hijo B de 1 ms**, comparado con un padre ya guardado que coincida en checkpoint, operador efectivo, preparación y entradas. Si esa correspondencia no existe, no utilizar otro padre aproximado.

Mantener los límites numéricos y de ejecución previamente fijados. Registrar aceptados, rechazados, evaluaciones, eventos, pared nativa y total. **Descartar B como ahorro de RHS en esa consulta si \(2N_B\ge6N_P\)**; también si incumple precisión, dominio o presupuesto. No cambiar `rtol`, `atol`, controlador o agenda para obtener ventaja. Aun con menos evaluaciones, debe ganar tiempo total: no basta el conteo.

## Qué falta para C y alcance de esta entrega

El par real q/s demuestra cambios graduados extensos, no elegibilidad espacial. Para C faltan el defecto temporal de la aproximación, pesos/ganancias **efectivos**, derivadas o cotas de sensibilidad del receptor, transiciones, masa y amplificación por recurrencia. Se necesita algo del tipo

\[
\|\delta x(t)\|
\le\int_0^t\|\Phi(t,s)\|\,\|r(s)\|\,ds,
\]

con \(r\) y el propagador/cota de influencia justificados. Un umbral de \(\Delta s\) entre dos snapshots no proporciona ninguno de los dos. 

**Leídos:** README y `density/RESULT.json` nuevos; completos `graph_core.py`, `gpu_coefficient_layout.py` y `organism_adapter.py` fijados. Materialicé el texto de `graph_core.py` y comprobé su blob Git y SHA256. El C++ padre se recuperó del artefacto anterior y coincide con el SHA exigido.

**Ejecutado:** una generación y un selftest CPU; proceso completo **1,17 s de pared, 1,55 s CPU y 92,9 MiB**. Sintaxis Python generada comprobada; **no compilé CUDA ni ejecuté el controlador C++, CuPy, el organismo o `state_pair.npz`**. No hubo fallos de test; el negativo de coste queda conservado.

SHA256 de `graph_core_B.py`:  
`aed03732ec7e971f883f6e4beddccc00a2b66675472af358ac0bba27b5dc7c43`

:chatgpt-content-reference{index="5"}[Paquete con generador, fuentes, diff, arrays y resultados](sandbox:/mnt/data/AXIOMA_B2_EMBEDDED_CPU.zip)

**B queda como rival falsable, no como mejora demostrada.** Los negativos históricos de ETDRK4/MOTOR14 permanecen intactos; esta prueba no califica la confirmación funcional ni modifica sus tolerancias.
