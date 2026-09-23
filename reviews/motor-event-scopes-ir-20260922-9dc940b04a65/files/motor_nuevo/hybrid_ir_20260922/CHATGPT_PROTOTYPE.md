**Entrego un prototipo B autocontenido de 209 líneas, ejecutado en CPU.** Ambos modelos completaron **un segundo simulado** y cumplieron el criterio externo de \(10^{-4}\). El control con realimentación congelada falló en ambos.

**El resultado de rendimiento es negativo:** esta implementación serial por trayectorias fue más lenta que integrar globalmente con DOP853. La conservo como **referencia funcional de acoplamiento y transacciones**, no como mejora de velocidad ni como ejecución del organismo.

## Resultado ejecutado

El proceso completo consumió **2,60 s de pared, 2,99 s de CPU y 121,04 MiB de memoria residente máxima**. Una corrida, sin reajustar tolerancias después de observar resultados.

| Modelo | Estados | Error máximo B frente a Radau | Error con realimentación congelada | Tiempo B¹ | Tiempo global DOP853¹ |
|---|---:|---:|---:|---:|---:|
| Lazo recurrente no lineal | 4 | \(6,624\times10^{-11}\) | **0,033661** | 0,6148 s | 0,0168 s |
| Reacción no lineal + oscilador | 5 | \(2,909\times10^{-10}\) | **0,009284** | 0,3192 s | 0,0178 s |

¹ Fase medida, incluido el guardado de sus arrays. Es una medición pequeña en orden fijo, no un benchmark estadístico.

También pasaron el rechazo sin publicar predictores, el rechazo transaccional por dominio, el orden no conmutativo ADD/SET y la masa no diagonal frente a una solución por exponencial matricial —error \(2,776\times10^{-17}\)—.

### Qué implementa exactamente

Cada bloque declara:

\[
M_i\dot x_i=F_i(t,x_i,u_i),\qquad
u_i(t)=\sum_j W_{ij}H_j(t,x_j)+u_{\mathrm{externo},i}(t).
\]

Los bloques intercambian interpolantes temporales mediante **Jacobi**: un barrido utiliza las trayectorias del anterior, no resultados parciales del mismo barrido. La masa se resuelve mediante LU; **no se diagonaliza**.

Cada llamada a `avanzar()` publica conjuntamente reloj, estados y eventos **solo después de completar el intervalo**. Los contadores diagnósticos sí incluyen intentos rechazados.

**Alcance:** eventos prescritos ADD/SET, masa constante no singular dentro de cada bloque, puertos escalares y callbacks puros. Se corta en cada evento. No admite masa entre bloques, DAE, retardos, emisión endógena, CUDA ni cuerpo. Los ejemplos están normalizados; **no contiene todavía un verificador dimensional ni sustituye vuestro IR**.

El error y los dominios se comprueban en **367 muestras por modelo**, incluidos ambos lados y el orden de los eventos. El test interno combina cambio de trayectoria y defecto de acoplamiento muestreado; **no es una cota rigurosa del error continuo**.

## Archivo completo: `trayectorias_cpu.py`

Requiere NumPy y SciPy. En WSL/Linux establece límites de CPU y memoria. La carpeta de resultados debe ser nueva.

```bash
python trayectorias_cpu.py resultado_B_cpu
```

```python
"""B-CPU: Jacobi por trayectorias. ODE, masa por bloque, eventos prescritos.
Unidades de los ejemplos: estados/masa adimensionales y tiempo en segundos.
No DAE, masa entre bloques, detección de espigas, retardos ni seguridad de callbacks.
"""
import os, sys, time, json, hashlib, traceback
from pathlib import Path
from dataclasses import dataclass
for nombre in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[nombre] = '1'
INICIO = time.process_time()
if __name__ == '__main__' and sys.platform.startswith('linux'):
    import resource, signal
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (59, 60))
    def agotado(*_): raise TimeoutError('Presupuesto CPU agotado')
    signal.signal(signal.SIGXCPU, agotado)
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve, expm
PLAN = dict(rtol=1e-9, atol=1e-11, convergencia=1e-9, iteraciones=24,
            error_externo=1e-4, ventana=.125, muestras_segmento=33,
            segundos=1., cpu_max_s=60, ram_max_bytes=2*1024**3)
def exigir(condicion, mensaje):
    if not condicion: raise ValueError(mensaje)
@dataclass
class Bloque:
    nombre: str
    masa: np.ndarray
    inicial: np.ndarray
    rhs: object                         # F(t, x_propio, entrada_escalar)
    salida: object                      # H(t, x_propio): escalar
    limites: tuple = (-10., 10.)
    escala: float = 1.
    def __post_init__(self):
        self.inicial = np.array(self.inicial, dtype=float, copy=True)
        self.masa = np.array(self.masa, dtype=float, copy=True)
        n = self.inicial.size
        exigir(self.inicial.shape == (n,) and self.masa.shape == (n,n), 'Forma de masa/estado')
        exigir(np.isfinite(self.masa).all() and np.linalg.cond(self.masa)<1e12, 'Masa singular/mal condicionada')
        exigir(np.isfinite(self.escala) and self.escala>0, 'Escala inválida')
        self.lu = lu_factor(self.masa); self.validar(self.inicial)
    def validar(self, x):
        exigir(np.isfinite(x).all() and np.all(x>=self.limites[0]) and np.all(x<=self.limites[1]), 'Dominio')
@dataclass(frozen=True)
class Evento:
    tiempo: float
    bloque: int
    coordenada: int
    operacion: str
    valor: float
class Sesion:
    def __init__(self, bloques, pesos, externo, eventos=()):
        self.bloques = tuple(bloques); self.W = np.array(pesos, dtype=float, copy=True)
        self.externo = externo; self.eventos = tuple(eventos)
        n = len(bloques)
        exigir(self.W.shape==(n,n) and np.isfinite(self.W).all(), 'Puertos')
        exigir(len({b.nombre for b in bloques})==n, 'Identidades repetidas')
        for e in eventos:
            exigir(type(e.bloque) is int and 0<=e.bloque<n, 'Bloque de evento')
            exigir(type(e.coordenada) is int and 0<=e.coordenada<bloques[e.bloque].inicial.size, 'Coordenada')
            exigir(np.isfinite([e.tiempo,e.valor]).all() and e.tiempo>0 and e.operacion in ('ADD','SET'), 'Evento')
        self.cortes = np.cumsum([0]+[b.inicial.size for b in bloques])
        self.contadores = dict(rhs=0, filas_puerto=0, solves_masa=0, iteraciones=0, rhs_control=0)
        iniciales = tuple(b.inicial.copy() for b in bloques)
        for x in iniciales: x.flags.writeable=False
        self.publicado = (0., iniciales, ())
    def entrada(self, i, t, xs):
        self.contadores['filas_puerto'] += 1
        valor = float(self.externo(t)[i])
        for j, b in enumerate(self.bloques):
            if self.W[i,j]: valor += self.W[i,j]*b.salida(t, xs[j])
        exigir(np.isfinite(valor), 'Puerto no finito')
        return valor
    def derivada(self, i, t, x, u):
        if time.process_time()-INICIO>55: raise TimeoutError('Reserva para guardar resultados')
        self.contadores['rhs'] += 1; self.contadores['solves_masa'] += 1
        b = self.bloques[i]; f = np.asarray(b.rhs(t,x,u), dtype=float)
        exigir(f.shape==x.shape and np.isfinite(f).all(), 'RHS inválido')
        return lu_solve(b.lu, f)
    def segmento(self, a, z, xs, modo, maxit):
        malla = np.linspace(a,z,PLAN['muestras_segmento'])
        if modo in ('global','referencia'):
            def funcion(t,y):
                estados = [y[l:r] for l,r in zip(self.cortes[:-1],self.cortes[1:])]
                return np.concatenate([self.derivada(i,t,x,self.entrada(i,t,estados)) for i,x in enumerate(estados)])
            fino = modo=='referencia'
            sol = solve_ivp(funcion,(a,z),np.concatenate(xs),method='Radau' if fino else 'DOP853',
                            rtol=1e-11 if fino else PLAN['rtol'],atol=1e-13 if fino else PLAN['atol'],dense_output=True)
            exigir(sol.success,sol.message)
            return sol.sol(malla), malla
        exigir(modo in ('iterado','congelado'), 'Modo desconocido')
        anterior = [lambda t,x=x.copy(): x for x in xs]
        for _ in range(maxit):
            self.contadores['iteraciones'] += 1; nuevas = []
            for i,b in enumerate(self.bloques):
                def funcion(t,x,i=i):
                    estados = [onda(t) for onda in anterior]; estados[i] = x
                    return self.derivada(i,t,x,self.entrada(i,t,estados))
                sol = solve_ivp(funcion,(a,z),xs[i],method='DOP853',rtol=PLAN['rtol'],atol=PLAN['atol'],dense_output=True)
                exigir(sol.success, sol.message); nuevas.append(sol.sol)
            error = max(float(np.max(np.abs(n(malla)-np.column_stack([v(t) for t in malla]))))/b.escala
                        for n,v,b in zip(nuevas,anterior,self.bloques))
            defecto = 0.
            if modo=='iterado':
                for t in malla:
                    actual = [onda(t) for onda in nuevas]; previa = [onda(t) for onda in anterior]
                    for i,b in enumerate(self.bloques):
                        usada = previa.copy(); usada[i] = actual[i]
                        self.contadores['rhs_control'] += 2
                        d = self.derivada(i,t,actual[i],self.entrada(i,t,actual))-self.derivada(i,t,actual[i],self.entrada(i,t,usada))
                        defecto = max(defecto,(z-a)*float(np.max(np.abs(d)))/b.escala)
            if modo=='congelado' or max(error,defecto)<=PLAN['convergencia']:
                return np.vstack([onda(malla) for onda in nuevas]), malla
            anterior = nuevas  # Jacobi: no usar actualizaciones parciales del mismo barrido.
        raise RuntimeError('Rechazo: no convergió la trayectoria; sin publicación')
    def avanzar(self, fin, modo='iterado', maxit=24):
        inicio, propios, confirmados = self.publicado
        exigir(np.isfinite(fin) and fin>inicio, 'Reloj')
        xs = [x.copy() for x in propios]; registro = list(confirmados); trazas=[]; tiempos=[]
        eventos = [(j,e) for j,e in enumerate(self.eventos) if inicio<e.tiempo<=fin]
        cortes = sorted({inicio,fin}|{e.tiempo for _,e in eventos})
        for a,z in zip(cortes[:-1],cortes[1:]):
            y,t = self.segmento(a,z,xs,modo,maxit)
            for i,b in enumerate(self.bloques): b.validar(y[self.cortes[i]:self.cortes[i+1]])
            trazas.extend(y.T.copy()); tiempos.extend(t.tolist())
            xs = [y[l:r,-1].copy() for l,r in zip(self.cortes[:-1],self.cortes[1:])]
            for j,e in eventos:         # El orden original decide eventos simultáneos.
                if e.tiempo!=z: continue
                x=xs[e.bloque]; k=e.coordenada
                x[k] = e.valor if e.operacion=='SET' else x[k]+e.valor
                self.bloques[e.bloque].validar(x); registro.append(j)
                trazas.append(np.concatenate(xs)); tiempos.append(z)  # Lado derecho explícito.
        for x in xs: x.flags.writeable=False
        self.publicado = (fin,tuple(xs),tuple(registro))  # Único commit del intervalo.
        return np.array(tiempos),np.array(trazas)
def modelos(caso):
    if caso==0:
        f=lambda t,x,u: np.array([-3*x[0]-.5*x[1]+2.5*np.tanh(u)+.2*np.sin(3*t), .6*x[0]-1.5*x[1]])
        g=lambda t,x,u: np.array([-2*x[0]+.4*x[1]-1.8*np.tanh(u), x[0]-4*x[1]])
        bs=[Bloque('uno',[[1,.2],[.2,1.5]],[.2,-.1],f,lambda t,x:x[0]),
            Bloque('dos',[[.8,-.1],[-.1,1.2]],[.1,.2],g,lambda t,x:x[0])]
    else:
        f=lambda t,x,u: np.array([1/(1+u*u)-x[0]-.3*x[0]*x[1], .5*x[0]-.8*x[1], x[1]-2*x[2]])
        g=lambda t,x,u: np.array([x[1],-4*x[0]-.4*x[1]-.8*x[0]**3+2*u+.1*np.cos(4*t)])
        bs=[Bloque('reaccion',np.diag([1.,.5,.25]),[.5,.3,.1],f,lambda t,x:x[2],(0.,10.)),
            Bloque('oscilador',[[1,.15],[.15,1]],[-.2,.25],g,lambda t,x:x[0])]
    ev=[Evento(.21,0,0,'ADD',.3),Evento(.57,1,0,'SET',-.25),Evento(.57,1,0,'ADD',.1),Evento(.87,0,0,'SET',.15)]
    return Sesion(bs,[[0.,1.],[1.,0.]],lambda t:np.zeros(2),ev)
def pruebas():
    s=modelos(0); antes=s.publicado
    try: s.avanzar(.125,maxit=1)
    except RuntimeError: pass
    else: raise AssertionError('Se aceptó sin convergencia')
    exigir(s.publicado is antes,'Se publicó un predictor rechazado')
    s.eventos=(Evento(.05,0,0,'SET',.7),Evento(.05,0,0,'SET',100.))
    try: s.avanzar(.1)
    except ValueError: pass
    else: raise AssertionError('Dominio no rechazado')
    exigir(s.publicado is antes,'Transacción parcial')
    b=Bloque('constante',[[1.]],[.8],lambda t,x,u:np.zeros(1),lambda t,x:x[0],(0.,1.))
    eventos=[Evento(.05,0,0,'SET',.7),Evento(.05,0,0,'ADD',.1)]
    for orden,esperado in ((eventos,.8),(eventos[::-1],.7)):
        ses=Sesion([b],[[0.]],lambda t:np.zeros(1),orden); ses.avanzar(.1)
        exigir(abs(ses.publicado[1][0][0]-esperado)<1e-14,'Orden ADD/SET')
    masa=np.array([[2.,.3],[.3,1.]])
    b=Bloque('masa',masa,[.4,-.2],lambda t,x,u:-x,lambda t,x:x[0])
    s=Sesion([b],[[0.]],lambda t:np.zeros(1)); s.avanzar(.125)
    exacto=expm(-np.linalg.solve(masa,np.eye(2))*.125)@b.inicial
    error=float(np.max(np.abs(s.publicado[1][0]-exacto)))
    exigir(error<1e-9,'Masa completa no conservada')
    return dict(rechazo_sin_publicar=True,dominio_transaccional=True,orden_ADD_SET=True,error_masa=error)
def main():
    salida=Path(sys.argv[1] if len(sys.argv)>1 else 'resultado_B_cpu'); salida.mkdir(exist_ok=False)
    resultado=dict(plan=PLAN,python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,modelos=[])
    resultado['sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (salida/'PLAN.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    pared=time.perf_counter()
    try:
        resultado['pruebas']=pruebas()
        for caso in (0,1):
            datos={}; registros={}; malla_comun=None
            for modo in ('referencia','global','iterado','congelado'):
                s=modelos(caso); trozos=[]; tiempos=[]; tic=time.perf_counter(); cpu=time.process_time()
                for fin in np.linspace(0.,PLAN['segundos'],1+round(PLAN['segundos']/PLAN['ventana']))[1:]:
                    t,y=s.avanzar(float(fin),modo,PLAN['iteraciones']); tiempos.extend(t); trozos.extend(y)
                datos[modo]=np.array(trozos)
                if malla_comun is None: malla_comun=np.array(tiempos)
                exigir(np.array_equal(malla_comun,tiempos),'Mallas de comparación distintas')
                np.savez(salida/f'modelo_{caso}_{modo}.npz',tiempo=tiempos,estado=datos[modo])
                registros[modo]=dict(pared_s=time.perf_counter()-tic,cpu_s=time.process_time()-cpu,**s.contadores)
                registros[modo]['eventos_confirmados']=list(s.publicado[2])
                (salida/f'medidas_{caso}.json').write_text(json.dumps(registros,indent=2),encoding='utf-8')
                exigir(s.publicado[2]==tuple(range(4)),'Eventos perdidos/duplicados')
            np.savez(salida/f'modelo_{caso}.npz',tiempo=tiempos,**datos)
            escala=np.concatenate([np.full(b.inicial.size,b.escala) for b in s.bloques])
            errores={k:float(np.max(np.abs(v-datos['referencia'])/escala)) for k,v in datos.items()}
            ok=errores['iterado']<=PLAN['error_externo'] and errores['global']<=PLAN['error_externo'] and errores['congelado']>PLAN['error_externo']
            resultado['modelos'].append(dict(caso=caso,error_max=errores,medidas=registros,contraste_correcto=ok))
            exigir(ok,'Contraste fallido; no se reajustan tolerancias')
        resultado['estado']='COMPLETO'
    except Exception:
        resultado['estado']='FALLO_CONSERVADO'; resultado['error']=traceback.format_exc()
    resultado.update(pared_s=time.perf_counter()-pared,cpu_s=time.process_time()-INICIO)
    resultado['rss_mib']=__import__('resource').getrusage(0).ru_maxrss/1024 if sys.platform.startswith('linux') else None
    resultado['alcance']='CPU serial; eventos prescritos; masa por bloque; error muestreado, no certificado continuo; no organismo'
    (salida/'RESULTADO.json').write_text(json.dumps(resultado,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(resultado,indent=2)); return 0 if resultado['estado']=='COMPLETO' else 1
if __name__=='__main__': sys.exit(main())
```

## Resultado JSON observado — resumen

Los errores también se recalcularon desde los arrays guardados, **sin otra integración**.

```json
{
  "estado": "COMPLETO",
  "ejecucion": "CPU local; una corrida sin reajustes",
  "python": "3.13.5",
  "numpy": "2.3.5",
  "scipy": "1.17.0",
  "codigo_lineas": 209,
  "codigo_sha256": "a43e828372091f7895992bf638e29a2d4ef75927d4bc875e2ec62103aa8ec287",
  "proceso_pared_s": 2.60,
  "proceso_cpu_s": 2.99,
  "rss_max_mib": 121.04296875,
  "modelos": [
    {
      "caso": 0,
      "estados": 4,
      "horizonte_s": 1.0,
      "muestras_incluidos_lados_evento": 367,
      "error_iterado": 6.623768200597624e-11,
      "error_global": 6.916511807730785e-11,
      "error_congelado": 0.03366065270843771,
      "barridos": 79,
      "rhs_iterado_incluido_control": 15574,
      "rhs_control_acoplamiento": 10428,
      "rhs_integracion_iterado": 5146,
      "rhs_integracion_global": 674
    },
    {
      "caso": 1,
      "estados": 5,
      "horizonte_s": 1.0,
      "muestras_incluidos_lados_evento": 367,
      "error_iterado": 2.909000540451956e-10,
      "error_global": 3.423720473794134e-10,
      "error_congelado": 0.009284370603968362,
      "barridos": 44,
      "rhs_iterado_incluido_control": 8729,
      "rhs_control_acoplamiento": 5808,
      "rhs_integracion_iterado": 2921,
      "rhs_integracion_global": 764
    }
  ],
  "pruebas": {
    "rechazo_sin_publicar": true,
    "dominio_transaccional": true,
    "orden_ADD_SET": true,
    "error_masa": 2.7755575615628914e-17
  },
  "alcance": "CPU serial; eventos prescritos; masa por bloque; error muestreado, no certificado continuo; no organismo"
}
```

## Interpretación y compatibilidad

**No ahorra evaluaciones en esta implementación.** Incluso descontando las comprobaciones del defecto de acoplamiento, B ejecutó **5.146 frente a 674** evaluaciones de RHS en el primer modelo y **2.921 frente a 764** en el segundo. La ejecución serial repite integraciones y consultas de interpolantes; no hay aquí comunicaciones distribuidas costosas que compensen esa repetición.

El antecedente primario de Hahne y colaboradores describe precisamente ese intercambio: la relajación de formas de onda puede reducir comunicación a costa de más cómputo, y su beneficio depende de la escala. **Nuestro negativo no refuta todas las implementaciones multirritmo; sí descarta presentar este prototipo como aceleración local.** :chatgpt-content-reference{index="0"}

DOP853 y Radau aportan integradores e interpolantes distintos para la comparación. Sin embargo, **comparten las funciones del modelo, la resolución de masa y el publicador de eventos**; no son dos reconstrucciones independientes de las ecuaciones. Las pruebas analíticas de masa y ADD/SET controlan parte de ese riesgo común. :chatgpt-content-reference{index="1"}

Para conectarlo posteriormente a vuestro IR, `Bloque.rhs` debe recibir **\(F\) antes de resolver la masa**: `Model.raw_rhs`/el equivalente generado, no `Model.rhs` si ya devuelve \(M^{-1}F\). El código publicado distingue ambas operaciones.  La semántica SET sin salto positivo está alineada con el `reset_ledger.py` corregido; este prototipo permite ADD firmado según el dominio declarado, sin atribuir esa ampliación al contrato LIF heredado. 

**Lecturas realizadas:** `organism_adapter.py` de `77611a3`, `coupled.py` de `3a5ea1f`, `model.py` líneas 130–280 y `reset_ledger.py` de `cb253a7`, más los índices consultados. Las rutas solicitadas para `block_midpoint.py` y `event_coupling.py` devolvieron 404; no atribuyo su lectura ni ejecución. **No ejecuté ningún componente de OpenMatrix ni recursos del organismo.**

El :chatgpt-content-reference{index="5"}[paquete completo con código, protocolo, arrays, logs y manifiesto](sandbox:/mnt/data/MATRIX_B_CPU_TRAYECTORIAS.zip) conserva la ejecución; también están disponibles el :chatgpt-content-reference{index="6"}[archivo Python](sandbox:/mnt/data/prototipo_B_cpu/trayectorias_cpu.py) y el :chatgpt-content-reference{index="7"}[resultado completo](sandbox:/mnt/data/prototipo_B_cpu/ejecucion_01/RESULTADO.json).

**Decisión:** conservar este B como comparador funcional y negativo de coste. L

