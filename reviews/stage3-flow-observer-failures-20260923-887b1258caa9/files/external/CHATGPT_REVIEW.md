**Mantendría el conectoma sin simetrizaciones y priorizaría medir el flujo efectivo hacia DNa02 en el pipeline protegido.** Los datos justifican ampliar prospectivamente el horizonte, pero **no seleccionar una ventana que convierta el giro observado en éxito**.

## 1. Ventana de 250–400 ms: qué permiten realmente las trazas

`TIME_WINDOWS.json` mide **100/200/300 ms después del ON a 11 ms**: corresponden a los instantes 111/211/311 ms del registro. Los datos hasta 335 ms cubren solamente **324 ms pos-ON**. No permiten afirmar qué sucede a 400 ms, que exista una meseta ni localizar el inicio de una respuesta sin un criterio de detección prefijado.  

Además, **derecha−sham = +0,0193186° significa más giro positivo que sham, no orientación correcta hacia la derecha**. A 300 ms pos-ON, los desplazamientos son izquierda **+0,162518°**, derecha **+0,185270°**, uniforme **+0,174588°** y sham **+0,165952°**. El contraste bilateral calculado desde esas cifras es:

\[
D_\psi=\frac{\Theta_{\mathrm{izquierda}}-\Theta_{\mathrm{derecha}}}{2}
=\mathbf{-0,01137635^\circ}.
\]

No reconstruí estas cifras desde los NPZ; calculé el contraste sobre la tabla publicada. 

**Para una nueva corrida congelaría:** horizonte de 400 ms desde el ON realmente consumido; resultado primario \(\Theta_c=\psi_c(t_\mathrm{ON}+400)-\psi_c(t_\mathrm{ON})\) y \(D_\psi\). Siempre acompañados por los cuatro \(\Theta_c\), contrastes frente a sham/uniforme y comprobación de signos absolutos opuestos. **Un \(D_\psi\) favorable por sí solo no basta.** La ventana tardía 250–400 ms sería secundaria y predeclarada, no elegida después.

## 2. Topología, Olsen–Wilson y candidatos LAL/MBON

**No compensaría manualmente las tres rutas derechas frente a cero izquierdas.** Son caminos de exactamente dos saltos en el grafo sin signo, no tres canales funcionales de fuerza comparable ni ausencia de rutas más largas. El código y el resultado lo delimitan así.  

Un matiz importante: **Olsen–Wilson sí estimuló la antena contralateral y observó supresión de EPSC ORN→PN**. Pero lo utilizó para estudiar inhibición interglomerular y presináptica; no demuestra una compensación que debamos añadir para equilibrar esos caminos estructurales. :chatgpt-content-reference{index="5"}

**LAL170/171 y MBON32 merecen observación, no prioridad automática de intervención.** LAL170 permanece en valores subnormales; LAL171 y MBON32 muestran poca discriminación tardía en `q`. Eso debilita una explicación basada exclusivamente en su actividad somática contemporánea, pero no determina sus transmisiones filtradas o contribuciones previas. MBON32 tiene motivación anatómica independiente como vía convergente a DNa02; no es su única entrada.   :chatgpt-content-reference{index="8"}

### Convención de corriente

Ambas expresiones son compatibles si se usa coherentemente su signo en la ecuación:

\[
I_{\mathrm{saliente}}=g(V-E),\quad C\dot V=\cdots-I_{\mathrm{saliente}},
\]

\[
I_{\mathrm{entrante}}=g(E-V),\quad C\dot V=\cdots+I_{\mathrm{entrante}}.
\]

Con \(g\) en nS y voltajes en mV, la corriente queda en pA. **No asignar signo negativo a \(g\)** para representar inhibición. Registrar también conductancia: una entrada puede modificar la respuesta eléctrica aun cuando su corriente instantánea sea pequeña. Las sumas abstractas `transmission × cap × weight` conservan sus unidades de modelo; no se renombran pA. Con masa no diagonal, distinguir igualmente \(F\) de \(M^{-1}F\).

## 3. Qué falta y cómo instrumentarlo

| Hipótesis | Información indispensable |
|---|---|
| **A. Periferia/ALLN** | Liberación ORN, modulación presináptica, entradas receptoras de DM1_lPN y **salida PN efectivamente consumida**, diferenciando ruta fina y legacy. |
| **B. Convergencia posterior** | Contribuciones firmadas por ruta hacia DNa02, después de sustituciones, con el resto de entradas contabilizado; no solo `q` y contactos. |
| **C. Basal/mando** | Estado preparado, exposición consumida, entrada/estado DNa02, mando solicitado/aplicado y movimiento corporal bajo la misma procedencia. |

**Instrumentación mínima:** capturar en los consumidores existentes, sin reevaluar ecuaciones; conservar propietario, ruta, receptor, tiempo y fase. Separar predictores de etapas confirmadas. Para conductancias, registrar \(g,V,E\); para entradas abstractas, el término efectivo. La suma de grupos más el resto debe reconstruir la entrada realmente utilizada. La inhibición presináptica puede aparecer como reducción de liberación ORN, **no como una corriente negativa ALLN→PN independiente**.

Primero comprobar observador **on/off** sin cambios de estados/eventos; después comparar con refinamiento temporal hasta la ventana interpretada, manteniendo \(10^{-4}\) y los demás criterios vigentes. No interpretar un efecto menor que la discrepancia numérica correspondiente. El historial publicado advierte que las ramas antiguas y actuales no son intercambiables. 

---

## Analizador ejecutable: `analizar_flujo.py`

**Contrato nuevo de exportación, no archivos de flujo que afirme ya publicados.** Procesa una interfaz bilateral por ejecución; no acepta `q` como sustituto de corriente.

Cada NPZ debe contener:

- `t_ns[T]`, `target_ids[2]` en orden **L,R**, `groups[G]`, `committed[T]`.
- `value[T,2,G]`: contribuciones firmadas; `other[T,2]`: todo lo no listado; `consumed[T,2]`: total capturado **independientemente en el consumidor**.
- Metadatos escalares `arm`, `interface`, `unit` y las cuatro identidades exigidas por el código.
- Alternativamente, modo `conductance_inward`: `g_nS`, `V_mV`, `E_mV`, todos `[T,2,G]`, en lugar de `value`. Si un grupo mezcla compartimentos o inversiones, sumar primero sus corrientes individuales, **no multiplicar promedios**.

La media trapezoidal es descriptiva sobre muestras; no certifica flujo integrado entre ellas. `reconstruction_atol` verifica contabilidad, **no sustituye \(10^{-4}\)**.

```python
"""Lectura offline: una interfaz bilateral, cuatro brazos, muestras confirmadas.
No simula, no modifica ecuaciones, no estima causalidad ni error entre muestras.
"""
import argparse, csv, hashlib, json, tempfile, traceback, zipfile
from pathlib import Path
import numpy as np
BRAZOS = ('odor_left', 'odor_right', 'uniform', 'sham')
IDENTIDAD = ('pipeline_sha256', 'operator_sha256', 'routes_sha256', 'initial_sha256')
def exigir(ok, mensaje):
    if not ok: raise ValueError(mensaje)
def guardar(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def huella(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def escalar(z, k):
    exigir(k in z and z[k].shape == (), 'Falta metadato escalar: '+k)
    return str(z[k].item())
def cargar(p, plan, brazo):
    with zipfile.ZipFile(p) as f:
        exigir(sum(x.file_size for x in f.infolist()) <= 128*1024**2, 'NPZ supera 128 MiB descomprimido')
    with np.load(p, allow_pickle=False) as f: z = {k: f[k].copy() for k in f.files}
    for k, esperado in [('arm', brazo), ('interface', plan['interface']), ('unit', plan['unit'])]:
        exigir(escalar(z, k) == esperado, 'Metadato distinto: '+k)
    for k in IDENTIDAD:
        exigir(escalar(z, k) == plan[k], 'Identidad distinta: '+k)
    t, ids, grupos = z['t_ns'], z['target_ids'], z['groups']
    exigir(t.ndim == 1 and len(t)>1 and t.dtype.kind in 'iu', 'Reloj entero requerido')
    exigir((t>=0).all() and (t[1:]>t[:-1]).all(), 'Reloj no creciente')
    exigir(ids.shape == (2,) and ids.dtype.kind in 'iu' and len(np.unique(ids))==2, 'Dos destinos distintos requeridos')
    exigir(np.array_equal(ids, plan['target_ids_L_R']), 'Orden/identidad de destinos L/R')
    exigir(grupos.ndim == 1 and len(grupos)>0 and grupos.dtype.kind in 'US', 'Grupos textuales requeridos')
    exigir(len(np.unique(grupos))==len(grupos) and all(str(g) and not str(g).startswith('__') for g in grupos), 'Grupos duplicados/reservados')
    exigir(z['committed'].shape == t.shape and z['committed'].dtype.kind == 'b' and z['committed'].all(), 'Muestras no confirmadas')
    forma = (len(t), 2, len(grupos))
    if plan['mode'] == 'conductance_inward':
        exigir(plan['unit'] == 'pA', 'nS*mV debe expresarse en pA')
        g, v, e = [z[k] for k in ('g_nS', 'V_mV', 'E_mV')]
        exigir(all(a.shape == forma and a.dtype.kind in 'fiu' and np.isfinite(a).all() for a in (g,v,e)), 'Conductancia/voltajes invalidos')
        exigir((g>=0).all(), 'Conductancia negativa')
        valores = g.astype(float)*(e.astype(float)-v.astype(float))
    else:
        valores = z['value'].astype(float)
        exigir(valores.shape == forma, 'Forma value incorrecta')
    otro, usado = z['other'], z['consumed']
    exigir(otro.shape == usado.shape == (len(t),2), 'Forma de other/consumed incorrecta')
    for a in (valores, otro, usado):
        exigir(a.dtype.kind in 'fiu' and np.isfinite(a).all(), 'Valor no finito/no numerico')
    error = float(np.max(np.abs(valores.sum(axis=2)+otro-usado)))
    exigir(error <= plan['reconstruction_atol'], 'No reconstruye la entrada consumida: '+str(error))
    return t, grupos, valores, otro, usado, error

def analizar(plan_path, out):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    exigir(plan['mode'] in ('conductance_inward','effective_input'), 'Modo no admitido')
    ids_plan = plan['target_ids_L_R']
    exigir(len(ids_plan)==2 and all(type(i) is int and i>=0 for i in ids_plan) and ids_plan[0]!=ids_plan[1], 'IDs L/R enteros y distintos')
    exigir(set(plan['arms']) == set(BRAZOS) and plan['sample_kind']=='committed_endpoint', 'Brazos/muestras incorrectos')
    exigir(isinstance(plan['interface'],str) and plan['interface'] and isinstance(plan['unit'],str) and plan['unit'], 'Interfaz/unidad')
    for k in IDENTIDAD:
        s = plan[k]; exigir(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s), 'SHA256 invalido: '+k)
    exigir(type(plan['onset_ns']) is int and plan['onset_ns']>=0, 'Onset entero no negativo')
    ventana = plan['window_after_onset_ns']
    exigir(len(ventana)==2 and all(type(i) is int for i in ventana) and 0<=ventana[0]<ventana[1], 'Ventana temporal')
    tolerancia = plan['reconstruction_atol']
    exigir(np.isfinite(tolerancia) and tolerancia>0, 'Tolerancia algebraica invalida')
    out.mkdir(exist_ok=False); guardar(out/'CONTRATO.json', plan)
    datos=[]; recibos={}; rutas=set()
    try:
        for brazo in BRAZOS:
            item = plan['arms'][brazo]; p = (plan_path.parent/item['file']).resolve()
            exigir(p not in rutas, 'Mismo archivo en dos condiciones'); rutas.add(p)
            digest=huella(p)
            if item.get('sha256'): exigir(digest==item['sha256'], 'Archivo modificado')
            t, grupos, valor, otro, usado, error = cargar(p,plan,brazo)
            if datos:
                exigir(np.array_equal(t,t0) and np.array_equal(grupos,g0), 'Tiempos/grupos distintos; no se alinean')
            else: t0,g0=t,grupos
            datos.append(np.concatenate((valor,otro[:,:,None],usado[:,:,None]),axis=2))
            recibos[brazo]={'sha256':digest,'reconstruction_max_abs':error}
        a,b=[plan['onset_ns']+i for i in ventana]
        ia,ib=np.flatnonzero(t0==a),np.flatnonzero(t0==b)
        exigir(len(ia)==len(ib)==1, 'Ventana incompleta: se requieren ambos extremos exactos')
        i,j=int(ia[0]),int(ib[0]); dt=np.diff(t0[i:j+1]).astype(float)
        promedios=[]
        for x in datos:
            y=x[i:j+1]; promedios.append(np.sum((y[:-1]+y[1:])*.5*dt[:,None,None],axis=0)/float(b-a))
        medias=np.array(promedios); bilateral=medias[:,0,:]-medias[:,1,:]
        nombres=list(map(str,g0))+['__OTHER__','__TOTAL_CONSUMED__']; filas=[]
        for k,nombre in enumerate(nombres):
            L,R,U,S=bilateral[:,k]
            fila={'group':nombre,'unit':plan['unit'],'odd_L_R':float((L-R)/2),
                  'common_vs_sham':float((L+R)/2-S),'L_sham':float(L-S),
                  'R_sham':float(R-S),'U_sham':float(U-S),'R_uniform':float(R-U)}
            for h,brazo in enumerate(BRAZOS):
                fila['input_L_'+brazo]=float(medias[h,0,k]); fila['input_R_'+brazo]=float(medias[h,1,k])
            filas.append(fila)
        with (out/'CONTRASTES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
        resultado={'status':'SUMA_Y_CONTRASTES_RECONSTRUIDOS','inputs':recibos,'total':filas[-1],
            'window_absolute_ns':[a,b],'samples_used':j-i+1,'source_sha256':huella(Path(__file__)),
            'limits':'Media trapezoidal entre muestras; no cota continua, causalidad o autenticidad de la captura. other no se resta del total. Una interfaz por ejecucion.',
            'stage3_admission':False}
        guardar(out/'RESULTADO.json',resultado); return resultado
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'inputs':recibos}); raise

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp); firma=hashlib.sha256(b'fixture_no_organismo').hexdigest()
        plan={'mode':'conductance_inward','interface':'fixture','unit':'pA','sample_kind':'committed_endpoint',
              'target_ids_L_R':[10,20],'onset_ns':0,'window_after_onset_ns':[0,335000000],
              'reconstruction_atol':1e-10,'arms':{},**{k:firma for k in IDENTIDAD}}
        for brazo in BRAZOS:
            t=np.array([0,100000000,200000000,300000000,335000000],dtype=np.int64)
            g=np.ones((5,2,2));v=np.full_like(g,-60.);e=np.zeros_like(g);e[:,:,1]=-70.
            if brazo in ('odor_left','uniform'):g[:,0,0]=2.
            if brazo in ('odor_right','uniform'):g[:,1,0]=2.
            valor=g*(e-v);z=dict(t_ns=t,target_ids=np.array([10,20]),groups=np.array(['exc','inh']),
                committed=np.ones(5,dtype=bool),g_nS=g,V_mV=v,E_mV=e,other=np.zeros((5,2)),consumed=valor.sum(2),
                arm=np.array(brazo),interface=np.array('fixture'),unit=np.array('pA'),**{k:np.array(firma) for k in IDENTIDAD})
            np.savez(r/(brazo+'.npz'),**z);plan['arms'][brazo]={'file':brazo+'.npz'}
        p=r/'plan.json';guardar(p,plan);res=analizar(p,r/'ok')
        exigir(abs(res['total']['odd_L_R']-60.)<1e-12,'Signo/unidad incorrectos')
        objetivo=r/'odor_right.npz'; original=objetivo.read_bytes();rechazos=[]
        for caso in ('consumed','ids','nonfinite','uncommitted','outward_sign','window400'):
            with np.load(objetivo,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
            if caso=='consumed':z['consumed'][0,0]+=.01
            elif caso=='ids':z['target_ids']=z['target_ids'][::-1]
            elif caso=='nonfinite':z['g_nS'][0,0,0]=np.nan
            elif caso=='uncommitted':z['committed'][-1]=False
            elif caso=='outward_sign':z['consumed']*=-1
            else:plan['window_after_onset_ns']=[250000000,400000000]
            np.savez(objetivo,**z);guardar(p,plan)
            try:analizar(p,r/caso)
            except ValueError:rechazos.append(caso)
            else:raise AssertionError('Corrupcion aceptada: '+caso)
            objetivo.write_bytes(original);plan['window_after_onset_ns']=[0,335000000]
        return {'scope':'SOLO aritmetica CPU sintetica; no OpenMatrix','odd_pA':60.,'rejected':rechazos}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selftest',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest: print(json.dumps(selftest(),indent=2))
    else:
        exigir(a.plan is not None and a.out is not None,'Se requieren --plan y --out')
        print(json.dumps(analizar(a.plan.resolve(),a.out),indent=2,ensure_ascii=False))
```

### Comandos y contrato

```bash
python analizar_flujo.py --selftest
python analizar_flujo.py --plan contrato.json --out flujo_01
```

Ejemplo de `contrato.json`; completar los hashes con identidades reales y fijar el ON correspondiente al nuevo protocolo:

```json
{
  "mode": "effective_input",
  "interface": "DNa02:entrada_nativa",
  "unit": "model_native_reference_v1",
  "sample_kind": "committed_endpoint",
  "target_ids_L_R": [523769, 10360],
  "onset_ns": 11000000,
  "window_after_onset_ns": [0, 400000000],
  "reconstruction_atol": 1e-8,
  "pipeline_sha256": "REEMPLAZAR_CON_SHA256_REAL_64_HEX",
  "operator_sha256": "REEMPLAZAR_CON_SHA256_REAL_64_HEX",
  "routes_sha256": "REEMPLAZAR_CON_SHA256_REAL_64_HEX",
  "initial_sha256": "REEMPLAZAR_CON_SHA256_REAL_64_HEX",
  "arms": {
    "odor_left": {"file": "odor_left_flow.npz"},
    "odor_right": {"file": "odor_right_flow.npz"},
    "uniform": {"file": "uniform_flow.npz"},
    "sham": {"file": "sham_flow.npz"}
  }
}
```

**Ejecutado aquí:** dos selftests aritméticos CPU; la versión final terminó en **1,48 s de pared y 1,47 s de CPU**. Rechazó seis casos: total incompatible, destinos invertidos, no finitos, muestra no confirmada, signo de corriente invertido y ventana incompleta hasta 400 ms. No ejecuté los cuatro nuevos archivos de flujo: **todavía no están en los paquetes examinados bajo este contrato**.

SHA256 del Python: `db74b46d18770475df8eb4d77de7d05e731d37aea1abffd4208360fa93915884`.

:chatgpt-content-reference{index="11"}[Fuente y pruebas completas](sandbox:/mnt/data/AXIOMA_FLUJO_FIRMADO_20260923.zip) · :chatgpt-content-reference{index="12"}[Python](sandbox:/mnt/data/AXIOMA_FLUJO_20260923/analizar_flujo.py)

**Lectura efectiva:** completos los índices, `ADDENDUM.md`, `probe_topology.py`, `TIME_WINDOWS.json` y `read_time_windows.py`; secciones inicial y final de `TOPOLOGY_EXPLORATORY.json`, y líneas 62–final de `FOCUS.csv`. Consulté las fuentes primarias citadas. **No descargué ni verifiqué los ZIP publicados, no ejecuté sus NPZ ni el organismo.** La conclusión es instrumentar y contrastar, no aprobar etapa 3 ni cambiar el conectoma.
