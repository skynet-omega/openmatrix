**La evidencia favorece comprobar primero C —lector y respuesta corporal—, sin abandonar A —adaptador PN— ni B —convergencia central—.** El cuerpo recibe comandos distintos según el olor, pero el comando derecho integrado casi cancela el sesgo positivo del sham: no produce un giro derecho absoluto.

**No pude descargar el ZIP ni ejecutar sus NPZ.** Sí leí las fuentes solicitadas, ejecuté cálculos sobre `CLOSE.json` y probé el analizador siguiente con datos sintéticos de interfaz. No ejecuté CUDA ni el organismo.

## 1. Descomposición que cambia la interpretación

El código confirma:

\[
d_k=(q^{usada}_{L,k}-b_L)-(q^{usada}_{R,k}-b_R),
\qquad
\omega_k=\operatorname{rad}(5)\tanh(250d_k),
\]

con \(q^{usada}_{k}=q^{actual}_{k-1}\): **un intervalo de 1 ms de desfase**. El baseline no se reinicia al ON. Los IDs lectores son **DNb05 10118 L y 10065 R**, no las DNa02 observadas por el tap.  

Recalculé desde los valores textuales de `CLOSE.json`:

| A 400 ms | Δyaw | Comando integrado | Δyaw − comando integrado |
|---|---:|---:|---:|
| Sham | +0,075607° | +0,078614° | −0,003008° |
| Izquierdo | +0,155823° | +0,159712° | −0,003889° |
| Derecho | +0,000218° | +0,002966° | −0,002748° |
| Uniforme | +0,093323° | +0,096895° | −0,003572° |

**Derecha−sham:** −0,075388° en yaw y −0,075648° en comando integrado; la diferencia entre ambos contrastes es **+0,000260°**. Esto favorece estudiar cómo se genera el comando basal antes de atribuir el fallo principalmente al cuerpo. **El residual es contabilidad, no una descomposición causal de fuerzas.** 

DNa02 tiene target cero **en las muestras capturadas**, no una demostración de silencio en todas las etapas internas. Asimismo, polaridad del target PN no equivale a polaridad de todas sus salidas receptor-específicas. 

## 2. Tres alternativas y experimentos fijables

### C — Primero: mando abierto sobre el cuerpo, sin nuevo controlador neural

Usaría **el estado corporal preparado completo**, incluidos actuadores, controladores, colas y estados internos; no solo `qpos/qvel`. Tres ramas independientes:

\[
\omega_\sigma=\sigma\,\operatorname{rad}(1),\quad
\sigma\in\{-1,0,+1\},\qquad T=200\,ms.
\]

El avance \(v_0\) se fija al valor del controlador en el estado preparado. Los demás comandos son idénticos entre ramas. No se reproduce una fuerza corporal grabada: las fuerzas se recalculan con el cuerpo de cada rama.

```text
para sigma en {-1, 0, +1}:
    restaurar exactamente el mismo estado corporal y de control
    durante 200 intervalos de 1 ms:
        sustituir únicamente la entrada yaw por sigma * rad(1)
        mantener avance v0 y los demás comandos comunes
        calcular fuerzas con el controlador físico existente y el estado actual
        integrar el cuerpo; NO actualizar yaw desde DNb05
        registrar comando realmente consumido, pose, velocidad y contactos
```

Mediría, sobre toda la trayectoria:

\[
O(t)=\frac{\Theta_+(t)-\Theta_-(t)}2,\qquad
E(t)=\Theta_+(t)+\Theta_-(t)-2\Theta_0(t).
\]

**Falsador:** comandos espejo comprobados no producen respuestas de signos opuestos o muestran asimetría persistente frente al refinamiento físico. Eso localiza una limitación de **esta planta y este estado**, no demuestra un defecto biológico.

Presupuesto: tres comandos × dos pasos físicos —1 y 0,5 ms, manteniendo el comando durante 1 ms—, **seis replays corporales y 120 s máximos de pared**. No son seis simulaciones del cerebro. Si no existe una restauración corporal exacta comprobada, se bloquea ese ensayo.

### A — Retirar solo las 629 sustituciones generales izquierdas

Para esas posiciones \(P_{629}\), la comparación es:

\[
W_{ij}\,\mathrm{cap}_j\,r^{local}_{ij}
\quad\longleftrightarrow\quad
W_{ij}\,\mathrm{cap}_j\,s^{legacy}_j.
\]

Mantener las **466 rutas dinámicas**, el productor PN fino, anatomía y estados iniciales. El código ya separa la bandera `general_outputs.enabled`; no hace falta copiar los 1.095 destinos al otro lado. La PN derecha tiene **1.103 destinos propios**.  

```text
desde dos estados preparados equivalentes:
    control: general_outputs.enabled = True
    intervención: general_outputs.enabled = False
    actualizar identidad del operador mediante su mecanismo de validación
    reconstruir el grafo capturado; conservar estados y reloj del integrador
    comprobar rutas/ecuaciones de los 466 destinos y W anatómica intactas
    ejecutar sham, uniforme, izquierda y derecha
```

**No basta cambiar la bandera después de capturar el grafo.** Tampoco exigiría que las señales futuras de los 466 destinos fueran idénticas: pueden cambiar legítimamente por realimentación. Se preservan sus operadores, no se congela su respuesta.

Falsador: la sustitución desaparece donde corresponde, pero el contraste DNb05/comando no cambia más que la incertidumbre numérica. Debilita esa sustitución como explicación suficiente.

Presupuesto: hasta cuatro brazos adicionales de 400 ms, **2.050 s por brazo**. Reutilizar los controles actuales exige verificar equivalencia de preparación y reconstrucción del grafo; si falla, no duplicar automáticamente el presupuesto.

### B — Convergencia central: observar al lector real

Extendería el tap a **DNb05 10118/10065**, comprobando primero todos sus escritores efectivos. Registrar dentro del consumidor:

`subtotal`, `drive`, `theta`, `gain`, argumento de `tanh`, target y rate.

Los buffers quedan residentes y dentro del CUDA Graph. Para latencias, añadir **origen de época y tiempo efectivo de etapa**, no solo `h.time_ns`. Para detectar activación transitoria, acumular mínimos/máximos de etapas **aceptadas**; descartar los del intento rechazado. Ninguna nueva llamada Python ni reevaluación del RHS en el bucle caliente.

El falsador de B es encontrar entradas y target DNb05 coherentes con el contraste requerido, mientras el problema aparece únicamente en el baseline/lector o cuerpo. Una futura ablación se elegiría por flujo efectivo predeclarado, no por el mayor `q`.

## 3. Error numérico y afirmaciones de Gemini

**DNb05 es una candidata biológica razonable, no un validador de esta fórmula.** Yang et al. registraron correlación bilateral de calcio con giro para DNb05; sus intervenciones detalladas se centraron en DNa02/DNg13. Eso no valida `tanh(250·Δq)`, la equivalencia de `q` a calcio/espigas ni los 5°/s elegidos aquí. :chatgpt-content-reference{index="6"}

La handedness observada por Buchanan et al. corresponde a preferencias persistentes en múltiples decisiones de giro. **Una deriva sham de una preparación, con este lector y baseline, no demuestra handedness biológica.** :chatgpt-content-reference{index="7"}

Hay además una sensibilidad matemática relevante:

\[
|\delta\omega|_{\mathrm{grados/s}}
\le1250\,|\delta d|.
\]

Si cada DNb05 tuviera error continuo acotado por \(10^{-4}\), con baseline idéntico, la cota del **comando integrado** en 400 ms sería **0,1°**. Es una cota condicional, no el error observado, y no acota por sí sola el cuerpo. Muestra por qué no basta citar un límite genérico: hay que medir error en DNb05, comando y yaw sobre la ventana interpretada.

Conservaría observador on/off y refinamiento temporal del mismo operador, incluyendo eventos e históricos que afectan emisiones futuras. El desacuerdo previo **0,065248 en `kc_axonal_state/trough` sigue sin resolverse**; no se descarta por observar targets focales precisos. Comparar dos resoluciones aporta evidencia de convergencia, no un certificado universal. 

## 4. Analizador completo: `analizar_dnb05.py`

Lee **solo NPZ/JSON** del paquete. Verifica IDs, preparación, exposición, relojes, desfase, baseline y fórmula. Separa yaw, comando integrado y residual; conserva sham/uniforme. El rebasado al ON es **un cálculo offline**, nunca una predicción del cuerpo bajo intervención.

```python
"""Descomposición offline de cuatro brazos del snapshot 50b9fd3e.
No simula cuerpo/CNS. Rebasar el lector abajo es SOLO cálculo contrafactual.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse, csv, hashlib, json, time, traceback, zipfile
from pathlib import Path
import numpy as np

ARMS=('sham','odor_left','odor_right','uniform')
DN_IDS=[10118,10065]
FLOW_IDS=[10176,10208,10360,523769]  # PN R,L; DNa02 R,L.
VENTANAS=(250,275,300,320,350,400)
def exigir(ok,mensaje):
    if not ok: raise ValueError(mensaje)
def guardar(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def maxabs(x): return float(np.max(np.abs(x),initial=0.))
def decodificar(q,base):
    d=q-base
    return np.tanh(250*(d[:,2]-d[:,3]))*np.deg2rad(5)
def yaw(qpos):
    w,x,y,z=qpos[:,3:7].T
    exigir(maxabs(w*w+x*x+y*y+z*z-1)<1e-8,'Quaternion no unitario')
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
def csvout(p,filas):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)

def analizar(root,out,manifest=None):
    out.mkdir(exist_ok=False);tic=time.perf_counter();hashes={};data={};checks={};common=None
    expected={} if manifest is None else {r['path']:r for r in json.loads(manifest.read_text())['files']}
    def entrada(rel):
        p=root/rel; exigir(p.is_file(),'Falta '+str(p));digest=hashfile(p);hashes[rel]=digest
        if manifest is not None:
            record=expected.get('campaign/'+rel)
            exigir(record is not None and record['sha256']==digest and record['bytes']==p.stat().st_size,'Hash/bytes distintos: '+rel)
        return p
    def js(rel): return json.loads(entrada(rel).read_text(encoding='utf-8'))
    def arrays(rel):
        p=entrada(rel)
        with zipfile.ZipFile(p) as z:
            exigir(sum(x.file_size for x in z.infolist())<=256*1024**2,'NPZ mayor de 256 MiB')
        with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    plan={'scope':'Cuatro brazos 40+400 ms; no adaptación ni simulación','windows_ms':VENTANAS,
          'decoder':'tanh(250*((qL-bL)-(qR-bR)))*deg2rad(5)','lag_expected_ms':1,
          'counterfactual':'Rebasado al ON SOLO offline; no predice otro cuerpo',
          'yaw_readback_atol_deg':1e-10,'limits':'No certifica error biológico ni numérico del organismo'}
    guardar(out/'PLAN_ANALISIS.json',plan)
    try:
        for arm in ARMS:
            folder='full_'+arm+'_01/'
            receipt=js(folder+'RESULT.json');contract=js(folder+'RUN_CONTRACT.json')
            inter=js(folder+'preparation_inputs/INTERVENCIONES.json');frozen=js(folder+'FROZEN.json')
            exigir(receipt['status']=='COMPLETE' and receipt['error'] is None and not receipt['cleanup_errors'],'Corrida incompleta')
            exigir(receipt['odor']==contract['odor']==arm,'Brazo incorrecto')
            exigir(receipt['completed_preparation_ms']==40 and receipt['completed_trial_ms']==400,'Horizonte incompleto')
            exigir(contract['preparation_ms']==40 and contract['trial_ms']==400 and contract['event_boundaries'] is True,'Contrato distinto')
            exigir(inter['dn_ids_lector'][2:]==DN_IDS and len(inter['dn_ids_lector'])==4,'IDs del lector')
            identity={k:contract[k] for k in ('engine','checkpoint_manifest_sha256','plan_sha256','event_representation')}
            firma=(identity,frozen,inter)
            if common is None: common=firma
            else: exigir(common==firma,'Operador/preparación declarados distintos entre brazos')
            t=arrays(folder+'traces.npz');flow=arrays(folder+'flow/FLOW.npz')
            phase=np.array(['preparacion']*40+['ensayo']*400)
            exigir(np.array_equal(t['fase'],phase) and np.array_equal(t['paso'],np.r_[np.arange(1,41),np.arange(1,401)]),'Fases/pasos')
            for k,v in t.items():
                exigir(v.ndim>=1 and len(v)==440,'Forma de traza: '+k)
                if v.dtype.kind in 'fiu': exigir(np.isfinite(v).all(),'No finito: '+k)
            clock=t['CNS_time_ns']
            exigir(clock.shape==(440,) and clock.dtype.kind in 'iu' and np.all(clock[1:]-clock[:-1]==1000000),'Reloj CNS')
            exigir(np.array_equal(clock,t['PN_time_ns']) and np.array_equal(clock,t['body_time_ns']),'Relojes distintos')
            exigir(t['sensores_usados'].shape==(440,3) and np.all(t['sensores_usados'][:40]==0),'Preparación no limpia')
            odor={'sham':[0,0,0],'odor_left':[1,0,0],'odor_right':[0,1,0],'uniform':[1,1,0]}[arm]
            exigir(np.array_equal(t['sensores_usados'][40:],np.tile(odor,(400,1))),'Exposición consumida')
            q,used,base=(t[k] for k in ('DN_q_actual','DN_q_usada','DN_baseline'))
            exigir(q.shape==used.shape==base.shape==(440,4),'Ejes DN')
            exigir(np.array_equal(used[1:],q[:-1]),'No conserva desfase declarado 1 ms')
            exigir(np.array_equal(base,np.tile(inter['baseline_inicial'],(440,1))),'Baseline cambió')
            command=decodificar(used,base)
            exigir(np.array_equal(command,t['command_yaw_rate_rad_s']),'Fórmula DNb05 no exacta')
            if arm!='sham':
                s=data['sham']['trace'];exigir(set(t)==set(s),'Campos distintos')
                exigir(all(np.array_equal(t[k][:40],s[k][:40]) for k in t),'Preparación registrada no idéntica')
                exigir(np.array_equal(clock,s['CNS_time_ns']),'Reloj absoluto distinto')
            exigir(np.array_equal(flow['ids'],FLOW_IDS) and np.array_equal(flow['time_ns'],clock)
                    and np.array_equal(flow['phase'],phase),'IDs/reloj de flujo')
            exigir(flow['target'].shape==flow['raw_signed'].shape==(440,4),'Forma de flujo')
            exigir(np.isfinite(flow['target']).all() and np.isfinite(flow['raw_signed']).all(),'Flujo no finito')
            psi=yaw(t['qpos']);theta=psi[40:]-psi[39]
            erryaw=maxabs(theta-t['yaw_delta_deg'][40:])
            exigir(erryaw<=1e-10,'Yaw guardado no coincide con qpos/origen')
            dt=(clock[40:]-clock[39:-1]).astype(float)*1e-9
            integral=np.rad2deg(np.cumsum(command[40:]*dt))
            nuevo_base=np.broadcast_to(q[39],base.shape)
            offline=np.rad2deg(np.cumsum(decodificar(used,nuevo_base)[40:]*dt))
            diff=used[:,2]-used[:,3]-(base[:,2]-base[:,3])
            checks[arm]={'lag1_exact':True,'decoder_exact':True,'yaw_readback_error_deg':erryaw,
                'lag0_command_error_rad_s':maxabs(decodificar(q,base)-command),
                'lag2_command_error_rad_s':maxabs(decodificar(q[:-2],base[2:])-command[2:]),
                'DNa02_nonzero_targets_sampled':int(np.count_nonzero(flow['target'][:,2:])),
                'DNb05_reader_ids_L_R':DN_IDS,'baseline_L_R':base[0,2:].tolist(),
                'q_used_ON_L_R':used[40,2:].tolist(),
                'qvel_max_abs_native_units':maxabs(t['qvel'])}
            data[arm]={'trace':t,'theta':theta,'command':integral,'residual':theta-integral,
                       'offline_rebased_command':offline,'d':diff[40:],'flow':flow['target'][40:]}
        rows=[]
        for ms in VENTANAS:
            i=ms-1
            for arm in ARMS:
                d=data[arm];r={'arm':arm,'ms':ms,'DNb05_delta_used':float(d['d'][i]),
                  'PN_target_L_R':float(d['flow'][i,1]-d['flow'][i,0]),
                  'OFFLINE_rebased_command_integral_deg':float(d['offline_rebased_command'][i])}
                for key in ('theta','command','residual'):
                    r[key+'_deg']=float(d[key][i])
                    for control in ('sham','uniform'):
                        r[key+'_minus_'+control+'_deg']=float(d[key][i]-data[control][key][i])
                rows.append(r)
        csvout(out/'DESCOMPOSICION.csv',rows)
        contrasts=[]
        for ms in VENTANAS:
            i=ms-1;r={'ms':ms}
            for key in ('theta','command','residual'):
                L,R,U,S=[data[a][key][i] for a in ('odor_left','odor_right','uniform','sham')]
                r[key+'_odd_deg']=float((L-R)/2);r[key+'_common_deg']=float((L+R)/2-S)
                r[key+'_uniform_minus_sham_deg']=float(U-S)
            contrasts.append(r)
        csvout(out/'CONTRASTES.csv',contrasts)
        np.savez_compressed(out/'CURVAS.npz',ms=np.arange(1,401),arms=np.array(ARMS),
            **{key:np.stack([data[a][key] for a in ARMS]) for key in ('theta','command','residual','offline_rebased_command')})
        result={'status':'OFFLINE_COMPLETE','checks':checks,'contrasts':contrasts,'input_sha256':hashes,
                'manifest_checked':manifest is not None,'code_sha256':hashfile(Path(__file__)),
                'wall_s':time.perf_counter()-tic,'stage3_admission':False,
                'limits':'Residual yaw-command es contabilidad, no torque causal. Rebasado offline no es intervención. Target DNa02 cero se refiere SOLO a muestras.'}
        guardar(out/'RESULTADO.json',result);return result
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'input_sha256':hashes});raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
    r=analizar(a.campaign.resolve(),a.out,a.manifest)
    print(json.dumps({'status':r['status'],'contraste_400':r['contrasts'][-1],'checks':r['checks']},indent=2,ensure_ascii=False))
```

```bash
python analizar_dnb05.py \
  --campaign /ruta/extraida/campaign \
  --manifest /ruta/al/MANIFEST.json \
  --out analisis_dnb05_01
```

La carpeta `--campaign` debe contener `full_sham_01`, `full_odor_left_01`, `full_odor_right_01` y `full_uniform_01`. `--manifest` es opcional; sin él se calculan hashes, pero no se contrastan con los publicados.

La suma del comando usa intervalos de 1 ms, no integración trapezoidal de órdenes mantenidas. Los controles de lag 0/2 se informan **sin elegir el que mejor se ajuste**. Los límites de lectura de quaternion/yaw no modifican las tolerancias del motor.

## 5. Ejecución y archivos revisados

**Ejecuté el analizador por CLI sobre un fixture de esquema**, no sobre el organismo: código de salida 0, **7,89 s de pared**. Seis corrupciones fueron rechazadas: desfase, comando, reloj, IDs DNb05, exposición y yaw incompatible con `qpos`. Conservé código, entradas sintéticas, resultados y fallos. También ejecuté la aritmética de `CLOSE.json` mostrada arriba.

**Leídos completos:** README raíz, `ARCHIVE.json`; `campaign/README.md`, `RESULTS.md`, `PLAN.json`, `CLOSE.json`, `analyze_long.py`, `native_tap.py`, `run_set.py`; `source/organism_adapter.py`, `pn_general_output_brain.py`; `prior/PN_PAIR_STRUCTURE.json`, `REFERENCE_20_COMPARE.json`; y los JSON de contrato/intervenciones del sham. **Parcialmente:** `matrix_olfactory_diagnostic.py` —líneas 1–230— y el manifiesto.

No ejecuté los NPZ publicados, ni el controlador físico, ni la prueba A. El código exacto de `bind_macro_advance` y la restauración corporal completa siguen siendo necesarios para implementar C contra vuestras clases sin inventar una API.

SHA256 de `analizar_dnb05.py`:  
`a5c39e25e17824e17cbc470c374e79a696bd5c111dbb9373634fde02c2a6a2a9`

:chatgpt-content-reference{index="10"}[Paquete reproducible — 230 KB](sandbox:/mnt/data/AXIOMA_DNB05_DESCOMPOSICION_REPRODUCIBLE.zip) · :chatgpt-content-reference{index="11"}[Python](sandbox:/mnt/data/AXIOMA_DNB05_20260923/analizar_dnb05.py)

**Decisión:** C es el diagnóstico inmediato más barato; A prueba una asimetría de implementación sin alterar anatomía; B debe observar DNb05 antes de atribuir una causa central. La ronda demuestra respuesta diferencial y ejecución completa, no orientación bilateral absoluta ni validación del motor.
