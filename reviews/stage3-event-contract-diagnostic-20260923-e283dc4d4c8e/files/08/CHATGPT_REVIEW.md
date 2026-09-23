**Conservaría SET/ADD como reparación candidata del evento real, pero encontré un defecto pequeño y reproducible de invariancia del prefijo en la consulta CPU.** Añadir un SET futuro cambia, en un ulp, una consulta anterior. **No refuta la semántica matemática de SET/ADD ni demuestra un error mayor que \(10^{-4}\)**; sí contradice que la misma historia pasada se evalúe siempre igual.

**Ejecuté pruebas CPU sobre la fuente publicada, verificada por SHA256. No pude descargar el ZIP —falló la resolución DNS— y no ejecuté CUDA, `run_set.py` ni el organismo.**

:chatgpt-content-reference{index="12"}[Fuentes verificadas, reproductores, resultados y registros de ejecución](sandbox:/mnt/data/AXIOMA_SET_C17D10_REVISION_CPU.zip)

## 1. Resultado empírico

| Prueba ejecutada | Resultado |
|---|---|
| 42 consultas con ADD/SET, marcas simultáneas y tres combinaciones de constantes temporales | Coincidencia con referencia independiente `scipy.linalg.expm` en las muestras ensayadas. |
| Continuidad de \(s\) en SET | Conservada en el fixture. |
| Publicación con un `post=inf` | Rechazada antes de añadir eventos parciales. |
| Productor LIF con cinco eventos, comparado con la fuente anterior | Estados, tiempos, saltos y conteos anteriores idénticos bit a bit. |
| Replay de los operandos publicados de KC41645 | ADD y `math.fsum` dan **1,0000000000000002**; SET al post declarado da **1**. |
| Consulta anterior a un SET futuro | **No idéntica bit a bit en CPU.** |
| Candidata CPU que considera únicamente SET ya ocurridos | Elimina esa discrepancia en los casos probados; conserva los controles anteriores. |

La suite principal tardó **5,30 s de pared y 5,28 s de CPU**, con aproximadamente **199,6 MiB** de memoria máxima. No reajusté el criterio de comparación de \(10^{-12}\).

### Defecto concreto: un evento futuro cambia la aritmética del pasado

`Waveform.at()` decide utilizar integración por segmentos cuando existe **cualquier** `post` finito, incluso posterior al instante consultado. El kernel CUDA, en cambio, activa `has_set` únicamente para marcas `et<=t`. Son condiciones diferentes.  

Caso ejecutado: ADD a 31,25 µs; consulta a 80 µs; después se añade SET a 100 µs.

| Consulta CPU a 80 µs | Antes de añadir SET futuro | Después |
|---|---:|---:|
| \(q\) | 0,7974676323973837 | 0,7974676323973838 |
| \(s\) | 0,20888706903475412 | 0,20888706903475407 |

**La diferencia es de redondeo por cambiar la forma de evaluación, no una influencia física del evento futuro.** La corrección localizada es restringir tanto las guardias como las filas reprocesadas a `post` finito **y marca ≤ tiempo consultado**. El reproductor genera esa candidata en una carpeta nueva; no modifica vuestra fuente.

### Segunda limitación: “ADD sin cambios” requiere precisar el orden

`FilterPorts.update()` ordena también las historias ADD-only por tiempo. La consulta CPU aditiva conserva el orden de inserción. Para una publicación fuera de orden, cambia la secuencia de sumas. Ejecuté el método de empaquetado publicado con buffers NumPy: una suma escalar CPU dio **0,5978377023018008 frente a 0,5978377023018009**. **No ejecuté ese contraste en CUDA ni afirmo que la fuente física real publique fuera de orden.** 

La condición mínima es explícita: o se exige publicación cronológica por fuente, validándola antes de mutar, o se conserva el orden aditivo anterior y se utiliza un índice ordenado separado para el recorrido SET. No anunciaría invariancia bit a bit para cualquier entrada admitida basándose en un solo ADD.

## 2. Qué no refuté

El orden simultáneo se conserva mediante `(tiempo, posición de inserción)`. SET→ADD produce \(0,7+0,1\); ADD→SET produce \(0,7\). La actualización de \(s\) usa el \(q\) anterior durante el segmento y no lo salta en el evento. Las pruebas GPU publicadas cubren esos casos, aunque **las ejecutó vuestro entorno, no el mío**.  

`lif_record` conserva el `event_value` calculado por el productor, y `event_coupling` lo entrega junto al salto. **Ese post pertenece a la historia física que generó el evento**: debe regenerarse si se descarta y recalcula dicha historia, no reutilizarse desde otra trayectoria para imponer su resultado.  

`run_set.py` comprueba la procedencia de los módulos antes de crear la sesión y prohíbe publicar un checkpoint del propietario fallido. Sin embargo, depende de componentes externos al paquete; **leerlo no demuestra su instalación o rollback integrales**. El recibo de 1 ms no es una garantía de 130/400 ms.  

## 3. A/B/C: qué control selecciona cada vía

| Vía | Control decisivo | Evaluación |
|---|---|---|
| **A. SET/ADD explícito** | Post del productor, orden de eventos, continuidad de \(s\), prefijos y reconstrucción del endpoint físico. | **La opción mínima mejor respaldada para este fallo**, con la corrección de prefijos señalada. |
| **B. ADD con precisión/compensación** | Evaluar la historia almacenada en alta precisión antes de cambiar el evaluador. | Los operandos reales ya dan \(1+1,77339566\times10^{-16}\). Mejorar solamente la suma no recupera el post perdido. B necesitaría preservar información adicional en el productor, no “redondear a uno” después. |
| **C. Segmentos desde el estado físico** | Misma historia de eventos, consultas por prefijo y retorno al endpoint del propietario; rechazo/reintento sin doble publicación. | Útil para evitar reconstrucciones duplicadas. **Es parcialmente complementaria a A**: el candidato ya emplea segmentos para filas con SET, no constituye una validación independiente de toda otra arquitectura. |

El control on/off publicado encontró el mismo fallo y prefijo exportado idéntico; eso debilita la explicación instrumental para **este** evento. No identifica todos los estados internos del intento fallido. 

**Control CUDA aún pertinente:** capturar una vez y reutilizar el mismo proyector tras actualizarlo sucesivamente con SET, ADD-only y ningún evento; consultar antes/en/después de la marca y comprobar contra CPU. Los fixtures publicados crean proyectores nuevos por caso: no ejercitan específicamente esa actualización bajo un mismo grafo capturado. No hace falta otra campaña del organismo para esa comprobación. 

---

## 4. Código exacto ejecutado: `revisar_set.py`

Requiere NumPy, SciPy y Numba. `--source` debe apuntar al `candidate/event_waveform.py` de este commit. `--parent` es opcional; la copia anterior verificada está incluida en mi paquete.

```bash
python revisar_set.py \
  --source /ruta/extraida/candidate/event_waveform.py \
  --out revision_set_cpu_01
```

```python
"""Pruebas CPU sobre c17d10f. No modifica fuente, tolerancias ni organismo.
El prefijo usa un caso sintético; los operandos de REAL provienen del evento 41645.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, hashlib, importlib.util, json, math, sys, time, traceback
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy.linalg import expm
SHA='2549fb0ca90afa90020d1dcd533e59a32fc59f0eee12605919832730e33f3feb'
SHA_PARENT='49986d70a5a72fe33267c7261b904d5ef68d0bde4e6344ff050e38dfeec75fae'
REAL=(.8061807853958571,.8830133842861765,.02426011860370636,.005,4.486788730489399e-6,.19396829995917275,.000125)
TOL=1e-12

def exigir(ok,msg):
    if not ok: raise AssertionError(msg)
def cargar(p,nombre,esperado):
    b=p.read_bytes(); exigir(hashlib.sha256(b).hexdigest()==esperado,'SHA de fuente distinto')
    spec=importlib.util.spec_from_file_location(nombre,p)
    m=importlib.util.module_from_spec(spec);sys.modules[nombre]=m;spec.loader.exec_module(m)
    return m

def oracle(w,t,incluir=True):
    """Referencia independiente: exponencial 2x2 y saltos ordenados por fila."""
    out=np.column_stack((w.q,w.s)).copy()
    for r in range(len(w.q)):
        A=np.array([[-1/w.tau[r],0.],[1/w.ts,-1/w.ts]])
        prev=0.; x=out[r].copy()
        orden=sorted((k for k,rr in enumerate(w.rows) if rr==r),key=lambda k:(w.times[k],k))
        for k in orden:
            e=w.times[k]
            if e>t or (not incluir and e==t): break
            x=expm(A*(e-prev))@x;prev=e
            if np.isfinite(w.posts[k]): x[0]=w.posts[k]
            else: x[0]+=w.jumps[k]
        out[r]=expm(A*(t-prev))@x
    return out.T

def prefix(W):
    w=W([.7],[.2],[.024],.005,.000125)
    w.add([.00003125],np.array([0]),[.1]);t=.00008
    a=np.array(w.at(t));w.add([.0001],np.array([0]),[0.],posts=[.7]);b=np.array(w.at(t))
    return {'tiempo_s':t,'SET_futuro_s':.0001,'antes':a.ravel().tolist(),
            'despues':b.ravel().tolist(),'delta':(b-a).ravel().tolist(),
            'bits_iguales':a.tobytes()==b.tobytes(),
            'hex_antes':[float(v).hex() for v in a.ravel()],
            'hex_despues':[float(v).hex() for v in b.ravel()]}

def positivos(m):
    W=m.Waveform;peor=0.;continuidad=0.;consultas=0
    for tq,ts in ((.024,.005),(.005,.005),(.005,.024)):
        for orden in (0,1):
            w=W([.4],[.2],[tq],ts,.000125)
            te=40e-6
            saltos=[0.,.1] if orden==0 else [.1,0.]
            posts=[.7,np.nan] if orden==0 else [np.nan,.7]
            w.add([te,te],np.array([0,0]),saltos,posts=posts)
            w.add([90e-6,70e-6],np.array([0,0]),[0.,.05],posts=[.6,np.nan])
            exigir(w.at(te)[0][0]==(.7+.1 if orden==0 else .7),'Orden simultáneo')
            for t in (0.,np.nextafter(te,0),te,np.nextafter(te,1),70e-6,90e-6,.000125):
                err=float(np.max(abs(np.array(w.at(t))-oracle(w,t))))
                peor=max(peor,err);consultas+=1
            continuidad=max(continuidad,float(abs(w.at(te)[1][0]-oracle(w,te,False)[1,0])))
    exigir(peor<TOL and continuidad<TOL,'Referencia/continuidad')
    w=W([.2],[.3],[.01],.005,.000125); antes=(list(w.times),list(w.posts))
    try:w.add([30e-6,30e-6],np.array([0,0]),[0.,0.],posts=[.7,np.inf])
    except ValueError:pass
    else:raise AssertionError('Publicación inválida aceptada')
    exigir((w.times,w.posts)==antes,'Publicación parcial')
    return {'consultas_expm':consultas,'max_error_expm':peor,
            'max_error_s_continua':continuidad,'rechazo_sin_eventos_parciales':True}

def caso_real(m):
    q,s,tq,ts,te,j,dur=REAL
    w=m.Waveform([q],[s],[tq],ts,dur);w.add([te],np.array([0]),[j])
    add=float(w.at(te)[0][0]);compensado=math.fsum([q*math.exp(-te/tq),j])
    w=m.Waveform([q],[s],[tq],ts,dur);w.add([te],np.array([0]),[j],posts=[1.])
    setq=float(w.at(te)[0][0]);expm_error=float(np.max(abs(np.array(w.at(dur))-oracle(w,dur))))
    with localcontext() as c:
        c.prec=100;D=Decimal.from_float
        alto=D(q)*(-D(te)/D(tq)).exp()+D(j)
        exceso=str(alto-Decimal(1))
    exigir(setq==1. and expm_error<TOL,'Replay real SET')
    return {'ADD_fp64':add,'ADD_fsum':compensado,'ADD_exceso_100digitos':exceso,
            'SET_post_declarado':setq,'error_expm_al_final':expm_error,
            'alcance':'Replay de operandos publicados; no simulación de la célula real.'}

def padre(m,p):
    dt=.01;tq=.024
    base=[np.array([-1.]),np.array([0.]),np.array([0],dtype=np.int64),np.array([.2]),
          np.array([1.]),np.array([math.log(2)/.00003125]),np.array([-1.]),np.array([0.]),
          np.array([120.]),np.array([tq])]
    a=[x.copy() for x in base];b=[x.copy() for x in base]
    old=p.lif_record(*a,dt);new=m.lif_record(*b,dt)
    exacto=all(x.tobytes()==y.tobytes() for x,y in zip(a,b)) and old[0]==new[0]
    exacto=exacto and all(x.tobytes()==y.tobytes() for x,y in zip(old[1:],new[1:3]))
    exigir(exacto,'Cambió el productor')
    mask=np.isfinite(new[1][0]);w=m.Waveform([.2],[.3],[tq],.005,dt)
    w.add(new[1][0,mask],np.zeros(mask.sum(),dtype=np.int64),new[2][0,mask],posts=new[3][0,mask])
    qend=float(w.at(dt)[0][0]);exigir(qend==b[3][0],'No recupera estado final de la fuente')
    return {'eventos':int(mask.sum()),'productor_y_eventos_anteriores_bitabit':True,
            'q_final_reconstruida':qend,'q_final_fuente':float(b[3][0])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--parent',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(exist_ok=False);inicio=time.perf_counter();cpu=time.process_time()
    r={'CUDA_ejecutado':False,'organismo_ejecutado':False,'criterio_referencia':TOL,'fuente_sha256':SHA}
    try:
        m=cargar(a.source.resolve(),'wf_c17d10_review',SHA)
        r['positivos']=positivos(m);r['evento_real']=caso_real(m);r['prefix_original']=prefix(m.Waveform)
        if a.parent:r['comparacion_productor']=padre(m,cargar(a.parent.resolve(),'wf_parent_review',SHA_PARENT))
        # Candidata local: tres guardias dependen sólo de SET ya ocurridos.
        texto=a.source.read_text()
        guardia='any(np.isfinite(self.posts))'
        exigir(texto.count(guardia)==2,'Anclas distintas')
        texto=texto.replace(guardia,'any(np.isfinite(p) and te<=time for p,te in zip(self.posts,self.times))')
        antes='if np.isfinite(post)}';exigir(texto.count(antes)==1,'Ancla filas distinta')
        texto=texto.replace(antes,'if np.isfinite(post) and self.times[i]<=time}')
        f=a.out/'event_waveform_prefix_candidate.py';f.write_text(texto)
        h=hashlib.sha256(f.read_bytes()).hexdigest();c=cargar(f.resolve(),'wf_prefix_candidate_review',h)
        r['prefix_candidata']=prefix(c.Waveform);r['positivos_candidata']=positivos(c)
        exigir(r['prefix_candidata']['bits_iguales'],'La candidata no restaura el prefijo probado')
        r['parche_solo_CPU_sha256']=h
        r['estado']='COMPLETO_REVISION_CPU'
    except Exception:r.update(estado='FALLO_CONSERVADO',error=traceback.format_exc())
    r.update(pared_s=time.perf_counter()-inicio,cpu_s=time.process_time()-cpu,
             codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'RESULTADO.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(r,indent=2,ensure_ascii=False));return 0 if r['estado']=='COMPLETO_REVISION_CPU' else 1
if __name__=='__main__':sys.exit(main())
```

La salida distingue el código publicado y la candidata CPU. **No instala la corrección ni altera el original.** La candidata resuelve el prefijo probado; no corrige por sí sola la diferencia de orden ADD en entradas publicadas fuera de orden.

## Alcance exacto

**Leídos completos:** README raíz, `ARCHIVE.json`, `MANIFEST.json`; los cuatro archivos solicitados; `test_set_cpu.py`, `test_set_gpu.py`, `candidate/PLAN.json`, `GPU_FIXTURE.json`, `SMOKE_PARENT_COMPARE.json`; `real_event/README.md`, `DOMAIN_CAPTURE.json`, `ARITHMETIC_AUDIT.json` y `control_off/COMPARE_PREFIX.json`.

**Materializados y comprobados:** `event_waveform.py` y `event_ports.py` coinciden con sus SHA256 del manifiesto. El padre proviene del adjunto anterior y coincide con su huella publicada.

**Ejecutado:** mi suite CPU sobre esas fuentes y el método host de empaquetado con buffers NumPy. **No ejecuté los fixtures originales completos, CUDA, `EventCoupling.install`, `run_set.py` ni sham130.** No verifico los 27 miembros del ZIP que no pude descargar.

**Dictamen:** el post explícito sigue siendo una reparación fundada para KC41645. Corregiría la dependencia aritmética del SET futuro y precisaría el contrato de orden ADD; luego comprobaría el replay del mismo grafo. Eso no modifica biología, no demuestra el resultado de sham130 y no supera Etapa 3.
