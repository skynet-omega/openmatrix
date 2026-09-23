**La continuación fría está razonablemente planteada para este checkpoint y perfil; no encontré una pérdida demostrada de estado en su ruta exitosa. Sí corregiría una invalidación en la ruta de fallo.** Para Etapa 4 conservaría la gaussiana existente y añadiría un diseño de transferencia que distinga realimentación espacial de reproducción abierta. Ejecuté únicamente controles CPU de geometría y del manejador de excepciones; **cero organismos, CUDA o MuJoCo**.

## 1. Continuación fría: revisión quirúrgica

El orden de restauración conserva lo esencial del contrato publicado: reconstruye el carrier, repone los campos serializados, estado neuronal, propiocepción, ojos, músculos y controlador; restaura el operador efectivo antes de crear el grafo; vuelve a publicar tasas, reloj y RNG al terminar los constructores. El campo recupera su origen, eje y reloj anteriores, sin recentrarlo en la postura reanudada. El comparador comprueba los cuatro árboles y la frontera antes del primer paso. No encontré una razón para reiniciar preparación, baseline o `pending_sensors`.   

Con la condición que declaras —**plasticidad `enabled=False`**— no hay fundamento para objetar esta continuación por aprendizaje de pesos durante la vida. Eso no equivale a haber ejecutado aquí los constructores ni sus validadores internos.

### Defecto localizado: propietario cerrado, pero no invalidado

En `restore_checkpoint.py`, línea **61**, se asigna `core.failed=False`. Si después falla, por ejemplo, la validación de la geometría del campo, las líneas **93–97** cierran cuerpo/híbrido, pero dejan `obj.core` apuntando a ese propietario sin marcarlo fallido. El `finally` de `run_continuation.py` puede intentar `snapshot()`, cuyo guard utiliza precisamente `core.failed`.  

**Corrección mínima**, únicamente en la rama de excepción:

```python
    except BaseException:
        core.failed = True
        obj.core = core
        if getattr(core, 'body', None) is not None:
            core.body.close()
        if getattr(core, 'hybrid', None) is not None and hasattr(core.hybrid, 'close'):
            core.hybrid.close()
        raise
```

Ejecuté el manejador extraído por AST de la fuente auténtica, con propietarios mínimos CPU:

| Resultado del fallo inducido | Publicado | Con la línea añadida |
|---|---:|---:|
| Cuerpo e híbrido cerrados | Sí | Sí |
| `core.failed` | **False** | **True** |
| Guard de snapshot bloquea por invalidez | **No** | **Sí** |

**No demostré que se escriba un checkpoint inválido con las clases reales**: sus validadores podrían rechazarlo después. Sí demostré que el manejador no lo impide por sí mismo. La modificación no afecta la trayectoria exitosa ni cambia criterios numéricos.

El resto debe decidirse mediante la prueba ya prevista: igualdad inicial y continuación 100→400 ms. No añado otra puerta universal ni altero sus límites. `check_remaining.py` conserva el origen angular preparado, verifica relojes, exposición geométrica, lag del lector y fórmula del comando antes de comparar los observables. 

## 2. Etapa 4: no hace falta otro campo

`AntennalBoundary.sample()` ya calcula

\[
c(p)=\exp\!\left[-\frac{\|p-S\|^2}{2\sigma^2}\right]
\]

sobre las posiciones actuales de las antenas. `AntennalWorld` mantiene el intervalo comprometido y produce la siguiente entrada bilateral; el factor \(80c\) pertenece a la frontera neuronal. Son centros de geometrías de colisión, **no posiciones de sensilios medidas**, y la concentración sigue sin calibración fisiológica.  

Para este piloto no usaría `StaticLateralField`: ese wrapper convierte el estímulo en semiplano 0/1. Conservaría la frontera gaussiana base, fuente fija, sin eventos futuros que la muevan, sin contacto-recompensa y con el muestreo/pending de 1 ms existente.  

### Transferencia propuesta: mismo olor inicial, otra dependencia espacial

Sea \(m\) el punto medio antenal, \(b\) la separación entre antenas, \(\ell\) el eje de derecha a izquierda y \(f\) su perpendicular orientada hacia delante. Propongo:

\[
S_+=m+\tfrac12bf+2b\ell,\qquad
S_-=m-\tfrac12bf+2b\ell,\qquad \sigma=2b.
\]

Las antenas están en \(m\pm b\ell/2\). Reflejar la fuente en \(f\) conserva **las dos distancias iniciales**: ambas geometrías empiezan con el mismo par de concentraciones, pero estas evolucionan de forma distinta al avanzar.

Con `GEOMETRY.json`, calculé:

| Magnitud | Donante \(S_+\) | Transferencia \(S_-\) |
|---|---:|---:|
| Fuente XY, mm | (0,918428; −0,094433) | (0,703369; −0,009587) |
| Distancia inicial desde el centro corporal | 1,197800 mm | 0,988345 mm |
| Concentraciones iniciales L/R | 0,731616 / 0,443747 | Iguales hasta \(1,11\times10^{-16}\) |
| Entrada declarada \(80c\), L/R | 58,5293 / 35,4998 | Igual |
| Distancia a la recta de avance inicial | 0,460122 mm | 0,460122 mm |

Aquí \(b=0,231191\) mm y \(\sigma=0,462382\) mm. Una región de llegada operacional de radio \(b/2=0,115596\) mm queda fuera de esa recta. **“Arranque fuera” significa fuera de esa región, no concentración exactamente cero:** la gaussiana tiene cola no nula. Los valores proceden de geometría publicada, no de una simulación nueva. 

### Cinco ramas futuras, una sola implementación ambiental

Después de la misma preparación limpia, sin reiniciar cerebro/lector:

1. Gaussiana espacial \(S_+\): donante.
2. Sham.
3. Uniforme constante \(c_0=0,587681\): media bilateral inicial, **no igualación retrospectiva de dosis**.
4. Gaussiana espacial \(S_-\): transferencia con feedback.
5. En la geometría \(S_-\), reproducción de la **secuencia sensorial consumida por el donante real \(S_+\)**, independiente de la pose actual.

La quinta rama no se compara con el donante para “descubrir” identidad determinista. Se compara con la cuarta **ante una fuente distinta**, que empieza con el mismo olor bilateral. La fuente permanece fija durante cada ensayo. No hay impulso mecánico, modificación neuronal ni objetivo comunicado al cerebro.

Sham y uniforme pueden reutilizarse para medir distancias a ambas fuentes **solo si se verifica que la posición de la fuente no tiene otra salida al organismo distinta del olor**. No se reutiliza una supuesta conducta bajo feedback.

| Rival | Falsador |
|---|---|
| **A. Feedback espacial útil** | La transferencia espacial no supera al replay donante y a los controles en acercamiento material. |
| **B. Giro abierto + avance tónico suficiente** | La transferencia espacial supera al replay con igual preparación y olor inicial; la secuencia abierta deja de explicar esa ventaja. |
| **C. Rango/latencia del gradiente continuo limita la respuesta** | Registrar concentración realmente consumida, ORN, liberación PN efectiva y mando. Si el gradiente se representa y transmite de forma coherente, C se debilita; si se pierde antes del mando, no culpar al cuerpo ni retocar ganancias. |

Un resultado favorable distinguiría el feedback de **ese replay abierto**, no de cualquier controlador imaginable ni demostraría navegación general entre especies.

## 3. Horizonte y efecto: 400 ms no bastan con la escala observada

Los endpoints laterales anteriores permiten una cota puramente geométrica:

\[
\bigl|\|x_L-S\|-\|x_{\rm sham}-S\|\bigr|
\le\|x_L-x_{\rm sham}\|.
\]

Calculada desde `APPROACH_CONTEXT.json`, vale **0,000435 mm** para izquierda y **0,000523 mm** para derecha. No predice los ensayos gaussianos, pero muestra la pequeñez de la separación corporal observada. 

Propongo para el **nuevo piloto**, no para la confirmación de etapa 3:

- Horizonte primario fijo **4 s**.
- Acercamiento \(P=d_{\rm cuerpo}(0)-d_{\rm cuerpo}(T)\).
- Ventaja material sobre controles \(\delta_d=b/20=\mathbf{0,011560\;mm}\).
- Reserva numérica propuesta por trayectoria \(\epsilon_d=\delta_d/10\); contraste exigido \(\delta_d+2\epsilon_d=\mathbf{0,013871\;mm}\).

La reserva procede de la desigualdad de distancias; la concordancia entre discretizaciones no se convierte por ello en una cota de solución exacta. Estas cifras necesitan un **prerregistro nuevo**; no alteran los 0,002° de etapa 3.

Cuatro segundos son un piloto de **acercamiento adicional**, no de llegada. Si la rapidez planar no supera 0,2 mm/s, solo recorrer la distancia inicial menos el radio exige al menos **5,41 s** y **4,36 s** respectivamente, incluso sin coste de giro. Es una cota cinemática condicional, no una garantía del cuerpo con contactos.

A la escala temporal comunicada, una extrapolación lineal da:

| Horizonte | Native por brazo | Reference por brazo |
|---|---:|---:|
| 4 s | ≈230 min | ≈320 min |
| 8 s | ≈460 min | ≈640 min |

Cinco ramas nativas de 4 s serían aproximadamente **19,2 horas**, antes de referencias adicionales. No son tiempos medidos del nuevo campo. Si ese coste no es aceptable, el objetivo honesto inmediato es comprobar **rango y flujo sensorial**, no llamar navegación a 400 ms. Un negativo a 4 s se conserva; no se amplía el horizonte dentro del mismo ensayo hasta conseguir éxito.

## 4. Código completo: `diseno_fuente_cpu.py`

Aporta cálculo geométrico y controles de transferencia, **no un nuevo simulador ni otro campo que sustituya al Gaussian**. Lee exclusivamente los dos JSON reales; verifica sus hashes. Las trayectorias rectas del informe son hipótesis geométricas y **no deben inyectarse como replay al organismo**.

```python
"""Diseño geométrico usando dos JSON reales. No ejecuta cuerpo, CNS o campos vivos.
La gaussiana coincide con AntennalBoundary.sample; sólo se evalúa su fórmula.
El replay propuesto usa un DONANTE REAL futuro, nunca las curvas proxy de este script.
"""
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
HASHES = {
    'geometry':'3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3',
    'context':'486d939c2f3bc55489763d9f1c5062cd31ac8c2a52a1524e547d235fcb8ccb39'
}
PLAN = {
    'scope':'Diseño propuesto; no prerregistro de ejecución del organismo',
    'source_longitudinal_in_baselines':0.5,
    'source_lateral_in_baselines':2.0,
    'sigma_in_baselines':2.0,
    'arrival_radius_in_baselines':0.5,
    'material_distance_in_baselines':0.05,
    'proposed_distance_error_fraction':0.1,
    'pilot_horizon_s':4.0,
    'readouts_s':[0.4,4.0,8.0],
    'cost_basis_min_per_0_4s':{'native':23.0,'reference':32.0},
    'organism_runs_executed':0
}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p,expected):
    need(sha(p)==expected,'SHA distinto de la entrada publicada: '+str(p))
    return json.loads(p.read_text(encoding='utf-8'))
def concentrations(points,source,sigma):
    # Misma fórmula y unidades mm del AntennalBoundary publicado.
    return np.exp(-np.sum((points[...,:2]-source)**2,axis=-1)/(2.*sigma**2))
def run(geometry,context):
    g=read(geometry,HASHES['geometry']);ctx=read(context,HASHES['context'])
    a=np.asarray(g['prepared_antennae_mm'],dtype=float)
    q=np.asarray(g['prepared_qpos_root'],dtype=float)
    need(a.shape==(2,3) and q.shape==(7,) and np.isfinite(a).all() and np.isfinite(q).all(),'Geometría inválida')
    scale=float(g['body_native_units_per_mm']);v=float(g['forward_command_mm_s'])
    need(scale>0 and v>0,'Escala/velocidad inválida')
    body=q[:2]/scale;mid=a[:,:2].mean(axis=0)
    baseline=float(np.linalg.norm(a[0,:2]-a[1,:2]));need(baseline>0,'Antenas coincidentes')
    lateral=(a[0,:2]-a[1,:2])/baseline
    forward=np.array([lateral[1],-lateral[0]])
    w,x,y,z=q[3:];need(abs(w*w+x*x+y*y+z*z-1)<1e-8,'Quaternion inválido')
    heading=np.array([1-2*(y*y+z*z),2*(w*z+x*y)])
    need(np.linalg.norm(heading)>0,'Rumbo horizontal degenerado')
    if forward@heading<0:forward=-forward
    sigma=PLAN['sigma_in_baselines']*baseline
    radius=PLAN['arrival_radius_in_baselines']*baseline
    effect=PLAN['material_distance_in_baselines']*baseline
    error=effect*PLAN['proposed_distance_error_fraction']
    offset=PLAN['source_longitudinal_in_baselines']*baseline
    side=PLAN['source_lateral_in_baselines']*baseline
    sources={'donor':mid+offset*forward+side*lateral,
             'transfer':mid-offset*forward+side*lateral}
    initial={k:concentrations(a,s,sigma) for k,s in sources.items()}
    initial_diff=float(np.max(abs(initial['donor']-initial['transfer'])))
    need(initial_diff<1e-12,'La transferencia no conserva ambos estímulos iniciales')
    specs={};rows=[]
    for name,source in sources.items():
        delta=source-body;along=float(delta@forward)
        lateral_offset=abs(float(delta@lateral))
        d0=float(np.linalg.norm(delta))
        raydistance=lateral_offset if along>=0 else d0
        need(d0>radius and raydistance>radius,'Fuente en la región inicial o recta tónica')
        c0=initial[name]
        specs[name]={'source_mm':source.tolist(),'sigma_mm':sigma,
            'initial_concentration_L_R':c0.tolist(),'initial_drive_80c_L_R':(80*c0).tolist(),
            'body_initial_distance_mm':d0,'distance_to_forward_ray_mm':raydistance,
            'arrival_radius_mm':radius,
            'conditional_arrival_lower_bound_s':(d0-radius)/v}
        for t in PLAN['readouts_s']:
            # Traslación geométrica hipotética: NO trayectoria predicha del cuerpo.
            da=v*t*forward;pts=a.copy();pts[:,:2]+=da
            straight_distance=float(np.linalg.norm(source-(body+da)))
            rows.append({'source':name,'seconds':t,'hypothetical_straight_distance_mm':straight_distance,
                'hypothetical_straight_progress_mm':d0-straight_distance,
                'hypothetical_concentration_L_R':concentrations(pts,source,sigma).tolist()})
    trials=ctx['derived_from_existing_native_trials']
    need(set(trials)=={'sham','odor_left','odor_right','uniform'},'Faltan brazos descriptivos')
    displacement={k:np.asarray(r['translation_mm'],dtype=float)[:2] for k,r in trials.items()}
    old_bounds={k:float(np.linalg.norm(d-displacement['sham'])) for k,d in displacement.items()}
    costs={str(t):{k:minutes*t/.4 for k,minutes in PLAN['cost_basis_min_per_0_4s'].items()} for t in PLAN['readouts_s']}
    T=PLAN['pilot_horizon_s'];donor=sources['donor'];transfer=sources['transfer']
    # Demuestra que la transferencia discrimina el campo sin cambiar el primer olor.
    points=a.copy();points[:,:2]+=v*.4*forward
    cd=concentrations(points,donor,sigma);ct=concentrations(points,transfer,sigma)
    need(np.max(abs(cd-ct))>0,'Transferencia degenerada')
    return {
      'status':'CPU_GEOMETRY_COMPLETE','plan':PLAN,'inputs_sha256':HASHES,
      'baseline_mm':baseline,'forward_axis':forward.tolist(),'left_axis':lateral.tolist(),
      'antennal_midpoint_mm':mid.tolist(),'prepared_body_mm':body.tolist(),
      'sources':specs,'equal_initial_exposure_max_abs':initial_diff,
      'uniform_concentration':float(initial['donor'].mean()),'sham_concentration':0.,
      'material_advantage_mm':effect,'proposed_pair_distance_error_mm':error,
      'proposed_advantage_with_two_error_reserves_mm':effect+2*error,
      'old_400ms_distance_contrast_upper_bounds_mm':old_bounds,
      'old_400ms_bound_scope':'Desigualdad triangular sobre endpoints de ensayos laterales antiguos; no predicción gaussiana.',
      'geometric_straight_proxies':rows,
      'transfer_exposure_gap_after_hypothetical_0_4s':(cd-ct).tolist(),
      'budget_projection_min_per_arm':costs,
      'five_arm_native_projection_hours':5*costs[str(T)]['native']/60,
      'primary_metric':'P=d_body(0)-d_body(T). Donor: ventaja sobre sham/uniform. Transfer: ventaja sobre sham/uniform/replay-donor.',
      'reuse_controls':'Sólo si fuente no tiene salida distinta del olor; sham/uniform tienen misma señal en ambas geometrías.',
      'future_replay':'Secuencia sensorial usada por el donante real, aplicada en geometría transfer; nunca estas curvas proxy.',
      'stage3_admission':False,'stage4_admission':False,
      'limits':'Sin simulación; cota de llegada condicional a rapidez planar<=0.2mm/s, no una cota del cuerpo. Coste extrapolado, no medido. Sin viento, recompensa, reescalado neuronal ni calibración fisiológica.'
    }
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'PLAN.json',PLAN)
    wall=time.perf_counter();cpu=time.process_time()
    try:result=run(a.geometry,a.context)
    except Exception:result={'status':'FAILED_RETAINED','error':traceback.format_exc()}
    result.update(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,code_sha256=sha(Path(__file__)))
    save(a.out/'RESULTADO.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['status']=='CPU_GEOMETRY_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
```

```bash
python diseno_fuente_cpu.py \
  --geometry /ruta/extraida/design/GEOMETRY.json \
  --context /ruta/extraida/design/APPROACH_CONTEXT.json \
  --out diseno_fuente_01
```

No modifica los JSON ni archivos del laboratorio. La carpeta de salida debe ser nueva.

## 5. Ejecución y alcance exactos

**Ejecutado:** el programa geométrico sobre copias textuales de los dos JSON, cuyos SHA256 coinciden con el manifiesto. El proceso completo tomó **0,73 s de pared, 1,04 s de CPU y aproximadamente 91 MiB**. También ejecuté el manejador de excepción aislado descrito arriba. No hubo ejecución neuronal ni corporal.

**Materializado y verificado:** `restore_checkpoint.py`, `GEOMETRY.json` y `APPROACH_CONTEXT.json`. El SHA256 de `diseno_fuente_cpu.py` es:

`c0637508b61bb663b2a96ed79e0947246096007e29f3c6c4f97825038f13973c`

**Leídos completos:** los cinco archivos de `continuation/`; los tres JSON de `design/`; `antennal_world.py`, `antennal_boundary.py`, `antennal_runtime.py` y `static_field.py`; README, ARCHIVE y MANIFEST. De `matrix_olfactory_diagnostic.py` leí el fragmento inicial, incluidas las funciones de preparación, instalación de campo, captura y métricas angulares; **no sus 114 KB completos**.

La descarga del ZIP original falló por DNS. No ejecuté la restauración integral, sus dependencias estáticas ni los snapshots de 100 ms. El ensayo frío real conserva la responsabilidad de confirmar esa continuación.

:chatgpt-content-reference{index="12"}[Paquete con originales verificados, corrección aislada, código y resultados CPU](sandbox:/mnt/data/AXIOMA_COLD_STAGE4_CD6710_REVISION.zip)

**Decisión:** corregir la invalidación de fallo sin tocar la trayectoria exitosa; terminar los controles reales de etapa 3; conservar la gaussiana y preparar una transferencia espacial con métricas de distancia. No hay resultado de Etapa 4 ni aprobación de etapa 3 en esta entrega.
