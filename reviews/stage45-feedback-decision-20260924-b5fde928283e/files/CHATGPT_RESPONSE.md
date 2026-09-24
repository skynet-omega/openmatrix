## Decisión: tres rutas, no una orden de congelar el motor

| Ruta | Experimento y controles | Coste condicionado a las campañas actuales | Falsador |
|---|---|---:|---|
| **A. Feedback tras perturbación corporal pequeña** | Donante sin perturbación; online y yoked con el mismo pulso. Propongo **2 s**, perturbación cercana a **1° a 0,5 s**, no 30–45° a 1,5 s. Dos referencias adicionales para online/yoked. | **≈12,26 h agregadas**: tres vidas nativas y dos referencias. | Online no mejora distancia/rumbo sobre yoked más allá de la incertidumbre numérica y mecánica. |
| **B. Transferencia sensorial sin impulso** | Cambiar prospectivamente la posición de la fuente. Comparar online con una cinta calculada para esa misma fuente sobre las poses del donante anterior. Dos nativas y dos referencias de 400 ms. | **≈2,02 h**, reutilizando el donante publicado solo si preparado y operador coinciden. | La actualización espacial no produce diferencia resuelta. Un positivo identifica dependencia del feedback, no navegación sostenida. |
| **C. Continuar migración nativa acotada** | Una frontera real y sus propietarios, padre/candidato ≤1 ms, manteniendo ecuaciones y calendario; fallo inyectado con rollback. | **Techo agregado 120 s** para el par, sin ampliación. | Estado/causalidad divergente o ausencia de ahorro integral. |

**Mi elección científica es A corregida; hoy haría primero su preflight corporal barato y mantendría el desarrollo del motor.** No exigiría alcanzar 600 s/5 s para autorizar un piloto exploratorio financiado, pero tampoco comprometería doce horas bajo la estimación de “2–2,5 horas”.

Los costes anteriores son extrapolaciones lineales, no presupuestos garantizados: reconstruí el promedio nativo de **1.564,146 s/400 ms** y el de referencia de **2.066,722 s/400 ms** a partir de los cierres. Una sola vida de 2 s sería aproximadamente **2,17 h nativas o 2,87 h de referencia**; a 3 s, **3,26 h o 4,31 h**.  

## 1. Qué es correcto y qué exagera Gemini

**0,057320516° es la diferencia final entre brazos**, no el giro de cada brazo. Las referencias terminan en **+0,044914336° y −0,012406180°**: hay signos opuestos al endpoint, pero no se demostró que la actualización espacial mejore la navegación. Además, S+ cambia su mando de positivo a negativo entre **351 y 352 ms**. No procede prolongar linealmente su giro anterior.   

El cierre conserva explícitamente abiertas navegación de Etapa 4 y Etapa 5. La diferencia de acercamiento de **0,000649 mm**, entre fuentes distintas y con avance común, no sustituye un contraste online/yoked hacia el mismo objetivo. Un video ilustra posiciones; no aporta el control causal ausente.  

La crítica a los costes del motor sí merece atención, pero **congelar C++ no se deduce de ella**. El controlador residente fue exacto y ligeramente más lento; el cuerpo completo ocupa **1,33 %** del milisegundo perfilado. Hacer gratuito ese cuerpo daría solo **1,0135×** en esa muestra. Hay que atacar recurrencia y fronteras reales, no atribuir el coste dominante a MuJoCo.  

## 2. ¿Dos segundos pueden discriminar algo?

**Sí, con una perturbación compatible con la autoridad disponible; no garantizan navegación.**

Con límite de mando \(|\omega|\le5^\circ/s\), tras una perturbación a 1,5 s quedan:

| Horizonte | Observación posterior | Integral máxima de mando |
|---|---:|---:|
| 2 s | 0,5 s | 2,5° |
| 3 s | 1,5 s | 7,5° |

Compensar **30–45° mediante ese mando** requiere al menos **6–9 s al techo**: terminaría no antes de 7,5–10,5 s desde ON. **No es una cota de toda rotación pasiva**, ni demuestra que el controlador alcanzará su techo. Es suficiente para rechazar “solo falta duración” como justificación del diseño propuesto.

La geometría publicada sitúa inicialmente las fuentes a **≈1,13 mm**, con errores de rumbo de **+17,35°/−18,38°**. Recorrer esa distancia a 0,2 mm/s exige ≈5,65 s incluso sin rodeos; es una referencia cinemática, no una predicción de llegada.  

Calculé una alternativa menor: rotar rígidamente la pose antenal inicial **1° alejándola de la fuente** cambia el índice olfativo bilateral normalizado en **≈0,016**. Por tanto, una perturbación pequeña puede ser sensorialmente visible. No demuestra respuesta neuronal.

Con 1,5 s posteriores, dos trayectorias ideales de igual rapidez y diferencia de rumbo ≤1° pueden separarse como máximo:

\[
2v\Delta t\sin(0,5^\circ)=0,005236\ \text{mm}.
\]

La ventaja de distancia puede ser bastante menor. **La precisión XY debe entrar al contrato:** para errores de posición \(\epsilon_x\), reservar hasta \(2\epsilon_x\) en el contraste de distancias. Para bearing, incluir también el término geométrico \(\arcsin(\epsilon_x/d_{\min})\), además del error angular. No basta conservar únicamente una reserva de yaw.

## 3. Yoked correcto y perturbación

En el yoked, **las decisiones neuronales deben seguir moviendo el cuerpo**. Lo que se retira es la dependencia del olor respecto de esa trayectoria:

\[
u_{\rm online}(t)=c(p_{\rm antenas}(t)),\qquad
u_{\rm yoked}(t)=u_{\rm donante}(t).
\]

Mantener cerebro, lector, propiocepción, otros sensores, avance y contactos. Registrar entradas **comprometidas y consumidas**, preservando el retardo de un intervalo. No reordenar la cinta ni usar muestras futuras.

Aplicar a ambos brazos el mismo pulso mecánico fechado, por ejemplo:

\[
\tau_z(t)=\tau_0\,\mathbf1_{[t_*,\,t_*+20\,ms)}.
\]

Fijar \(\tau_0\) mediante el preflight corporal, antes de las vidas neuronales; no ajustarlo según el giro olfativo. Añadirlo a las fuerzas existentes antes de cada subpaso físico: no teletransportar `qpos` ni sustituir el mando. MuJoCo admite fuerzas aplicadas antes de `mj_step`. :chatgpt-content-reference{index="11"}

Tras el pulso, comparar progreso hacia **la misma fuente**, error de bearing y apoyo. Un positivo exige ventaja geométrica resuelta, no solamente comandos diferentes. Una ejecución solitaria no proporciona ese contraste.

## 4. Prueba útil hoy

**Tres replays corporales de 200 ms:** pulso previsto, fuerza cero y pulso con refinamiento físico; estados iniciales completos emparejados, techo conjunto **120 s**. Verificar magnitud realmente inducida, apoyo, retorno de fuerzas a cero y diferencia sensorial. No son pruebas neuronales.

Si el pulso no mantiene apoyo o la señal geométrica queda debajo de la incertidumbre, no lanzar A. Si pasa y se acepta el coste agregado, **un piloto de 2 s es científicamente defendible sin esperar un motor perfecto**. Su negativo quedaría limitado a ese horizonte y perturbación. Si no se financia ese coste, elegir C, sin afirmar que otro milisegundo exacto garantiza aceleración.

## Código comprobable: `criba_feedback.py`

Calcula horizonte, geometría y costes usando los JSON originales. No simula. **Lo ejecuté en CPU** con geometría y CAMPOS materializados desde los textos públicos y hashes coincidentes; los tiempos se transcribieron de campos de los recibos.

```python
"""Criba cinemática y de coste; sólo biblioteca estándar, sin simulación.
Usa geometría/tiempos del paquete 7186929. No estima respuesta neuronal.
Los costes son escalados linealmente desde corridas de 400 ms con preparación.
"""
import argparse, hashlib, json, math
from pathlib import Path
GEOM = "3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3"
FIELDS = "fcf9f2879d703cc03e3b8dca3908701279f12ce113188ce1eaf8c65484401133"
def need(ok, msg):
    if not ok: raise ValueError(msg)
def read(p, hashes, expected=None):
    b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
    need(expected is None or h==expected, "Hash geométrico distinto: "+str(p))
    hashes[str(p)]=h
    return json.loads(b)
def run(folder, T, tp, angle):
    need(all(math.isfinite(x) for x in (T,tp,angle)) and 0<=tp<T and 0<angle<=180,
         "Se requiere 0<=pulso<horizonte y 0<ángulo<=180")
    hashes={}
    g=read(folder/"runs/plus/executed_sources/16__etapa4_diseno_20260923_17__GEOMETRY.json",hashes,GEOM)
    fields=read(folder/"recovery/CAMPOS.json",hashes,FIELDS)
    close=read(folder/"recovery/CLOSE_01.json",hashes)
    old=read(folder/"prior/CLOSE_01.json",hashes)
    n400=(old["budget"]["aggregate_wall_s"]-old["reference_plus"]["wall_s"])/2
    r400=sum(close["runs"][s]["wall_s"] for s in ("plus","minus"))/2
    need(n400>0 and r400>0, "Tiempos no positivos")
    q=g["prepared_qpos_root"]; a=g["prepared_antennae_mm"]
    scale=g["body_native_units_per_mm"]; v=g["forward_command_mm_s"]
    need(scale>0 and v>0 and len(q)==7 and len(a)==2, "Geometría inválida")
    origin=[x/scale for x in q[:2]]
    w,x,y,z=q[3:]; yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    rest=T-tp; alpha=math.radians(angle); out={}
    for side,f in fields.items():
        need(f["geometry_sha256"]==GEOM, "Fuentes ligadas a otra geometría")
        source=f["source_mm"];sigma=f["sigma_mm"]
        beta=math.atan2(source[1]-origin[1],source[0]-origin[0])-yaw
        beta=math.atan2(math.sin(beta),math.cos(beta))
        # Rotación rígida contrafactual que aleja el rumbo inicial de la fuente.
        turn=-math.copysign(alpha,beta);ca,sa=math.cos(turn),math.sin(turn)
        def concentrations(rotated):
            result=[]
            for p in a:
                dx,dy=p[0]-origin[0],p[1]-origin[1]
                xy=[origin[0]+ca*dx-sa*dy,origin[1]+sa*dx+ca*dy] if rotated else p[:2]
                result.append(math.exp(-math.dist(xy,source)**2/(2*sigma*sigma)))
            return result
        c0,c1=concentrations(False),concentrations(True)
        out[side]={"bearing_initial_deg":math.degrees(beta),
            "distance_initial_mm":math.dist(origin,source),
            "straight_distance_over_v_s":math.dist(origin,source)/v,
            "counterfactual_rotation_deg":math.degrees(turn),
            "c_initial":c0,"c_after_rigid_rotation_at_initial_pose":c1,
            "delta_normalized_LR":(c1[0]-c1[1])/sum(c1)-(c0[0]-c0[1])/sum(c0)}
    n,r=n400*T/.4/3600,r400*T/.4/3600
    return {"inputs_sha256":hashes,"T_s":T,"perturbation_s":tp,
        "remaining_s":rest,"assumed_decoder_limit_deg_s":5.0,
        "maximum_command_integral_after_perturbation_deg":5*rest,
        "command_time_for_angle_at_ceiling_s":angle/5,
        "earliest_command_only_compensation_from_on_s":tp+angle/5,
        "geometry":out,
        "ideal_equal_speed_position_separation_upper_mm":2*v*rest*math.sin(alpha/2),
        "cost_linear_h":{"one_native":n,"one_reference":r,
            "donor_and_two_native_and_two_reference":3*n+2*r,
            "three_native_and_three_reference":3*(n+r)},
        "limits":["Geometría en el preparado, NO en el instante futuro del pulso.",
            "La integral de mando no acota toda rotación física pasiva.",
            "Cota XY sólo para dos trayectorias ideales de igual rapidez y diferencia de rumbo <=ángulo.",
            "Tiempo extrapolado, no corrida larga medida ni presupuesto autorizado.",
            "No efectos neurales, contactos, navegación o etapa5 simulados."],
        "organism_executed":False,"CUDA_executed":False}
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",type=Path,required=True)
    p.add_argument("--horizon",type=float,default=2.)
    p.add_argument("--pulse",type=float,default=1.5)
    p.add_argument("--angle",type=float,default=30.)
    args=p.parse_args()
    print(json.dumps(run(args.files,args.horizon,args.pulse,args.angle),
                     indent=2,ensure_ascii=False,allow_nan=False))
```

```bash
python criba_feedback.py --files /ruta/extraida/files \
  --horizon 2 --pulse 1.5 --angle 30

python criba_feedback.py --files /ruta/extraida/files \
  --horizon 2 --pulse 0.5 --angle 1
```

**Ejecución:** ambos comandos CPU y tres controles de entradas inválidas; resultados conservados. Compensación mínima de mando calculada: **7,5 s desde ON** para el primer caso y **0,7 s** para el segundo. No son tiempos de recuperación corporal.

**Lectura efectiva:** cierre, README, hoja de ruta, CAMPOS, geometría y cierre previo de campaña27; fragmentos CSV 349–354, 399–400 y 397–400 del brazo opuesto. Del snapshot `619942…`: README, `BODY_COST_NOTE_01.md` y `PREPARATION_PROVENANCE_01.json`. **No descargué el CSV completo/NPZ ni ejecuté organismo, CUDA o MuJoCo.** No leí el borrador local `STAGE5_ABC_DRAFT`; utilicé su descripción proporcionada.

:chatgpt-content-reference{index="13"}[Código y resultados CPU, con procedencia de los datos transcritos](sandbox:/mnt/data/AXIOMA_FEEDBACK_DECISION_CPU.zip)

**Conclusión:** no congelar el motor por consenso ni lanzar la perturbación de Gemini como supuesto cierre. Preparar un contraste causal pequeño, presupuestar todas sus vidas y conservar el desarrollo del motor como vía paralela de ingeniería.
