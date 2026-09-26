## Decisión

**No lanzaría otra vida neural larga todavía.** Primero separaría geometría y giro, y probaría la autoridad del mando sobre el cuerpo actual. Mantendría congeladas las ganancias; no introduciría músculos ni ampliaría CNS→VNC→MN para explicar esta vida.

**Exposición y alcance:** las tres hipótesis iniciales —mando ineficaz, cambio geométrico y persistencia mecánica— parten únicamente de tu resumen. Después consulté documentación oficial de MuJoCo. **No pude acceder a los archivos del commit fijado: no revisé código/recibos/hashes ni descargué o ejecuté sus NPZ.** Este análisis está condicionado a tus datos. El Python incluido sí fue ejecutado, exclusivamente con pruebas sintéticas.

## 1. Separación que puede cambiar la decisión

Con tus cifras, los cambios del **error informado** son:

| Tramo | Cambio |
|---|---:|
| Preparado → antes del pulso | +2,31215° |
| Durante el pulso | −1,275° |
| Fin del pulso → final | +8,144° |

Deterioro neto: **+9,18115°**. Si esa métrica es absoluta, estas diferencias **no son cambios de error firmado**. Tampoco asumiría que «preparado» coincide con el primer estado dinámico registrado.

En 20 ms, la orden limitada a 5°/s suma como máximo **0,1° de giro solicitado**, frente a 1,275° de disminución del error. No atribuiría esa disminución al mando sin separar geometría y respuesta física. **Limitar la orden no limita necesariamente el yaw real**: importa la dinámica y semántica del actuador. :chatgpt-content-reference{index="0"}

Para fuente fija \(s\), posición \(p=(x,y)\), \(r=s-p\) y orientación corporal \(\psi\):

\[
\beta=\operatorname{atan2}(r_y,r_x),\quad e=\beta-\psi,
\qquad\boxed{\Delta e=\Delta\beta-\Delta\psi}.
\]

Desenvolver los ángulos usando el eje corporal y punto de referencia correctos, en un mismo marco. Con velocidad real \(v\):

\[
\dot\beta=\frac{r_yv_x-r_xv_y}{\|r\|^2},\qquad
\frac{d(e^2/2)}{dt}=e(\dot\beta-\dot\psi).
\]

**Un giro correctivo, \(e\dot\psi>0\), puede coexistir con error creciente si el bearing cambia más deprisa.** Esta separación es cinemática: no atribuye automáticamente \(\Delta\beta\) al avance tónico, porque el pulso también puede modificar la trayectoria.

Escala relevante: 0,2 mm/s supone 0,4 mm nominales en 2 s. A 2,29 mm de la fuente, un movimiento transversal de 0,2 mm/s produce aproximadamente 5°/s de cambio de bearing. **No afirmo que ésa sea vuestra distancia**; necesitamos \(r(t)\) y desplazamiento real, no sólo la consigna.

**Primer entregable para Codex:** \(\Delta\beta\), \(-\Delta\psi\) y \(\Delta e\) en prepulso, pulso y pospulso, con timestamps efectivos. Si domina \(\Delta\beta\) mientras el giro es correctivo, aislar orientación del avance antes de aumentar actividad neural.

## 2. Auditorías baratas antes de interpretar neuronas

**Filtro.** La diferencia de integrales es **+1,0503°**. Para un EMA continuo ideal, \(\tau\dot z=u-z\):

\[
\int z\,dt-\int u\,dt=\tau[z(0)-z(T)].
\]

La memoria y una ventana finita pueden cambiar el signo del integral. Hay que comprobar la recurrencia discreta real, estado inicial y orden de umbral/saturación. Si el umbral sólo anula salidas menores que 0,0005 rad/s, su efecto integral máximo en 2 s es **0,0573°**: no explica solo 1,0503°. La saturación puede afectar desigualmente lóbulos de distinto signo.

Recalcularía sin red las integrales de crudo, EMA interno y etapas posteriores sobre **el mismo intervalo**. Un integral crudo casi cero puede ocultar órdenes opuestas en momentos distintos; no equivale a falta de mando ni demuestra inversión neural.

**Relojes.** Emparejar eventos reales: adquisición sensorial, tiempo neural, filtro, aplicación motora y `mjData.time`. Verificar unidades, origen, deriva y subpasos. `mj_step` avanza estado y tiempo: asignar la pose posterior al timestamp de entrada puede introducir un desfase de un paso. :chatgpt-content-reference{index="1"} No «corregir» alineación eligiendo el desfase que mejor explica el resultado.

**Pulso.** Para exactamente 0,020 s, el impulso externo nominal es:

\[
J_\tau=-9{,}345395714306934\times10^{-12}\ \mathrm{N\,m\,s}.
\]

Integrar el **torque aplicado por subpaso**, verificar cuerpo/eje y comprobar cero después del pulso. MuJoCo no limpia automáticamente controles/fuerzas aplicados: el flag «viento terminado» no demuestra que la fuerza desapareció. :chatgpt-content-reference{index="2"} Revisar también si el extremo inclusivo `1000..1020` añadió un paso. No convertir \(J_\tau\) directamente en \(\Delta yaw\): faltan inercia efectiva, velocidad inicial y contactos.

## 3. Controles físicos y criterio de continuación

R = orden final registrada de giro; 0 = anular sólo ese mando, conservando las demás entradas. **No reproducir posiciones:** el cuerpo evoluciona físicamente. Mantener modelo, inicialización, estabilización y reloj.

**Primera tanda: cuatro rollouts sin actualizar la red.**

| Ensayo | Giro | Pulso | Avance |
|---|---|---|---|
| A | R | No | 0,2 mm/s |
| B | R | Sí | 0,2 mm/s |
| C | 0 | No | 0,2 mm/s |
| D | 0 | Sí | 0,2 mm/s |

Comparar \(\beta,\psi,e\) y velocidad de giro durante todo el tramo. **B−A** mide el efecto del pulso con R fijado; **D−C**, con giro anulado. **B−D** y **A−C** muestran el efecto de introducir R. La diferencia entre contrastes mide interacción pulso×mando en la escala de cada métrica; no asumir aditividad.

**Si importa la geometría**, repetir A/B retirando sólo el avance tónico. No bloquear x/y ni mover la fuente. Retirar avance también cambia contactos: separar ese efecto causal de la descomposición cinemática anterior. Añadir C/D sin avance únicamente si hace falta resolver la interacción triple.

**Prueba de autoridad:** órdenes prescritas +5 y −5°/s, inyectadas después del filtro, sin ajustar ganancias. Medir respuesta diferencial, giro alcanzado y retraso; no exigir inversión instantánea frente a inercia. Prueba la planta, no competencia neural.

- **La planta no consigue corrección con órdenes prescritas:** revisar interfaz, signos, contactos y torque residual antes de ampliar el circuito neural.
- **Gira correctamente, pero el bearing se aleja por el avance:** orientación sin avance como control de referencia; después estudiar la combinación.
- **Hay autoridad y geometría/relojes están aclarados, pero R es débil, tardío o equivocado:** autorizar una prueba neural corta sobre generación/decodificación del mando. Todavía no localizar el defecto en CNS, VNC o MN.

**No inventar un checkpoint a 1000 ms a partir de una pose.** Pueden importar activaciones, entradas persistentes, warmstart y estado externo del controlador. La reproducción exacta requiere estado pertinente y condiciones compatibles de versión/arquitectura. :chatgpt-content-reference{index="3"} Reconstruir desde el reset físico y validar el prefijo contra la traza. Sin esa validación son pruebas del cuerpo, **no contrafactuales exactos de campaña40**. No tener checkpoint neural completo no impide por sí solo controles físicos.

**Replay no prueba feedback neural:** responde «qué hace la planta con \(u_{40}(t)\) fijado», no «qué habría decidido la red con otra trayectoria». Puede subsistir feedback del controlador físico de bajo nivel; no confundirlo con cierre olfativo neural.

## 4. Python

No inventa claves NPZ: recibe arrays identificados explícitamente. Metros, radianes y segundos físicos. `relojes` exige pares del mismo evento; informa desfases sin corregirlos. Las integrales requieren valores mantenidos por intervalo, no muestras post-step.

```python
"""Auditoría cinemática: no carga campañas ni demuestra feedback neural.
Posiciones en metros, ángulos en radianes y tiempo físico en segundos.
Las muestras de pose deben describir el mismo punto/eje usados por el error.
"""
import numpy as np


def reloj(t, nombre: str = "tiempo") -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or t.size < 2 or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError(f"{nombre}: exigir muestras finitas y crecientes")
    return t


def relojes(t_mj_s, t_neural_ms) -> dict:
    """Pares del MISMO evento; no emparejar por longitud ni corregir desfases."""
    p = reloj(t_mj_s, "MuJoCo")
    n = reloj(t_neural_ms, "neural") / 1000.0
    if p.shape != n.shape:
        raise ValueError("Falta correspondencia explícita entre eventos")
    x, y = p - p.mean(), n - n.mean()
    escala = float((x @ y) / (x @ x))
    offset = float(n.mean() - escala * p.mean())
    return dict(escala_neural_fisica=escala, offset_s=offset,
                residuo_max_s=float(np.max(np.abs(n - escala*p - offset))),
                deriva_s=float((n[-1]-n[0]) - (p[-1]-p[0])),
                dt_mj_min_max_s=(float(np.diff(p).min()),
                                 float(np.diff(p).max())))


def descomponer(t_s, xy_m, yaw_rad, fuente_xy_m, ventanas,
                error_firmado_rad=None, distancia_min_m: float = 1e-9) -> dict:
    """ventanas: [(inicio_s, fin_s), ...]. No usar error absoluto.
    unwrap exige muestreo suficiente; no puede detectar vueltas ocultas.
    """
    t = reloj(t_s)
    p, y, s = [np.asarray(a, dtype=float)
               for a in (xy_m, yaw_rad, fuente_xy_m)]
    if p.shape != (len(t), 2) or y.shape != t.shape or s.shape != (2,):
        raise ValueError("Formas requeridas: xy=(N,2), yaw=(N,), fuente=(2,)")
    if not all(np.isfinite(a).all() for a in (p, y, s)):
        raise ValueError("Pose/fuente no finita")
    if not np.isfinite(distancia_min_m) or distancia_min_m <= 0:
        raise ValueError("distancia_min_m debe ser positiva")
    r = s - p
    if np.any(np.linalg.norm(r, axis=1) <= distancia_min_m):
        raise ValueError("Bearing indefinido o mal condicionado junto a fuente")
    b = np.unwrap(np.arctan2(r[:, 1], r[:, 0]))
    y = np.unwrap(y)
    e = b - y
    e -= 2*np.pi*np.floor((e[0] + np.pi) / (2*np.pi))
    discrepancia = None
    if error_firmado_rad is not None:
        obs = np.asarray(error_firmado_rad, dtype=float)
        if obs.shape != t.shape or not np.isfinite(obs).all():
            raise ValueError("Error firmado incompatible")
        discrepancia = float(np.rad2deg(np.max(np.abs(
            np.angle(np.exp(1j*(obs-e)))))))
    filas = []
    for a, z in ventanas:
        if not (t[0] <= a < z <= t[-1]):
            raise ValueError("Ventana fuera del registro: no extrapolar")
        db, dy, de = [float(np.diff(np.interp([a, z], t, v))[0])
                      for v in (b, y, e)]
        filas.append(dict(inicio_s=a, fin_s=z, geometria_deg=np.rad2deg(db),
                          giro_deg=np.rad2deg(-dy), delta_error_deg=np.rad2deg(de),
                          bordes_interpolados=not (np.any(t == a) and np.any(t == z))))
    return dict(ventanas=filas, discrepancia_error_deg=discrepancia,
                alcance="Cinemática; no identifica feedback neural")


def integral_zoh(t_bordes_s, valor_intervalo, inicio: float, fin: float) -> float:
    """N-1 valores aplicados en [t[i],t[i+1]); NO muestras post-step."""
    t = reloj(t_bordes_s)
    u = np.asarray(valor_intervalo, dtype=float)
    if u.shape != (len(t)-1,) or not np.isfinite(u).all():
        raise ValueError("Exigir un valor finito por intervalo físico")
    if not (t[0] <= inicio < fin <= t[-1]):
        raise ValueError("Ventana inválida")
    dt = np.maximum(0.0, np.minimum(t[1:], fin) - np.maximum(t[:-1], inicio))
    return float(u @ dt)
```

:chatgpt-content-reference{index="5"}[Auditor Python](sandbox:/mnt/data/axioma_stage45_auditoria.py) · :chatgpt-content-reference{index="6"}[Pruebas sintéticas](sandbox:/mnt/data/test_axioma_stage45_sintetico.py)

Las pruebas comprobaron descomposición, cruce angular, desfase/deriva, impulso de 20 ms y rechazo de entradas inválidas. Interpolar bordes no recupera resolución perdida; `unwrap` tampoco descubre vueltas ocultas entre muestras.

**No identificable todavía:** contribución neural causal, localización del defecto, contrafactual neural de campaña40 y reparto inercia/contactos a partir de cuatro errores. El fracaso en 2 s tampoco demuestra incapacidad a otro horizonte. Los hashes acreditarían identidad de archivos, no suficiencia causal.

**La decisión de esta tanda es corregir geometría/planta/temporización, o autorizar una prueba neural corta porque esos frentes ya fueron descartados.** Ningún resultado de estos controles acredita, por sí solo, marcha natural o avance de etapa biológica.
