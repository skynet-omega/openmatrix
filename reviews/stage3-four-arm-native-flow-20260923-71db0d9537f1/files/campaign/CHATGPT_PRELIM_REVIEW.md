# Revisión externa ChatGPT de consejos etapa 3

Turno: aa4e56ad-d1c7-4d41-83bc-fed5409126b7
Finalización: 1790150457.776277
Alcance: texto externo; verificar localmente código y resultados.

**El consejo 1 ya está implementado; el 2 confunde igualdad de número de destinos con equivalencia del operador; el 3 utiliza un contraste insuficiente para declarar orientación.** No modificaría las cuatro corridas en curso. Sus resultados deberán quedar separados de las trazas PFG históricas.

## 1. Qué ya se hizo y qué no puede afirmarse

### 1) «Capturar dentro del kernel CUDA»

**Ya se hace en `09/native_tap.py`.** Se añade una escritura FP64 del subtotal inmediatamente antes de `tanh`, tanto en el kernel general como en el ORN→PN. El buffer permanece en el grafo capturado; Python conserva referencias al construirlo y lee después del avance exitoso. Además, se rechazan las cuatro filas focales si sus manifiestos indican reemplazos regionales posteriores. 

La captura proporciona **entrada efectiva agregada**, no todavía contribuciones por aferente ni corriente en pA. Conserva información incluso cuando `tanh` satura: no necesita invertir el target.

Dos precauciones prácticas:

**El orden exportado es PN_R, PN_L, DNa02_R, DNa02_L**, según `TARGET_IDS=(10176,10208,10360,523769)`. No interpretar las primeras dos columnas automáticamente como L/R.

**La marca guardada es `h.time_ns`**, mientras el valor corresponde a la última evaluación de coeficientes del intento aceptado. Para medir latencias entre interfaces hay que registrar el tiempo efectivo de esa etapa y el de la emisión/consumo del puerto. Una muestra etiquetada por endpoint no proporciona esos tiempos por sí sola. 

### 2) «Homologar ambas PN porque tienen 1.095 destinos»

**La fuente leída no sostiene “1.095 ambas”.** `pn_general_output_brain.py` trabaja con **una** fuente, `self._online_ports.pn_row`. Comprueba una partición de sus consumidores: **629 generales y 466 dinámicos**, total 1.095. En los generales utiliza liberación local, modifica temporalmente los pesos efectivos y sustituye la transmisión de esa fuente por uno; su `q` anterior permanece como diagnóstico legacy. 

La ruta solicitada `files/campaign/PN_PAIR_STRUCTURE.json` devuelve **404 y no figura en el manifiesto de ese paquete**. No puedo verificar el inventario derecho desde ese archivo. `parent/AUDIT.json` confirma la identidad/lado de ambas PN, pero sus cifras 478/493 son **entradas**, no destinos de salida.  

Aunque otra captura demostrara 1.095 destinos para ambas, seguirían faltando identidades de receptores, compartimentos, pesos, filtros, escalas y fronteras temporales. **Homologar la infraestructura es razonable; copiar rutas o fisiología izquierda a derecha sería una intervención de modelo, no una optimización neutral.** Solicitaría el archivo faltante o las listas efectivas de destinos antes de cuantificar esa diferencia.

### 3) «Prueba definitiva a 320 ms porque supera 0,020°»

Recalculé, desde los valores textuales publicados:

| A 320 ms pos-ON —muestra 331 del registro histórico— | Resultado |
|---|---:|
| Derecha − sham | **+0,02096717°** |
| Contraste bilateral \((\Theta_L-\Theta_R)/2\) | **−0,01250154°** |
| Desplazamientos izquierda/derecha/uniforme/sham | **Todos positivos** |

Por tanto, superar 0,020° en **derecha−sham no demuestra giro derecho**. Tampoco convierte 320 ms en horizonte definitivo. Es una lectura de la rama PFG histórica, no de las cuatro corridas protegidas actuales. 

Conservaría el horizonte y la métrica predeclarados de la campaña actual: desplazamientos desde ON de los cuatro brazos, contraste bilateral, mando aplicado y controles físicos. No elegiría retrospectivamente el instante que cruza un umbral.

## 2. Tres experimentos rivales, no tres cambios simultáneos

**No ejecutaría A+B+C a la vez.** Primero cerrar las cuatro referencias actuales; después elegir una intervención con su plan propio.

| Experimento | Intervención concreta y control | Falsador e interpretación |
|---|---|---|
| **A. La sustitución de salida PN contribuye materialmente a la inversión** | Retirar **solo la sustitución local de las 629 salidas generales** de PN10208 mediante la ruta deshabilitada ya prevista por la clase; conservar las 466 dinámicas, anatomía, PN fina y demás ecuaciones. Comparar cuatro condiciones con el padre emparejado. No saltarse validadores ni copiar destinos. | Si la intervención se verifica en esas salidas, pero no modifica el contraste de entrada DNa02 por encima de la incertidumbre numérica, debilita esa sustitución como explicación suficiente. **No refuta toda la periferia/ALLN.** |
| **B. Una convergencia posterior concreta transmite el contraste invertido** | Elegir un solo grupo mediante una regla de flujo firmado fijada antes de la intervención, no por yaw. Sustituir únicamente su contribución hacia DNa02 por la historia sham emparejada; conservar sus otras salidas y el resto de la recurrencia. Cuatro condiciones. | La modificación de la suma intervenida es predecible por construcción; **no cuenta como confirmación causal**. El discriminador es la respuesta de DNa02, mando y cuerpo. Si no responde como se predijo, ese grupo no basta para explicar la conducta. |
| **C. El sesgo está en lector/actuación o cuerpo** | Desde un estado corporal idéntico, aplicar \(0,+\omega_0,-\omega_0\), con igual avance y exposición sham. Fijar \(\omega_0\) por el contrato del actuador antes de observar yaw. Es un control diagnóstico, no un nuevo controlador operativo. | Una respuesta espejo, después de restar el control cero, debilita un sesgo puramente mecánico. Una respuesta no espejo localiza un problema de actuación/cuerpo, pero no explica automáticamente la inversión ya observada aguas arriba. |

**Presupuesto propuesto:** para A o B, hasta cuatro brazos adicionales de 400 ms, máximo 2.050 s por brazo; reutilizar controles solamente si estado preparado, operador y pipeline coinciden. Para C, un triplete de 200 ms, máximo 1.000 s por brazo. Son techos de ejecución, no estimaciones de duración. Ningún fallo habilita otra ganancia o ventana.

El comparador numérico sigue siendo obligatorio. El paquete actual documenta **1.036 eventos aceptados en ambos motores**, pero hasta **3.125 ns de desfase**; la criba de 20 ms conserva `screen_pass=false`, con mayor discrepancia en `kc_axonal_state/trough`. Los recuentos iguales no certifican historia temporal ni fidelidad a 400 ms.  

Antes de interpretar una intervención: observador on/off, misma aritmética y eventos; después, refinamiento del brazo y control pertinentes hasta la ventana interpretada, sin aflojar \(10^{-4}\) ni convertir discrepancias de variables distintas en una unidad común.

## 3. Analizador local de desfases —sin simular—

El siguiente programa recalcula los **desfases entre eventos aceptados** desde los dos `EVENT_AUDIT.json` publicados. No estima latencia olfativa ni latencia PN→DNa02.

Utiliza el contrato específico documentado: bloques completos de 125.000 ns y predictores restaurados de 62.500 ns. Empareja por productor, fila, ID y orden temporal, sin buscar un desplazamiento que mejore la coincidencia. Ese criterio procede del comparador publicado. 

**Archivo: `desfase_eventos.py`**

```python
"""Lectura de EVENT_AUDIT del snapshot 2ed2781; no simulador.
Emparejamiento ordinal por neurona, no estimación de latencia biológica.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

BLOQUE_NS = 125000

def exigir(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)

def cargar(ruta):
    contenido = ruta.read_bytes()
    datos = json.loads(contenido)
    eventos = defaultdict(list)
    bloques = []
    predictores = 0
    cuantizacion = 0.0
    for bloque in datos["blocks"]:
        duracion = bloque["duration_ns"]
        exigir(duracion in (BLOQUE_NS // 2, BLOQUE_NS),
               "Duración ajena al contrato revisado")
        if duracion == BLOQUE_NS // 2:
            predictores += len(bloque["events"])
            continue
        inicio = bloque["start_elapsed_ns"]
        exigir(type(inicio) is int and inicio >= 0, "Reloj inválido")
        bloques.append(inicio)
        for e in bloque["events"]:
            fila, identidad = e["row"], e["neuron_id"]
            exigir(type(fila) is int and type(identidad) is int,
                   "Fila/ID deben ser enteros")
            exigir(fila >= 0 and identidad >= 0, "Fila/ID negativos")
            tiempo = float(e["time_s"]) * 1e9
            exigir(math.isfinite(tiempo) and 0 <= tiempo <= duracion,
                   "Evento fuera del bloque")
            redondeado = round(tiempo)
            cuantizacion = max(cuantizacion, abs(tiempo-redondeado))
            post = e["post_q"]
            exigir(post is None or math.isfinite(float(post)),
                   "Post no finito")
            clave = (e["producer"], fila, identidad)
            eventos[clave].append((inicio+redondeado, post))
    exigir(bloques and all(a < b for a, b in zip(bloques, bloques[1:])),
           "Bloques aceptados vacíos, duplicados o desordenados")
    for lista in eventos.values():
        lista.sort(key=lambda x: x[0])  # Conserva orden en marcas iguales.
    return eventos, bloques, {
        "sha256": hashlib.sha256(contenido).hexdigest(),
        "eventos_predictores": predictores,
        "eventos_aceptados": sum(map(len, eventos.values())),
        "redondeo_max_ns": cuantizacion,
    }

def comparar(ruta_a, ruta_b):
    a, bloques_a, meta_a = cargar(ruta_a)
    b, bloques_b, meta_b = cargar(ruta_b)
    exigir(bloques_a == bloques_b, "Cronología de bloques distinta")
    diferencias, tiempos, posts = [], [], []
    peor = None
    for clave in sorted(set(a) | set(b)):
        aa, bb = a.get(clave, []), b.get(clave, [])
        if len(aa) != len(bb):
            diferencias.append({
                "productor_fila_id": list(clave),
                "n_A": len(aa), "n_B": len(bb)
            })
            continue
        for orden, ((ta, pa), (tb, pb)) in enumerate(zip(aa, bb)):
            delta = abs(ta-tb)
            tiempos.append(delta)
            if pa is not None and pb is not None:
                posts.append(abs(float(pa)-float(pb)))
            if peor is None or delta > peor["desfase_ns"]:
                peor = {
                    "productor_fila_id": list(clave),
                    "orden": orden, "t_A_ns": ta, "t_B_ns": tb,
                    "desfase_ns": delta
                }
    return {
        "A": meta_a, "B": meta_b,
        "diferencias_recuento": diferencias,
        "eventos_emparejados": len(tiempos),
        "max_desfase_ns": max(tiempos) if tiempos else None,
        "max_diferencia_post": max(posts) if posts else None,
        "peor": peor,
        "alcance": "Desfase numérico de eventos; no latencia de circuito.",
        "admisión_etapa3": False
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("causal", type=Path)
    p.add_argument("referencia", type=Path)
    args = p.parse_args()
    print(json.dumps(comparar(args.causal, args.referencia),
                     indent=2, ensure_ascii=False, allow_nan=False))
```

```bash
python desfase_eventos.py \
  /ruta/extraida/09/audit_causal_20_01/EVENT_AUDIT.json \
  /ruta/extraida/09/audit_native_20_01/EVENT_AUDIT.json
```

**No ejecuté este programa sobre esos archivos.** La salida publicada que debe contrastar contiene 528/527 eventos predictores, 1.036/1.036 aceptados y máximo desfase de 3.125 ns; no convierto esas cifras en un criterio nuevo de aceptación.

## Alcance de esta revisión

**Leídos completos:** índices de ambas entregas; `09/native_tap.py`, `09/README.md`, `09/ACCEPTED_EVENT_COMPARE.json`, `09/compare_accepted_events.py`; `pn_general_output_brain.py`, `audit_load.py`, manifiesto de `116fb6…` y `HISTORICAL_LATE_READBACK.json`. **Parcialmente:** `AUDIT.json` —líneas 1–210 y 280–380— y `08/REFERENCE_20_COMPARE.json` —líneas 1–70—.

**Ejecuté únicamente la aritmética de la tabla histórica a 320 ms.** La descarga del ZIP no se completó; no verifiqué sus miembros ni ejecuté CUDA, los NPZ, el analizador de eventos o las cuatro corridas actuales. `PN_PAIR_STRUCTURE.json` sigue siendo el archivo puntual faltante para comprobar esa afirmación de destinos.

**Conclusión:** conservar el tap, no homologar anatomía por un número de destinos y no declarar definitiva una separación respecto de sham. La próxima evidencia útil es el contraste entre entradas PN efectivas, entrada DNa02 y actuación en las cuatro condiciones del mismo pipeline.

