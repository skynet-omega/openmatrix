**Conservaría íntegro el negativo B y priorizaría comprobar si el contraste probado era principalmente común, mientras el lector necesita una diferencia bilateral.** Hay un resultado cuantitativo adicional: **en el último intervalo, el `tanh` del lector conserva aproximadamente el 99,82 % de su sensibilidad máxima para esta pareja**. Eso no respalda una saturación fuerte de ese lector en ese instante; tampoco identifica dónde se atenuó la señal.

**No ejecuté Python ni los NPZ:** el entorno CPU devolvió `ClientError`. Leí los archivos públicos y comprobé la aritmética indicada con calculadora. El código completo siguiente queda preparado para ejecución local, sin atribuirle una prueba que no ocurrió aquí.

## 1. Qué añade el diagnóstico, sin cambiar el dictamen

`compare_layers.py` guarda diferencias **firmadas S+−S−** en `last_difference`; las medias tardías, en cambio, son absolutas. No son intercambiables para localizar cancelaciones. 

### A. El promedio sensorial mezcla un canal nulo

`mean_abs_300_400ms` de `sensores_usados` es **0,02881976**, pero promedia tres canales, incluido el tercero nulo. Para los dos canales olfativos es:

\[
0,0288197599\times\frac32=\mathbf{0,0432296399}.
\]

La diferencia respecto de la concentración de campo —0,04335093— incluye el desplazamiento de una muestra entre campo observado y entrada consumida. **No representa por sí sola pérdida de intensidad en el transductor.**  

### B. El lector resta dos cambios muy parecidos

En las columnas **2 y 3 de `DN_q_usada`**, las diferencias finales son:

\[
\delta q_2=4,27456144\times10^{-5},\qquad
\delta q_3=3,50008252\times10^{-5}.
\]

Su componente común es \(3,88732198\times10^{-5}\), pero el lector recibe:

\[
\delta d=\delta q_2-\delta q_3
=\mathbf{7,74478916\times10^{-6}}.
\]

El índice descriptivo

\[
1-\frac{|\delta q_2-\delta q_3|}
{|\delta q_2|+|\delta q_3|}
=\mathbf{0,900384}
\]

muestra que el 90,04 % de la suma de magnitudes se cancela en esa resta. **Es contabilidad del lector, no demostración de un mecanismo neuronal de normalización.** Uso las columnas que resta el código; este JSON no incorpora sus IDs anatómicos.  

Con baseline común y la fórmula publicada,

\[
\omega=5^\circ/s\,\tanh(250d),
\]

la secante normalizada en ese último intervalo es:

\[
\rho=
\frac{\delta\omega}
{1250\,\delta d}
=\mathbf{0,99819535}.
\]

Por tanto, **la pequeña separación ya existe en la entrada diferencial del lector**; no se explica principalmente por una fuerte saturación de su `tanh` al final. Esto no evalúa saturación en ORN, PN u otras capas, ni durante toda la vida.

Otro dato útil: la diferencia de comandos integrados firmados, **0,0016892447178007766°**, coincide con el L1 publicado dentro del redondeo del resumen. No aparece una pérdida apreciable por cancelación temporal de signos en esa integral. Sigue estando por debajo de 0,004° y **no hubo referencias que calificaran numéricamente este nuevo efecto pequeño**. 

## 2. Tres hipótesis causales propias

Son explicaciones rivales para priorizar intervenciones; pueden coexistir. **No llamaría a ninguna confirmada mediante cocientes entre capas.**

### H1 — Selectividad modal: intensidad común frente a diferencia bilateral

**Hipótesis.** La pareja S+/S− excita sobre todo el modo común. El sistema conserva capacidad direccional, pero ese cambio llega a ambas DNb05 de manera semejante y se resta en el lector.

**Experimento mínimo identificable:** cuatro cintas sensoriales nuevas, obtenidas de las entradas realmente consumidas, no de `q`:

\[
\bar c_k=\frac{c_k^++c_k^-}{2},\qquad
a_k=\frac{|\delta c_{L,k}|+|\delta c_{R,k}|}{4}.
\]

Construir, desde el mismo preparado:

\[
c^{C\pm}_k=\bar c_k\pm a_k(1,1),\qquad
c^{D\pm}_k=\bar c_k\pm a_k(1,-1).
\]

Ambos pares tienen igual magnitud de perturbación por muestra, pero distinta dirección en el espacio de entradas. Mantener tercer canal cero, reloj y lector. **Validar previamente `[0,1]`; no recortar ni reescalar si falla.**

Medir \(S_C=\|\omega_{C+}-\omega_{C-}\|_{L1}\) y \(S_D=\|\omega_{D+}-\omega_{D-}\|_{L1}\), además de flujos efectivos y signos.

**Falsador:** una excitación diferencial correctamente consumida no produce mayor separación que la común, por encima de la incertidumbre numérica. Eso debilita esta explicación para este rango y horizonte.

**Control/coste:** cuatro vidas de 400 ms; máximo 2.050 s por vida, **8.200 s nativos**. No llamar navegación a estas cintas: desacoplan deliberadamente el olor de la pose para identificar sensibilidad.

**Es mi primera opción:** no requiere intervenir un circuito todavía mal localizado ni otro cambio del motor.

### H2 — Realimentación periférica PN→ORN atenúa la respuesta incremental

**Hipótesis.** La realimentación periférica conservada contribuye materialmente a reducir la diferencia transmitida. El recibo identifica **31 contactos de feedback**, distintos de las 629 sustituciones generales desactivadas. Su mera existencia no prueba inhibición, saturación ni eficacia. 

**Experimento:** reproducir las cintas reales S+/S− bajo dos condiciones: feedback intacto frente a **entrada del feedback mantenida en su valor preparado**, exclusivamente en el consumidor periférico. No cambiar pesos, otras salidas PN, los 466 destinos dinámicos ni el lector.

Registrar que la intervención realmente cambia ese término; comparar la diferencia S+−S− con y sin feedback. Utilizar cintas idénticas evita confundir la intervención con cambios posteriores del campo por movimiento.

**Falsador:** el término intervenido cambia, pero la diferencia ORN→salida PN efectiva→DN/mando no se incrementa de forma resuelta. Debilita a ese feedback como explicación suficiente.

**Control/coste:** cuatro vidas de 400 ms, máximo **8.200 s nativos**. Es más caro de implementar que H1: falta el punto consumidor concreto y su mapa de rutas; no sustituiría todo un buffer PN sin comprobar a quién más alimenta.

### H3 — Selectividad temporal: importa el orden de la historia, no solo sus niveles

**Hipótesis.** La rampa lentamente divergente es poco eficaz por la dinámica temporal del circuito. No afirmo una constante de tiempo concreta ni extrapolo el mando final hasta que cruce el umbral.

**Experimento:** dos cintas originales y dos con una permutación temporal fijada:

\[
\pi(k)=
\begin{cases}
399-k,&40\le k<360,\\
k,&\text{resto},
\end{cases}
\qquad k=0,\ldots,399.
\]

Esto conserva primeros y últimos 40 ms, niveles, histograma y exposición acumulada de cada antena, pero invierte la historia intermedia. Los saltos de muestra siguen siendo entradas legales en la frontera de 1 ms; no se omiten eventos.

**Falsador:** cambiar el orden no altera materialmente la transmisión diferencial ni el mando, con los controles numéricos correspondientes. Un positivo demuestra dependencia temporal para esa intervención, **no distingue por sí solo adaptación, filtrado y memoria recurrente**.

**Control/coste:** cuatro vidas de 400 ms, máximo **8.200 s nativos**; originales y permutadas desde el mismo preparado.

### Selección económica

**No ejecutaría las tres campañas.** Primero el análisis CPU y después H1. Para confirmar cualquier efecto nuevo, reservar hasta cuatro referencias emparejadas: techo adicional 8.200 s, **16.400 s máximos por diseño completo**, sin ampliarlo tras fallos.

La diferencia entre dos contrastes de cuatro vidas requiere una reserva de cuatro errores, no de dos. Con 0,002° por vida, la reserva conservadora sería **0,008°**. Es un contrato prospectivo nuevo; **no modifica el gate negativo de 0,004°**. Un contraste nativo prometedor seguiría siendo exploratorio hasta resolverlo frente a las referencias.

## 3. Código CPU sobre los JSON y la geometría reales

`diagnostico_modal.py` usa solo biblioteca estándar. Verifica hashes, calcula la cancelación final, la secante del lector y la geometría inicial. **No genera todavía las cuatro cintas: faltan los NPZ originales en este paquete.**

No escribe en los datos ni ejecuta el código del organismo.

```python
"""Aritmética de resúmenes REALES del snapshot ce920ee; sin NPZ/GPU/organismo.
No modifica el gate, no atribuye causalidad ni clasifica saturación neuronal.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path

HASHES = {
 "result/CLOSE.json":
 "ec4c74ca5f97a7e107b1520727bfa3e1c08c8680532ea6f98235771ef623e933",
 "result/LAYER_DIAGNOSTIC_01.json":
 "217f2a61b464b3b18c1370df584a35a486f09184476f72cae450bb08264de792",
 "result/RAW_VERIFIED_01.json":
 "02c7c1e526542d5e2ac6d1cb1ce5ce64e6b14a2f61f15fba60c689de95e0653e",
 "contract/PLAN.json":
 "5cf0a9b06116686a845bd893a57bd35d58fbb81a81d00238f6d4ac83fc00545f",
 "contract/GEOMETRY.json":
 "3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3",
 "result/compare_layers.py":
 "3d6cc77af9eefd72dc5a54d808d4ddbcd8448b44f6ecc6a5c56a547251d6074c",
 "code/gaussiano_400.py":
 "3b800923c0fc39589c905b7f3d5171a0476ed770f5a77bb28491c84d43be25f5"
}
def exigir(ok, mensaje):
    if not ok:
        raise ValueError(mensaje)
def unico(pares):
    d = {}
    for k, v in pares:
        exigir(k not in d, "Clave JSON duplicada: " + k)
        d[k] = v
    return d
def rechazar(x):
    raise ValueError("Constante JSON no finita: " + x)
def finitos(x):
    if isinstance(x, float):
        exigir(math.isfinite(x), "Número no finito")
    elif isinstance(x, dict):
        for v in x.values():
            finitos(v)
    elif isinstance(x, list):
        for v in x:
            finitos(v)
def cargar(raiz):
    if not (raiz / "result/CLOSE.json").is_file():
        raiz = raiz / "files"
    datos, recibos = {}, {}
    for nombre, esperado in HASHES.items():
        p = raiz / nombre
        bruto = p.read_bytes()
        h = hashlib.sha256(bruto).hexdigest()
        exigir(h == esperado, "Archivo distinto: " + nombre)
        recibos[nombre] = h
        if p.suffix == ".json":
            d = json.loads(bruto, object_pairs_hook=unico,
                           parse_constant=rechazar)
            finitos(d)
            datos[nombre] = d
    return datos, recibos

def geometria(g):
    antenas = g["prepared_antennae_mm"]
    q = g["prepared_qpos_root"]
    exigir(len(antenas) == 2 and all(len(p) == 3 for p in antenas),
           "Geometría antenal")
    exigir(len(q) == 7, "Pose de raíz")
    d = [antenas[0][i] - antenas[1][i] for i in (0, 1)]
    b = math.hypot(*d)
    exigir(b > 0, "Antenas coincidentes")
    lateral = [v / b for v in d]
    frente = [lateral[1], -lateral[0]]
    w, x, y, z = q[3:]
    heading = [1 - 2*(y*y + z*z), 2*(w*z + x*y)]
    if sum(a*c for a, c in zip(frente, heading)) < 0:
        frente = [-v for v in frente]
    medio = [(antenas[0][i] + antenas[1][i])/2 for i in (0, 1)]
    sigma = 2*b
    fuentes = {}
    for nombre, signo in (("plus", 1), ("minus", -1)):
        s = [medio[i] + signo*.5*b*frente[i] + 2*b*lateral[i]
             for i in (0, 1)]
        c = [math.exp(-sum((p[i]-s[i])**2 for i in (0, 1)) /
                      (2*sigma*sigma)) for p in antenas]
        argumento = sum((s[i]-medio[i])*d[i] for i in (0, 1)) / (2*sigma*sigma)
        fuentes[nombre] = {
            "source_mm": s,
            "concentracion_inicial_L_R": c,
            "indice_bilateral": (c[0]-c[1])/(c[0]+c[1]),
            "indice_formula_tanh": math.tanh(argumento)
        }
    return {
        "baseline_mm": b, "sigma_mm": sigma, "fuentes": fuentes,
        "alcance": "Geometría inicial; no trayectoria gaussiana ni navegación."
    }

def analizar(raiz):
    d, hashes = cargar(raiz)
    cierre = d["result/CLOSE.json"]
    capas = d["result/LAYER_DIAGNOSTIC_01.json"]
    raw = d["result/RAW_VERIFIED_01.json"]
    plan = d["contract/PLAN.json"]
    exigir(cierre["scientific_classification"] == "DESCARTADO_EN_ESTE_CONTRATO"
           and cierre["rival"] == "B", "Dictamen distinto")
    exigir(raw["decision"]["native_pair"] == cierre["native_pair"],
           "Recibos del par distintos")
    exigir(cierre["thresholds"] == plan["thresholds"], "Umbrales distintos")
    exigir(capas["trial_late_rows"] == [339, 439], "Ventana descriptiva distinta")
    f = capas["fields"]
    exigir(f["DN_q_usada"]["shape"] == [440, 4], "Layout del lector")
    exigir(f["sensores_usados"]["shape"] == [440, 3], "Layout sensorial")
    for arm in ("plus", "minus"):
        r = raw["runs"]["causal_cuda/" + arm]
        exigir(r["metrics"]["complete"] and r["metrics"]["steps"] == 400,
               "Brazo incompleto")
        exigir(capas["sources"][arm]["traces_sha256"] ==
               r["raw_hashes"]["traces.npz"], "Traza de otra procedencia")
        exigir(capas["sources"][arm]["flow_sha256"] ==
               r["raw_hashes"]["flow/FLOW.npz"], "Flujo de otra procedencia")
    usado = f["DN_q_usada"]["last_difference"]
    delta2, delta3 = usado[2], usado[3]
    diferencial = delta2 - delta3
    comun = (delta2 + delta3)/2
    magnitud = abs(delta2) + abs(delta3)
    dw = math.degrees(f["command_yaw_rate_rad_s"]["last_difference"])
    rho = dw/(1250*diferencial) if diferencial != 0 else None
    plus = raw["runs"]["causal_cuda/plus"]["metrics"]
    minus = raw["runs"]["causal_cuda/minus"]["metrics"]
    firmado = plus["signed_command_deg"] - minus["signed_command_deg"]
    l1 = cierre["native_pair"]["command_L1_deg"]
    return {
        "estado": "ARITMETICA_DE_RESUMENES_VERIFICADOS",
        "dictamen_original": cierre["scientific_classification"],
        "endpoint_400ms": {
            "columnas_lector": [2, 3],
            "delta_q_usada": [delta2, delta3],
            "componente_comun": comun,
            "diferencia_que_lee_decoder": diferencial,
            "indice_cancelacion": None if magnitud == 0 else
                                  1-abs(diferencial)/magnitud,
            "delta_mando_deg_s": dw,
            "secante_tanh_normalizada": rho,
            "alcance": "Baseline común según recibo; no se relee aquí su array."
        },
        "ventana_tardia": {
            "media_tres_canales": f["sensores_usados"]["mean_abs_300_400ms"],
            "media_dos_canales_condicionada_a_tercero_nulo":
                1.5*f["sensores_usados"]["mean_abs_300_400ms"],
            "media_campo": f["concentracion_campo"]["mean_abs_300_400ms"]
        },
        "integrales_comando": {
            "diferencia_firmada_deg": firmado,
            "L1_diferencia_deg": l1,
            "L1_menos_abs_firmada_deg": l1-abs(firmado),
            "nota": "Residuo de redondeo del resumen, no cota por intervalos."
        },
        "geometria": geometria(d["contract/GEOMETRY.json"]),
        "input_sha256": hashes,
        "codigo_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "NPZ_grandes_leidos": False, "CUDA_ejecutado": False,
        "organismo_ejecutado": False, "navegacion_admitida": False,
        "limites": [
            "No asigna IDs anatómicos ausentes a las columnas del lector.",
            "No estima causalidad ni ganancias entre capas de unidades diferentes.",
            "La secante corresponde sólo al último intervalo de la pareja.",
            "No reconstruye error entre perfiles: no se ejecutaron referencias."
        ]
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    try:
        resultado = analizar(args.root)
        codigo = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        resultado = {"estado": "BLOQUEADO", "error": str(error)}
        codigo = 2
    print(json.dumps(resultado, indent=2, ensure_ascii=False, allow_nan=False))
    sys.exit(codigo)
```

```bash
python -B diagnostico_modal.py --root /ruta/extraida
```

**Valores esperados por aritmética de los textos, no salida de una ejecución del script:**

```json
{
  "componente_comun": 3.8873219777968426e-05,
  "diferencia_que_lee_decoder": 7.744789161656662e-06,
  "indice_cancelacion": 0.9003840020727322,
  "delta_mando_deg_s": 0.009663515679428438,
  "secante_tanh_normalizada": 0.9981953520202838,
  "media_dos_canales": 0.04322963985351485,
  "dictamen_original": "DESCARTADO_EN_ESTE_CONTRATO"
}
```

## 4. Dependencias faltantes y alcance

Para generar las cintas H1/H3 faltan aquí los dos `traces.npz` completos, cuyos hashes figuran en `LAYER_DIAGNOSTIC_01.json`, sus ejes temporales, preparación y mapa de entradas. Para H2 falta el código del **consumidor periférico PN→ORN y su mapa de 31 contactos**, además de los valores efectivos que recibe. No usaría como sustituto `PN_q_legacy` ni el net PN total.

**Leí completos:** REQUEST, README, CLOSE, LAYER_DIAGNOSTIC, RAW_VERIFIED, `compare_layers.py`, PLAN, GEOMETRY, MANIFEST y ARCHIVE. De `gaussiano_400.py` leí geometría, instalación y auditor de intervalos; la respuesta quedó truncada en el selftest.

**No descargué/verifiqué el ZIP, no ejecuté el código nuevo, CUDA, MuJoCo ni los NPZ grandes.** La calculadora permitió verificar los resultados aritméticos mostrados; los intentos de ejecución CPU devolvieron `ClientError`.

**Decisión:** conservar B como negativo de intensidad/historia en 400 ms. Antes de pagar un ensayo de navegación más largo, probar H1 con excitaciones comunes y diferenciales de magnitud comparable. El objetivo es identificar qué entrada puede producir una respuesta útil, no convertir la pareja fallida en PASS.
