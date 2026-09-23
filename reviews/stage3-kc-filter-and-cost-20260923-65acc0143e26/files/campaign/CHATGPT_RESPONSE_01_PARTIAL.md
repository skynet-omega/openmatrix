**No cambiaría aún el detector ni el integrador. La siguiente decisión puede obtenerse de las dos épocas ya previstas: comparar primero restauración e inputs; después, propuestas con el mismo paso; finalmente, el detector sobre las muestras efectivamente aceptadas.**

**Esta vez no pude ejecutar código:** los intentos de Python y contenedor devolvieron `ClientError`; tampoco se completó la descarga del ZIP. Sí leí las fuentes públicas. Entrego un comparador completo para las capturas que preparará Codex, **sin atribuirle resultados todavía**.

## 1. Qué aportan las fuentes nuevas

### A. Muestreo y detector

`previous_slope` conserva \(V_k-V_{k-1}\), no una derivada. Compararlo entre pasos diferentes mezcla **trayectoria e intervalo de observación**. Dividirlo por \(\Delta t\) ayuda al diagnóstico, pero tampoco produce cantidades idénticas si corresponden a intervalos distintos.

`trough` es memoria del detector: se reinicia al detectar un máximo muestreado, incluso sin espiga. No es directamente “el mínimo físico desde la última espiga”. **Sigue siendo causalmente relevante**, porque entra en la siguiente prueba de prominencia. No debe borrarse del gate ni reemplazarse por el valor del otro motor. 

**Importante:** el negativo no se reduce a campos dependientes de la malla. A 1 ms, `somatic_delta` ya difiere **0,00104239 mV**; a 2 ms, **0,00274922 mV**. Son estados eléctricos comparables en el mismo instante. Ambas capturas son posteriores a la primera discrepancia de eventos, por lo que todavía no separan causa y consecuencia. 

### B. Integración y control

El WARP conserva la masa completa \(C\). Su paso hace:

\[
(K_0+2C/h)V_*=(2C/h)V_0+b_0,
\]

recalcula compuertas y operadores en \(V_*\), y después:

\[
(K_*+2C/h)V_1=(2C/h-K_*)V_0+2b_*.
\]

Las dos resoluciones tienen control de residuo. El estimador compara completo frente a dos medios pasos con **\(V_{\mathrm{atol}}=2\times10^{-5}\)** y **\(G_{\mathrm{atol}}=2\times10^{-7}\)**. Eso no certifica por sí solo la memoria del detector.  

No encontré una discrepancia algebraica evidente entre la política escalar C++ y la política por bloque previamente publicada: ambas aceptan con \(e\le1\), duplican si \(e<0,1\) y dividen tras rechazo. La diferencia sustancial es **qué conjunto de células determina el paso**. El replay decisivo debe usar iguales inputs y \(h\), no comparar directamente dos historias adaptativas distintas. 

### C. Restauración

`restore_parent` restaura explícitamente estado espacial y axonal. No encontré una omisión demostrada en esa función. Sin embargo, la comparación exhaustiva `predictor_restore_exact` sigue ejecutándose explícitamente solo la primera vez. Además, un snapshot correcto no demuestra que el siguiente consumidor use el mismo contenido tras reenlazar buffers.   

**Discriminador de C:** comparar antes/después de restaurar tanto el estado publicado como **los valores efectivamente entregados al siguiente ensayo**, excluyendo únicamente scratch descartable y estadísticas diagnósticas.

## 2. Decisión con una sola captura acotada

Mantendría la selección y presupuesto de vuestro plan: dos épocas y las cuatro células **76431, 81004, 544736 y 45199**. El mapa confirma sus índices locales; no confundiría el máximo de trough de KC544736 con la primera emisora divergente KC76431.  

| Resultado | Decisión |
|---|---|
| Estado/restauración o inputs del siguiente consumidor difieren antes de integrar | **C:** localizar y reparar esa transferencia. No modificar el detector. |
| Mismos inputs y mismo \(h\), pero cambian completo/medio/fino | **B:** investigar cálculo/compilación/control local. El comparador identifica la primera salida diferente. |
| Propuestas emparejadas coinciden, pero cambian los pasos aceptados y la memoria del detector | **A:** cuantificar el efecto temporal sobre los puertos antes de proponer otro detector. |

Si las capturas no contienen ningún ensayo con inputs y \(h\) iguales, basta **un replay emparejado de una propuesta capturada**, dentro de los casos portátiles presupuestados. No hace falta otra campaña larga.

Una futura localización continua de máximos sería **otro algoritmo**, no una corrección cosmética. Habría que definir cómo conserva umbral, prominencia y estado previo. La búsqueda de raíces de \(\dot V\) tampoco garantiza detectar múltiples extremos ocultos dentro de un paso; los métodos basados en cambios de signo tienen límites conocidos. :chatgpt-content-reference{index="10"}

## 3. Qué puede acotarse sin excluir KC

Entre eventos, para dos historias con las mismas constantes:

\[
\Delta q'=-\Delta q/\tau_q,\qquad
\Delta s'=(g\Delta q-\Delta s)/\tau_s.
\]

En un intervalo de longitud \(d\), sin eventos interiores:

\[
|\Delta q(t)|\le D_q,
\]

\[
|\Delta s(t)|\le
D_s+|g|D_q(1-e^{-d/\tau_s}),
\]

donde \(D_q,D_s\) son las diferencias al inicio. Separando por la **unión de las marcas de ambas historias**, y aplicando sus SET reales, obtenemos una envolvente para todo el intervalo, no solo endpoints. El código implementa esa envolvente y una cota de área de error.

**Alcance exacto:** cota algebraica de dos historias prescritas, con constantes iguales y en aritmética exacta. La evaluación FP64 no usa redondeo dirigido; no es un certificado por intervalos ni una cota de la realimentación del organismo.

Un desfase diminuto de una espiga puede causar un error instantáneo en \(q\) del tamaño del salto. **No se oculta alineando curvas.** El efecto sobre \(s\) puede ser pequeño porque filtra ese intervalo; ambas magnitudes se informan.

Para llegar a yaw falta la propagación recurrente. La desigualdad del lector es:

\[
|\delta\omega|_{\mathrm{grados/s}}
\le1250\bigl(|\delta q_L|+|\delta q_R|\bigr)
\]

con baseline idéntico. Integrarla acota **comando acumulado**, no directamente yaw. Este último necesita una cota de la planta o un replay corporal emparejado de ambos comandos, manteniendo su incertidumbre numérica explícita. No extrapolaría el pequeño error DNb05 de 20 ms a 400 ms.

---

## 4. Comparador completo: `localizar_kc.py`

No cambia el motor. Tiene dos usos:

1. Leer inmediatamente `BOUNDARY.json` y `KC_LOCAL_MAPPING.json`.
2. Comparar un **puerto de una célula y una época** de las capturas nuevas.

### Contrato de la captura nueva

Cada JSON contiene:

- `schema: "kc_causal_probe_v1"`.
- `meta`: `neuron_id`, `local_index`, `global_row`, `port`, `origin_ns`, `duration_ns`, `clock_domain`, `operator_sha256`, `role: "accepted_full_epoch"`.
- `parameters`: `tau_q_s`, `tau_s_s`, `gain`, `cap`, `threshold_mV`, `prominence_mV`.
- `initial`: `last`, `slope`, `trough`, `q`, `s`, `count`, `clipped`.
- `samples`: cada media etapa **aceptada**, con `t_ns` relativo a la época, `voltage_mV` realmente consumido, `delta`, `gates`, `memory` con los campos anteriores, y los booleanos `peak`, `event`. Para soma puede omitirse `memory.s`; el filtro se reconstruye aparte.
- `trials`: `start_ns`, `h_ns`, `inputs_sha256`, `accepted`, y arrays `full_v/full_g`, `middle_v/middle_g`, `fine_v/fine_g`.
- `rollback`: lista de recibos con `before` y `after`, cada uno con hashes de `parent`, `pn`, `published`, `next_consumer`.

`inputs_sha256` debe incluir **valores y layout** de estado inicial, geometría/masa, conductancias, corriente y parámetros usados; no punteros. Los hashes de rollback tienen que proceder de copias propias, no de dos referencias al mismo buffer. Su exhaustividad sigue siendo parte del contrato de captura.

```python
"""Comparador externo KC. No integra membranas ni modifica el organismo.
Captura: kc_causal_probe_v1; una célula/puerto y una época confirmada.
La envolvente q/s es algebraica para historias prescritas y constantes iguales.
No certifica redondeo, eventos ausentes, recurrencia ni yaw.
"""
import argparse, hashlib, json, math, sys, traceback
from pathlib import Path
import numpy as np
from scipy.linalg import expm

PLAN = {
    "historical_abs_limit": 1e-4,
    "V_ATOL": 2e-5,
    "G_ATOL": 2e-7,
    "comparison": "tiempos idénticos; sin interpolación ni ajuste de lag",
    "stage3_admission": False
}
MEM = ("last", "slope", "trough", "q", "s")
ROLL = ("parent", "pn", "published", "next_consumer")

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def read(p):
    require(p.stat().st_size <= 64*1024**2, "JSON mayor de 64 MiB")
    return json.loads(p.read_text(encoding="utf-8"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def digest_ok(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)

def array(x):
    a = np.asarray(x, dtype=np.float64)
    require(a.size > 0 and np.isfinite(a).all(), "Array vacío/no finito")
    return a

def difference(a, b):
    x, y = array(a), array(b)
    require(x.shape == y.shape, "Layouts distintos")
    return float(np.max(np.abs(x-y)))

def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                           allow_nan=False)+"\n", encoding="utf-8")

def flow(q, s, dt, p):
    tq, ts, gain = p["tau_q_s"], p["tau_s_s"], p["gain"]
    A = np.array([[-1/tq, 0.], [gain/ts, -1/ts]])
    return tuple(expm(A*dt) @ np.array([q, s]))

def load_capture(path):
    d = read(path)
    require(d["schema"] == "kc_causal_probe_v1", "Esquema desconocido")
    m, p = d["meta"], d["parameters"]
    require(m["role"] == "accepted_full_epoch", "No mezclar predictor con confirmado")
    for k in ("neuron_id", "local_index", "global_row", "origin_ns", "duration_ns"):
        require(type(m[k]) is int and m[k] >= 0, "Identidad/reloj no entero")
    require(m["duration_ns"] > 0 and digest_ok(m["operator_sha256"]),
            "Duración/operador inválidos")
    require(isinstance(m["clock_domain"], str) and m["clock_domain"], "Dominio de reloj")
    for k in ("tau_q_s", "tau_s_s", "gain", "cap", "threshold_mV", "prominence_mV"):
        require(np.isfinite(p[k]), "Parámetro no finito")
    require(min(p["tau_q_s"], p["tau_s_s"], p["cap"]) > 0, "Tau/cap inválidos")
    initial = d["initial"]
    for k in MEM:
        require(np.isfinite(initial[k]), "Estado inicial no finito")
    for k in ("count", "clipped"):
        require(type(initial[k]) is int and initial[k] >= 0, "Contador inicial")
    previous = 0
    for s in d["samples"]:
        t = s["t_ns"]
        require(type(t) is int and previous < t <= m["duration_ns"], "Orden de muestras")
        previous = t
        require(type(s["event"]) is bool and type(s["peak"]) is bool, "Flags no booleanos")
        require(np.isfinite(s["voltage_mV"]), "Voltaje no finito")
        array(s["delta"]); array(s["gates"])
        require(set(s["memory"]) in (set(MEM)|{"count","clipped"},
                                    (set(MEM)-{"s"})|{"count","clipped"}),
                "Campos de memoria incompletos")
        for k, value in s["memory"].items():
            require(np.isfinite(value), "Memoria no finita")
        for k in ("count", "clipped"):
            require(type(s["memory"][k]) is int, "Contador no entero")
    require(previous == m["duration_ns"], "Época incompleta")
    return d

def replay_detector(d):
    """Réplica de la ley publicada; min(...,1) pertenece a esa ley KC."""
    p = d["parameters"]; state = dict(d["initial"]); t0 = 0
    errors = {k: 0. for k in MEM}; discrete = []
    for sample in d["samples"]:
        dt = (sample["t_ns"]-t0)*1e-9
        voltage = sample["voltage_mV"]
        slope = voltage-state["last"]
        peak = state["slope"] > 0 and slope <= 0
        event = (peak and state["last"] > p["threshold_mV"]
                 and state["last"]-state["trough"] >= p["prominence_mV"])
        before, snew = flow(state["q"], state["s"], dt, p)
        after = before + (1/(p["cap"]*p["tau_q_s"]) if event else 0.)
        state = {
            "last": voltage, "slope": slope,
            "trough": voltage if peak else min(state["trough"], voltage),
            "q": min(after, 1.), "s": snew,
            "count": state["count"]+int(event),
            "clipped": state["clipped"]+int(after > 1.)
        }
        actual = sample["memory"]
        for k in MEM:
            if k in actual:
                errors[k] = max(errors[k], abs(state[k]-actual[k]))
        if (peak != sample["peak"] or event != sample["event"]
                or any(state[k] != actual[k] for k in ("count","clipped"))):
            discrete.append(sample["t_ns"])
        t0 = sample["t_ns"]
    return {"max_memory_error": errors,
            "discrete_mismatch_times_ns": discrete,
            "scope": "Mismo voltaje consumido; replay CPU, no paridad CUDA garantizada"}

def events(d):
    # Cada post procede de la captura física, no del replay anterior.
    return [(s["t_ns"], s["memory"]["q"])
            for s in d["samples"] if s["event"]]

def port_envelope(a, b):
    p = a["parameters"]
    if any(p[k] != b["parameters"][k] for k in ("tau_q_s","tau_s_s","gain")):
        return {"status": "PENDIENTE: constantes de flujo distintas"}
    ea, eb = dict(events(a)), dict(events(b))
    stop = a["meta"]["duration_ns"]
    marks = sorted({0, stop} | set(ea) | set(eb))
    xa = (a["initial"]["q"], a["initial"]["s"])
    xb = (b["initial"]["q"], b["initial"]["s"])
    supq = sups = areaq = areas = 0.
    for j, mark in enumerate(marks):
        if mark in ea: xa = (ea[mark], xa[1])
        if mark in eb: xb = (eb[mark], xb[1])
        dq, ds = abs(xa[0]-xb[0]), abs(xa[1]-xb[1])
        supq = max(supq, dq); sups = max(sups, ds)
        if j == len(marks)-1: break
        dt = (marks[j+1]-mark)*1e-9
        rq = -math.expm1(-dt/p["tau_q_s"])
        rs = -math.expm1(-dt/p["tau_s_s"])
        envelope_s = ds+abs(p["gain"])*dq*rs
        sups = max(sups, envelope_s)
        areaq += dq*p["tau_q_s"]*rq
        areas += ds*p["tau_s_s"]*rs + abs(p["gain"])*dq*dt*rs
        xa = flow(*xa, dt, p); xb = flow(*xb, dt, p)
    return {"status": "ENVOLVENTE_ALGEBRAICA_CALCULADA",
            "sup_abs_q": supq, "upper_sup_abs_s": sups,
            "upper_integral_abs_q_s": areaq,
            "upper_integral_abs_s_s": areas,
            "scope": "Historias SET capturadas; sin cota FP64 ni de realimentación"}

def trial_index(d):
    indexed = {}
    for t in d.get("trials", []):
        h = t["h_ns"]
        require(type(h) is int and h >= 2 and type(t["start_ns"]) is int,
                "Reloj de propuesta inválido")
        require(digest_ok(t["inputs_sha256"]) and type(t["accepted"]) is bool,
                "Contrato de propuesta")
        for field in ("full_v","full_g","middle_v","middle_g","fine_v","fine_g"):
            array(t[field])
        x, y = (h//2)/h, (h-h//2)/h
        cubes = x**3+y**3; factor = cubes/(1-cubes)
        estimate = max(difference(t["fine_v"],t["full_v"])/PLAN["V_ATOL"],
                       difference(t["fine_g"],t["full_g"])/PLAN["G_ATOL"])*factor
        key = (t["start_ns"], h, t["inputs_sha256"])
        require(key not in indexed, "Propuesta duplicada: separar época/replay")
        indexed[key] = (t, estimate)
    return indexed

def rollback(d):
    records = d.get("rollback", [])
    if not records: return {"status": "NO_CAPTURADO"}
    changed = []
    for i, r in enumerate(records):
        require(set(r["before"]) == set(r["after"]) == set(ROLL),
                "Cobertura de rollback incompleta")
        for k in ROLL:
            require(digest_ok(r["before"][k]) and digest_ok(r["after"][k]),
                    "Hash de rollback inválido")
            if r["before"][k] != r["after"][k]:
                changed.append({"record": i, "field": k})
    return {"status": "COMPARADO", "differences": changed,
            "scope": "Exactitud de los valores representados por esos hashes"}

def compare(a, b):
    for k in ("neuron_id","local_index","global_row","port","origin_ns",
              "duration_ns","clock_domain"):
        require(a["meta"][k] == b["meta"][k], "Identidad/tiempo distinto: "+k)
    sa = {s["t_ns"]: s for s in a["samples"]}
    sb = {s["t_ns"]: s for s in b["samples"]}
    common = sorted(set(sa)&set(sb)); rows = []
    for t in common:
        x, y = sa[t], sb[t]
        r = {"t_ns": t, "voltage_mV": abs(x["voltage_mV"]-y["voltage_mV"]),
             "delta": difference(x["delta"],y["delta"]),
             "gates": difference(x["gates"],y["gates"])}
        for k in set(x["memory"]) & set(y["memory"]):
            r["memory_"+k] = abs(x["memory"][k]-y["memory"][k])
        rows.append(r)
    ia, ib = trial_index(a), trial_index(b); paired = []
    same_operator = a["meta"]["operator_sha256"] == b["meta"]["operator_sha256"]
    if same_operator:
        for k in sorted(set(ia)&set(ib)):
            x, ex = ia[k]; y, ey = ib[k]
            r = {"start_ns": k[0], "h_ns": k[1],
                 "local_error_A": ex, "local_error_B": ey,
                 "accepted_A": x["accepted"], "accepted_B": y["accepted"]}
            for f in ("full_v","full_g","middle_v","middle_g","fine_v","fine_g"):
                r[f] = difference(x[f], y[f])
            paired.append(r)
    ea, eb = events(a), events(b)
    event_pairs = [] if len(ea) != len(eb) else [
        {"ordinal": j, "time_delta_ns": x[0]-y[0], "post_delta": x[1]-y[1]}
        for j,(x,y) in enumerate(zip(ea,eb))]
    return {
        "meta": a["meta"], "same_operator": same_operator,
        "same_initial": a["initial"] == b["initial"],
        "same_parameters": a["parameters"] == b["parameters"],
        "common_times_only": rows, "matched_input_step_trials": paired,
        "trial_pairing": "NO_IDENTIFICABLE" if not paired else "MISMO_INPUT_Y_H",
        "event_counts": [len(ea),len(eb)], "event_pairs": event_pairs,
        "detector_replay_A": replay_detector(a),
        "detector_replay_B": replay_detector(b),
        "rollback_A": rollback(a), "rollback_B": rollback(b),
        "ports": port_envelope(a,b),
        "scope": "Primera diferencia sólo en muestras disponibles; sin interpolación",
        "stage3_admission": False
    }

def boundary(path, mapping):
    d = read(path); m = read(mapping)["selected_local_to_global"]; out = []
    for ms, fields in d["state_max_abs_by_ms"].items():
        for name, r in fields.items():
            local = str(r["index"][0])
            out.append({"ms": int(ms), "field": name, "index": r["index"],
                        "identity": m.get(local), "max_abs": r["max_abs"],
                        "exceeds_old_limit": r["max_abs"] > PLAN["historical_abs_limit"]})
    return {"rows": out,
            "first_event": d["first_recorded_accepted_event_difference"],
            "scope": "Lectura de máximos publicados; no reconstrucción de arrays",
            "first_continuous_divergence": None, "stage3_admission": False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path); p.add_argument("--b", type=Path)
    p.add_argument("--boundary", type=Path); p.add_argument("--mapping", type=Path)
    p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    require(bool(args.boundary) != bool(args.a), "Elegir captura o BOUNDARY")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/"PLAN.json", PLAN)
    try:
        if args.boundary:
            require(args.mapping is not None, "Falta mapa de IDs")
            result = boundary(args.boundary, args.mapping)
            paths = [args.boundary,args.mapping]
        else:
            require(args.b is not None, "Falta segunda captura")
            result = compare(load_capture(args.a), load_capture(args.b))
            paths = [args.a,args.b]
        result["inputs_sha256"] = {str(x): sha(x) for x in paths}
        result["code_sha256"] = sha(Path(__file__))
        save(args.out/"RESULTADO.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        save(args.out/"FALLO.json", {"error": traceback.format_exc()})
        raise

if __name__ == "__main__":
    sys.exit(main())
```

### Comandos

```bash
# Eje
