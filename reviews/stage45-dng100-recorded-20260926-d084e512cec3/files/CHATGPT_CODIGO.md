# Revisión externa del código — 26-09-2026

Conversación autorizada ChatGPT C++/CUDA; revisión de fuentes, sin ejecución remota. Snapshot 86044c7b3b88c97e0c1036fd2eeb7b7c25b193ca. La adquisición sigue congelada. La observación sobre el clasificador se resolverá mediante un suplemento posterior, preservando el analizador original y su hash.

## Dictamen

**No encuentro un fallo material en la adquisición que obligue a detener la rama en curso ni a modificar el scheduler.** La instrumentación distingue correctamente las cuatro etapas, los intentos aceptados y los contextos predictor/comprometido. Sí corregiría **una condición del veredicto automático del analizador**, usando los registros existentes, sin cambiar el ensayo.

Leí `observer.py`, ambos kernels observados, `run.py`, `analyze.py`, `test_observer.py`, `PLAN.json`, `launch.py` y las fuentes originales de CSR, `GraphRK23`, `RealCNS`, `OrganismAdapter`, `block_midpoint` y protocolo45. **No ejecuté CUDA, las pruebas ni el organismo.** La coincidencia de los primeros100 ms procede de tu comunicación; el snapshot contiene el resultado publicado del test de software, no una reproducción realizada aquí. 

## 1. La captura corresponde a las evaluaciones realmente ejecutadas

**CSR:** comparado con `original/fast_fp32_coefficient.cu`, `coefficient_observed.cu` conserva la acumulación original de `a`, su reducción por shuffle y las expresiones de `target/rate`. `positive/negative` se calculan aparte y no realimentan esos resultados. El valor `net` procede de `a`, no de reconstruirlo como suma de auxiliares. Esto preserva la separación que necesitabas.  

**Callback completo:** `ObservedCNS.rhs()` recibe el resultado de `coefficient(y)`, calcula la misma expresión original `rate*(target-y)` y después copia estado, coeficientes finales y derivada a `scratch`. En el adaptador, las sustituciones de puertos ocurren **antes del retorno de `coefficient`**; por tanto, la captura final no se limita al objetivo provisional del CSR. Los índices de los 16 campos coinciden entre kernel, `FIELDS` y analizador.  

**Etapas y contador:** `stage_id` se utiliza durante la construcción del grafo para fijar los cuatro destinos de captura; no pretende contar reproducciones en Python. El ordinal de cada trial lo incrementa `capture_trial` en GPU. El reinicio de `observer.count` en el stream del integrador, antes del avance real, excluye los calentamientos. No veo aquí el error de contador congelado durante captura.  

**k4:** el código original usa `nextafter(clock[2],clock[0])`; el observador registra ese reloj recibido, no reconstruye un extremo derecho. El analizador pondera únicamente k1–k3 con \(h(2/9,1/3,4/9)\), y cuenta los objetivos positivos de k4 separadamente. Esta parte está correctamente implementada.  

## 2. La reconstrucción de aceptación es válida en el alcance publicado

En esta versión, `super().advance()` sólo retorna normalmente después de comprobar el resultado del controlador; una `failure` produce excepción. Después, el observador exige finitud, flags nulos, coincidencia con los contadores C++, continuidad temporal, continuidad de las dos filas y coincidencia con el estado final real.

**Dentro de esas épocas exitosas**, clasificar con `error <= 1` y flags nulos coincide con la decisión inspeccionada. No hace falta añadir otra escritura a `decide` para esta adquisición.   

También está cubierta la ruta de exceso de intentos: la escritura tiene guarda de capacidad y el contador sigue aumentando. Si el controlador falla, no se descarga ni clasifica como época válida; si retornara con un conteo fuera de capacidad, el observador lo rechaza. No hay sobrescritura circular silenciosa. 

### Predictor aceptado no se convierte en trayectoria comprometida

`bind_context()` obtiene el diccionario `active` del cierre real. Sus relojes se contrastan antes de añadir la época. Es correcto recordar que `active['accepted']=True` identifica **la rama comprometida prevista**, todavía susceptible de fallar en la segunda mitad PN.

Tu runner resuelve esa diferencia: sólo publica el bloque después de que los pasos completos hayan terminado y coincidan con45. Un fallo posterior no convierte los registros pendientes en datos científicos publicados. Además, `analyze.py` excluye los directorios `.partial` y exige ambos brazos `COMPLETE`.   

## 3. Corrección concreta del analizador: «sin positivos» no siempre significa «cero»

Actualmente:

```python
zero = all(
    all(v == 0 for v in d['positive_target_evaluations'])
    for a in result.values()
    for d in a['phases'].values()
)
```

Esto demuestra **ausencia de objetivos positivos**, pero también daría `True` si aparecieran objetivos finales negativos. El CSR genérico impide valores negativos mediante `fmaxf`, pero precisamente estás observando el resultado **posterior a posibles sustituciones**. El analizador no debe asumir esa propiedad para descartar una anomalía. 

Antes de emitir `COHERENT_ZERO_TARGET…`, utilizaría los extremos que ya calculas:

```python
zero = all(
    cell["final_target"]["min"] == 0.0
    and cell["final_target"]["max"] == 0.0
    for arm in result.values()
    for phase in arm["phases"].values()
    for cell in phase["cells"]
)
```

Un objetivo negativo debe informarse como anomalía localizada, no como reclutamiento positivo. **No he encontrado evidencia de objetivos negativos en esta vida**: señalo una posibilidad real de clasificación incorrecta, corregible sin tocar adquisición, límites ni datos.

## 4. Qué podrá afirmarse al cerrar

La partición temporal del analizador concuerda con protocolo45: la concentración se publica al terminar el intervalo1000 y empieza a consumirse en1001; por eso `ms<=1000` es basal y `ms>1000` corresponde al estímulo consumido hasta3000. 

Conservaría dos límites de interpretación:

**Las medias ponderadas son resúmenes de evaluaciones RK**, no muestras independientes ni medidas continuas exactas del circuito. La integral de `target` tampoco es directamente el incremento de estado: éste depende de la derivada registrada y del esquema de integración.

**La comparación contra45 cubre sus observables guardados.** El runner comprueba todos los campos de traza, publicaciones y registros de eventos por bloque, pero no dispone de cada estado interno original45. El resultado legítimo será «objetivos realmente consumidos en47, con reproducción de los observables45», no recuperación retrospectiva de operandos45 no guardados. 

**Mantendría47 en ejecución bajo sus guardas actuales.** No veo motivo para otra campaña, otra evaluación del coeficiente ni cambios al controlador. La corrección del test literal de cero pertenece únicamente al análisis posterior; una discrepancia real de observación, cobertura o publicación sí debe detener la interpretación conforme al plan existente.
