**No discrepo de completar la única confirmación de 1 s con las fuentes congeladas.** Los números publicados respaldan la mejora de 100 ms. Sí encontré huecos concretos en el comparador: **un `PASS` de 1000 ms no basta, tal como está, para confirmar procedencia inicial, integridad de todas las observaciones y rendimiento.** No he demostrado que esos huecos afecten a tus archivos reales.

### Lo inspeccionado y comprobado

Leí íntegramente los cuatro archivos pegados —`evaluate.py`, `compare_runs.py`, `run_trial.py` y `comparison_math.py`—, el JSON de resultados y las métricas neuronales incluidas. Ejecuté **copias de las funciones de comparación sobre fixtures sintéticos CPU**. No ejecuté el organismo ni CUDA; tampoco dispuse de los NPZ, manifiestos y árboles PN reales para recalcular su contenido.

Tus cifras son coherentes: `4 × 18361 = 73444` RHS y `4 × (11453 + 1) = 45816`. Los intentos bajan **37,62 %** y el tiempo CNS **37,55 %**. El ahorro CNS, **32,70 s**, explica aproximadamente **98,17 %** de los **33,31 s** ahorrados en avance. Esto respalda **menos ensayos RK como explicación principal en esta pareja**, no una aceleración sustancial por RHS.

### Hallazgos materiales

Los contraejemplos siguientes son **sintéticos**, no resultados de tus ejecuciones:

| Punto | Comportamiento comprobado | Corrección acotada |
|---|---|---|
| **Inicialización en la confirmación** | Con `paired_timing=False`, desaparece la exigencia de igualdad del estado inicial registrado neuronal/PN/publicado. Cambié esos estados iniciales y `source_trace_sha256`: siguió dando `PASS`. | Mantener la verificación de preparación también contra 07 de 1 s. `INITIAL.exact` de cada corrida no sustituye la comparación entre sus preparaciones. |
| **PN/publicación finales opcionales** | Si falta uno de los JSON finales, el comparador omite ese árbol. Retiré los finales de la candidata: `PASS`, con únicamente `"0"` en `PN` y `published`. | Exigir las observaciones finales comprometidas y su finitud. **No exigir igualdad PN ni introducir tolerancias nuevas.** |
| **Cobertura funcional y relojes** | Sólo `paso` exige exactamente `ms` filas. Ambos archivos con 99 filas en las demás columnas y 100 en `paso` pasan. También pasan relojes PN idénticos pero detenidos. | Exigir longitud de todas las columnas y verificar el contrato temporal PN/body, no sólo igualdad entre corridas. El runner sí comprueba avance CNS y número de llamadas al cuerpo. |
| **Eventos: información no comparada** | Cambiar `jump` y convertir `post_q` a `None` dejó el informe de eventos idéntico. Cambiar el orden puede excluir pares del cálculo temporal sin informar cuántos quedaron excluidos. | Añadir diferencias de payload, cambios de presencia de `post_q` y cobertura del emparejamiento temporal. Mantenerlos como diagnóstico separado, **no como puertas nuevas**. |
| **Rendimiento/presupuesto de 1 s** | Sin `paired_timing`, no hay puertas temporales ni de presupuesto. Una candidata sintética más lenta que 07 y con **2450 s** de proceso devolvió `PASS`. | Identificar ese resultado como criba funcional; informar mejora real y comprobar el presupuesto congelado separadamente. No trasladar automáticamente el umbral del 10 % a 1 s. |
| **Dos entradas con distinta decisión** | Con forward diferente, `compare_runs.compare()` devolvió `PASS`; `evaluate.evaluate()` devolvió `FAIL`. | Usar siempre `evaluate.py`, o incorporar la misma comprobación forward al comparador base. Tu resultado de 100 ms sí incluye esa puerta. |

:chatgpt-content-reference{index="2"}[Resultados de los contraejemplos CPU](sandbox:/mnt/data/revision11_cpu/synthetic_results.json).

**Sobre eventos:** lo publicado sí permite decir «mismas identidades y multiplicidades por bloque». No permite concluir que también coincidan los payloads ni que el máximo temporal cubra todos los eventos. La etiqueta actual limita correctamente el máximo a órdenes coincidentes; falta cuantificar esa cobertura.

### Observación concreta del runner

`run_trial.py` cancela la alarma al entrar en `finally`, **antes** de guardar los NPZ/EVENTS y cerrar recursos. Por ello, esa parte puede llevar `wall_total_s` por encima del límite sin cambiar un `COMPLETE` previo. Además, una señal manejada por Python no garantiza interrumpir inmediatamente una operación prolongada en C. No he inspeccionado ningún supervisor externo que complemente ese límite. :chatgpt-content-reference{index="0"}

Esto no afecta al presupuesto observado de la pareja, **404,251 s**, pero hace necesaria la comprobación final del límite ya fijado, sin modificar ahora el runner.

### Parche externo acotado

Preparé un :chatgpt-content-reference{index="3"}[parche de integridad para una copia externa de `compare_runs.py`](sandbox:/mnt/data/revision11_cpu/integrity_subset.patch). Cubre cuatro puntos: longitud de columnas, igualdad inicial independientemente del modo temporal, presencia de PN/publicación finales y forward en el comparador base. No cambia umbrales ni código del motor.

Lo probé con fixtures válidos de 100/1000 ms y con los cuatro contraejemplos correspondientes: los válidos pasan y los defectuosos dejan de pasar. :chatgpt-content-reference{index="4"}[Resultados del parche](sandbox:/mnt/data/revision11_cpu/patch_synthetic_results.json). No incluye las métricas adicionales de eventos ni la validación temporal específica PN/body.

**Conclusión:** mantener 11 como candidata y completar la confirmación sigue estando fundamentado. Conservaría el informe original y añadiría la comprobación externa sobre los mismos artefactos. La decisión final debe distinguir **criba funcional superada, mejora respecto de 07 y cumplimiento de los 1500 s de proceso correspondientes a la meta nominal de 25 minutos**; son tres afirmaciones diferentes.

## Revisión del auditor externo

**Veo un único hueco material pendiente: comprobar el instante de los snapshots finales, no sólo su presencia y finitud.** No indica que tus archivos reales estén mal.

### El punto concreto

En `final_trees()`:

```python
rows = saved_tree_errors(path, path)
```

Esto verifica contenido legible, referencias a arrays y finitud, pero **no demuestra que corresponda al final solicitado**. `check_trace()` valida los relojes de las trazas, no los del árbol guardado.

Un snapshot inicial colocado en `state_100ms/published` puede superar esta comprobación. El comparador original tampoco exige coherencia temporal de ese árbol: registra la diferencia de `/time_ns` como diagnóstico, sin convertirla en fallo.

**Lo reproduje con un fixture CPU:** un árbol «final» con reloj inicial superó `final_trees`; la comparación cruzada devolvió `/time_ns: exact=False` sin lanzar error. Es un contraejemplo sintético, no un hallazgo en tus datos. La consecuencia sería una **confirmación de integridad excesiva**, aunque la criba funcional de las trazas fuese legítima.

### Corrección externa mínima

Vincula el reloj serializado del snapshot con el instante final exigido por su contrato. El runner mostrado guarda explícitamente `published.time_ns`; compruébalo contra el tiempo de publicación esperado al cierre. Será `origin + ms*1_000_000` **si ese es efectivamente su contrato**, sin imponerlo por suposición. Aplica lo mismo al reloj propio de PN cuando esté serializado.

Si un árbol no contiene información temporal verificable, limita esa parte del informe a **presencia, estructura y finitud**, sin presentar el autocotejo como comprobación de su instante. Esto no introduce tolerancias fisiológicas ni requiere repetir la simulación.

**Fuera de ese punto, no identifico otro defecto material en la combinación mostrada del auditor y los comparadores anteriores.** La referencia inicial mediante control100 está correctamente delimitada, y el informe de eventos ahora explicita cobertura y diferencias de payload sin confundirlas con identidad exacta.

Inspeccioné el código y `ACTUAL100` pegados, y ejecuté únicamente el contraejemplo descrito. No inspeccioné los archivos reales ni la implementación de `verify_saved`, que no está incluida. **No discrepo de continuar la confirmación congelada.**

Aplicación local: se verifican también los relojes de publicación y de cada dueño PN respecto de su origen inicial propio. No se iguala el reloj PN interno al del CNS porque sus orígenes difieren. El fixture de publicación inicial colocada como final se rechaza; los datos reales de100ms conservan el mismo resultado. No hubo nueva ejecución del organismo ni cambio del motor.
