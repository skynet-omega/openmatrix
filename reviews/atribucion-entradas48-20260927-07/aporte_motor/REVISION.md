# Checkpoint48 y reconstrucción de entradas — revisión del motor

Inicio: 27-09-2026 12:43:15 UTC. Alcance coordinado: revisar esquema guardado, operador efectivo y correspondencia temporal de DNg100/DNb05. Presupuesto 20 minutos de pared, 60 s CPU, 4 GiB; cero GPU, descargas, integración o nuevos asesores. Sólo se escribe en esta subcarpeta. Matrix Astra realiza la descomposición numérica por separado.

Hipótesis de revisión previas: (A) el checkpoint permite una evaluación en su frontera pero no reproduce el último RHS aceptado; (B) usar release publicado como transmisión filtrada, o el grafo original como operador efectivo, confunde buffers distintos; (C) una incompatibilidad concreta de identidad/época podría impedir reconstruir incluso el estado de frontera. Se decidirán leyendo los campos y sus consumidores, no ejecutando el organismo.

## Resultado

Los cuatro checkpoints permiten una atribución **algebraica de las entradas directas en la frontera final** de los seis destinos solicitados. No permiten identificar esa suma con el último RHS histórico ni demostrar causalidad recurrente. No encontré una contradicción concreta de implementación en este examen acotado. La descomposición y la decisión científica pertenecen al análisis principal de Matrix Astra; no se repitieron aquí.

La advertencia inicial sobre APL era pertinente para una reconstrucción general del cerebro, pero **no afecta directamente estas seis sumas**: ninguno de los seis destinos recibe aristas de APL en las rutas canónicas guardadas. Tampoco aparece entre los destinos de las sustituciones regionales KC, salida KCgamma, PN→KC, visuales, ORN o CXHP8 examinadas. Esto no excluye una influencia indirecta de APL sobre sus presinápticas.

| ID canónico | Tipo solicitado | Fila CSR comprobada | Cap heredado |
|---|---|---:|---:|
|10045|DNg100 L|36|201.9180450439453|
|10056|DNg100 R|46|190.84814453125|
|10118|DNb05 L|104|194.21115112304688|
|10065|DNb05 R|52|200.49769592285156|
|523769|DNa02 L|131957|199.53604125976562|
|10360|DNa02 R|332|200.26658630371094|

Estas identidades se comprobaron contra `brain/state.npz` del cargador congelado; no se infirieron de números consecutivos ni del orden de una tabla externa. Los seis son no visuales. La salida visual permanece conectada y la salida PN general permanece deshabilitada en los cuatro brazos. DNa02 se incluye como observador solicitado, sin sustituir el lector actual.

## Estado y operador que corresponden

Todos los propietarios inspeccionados terminan en **47486000000 ns**. Hay 166700 neuronas, 5529 entradas ópticas y 359373 coordenadas híbridas. La transmisión sináptica genérica es exactamente `state[177758:344458]`: `transmission_start = n + 2*len(photo_ids)`. Tomar toda la cola incluye coordenadas de otros modelos; usar `state[:n]` toma salida neuronal instantánea, no su transmisión filtrada.

Para estos seis destinos y en esta frontera, la suma descriptiva es `sum(W_eff[i,j] * s[j] * caps[j])`. Con la salida visual conectada no se excluyen presinápticas visuales. El término externo `drive[i]`, el umbral y la no linealidad son operaciones separadas: la suma recurrente sola no certifica el objetivo final. Los campos `pending_*` guardados no se deben suponer iguales al último input consumido.

Procedencia exacta de los operandos:

* **CSR/IDs/caps:** `/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain/weights_post_pre.npz` aporta `indptr/indices`; su `state.npz` aporta `node_ids/r_max`. Son archivos fijados en `SOURCES.json`. `motor_runtime.py:23` carga el checkpoint del plan; `anatomical_rate_brain.py:301` recupera arrays y `:306` recupera la matriz. No usar `counts_pre_post.npz` sin la transformación correspondiente: tiene orientación distinta.
* **Caps consumidos:** `hybrid_visual_brain.py:69` toma `brain.r_max`; `fp32_operator.py:36` prepara su copia FP32. `restore_prepared.py:35` conserva el cerebro estático del cargador original. `effective_operator` no contiene caps y no debe inventárselos desde tasas publicadas o un valor constante 200.
* **Pesos y parámetros efectivos:** `final_state/effective_operator.json/.npz`, con bindings explícitos a `h.weights64` y `h.cuda.weights`, además de tau/theta/gain. `checkpoint48.py:23` los guarda después del avance. `stored_operator` y `static_inputs/intervenciones_W.npz` cumplen otra función; no reemplazan el operador efectivo. `static_inputs` tampoco contiene el grafo completo.
* **Salida publicada:** `hybrid_visual_brain.py:73–81` transforma el estado visual y publica `release*caps` en FP32. No es el buffer `s*caps` consumido por la suma recurrente.

El verificador fija las diez fuentes estáticas pertinentes contra el lock de campaña. No certifica por sí solo el contenido completo de los grandes checkpoints; sus hashes y la extracción numérica pertenecen al análisis principal.

## Por qué no es el último RHS

`block_midpoint.py:108–118` produce y conserva un estado intermedio temporal, luego restaura al padre. Durante la evaluación aceptada `:69–76` usa parte de ese estado y pesos APL temporales; `:127–135` exige que los propietarios físicos reemplacen las interfaces antes del final. El cierre temporal `mid` no está serializado como un registro completo de la última evaluación.

Además, `graph_runtime.py:125–130` evalúa la derivada de extremo por el límite izquierdo de un evento, mientras el estado comprometido usa el derecho. El acoplamiento por eventos mantiene una forma de onda temporal que se libera al terminar el intervalo (`event_coupling.py:67–89`). La ausencia de conexiones APL directas no demuestra que todas las transmisiones presinápticas del último RHS coincidan con las del checkpoint.

El observador de48 guarda operandos agregados, objetivo y derivada de **dos DNg100**, con las épocas y su condición de aceptación. No guarda el estado completo de cada presináptica de los seis destinos en cada RHS. Para reconstruir exactamente esa evaluación serían necesarios sus buffers de transmisión realmente consumidos, operador temporal, input externo, reloj/estado de eventos y estados intermedios. No se solicitan nuevas simulaciones para suplirlos: el informe debe limitarse a lo identificable con lo existente.

`coefficient_observed.cu:14–31` suma términos FP32 por 32 carriles y después reduce; `:39` aplica la ley rectificada. Una suma CPU FP64 es válida como atribución matemática en la frontera, pero no es una reproducción bit a bit de esa aritmética. Las sumas auxiliares positiva y negativa tienen otro orden de acumulación; una pequeña diferencia de redondeo al recombinarlas no basta para alegar un bug. Tampoco hace falta perseguir diferencias irrelevantes para decidir este negativo.

## Verificador y falsador acotados

Ejecutado `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B verificar_esquema.py`: **4.527 s CPU, 5.188 s de pared y 302821376 bytes RSS**. Cero GPU, pasos neuronales, pasos corporales o descargas. Estos tiempos son los del verificador, no una medición agregada de todas las lecturas de código de la revisión.

`VERIFICACION.json` conserva identidad/filas, caps, reloj, slice, encabezados y membership de cada brazo. Sólo se leen pequeños arrays de selección e identidad; no se materializan W ni el estado neuronal completo, ni se calcula la descomposición asignada a Matrix Astra. Un ID inválido se rechaza en vez de seleccionar silenciosamente su vecino.

Falsadores concretos para la atribución de frontera: hash estático incompatible, ID/orden incorrecto, relojes distintos, slice incompleto, un destino en una sustitución especializada o una arista APL entrante invalidan el camino genérico y obligan a tratar esa fila por su consumidor real. Ninguno apareció aquí. Una diferencia con el último RHS **no** es ese falsador mientras no se demuestre igualdad de todos sus operandos temporales.

No procede usar un Jacobiano infinitesimal de la salida DNg100 rectificada para descartar una desinhibición finita: estando el margen bajo cero, la derivada local puede ser cero aunque una modificación finita cruce el umbral. Tampoco procede retirar fuentes inhibidoras por ocupar los primeros lugares de una suma. Aporte directo actual, sensibilidad local y efecto causal recurrente son preguntas distintas.

Se entrega esta revisión sin modificar el motor, los parámetros, las fuentes históricas ni los criterios de las etapas 4/5.
