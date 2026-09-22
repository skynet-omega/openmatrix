**Sí: las fuentes examinadas permiten integrar cada célula espacial con su propio reloj dentro de una llamada con entradas fijadas. El callback axonal auditado no introduce realimentación entre células durante ese intervalo.** La salvedad principal es otra: **cambiar los relojes locales cambia los instantes en que el detector busca máximos; conservar ecuaciones y tolerancias de voltaje/compuertas no garantiza conservar los eventos.**

**Revisión documental, sin ejecución ni lectura de fuentes nuevas.** Mantendría A dentro de la ronda propuesta, con las condiciones siguientes.

## 1. Independencia real: entre células, no entre sus 17 coordenadas

En `kc_fused_warp.py`, cada bloque utiliza exclusivamente el voltaje, las compuertas, `ge/gi/current` de su célula y matrices geométricas compartidas de solo lectura. Los intercambios `__shfl_sync` conectan las **17 coordenadas de esa misma célula**. No hay lecturas del estado de otra célula. Esa estructura admite relojes distintos entre células, pero no eliminar el acoplamiento eléctrico interno. 

El acoplamiento entre células que introduce actualmente el controlador es **numérico**: `kc_adaptive.py` calcula un máximo sobre todo el lote, adopta un único paso y confirma ambas medias etapas para todas las células. Sustituirlo por máximos y decisiones por célula elimina esa dependencia del lote, conservando el mismo estimador local —incluido su factor para mitades de duración desigual—. **No reproduce necesariamente la trayectoria discreta del controlador global**, aunque conserve las ecuaciones. 

La independencia es válida **hasta la siguiente frontera declarada de intercambio**, no durante cualquier ventana elegida por el ejecutor. En particular, el esquema publicado también ejecuta un predictor de media ventana y luego lo descarta. El reloj local debe terminar en la duración recibida —por ejemplo, 62,5 µs en ese predictor—, no avanzar siempre 125 µs. 

## 2. El callback axonal es local, pero su contabilidad temporal no lo es todavía

El `Publisher` publicado captura `gain` al construirse. En cada llamada, cada puerto axonal lee un voltaje de su célula y actualiza únicamente sus propios filtros, detector y contadores. **No devuelve una corriente a las membranas durante el subpaso.** Las combinaciones de los doce puertos mediante `summaries()` también se realizan por célula, no mezclando células.  

Por tanto, una vez fijados `ge/gi/current`, deben permanecer fijados también **la supresión `gain` y los demás parámetros consumidos por el publicador**. No recalcular `gain` a partir de un CNS parcialmente actualizado porque algunas células hayan terminado antes. En la fase aceptada del acoplamiento publicado, esa ganancia puede proceder del estado de punto medio previsto. 

**La adaptación necesaria está en los relojes:** `Publisher.elapsed` y `batch.elapsed_ns` son escalares; el registrador heredado obtiene la marca del evento desde ese reloj común. No pueden incrementarse una vez por cada célula terminada. Deben existir tiempos locales privados, y el reloj publicado avanzar una sola vez cuando todas hayan alcanzado la frontera.  

La admisión debe ser para **este publicador auditado**, no para cualquier función instalada arbitrariamente en `_motor_axonal_callback`.

## 3. Falsador adicional: pico perdido al liberar a una célula del reloj del lote

El detector publicado compara signos de diferencias de voltaje, exige que el valor anterior supere −40 mV y tenga prominencia de al menos 20 mV. Confirma el evento en el extremo derecho. **Es un detector sobre muestras, no un localizador continuo del máximo.** 

Propongo este falsador del publicador con **voltajes prescritos**, no una trayectoria que afirme haber obtenido de las membranas:

| Tiempo desde el inicio | Voltaje |
|---:|---:|
| 0 µs | −65 mV |
| 6,25 µs | −30 mV |
| 12,5 µs | −65 mV |
| 25 µs | −65 mV |

Con observaciones en **6,25 y 12,5 µs**, se detecta el pico y se emite un evento a 12,5 µs. Con observaciones solo en **12,5 y 25 µs**, no se detecta ninguno. Ambas secuencias pueden terminar con exactamente el mismo voltaje.

**Consecuencia:** confirmar eventos en cada media etapa local conserva la regla, pero puede cambiar el tren de eventos respecto del reloj global. Los límites `2e-5 mV/2e-7` no controlan directamente esta discontinuidad. No aceptaría justificarla solamente mediante error pequeño del estado final.

Lo complementaría con una prueba de independencia: ejecutar una célula testigo sola, acompañada de una célula difícil y con el lote permutado. **Con las mismas entradas y estado propios, el nuevo controlador local debe producir para la testigo la misma secuencia local y los mismos eventos**, independientemente del trabajo de sus vecinas. Eso no exige identidad con el controlador global anterior, cuya dependencia del lote precisamente se está retirando.

## 4. Condición de confirmación

Las dos medias etapas se confirman privadamente **solo después de aceptar la propuesta completa**. No pueden sobrevivir de un intento rechazado `q/s`, pendiente previa, valle, conteos, clipping ni eventos encolados. El orden fisiológico de cada fuente se determina por su reloj, nunca por el orden de terminación de los bloques CUDA. El salto recién confirmado no debe entrar retroactivamente en el filtro axonal del intervalo ya transcurrido.  

**La barrera debe comprobar finalización y éxito de todas las células**, no solo terminación del kernel. Si una falla, las demás pueden haber avanzado sus buffers privados, pero no se publica un estado mixto al CNS.

**Dictamen:** A tiene fundamento estructural en estas fuentes y merece el contraste. Su riesgo principal es la semántica temporal de eventos y confirmación, no una dependencia eléctrica oculta entre las células examinadas.

Releí `kc_adaptive.py`, `kc_fused_warp.py`, `physical_events.cu`, `kc_axonal_brain.py` y `kc_axonal_release.py` en `8ff835…`; y `axon_gpu.py`, `block_midpoint.py` y `event_coupling.py` en `618736…`. **No ejecuté los falsadores ni doy por comprobada su futura implementación.**
