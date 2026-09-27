# Continuación49 cualificada en el caso profile: estado y trayectoria exactos

**Resultado actual: PASS del contraste acotado.** Tras reparar únicamente los dos predicados de carga del reloj,4ms continuos desde `profile/final_state48` coinciden exactamente con guardar en2ms, reconstruir en otro proceso y continuar2ms. Se consumieron6ms neuronales en total. Ambos comienzos restauraron exactamente los ocho propietarios científicos serializados más boundary. El estado final completo, las33señales de la traza y los202registros observados de140campos del tramo compartido coinciden. No se excluyó ningún campo del estado científico ni se modificaron tolerancias. No se necesitó cambiar auxiliares MuJoCo, guardar MjData adicional ni llamar `mj_forward`.

La evidencia decisiva es `COMPARISON_02.json`, `TRAJECTORY_COMPARISON_02.json` y los `INITIAL.json`/`RESULT.json` de `profile_whole_02` y `profile_split_02`. El ordinal del observador reinicia por proceso; en la comparación de su sidecar se alinearon las épocas por tiempo y se compararon registros, decisiones y duraciones, sin tratar ese ordinal administrativo como una coordenada neuronal.

El alcance es esta cola OFF, cuerpo actual, motor congelado y observador48. No demuestra continuidad histórica de auxiliares no guardados en48 bajo cualquier fuerza externa, no valida el nuevo observador49, otras entradas, viento o plasticidad, y no admite etapas4/5. Los checkpoints históricos48 conservan `restart_tested:false`; ésta es una cualificación separada del harness49. La GPU quedó liberada a Matrix Astra tras estos6ms.

## Intento01 conservado: defecto real del guard de reloj

El primer proceso real, desde `profile/final_state`, falló durante la construcción corporal, antes de intentar ningún paso neuronal o físico. No se borró ese negativo.

El fallo reproducible es `ValueError: FeTi physical clock differs`, en el guard de `GuardedTibiaBody.from_state`, `guarded_tibia_body.py:114`. `rh_tarsal_body.py:146` contiene el mismo criterio para el cuerpo exterior. Ambos comparan el tiempo MuJoCo flotante con `steps*dt` usando el límite absoluto heredado1e-10s.

Los cuatro estados finales48 tienen `steps=1899440`, `dt=25us`, reloj corporal y paterno `47.486000000126744s`. El reloj de pasos produce `47.486000000000004s`; su diferencia supera ese guard. Una reproducción CPU escalar desde el reloj preparado, sumando exactamente120000 veces el mismo dt, obtiene **bit a bit el tiempo guardado**. Esto respalda una incompatibilidad del criterio de carga con la acumulación flotante normal de esta trayectoria, no una corrupción identificada del checkpoint. Los relojes neuronales enteros no se alteraron.

La evidencia está en `CLOCK_BARRIER.json`; `reproduce_clock.py --out <archivo_nuevo.json>` reproduce el cálculo sin importar el organismo, usar GPU ni avanzar MuJoCo. La lectura de los cuatro checkpoints es sólo de pequeños arrays de integración corporal. No se modificaron umbrales, pesos, ecuaciones, dt, tolerancias ni archivos de48/históricos.

## Trabajo preparado y límites

`restore48.py` adapta localmente la restauración40 al schema48: recupera stored/effective W, cerebro/PN/KC, cuerpo, RNG, publicación y base del mundo. `resume49.py` reconstruye los propietarios input/motor/interval, mantiene sus relojes y baselines, y contiene una comparación exacta de los ocho propietarios serializados más boundary antes del primer paso. El olor conserva su apagado original en3000ms; sólo se extiende el guard de duración para observar la cola OFF.

La fábrica acepta `observer_installer(brain, stimulus)` compatible con el observador48. El nuevo observador de Matrix Astra no se instaló ni se probó aquí. En el intento01, **ni la identidad completa inicial ni el split alcanzaron su comprobación**, porque el constructor corporal falló antes. El intento02 comprobó posteriormente ambos, como se informa arriba. El `restart_tested:false` histórico no cambia.

El fallo ocurrió tras restaurar el operador efectivo sin diferencias, pero antes de terminar de reconstruir el organismo. Esto no certifica por separado toda la dinámica neuronal. Las mediciones negativas de48 y la atribución algebraica07 no cambian por este fallo de continuación.

La limpieza de un propietario parcialmente construido produjo un segundo error que ocultaba el primero como excepción final, aunque ambos quedaron en el traceback. Se corrigió la propagación local: ahora conserva la excepción primaria y añade el fallo de limpieza como nota. El código realmente ejecutado antes de esa reparación está preservado en `profile_whole_01/harness_source` y sus hashes figuran en el resultado del proceso. Las dos ejecuciones posteriores terminaron con limpieza normal.

## Coste y decisión

Primer proceso: **27.422s pared,28.462sCPU,6.605GB RSS;0ms neuronales intentados/comprometidos y0 pasos corporales nuevos**. Reproducción de relojes:3.427sCPU/3.191s pared; tampoco simuló.

Proceso continuo02:123.652s pared/130.314sCPU,4ms; reanudado02:82.067s pared/87.607sCPU,2ms. Pico RSS de ambos inferior a7.75GB. Comparación completa:15.302s pared/16.048sCPU; comparación de trazas y RHS:0.008sCPU. Los6ms corresponden a240 subpasos corporales de25us. El cierre agrega estos costes sin atribuir el tiempo de escritura del checkpoint al rendimiento neuronal. Se conserva margen dentro de los24ms asignados al aporte y la GPU se devolvió al coordinador.

No se ensanchó la tolerancia ni se normalizó el reloj para obtener PASS. Una reparación posterior debe definir una autoridad temporal coherente y comprobar el reloj contra un testigo exacto de pasos y procedencia; debe preservar el estado guardado y volver a demostrar identidad inicial y continuidad corporal/neural. Cambiar silenciosamente `data.time`, omitir el guard o escoger otro checkpoint para aprobar no resuelve el defecto.

El intento01 se cerró al encontrar la barrera real prevista por el mandato. La continuación científica, el observador nuevo y las etapas4/5 no quedan cualificados por ese intento.

## Reparación local02, posterior al diagnóstico

Matrix Astra pidió continuar dentro del mismo presupuesto con una comparación exacta contra la recurrencia desde el ancla preparada, preservando el tiempo guardado. `clock_guard49.py` sustituye únicamente los dos predicados de carga mediante un adaptador reversible; comprueba por AST que ninguna otra operación de esos métodos cambia. Los archivos históricos y sus identidades siguen intactos, y un recibo adicional declara los métodos efectivamente adaptados y el hash del overlay. Los linajes anteriores al ancla conservan su criterio; el nuevo camino está acotado a25us y hasta3200ms de48/49.

`CLOCK_OVERLAY_CPU_TEST.json` registra la admisión del tiempo exacto de48 y el rechazo de ±un paso físico, ±una unidad representable del reloj, no finitos, paso equivocado y dt diferente. No se añadió margen de tolerancia. El intento02 superó posteriormente la carga real, identidad científica y comparación split; esas pruebas son distintas de los ensayos CPU del predicado.

## Uso por el observador49

Importar `resume49.build(source, output, observer_installer=install)` en un proceso nuevo. `source` es el checkpoint; `output` debe ser una carpeta nueva. El callback retorna `(observer, undo)` y ofrece `bind_context`, `current_ms`, `flush` y `report`. Con `observer_installer=None` se usa el observador48 ya comparado. `run.step()` consume el siguiente milisegundo conservando el índice global; `run.save(folder)` guarda los propietarios; `run.close()` revierte todas las instalaciones locales.

La duración permitida se extiende a3200ms, manteniendo `odor_off_ms=3000`: al entrar en3001 se vuelve al basal nominal, sin clamping instantáneo de las neuronas. La referencia angular de la telemetría usa la pose inicial guardada de48 y no se recentra al reanudar. Se requiere una prueba separada de no interferencia para el nuevo observador antes de atribuirle la cualificación de este ensayo.
