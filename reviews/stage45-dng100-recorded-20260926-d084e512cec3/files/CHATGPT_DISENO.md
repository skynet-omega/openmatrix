# Revisión externa del diseño47

ChatGPT C++/CUDA, conversación designada previamente por el usuario. No ejecutó datos ni inspeccionó el árbol local. Citas opacas pertenecen a su conversación. Recomendación consultiva; Codex verifica la implementación y conserva el presupuesto.

**La adquisición no es redundante con la evidencia disponible. Mantendría el diseño, simplificando el registro y restringiendo la reconstrucción de aceptación a épocas terminadas correctamente.** El principal riesgo no es científico sino instrumental: mezclar evaluaciones especulativas, pasos aceptados y estados finalmente publicados.

Releí el informe público45/46 y el controlador residente archivado en el mismo commit. El informe confirma que **los operandos dinámicos de DNg100 no se guardaron durante45**. Los antecedentes del7 y13 de septiembre no sustituyen esa información: otra ley/preparación o activación directa responden preguntas diferentes. No he comprobado la existencia de un reinicio intermedio local ni ejecutado código. 

## 1. Captura mínima: resultado del operador, no una segunda evaluación

Tu ubicación es correcta, pero distinguiría **dos puntos de observación**:

**Dentro del CSR:** guardar la suma neta **ya reducida**, junto con `drive`, `theta` y `gain` en la precisión realmente utilizada. Escribirla después de la reducción, sin modificar sus operaciones ni sustituirla por `positivo + negativo`. Las sumas auxiliares por signo pueden ayudar, pero **no son necesarias para decidir si el objetivo permanece cero**; si las conservas, etiquétalas como auxiliares, no como reproducción exacta de la suma original.

**Después del callback completo:** copiar inmediatamente las dos filas de `target`, `rate`, estado proyectado `z` y, preferiblemente, la derivada `rate*(target-z)` efectivamente devuelta. Así detectas una sustitución posterior al CSR y no confundes estado bruto con estado consumido.

No guardes referencias a esos arrays: **cópialos al buffer diagnóstico antes de la siguiente evaluación**, que puede reutilizarlos. No reconstruyas después los operandos mediante otra llamada al coeficiente.

Según el RK3 que compartiste, **`k4` participa en el estimador, pero no en la combinación de tercer orden que construye `high`**. Un objetivo positivo únicamente en `k4` no prueba que el estado aceptado de ese intento debiera aumentar. Registra también su reloj izquierdo real; no lo etiquetes artificialmente como el extremo derecho.

## 2. Aceptación: puedes evitar modificar `decide`, con una condición

En el controlador archivado, aceptar no depende únicamente de `error <= 1`: intervienen `failure`, finitud, validez del ensayo y dominio. `commit` exige además `accept && !failure`. **Reproduce la condición de la versión efectivamente cargada**, no una aproximación ni una versión histórica asumida. 

Tu reconstrucción es suficiente **para una época que retorna con éxito**, si:

- Registras el vector final de estado del estimador por intento, sin redondearlo.
- La clasificación reproduce los contadores aceptados/rechazados.
- Los tiempos y la secuencia de estados concuerdan con las decisiones reconstruidas.

**La igualdad de las dos filas DNg100 no verifica por sí sola aceptación:** con objetivo cero pueden permanecer iguales tanto al aceptar como al rechazar.

Para épocas fallidas, no inferiría aceptación sólo desde el estimador. Las conservaría como **fallidas/no publicadas**, fuera de la interpretación principal. Si necesitas clasificar sus decisiones internas, entonces sí: una escritura diagnóstica de `accept/failure` después de `decide`, sin alterar la decisión, es más fiable que deducirla.

Hay tres etiquetas distintas:

**aceptado por el integrador → época CNS exitosa → resultado retenido por el acoplamiento.**

Un predictor puede contener pasos aceptados y después restaurarse deliberadamente. El adaptador debe registrar esa disposición final; el contexto «comprometido» asignado al comenzar tampoco basta si ocurre rollback posterior.

## 3. Dos trampas concretas del buffer capturado

**El índice debe avanzar en GPU una vez por ejecución real del trial.** Un contador Python empleado durante captura queda fijado al construir el grafo y no proporciona un ordinal nuevo por reproducción. Usa `(época, trial, etapa, fila)`, con las cuatro etapas escribiendo en posiciones distintas. Excluye calentamientos y reinicia el contador antes de la ejecución real. La captura de CuPy registra operaciones para reproducirlas posteriormente; además, no admite transferencias síncronas dispositivo→host dentro de ella. :chatgpt-content-reference{index="2"}

**10.000 intentos no garantizan sólo 10.000 ejecuciones del grafo hijo en una ruta fallida.** En el controlador publicado, `prepare` puede marcar `trial_limit`, pero el hijo sigue conectado antes de `decide`; puede ejecutarse con datos anteriores. Esto importa tanto para clasificación como para evitar escritura fuera del buffer.  

Exige una guarda de capacidad y un indicador de desbordamiento **separado de los flags numéricos del motor**. Nunca sobrescribas muestras anteriores mediante un índice circular silencioso. Al volver al host, el desbordamiento invalida la adquisición. Ordena copia y reutilización del buffer para no vaciarlo mientras la GPU lo escribe.

## 4. Falsadores y decisión final

El registro permitiría distinguir:

**Entrada modulada, objetivo final siempre cero:** bloqueo localizado en la respuesta de esa ley bajo el estímulo; no demuestra un defecto de implementación.

**Objetivo final incompatible con la ley o sustitución declarada:** discrepancia concreta que justifica revisar esa operación.

**Objetivos/derivadas de etapas que alimentan la solución aceptada incompatibles con la transición registrada:** revisar integración/publicación, excluyendo primero predictores descartados y proyecciones.

El instrumento queda invalidado si pierde registros, mezcla etapas, no reproduce contadores/transiciones o altera los observables históricos fuera del contrato previamente fijado. **La coincidencia con45 respalda ausencia de interferencia observada, no certifica todas las variables no guardadas.**

**Procedería con esa adquisición única hasta3 s por brazo, dentro del presupuesto propuesto, sin ampliar si el resultado es nulo.** Sólo un reinicio completo intermedio realmente verificado permitiría omitir parte del prefijo. Si el registro muestra una ley coherente manteniendo objetivo cero, habrás localizado mejor el negativo, pero todavía no identificado una reparación legítima del circuito.

