**Conservaría el transporte y sus resultados, pero corregiría el verificador antes de utilizar su `screen_pass` como comprobación estructural. Además, reproduje un caso donde el puerto de eventos es exacto y, aun así, el control temporal no detecta un error aguas abajo.** Ese segundo límite está también en el método heredado; no demuestra que el adaptador nativo haya introducido la discrepancia.

**Alcance:** leí las fuentes del commit indicado. El entorno permitió ejecutar pruebas CPU, pero no descargar las partes binarias: **no verifiqué tus arrays, el SHA-256 del paquete ni una simulación CUDA/MuJoCo**. Ejecuté dos reproductores pequeños sobre datos artificiales. Las copias de las fuentes utilizadas coinciden con los identificadores Git de los archivos publicados.

:chatgpt-content-reference{index="17"}

## 1. Verificador: acepta cambios estructurales que debería distinguir

`history_difference()` comprueba que `left` y `right` tengan igual forma **dentro de cada segmento**, pero no exige que las historias de ambas ramas tengan la misma forma y tipo. Posteriormente, `verify()` excluye de los errores estructurales cualquier ruta que contenga `/history/`. Esa exclusión permite más que diferencias legítimas de segmentación. 

Ejecuté los verificadores publicados, sin modificarlos, sobre siete fixtures. Estos fueron los resultados relevantes:

| Alteración artificial | `screen_pass` |
|---|---:|
| Historia de un canal → dos canales con valores repetidos | **True** |
| Historia FP64 → FP32, con valores exactamente representables | **True** |
| Reloj interno PN cambiado, conservando el reloj global | **True** |
| Tiempo de la traza cambiado, conservando sus valores | **True** |
| Identidad PN anidada cambiada | **True** |

El control idéntico pasa y una alteración suficientemente grande de los valores de historia se rechaza. **No son corrupciones encontradas en tus archivos: son contraejemplos ejecutados que muestran qué podría aceptar el verificador.**

La causa de los tres últimos casos es distinta: `compare.py` comprueba el reloj global y ciertos metadatos seleccionados, pero no convierte todos los cambios de relojes e identidades anidados en condiciones de rechazo. Puede informar diferencias y mantener `screen_pass=True`. 

**Reparación acotada:** permitir diferencias en el número y las fronteras de segmentos, pero exigir forma, dtype e identidad de canales compatibles antes de interpolar. Para relojes y metadatos, definir las rutas obligatoriamente exactas según el esquema; no exigir igualdad indiscriminada de estadísticas numéricas que legítimamente cambien entre integradores.

La comparación en la unión de nudos es adecuada para historias continuas lineales por tramos **una vez establecidos esos invariantes**. No propongo volver a exigir el mismo número de segmentos.

## 2. Eventos: el estimador puede omitir su efecto sobre otros estados

`FilterPorts` reconstruye los filtros a partir de sus eventos. Sin embargo, el controlador C++ selecciona el paso usando duración restante, paso propuesto y límites; **no consulta la próxima marca de evento**. El punto medio evalúa los coeficientes en tiempos determinados dentro del paso.   

Reproduje en CPU la aritmética publicada de `coeff()` y `midpoint()`, con un puerto analítico y tres estados:

\[
q'=-q/\tau,\qquad
s'=(q-s)/\tau,\qquad
z'=(s-z)/\tau.
\]

Estado inicial nulo; \(h=\tau=125\,\mu s\); salto \(q\leftarrow q+0,5\) en \(0,9h\).

| Resultado | Valor |
|---|---:|
| \(z\) con un paso completo | **0** |
| \(z\) con dos medios pasos | **0** |
| Estimación por duplicación de paso | **0** |
| \(z\) analítico al final | **0,002262093545** |

El puerto entrega correctamente \(q\) y \(s\) al final. Pero todas las evaluaciones que alimentan a \(z\) ocurren antes del salto: ambos aproximantes omiten su efecto y coinciden entre sí. **Con esos valores, las condiciones del controlador aceptarían el paso; no ejecuté esa aceptación en GPU.**

Estos parámetros son un falsador del método, **no parámetros extraídos de la mosca**. El mismo patrón de muestreo aparece en `event_coupling.py`, por lo que la igualdad nativo–referencia a igual partición no elimina este límite compartido. 

**Acción concreta:** probar que el controlador termina subpasos en las marcas de eventos conocidas, o justificar otra integración de su efecto sobre los receptores. No cambia ganancias, eventos físicos ni el reloj corporal de 1 ms. La corrección debe quedar versionada y comparada; no introducirla silenciosamente en una corrida iniciada.

La prueba publicada `check_core.py` contrasta el puerto analítico al final, pero no este acoplamiento hacia un tercer estado. 

## 3. Punteros y rollback: camino ordinario razonable, recuperación aún no cerrada

En el camino ordinario, el adaptador copia estado, entradas, transmisión PN y campos retenidos a los buffers utilizados por el grafo. `NativeGraph` conserva el grafo, su stream y su pool; el controlador confirma los subpasos copiando sobre `x`. **No encontré una sustitución habitual de punteros que pueda declarar incorrecta en esa secuencia.**  

Tampoco falta necesariamente la química de calcio porque `pn_execution.py` no la exporte junto con voltajes: `Calcium` mantiene el propietario original y su `commit()` confirma sobre él. Los estados eléctricos se vuelven a importar antes de cada llamada.  

**La recuperación merece una prueba específica:** el controlador puede haber confirmado subpasos en su buffer GPU cuando falla una época. El adaptador no publica entonces el estado final en `brain`; la capa exterior invoca `_restore_joint`. Además, el adaptador conserva referencias tomadas al instalarse —lector PN, buffers y objeto PN—.  

No afirmo que ese rollback falle. **La implementación completa heredada de `_restore_joint` y de `ParentSnapshotFrames/copy_values` no está entre las fuentes que pude examinar**, por lo que no puedo establecer si mantiene o sustituye todos esos propietarios.

El falsador útil sería: provocar una excepción después de un subpaso aceptado, restaurar y continuar; comparar con una ejecución sin el intento fallido. Deben coincidir estado publicado, membranas, calcio, eventos, historias y relojes. Si la restauración cambia algún propietario capturado, debe reconstruirse o revalidarse explícitamente el adaptador, no comprobar únicamente formas.

## 4. Guardia menor: la vía especulativa PN no recibe `max_newton`

`advance_graph_resident()` admite y valida `max_newton`, pero no lo transmite a `graph_stage()`. El grafo ejecuta dos correcciones y puede aceptar aunque se haya solicitado `max_newton=1`. **No afecta al caso documentado que utiliza ocho; sí deja abierta una entrada admitida que incumple su presupuesto.**  

La reparación mínima es impedir esa vía cuando el presupuesto no permite sus dos correcciones y utilizar el camino original, sin cambiar tolerancias.

## Dictamen operativo

**No descartaría los resultados cortos ni modificaría en caliente el piloto en marcha.** Repararía ahora el verificador y volvería a aplicarlo a los arrays ya existentes. El contraejemplo temporal requiere una prueba numérica dirigida, no otra campaña sintética de optimización ni un retorno al estudio general del motor.

El resultado largo permanece desconocido para esta revisión. El archivo publicado `VERIFIED_LEFT.json` comunica una criba corta favorable; **yo no reconstruí esas cifras desde sus arrays**. 

Leí los seis archivos principales solicitados y sus dependencias pertinentes de eventos, acoplamiento, coeficientes, KC, PN, comparación y ejecución: **28 archivos**, enumerados exactamente en `ALCANCE_LECTURA.json` dentro de la entrega. Lo ejecutado fueron **siete fixtures del verificador y un caso analítico CPU**, no el organismo. Los hallazgos justifican cerrar esas condiciones concretas antes de ampliar la interpretación de fidelidad temporal, **no afirmar que el transporte o la etapa 3 hayan fallado biológicamente**.
