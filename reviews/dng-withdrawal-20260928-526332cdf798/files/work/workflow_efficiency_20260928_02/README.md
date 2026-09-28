# Revisión acotada del flujo de trabajo — 28 de septiembre de 2026

El flujo real de recuperación, análisis y registro ya es utilizable: una consulta
de evidencias tarda unos 0,04 s y reutilizar el análisis DNg100 existente tarda
unos 0,30 s. No apareció fundamento para rehacer el laboratorio. Sí se reprodujeron
dos defectos evitables de concurrencia y cierre de recursos; ambos quedaron
corregidos. Queda una oportunidad concreta en la granularidad de la caché.

Esta revisión operativa no acredita navegación ni recuperación frente al viento.
Las etapas 4/5 siguen abiertas. No se cambiaron ecuaciones, parámetros, contratos
científicos, selección de grupos ni resultados retenidos.

## Alcance y presupuesto

Encargo coordinado con Matrix Astra: un flujo real, máximo tres hallazgos, dos
reparaciones, 15 minutos de reloj y 300 s de CPU de pruebas/mediciones; cero CNS y
cero GPU. Inicio: 2026-09-28 22:44:45 UTC. Flujo elegido:
`context → workbench dng-context-capacity → tablas/assessment → lab`.
Las pruebas adicionales usan bases temporales y una copia mínima aislada del
workbench. No se ejecutó la suite completa; queda a cargo de Matrix al integrar.

## 1. Consultar el catálogo de código adquiría acceso de escritura — corregido

`scripts/code_catalog.py:connect(create=False)` inicializaba esquema y metadatos
en cada consulta. Por eso `search`, `stats` y `duplicates` fallaban con
`database is locked` si otro escritor mantenía una transacción, aunque había
datos confirmados disponibles. Una consulta también modificaba los bytes del
catálogo.

La conexión de consulta ahora abre SQLite en modo de solo lectura, activa
`query_only`, verifica esquema/raíz y cierra la conexión si falla la validación.
La construcción del índice conserva su ruta de escritura. Las pruebas reproducen
un escritor concurrente real, comprueban las tres consultas, la conservación del
archivo y el rechazo de una escritura por la conexión lectora.

## 2. Algunas excepciones retenían el bloqueo del catálogo de evidencias — corregido

En `src/lab_evidence.py`, un error durante la inicialización de tablas podía dejar
abierta la conexión y su bloqueo exclusivo. Además, `refresh(only=...)` validaba
una ruta inválida después de tomar ese bloqueo, antes de entrar en su cierre
protegido.

Ahora se cierra la conexión o el bloqueo ante cualquier fallo de inicialización,
y se valida la selección antes de adquirirlos. Las pruebas retienen la excepción
y su traceback para impedir que la recolección de objetos oculte el defecto;
comprueban que otro escritor puede entrar y que una actualización posterior
funciona. Se conserva el bloqueo necesario para serializar escritores.

## 3. Cambios ajenos pueden invalidar la caché de un componente — pendiente

`Workbench.code_identity()` incluye todos los módulos de `matrix_workbench`,
incluido el registro global de componentes de `tasks.py`, para cada unidad.
En una copia temporal, añadir un registro ajeno a `compare_arrays` cambió la
clave y volvió a ejecutar esa misma comparación con entradas idénticas:

| Caso aislado | Reutilizado | Tiempo de reloj |
| --- | --- | ---: |
| Primera ejecución | No | 0,298 s |
| Repetición idéntica | Sí | 0,169 s |
| Registro de un componente ajeno | No | 0,288 s |

El ensayo demuestra invalidación demasiado amplia; no estima su coste en una
campaña CNS. No se modificó `runtime.py` ni `tasks.py`. Una reparación posterior
debe distinguir el código del componente ejecutado y sus dependencias reales,
manteniendo la invalidación por cambios científicos o del ejecutor. Borrar hashes
o relajar la verificación de fuentes sería una solución incorrecta.

## Mediciones del flujo real y preservación científica

| Operación | Antes | Después |
| --- | ---: | ---: |
| Recuperar contexto | 0,948 s | 0,950 s |
| Buscar evidencias DNg100 | 0,041 s | 0,034 s |
| Reutilizar análisis y registrar | 0,604 / 0,298 s | 0,296 s |

Son muestras puntuales con hardware compartido; no demuestran una aceleración
estadística. El beneficio probado de las dos reparaciones es evitar escrituras
innecesarias y bloqueos persistentes, no acelerar el motor neuronal. La consulta
real de código posterior devolvió tres resultados correctamente en 2,744 s.

Hubo también una ejecución CPU de 3,221 s que **no reutilizó** el resultado. Se
comparó su procedencia: había cambiado legítimamente
`src/dng_context_capacity.py` durante el trabajo paralelo. No se atribuye este
caso al hallazgo 3. Las cuatro tablas (`groups`, `windows`, `aligned`,
`differences`) conservan exactamente sus bytes y `assessment.json` conserva su
contenido. La siguiente invocación reutilizó la nueva clave en 0,296 s. Esta
revisión produjo cero milisegundos CNS y no inició GPU.

Las seis nuevas regresiones fallaban antes de las reparaciones. Después,
**27 pruebas específicas pasaron**: las seis nuevas y los tests existentes de
`lab_evidence` y `code_catalog`. Pytest informó 3,05 s; el proceso completo tardó
3,592 s. Los registros de medición acumulan **9,603 s de CPU contabilizada**;
incluyen los procesos medidos y no constituyen un censo perfecto de todos los
descendientes ni del coste del editor. El margen frente a 300 s es amplio.

## Decisión práctica para avanzar etapas 4/5

Conservar este flujo y usar la evidencia ya disponible para el siguiente contraste
causal de contexto. No exigir una reestructuración ni una actualización global de
catálogos antes del experimento. Registrar solo el trabajo que cambia, mantener
los controles y resultados negativos, y reservar una intervención posterior
acotada para la invalidación por componente si genera repeticiones costosas.
La recuperación y la reutilización medidas no justifican otra ronda general de
microoptimización. El progreso biológico depende del experimento científico de
Matrix, no de declarar terminadas estas reparaciones operativas.

## Evidencia y entrega

- `changes.patch` y `source_versions.json`: dos archivos compartidos modificados;
  `runtime.py` permanece idéntico.
- `before/`: versiones anteriores preservadas.
- `test_workflow_regressions.py`, `regressions_before.*`,
  `regressions_after.*`: reproducción y verificación focalizada.
- `flow_before.json`, `flow_after.json`, `reuse_final.json`: mediciones del flujo
  y confirmación de reutilización/registro.
- `real_reuse_identity_diff.json`: causa de la nueva ejecución y comparación de
  salidas científicas.
- `cache_scope_reproduction.json`: reproducción aislada del tercer hallazgo.

No se alteraron contratos sellados, `INDEX`, continuidad, recetas científicas ni
`tasks.py`; no se eliminó evidencia. Matrix puede ejecutar una vez su suite de
integración y actualizar el índice de código al incorporar estos cambios.
