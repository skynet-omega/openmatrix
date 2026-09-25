# Reinicio real de la candidata CNS + membranas

**No pasa aún como checkpoint íntegro.** Se ensayó el organismo real con
`odor_left`, sin cambiar ecuaciones ni umbrales. Tras 1 ms de avance se escribió
un checkpoint de 463.979.491 bytes. La carga en otro proceso falló con
`Pending odor disagrees with world/body`: el estado del mundo conserva la señal
olfativa pendiente, pero el campo lateral externo instalado en `world.boundary`
no forma parte de `AntennalWorld.state_dict()` y se reconstruye el campo base.

Para aislar ese defecto, una carga **sólo experimental** reconstruyó el campo
desde sus metadatos antes de la validación heredada. Así se recuperaron
exactamente todos los campos serializados de `core` y `prosthesis`, el reloj,
las señales pendientes, el CNS y los metadatos del campo **antes** del siguiente
paso. No se modificó el cargador histórico; el checkpoint publicado sigue sin
ser autónomo.

Al continuar 1 ms, la rama reiniciada difirió de la rama continua en CNS, PN,
membranas, cuerpo y tiempos de eventos. La primera diferencia registrada en
un evento fue 6,78e-21 s, pero la posición corporal indicada por
`body.integration[1]` difirió en 5,88e-8 y `published.rates[52]` en
7,32e-4. Estos números prueban ausencia de identidad bit a bit para este
reinicio; no se evaluó una puerta de error biológico de reanudación. Dos corridas
**continuas independientes** de los mismos 2 ms sí coincidieron exactamente en
estado neural, cuerpo, señales publicadas y 16 bloques de eventos del segundo
milisegundo. Por tanto, la divergencia observada está asociada a restaurar y
reconstruir ejecutores, no a variabilidad ordinaria de esas corridas continuas.

No se atribuye todavía la divergencia posterior a un único propietario. El
cuerpo usa `mjSTATE_INTEGRATION`, que incluye `qacc_warmstart` según la
[definición de MuJoCo](https://mujoco.readthedocs.io/en/3.3.5/APIreference/APItypes.html);
por tanto, no se puede imputar esta divergencia a la omisión de ese campo.
La
fuente PN sí publica calcio: `Calcium.commit` llama al `commit` del puerto
original, y la fuente PN restaura su estado si falla tras avanzar. No se hizo
un parche de calcio. Tampoco se promovió esta etapa como motor completo o más
rápido.

La decisión siguiente fue versionar el campo externo y probar una única
caché de ejecución identificada por código, sin ajustar tolerancias ni CUDA.
Para el estado interno de los ejecutores siguen existiendo tres rutas de fondo:
persistir el estado causal, reconstruirlo exactamente desde el estado físico,
o definir prospectivamente una tolerancia y demostrar su efecto en eventos,
cargas y conducta.

Evidencia: `run_04/PREPARE.json` y `continuous_2ms/` contienen la rama continua;
`run_05_control/PREPARE.json` registra el control exacto;
`reload_02/RELOAD.json` registra el fallo del cargador;
`reload_07/RELOAD.json` y `resumed_2ms/` registran la reconstrucción experimental
y la continuación divergente. `check_restart.py` reproduce la prueba con procesos
separados. Las corridas 01–02 y 04–06 de recarga incluyen correcciones del
instrumento de prueba y permanecen como historial, sin reclamaciones científicas.

## Continuación: campo externo versionado y caché PN falsada

Se añadió `external_checkpoint.py`, una envoltura genérica que guarda el
checkpoint del modelo junto al estado versionado del controlador externo,
con hashes de ambos manifiestos y guardado atómico. El adaptador
`static_lateral_checkpoint.py` conserva los parámetros geométricos y el reloj
del campo olfativo, comprueba la identidad de su implementación y lo restaura
durante el enlace del cuerpo, **antes** de la validación de señales pendientes.
El cargador histórico carece de esa interfaz; el adaptador usa un enlace
temporal y delimitado que retira al salir. No se cambió el repositorio histórico
ni se incorporó olor al núcleo genérico.

La nueva corrida real `run_10_versioned_driver` guardó un checkpoint autónomo
de 463.980.429 bytes tras 1 ms (`checkpoint_wall_s=27,556`). Un segundo proceso
lo cargó sin error de campo. `core` y `prosthesis` serializados, reloj, CNS,
sensores pendientes y metadatos del campo coincidieron **exactamente antes**
del paso. El segundo ms continuo coincidió exactamente con `run_04` en
`core`, `prosthesis`, salida publicada y eventos. La continuación versionada
coincidió exactamente con la reanudación diagnóstica `reload_07`, incluida
su divergencia frente a la rama continua. `VERSIONED_COMPARISON.json` registra
las cuatro comparaciones completas. La alteración deliberada del estado del
driver, del manifiesto del modelo y del hash de fuente fue rechazada antes
de ejecutar el cargador (`CHECKPOINT_GUARDS.json`). Por ello, **el defecto
de carga del campo queda reparado en este puente, pero la reanudación bit a
bit del organismo sigue sin aprobar**.

El único control de caché elegido reconstruyó `view._stage_graph` y
`view._stage_graph_key` del ejecutor PN en la rama viva al límite de 1 ms.
Antes de avanzar, `core` y `prosthesis` comprometidos permanecieron iguales.
En `run_08_stagegraph_cache`, la continuación fue exactamente la de la rama
viva en los cuatro componentes comparados y los 16 bloques de eventos.
Esta caché específica **no explica por sí sola** la divergencia del reinicio.

El primer evento diferente entre A y C aparece en el bloque 2 del segundo
milisegundo, neurona 52140 (`nongamma_lif`, fila 37405), en `time_s` por
aproximadamente 6,78e-21 s. Se intentó registrar sus operandos en una sola
corrida adicional. Aunque eventos, cuerpo y tasas finales no variaron, la
instrumentación alteró `pn_online_state.base.apl_charge_pC` respecto al
control no instrumentado (`trace_lif_continuous_01/TRACE.json`). Su criterio
previo de no interferencia falló; **esa traza no se interpreta** y no se abrió
la segunda corrida prevista. No hay aún propietario probado para la
divergencia posterior ni medida de coste comparable que demuestre aceleración.
