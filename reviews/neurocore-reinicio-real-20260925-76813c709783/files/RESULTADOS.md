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

Próxima decisión de fondo: el estímulo externo debe tener estado versionado y
restaurarse antes de validar sensores. Después, para el estado interno de los
ejecutores hay tres rutas rivales: persistir lo que afecte a la continuación;
reconstruirlo de forma determinista desde el estado físico comprometido; o
definir prospectivamente una tolerancia de reinicio y demostrar que conserva
eventos, cargas y conducta. El único discriminador siguiente será reconstruir
una caché de ejecución identificada por código en la rama viva a los 1 ms,
sin tocar historia física ni llamar a una reinstalación general, y contrastar
el primer intento siguiente con la rama intacta y la reiniciada. Si no se
puede nombrar y aislar esa caché, no se ejecutará la sonda. No se ajustarán
registros CUDA ni tolerancias a posteriori.

Evidencia: `run_04/PREPARE.json` y `continuous_2ms/` contienen la rama continua;
`run_05_control/PREPARE.json` registra el control exacto;
`reload_02/RELOAD.json` registra el fallo del cargador;
`reload_07/RELOAD.json` y `resumed_2ms/` registran la reconstrucción experimental
y la continuación divergente. `check_restart.py` reproduce la prueba con procesos
separados. Las corridas 01–02 y 04–06 de recarga incluyen correcciones del
instrumento de prueba y permanecen como historial, sin reclamaciones científicas.
