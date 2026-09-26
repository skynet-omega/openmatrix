# RK3 residente: memoria de propuesta tras cortes por eventos

Candidata aislada del motor07. El único cambio de cálculo está en
`engine/resident_controller.cu`, visible en `ENGINE.diff`.

Un evento interior puede recortar un paso por calendario aunque el error permita
uno mayor. Si ese paso se acepta con error<0,1, la próxima propuesta toma el máximo
entre la propuesta normal y la anterior, limitado por el máximo permitido.
No se aplica al final de época, ante rechazo ni cuando no hubo truncamiento.
El siguiente intento sigue comprobando eventos, error y dominio.

La pareja usa el organismo completo, la preparación guardada de campaña40,
el campo gaussiano derecho y la misma interfaz motora de la comparación09.
PN/membranas, pesos, tolerancias, parámetros y estimador RK3 no cambian.
`run_trial.py` obliga a BLAS1 y compara directamente el estado preparado.
Cada corrida registra también PN, publicación/RNG y estado neural al inicio.

`PLAN.json` fija dos corridas de100ms,900s de proceso entre ambas, y una sola
confirmación1s/2400s si hay al menos10% menos avance, proceso sin regresión y
compatibilidad funcional. El revisor señaló el mando forward además del yaw;
`evaluate.py` lo comprueba desde las trazas existentes. Esa aclaración y su hash
se guardaron antes de ejecutar la candidata, sin cambiar fuentes de ejecución.

El control de100ms ya reprodujo35/35campos del primer tramo07 guardado en09.
La revisión de ChatGPT fue estática sobre código real; Jev clasificó el protocolo.
No son ejecuciones independientes del organismo.

## Ejecutar y comprobar

Requiere el modelo/checkpoint histórico instalado, CuPy/CUDA y el entorno local.
Las rutas de `SOURCES.json` identifican las fuentes que realmente se ejecutaron.
No se debe editar una fuente congelada para repetir: usar una carpeta nueva.

```bash
/home/daroch/miniconda3/envs/GPU/bin/python engine/build.py --out /tmp/libevent_memory_new.so
/home/daroch/miniconda3/envs/GPU/bin/python -O check_policy.py
/home/daroch/miniconda3/envs/GPU/bin/python run_trial.py --engine control --ms 100 --out control_new --wall-limit 400
/home/daroch/miniconda3/envs/GPU/bin/python run_trial.py --engine event_memory --ms 100 --out candidate_new --wall-limit 400
python evaluate.py --control control_new --candidate candidate_new --ms 100 --paired-timing --out pair_new.json
```

Las pruebas originales están además envueltas por un límite externo de450s por
proceso corto. El temporizador interno protege el avance principal; el externo
incluye el cierre. `check_policy.py` escribe `FIXTURE.json`: ejecutarlo sólo en
una copia nueva para preservar la salida original. Las comparaciones de datos guardados no
requieren GPU ni reconstruir el organismo.

`check_integrity.py` comprueba externamente filas, relojes, procedencia,
observaciones iniciales/finales y presupuesto sobre los mismos datos. Añade
payload y cobertura de emparejamiento de eventos como diagnóstico, sin nuevas
tolerancias. `verify_saved.py --long` recalcula también esa comprobación y prueba
que registros truncados, PN detenido y PN final ausente se rechacen.

El informe final separará reducción medida de trabajo/tiempo, cribas funcionales,
estados internos, eventos comprometidos y predictores descartados. Cumplir la
criba en esta preparación no demuestra equivalencia biológica general.
