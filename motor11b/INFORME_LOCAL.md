# Resultado local para el analista externo

Estado del ejecutor: **BLOQUEADO_MOTOR_11B**. Etapa 3 abierta.

Se revisó y ejecutó MOTOR11B sin modificar el paquete recibido. A1/B1 son datos previos; B2/A2 y SONDA son las tres ejecuciones nuevas autorizadas. Cero subagentes, entrenamiento o reajustes. No se promovió una política del motor.

## Coste y repetibilidad

| Brazo | Pared por 40 ms medidos (s) | CPU acumulada (s) |
|---|---:|---:|
| A1 | 298.116884 | 4290.162772 |
| B1 | 197.223357 | 235.130253 |
| B2 | 206.909248 | 227.157833 |
| A2 | 337.117172 | 4454.562956 |

BLAS=1 redujo la mediana de tiempo de pared un 36.3805%; aceleración 1.5718×.
Extrapolación lineal de cinco segundos simulados: 11.028 h con A y 7.016 h con B. Es una estimación desde 40 ms medidos por brazo; no se ejecutaron cinco segundos.
Cada ejecución incluye 40 ms de preparación y 10 ms de ensayo; el benchmark excluye los primeros 10 ms. ABBA está repartido en dos invocaciones y sólo contiene dos mediciones por política. No hay intervalo de confianza ni estabilidad general demostrada.

- repetibilidad_A: igualdad exacta **True**.
- repetibilidad_B: igualdad exacta **True**.
- cruzada_A1_B1: igualdad exacta **False**.
- cruzada_A2_B2: igualdad exacta **False**.

Diferencias A2/B2 reconstruidas por el comparador:

| Archivo | Array | Elementos distintos | Máximo absoluto |
|---|---|---:|---:|
| traza | PN_q_legacy | 10 | 1.11022302463e-16 |
| traza | PN_general_transmission | 67 | 2.77555756156e-17 |
| traza | PN_gamma_nS | 28 | 1.73472347598e-18 |
| traza | PN_additional_nS | 89 | 8.67361737988e-19 |
| final | q | 418 | 1.09079412169e-14 |

Estos arrays cubren observaciones del prefijo y tres arrays finales. No contienen todo el estado interno ni una reanudación íntegra. La igualdad del cuerpo o de la salida motora en este intervalo no prueba equivalencia a segundos.

## Capturas para optimización

Capturadas 0 cápsulas; omitidas 3. Llamadas al plan: 1280; al solver completo: 1280; fallbacks registrados: 0; máximo de iteraciones informado: 1.

Omisiones: [{"llamada": 1, "bytes": 31434736, "motivo": "Límite de captura, no ausencia de modelo"}, {"llamada": 64, "bytes": 31434736, "motivo": "Límite de captura, no ausencia de modelo"}, {"llamada": 640, "bytes": 31434736, "motivo": "Límite de captura, no ausencia de modelo"}]

Consultar tiempos de regiones y Nsight en `ejecucion/SONDA/` y `ejecucion/NSIGHT_STATS.txt`, cuando estén disponibles. Las regiones inclusivas se solapan; no sumar sus tiempos. SONDA usa la política A. Su perfil no describe automáticamente el coste residual con BLAS=1. Las cápsulas son sistemas PN concretos, no el cerebro completo; el replay no mide la aceleración integrada.

## Autocrítica y decisión

El requisito de identidad exacta sirvió para detectar una diferencia real. Habría sido excesivo tratar ese fallo como refutación de toda optimización; también sería injustificado declarar equivalencia porque la diferencia es pequeña. MOTOR11 conserva su fallo, y MOTOR11B aporta repetibilidad y entradas para estudiar el error sin mover la tolerancia histórica.

La principal limitación práctica sigue siendo el coste por segundo simulado. Una mejora del solver sólo se justifica por su contribución al coste total: el perfil04 también señala proyecciones KC, ensamblado PN y numerosas operaciones pequeñas. Conviene probar como máximo dos alternativas sobre estas capturas antes de otro ensayo integrado. No se infiere que exista un fallo biológico por esta medición del rendimiento.

Clasificación de BLAS=1: **PROMETEDOR_NO_CONFIRMADO** como mejora de ingeniería. El motor operativo conserva su política previa. No se cambian ecuaciones, dt, precisión, pesos ni criterios de etapa 3.

## Integridad, presupuesto y reproducción

Antes de ejecutar: 21 archivos del paquete, 426 fuentes y 37 referencias contrastados; 34 pruebas CPU aprobadas. Presupuesto previo en `REGISTRO_LOCAL.json`: tres trabajadores de hasta 1200 s, 3800 s agregados incluyendo exportación, 57000 s CPU, 8 GiB RAM y 16 GiB GPU; cero reintentos.

Los resultados originales y sus manifiestos se conservan dentro de `referencia11/`; los nuevos en `ejecucion/`; el código recibido en `propuesta_recibida/`. La comprobación CPU del retorno reconstruye comparaciones y tiempos desde datos, además de verificar hashes. No repite las simulaciones ni reemplaza sus checkpoints.

Desde una extracción limpia, con el entorno descrito en `ENTORNO.md`:

```bash
cd MATRIX_MOTOR_11B_RESULTADOS_CODEX_2026-09-22_02
python -I -B verificar_retorno.py
```

El comando para el ensayo integrado está en el paquete recibido; requiere el laboratorio y checkpoint locales. Este retorno es autocontenido para reconstruir comparaciones y tiempos. No contiene cápsulas PN completas ni reconstruye el organismo completo. El mensaje de siguiente trabajo está en `MENSAJE_PARA_CHATGPT.md`.

## Fallo conservado: error

{'tipo': 'RuntimeError', 'mensaje': 'Subproceso terminó con código 2: SONDA.log', 'traceback': 'Traceback (most recent call last):\n  File "/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor11b_external_20260922/MATRIX_MOTOR_11B/motor11.py", line 210, in coordinador\n    ejec(cmd,raiz,salida/f"{nombre}.log",limite_s=plan["limite_s_por_proceso"])\n  File "/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor11b_external_20260922/MATRIX_MOTOR_11B/vendor/motor11_original.py", line 497, in ejecutar_proceso\n    exigir(codigo == 0, f"Subproceso terminó con código {codigo}: {log.name}")\n  File "/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor11b_external_20260922/MATRIX_MOTOR_11B/vendor/motor11_original.py", line 55, in exigir\n    raise RuntimeError(mensaje)\nRuntimeError: Subproceso terminó con código 2: SONDA.log\n'}

## Recuperación local posterior, sin nueva simulación

SONDA/A1 exacto reconstruido desde arrays: True. El trabajador terminó el CNS y guardó las salidas antes del error de lectura. La ausencia de cápsulas completas se confirma: cada una requería31,43MB y el tope era20MB. El índice conserva las tres omisiones; no se realizó replay del solver.

El fallo posterior es UnicodeDecodeError por lectura ASCII bajo Nsight. Se adjunta un diff prospectivo con UTF-8 explícito y32MB por captura, conservando64MB total (alcanzaría para dos capturas del tamaño observado). No se aplicó al paquete original ni se relanzó el CNS. Falta además el importador Nsight2022.4.2; el Windows2024.6.2 instalado es otra versión. La traza qdstrm se preserva, sin inventar duraciones GPU exclusivas.

Consumo de los bloques CNS nuevos: 1096.676s de pared, 11332.154s CPU acumulada, RSS máximo 4.767GiB. Estos tiempos excluyen inicialización, exportación y empaquetado. Tres intentos, cero reintentos. 426fuentes y37referencias permanecen idénticas según INTEGRIDAD_FINAL.json.

Motivo de parada: presupuesto experimental acotado cumplido; diagnóstico útil con fallos conservados. Siguiente paso: reparar la captura y comparar dos arquitecturas de ejecución según PLAN_MOTOR_Y_WORKFLOW.md. No hay simulación neuronal activa.
