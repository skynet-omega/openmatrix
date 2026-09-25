# Diagnóstico causal del contraste entre antenas

Estado: ambos brazos de 1 s terminaron, con salida correcta y sin errores de cierre. El [informe calculado desde las trazas](analysis_01/REPORT.md) muestra un error final hacia la fuente de **22,11678° con L/R originales** y **22,24512° con antenas igualadas**. La diferencia de **+0,128338°** favorece conservar el contraste en este modelo discretizado. Ambas condiciones empeoran desde los 18,37685° iniciales: **etapas 4 y 5 abiertas**.

Clasificación: **PROMETEDOR_NO_CONFIRMADO** para la contribución del contraste. El control coincide exactamente con el donante; los 588 arrays preparados de cada brazo coinciden con la referencia. El verificador local y el código proporcionado por ChatGPT recalculan el mismo efecto desde datos crudos; cada uno conserva resultados idénticos con Python normal y `-O`. ChatGPT no ejecutó los NPZ en su entorno. No hay todavía referencia refinada de ambos brazos a 1 s.

Presupuesto consumido: 3.739,031 s + 4.290,544 s = **8.029,575 / 8.600 s**, por debajo de los 4.300 s de cada brazo. Dos vidas de 40 ms de preparación + 1.000 ms de ensayo; salidas de aproximadamente 1,8 GiB. No quedan organismos activos en esta campaña. Se conservan los checkpoints completos localmente, sin afirmar que su reanudación esté validada. La contención de GPU registrada impide usar los tiempos de esta pareja para medir rendimiento del motor.

ChatGPT y Codex coinciden en analizar el lector y comparar dos vidas de 1 s antes de prolongar otra vida a 2 s. Jev priorizó este par de manera consultiva. Las tres alternativas, sus falsadores y el presupuesto están en `DECISION.md` y `PLAN.json`.

El código original de ChatGPT, SHA256 `21373acbd1323b52375dd5892d3a198996e4695fd42cfa879b9ba187255510ca`, se ejecutó localmente sobre la traza real de campaña 35. La cancelación temporal del mando fue 85,516 % y la cancelación en la resta bilateral de señales DN corregidas por baseline, 29,731 %. Son descripciones de la señal, no una causa demostrada. ChatGPT revisó fuentes y recibos pero no ejecutó nuestros NPZ en su entorno.

## Comparación fijada antes de ejecutar

1. `identity_01`: repetir exactamente L/R consumidos por el donante durante 1 s, desde la misma preparación de 40 ms. Deben coincidir todos los datos de la traza y los 588 arrays del preparado, incluido el operador.
2. `no_contrast_01`: sólo si el control pasa, asignar la media bilateral FP64 a ambas antenas, conservando tiempo, intensidad común, tercer canal, cerebro, lector, cuerpo y propiocepción.

Ambos brazos reproducen entradas grabadas y, por tanto, no prueban feedback espacial. Se conserva separadamente el campo gaussiano que encontraría el cuerpo en su posición real: `concentracion_fisica`. La columna heredada `concentracion_campo` contiene aquí la señal entregada por la cinta, como declara el contrato. No interpretar ésta como un campo leído en bucle cerrado.

Positivo en `error_sinD − error_identidad` significa que D ayuda al rumbo; negativo significa que D lo empeora en esta condición. La criba de tamaño 0,022° incorpora un margen tentativo de 0,002°, cuya suficiencia **no está validada a 1 s**. Toda interpretación es provisional hasta comprobar precisión en ambos brazos. El mínimo histórico de mejora de orientación 0,5°, desde preparación y desde 400 ms, sigue intacto.

El presupuesto máximo es 4.300 s por brazo y 8.600 s agregados, 18 GiB de RAM, 12 GiB de GPU y 3 GiB de resultados. Se detiene ante fallo del control, pérdida de apoyo, estado no finito o límite de recursos. No se utiliza reanudación no validada ni se retocan parámetros tras ver el resultado.

## Reproducción local

En este host, con los recursos históricos disponibles en sólo lectura:

```bash
cd /home/daroch/AXIOMA_ASTRA/campanas/etapa4_contrast_ablation_20260925_36
python3 test_replay.py
python3 -O test_replay.py
# Directorios de salida deben ser nuevos; no repetir corridas sólo para empaquetar.
/home/daroch/miniconda3/envs/GPU/bin/python -u run_replay.py --mode identity --out identity_01
python3 verify_pair.py --identity identity_01 --out IDENTITY_VERIFIED_01.json
/home/daroch/miniconda3/envs/GPU/bin/python -u run_replay.py --mode no_contrast --out no_contrast_01
python3 verify_pair.py --identity identity_01 --no-contrast no_contrast_01 --out PAIR_VERIFIED_01.json
python3 chatgpt_verificar_cintas_cd.py --donante ../etapa4_long_trajectory_20260925_35/native_minus_02/traces.npz --control identity_01/traces.npz --sinD no_contrast_01/traces.npz --campos ../etapa4_long_trajectory_20260925_35/CAMPOS.json --out EXTERNAL_VERIFIED_01.json
```

Para etapa 5 aún falta mostrar una ventaja del bucle cerrado frente a un control de reproducción bajo la misma perturbación, autoridad motora y preparación. La intervención actual mide el efecto total de igualar entradas: conserva su concentración media, pero modifica la actividad neural media por no linealidad e historia. No localiza una sinapsis defectuosa ni valida fisiológicamente el lector.

Siguiente decisión: preparar identificación independiente del lector actividad→mando, conservando como rivales el diagnóstico de transformación neural y la prueba online/replay con separación observable justificada. No invertir signos ni ajustar ganancia para hacer aprobar esta trayectoria expuesta. La revisión de Gemini y la respuesta real de ChatGPT están en [CHATGPT_GEMINI_REVIEW.md](CHATGPT_GEMINI_REVIEW.md); las alternativas y limitaciones, en [DECISION.md](DECISION.md).
