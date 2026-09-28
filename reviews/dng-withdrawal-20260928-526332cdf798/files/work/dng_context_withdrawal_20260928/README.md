# Retirada directa hacia DNg100: contraste cerrado

28-09-2026. **DESCARTADO como solución directa suficiente en la cobertura observada. Etapas 4/5 abiertas.**

Reutilizamos diez capturas de la campaña 49 y el componente de contexto ya registrado. La hipótesis era que retirar la recepción negativa descendente externa bastaría para abrir ambas DNg100. Se fijaron cinco casos antes del cálculo; no se combinaron grupos ni cambiaron parámetros tras observarlos. No se inició una nueva vida CNS.

## Resultado reconstruido

Cada caso conserva 2.820 evaluaciones emparejadas del integrador y 80 puntos de alineación entre neuronas, de dos historias y cinco ventanas. No son animales, semillas ni observaciones independientes. Los valores son entrada ponderada nativa del modelo, no voltios, corriente o velocidad. Los máximos de la tabla son descriptivos por neurona; el criterio utiliza ambas en el mismo instante, nunca mezcla máximos.

| Caso | Mayor margen 10045 | Mayor margen 10056 | Ambas positivas |
|---|---:|---:|---:|
| Identidad | -1331.923462 | -1234.830078 | 0/2820 |
| Retirar recepción negativa descendente | -362.051758 | -303.917847 | 0/2820 |
| Retirar recepción positiva descendente | -1377.873169 | -1295.647583 | 0/2820 |
| Retirar todo el grupo descendente | -408.001221 | -364.735352 | 0/2820 |
| Retirar recepción negativa ascendente | -1124.007935 | -979.556824 | 0/2820 |

Incluso la retirada total de los términos negativos descendentes deja ambas neuronas bajo umbral. La retirada ascendente tampoco abre el margen. Identidad y control de signo se verificaron; la fila completa se recalculó en el orden FP32 del consumidor. Se conservaron todos los valores, tiempos, pesos, capacidades y máscaras. El cálculo consumió **2.612 s CPU**, con 0 CNS/GPU.

Resultados originales: `runs/workbench/units/54826ab8b87d528c8e27261660fd2e93fbed34363c4eecabb3a2d9c5e5585bf3/attempts/23ddd927ccec45808af1d4fb440e18e4`. `withdrawals.csv` conserva los contrafactuales; `assessment.json` y `REPORT.md` se generan desde ellos. El ciclo `3acb677795ab4f15adb1c4f44418c2e3` quedó DECIDED/NEGATIVE con validez comprobada. Las cuatro tablas del análisis anterior permanecen idénticas byte a byte (`PARENT_REGRESSION.json`).

## Decisión y límites

Se cierra esta familia de retiradas directas con resto fijo. No se lanza una simulación para repetir el mismo mecanismo insuficiente ni se recorren más clases buscando un resultado favorable. El nulo no excluye efectos recurrentes, otros estados, los intervalos no capturados o contexto corporal natural. Tampoco convierte los signos del operador en signos receptoriales demostrados.

La prioridad pasa a identificar la escala de transferencia y el umbral: véase [decisión y tres alternativas](DECISION.md). La revisión de procedencia encontró cero contradicciones entre los signos consumidos de 2.289 aristas y la anotación usada para construirlos; eso es consistencia interna, no validación biológica independiente (`PARAMETER_PROVENANCE_CHECK.json`).

El recibo original conserva explícitamente `UNVALIDATED_WHOLE_CNS_VOLUME_SCALING_HYPOTHESIS`: volumen de segmentación normalizado por la mediana de todo el CNS aumenta theta y reduce gain. No equivale a medir área de membrana, resistencia o capacitancia. No se elimina esa normalización ni se retoca el umbral para producir apertura. Con gain positiva, cambiar sólo gain no cambia el signo de `net + drive − theta`; tau afecta la trayectoria, no ese cruce instantáneo.

## Revisión y reproducción

ChatGPT ASTRA_V2 revisó el discriminador y buscó fuentes primarias B1/B2/B3. ChatGPT_Motor_V2 revisó la inferencia y distinguió margen, ganancia y dinámica. Sus respuestas se conservan en esta carpeta: son revisiones conceptuales, no reproducciones de los arrays ni modo PRO verificado. La fuente primaria consultada y sus límites están en `DECISION.md`. No se descargó un dataset nuevo para repetir información disponible.

Motor C++/CUDA realizó la revisión operativa en paralelo: [informe](../workflow_efficiency_20260928_02/README.md). Corrigió consultas que adquirían bloqueo de escritura y liberación incompleta del bloqueo del índice tras excepciones. Sus seis regresiones se incorporaron al conjunto operativo. Las comprobaciones de integración, publicación y reproducción de esta entrega se consignan en `CIERRE.json` al concluir; las pruebas no admiten etapas.

Presupuesto científico previo: 300 s CPU, 4 GiB RAM, 128 MiB de salidas nuevas; no entrenamiento ni CNS. La revisión operativa tuvo su propio límite de 300 s CPU y 15 minutos. Fuentes anteriores en `before/`; contrato y fecha de congelación en `CONTRACT.json` y `FROZEN.json`. La receta reutilizable es `config/workflows/dng-context-withdrawal.json`.

Motivo de parada: discriminador preregistrado resuelto negativamente y familia directa acotada; falta identificación independiente para la siguiente hipótesis, no más variantes de parámetros del mismo contraste.
