# Motor CUDA: continuidad funcional del 25 de septiembre

**PASA la comparación numérica completa de 100 ms.** Clasificación del núcleo CNS en este alcance:
CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA. No es admisión de navegación/vuelo ni prueba de todos los
modelos neurocientíficos o duraciones mayores.

## Qué quedó implementado

El controlador C++/CUDA recibe el extremo temporal autorizado. Los modelos
declaran explícitamente lado izquierdo/derecho en eventos, sin representar el
lado mediante nextafter. El integrador conserva el punto medio exponencial con
paso doble de la referencia y comparte su evaluación inicial idéntica: cinco
evaluaciones por intento en lugar de seis. Las operaciones elementales se
fusionan en CUDA, con FMA desactivado. Núcleo, modelo y evaluador están separados.

El núcleo acepta estados, ecuaciones target/rate, tolerancias y dominios por
variable; no contiene anatomía ni ganancias conductuales. Se validan buffers
contiguos y se rechazan operaciones solapadas antes de tocar el respaldo.
Esta ronda sustituye únicamente el ejecutor CNS; PN y membranas conservan sus
solvers, masas y ecuaciones. El modelo heredado conserva sus limitaciones.

## Prueba empírica

Dos corridas de 100 ms con el organismo completo, mismo checkpoint, 166.700
neuronas canónicas, 359.373 estados CNS y 25,582,938 pesos
originales. Olor lateral estático, precisión nominal sin afinamientos.
Se conservaron 101 muestras completas por trayectoria, eventos, estados finales
PN, cuerpo, axones y RNG. Ambos procesos terminaron con código 0 y sin fallos de
limpieza. Los umbrales del ensayo anterior se mantuvieron intactos.

| Comparación máxima | Diferencia | Límite |
|---|---:|---:|
| CNS normalizado | 0 | 1 |
| Voltaje espacial, mV | 0 | 2e-05 |
| Compuertas | 0 | 2e-07 |
| Tiempo de evento, segundos | 0 | 1e-9 |

Veredicto mecánico: **PASS_COUPLED_NUMERICAL**. Se compararon
7,788 registros de eventos,
incluidos predictores; no son esa cantidad de espigas físicas distintas.
Los detalles de todos los campos y propietarios están en PAIR100.json.

## Coste y alcance del ahorro

| Medida | Referencia | Candidata |
|---|---:|---:|
| Evaluaciones CNS | 112,620 | 93,850 |
| Intentos CNS | 18,770 | 18,770 |
| Avance completo, segundos | 637.988 | 421.063 |
| Proceso, segundos | 670.036 | 451.304 |

El recuento CNS disminuye 16.667 %. El tiempo observado de avance
cambia -34.00 %. **La GPU y CPU estuvieron compartidas y su carga
varió durante la pareja: estos tiempos no certifican una aceleración atribuible
al motor.** La eliminación de evaluaciones repetidas y la comparación numérica
se juzgan por separado de esa limitación temporal.

Se consumieron 1123.047 s de procesos
supervisados entre ambos brazos, de un máximo conjunto de 1.400 s. No hubo
tercera tolerancia ni una colección de candidatas. Los controles sintéticos
complementarios probaron lados/eventos en GPU, solución analítica, alias de
buffers, rechazos y rollback. No sustituyen esta prueba real.

## Decisión

Se conserva esta implementación como base CNS compatible en el alcance probado. La etapa de compatibilidad real de 100 ms queda cerrada.
El RK3(2) anterior queda como candidato histórico, con sus resultados positivos
y negativos intactos. Esta prueba no demuestra que el defecto de nextafter
explicara su diferencia de 100 ms. Tampoco se rebajan límites usando el ruido
biológico como excusa.

El siguiente escalamiento debe medir duración/heterogeneidad y rendimiento con
recursos controlados sobre esta base, conservando la separación de modelos.
No se añaden ahora FP32, poda ni algoritmos solapados. No se cambió la selección
del motor de la otra sesión.

## Revisión y reproducción

El usuario eligió la tarea «ChatGPT C++/Cuda» para el motor; la app la identifica
como Codex. Su revisión fue estática, sin GPU. Se resolvieron sus dos hallazgos
sobre memoria no contigua y llamadas solapadas antes de la pareja. No se crearon
subagentes ni se reutilizó para esta ronda la conversación de las etapas.

El ZIP pequeño permite compilar y comprobar el núcleo. El ZIP de evidencia
incluye las dos trayectorias y todos los estados necesarios para recalcular
PAIR100.json con verify_saved.py. Reejecutar el organismo requiere el modelo y
checkpoint locales originales. Los snapshots PN compactos no son reinicios
completos. Recalcular datos guardados no equivale a ejecutar nuevamente el organismo.

[Código y comandos](/mnt/c/Users/gonza/Documents/Codex/2026-09-25/pu/outputs/Motor_CUDA_continuidad_codigo.zip) ·
[Evidencia de la pareja](/mnt/c/Users/gonza/Documents/Codex/2026-09-25/pu/outputs/Motor_CUDA_continuidad_evidencia.zip)
