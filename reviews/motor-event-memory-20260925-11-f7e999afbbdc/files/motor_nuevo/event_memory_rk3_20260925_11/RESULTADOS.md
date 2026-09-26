# Motor RK3 con memoria de propuesta: resultado real

**Candidata conservada: comparación funcional de 1 s aprobada.**

La vida completa consumió **23.77 minutos de avance y 24.63 minutos totales**.
Margen de 25 minutos totales: **cumplido**; diferencia 0.37 minutos por debajo.
Metas de 20 y 12 minutos: 20 pendiente; 12 pendiente.
Una condición con olor y cuerpo, sin viento. No equivale a validación biológica general, navegación ni sustitución automática del motor estable.

## Cambio concreto

El controlador distingue un paso pequeño impuesto por un evento de uno exigido por el error. Tras un corte interior aceptado con error<0,1, la propuesta anterior actúa como suelo de la propuesta normal siguiente. Nunca se aplica al final de época ni ante rechazo. El siguiente intento vuelve a comprobar error y dominio.
Sólo cambió esa regla en `engine/resident_controller.cu`; `ENGINE.diff` muestra el porte. RK3, proyección izquierda/derecha, pesos persistentes FP32, estados FP64, PN, membranas, cuerpo y tolerancias permanecieron iguales. El motor07 y el estable no se editaron.

## Pareja controlada de 100 ms

| Medida | Motor07 | Candidata11 |
|---|---:|---:|
| Avance integral, s | 176.400 | 143.093 |
| Proceso completo, s | 219.155 | 185.096 |
| Evaluaciones CNS | 73444 | 45816 |
| Tiempo CNS, s | 87.073 | 54.374 |

Reducción observada: **18.88% de avance y 15.54% de proceso**. Pasó la puerta fijada de10% de ahorro de avance, proceso sin regresión y compatibilidad funcional; por eso se ejecutó una sola confirmación de1s.
El control reprodujo exactamente35/35campos del primer100ms del motor07 guardado previamente. Ambas ejecuciones usaron BLAS1, misma preparación y registro. Una pareja y orden fijo: no estima variabilidad ni elimina todo efecto de sistema/cachés. Los muestreos no encontraron otra simulación; durante el control apareció un verificador remoto de archivos. No se acredita exclusividad completa de CPU/GPU.

## Comparación funcional durante el segundo

- Criba frente a07: PASS; frente al estable: FUNCTIONAL_SCREEN_PASS.
- Integridad externa: INTEGRITY_CONFIRMED; filas completas, relojes CNS/PN/cuerpo a1ms, árboles finales presentes y finitos, presupuesto respetado. Estados iniciales de las tres corridas nuevas coinciden directamente.
- Mandos de giro diferentes: 0/1000; mando de avance exacto: True.
- Pose/cuerpo exacto: True; velocidades exactas: True; contactos exactos: True.
- Campos de traza exactos: 24/35. Máxima diferencia CNS a1s: 3.84315e-05; voltaje celular: 0.00587891mV.
- Eventos comprometidos estable/candidata: 51977/51977; bloques con distinta identidad: 2.
- Predictores descartados estable/candidata: 25981/25983; bloques diferentes: 4.
- Frente a07, mismos conteos totales por identidad comprometida: True.

Las diferencias neuronales y de eventos se conservan en los JSON; mismas órdenes no significan estados internos idénticos. La posición de un evento en un bloque vecino tampoco mide por sí sola su retraso físico. El criterio de mando incluye yaw y forward; esa aclaración se registró antes de ejecutar la candidata.

La auditoría externa `INTEGRITY1000.json` informa también diferencias de payload y cuántos eventos se excluyen del emparejamiento temporal. No añade umbrales de aceptación. La corrida07 histórica no guardó PN/publicación al inicio: se cotejaron su estado neuronal inicial y procedencia, y los árboles iniciales de la candidata contra el control nuevo. No se atribuye al histórico una observación inexistente.

## Coste y recursos

| Región en1s | Segundos |
|---|---:|
| CNS residente | 537.648 |
| Región celular | 266.548 |
| PN | 206.313 |
| Diferencia sin atribuir | 415.578 |

CNS: 113138 pasos aceptados, 26 rechazados y 452656 evaluaciones. La diferencia de tiempos no se atribuye automáticamente a Python: los temporizadores cubren regiones distintas y algunos incluyen esperas.
Presupuesto: pareja 404.251s/900s; confirmación 1477.627s/2400s. Exposición:1200ms nuevos en tres procesos, más fixtures sintéticos. La carga/archivo se informa dentro del proceso, fuera del avance.
La referencia estable y el segundo previo07 tienen tiempos históricos con condiciones distintas; sus cocientes no son una pareja de rendimiento controlada.

## Revisión y reproducción

ChatGPT Motor revisó diff/controlador y posteriormente código de evaluación y cifras. Sus hallazgos sobre registros incompletos motivaron la comprobación externa sin tocar el motor. Jev priorizó la medición integral. Sus revisiones no son ejecuciones independientes; véanse `CHATGPT.md`, `CHATGPT_DATOS.md` y `jev_01/response.json`.
La evidencia se recalcula mediante `python -O verify_saved.py --long`. El paquete de reproducción incluye código del núcleo, comparadores y datos necesarios para estas métricas. Reejecutar el organismo requiere el proyecto y checkpoint originales instalados: el paquete no se presenta como un simulador autónomo.

Se conserva la candidata para el alcance observado. La residencia conjunta de estados/puertos sigue como posible mejora futura, con coste por medir; no se mezclaron otras optimizaciones en esta ronda.
