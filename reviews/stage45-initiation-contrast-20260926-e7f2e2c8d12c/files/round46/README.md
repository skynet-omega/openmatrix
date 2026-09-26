# Ronda46 — contraste lateral pendiente completado

**Resultado: DESCARTADO_EFECTO_MATERIAL_100MS. Etapas4/5 abiertas.** Dos ramas de120ms,240ms neuronales nuevos, conservaron el motor original y las cintas/criterios de43/44. El STOP44 permanece intacto.

El contraste virtual−common cambia la respuesta de ORN, PN y DN, pero no supera ninguno de los dos mínimos prospectivos del lector no aplicado:

| Medida | Observado | Mínimo anterior | Brecha |
|---|---:|---:|---:|
| Integral absoluta de diferencia de yaw crudo | 0.0000165° | 0,001° | 60.6 veces por debajo |
| Magnitud de media últimos50ms | 0.0002869°/s | 0,02°/s | 69.7 veces por debajo |

El signo neto fue negativo. El relé calculado da la misma secuencia entre ramas. Este es un negativo de efecto material bajo el protocolo y ventana registrados, **no ausencia total de sensibilidad, ni prueba de que nunca pueda haber respuesta más tardía**. No se alarga la ventana ni se modifica ganancia para rescatar el resultado.

Controles completos: estado inicial y prefijos frente al sham histórico iguales; cuerpo y fuerzas subpaso idénticos; entradas efectivamente consumidas con media emparejada; contexto no olfativo, retina, propiocepción y RNG emparejados. Comparación local por `verify_pair.py` original. El cuerpo sigue una cinta externa común y no recibe el nuevo mando neuronal; no acredita arranque, navegación ni recuperación física.

Recursos: 994.4s de pared agregada, 6521.2sCPU de procesos neuronales; topes1800/8000s. Common=483.2s y virtual=486.0s. El tiempo CPU del verificador en el supervisor no se suma al contador de procesos neuronales; el tiempo de pared final sí lo incluye. RAM y recursos de procesos vigilados en vivo; no reintentos. No se modificó modelo ni protocolo44.

[Plan previo](PLAN.json) · [resultado](RESULTADOS.json) · [contabilidad](QUEUE.json) · [revisión ChatGPT](CHATGPT.md) · [informe conjunto y decisión](../../investigacion/avance_etapas45_20260926_02/INFORME.md).

La salida pública es un subconjunto de observaciones, código y recibos; los checkpoints completos permanecen locales. El cierre no convierte un verificador CPU o una revisión de texto en una nueva réplica biológica.

**Decisión:** no lanzar otra vida larga con esta transferencia. La prioridad es discriminar balance neuronal/reclutamiento, lector de acción e interfaz corporal con evidencia independiente. No elegir a posteriori la neurona más activa ni ajustar umbrales hasta caminar. Orientación cerrada y perturbación se ensayan después de obtener una señal propulsiva y una señal de giro útiles y atribuibles.
