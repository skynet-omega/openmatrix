# Primer pulso: diferencia concreta entre la rama PFG histórica y el pipeline protegido

ChatGPT priorizó aislar el pulso olfativo consumido en el primer intervalo del experimento PFG. Antes de presupuestar ocho corridas de 335 ms, se cargó **una vez** el organismo del pipeline actual, sin dar ningún paso. [Plan](PLAN_HISTORY_LOAD_PROBE.json), [script](inspect_history_load.py), [recibo](history_load_01/RESULT.json). Carga 12,20 s; mismo checkpoint `settled_700ms`, SHA256 de manifiesto `1817df128eb1f79953a6e45a24863465b6fe6c59b71fa0ddd6af4edde541ba15`.

Los tres vectores, identidad del checkpoint y hashes de ambas trazas están en [HISTORY_PORTS.json](HISTORY_PORTS.json); se comprobaron las igualdades y el pulso esperado mediante aserciones explícitas.

| Preparación | Primer vector olfativo consumido |
|---|---|
| Rama PFG histórica, cuatro brazos | `[0,59088072, 0,58233249, 0]` |
| Pipeline protegido actual, carga antes de `step()` | `pending_sensors=[0, 0, 0]` |
| Pipeline protegido actual, smoke real de 1 ms | `sensores_usados=[0, 0, 0]` |

La rama PFG histórica realmente consumió ese pulso; la carga del pipeline actual lo limpia durante su preparación. **No sería un experimento válido comparar directamente la rama PFG vieja con el pipeline nuevo y atribuir las diferencias sólo al pulso**, porque también cambió el motor/ensamblaje. Tampoco tiene sentido «eliminar» de nuevo el pulso pendiente del pipeline actual: ya es cero. La opción B continúa abierta para otras dependencias de historia (los estados iniciales ORN/PN no son neutros), pero el pulso específico de ChatGPT no es el próximo discriminador en el pipeline protegido.

Decisión operativa: priorizar A, localizar la inversión direccional dentro del **mismo pipeline protegido**, con sham/uniform y un contraste numérico frente a referencia/refinamiento antes de interpretar conducta. C, comprobar mando espejo desde un estado corporal idéntico, sigue como control. Si se quiere probar la hipótesis exacta del pulso viejo, se requiere un toggle presente/ausente en **la misma** rama PFG y mismo checkpoint, no mezclar ramas ni presupuestar una campaña larga antes de verificar ese contrato.
