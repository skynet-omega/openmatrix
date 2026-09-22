**Elijo B ahora: reparar el pipeline y ejecutar un único discriminador causal del coste CNS. No lanzaría una campaña larga de etapa 3 con el runner publicado, ni comenzaría otra arquitectura todavía.** No es volver a exigir un motor perfecto: hay discrepancias concretas entre la variante ensayada, la que instala el piloto y lo que realmente cubre su aceptación.

**Leí completos** `run_pilot.py`, `organism_adapter.py`, `verification_vendor/compare.py` y `verify_transport.py` del commit `5f3327…`. **No ejecuté código, CUDA ni tus nuevas corrupciones.**

## Qué cambia la decisión

**El runner no transporta al piloto la variante más reciente.** Instala `kc_adaptive`, no `device_cell`, y llama a `organism_adapter.install(h,events)` sin activar `event_boundaries`, cuyo valor predeterminado es `False`. Por tanto, ejecutar ese archivo no comprueba la variante con relojes locales y fronteras de eventos que acabáis de medir.  

También confirmo los problemas de finalización: el `finally` puede escribir `RESULT.json` en un directorio existente después de fallar `load`; `KeyboardInterrupt` no entra en `except Exception`, de modo que puede quedar registrado un estado de fase con `error=None`. **No declara automáticamente éxito, pero deja un recibo ambiguo y permite alterar evidencia anterior.** 

**Las corrupciones que comunicas son coherentes con el verificador leído:** calcula diferencias para muchas hojas, pero sus límites numéricos no incluyen `kc_spatial_state.delta` ni `body_final.qvel`. La capa adicional exige estructura, identidades y relojes, no límites para esos arrays continuos. **Comparar todas las hojas no equivale a calificar todas las variables.** Mantendría el contrato histórico intacto y corregiría cualquier interpretación anterior más amplia.  

## Comparación de las tres opciones

| Opción | Dictamen |
|---|---|
| **A. Diagnóstico etapa 3 muy limitado** | Útil después de reparar el runner. Ahora aprovecharía los registros existentes descriptivamente, sin atribuir el fallo conductual a una ruta biológica ni iniciar cuatro corridas largas. |
| **B. Reparación + discriminador causal CNS** | **Prioridad inmediata.** Resuelve trazabilidad y permite decidir qué reforma del motor merece trabajo, sobre el organismo real. |
| **C. Arquitectura alternativa general** | **Condicionada al resultado de B.** No reescribir por insatisfacción con los tiempos ni sumar otra especialización sin identificar qué trabajo elimina. |

## Los tres mínimos para cerrar B

**1. Un ejecutor inequívoco y seguro ante fallos.** Reservar exclusivamente un directorio nuevo antes de cualquier escritura; no tocar una salida anterior si falla la carga. Registrar interrupción, fallo y finalización como estados distintos, conservando el último intervalo confirmado. El preflight debe comprobar **las implementaciones efectivamente instaladas**, las fronteras de eventos, el operador efectivo y sus identidades; no solamente los archivos presentes. Mantener bloqueada la sesión corporal fallida.

**2. Cobertura de aceptación explícita, sin recalificar el pasado.** Separar `criba_histórica` de `calificación_del_piloto`. Antes de interpretar conducta, añadir controles prospectivos para el voltaje espacial y el estado dinámico corporal pertinente, además de las señales entregadas al lector y actuadores. Los límites deben tener unidades y fundamento; **no copiarlos de otro campo ni escogerlos para admitir la trayectoria observada**. Lo no cubierto debe figurar como **no evaluado**, no como aprobado. Esto no exige certificar toda la fisiología para registrar un piloto.

**3. Una sola medición que decida la reforma macro.** En una época real aceptada con eventos, distinguir los cortes debidos a eventos, precisión, máximo permitido y fin del intercambio; separar trabajo predictor del confirmado. Después estimar qué parte del operador efectivo debe actualizarse por cada evento, incluyendo recurrencia y rutas especializadas.

Si predominan cortes por eventos con dependencias confinadas, probar partición causal. Si alcanzan prácticamente todo el operador, investigar respuesta de puertos con cota no lineal. Si predominan otros motivos, **no implementar ninguna de esas dos soluciones por anticipado**. Es un discriminador, no otra batería abierta de optimización.

## Cuándo volver a A

**B termina cuando esos mínimos están comprobados y produce una decisión, no cuando desaparecen todos los problemas del motor.** Entonces sí haría un piloto acotado del organismo durante segundos, con presupuesto previo y alcance explícito; un prefijo interrumpido sigue siendo evidencia parcial, no el ensayo completo.

**No convertiría la meta de 1 s/60 s en requisito científico para empezar ese piloto. Pero tampoco aceptaría que un runner desactualizado o un `screen_pass` incompleto sustituyan la comprobación del transporte. La secuencia recomendada es B breve y cerrada → A interpretable; C únicamente si la medición la justifica.**
