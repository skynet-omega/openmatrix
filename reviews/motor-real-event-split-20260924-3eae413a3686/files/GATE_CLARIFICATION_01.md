# Precisión y alcance antes del nuevo replay

24-09-2026. Aclaración prospectiva de `../architecture_round_20260924_01/MULTIRATE_NEXT_GATE_01.json`, antes de ejecutar la captura o un candidato. Conserva el presupuesto y la puerta de trabajo 10×. No modifica ningún resultado histórico.

La captura usa el mismo runtime y debe dejar idénticos todos los estados científicos serializados, RNG, entradas/eventos y contadores; los tiempos de construcción/medición y memoria auxiliar no son estado científico. El puerto cargado realmente debe ser `motor_nuevo/epoch_cost_20260923/event_ports.py`, con SET/ADD; se registra su hash.

Un método nuevo puede elegir estimador propio, pasos internos y número de rechazos/aceptaciones. No hereda el divisor 3 del midpoint ni debe reproducir sus ramas especulativas. En el replay, los eventos grabados son entradas exógenas exactas; esta exigencia no se extiende a tiempos de eventos endógenos de futuros algoritmos.

Para este bloque normalizado, la comparación al endpoint físico es `max(abs(x_new-x_ref)/(1e-7+1e-5*max(abs(x_new),abs(x_ref)))) <= 1`, sobre cada coordenada de estado CNS; además se exige finitud, dominio declarado, evolución exacta de los puertos impuestos y defecto acoplado entre muestras. Ésta es cercanía a la referencia del modelo, no una cota contra la solución exacta. Cualquier modelo con otra unidad necesita sus propias escalas explícitas. El coste incluye auditoría, reintentos y fallback; un ensayo corto aprobado sólo permite otro ensayo, no etapa4/5.

El modo rápido del organismo se evaluará con un contrato funcional nuevo por brazo y errores por clase de estado. No exige un motor universal perfecto antes de explorar; tampoco convierte error numérico en ruido biológico. `3,42e-7°` fue una diferencia observada de giro, mientras el tope prospectivo vigente era `0,002°`.
