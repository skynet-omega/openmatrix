# Hipótesis adicional: recuperación del paso después de un evento

Inspección de `native_hybrid_20260922/graph_control_v2.cpp`, sin cambiar el motor ni ejecutar otro organismo. En la corrida PN629 sham40+400ms, el recibo cuenta81.297 intentos CNS aceptados y4 rechazados;34.053 eventos físicos incluyen predictor y camino aceptado. Son conteos, no una atribución de tiempo. No respaldan que los rechazos por rigidez del CNS dominen esta corrida; la membrana tiene un controlador y conteos diferentes.

En `engine_advance_events`, `h=min(available, requested)` respeta el próximo evento. Tras aceptarlo, la propuesta siguiente se calcula desde `h`, incluso si el tamaño pequeño se debió sólo a ese corte. El mínimo impide quedar por debajo de `minstep`, pero no conserva la propuesta previa a un corte corto. Es una posible fuente de trabajo global adicional, aún no medida de forma causal.

Tres alternativas para esa decisión:

- A: conservar o recuperar la propuesta anterior cuando el corte, y no un rechazo, limitó el paso. Cada intento posterior mantiene la misma prueba de error, dominio y cortes obligatorios. Puede aumentar rechazos; no garantiza aceleración ni igualdad microscópica.
- B: elegir una propuesta según el error observado y distancia al siguiente evento, con un controlador matemático explícito. También requiere contrastar el error real y el coste de rechazos.
- C: conservar la política y reducir el trabajo espacial por intervalo mediante estados/forzamientos locales y cotas de influencia. Es una reforma más amplia, no una licencia para omitir eventos.

El discriminador inmediato es la sonda C++ pendiente: distribución de `h`, error, cortes y tiempo de grafo. Cualquier prototipo posterior debe tener presupuesto nuevo, fuentes congeladas, referencia y controles completos del organismo. No se ha modificado el algoritmo ni se atribuye una mejora de rendimiento a esta hipótesis.
