# Motor nativo: resultado de la ronda del 22-09-2026

**La meta de un segundo simulado por minuto real NO se ha alcanzado. Etapa 3 sigue abierta.**

La medición del organismo refuta el reparto de costes atribuido por Gemini: cuerpo más captura ocupan aproximadamente 1,30% del paso; eliminarlos por completo sólo permitiría 1,013× en ese régimen. Las membranas espaciales, PN y el operador CNS dominan. GPU real: RTX 4070 Ti SUPER, 16 GiB.

## Mediciones reproducidas desde recibos

| Ejecución | ms completos | Proceso total (s) | Media pasos 2–5 (s/ms) | Criba corta |
|---|---:|---:|---:|---|
| baseline_01 | 5 | 28.660 | 2.8198 | PASA |
| native_01 | 5 | 28.229 | 2.4312 | PASA |
| compressed_01 | 5 | 25.563 | 1.8450 | PASA |
| ros_01 | 5 | 28.960 | 2.7327 | NO PASA |
| certified_01 | 0 | 16.845 | — | INCOMPLETO |
| certified_02 | 5 | 25.147 | 1.8648 | PASA |
| aligned_01 | 5 | 32.954 | 3.6305 | PASA |

A, con control C++ y bases agrupadas más cota respecto del operador original, mejora 1.512× la ventana estable del mismo método. Es una medición única de 5 ms, no una extrapolación validada de un segundo. El arranque y compilación están incluidos en la columna de proceso; no se suman a la media estable.

## Decisiones e interpretación

- **A conservada provisionalmente.** Control adaptativo y confirmación de eventos en C++/CUDA; reducción de 51 a 15 bases geométricas conservando todos los canales, inversiones, compuertas y coordenadas. Diferencia geométrica relativa máxima 6,50e-16. El residuo incluye una cota de agrupación dependiente de los coeficientes actuales. El intento con una cota uniforme excesivamente conservadora se conserva como fallo. El nombre `certified_*` identifica ese experimento de cota; no significa certificación del motor.
- **B descartada en esta implementación.** RA34PW2 con Jacobiano aproximado por bloques usa más subpasos y no mejora el coste; el error normalizado global 1,09126e-4 supera 1e-4. No se aflojó el criterio. Esto no descarta Rosenbrock con un Jacobiano acoplado completo: la aproximación diagonal por bloques aquí probada es más simple que la propuesta de Schur de ChatGPT.
- **C corporal/telemetría no priorizada**, por su coste medido. El contraste adicional de actualizaciones de rango bajo encuentra rango 16 en dimensión 17 para tres combinaciones: el rango 15 del espacio de bases no autoriza una reducción barata de la resolución.

## Fiabilidad y limitaciones accionables

ChatGPT encontró fallos reales del verificador: forma/tipo de historias, identidad PN anidada, reloj PN y tiempo de trazas. Las cinco corrupciones ahora se rechazan, también bajo Python -O; se conserva la comparación de historias con segmentaciones distintas. Las comparaciones anteriores de izquierda se revalidaron sin relajar sus límites.

El evento tardío fue reproducido en CUDA: el método heredado y su transporte daban z=0 cuando el valor analítico es 0,00226209, con estimador cero. Terminar subpasos en los eventos reduce el error a 5,89e-8 en ese caso y también admite marcas fraccionarias de nanosegundo. La variante `aligned_01` pasa la criba del organismo y la referencia fina, pero cuesta más: **la mejora rápida no puede presentarse como solución fiable general mientras conserve el defecto temporal compartido**.

El grafo nativo revierte exactamente su estado privado tras un fallo posterior a un subpaso aceptado y continúa igual al control. En el organismo, el cargador frío restaura exactamente el estado neural serializado; se bloquea reutilizar los punteros anteriores. La sesión corporal fallida ya rechaza continuar. **La reconstrucción manual del runtime neural no reproduce la continuación dentro del límite de ejecución**; se conservan arrays y diferencias en `recovery_04`. La causa de esa discrepancia sigue pendiente. No se autoriza recuperación automática ni se promueve el motor como estable.

El límite Newton=1 ya evita el grafo especulativo de dos correcciones; una entrada PN real confirma que respeta ese presupuesto. No cambia las corridas normales, que solicitan ocho.

El controlador nativo y el compilador de bases son reutilizables; el adaptador de membrana probado sigue describiendo 17 coordenadas y cinéticas heredadas. No es todavía una arquitectura general terminada para cualquier cerebro. Los ajustes PN/DNb05, lector motor y prótesis de la preparación se conservaron; ninguna comparación aquí los valida biológicamente.

## Autocrítica y siguiente decisión

La expectativa anterior confundió rendimiento de una carga sintética con el organismo. También era insuficiente comparar contra un método que comparte el mismo defecto de eventos. La revisión externa cambió decisiones concretas; no constituye reproducción independiente de CUDA.

La próxima ronda debe atacar el acoplamiento y el coste numérico completo, con tres rivales: (A) sesión unificada con propietarios y reanudación verificables, fusión de operadores e intercambio residente; (B) integración multirritmo que incorpore momentos de puertos de eventos sin detener globalmente todas las variables por cada espiga; (C) Jacobiano acoplado y eliminación de compuertas en el sistema lineal, conservando todos sus estados. Antes de implementar, cada ruta necesita un falsador, presupuesto nuevo finito y un máximo de dos prototipos. No prolongar simulaciones caras de etapa 3 para ocultar este límite.

La ronda actual termina tras comparar sus dos prototipos completos y documentar los defectos; no por haber terminado el motor. Los intentos de recuperación fallidos forman parte de la evidencia. Jev sólo clasificó tareas en la llamada previa de esta sesión; no emitió un dictamen científico. No se usaron subagentes Codex.

El método B y sus coeficientes proceden de [PETSc RA34PW2](https://petsc.org/release/manualpages/TS/TSROSWRA34PW2/), [fuente oficial](https://petsc.org/release/src/ts/impls/rosw/rosw.c.html). Licencia y procedencia incluidas.
