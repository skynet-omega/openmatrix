# Campaña 50 cerrada: patrón temporal insuficiente

**Etapas4/5 abiertas.** Las ocho condiciones terminaron y no alcanzaron los criterios previos de efecto material. [Resultados y alcance](RESULTADOS.md), [figura](RESULTADOS.png), [decisión](DECISION.md), [reproducción](REPRODUCIR.md).

Se conserva una preparación expuesta, entrada de igual cantidad por antena y un contraste factorial prospectivo. No se cambió el circuito ni se buscó otra ventana para obtener un resultado favorable.

## Diseño ejecutado

Ocho condiciones de 140 ms parten del mismo estado guardado. Conservan la dosis por antena y el circuito; cambian el orden temporal de la entrada. El diseño factorial permite separar una interacción entre lados de dos respuestas independientes, incluso cuando cada una tiene memoria.

`PLAN.json` y `PROPUESTA_PREVIA.md` se fijaron antes de iniciar. La cola ejecutó una condición a la vez, con detención prevista ante un fallo o el límite de recursos. Las ocho finalizaron: `QUEUE_RESULT.json`.

No es todavía una prueba de navegación: el contrato mantiene el giro neural sin aplicar y no introduce viento. DNb05 sigue siendo el lector; DNa02 sólo se observa. Un resultado positivo justificaría el ensayo posterior de orientación con feedback, sin aprobar por sí mismo las etapas 4/5.

El campo heredado `fase=cola_OFF` describe el mundo sensorial antiguo, que permanece apagado. La intervención terminal ORN está definida por `TEMPORAL_OWNER.json` y por las tasas realmente consumidas en `input_and_observers.npz`. No interpretar ese rótulo heredado como ausencia de la intervención de esta campaña.

Los estados finales usan el esquema explícito `temporal50_scientific_state_v1`; el cargador de 48/49 debe rechazarlos hasta incorporar el propietario temporal. Se conserva el cuerpo, RNG y resto de propietarios científicos.
