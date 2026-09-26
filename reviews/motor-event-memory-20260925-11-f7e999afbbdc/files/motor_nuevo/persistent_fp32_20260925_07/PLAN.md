# Matriz FP32 persistente, organismo real — 25-09-2026

Se ejecuta B de la revisión de viabilidad: datos estáticos convertidos una vez,
posiciones mutables declaradas por el adaptador y refrescadas en GPU justo antes
de cada consumo. Se conserva el mismo CSR FP32, integrador RK3, ecuaciones,
parámetros, eventos, cuerpo, preparación y registro del control mixto anterior.
El núcleo de memoria no conoce clases neuronales. A (coordinación residente) y
C (menos evaluaciones con otro método) siguen alternativas, no se fusionan aquí.

Un kernel indexado lee pesos efectivos FP64 actuales y escribe sólo posiciones
PN/APL/diagnóstico declaradas. Los consumidores especializados mantienen su
fuente FP64 original. Plasticidad explícita invalida toda la copia mediante un
refresco completo fuera del grafo, sólo cuando ocurre ese cambio; se conserva el
puntero capturado. Cambiar almacenamiento requiere reconstrucción, no silencio.

Presupuesto previo: una prueba pequeña GPU de cambios/restauraciones/grafo,
una preprueba real de 1 ms con auditoría de pesos (máximo 120 s), una pareja
control/candidata de 100 ms (máximo 600 s/proceso). Segunda pareja en orden inverso
sólo si la primera conserva funcionamiento y reduce al menos 20% el avance.
Máximo 4 corridas de 100 ms y 1 de 1 ms, 2520 s de procesos del organismo,
18 GiB RSS/proceso y 12 GiB VRAM. Corrección de un bug de implementación permite
repetir sólo su preprueba fallida dentro del mismo techo agregado, conservándola.
Las mediciones largas esperan a que termine la simulación ajena; no se detiene.

Fidelidad: la representación debe producir los mismos pesos FP32 efectivos que
el control mixto en cada consumo auditado; ejecución completa, finita, reloj
correcto y comparación guardada de estados neuronales, PN, eventos y cuerpo
contra ese mismo control. No se investiga precisión adicional contra FP64, se
cambian umbrales, ni se reabre reinicio. El contrato numérico existente se aplica
una vez a la pareja; no se afina para pasar.

Utilidad: >=20% menos tiempo total de avance, con preparación y proceso total
informados aparte. Hitos 150/120/72 s por 100 ms equivalen a 25/20/12 min/s sólo
como proyección. Una mejora parcial no acredita la meta. Si B falla utilidad,
se cierra B, sin variantes pequeñas. La decisión sobre A requiere los costes
integrales observados y un mecanismo concreto. No se sustituye el motor estable.
