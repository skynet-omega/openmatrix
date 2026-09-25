# Consulta externa: continuar Neurocore con evidencia real

Objetivo del usuario: concretar un motor general de neurociencia a bajo nivel,
limpio, rápido en hardware local y sin sesgos conductuales dentro del núcleo.
No una nueva colección de algoritmos o microbenchmarks. Otra sesión trabaja
el organismo estable. Esta consulta no modifica esa sesión ni sus procesos.

Este paquete contiene código candidato, código de referencia pertinente,
los seis recibos, progresos y eventos, los comparadores y las tres comparaciones
ya calculadas. No contiene los 1,63 GB de trayectorias y estados completos ni
el checkpoint/modelo completo: permite revisión del código y de los registros,
no recomputar de forma independiente todos los máximos neuronales reportados.
La evidencia completa se conserva localmente. No confundir publicación con
reproducción de CUDA u organismo por ChatGPT.

## Hechos que debe contrastar el revisor

- Mismo organismo real: 166.700 neuronas, 359.373 estados CNS y 25.582.938 pesos
  originales. Cuerpo y sensores ejecutados; olor lateral estático, no pluma.
- Núcleo genérico RK3(2) FP64, 4 RHS por intento, controlador CUDA residente.
  Referencia: integrador exponencial de coeficientes congelados / paso doble,
  6 RHS por intento. Sólo se sustituyó CNS; PN y membranas espaciales siguen
  sus solvers originales. No es todavía reemplazo integral de todos los solvers.
- 20 ms sham: candidata 104,466 s, referencia 98,785 s; comparación PASA.
- 100 ms olor: candidata 287,895 s, referencia 363,363 s; comparación FALLA.
- 100 ms con rtol/atol CNS diez veces menores EN AMBAS: 292,664 / 371,029 s;
  comparación FALLA. Ambas parejas de 100 ms consumen aproximadamente 21 %
  menos tiempo en la candidata. GPU sin exclusividad instrumentada; no son
  réplicas idénticas de rendimiento ni prueba de ganancia universal.
- Refinada: error CNS máximo normalizado 3,248455 (límite 1), diferencia
  máxima absoluta CNS 1,383115e-6, voltaje espacial 2,81492e-5 mV (límite
  2e-5), compuertas 4,505567e-7 (límite 2e-7), evento 9,898693 ns (límite 1).
  Cuerpo, sensores, axones y 44 campos PN pasan; estructuras, conteos y RNG
  coinciden. 7.788 registros de eventos incluyen predictores, no sólo espigas.
- Peor error CNS en coordenada 58.613 a 89 ms. Nominal: 3,510266; refinada:
  3,248455. Referencia nominal -> fina: 0,524255; candidata nominal -> fina:
  0,461080 en la misma escala de comparación. El único refinamiento no separa
  todavía la causa. No hay prueba de que la referencia sea verdad exacta.
- Los límites se fijaron antes de correr, pero esta campaña no demostró su
  necesidad biológica. Un fallo del contrato no demuestra inutilidad del
  método; tampoco la coincidencia corporal prueba equivalencia neuronal.
- En candidata nominal CNS residente: 146,665 s de 287,895 s de avance total;
  el resto consume 141,230 s. Perfil no exclusivo.
- Seis corridas principales completaron sus datos; candidata fina devolvió
  143 después de COMPLETE y de escribir muestras/estado íntegros; causa
  pendiente. Las otras cinco terminaron en 0. No afirmar cierre limpio de esa.

## Preguntas

Formula tu recomendación propia antes de contrastar mi propuesta. El usuario
aporta otra opinión: deberíamos considerar funcional la mejora del 21 %, que
8 nV por encima del límite no justifican detenerse, y pasar a FP32 y
event_sparse/poda para ganar 100-200 %. Examina lo acertado y lo no demostrado
de esa opinión y también de mi evaluación. No necesitamos defender a Codex.

1. ¿Qué estado práctico darías a este motor: usar experimentalmente, corregir
   antes de sustituir el estable, rediseñar o abandonar? Fundamenta sin usar
   ruido biológico como sustituto automático de un presupuesto numérico.
2. ¿Qué límites son demasiado estrictos o faltan para un motor general?
   Distingue verificación numérica, observables científicos y equivalencia
   biológica. Si cambiarías el criterio, define uno prospectivo y justificable,
   preservando los resultados históricos, no sólo elevándolo hasta aprobar.
3. Revisa graph_runtime.py, resident_controller.cu, real_model.py, graph_core.py,
   organism_adapter.py y event_ports.py: ¿hay una causa concreta demostrable
   de la discrepancia o sólo hipótesis? Distingue defecto de implementación,
   error global/control de eventos e insuficiencia del método de referencia.
4. Elige UNA próxima intervención y máximo dos pruebas reales que decidan
   algo. Precisa qué conservar, qué tocar, qué medir y cuándo terminar. Si
   basta análisis offline, priorízalo. Evita rondas ilimitadas de tolerancias.
5. Compara aislar la discrepancia actual, cálculo disperso incremental exacto,
   y precisión mixta selectiva. ¿Dónde invertirías a continuación? Distingue
   poda anatómica de evitar recomputación exacta y no prometas 2-3x sin medir.

Mi propuesta provisional (expuesto ya a los resultados): conservar la candidata
ejecutable y su mejora como ingeniería experimental; localizar primero la
discrepancia 70-90 ms con los datos existentes y, si hace falta, un replay corto
desde estado común con forzamiento común. Decidir con ese resultado si se
corrige implementación/acoplamiento o se necesita un contrato nuevo de precisión
basado en uso. Después elegir UNA optimización respaldada por costes medidos,
sin añadir FP32 y dispersión a la vez. Cuestiónala si bloquea avance útil.

Respuesta acotada, aplicable. Declara archivos realmente leídos, cálculos y
código ejecutados, y qué no pudiste verificar. Una propuesta de revisor no
autoriza cambiar ganancias biológicas ni intervenir otra sesión.
