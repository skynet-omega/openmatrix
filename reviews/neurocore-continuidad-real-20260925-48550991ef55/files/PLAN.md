# Continuidad del motor: una candidata, una comparación real

Se conserva la referencia y todos los resultados anteriores. El usuario pide
soluciones estables y terminar etapas, no ajustes sucesivos de tolerancias.

Decisión: método de punto medio exponencial con estimación por paso doble,
idéntico al método de referencia, compartiendo la evaluación inicial que ambos
caminos calculaban dos veces. Son cinco evaluaciones por intento, antes seis.
El controlador CUDA sigue siendo independiente del modelo. El paso exponencial
admite ecuaciones y'=rate(y,t)*(target(y,t)-y); no se afirma que cualquier
ecuación neuronal tenga esta forma ni que se hayan sustituido PN y membranas.

El tiempo de evaluación es un valor de dispositivo junto con lado LEFT/RIGHT
explícito. El controlador conserva el extremo autorizado del intento; ninguna
proyección reconstruye ese extremo ni codifica el lado con nextafter. Se mantiene
FP64, pesos, ecuaciones, tolerancias originales, acoplamiento y política adaptativa.
Sólo se fusiona álgebra elemental por estado y se elimina una evaluación idéntica.

Alternativas consideradas: A preservar método y eliminar trabajo repetido
(elegida); B continuar RK3(2) y aislar su error global (conservada, no otra candidata
en esta ronda); C precisión mixta/dispersión (requiere justificación de error y
coste, no se mezcla aquí). La corrección temporal es necesaria en A/B.

Presupuesto: controles analíticos y de transacción CUDA de hasta 60 s de proceso;
dos corridas principales como máximo, referencia y candidata, 100 ms cada una,
olor lateral real, nominal sin refinamiento. Hasta 700 s por proceso y 1.400 s
sumados, 18 GiB RSS por proceso, 8 GiB de incremento de VRAM, 2 GiB de evidencia.
La GPU puede estar compartida: no se certificará exclusividad ni una aceleración
universal. No se detienen ni se modifican procesos de la otra sesión.

Se usa compare_real.py sin alterar sus umbrales históricos: CNS <=1 normalizado,
voltaje 2e-5 mV, gates 2e-7, eventos 1 ns; estructura, enteros, RNG y relojes
exactos; todas las muestras y estados PN finales. No se cambia el contrato para
aprobar. El coste y la equivalencia se juzgan por separado.

Parada: tras la pareja se conserva o rechaza esta implementación con una decisión
de alcance explícito. No tercera tolerancia ni cadena de variantes. Un fallo de
implementación concreto se conserva y se repara con una comprobación mínima;
no se amplía automáticamente el presupuesto científico.
