# Contrato del ejecutor causal — ronda del 22 de septiembre

El núcleo recibe bloques de estado, ecuaciones y puertos declarados por adaptadores. No selecciona tratamientos por nombre anatómico. El adaptador espacial aún corresponde al modelo heredado de 17 voltajes y 68 compuertas: no representa cualquier modelo neuronal. La prueba adicional de tres estados de decaimiento utiliza el mismo controlador sin editarlo; el propagador afín admite dimensiones 1–32 y se prueba también con siete estados.

## A: reloj local dentro de GPU

`block_runtime.hpp` administra propuesta, error, rechazo, confirmación y reloj por bloque. Sólo se admite independencia dentro de la época recibida: el predictor termina a 62,5 µs y el avance aceptado a 125 µs. Se conservan las fronteras del acoplamiento original. Todas las 17 coordenadas de una célula siguen eléctricamente acopladas; no son 17 modelos independientes.

El controlador consulta al modelo, no a la CPU, en cada ensayo interno. Cada bloque conserva físicamente sus eventos en ambas medias etapas después de aceptar. Estado, filtros, detector, contadores y eventos permanecen privados. Se publica al propietario únicamente después de comprobar que todos los bloques terminaron sin error. La ganancia axonal queda fijada por el propietario al inicio; no se realimenta desde bloques que terminaron antes. No se admite un publicador arbitrario: el adaptador comprueba la clase auditada. La prueba de operador tiene un constructor de fixture explícito, separado de la instalación normal.

No hay geometría comprimida en esta candidata: utiliza las 51 matrices originales y conserva el umbral de residuo 1e-12. La cota aritmética de agrupación anterior sigue pendiente de demostración y se evita como dependencia.

La independencia y la confirmación transaccional se prueban ejecutando una célula sola, en un lote, con orden invertido y después de un ensayo fallido. La semántica de eventos tiene una limitación heredada: el detector observa picos muestreados. El falsador de ChatGPT demuestra que dos rejillas pueden terminar con idéntico voltaje y distinto número de espigas. Las tolerancias de voltaje no son una prueba de topología de eventos. Se conservan controles de conteos exactos; no se promueve fidelidad universal.

## B: respuesta afín de eventos

`affine_ports.cu` propaga el estado receptor completo mediante la acción de la exponencial matricial entre cambios y saltos cronológicos. Con una coordenada constante se representan entradas afines. Permite consultas de prefijo, incluidas marcas fraccionarias de nanosegundo. Aplica el salto en la frontera de consulta, con convención continua por la derecha.

La acción se evalúa con Taylor de grado 20, subdividiendo para que la norma infinito de cada matriz escalada no supere 0,5. La cola en aritmética exacta está acotada por exp(0,5)·0,5^21/21! por la norma del estado de entrada. Esa cota no es una certificación completa de redondeo FP64; los contrastes usan `scipy.linalg.expm` y fórmulas cerradas. Existen topes de trabajo explícitos, errores ante no finitos y publicación sólo de un resultado completo.

La prueba incluye el evento tardío omitido por el método anterior, constantes de tiempo iguales y casi iguales, prefijos y dos conductancias no conmutativas. Cambiar el orden de sus inversiones produce la diferencia (1−exp(−g))²(E2−E1), aun con idénticas áreas. El propagador conserva ese efecto.

**B no sustituye el CNS real en esta ronda.** El CNS contiene tanh, conductancias dependientes del estado, recurrencia y varias rutas heredadas. Integrar exactamente un puerto lineal no resuelve el error de congelar esos receptores. Mantener las fronteras de eventos es el control operativo actual. La próxima decisión debe establecer un contrato de flujo/error por receptor, o una partición por dependencias que evite detener toda la red ante eventos locales.

## C: Jacobiano acoplado y Schur

La anterior RA34PW2 usó bloques cruzados nulos y queda descartada sólo en esa forma. La alternativa conserva todas las compuertas y elimina únicamente sus correcciones lineales. Para bloques de una etapa A,B,C,D, se resuelve (A−BD⁻¹C)δv = rv−BD⁻¹rg y δg=D⁻¹(rg−Cδv). Con 17 voltajes y 68 compuertas locales, D es diagonal si lo permiten las cinéticas declaradas; un modelo nuevo puede requerir otra estructura.

Esta ruta podría reducir factorizaciones pero requiere ensamblar derivados y evaluar cuatro etapas. No se implementa un tercer prototipo para rescatar el anterior. Con A ejecutándose, las membranas ya no son el principal coste: Schur por sí solo no elimina el trabajo de recorrer el grafo completo en cada evento ni el coste PN. Se conserva como rival, no como descartado por analogía.

## Operador y reanudación

`operator_state.py` conserva exactamente los arrays efectivos registrados por un adaptador, incluidas copias host/CUDA que pueden diferir. Captura, valida disposición y hashes, y repone valores; no vuelve a calcular medias ni a multiplicar pesos. La reconstrucción heredada omitía tau/theta preparados. Tres caminos (continuo, carga fría, fallo seguido de reconstrucción) comparan el estado neural completo después de 1 ms. La sesión corporal fallida sigue bloqueada. No se implementó recuperación automática del organismo ni un checkpoint corporal nuevo.

## Procedencia matemática

Ingeniería de métodos conocidos, no novedad biológica. [SUNDIALS y múltiples sistemas independientes](https://arxiv.org/abs/2405.01713) discute compromisos de agrupación y ejecución GPU; no demuestra esta implementación ni su rapidez. [NVIDIA, primitivas de warp](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) fundamenta las barreras explícitas para compartir memoria dentro del bloque. [PETSc RA34PW2](https://petsc.org/release/manualpages/TS/TSROSWRA34PW2/) es la procedencia del rival Rosenbrock. ChatGPT revisó las fuentes anteriores y sugirió falsadores; no ejecutó CUDA.
