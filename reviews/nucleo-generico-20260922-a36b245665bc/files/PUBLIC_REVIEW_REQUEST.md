# Encargo de revisión del núcleo genérico

Revisar los archivos reales model.py, autodiff.py y runtime.py; después
validate.py, extensions.py, extension_test.py, review_falsifiers.py y verify.py.
PLAN.json fija tolerancias. CORE_FREEZE.json precede las extensiones de siete
estados y doce especies; no es una reserva ciega independiente.

El punto central: ¿la separación descriptor/compilador/ejecutor conserva las
mismas ecuaciones CPU/GPU y permite nuevas áreas sin reglas anatómicas? Buscar
fallos de captura CUDA, propiedad de buffers, diferenciación, duplicación de
paso, estados/clamps y transacciones. Hay dos perfiles numéricos FP64; no son dos
biologías ni equivalencia garantizada de espigas.

C (punto medio exponencial con separación diagonal local) ya falló HH en ambos
perfiles. No rescatar C con otra tolerancia ni interpretar ese fallo como
refutación de exponenciales/Krylov. A es RK4 adaptativo global residente; aún
no es el backend implícito general propuesto. B multirrate sigue sin implementar.

La carga grande contiene 726900estados y23296700conexiones, pero es sintética,
suave y sin espigas discretas, retardos, ruido o cuerpo. No es prueba del cerebro
completo ni de etapa3. La exactitud grande se evalúa en4096sondas/21instantes y
todos los estados finales; no en todos los estados en todo instante.

Distinguir hallazgo en código, propuesta, reproducción y resultado leído.
Indicar archivos efectivamente abiertos. Siguiente decisión propuesta por la
revisión previa: ARKStep/Ginkgo, masa dispersa constante por época y residuo
implícito global con JVP de las conexiones completas. No declarar que esa pieza
ya está implementada.
