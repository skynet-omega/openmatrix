# Revisión acotada elegida por el usuario

Destino: tarea «ChatGPT C++/Cuda», id `01a0d769-1b01-7de1-b12e-43bb30763413`.
La app identifica ese destino como Codex. No se creó un subagente ni se utilizó
la conversación de etapas para esta ronda. El revisor recibió la implementación
y la referencia, efectuó una revisión estática y no ejecutó GPU ni pruebas.

La revisión confirmó que compartir el coeficiente inicial conserva
algebraicamente el método para el adaptador leído: copia de target/rate antes
de reutilización, proyección idempotente y tiempos interiores. El extremo exacto
y RIGHT en la proyección final son coherentes con punto medio: no existe una
evaluación de coeficientes en el extremo final.

Dos hallazgos P2, ambos resueltos antes de correr el organismo:

1. Los kernels planos no respetan strides arbitrarios. El núcleo ahora rechaza
   coeficientes/proyecciones que no sean CuPy FP64 contiguos.
2. Respaldo/control/estado compartidos podían sufrir llamadas solapadas.
   La interfaz ahora excluye advance/close desde antes del respaldo; se rechaza
   una segunda operación. El ABI C exige propiedad serializada del handle.

Ambos contratos se comprobaron en `check_runtime.py` bajo Python `-O`, además
de los controles analíticos/de eventos/rollback. La revisión no certifica
equivalencia del organismo, rendimiento ni que la fragilidad de nextafter causara
el FAIL histórico de RK3(2).
