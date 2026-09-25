# Etapa 2: membranas como modelo del núcleo

Limitación: el CNS está verificado a 100 ms, pero las membranas siguen ligadas a
un adaptador histórico que ensambla CUDA mediante sustituciones de texto y usa
importaciones transitivas implícitas. No hay una entrega limpia que demuestre
un segundo modelo con matriz de masa no identidad en la infraestructura común.

A: núcleo reutilizable de bloques independientes, solución densa FP64 con
residual del operador original y modelo celular declarado por separado (elegida).
Conserva método, masa, matrices, canales, eventos y política adaptativa; extrae
infraestructura en código explícito y elimina dependencias históricas del backend.
B: un integrador monolítico para todo el estado; su paso común puede quedar
limitado por el bloque más rígido. Falsador: el coste conjunto no compensa eliminar
las interfaces. C: agenda asíncrona por influencias/eventos; necesita una cota del
error de entradas mantenidas. No mezclar B/C en esta implementación.

Predicción falsable de A: CNS+membranas del nuevo backend conservan todas las
muestras reales y eventos bajo los criterios anteriores, y un modelo de masa
densa de dimensión distinta de 17 se ejecuta con el mismo núcleo sin editarlo.
El modelo de referencia es el control; no se normalizan pesos ni masas, no se
poda y no se usan tolerancias nuevas. PN queda conservado durante esta etapa.

Trabajo externo: una petición de desarrollo a la tarea escogida por el usuario
«ChatGPT C++/Cuda», que la app identifica como Codex; un panel Jev de prioridades,
sin delegaciones recursivas ni atribuir generación de código a Jev Choice.

Presupuesto prospectivo: una candidata; controles analíticos/fallos de hasta
180 s de proceso; una preprueba real de 1 ms <=120 s para detectar integración;
una pareja completa de 100 ms, <=900 s por brazo, total real <=1920 s; RSS <=18
GiB por proceso, incremento de VRAM <=8 GiB, evidencia nueva <=2 GiB. Hardware
compartido: los tiempos se informan, no certifican aceleración causal. Ningún
proceso ajeno se modifica o detiene. No se amplía el presupuesto al ver fallos.

Comparador conservado: CNS <=1 normalizado; voltaje 2e-5 mV; gates 2e-7; tiempos
de evento <=1 ns; estructura, enteros, RNG y relojes exactos. Criterios completos
en compare_real.py del cierre anterior. Tras la pareja se acepta el hito en su
alcance o se conserva el fallo. Nada de nuevas tolerancias o variantes sucesivas.
