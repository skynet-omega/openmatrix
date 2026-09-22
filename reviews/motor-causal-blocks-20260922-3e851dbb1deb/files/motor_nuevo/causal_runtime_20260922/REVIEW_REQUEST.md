# Revisión acotada de la nueva ejecución CUDA

Lee `RESULTADOS.md`, `VERIFIED.json`, `ARCHITECTURE.md`, `device_cell.py`,
`device_cell.cu`, `block_runtime.hpp`, `operator_state.py` y sus comprobadores.
Los arrays completos de las comparaciones están en el ZIP por partes; no
atribuir validación independiente a una lectura del informe. Declara qué
archivos pudiste leer/ejecutar y qué no.

La pérdida de tau/theta sugerida por ChatGPT se reprodujo tanto en carga fría
como después de un fallo. Tras restaurar el operador efectivo, los tres estados
neurales al cabo de 1ms son idénticos. El cuerpo fallido permanece bloqueado.

A ya ejecuta ensayos adaptativos y confirmación de eventos por bloque dentro
de un único lanzamiento CUDA por época. Operador original completo, sin
agrupación de bases; reloj por bloque, fronteras de comunicación conservadas,
ganancia axonal fijada, barrera de publicación y tope de trabajo. Modelo
adicional de3estados sin editar el núcleo. Misma célula sola/lote/permutada y
tras rechazo: estados y eventos comparados exactamente.

Organismo20ms: referencia82,82s de avance, candidata60,32s. Membranas28,06→5,38s.
Criba normalizada4,29251e-5, compuertas5,87559e-5, límite1e-4; conteos/relojes
exactos. No se ejecutó1s y no se alcanzó la meta. El falsador de pico muestreado
se conserva: no hay garantía universal de eventos por tolerancias de voltaje.

B es un propagador separado de bloques afines y respuesta del receptor;
incluye eventos tardíos, marcas fraccionarias, prefijos, constantes casi iguales,
no conmutación y modelo7estados. No está integrado en el CNS no lineal.

Busca errores concretos de sincronización/confirmación/registro de operador.
Luego propón un único discriminador barato para decidir la siguiente ronda
macro: partición por dependencias de eventos frente a integración de puertos
con cota de respuesta no lineal. No recomendar más trabajo de membranas sin
justificarlo por su fracción actual del tiempo. Schur acoplado sigue como tercer
rival. No dar PASS general al motor por las pruebas cortas.
