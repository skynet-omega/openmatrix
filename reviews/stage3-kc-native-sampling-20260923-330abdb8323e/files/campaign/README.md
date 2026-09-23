# Muestreo KC: captura nativa y reproducción sobre entradas reales

Etapa 3 abierta. Confirmado localmente sólo el diagnóstico acotado: captura on/off exacta en ambos motores y ocho comprobaciones de rollback por corrida. 4 cargas completas de 1 ms, 144.950 s de pared agregada; sin modificación de pesos ni decodificador.

Las dos primeras entradas físicas (predictor y aceptación) de las cuatro KC son idénticas entre motores. La captura almacena valores ya calculados dentro de CUDA, antes de actualizar la memoria del detector; no reevalúa ecuaciones. Dos épocas de 125 us contienen cadencias distintas: KC76431 usa pasos aceptados 6250 ns frente a3125 ns, y KC544736 puede usar25000 ns mientras la referencia usa3125 ns. En marcas temporales exactamente comunes, máximo error de voltaje 6.68534966e-05 mV y q 2.65965124e-05. Los incrementos y trough pueden diferir mucho más; un máximo detectado en distinto instante reinicia el trough en momentos diferentes. No se demostró inocuidad motora a400 ms.

El primer evento somático divergente permanece en KC76431 a1.482100000 s frente a1.482098437 s absolutos. El reloj del ensayo CNS y el reloj de la membrana son distintos; conservar ambos sin mezclar sus orígenes.

`matched_replay.py` compila WARP original y la variante usada por el controlador independiente (__device__ más unroll); ocho casos con entradas reales y pasos idénticos, incluyendo dos medios pasos, coinciden a1e-12 y conservan decisiones de aceptación. Debilita el desacuerdo aritmético como causa del primer evento en estas cuatro KC. No certifica todos los estados ni todos los tiempos.

Alternativas: A muestreo/detector recibe apoyo directo; B distinto error de trayectoria por control local sigue posible y requiere convergencia/impacto funcional; C fuga del predictor queda debilitada en las ocho épocas observadas, no universalmente excluida. El gate histórico de1e-4 no se modifica.

ChatGPT recibió las fuentes del controlador para proponer una corrección o contrato causal; respuesta todavía pendiente al escribir este cierre. Jev completó una consulta de clasificación de tareas, sin evaluación numérica. Paquete con fuentes, entradas seleccionadas, trazas y verificadores; no incluye los snapshots integrales (~359 MB cada uno), por lo que no permite repetir todo el organismo fuera del entorno local.

Reproducir análisis compacto: `python3 -B analyze_trace.py`; reproducción matemática GPU: `python3 -B matched_replay.py` (salida nueva, CuPy y CUDA). Los controles de estado completo se verificaron localmente con `compare_capture.py`, cuyos snapshots no se incluyen en el compacto.
