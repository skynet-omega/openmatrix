# Contrato propuesto: runtime nativo para un organismo heterogéneo

**Estado:** diseño, sin implementación ni prueba del organismo. Elaborado el 24-09-2026 mediante inspección de fuentes. La ruta conservadora propuesta mantiene primero las ecuaciones, la agenda física y la precisión declarada por cada campo del padre; cualquier integrador alternativo requiere un contrato y una confirmación propios. «Unificado» significa que un ejecutable C++ posee el reloj, la memoria, las transacciones y la planificación de *todos* los módulos, aunque MuJoCo siga calculando física en CPU mediante su API C. El intercambio CPU↔GPU de puertos de cuerpo/sensores en el borde de 1 ms sigue siendo necesario mientras se use MuJoCo CPU; eliminar Python del bucle no equivale a eliminar ese intercambio.

## Inventario del organismo vigente

| Dueño actual | Estado y operaciones que posee | Borde visible |
| --- | --- | --- |
| CNS recurrente (`hybrid`) | Vector FP64 `state`, topología/pesos, coeficientes `target,rate`, liberación y `next_step_ns`. `NativeGraph` captura un CUDA Graph con midpoint exponencial y comparación paso completo/dos medios; la decisión de aceptación, límites y cortes de evento está en C++ host. | [organism_adapter.py](../../campanas/etapa3_motor_nuevo_20260922/organism_adapter.py), [graph_core.py](../../campanas/etapa3_motor_nuevo_20260922/graph_core.py), [graph_control_v2.cpp](../native_hybrid_20260922/graph_control_v2.cpp), [gpu_coefficient_layout.py](/home/daroch/AXIOMA_FLYWIRE/matrix/src/gpu_coefficient_layout.py). |
| Productores físicos KC/APL y membranas | Estado espacial, voltajes, compuertas, calcio, corriente y filtrado; publican q/s y eventos LIF o gamma. La física de membrana tiene su propio criterio de aceptación. El predictor de medio bloque se descarta y se restauran los estados antes del avance aceptado. | [block_midpoint.py](/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922/block_midpoint.py), [event_coupling.py](../../campanas/etapa3_pn629_intervention_20260923_15/event_coupling.py), [device_cell.py](../../campanas/etapa3_pn629_intervention_20260923_15/device_cell.py). |
| PN de masa y salidas eléctricas | Voltajes, compuertas, cargas y calcio; historias de entrada y salida, aceptaciones del solver. El adaptador vigente importa/exporta estado para cada avance PN; el estado físico PN y el vector CNS deben tener un solo publicador cada uno. | [pn_execution.py](../../campanas/etapa3_motor_nuevo_20260922/pn_execution.py), [block_midpoint.py](/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922/block_midpoint.py). |
| Eventos y puertos | Filtros q/s con decaimiento analítico entre eventos, saltos `ADD` y publicaciones `SET(post_q)` ordenadas. Se requieren cortes de integración en eventos físicos; exceder la capacidad actual de 8 eventos por célula y bloque falla explícitamente. | [event_waveform.py](../../campanas/etapa3_pn629_intervention_20260923_15/event_waveform.py), [event_ports.py](../../campanas/etapa3_pn629_intervention_20260923_15/event_ports.py), [graph_control_v2.cpp](../native_hybrid_20260922/graph_control_v2.cpp). |
| Mundo, cuerpo, transducción y efector | MuJoCo y controlador de fuerza por contacto son CPU. La pose determina geometría de antenas y concentración; la señal disponible al inicio del intervalo de 1 ms se consume entonces. La liberación descendente leída **antes** del `core.step()` manda el siguiente intervalo corporal según el contrato heredado; la fuerza sintética se aplica en `mj_step`. | [antennal_world.py](/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage4_antennal_contact_adapter_20260915/antennal_world.py), [antennal_runtime.py](/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage4_antennal_contact_adapter_20260915/antennal_runtime.py), [contact_runtime.py](/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage2_contact_cns_20260915/contact_runtime.py). |
| Sesión y persistencia | Relojes de cerebro/cuerpo/mundo, sensores y acción pendientes, estado del controlador, RNG, identidad de fuentes y checkpoint. El acoplamiento vigente se instala con parches de clases y bloqueo de una sesión por proceso. | [runtime_session.py](../pipeline_review_20260922/runtime_session.py), [run_pipeline.py](../pipeline_review_20260922/run_pipeline.py), [operator_state.py](../causal_runtime_20260922/operator_state.py). |

El reloj externo de esta preparación es **1 ms**. Ocho bloques físicos de **125 µs** realizan cada uno predicción descartada de 62,5 µs y avance aceptado de 125 µs; no son 16 microintegraciones fijas del CNS. El controlador CNS registra pasos adaptativos interiores y cortes de evento adicionales. El perfil real de 19 ms atribuyó 67,2 % de la pared a `NativeGraph.advance`, incluyendo espera GPU; ese dato **no** descompone kernels y host. [Evidencia de coste](../epoch_cost_20260923/README.md), [reconstrucción de 1 ms](RESULT_01.json).

## Contrato mínimo del runtime

Una instancia de runtime posee `Clock(int64 ns, sub_ns_event_fraction)`, `DeviceArena`, `ModelRegistry`, `EventQueue`, `Scheduler`, `BodyPort` y `CheckpointWriter`. Los modelos se registran mediante descriptores de datos y puntos de extensión; el núcleo nunca contiene los nombres PN, KC, retina o DNb05. Se rechaza explícitamente cualquier modelo que no declare capacidad numérica suficiente. Un nuevo cerebro puede añadir poblaciones, estados, conexiones, conductancias y lectores sin editar el planificador.

```cpp
struct StateField {
  Id id; Unit unit; DType dtype; Shape shape; Bounds bounds;
  OwnerId sole_writer; MemorySpace placement; ErrorScale error;
};
struct Port {
  Id id; Unit unit; Shape shape; OwnerId publisher;
  Phase sampled_at; Phase visible_from; Reduction reduction;
};
struct Event {
  TimeStamp when; uint64_t sequence; PortId dst;
  enum { Add, SetPostState, ChangeParameter } operation; Payload value;
};
struct Model {
  Descriptor schema();
  TrialResult trial(TimeStamp t0, TimeStamp t1, ReadOnlyState,
                    ReadOnlyPorts, ScratchState&); // no commit
  EventSpan emitted_events(ScratchState const&);
  void commit(ScratchState const&, OwnedState&);
  void snapshot(Writer&) const; void restore(Reader&);
};
```

`Descriptor` fija versión, unidades, FP64/FP32 permitido por campo, identidad de filas/columnas, CSR/CSC y pesos, mecanismo de evaluación (`target-rate` exponencial, ODE explícita, masa/implícito, flujo exacto de filtro, evento discreto), grafo de dependencias, dominio y escala de error. Su compilador genera un **plan de ejecución por época** que concatena operaciones compatibles y conserva barreras impuestas por retroalimentación, predictores y eventos. Una dependencia recurrente obliga a releer el estado de la etapa correspondiente; declarar localidad o frecuencia lenta sin cota no autoriza omitirla. Para el legado el bloque `target-rate` evalúa $x_{n+1}=x_n+(1-e^{-h r(x_*,u_*)})(a(x_*,u_*)-x_n)$, con el mismo punto medio/estimador que el padre. El descriptor permite añadir RHS $M(x,t)\dot x=F(x,u,t)$, pero **no** convierte por sí solo un solver implícito en fiable: exige residual, convergencia, eventos y tolerancias específicos de ese modelo.

Orden de una época de 1 ms, entendido como contrato observable: (1) leer los puertos sensoriales y de comando *pendientes* de la época previa; (2) iniciar transacción de todos los dueños y completar ocho bloques físicos con predicción/restauración y eventos ordenados; (3) dentro de cada bloque, integrar CNS con cortes en eventos y decisión adaptativa; (4) publicar salidas neuronales/efector con la latencia declarada; (5) llamar a MuJoCo y al mundo, muestrear y dejar sensores pendientes para la época siguiente; (6) confirmar todos los estados y relojes o marcar la sesión fallida. El planificador del prototipo 1 conserva incluso las operaciones que parecen redundantes hasta que la paridad de estados y eventos permita simplificarlas. La agenda ordena `(tiempo físico, fase, propietario, secuencia)`; eventos coincidentes mantienen el orden de emisión comprobado en el padre. Un `SET(post_q)` prevalece según ese orden, y no se reduce a una suma; desbordamiento de cola y no finitos fallan sin descarte ni clipping.

La interfaz `BodyPort` es C++↔MuJoCo C y transporta sólo comandos, observaciones y señales sensoriales declarados, con unidades y estampas de tiempo. La GPU retiene CNS, grafo, estados físicos neuronales, pesos y scratch. Se transfieren al host los puertos necesarios para la física y, en puntos de muestreo elegidos, telemetría de diagnóstico; la traza completa no se descarga cada milisegundo. Python queda en preparación, lanzamiento/análisis fuera del bucle y no decide pasos. Un `checkpoint` verificable incluye vector completo y estados de módulos, pesos/parámetros efectivos, agenda y secuencia de eventos, `next_step_ns`, reloj, RNG, buffers pendientes, estado del cuerpo/MuJoCo y del controlador, identidad/hash de modelos y software. El importador de snapshots históricos debe verificar equivalencia campo a campo y abortar ante un dueño ambiguo.

El modo **full** conserva FP64, orden de acumulación y tolerancias del contrato actual. El modo **fast** es una política separada de precisión/integración, admitida sólo contra errores medidos de estado, eventos, puertos y conducta en casos heterogéneos. La selección fast/full se congela por corrida y aparece en el checkpoint; nunca se etiqueta una diferencia de modo como continuidad numérica exacta.

## Tres falsadores medibles previos a promoción

1. **Fidelidad de una misma vida.** Desde idéntico snapshot, medir por época todos los estados propiedad de cada módulo, eventos `(tiempo, fila, operación, post)`, aceptados/rechazados y próximos pasos, puertos, sensores usados/pendientes, cuerpo y reloj. Una inversión `SET/ADD`, desbordamiento oculto, latencia de efector distinta, no finito o ausencia de reanudación exacta descarta el prototipo 1. Las tolerancias numéricas se registran antes de correr y no se amplían al ver un fallo.
2. **Rendimiento del organismo completo.** Medir pared, GPU activa/espera, copias, pico RAM/VRAM y energía aproximada para el mismo prefijo heterogéneo con y sin telemetría, con warm-up separado. Si mover sólo el controlador no reduce la pared neta o sacrifica memoria/precisión, no se promociona; el objetivo 1 s simulado/60 s real exige hoy ≈52,5× respecto al prefijo de 19 ms, no extrapolable desde kernel sintético. [Base medida](RESULT_01.json).
3. **Generalidad y eventos difíciles.** Cargar sin cambiar el núcleo un modelo visual/conductancia con estados y dominio distintos, otro cerebro pequeño con distinta topología, y un motivo recurrente con evento tardío/`SET` y cruce rasante. Deben conservar residual/error y tiempos de evento contra referencia fina, fallar explícitamente ante descriptor incompleto y mostrar coste por módulo. Si el compilador exige un parche PN/KC/visual, o sólo funciona sobre una red homogénea, falla la pretensión de motor general.

## Migración acotada: dos prototipos como máximo

**P1, runtime residente conservador.** Exportar snapshot y descriptores inmutables desde el padre; crear un ejecutable C++ que administra reloj, sesiones, plan compilado, checkpoints y MuJoCo C. Portar primero las operaciones de coeficientes y el controlador adaptativo a CUDA/C++ con semántica sin cambio; incorporar PN/KC/visión como modelos registrados, eventos `ADD/SET` y puerto corporal de la misma vida. Benchmark escalonado: descriptor ajeno → 1 ms real → 20 ms real → episodio suficiente para responder una pregunta de etapa 4. Criterios previos: paridad de eventos/propietarios y pared completa, no sólo `NativeGraph.advance`. Los resultados negativos se conservan. El código vigente se usa como referencia/donante, no se sustituye hasta pasar esos gates.

**P2, sólo si P1 demuestra coste residual dominante y paridad.** Elegir *una* reforma matemática según medida: (A) compilar/fusionar mayor parte de la evaluación recurrente, (B) multirritmo/eventos con cota de defecto y retorno recurrente, o (C) integración implícita para una rigidez demostrada. No mezclar las tres en un primer benchmark. Congelar ecuaciones, interfaz, tolerancias y control padre; validar motivos adversarios y organismo heterogéneo. Un resultado favorable de P2 no sustituye la admisión científica de etapas 4–5, que exige experimentos de feedback y navegación.

La recomendación inmediata es P1 como experimento de arquitectura. El cuello observado está repartido entre la llamada al grafo recurrente y el resto del organismo: si el nuevo plan sólo acelera una de esas mitades, la meta de minutos seguirá abierta. P1 debe reducir **coste de evaluación global y coordinación** sin alterar la latencia causal de eventos/puertos. Todo factor de aceleración es una hipótesis hasta medirlo en la misma preparación.
