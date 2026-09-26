# Revisión C++/CUDA y rendimiento — 25-09-2026

Alcance: motor `event_memory_rk3_20260925_11/engine`, padre 07 y dependencias que realmente carga `run_trial.py`. Lectura de fuentes y resultados; **ningún benchmark GPU ni vida nueva**. Se ejecutó únicamente una prueba CPU de orden de llamadas. No se modificaron 11, 07 ni históricos.

**Resultado:** se detectó una dependencia de stream ausente en la inicialización PN y se corrigió en las copias de `12/pn`, por instrucción del coordinador. La corrección y su selección de importaciones pasan pruebas CPU; su validación GPU queda a cargo del coordinador. No se encontró otro bloqueador CUDA concreto del recorrido ordinario. Los 1.477,627 s existentes no prueban ausencia de carreras ni explican por sí solos sus cuellos de botella.

## Hallazgo material C1 — ordenar la inicialización del árbol PN antes de su warmup

**Prioridad P2, defecto estático de sincronización; fallo GPU no reproducido.**

- [resident_cyclic.py:6](/home/daroch/AXIOMA_ASTRA/motor_nuevo/resident_pn_20260922/resident_cyclic.py:6) crea `order`, índices, coeficientes, ceros y gathers en el stream corriente, hasta la línea 11. En líneas 14 y 23–24 crea un stream `non_blocking=True` y ejecuta `launch()` en él sin dependencia respecto de los productores.
- El primer `cp.add(base_d, diagonal, out=d)` de [resident_cyclic.py:16](/home/daroch/AXIOMA_ASTRA/motor_nuevo/resident_pn_20260922/resident_cyclic.py:16) ya puede consumir datos pendientes. Después los kernels consumen `steps`, `pairs`, `core` y gathers. También puede competir una inicialización tardía con escrituras del warmup. Un índice todavía no transferido podría causar acceso inválido; un cero tardío podría alterar un resultado temporal. No se observó ninguno de esos efectos en GPU durante esta revisión.
- Es alcanzable: [pn_execution.py:16](/home/daroch/AXIOMA_ASTRA/campanas/etapa3_motor_nuevo_20260922/pn_execution.py:16) → `graph_step` → [graph_stage.py:22](/home/daroch/AXIOMA_ASTRA/motor_nuevo/resident_pn_20260922/graph_stage.py:22) construye ese árbol. La sincronización de `StageGraph` en su línea 30 ocurre **después** del warmup problemático. La sincronización del stream del árbol tampoco espera trabajo pendiente del stream productor.
- El fallback `resident.Backend` importa otro árbol con el mismo patrón en [cyclic_tree.py:6–23](/home/daroch/AXIOMA_ASTRA/motor_nuevo/pn_abc_20260922/cyclic_tree.py:6). Aunque hubo cero fallbacks en la vida medida, debe reparar también esa ruta; es el mismo defecto de orden.

CuPy 13.6 documenta que [`asarray(blocking=False)`](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.asarray.html) encola H2D asíncrono en el stream corriente y que un [`Stream(non_blocking=True)`](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.cuda.Stream.html) no sincroniza con NULL. No se puede usar la latencia incidental de compilación/carga de `RawModule` como contrato de orden: una caché caliente o cambios de carga pueden variar esa latencia.

**Remedio concreto:** después de terminar las inicializaciones y antes de entrar en `with self.stream`, sincronizar el stream productor. Es setup, no el bucle caliente. Alternativamente registrar un evento productor y esperarlo, o realizar inicialización y warmup íntegramente en el mismo stream. Preservar fuentes congeladas y registrar el hash de la dependencia corregida en 12.

**Evidencia CPU:** [check_pn_stream_order_cpu.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/check_pn_stream_order_cpu.py) ejecuta el constructor original con un espía de CuPy; no importa CuPy real. [PN_STREAM_ORDER_CPU.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/PN_STREAM_ORDER_CPU.json) registra el primer consumidor `base_d`, productor `caller`, consumidor `tree_nonblocking`. El control inserta la sincronización **sólo en memoria** y no deja lecturas sin dependencia. El número de lecturas del espía no es un conteo de fallos reales ni una medición de riesgo.

Comando ejecutado:

```bash
/home/daroch/miniconda3/envs/GPU/bin/python -B /home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/check_pn_stream_order_cpu.py
```

**Reparación implementada, 26-09:** [resident_cyclic.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/pn/resident_cyclic.py:23) y [cyclic_tree.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/pn/cyclic_tree.py:22) añaden una sola barrera del stream productor antes del warmup. El único otro cambio adapta la ruta a los mismos tres archivos `.cu` congelados. [PROVENANCE.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/pn/PROVENANCE.json) identifica originales, copias y kernels por hash. No cambia la aritmética ni se añade sincronización por paso.

[install_pn.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/pn/install_pn.py) precarga ambos nombres que importan los módulos heredados; rechaza instalar después de cargar un consumidor o repetir la instalación. Uso una vez por proceso fresco, antes de `RuntimeSession` y antes de `resident`, `graph_step` o `graph_stage`:

```python
sys.path.insert(0, str(HERE / 'pn'))  # HERE = carpeta full_pipeline_review_20260925_12
from install_pn import install as install_pn_overlay
result['pn_overlay'] = install_pn_overlay()
session = RuntimeSession(h, 'causal_cuda')
```

La prueba de orden fue repetida sobre **los dos archivos reparados del disco**, ambos sin lecturas sin dependencia en el modelo CPU. [check_pn_import_cpu.py](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/check_pn_import_cpu.py) y [PN_IMPORT_CPU.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/PN_IMPORT_CPU.json) comprueban que las inserciones posteriores del path heredado conservan las copias y que una instalación tardía falla antes de cambiar bindings. Tampoco importan CuPy real ni construyen una simulación.

## Cobertura del recorrido y garantías observadas

| Tramo real | Evidencia y conclusión |
|---|---|
| Carga/restauración | `run_trial.py:75–115` restaura antes de instalar/capturar. `restore_prepared.py:49–53` aplica pesos/plasticidad/máscara y restaura el operador efectivo antes de crear el espejo. Las mutaciones de preparación no dejan el espejo obsoleto porque aún no existe. |
| Captura CNS | `real_model.py:31–63` instala `FastCSR`; `organism_adapter.py:23–41` congela buffers de frontera/PN y conserva `CoefficientBuffers`. `graph_runtime.py:71–93` sincroniza productores, inicializa y captura en su stream privado. Las copias RHS devueltas por `coefficient_buffer_brain.py:44–50` evitan alias entre las cuatro derivadas RK. |
| Vida de punteros CNS | `GraphRK23` conserva estado, stages, reloj, tolerancias, módulo y pool privado. `FastCSR` conserva buffers/espejo. Las temporales de captura quedan respaldadas por el pool privado; no hay `free_all_blocks` ni reutilización externa de ese pool en el recorrido. Esto depende de mantener el pool hasta destruir el grafo, como hace la clase. No trasladar temporales al pool global sin ownership explícito. |
| Grafo residente C++ | `resident_controller.cu:155–185` tiene limpieza en error de construcción, copia el child graph, instancia/upload y sincroniza antes de usar. `202–213` sincroniza antes de destruir recursos. `prepare → child → decide → commit → relaunch` tiene dependencias explícitas. El tail launch espera la terminación del entorno anterior y el retorno sincronizado host engloba sus descendientes; este orden coincide con la [guía CUDA 12.1 de device graphs](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-c-programming-guide/index.html#device-graph-launch). |
| Fronteras/streams ordinarios | `organism_adapter.py:49–57` termina trabajo del stream corriente antes de escribir entradas en el residente. `resident_advance` sincroniza antes de devolver. El refresh completo de `real_model.py:83` entra en el stream corriente y esa barrera también lo cubre. No se encontró carrera allí. |
| Rechazo RK y fallo de época | Un ensayo rechazado no ejecuta commit (`resident_controller.cu:114–126`). En fallo, `graph_runtime.py:132–144` restaura `x` desde backup. No es una transacción genérica sobre efectos arbitrarios de un RHS; el modelo capturado debe ser especulativo. Los escritores temporales PN restauran sus pesos dentro del grafo. Un error CUDA puede interrumpir esa restauración; el organismo real hace rollback estricto y se invalida, no debe reintentarse sólo con el backup de `x`. |
| Rollback externo | `block_midpoint.py:105–139` descarta el predictor y guarda/publica por bloque. `waveform_coupling.py:18–30` restaura estados físicos. `rollback_guard.py:12–21` invalida el runtime cuando `_restore_joint` reemplaza owners. Eso evita reutilizar los punteros capturados tras reconstrucción. |
| Células y eventos | `DeviceCell` mantiene un estado privado y publica sólo cuando todos los bloques pasan (`device_cell.py:59–87`). Capacidad de ocho eventos por célula; overflow marca error antes de escribir fuera del buffer (`physical_events.cu:15–19`). El kernel opera con 17 lanes y máscara `0x1ffff`; las reducciones y sincronizaciones respetan esa participación. Los flags compartidos usan atómicos. No se encontró desborde en el layout fijo 17/12. |
| PN | Se siguieron `pn_execution`, `resident`, `graph_step`, `graph_stage`, `resident_cyclic`, kernels de canales/reducción/árbol/core y constructor `CyclicPlan`. `StageGraph` conserva explícitamente arrays y reductores; sus ensayos no publican estado hasta pasar residual y química. La capacidad Ca ≤4 y core ≤64 se valida antes de sus arrays/kernel correspondientes. Hallazgo C1 se concentra en el cambio de stream de setup. |
| Límites CNS/ABI | CSR usa indptr int64 e índices int32; grid real de 256 conserva warps completos por fila. El refresh indexado valida posiciones y usa int64. Eventos validan capacidad, rango y orden en C++. Estado/tolerancias/domino se validan antes de captura, incluyendo coordenadas fuera de la norma. `ct.c_long` coincide con `long` Linux x86_64 del binario local; no se afirma portabilidad a Windows nativo. |

Los buffers y puertos son válidos para el layout congelado. `writer_layout` detecta cambio de objeto/puntero/forma de listas de posiciones, pero no una edición de sus contenidos; `FastCSR` copia `caps/tau/gain/theta` una sola vez. Una nueva API que permita cambiar topología, esas constantes o máscaras capturadas debe invalidar/reconstruir, aunque la forma y el puntero no cambien. **No se encontró esa mutación posterior a captura en el ensayo ordinario revisado.**

En errores de captura de `StageGraph`/árbol PN conviene emparejar `begin_capture` con `end_capture` usando `try/finally` y cerrar el owner fallido. En el flujo actual una excepción aborta la vida; no se propone convertirla en retry silencioso. `GraphRK23.advance(budget=30)` no usa ese argumento: el límite activo es 10.000 ensayos en C++, y el timeout Python no garantiza interrumpir una llamada CUDA en curso. Si el ensayo requiere límite duro de pared, supervisar el proceso externamente.

## Inventario de escritores FP32 y restauraciones

| Escritor | Cobertura real del espejo |
|---|---|
| PN general, `pn_general_output_brain.py:84–97` y `gpu_coefficient_layout.py:37–51` | `_general_positions`, refresco indexado justo antes del consumidor en cada RHS; restauración FP64 posterior. En esta vida `general_outputs.enabled=False`, pero la lista sigue declarada. |
| APL, `kc_apl_dynamic_brain.py:198–203`, `kc_spatial_brain.py:91–94`, `block_midpoint.py:73–76` | `_apl_gpu_positions`; incluye instalación/restauración de pesos del intercambio/predictor. Refresco en cada RHS. |
| Corte paralelo, `dnge035_parallel_brain.py:84–92` | `_parallel_position` se incluye si existe. El objeto de esta vida no lo declara: el reporte sólo enumera PN general y APL. No atribuir este escritor a la vida sin evidencia. |
| Plasticidad, `hybrid_visual_brain.py:84–85`, `gpu_visual_brain.py:69–71` | Hook `real_model.py:43–58` espera el residente, ejecuta el método completo y refresca todo. Ante error invalida el espejo. |
| Reinterpretación visual/cortes heredados, `receptor_visual_brain.py:108–121`, `r8_mi4_visual_brain.py:123–143`, `graded_descending_brain.py:256–271` | Se ejecutan dentro de la cadena `sync_plastic_weights`; el hook exterior refresca tras terminarla. `measured_t4_visual_brain.py` y `lamina_afferent_clamp.py` añaden validaciones/delegación. No reducir el hook sólo a `plasticity.positions`: omitiría esas escrituras. |
| Máscara de preparación `kcgamma_sensory_probe.py:117–129`; `OperatorState.restore` | Antes de captura en `restore_prepared.py:51–53`; cubiertas por conversión inicial. Restauración externa posterior requiere declarar dirty set o reconstruir. |
| Escrituras fuera del hook, p.ej. `canonical_pathway_probe.py:84–101`, `antennal_lobe_probe.py:74–92` | No forman parte de `run_trial.py`. No quedan automáticamente cubiertas por la lista de escritores. Una nueva lesión/intervención después de capturar necesitaría refresh/invalidación explícita, especialmente antes del primer sync que activa el refresh de frontera. |

`mirror.refresh_at_boundary=True` después del primer sync vuelve a convertir todos los pesos en cada frontera y cubre restauraciones in-place fuera del listado. Esto conserva coherencia a costa de tráfico considerable; **no eliminarlo sin cerrar el inventario de writers y rollbacks**. La vida de 1 s tiene `audit_enabled=false`: sus cero mismatches y cero evaluaciones auditadas no certifican igualdad exhaustiva del espejo.

## Rendimiento: medición existente, cotas y propuestas

Fuente: [RESULT.json de 1 s](/home/daroch/AXIOMA_ASTRA/motor_nuevo/event_memory_rk3_20260925_11/candidate_1000ms_01/RESULT.json:99). Son temporizadores host alrededor de regiones distintas, no perfiles de kernels.

| Tramo | Segundos medidos/derivados |
|---|---:|
| Avance total | 1.426,085698 |
| Dentro de `resident_advance` CNS | 537,647900 |
| Región temporal células | 266,547501 |
| Región temporal PN | 206,312703 |
| Avance sin atribuir a esas tres regiones | 415,577594 |
| Fuera del avance, incluido setup/trazas/cierre | 51,541304 |
| Pared total | 1.477,627002 |

El residuo contiene sincronizaciones, conversiones de pesos, H2D/D2H, preparación de fronteras, copias/restauración física, otros cálculos CPU/GPU y cuerpo. No es una medida de «Python». El temporizador células comienza después de sincronizar el stream corriente y termina antes de su publicación. El CNS excluye `FastCSR.refresh_all` de frontera, la entrada/salida del adaptador y snapshots.

Hechos para orientar una intervención de fondo:

- 16.000 épocas CNS, 113.164 ensayos, 452.656 RHS; 25.582.938 pesos. El recorrido CSR nominal de pesos FP32 + índices int32 representa **92,642 TB de accesos lógicos** si todos se consumen, antes de otros accesos. No es tráfico DRAM medido: caché, relecturas y filas omitidas por condiciones cambian el tráfico físico.
- Sólo 6.474 posiciones se declaran dinámicas (0,0253%), pero se registran **16.984 conversiones completas**. Leer FP64 y escribir FP32 para ellas representa **5,214 TB lógicos** adicionales. El total es consistente con 1.000 sync y 15.984 fronteras posteriores; esa descomposición no se instrumentó por separado. Su duración debe medirse antes de atribuirle el residuo.
- Células: 102.219.560 ensayos aceptados, 484.385 rechazados. El kernel registrado usa 250 registros/thread y un bloque de 32 threads con 17 activos. Es una pista de ocupación/latencia, no prueba del cuello. Ada tiene 64K registros de 32 bits y hasta 48 warps por SM según la [guía NVIDIA](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#occupancy). La cota aproximada por registros es ocho bloques de ese tamaño; no se propone ajustar registros a ciegas.
- Eliminar sólo todo el tiempo CNS, manteniendo los otros costes, da una aceleración máxima **1,572×**: quedan unos 940 s. El objetivo de 60 s necesita cambios transversales. Esta es una cota Amdahl condicionada a mantener el resto, no una predicción de otra arquitectura.

Alternativas sustanciales para discriminar, sin abrir tres implementaciones completas:

| Alternativa | Cambio y condición de rentabilidad | Falsador/validación necesaria |
|---|---|---|
| A. Fronteras residentes y generaciones de escritura | Mantener estados, snapshots de predictor y dirty sets en dispositivo; refrescar unión de posiciones realmente escritas por plasticidad, políticas y restauraciones. Publicar al host sólo lo requerido por cada frontera externa. Rentable si el perfil atribuye coste material a conversiones/copia/sync. | Ledger con todos los writers, comparación completa de rollback, eventos y fronteras; auditoría espejo temporal. Falla si dirty-set bookkeeping cuesta más o falta un escritor. No suprimir owners científicos. |
| B. Compilación del operador por sus salidas efectivas | Plan común que identifique filas/segmentos cuyos outputs son sustituidos después del CSR base y evite calcular dos veces trabajo finalmente descartado; fusionar conversiones y ensamblado cuando conserve el orden necesario. Rentable si esas filas contienen fracción material de aristas/tráfico o si domina la cadena de kernels cortos. | Perfil de kernels y aristas por output; mismas ecuaciones y sumas de los outputs conservados; error y conducta dentro de contrato. Falla si el CSR útil restante domina y las particiones añaden coste. No consiste en optimizar PN/KC aisladamente. |
| C. Programación por bloques independientes con layout apropiado | Rediseñar almacenamiento/resolución batched de sistemas pequeños y separar su publicación transaccional. Comparar CPU SIMD/batches, bloques CUDA con estado cooperativo y actual warp por célula; mantener geometría y residual FP64. Rentable si latencia/ocupación explica una fracción material de los 266,55 s. | Medir cuerpo completo además de kernel, conservar residual, rechazos, estados y eventos. Falla si tráfico/shared-memory/sincronización supera el ahorro; sin bajar tolerancias ni comprimir geometría por rendimiento. |

Primer discriminador recomendado: instrumentación de fases disjuntas con CPU wall + eventos CUDA y una captura breve de timeline autorizada por el coordinador. Separar refresh completo, copia/restauración, actualización de puertos, publicación, cuerpo y trabajo de kernels; los eventos CUDA no sustituyen wall para regiones CPU/transferencias fuera de su stream. Las muestras `nvidia-smi` tomadas entre pasos no permiten inferir utilización media ni ociosidad durante los kernels.

## Hardware, procedencia y límites

Consulta sólo lectura actual: RTX 4070 Ti SUPER, 16.376 MiB reportados, driver 591.86, capacidad 8.9, límite de potencia 285 W. CPU expuesta a WSL: Core Ultra 5 245K, 14 CPUs, un thread/core, hipervisor Microsoft. No se cambió potencia, clocks, afinidad ni configuración. El `build.py` apunta a `sm_89`, CUDA compiler 12.1.66 y headers/runtime 12.1.55; el binario local requiere libstdc++/libgcc/libc y no una libcudart dinámica en `NEEDED`. El registro de la vida declara NumPy 1.26.4, CuPy 13.6.0, Python 3.10.18 y BLAS a un thread. No prometer ganancia subiendo threads: los costes de esta ruta incluyen kernels, transferencias y sincronización.

Los 30 hashes de `SOURCES.json` de 11 y los 46 del `SOURCE_LOCK.json` de campaña 40 coinciden con los archivos. Esos manifiestos **no incluyen todas las dependencias PN reales** descritas arriba. El expediente 12 debe sellar también `resident_pn`, `pn_abc` y fuentes históricas realmente importadas. [REVIEW_EVIDENCE.json](/home/daroch/AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/reviews/cuda/REVIEW_EVIDENCE.json) guarda los hashes leídos y la aritmética de los tiempos, sin afirmar cierre transitivo completo.

No se ejecutaron sanitizers, racecheck, Nsight ni reproducción GPU de C1; esas ausencias delimitan la conclusión. La corrección propuesta de setup conserva el algoritmo. Cualquier optimización posterior debe compararse con el mismo estado preparado, parámetros, publicación y presupuesto del ensayo autorizado.
