# MOTOR14 — cambio matemático útil y límite de integración

**PROMETEDOR_NO_CONFIRMADO. Etapa3 abierta.** La petición posterior del usuario abre un motor independiente con arquitecturas B/C; no continuar sólo con parches. Entrada activa: `/home/daroch/AXIOMA_ASTRA/motor_nuevo/README.md`.

La solución analítica del filtro q→s elimina un error de retención temporal bajo los mismos eventos. Replay de3ms:141eventos, diferencia máxima retenido/exacto0.007184; cambio por retener conductancias0.00005799. Control por exponencial independiente1.11e-16 y composición temporal2.22e-16. Exactitud condicional a los eventos prescritos, no biología ni red completa.

La parte pasiva compartida de KC tiene modos de≈199545/s; ETDRK4 integra esa parte mediante propagadores precalculados y evalúa cuatro veces canales/compuertas actuales. Conserva17voltajes y68compuertas por célula. No se cachean respuestas a entradas anteriores ni se congela la matriz activa. ETD2 falla las células exigentes; ETDRK4 pasa las8células expuestas frenteRadau. A25us, error máximo1.144e-5mV y2.966e-6en compuertas.

En1557réplicas de4células exigentes y3ms de entradas fijas, el costeGPU baja2.091538→0.294948s:7.091×. CPU/GPU discrepan7.11e-14. Incluye la misma transferencia de muestras; no son1557casos independientes y estas8células no añaden nuevas espigas. Desenrollar el kernel previo dio≈1× y se conserva como negativo. El primer intento de captura ETD falló porque CuPy no permite esa llamada cuBLAS durante captura; se sustituyó la ejecución por operadores FP64 explícitos, conservando ese fallo y comprobando CPU/GPU.

## Organismo completo

166700neuronas y cuerpo, mismo inicio,5ms. Los límites predeclarados son estado1e-4,compuertas1e-4,PN0.005mV,yaw1e-4grados,posicion1e-6m y conteos finales exactos.

| Comparación contra referencia7.812us | Error estado | Error compuertas | Resultado |
|---|---:|---:|---|
| reference_refinement | 5.6535727e-06 | 2.2436717e-06 | PASA métricas5ms |
| event125 | 5.4580147e-05 | 6.3967986e-05 | PASA métricas5ms |
| event250 | 5.4580147e-05 | 0.00024460129 | FALLA |
| etd25 | 0.001288874 | 6.3969717e-05 | FALLA |
| etd625 | 0.00024838826 | 6.3968101e-05 | FALLA |

El resultado PASA se refiere sólo a las métricas predeclaradas: los historiales receptores tienen particiones temporales distintas, y su equivalencia funcional completa no se ha demostrado. La diferencia de referencia15.625→7.812us apoya convergencia de esas métricas, no una cota rigurosa de toda trayectoria.

Con ETDRK4 a25us,5ms cuestan10.414s frente17.799s del CN con filtro exacto, **pero falla fidelidad**. A6.25us cuesta12.221s y todavía falla. Conteos iguales con diferencias de q/s son compatibles con desplazamiento de emisión, sin aislarlo aún de realimentación/waveform. No hay ejecución de1segundo ni extrapolación calificada a segundos/minutos.

El perfil posterior de2ms, cuyo último ms se instrumentó, muestra2.097s: CNS0.946s,PN0.652s,KC0.137s; los hijos de esas funciones son tiempos inclusivos. Justifica replantear el intercambio y número de evaluaciones globales, no seguir optimizando únicamente KC. MuJoCo no aparece como cuello principal.

## Autocrítica y estrategia

Gemini acierta al cuestionar la arquitectura y proponer residenciaGPU. No demuestra las promesas82min→5min o5s/simseg: PN representaba≈43% en el perfil anterior, no80%; el porcentaje cambia con la variante.5s reales por1simulado es5veces más lento que tiempo real. La GPU real es4070TiSUPER y el motor usaFP64; el picoFP32 no es su rendimiento. La documentación de [MuJoCoWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/) distingue throughput/latencia y precisión: no garantiza ventaja para una sola mosca. No se congela la realimentación corporal50–100ms para fabricar velocidad.

Nuestra autocrítica: una mejora celular no paga por sí sola la interacción completa; el conteo no certifica timestamps; una referencia original no convergida no puede ser verdad por antigüedad. Corregir estos problemas ayuda a construir el nuevo motor, pero no se presenta como objetivo cumplido. La fuente matemática utilizada es [Kassam–Trefethen2005](https://people.maths.ox.ac.uk/trefethen/fourth-order.pdf); adaptación de ingeniería, no método numérico inventado.

## Alcance de entrega

Fuentes originales:426hashes sin cambios. Ocho intentos de organismo completos, incluido perfil2ms; una captura adicional del modelo no avanzó neuronas. Una llamada real nueva aJev clasificó seis tareas, coste estimadoUSD0.000076146. ChatGPT revisó fórmulas y conclusiones comunicadas; no ejecutó ni recibió este ZIP. PRO máximo confirmado por el usuario, selector no observable. Sin subagentesCodex.

El paquete permite reconstruir comparaciones desde arrays, repetir filtros y pruebas celulares, y compilar el banco C++/CUDA independiente. Incluye fallos, contratos, trazas, fuentes y entorno. No incluye anatomía/checkpoint/cuerpo padre suficientes para arrancar el organismo completo: los exports integrados son estado diagnóstico, no reanudación autónoma. No se anuncia reproducción completa del organismo.

Desde la raíz del ZIP extraído, con el entorno descrito en `ENVIRONMENT.json`:

```bash
python -I -B verify_package.py
python -I -B matrix/work/motor14_20260922/summarize.py --check
python -I -B matrix/work/motor14_20260922/test_event_waveform.py
python -I -B matrix/work/motor14_20260922/test_etd_ownership.py ownership_replay.json
python -I -B matrix/work/motor14_20260922/validate_etdrk4.py replay_etdrk4
python -I -B matrix/work/motor14_20260922/benchmark_etdrk4.py replay_kc_load
python -I -B matrix/work/motor14_20260922/replay_filters.py matrix/runs/motor14_20260922/capture_lif replay_filters
python -I -B motor_nuevo/benchmark_connections.py matrix/runs/motor14_20260922/architecture_input/connections.npz replay_connections
```

Los primeros dos comandos verifican el paquete y resultados; el resto repite experimentos locales. No amplía presupuesto de pruebas integradas ni califica etapas. Siguiente desarrollo: arquitecturas B/C en `motor_nuevo/PLAN_B_C.md`. Se cierra este hito de ingeniería/diagnóstico, sin agotar artificialmente intentos restantes ni mantener simulaciones activas.
