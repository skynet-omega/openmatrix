# Estado vigente — controlador causal CUDA, 22-09-2026

**Meta1segundo simulado/minuto todavía incumplida. Etapa3 abierta. No hay simulaciones ni cola activas.** No reanudar automáticamente el sham histórico ni sus checkpoints sin validar.

## Ronda actual cerrada para revisión

`motor_nuevo/causal_runtime_20260922`, contrato prospectivo `PLAN.json`: tres rivales, dos implementaciones, sin subagentes Codex. Presupuesto5400s,12 cargas y20 comprobaciones; ejecutadas6 cargas. Detalles en `BUDGET_RECEIPT.json`.

A: controlador C++/CUDA de pasos adaptativos por bloque independiente, completamente en GPU dentro de la época recibida. Conserva las fronteras62,5/125µs, todas las coordenadas/compuertas,51 matrices originales, error y residuo1e-12. Los eventos y estados quedan privados hasta completar todos los bloques. No usa la agrupación de bases cuya cota de redondeo sigue pendiente.

**Organismo20ms:** referencia82,8197s de avance; candidata60,3165s. Media pasos2–20:4,11443→2,99389s/ms,1,37427×. Trabajo nativo de membranas28,0554→5,38294s,5,21192×. Proceso total con carga/guardado96,3559→73,8084s. No es un segundo ejecutado ni se afirma alcanzar el objetivo.

Verificador congelado reconstruido desde arrays: normalizado4,29251e-5, compuertas5,87559e-5 (<1e-4); PN6,88710e-8mV, guiñada4,04498e-10grados. Conteos, relojes, banderas e identidades exactos. También pasa5ms contra referencia con eventos y referencia fina histórica. No hay validación biológica por esta comparación.

La prueba de célula sola/lote/permutado conserva estado y secuencia de eventos; fallo privado no publica y reintenta exactamente. Otro modelo de3estados usa el mismo núcleo. El detector heredado todavía puede perder un pico entre muestras: el falsador sugerido por ChatGPT queda en DEVICE_CHECK.json. No afirmar fidelidad universal de espigas por error pequeño de voltaje.

B: propagador separado de bloques afines y respuesta de receptor a eventos ordenados. Pasa exponencial matricial, evento tardío, prefijos, constantes iguales/casi iguales, inversión no conmutativa y modelo7estados. **No sustituye el CNS no lineal.** C: Jacobiano realmente acoplado y Schur permanece como rival analítico; no se implementó un tercer prototipo.

## Reparación confirmada y límite

ChatGPT señaló que la reconstrucción perdía tau/theta efectivos de la preparación. Se confirmó en carga fría y en fallo. `operator_state.py` conserva valores efectivos e identidad sin recalcular intervenciones. Tras restaurarlos, las continuaciones neurales de1ms continua/fría/fallo son exactamente iguales en todos los estados serializados guardados. Esto resuelve la discrepancia observada de la ronda anterior en ese contraste. La sesión corporal fallida sigue bloqueada: no existe aún recuperación automática integral del organismo.

## Siguiente decisión

El CNS ocupa2,10333s/ms de2,99389 y PN0,30990s/ms en la candidata. Priorizar planificación por dependencias de eventos o integración de sus efectos con cota para receptores no lineales; mantener frontera de eventos como control. No quitar fronteras ni subir tolerancias para obtener velocidad. No seguir invirtiendo sólo en membranas sin un techo de ganancia explícito.

Clasificación: **PROMETEDOR_NO_CONFIRMADO**; mejora de ingeniería medida20ms, sin admisión etapa3, sin motor general terminado. Se cierra el hito acotado de dos prototipos y reparación de identidad para revisión externa; el objetivo general sigue pendiente.

## Entrega y coordinación

ZIP verificado desde extracción limpia y publicado en OpenMatrix; SHA256 `d81faba00fbc0948097989294622b4a515c4c1ca345f3fc1b18963ce1e574dd9`. Disponible en `/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_MOTOR_CAUSAL_BLOCKS_2026-09-22.zip`; 468archivos; detalles en DELIVERY.json/CLEAN_CHECK.json y PUBLICATION.json. El paquete reproduce comprobaciones y análisis desde arrays; no distribuye todos los recursos estáticos del organismo.

ChatGPT revisó documentalmente las fuentes anteriores (abd7403e y f39e03ff), aportó la causa de reanudación y el falsador de eventos; no ejecutó CUDA. Jev realizó una clasificación acotada1064/214tokens; no valida ciencia. Ver INTERCAMBIO_CHATGPT.md para el envío de fuentes nuevas.

[Resultados](motor_nuevo/causal_runtime_20260922/RESULTADOS.md) · [Verificación](motor_nuevo/causal_runtime_20260922/VERIFIED.json) · [Reproducir y límites](motor_nuevo/causal_runtime_20260922/README.md) · [Arquitectura A/B/C](motor_nuevo/causal_runtime_20260922/ARCHITECTURE.md).

Estado anterior preservado en `motor_nuevo/causal_runtime_20260922/continuity_before/`. Sus pendientes son históricos, no instrucciones actuales.
