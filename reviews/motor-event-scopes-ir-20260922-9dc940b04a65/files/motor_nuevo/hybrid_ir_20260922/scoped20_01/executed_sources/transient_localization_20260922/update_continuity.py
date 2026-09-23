from pathlib import Path
import json
T=Path(__file__).resolve().parent;ROOT=T.parents[1]
r=json.loads((T/'VERDICT.json').read_text());b=json.loads((T/'BUDGET_USED.json').read_text());d=json.loads((T/'DELIVERY.json').read_text())
(ROOT/'ESTADO_ACTUAL.md').write_text(f'''# Estado vigente — resolución temporal de eventos, 22-09-2026

**Motor PROMETEDOR_NO_CONFIRMADO; etapa 3 abierta. No hay campaña de organismo ni cola activa.** El usuario autoriza continuar a etapa 3 cuando la evidencia y revisión externa lo justifiquen; no hace falta volver a pedir permiso. La meta aproximada de 1 segundo simulado por minuto real sigue incumplida.

Ronda vigente cerrada: `motor_nuevo/transient_localization_20260922`. Consumidas las seis cargas previstas, incluidos ambos fallos; {b['aggregate_runner_wall_s']:.3f} s de runners frente al límite agregado de 1500 s. Dos candidatos numéricos, una consulta Jev real. No repetir ensayos para rescatar un resultado ni modificar el umbral de esta ronda.

## Evidencia que cambia la decisión

- Primera separación del filtro localizada en marcas de eventos distintas con inputs, saltos, operador y estado inicial iguales. Cambiar solo marcas elimina la separación del filtro en esa primera frontera; no se afirma que borre toda historia posterior. Replay independiente de dos historias: 166 consultas, residual máximo 3.47e-18. El filtro funciona para los eventos suministrados.
- Candidata A: límite de paso de 1562 ns cerca del umbral somático/axonal existente, detector muestreado intacto. Frente al refinamiento uniforme, discrepancia normalizada de {r['one_ms_normalized_difference']:.12g} a 1 ms y {r['five_ms_normalized_difference']:.12g} a 5 ms. Campos discretos iguales en esas instantáneas. Mejora de {r['one_ms_error_reduction_factor']:.2f} veces a 1 ms, no certificación de toda la trayectoria.
- Guardia completa 20 ms: {r['guard_advance_s']:.3f} s de cálculo. Media 3.655 s por ms simulado, aproximadamente {r['measured_speed_gap_to_target']:.2f} veces el presupuesto temporal objetivo. El cuello global CNS sigue siendo relevante. No confundir esta mejora de fidelidad con velocidad suficiente.
- Referencia fina completa 14 ms y falla durante el paso 15: `accepted event state outside domain`. Sin extremo refinado de 20 ms. La referencia antigua también difiere del refinamiento; no usarla como verdad absoluta. La comparación guardia/referencia antigua a 20 ms excede 1e-4 (1.36645e-4); hay además dos trazas centrales ausentes en el exportador antiguo.
- Fallo inicial de conexión del prototipo corregido: el publicador axonal se crea al primer avance, por lo que la guardia también debe inicializarse allí. Conservado `guard20_01` fallido y `guard20_02` posterior, sin sobrescribir.
- Candidata B: conservar propuesta nominal CNS tras recortes obligatorios. Fixture pequeño: 10→9 pasos, fronteras preservadas. No corrida de organismo ni ganancia global demostrada. No se combinó con A.
- Reparación de diagnóstico: `graph_core.py` conserva índices/valores/reloj del último ensayo antes de rollback; ambos runners guardan el diagnóstico en errores. Rechazo y restauración verificados con violación inducida. Esta reparación ocurrió después del fallo real: su índice/valor exactos siguen desconocidos.

## Verificación y entrega

[Informe](motor_nuevo/transient_localization_20260922/README.md), [comparaciones reconstruidas](motor_nuevo/transient_localization_20260922/CANDIDATES.json), [presupuesto y fallos](motor_nuevo/transient_localization_20260922/BUDGET_USED.json), [entrega](motor_nuevo/transient_localization_20260922/DELIVERY.json).

ZIP: `{d['archive']}`. SHA256 `{d['sha256']}`; {d['files']} archivos, {d['bytes']} bytes. Siete comprobaciones desde extracción nueva, incluidas reconstrucción de arrays y fixtures CUDA. No reproduce todo el organismo desde el ZIP: recursos estáticos anatómicos/corporales completos permanecen en el histórico local de solo lectura. No se validó reanudación corporal completa. Publicación final y descarga remota: consultar `FINAL_PUBLICATION.json` y `REMOTE_CHECK.json` cuando existan; no inferir éxito sin recibo.

## Revisión y siguiente decisión

ChatGPT examinó documentación y código del replay, respaldó la causa local y pidió validar el falsador de un estado ya sobre umbral; ese falsador CUDA pasa. No ejecutó los arrays ni aprobó el motor. Revisión guardada en `CHATGPT_LOCALIZATION_REVIEW.md`. Jev clasificó tareas en una llamada real (1071/216 tokens), no emitió aprobación científica. Sin subagentes Codex.

Próxima ronda debe registrar hipótesis y presupuesto nuevos: localizar el valor que disparó el dominio, sin clamp ni tolerancia cambiada por conveniencia. Mantener como alternativas (A) esquema nativo actual con diagnóstico, (B) integración multirritmo/respuesta de puertos para reducir recomputación global sin eliminar realimentación, (C) backend general alternativo con ecuaciones equivalentes. Los 1562 ns de A no son una solución universal ni localización continua de eventos. El motor generalista y la etapa 3 siguen pendientes.

Motivo de parada: seis cargas consumidas y refinamiento incompleto que impide interpretar el extremo largo. Hito obtenido: causa reproducida, candidata acotada, dos fallos conservados y diagnóstico reparado. No se mantiene ningún proceso del organismo por este archivo. Estados previos en `continuity_before/`; negativos del pipeline conservados en su campaña original.
''')
(ROOT/'README.md').write_text('''# AXIOMA_ASTRA — entrada vigente

Prioridad: motor neuronal general, eficiente y comprobado con el organismo real. Etapa 3 abierta; meta aproximada de 1 segundo simulado por minuto real pendiente.

La discrepancia transitoria se localizó en la temporización de eventos. Una guardia temporal mejora la precisión a 1/5 ms, pero el organismo sigue tardando 73.1 s por 20 ms simulados. La referencia más fina se detuvo durante el paso 15 por un estado fuera de dominio. No se promueve el motor todavía.

[Estado actual](ESTADO_ACTUAL.md) · [Informe y reproducción](motor_nuevo/transient_localization_20260922/README.md) · [Recursos externos investigados](motor_nuevo/fly_resources_20260922/LOCAL_REVIEW.md) · [ChatGPT/Jev](INTERCAMBIO_CHATGPT.md) · [Objetivo científico](OBJETIVO_DE_FONDO.md).

Sin procesos de organismo ni colas activas. Sin subagentes Codex. La autorización para avanzar a etapa 3 al contar con evidencia suficiente continúa vigente.
''')
(ROOT/'INTERCAMBIO_CHATGPT.md').write_text('''# Coordinación vigente — resolución de eventos, 22-09-2026

Codex programa y ejecuta; ChatGPT revisa/investiga; Jev clasifica tareas acotadas. Sin subagentes Codex. Conversación autorizada: «Investiga el conectoma de mosca», `6ab06db7-9908-83e9-a515-58c9e6e18a1a`. El usuario confirmó PRO/máximo; la herramienta no observa el selector y no se afirma verificación independiente.

## Respuestas usadas en esta ronda

- `e00c2f58-12fa-4314-a5b0-461749b3c6b9`: sugiere replay independiente del filtro, historial y operador; `motor_nuevo/transient_localization_20260922/CHATGPT_REVIEW.md`.
- `2caa7e0b-4b07-40c8-8d5f-435dfbe00b3f`: revisión documental de fuentes y PORT_75907. Respalda temporización como causa local y solicita verificar guardia ya por encima del umbral; `CHATGPT_LOCALIZATION_REVIEW.md`. Falsador CUDA ejecutado y aprobado, pero no implica motor admitido.

Jev realizó una llamada real: `motor_nuevo/transient_localization_20260922/jev_01`, modelo jev-1.13.0, 1071/216 tokens. Clasifica diagnóstico→Codex y revisión→ChatGPT. No confundir probabilidades de clasificación con aprobación científica. Credenciales excluidas de los paquetes.

## Entrega y coordinación

La primera publicación del puerto y la localización es el commit `b2ecd5d7d9b8e2d53058a4907b4af34d121e896e`, snapshot `reviews/motor-event-localization-20260922-69da9b3ea539` de skynet-omega/openmatrix. La revisión anterior corresponde a ese snapshot, no a los resultados posteriores de la guardia.

La entrega final de esta ronda queda en `DELIVERY.json`, `FINAL_PUBLICATION.json`, `REMOTE_CHECK.json` y `CHATGPT_FINAL_DISPATCH.json` de la campaña. Consultar esos recibos antes de afirmar publicación/descarga/envío; un archivo ausente significa que no está confirmado. La nueva pregunta pide decisión entre tres rutas generales, con código y negativos del organismo. No hay aprobación general pendiente de fingir ni ejecución autónoma atribuida a ChatGPT.

La herramienta rechaza mensajes si la conversación está generando. El historial anterior de coordinación queda en `motor_nuevo/transient_localization_20260922/continuity_before/INTERCAMBIO_CHATGPT.md`.
''')
print('Continuity updated from measured receipts')
