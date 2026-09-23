# stage3-event-contract-diagnostic-20260923

Reparación SET del evento real supera 130 ms sham; control de dos motores a 20 ms no cumple el criterio congelado. Auditoría posterior distingue 528/527 eventos predictores descartados de 1036/1036 aceptados, máximo desfase 3125 ns; etapa 3 abierta. Incluye código, trazas, eventos y revisión externa; no contiene checkpoint completo del organismo.

Snapshot inmutable: `stage3-event-contract-diagnostic-20260923-e283dc4d4c8e`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [07/README.md](files/07/README.md)
- [07/PLAN.json](files/07/PLAN.json)
- [07/CLOSE.json](files/07/CLOSE.json)
- [07/GPU_MULTI_FIXTURE.json](files/07/GPU_MULTI_FIXTURE.json)
- [07/PREFIX_130_COMPARE.json](files/07/PREFIX_130_COMPARE.json)
- [07/REFERENCE_PLAN.json](files/07/REFERENCE_PLAN.json)
- [07/REFERENCE_20_COMPARE.json](files/07/REFERENCE_20_COMPARE.json)
- [07/sham_130_01/RESULT.json](files/07/sham_130_01/RESULT.json)
- [07/reference_causal_20_01/RESULT.json](files/07/reference_causal_20_01/RESULT.json)
- [07/reference_native_20_01/RESULT.json](files/07/reference_native_20_01/RESULT.json)
- [08/README.md](files/08/README.md)
- [08/PLAN.json](files/08/PLAN.json)
- [08/CLOSE.json](files/08/CLOSE.json)
- [08/GPU_PHYSICAL_FIXTURE.json](files/08/GPU_PHYSICAL_FIXTURE.json)
- [08/SMOKE_COMPARE.json](files/08/SMOKE_COMPARE.json)
- [08/REFERENCE_20_COMPARE.json](files/08/REFERENCE_20_COMPARE.json)
- [08/CHATGPT_REVIEW.md](files/08/CHATGPT_REVIEW.md)
- [08/chatgpt_revisar_set.py](files/08/chatgpt_revisar_set.py)
- [08/chatgpt_local_repro_01/RESULTADO.json](files/08/chatgpt_local_repro_01/RESULTADO.json)
- [08/chatgpt_local_repro_02/RESULTADO.json](files/08/chatgpt_local_repro_02/RESULTADO.json)
- [08/physical_events.cu](files/08/physical_events.cu)
- [08/device_cell.cu](files/08/device_cell.cu)
- [08/device_cell.py](files/08/device_cell.py)
- [08/native_cell.py](files/08/native_cell.py)
- [08/block_runtime.hpp](files/08/block_runtime.hpp)
- [09/README.md](files/09/README.md)
- [09/PLAN.json](files/09/PLAN.json)
- [09/CLOSE.json](files/09/CLOSE.json)
- [09/PREFIX_FIXTURE.json](files/09/PREFIX_FIXTURE.json)
- [09/NEUTRALITY_CAUSAL.json](files/09/NEUTRALITY_CAUSAL.json)
- [09/NEUTRALITY_REFERENCE.json](files/09/NEUTRALITY_REFERENCE.json)
- [09/EVENT_COMPARE.json](files/09/EVENT_COMPARE.json)
- [09/ACCEPTED_EVENT_COMPARE.json](files/09/ACCEPTED_EVENT_COMPARE.json)
- [09/event_waveform.py](files/09/event_waveform.py)
- [09/event_ports.py](files/09/event_ports.py)
- [09/event_coupling.py](files/09/event_coupling.py)
- [09/run_set.py](files/09/run_set.py)
- [09/native_tap.py](files/09/native_tap.py)
- [09/compare_audits.py](files/09/compare_audits.py)
- [09/compare_accepted_events.py](files/09/compare_accepted_events.py)
- [09/audit_causal_20_01/RESULT.json](files/09/audit_causal_20_01/RESULT.json)
- [09/audit_native_20_01/RESULT.json](files/09/audit_native_20_01/RESULT.json)
- [09/audit_causal_20_01/EVENT_AUDIT.json](files/09/audit_causal_20_01/EVENT_AUDIT.json)
- [09/audit_native_20_01/EVENT_AUDIT.json](files/09/audit_native_20_01/EVENT_AUDIT.json)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
