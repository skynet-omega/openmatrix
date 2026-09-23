# stage3-flow-observer-failures-20260923

Revisión de ventana tardía, auditoría del pipeline vigente y dos campañas de instrumentación detenidas por discrepancia PN; código ejecutado, fallos, fixture CUDA y revisión externa. Sin PASS de etapa 3.

Snapshot inmutable: `stage3-flow-observer-failures-20260923-887b1258caa9`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [campaign/README.md](files/campaign/README.md)
- [campaign/PLAN_DIRECT.json](files/campaign/PLAN_DIRECT.json)
- [campaign/CLOSE.json](files/campaign/CLOSE.json)
- [campaign/close_campaign.py](files/campaign/close_campaign.py)
- [campaign/flow_observer_direct.py](files/campaign/flow_observer_direct.py)
- [campaign/run_flow_direct.py](files/campaign/run_flow_direct.py)
- [campaign/test_graph_capture.py](files/campaign/test_graph_capture.py)
- [campaign/GRAPH_CAPTURE_FIXTURE.json](files/campaign/GRAPH_CAPTURE_FIXTURE.json)
- [campaign/smoke_direct_01_RESULT.json](files/campaign/smoke_direct_01_RESULT.json)
- [campaign/smoke_direct_01_FROZEN.json](files/campaign/smoke_direct_01_FROZEN.json)
- [campaign/smoke_direct_02_RESULT.json](files/campaign/smoke_direct_02_RESULT.json)
- [campaign/smoke_direct_02_FROZEN.json](files/campaign/smoke_direct_02_FROZEN.json)
- [campaign/smoke_direct_02_FLOW_METADATA.json](files/campaign/smoke_direct_02_FLOW_METADATA.json)
- [parent/PLAN.json](files/parent/PLAN.json)
- [parent/audit_load.py](files/parent/audit_load.py)
- [parent/AUDIT.json](files/parent/AUDIT.json)
- [parent/flow_observer_corrected_unrun.py](files/parent/flow_observer_corrected_unrun.py)
- [parent/flow_observer_executed_failed.py](files/parent/flow_observer_executed_failed.py)
- [parent/run_flow.py](files/parent/run_flow.py)
- [parent/smoke_off_RESULT.json](files/parent/smoke_off_RESULT.json)
- [parent/smoke_on_RESULT.json](files/parent/smoke_on_RESULT.json)
- [parent/historical_late_readback.py](files/parent/historical_late_readback.py)
- [parent/HISTORICAL_LATE_READBACK.json](files/parent/HISTORICAL_LATE_READBACK.json)
- [external/CHATGPT_REVIEW.md](files/external/CHATGPT_REVIEW.md)
- [external/chatgpt_analizar_flujo.py](files/external/chatgpt_analizar_flujo.py)
- [source/organism_adapter.py](files/source/organism_adapter.py)
- [source/graph_core.py](files/source/graph_core.py)
- [source/graph_control_v2.cpp](files/source/graph_control_v2.cpp)
- [source/gpu_coefficient_layout.py](files/source/gpu_coefficient_layout.py)
- [source/prosthetic_olfactory_brain.py](files/source/prosthetic_olfactory_brain.py)
- [source/orn_peripheral_terminal_brain.py](files/source/orn_peripheral_terminal_brain.py)
- [source/pn_online_cns_brain.py](files/source/pn_online_cns_brain.py)
- [source/pn_general_output_brain.py](files/source/pn_general_output_brain.py)
- [source/pn_electrical_output_brain.py](files/source/pn_electrical_output_brain.py)
- [source/block_midpoint.py](files/source/block_midpoint.py)
- [source/event_coupling.py](files/source/event_coupling.py)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
