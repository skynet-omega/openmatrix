# motor-temporal-guard-20260922

External review found a late-event false negative; reproduced with actual C++/CUDA. C0-only crossing is not promoted; current wrapper retains event cuts. Earlier2.08x whole-body measurement remains experimental. Complete small GPU reproducer, verified errors and corrected runner included; no extra organism campaign.

Snapshot inmutable: `motor-temporal-guard-20260922-26ce991e4ac1`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [motor_nuevo/hybrid_ir_20260922/README.md](files/motor_nuevo/hybrid_ir_20260922/README.md)
- [motor_nuevo/hybrid_ir_20260922/CHATGPT_FINAL_REVIEW.md](files/motor_nuevo/hybrid_ir_20260922/CHATGPT_FINAL_REVIEW.md)
- [motor_nuevo/hybrid_ir_20260922/CHATGPT_RECEIPT.json](files/motor_nuevo/hybrid_ir_20260922/CHATGPT_RECEIPT.json)
- [motor_nuevo/hybrid_ir_20260922/CHATGPT_ARCHITECTURE.md](files/motor_nuevo/hybrid_ir_20260922/CHATGPT_ARCHITECTURE.md)
- [motor_nuevo/hybrid_ir_20260922/TEMPORAL_GUARD_PLAN.md](files/motor_nuevo/hybrid_ir_20260922/TEMPORAL_GUARD_PLAN.md)
- [motor_nuevo/hybrid_ir_20260922/scoped_graph.py](files/motor_nuevo/hybrid_ir_20260922/scoped_graph.py)
- [motor_nuevo/hybrid_ir_20260922/scoped_graph_before_temporal_guard.py](files/motor_nuevo/hybrid_ir_20260922/scoped_graph_before_temporal_guard.py)
- [motor_nuevo/hybrid_ir_20260922/dependency_ir.py](files/motor_nuevo/hybrid_ir_20260922/dependency_ir.py)
- [motor_nuevo/hybrid_ir_20260922/model_event_contract.py](files/motor_nuevo/hybrid_ir_20260922/model_event_contract.py)
- [motor_nuevo/hybrid_ir_20260922/check_temporal_fallback.py](files/motor_nuevo/hybrid_ir_20260922/check_temporal_fallback.py)
- [motor_nuevo/hybrid_ir_20260922/TEMPORAL_FALLBACK_CHECK.json](files/motor_nuevo/hybrid_ir_20260922/TEMPORAL_FALLBACK_CHECK.json)
- [motor_nuevo/hybrid_ir_20260922/check_late_event_cuda.py](files/motor_nuevo/hybrid_ir_20260922/check_late_event_cuda.py)
- [motor_nuevo/hybrid_ir_20260922/verify_temporal.py](files/motor_nuevo/hybrid_ir_20260922/verify_temporal.py)
- [motor_nuevo/hybrid_ir_20260922/bundle_temporal.py](files/motor_nuevo/hybrid_ir_20260922/bundle_temporal.py)
- [motor_nuevo/hybrid_ir_20260922/run_scoped.py](files/motor_nuevo/hybrid_ir_20260922/run_scoped.py)
- [motor_nuevo/hybrid_ir_20260922/RUNNER_CLI_REPAIR.md](files/motor_nuevo/hybrid_ir_20260922/RUNNER_CLI_REPAIR.md)
- [motor_nuevo/hybrid_ir_20260922/RUNNER_CLI_CHECK.json](files/motor_nuevo/hybrid_ir_20260922/RUNNER_CLI_CHECK.json)
- [motor_nuevo/hybrid_ir_20260922/BUDGET_USED.json](files/motor_nuevo/hybrid_ir_20260922/BUDGET_USED.json)
- [motor_nuevo/hybrid_ir_20260922/ENVIRONMENT.json](files/motor_nuevo/hybrid_ir_20260922/ENVIRONMENT.json)
- [motor_nuevo/hybrid_ir_20260922/late_event_cuda_01/RESULT.json](files/motor_nuevo/hybrid_ir_20260922/late_event_cuda_01/RESULT.json)
- [motor_nuevo/hybrid_ir_20260922/late_event_cuda_01.log](files/motor_nuevo/hybrid_ir_20260922/late_event_cuda_01.log)
- [motor_nuevo/hybrid_ir_20260922/DELIVERY.json](files/motor_nuevo/hybrid_ir_20260922/DELIVERY.json)
- [motor_nuevo/hybrid_ir_20260922/REMOTE_CHECK.json](files/motor_nuevo/hybrid_ir_20260922/REMOTE_CHECK.json)
- [campanas/etapa3_motor_nuevo_20260922/graph_core.py](files/campanas/etapa3_motor_nuevo_20260922/graph_core.py)
- [campanas/etapa3_motor_nuevo_20260922/event_ports.py](files/campanas/etapa3_motor_nuevo_20260922/event_ports.py)
- [motor_nuevo/macro_abc_20260922/reset_ports.py](files/motor_nuevo/macro_abc_20260922/reset_ports.py)
- [motor_nuevo/native_hybrid_20260922/graph_control_v2.cpp](files/motor_nuevo/native_hybrid_20260922/graph_control_v2.cpp)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
