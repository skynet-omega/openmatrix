# stage3-real-event-set-candidate-20260923

Evento real KC41645: post físico exacto 1, historia ADD codifica 1+1,77e-16 y bloquea CNS; control on/off igual. Candidato SET/ADD CPU+CUDA con fixtures y smoke1ms aprobados, sham130ms aún en ejecución; sin cuatro brazos ni PASS.

Snapshot inmutable: `stage3-real-event-set-candidate-20260923-e0327a803ef6`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [real_event/README.md](files/real_event/README.md)
- [real_event/PLAN.json](files/real_event/PLAN.json)
- [real_event/capture_event.py](files/real_event/capture_event.py)
- [real_event/analyze_capture.py](files/real_event/analyze_capture.py)
- [real_event/close_capture.py](files/real_event/close_capture.py)
- [real_event/CLOSE.json](files/real_event/CLOSE.json)
- [real_event/DOMAIN_CAPTURE.json](files/real_event/DOMAIN_CAPTURE.json)
- [real_event/RUN_RESULT.json](files/real_event/RUN_RESULT.json)
- [real_event/ARITHMETIC_AUDIT.json](files/real_event/ARITHMETIC_AUDIT.json)
- [real_event/REAL_EVENT_CASE.json](files/real_event/REAL_EVENT_CASE.json)
- [review/chatgpt_auditar_evento.py](files/review/chatgpt_auditar_evento.py)
- [review/REAL_CASE_RESULT.json](files/review/REAL_CASE_RESULT.json)
- [review/CHATGPT_REVIEW.md](files/review/CHATGPT_REVIEW.md)
- [control_off/README.md](files/control_off/README.md)
- [control_off/COMPARE_PREFIX.json](files/control_off/COMPARE_PREFIX.json)
- [candidate/PLAN.json](files/candidate/PLAN.json)
- [candidate/event_waveform.py](files/candidate/event_waveform.py)
- [candidate/event_coupling.py](files/candidate/event_coupling.py)
- [candidate/event_ports.py](files/candidate/event_ports.py)
- [candidate/run_set.py](files/candidate/run_set.py)
- [candidate/test_set_cpu.py](files/candidate/test_set_cpu.py)
- [candidate/test_set_gpu.py](files/candidate/test_set_gpu.py)
- [candidate/CPU_FIXTURE.json](files/candidate/CPU_FIXTURE.json)
- [candidate/GPU_FIXTURE.json](files/candidate/GPU_FIXTURE.json)
- [candidate/compare_smoke_parent.py](files/candidate/compare_smoke_parent.py)
- [candidate/SMOKE_PARENT_COMPARE.json](files/candidate/SMOKE_PARENT_COMPARE.json)
- [candidate/SMOKE_RESULT.json](files/candidate/SMOKE_RESULT.json)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
