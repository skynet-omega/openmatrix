# neurocore-real-review-20260925

Revisión acotada de núcleo RK3(2) CUDA: seis corridas reales, mejora observada de tiempo y discrepancia numérica de 100 ms. Código y registros; no incluye trayectorias CNS completas ni modelo/checkpoint para reproducción integral.

Snapshot inmutable: `neurocore-real-review-20260925-c3a5a4aa112f`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [CONSULTA.md](files/CONSULTA.md)
- [README.md](files/README.md)
- [RESULTADOS.md](files/RESULTADOS.md)
- [PLAN.md](files/PLAN.md)
- [PRECISION.md](files/PRECISION.md)
- [graph_runtime.py](files/graph_runtime.py)
- [real_model.py](files/real_model.py)
- [resident_controller.cu](files/resident_controller.cu)
- [build.py](files/build.py)
- [run_real.py](files/run_real.py)
- [run_real_initial.py](files/run_real_initial.py)
- [compare_real.py](files/compare_real.py)
- [check_runtime.py](files/check_runtime.py)
- [PAIR20_FULL.json](files/PAIR20_FULL.json)
- [PAIR100.json](files/PAIR100.json)
- [PAIR100_FINE.json](files/PAIR100_FINE.json)
- [CONVERGENCE.json](files/CONVERGENCE.json)
- [candidate20_01/RESULT.json](files/candidate20_01/RESULT.json)
- [candidate20_01/PROGRESS.jsonl](files/candidate20_01/PROGRESS.jsonl)
- [candidate20_01/EVENTS.json](files/candidate20_01/EVENTS.json)
- [reference20_01/RESULT.json](files/reference20_01/RESULT.json)
- [reference20_01/PROGRESS.jsonl](files/reference20_01/PROGRESS.jsonl)
- [reference20_01/EVENTS.json](files/reference20_01/EVENTS.json)
- [candidate100_01/RESULT.json](files/candidate100_01/RESULT.json)
- [candidate100_01/PROGRESS.jsonl](files/candidate100_01/PROGRESS.jsonl)
- [candidate100_01/EVENTS.json](files/candidate100_01/EVENTS.json)
- [reference100_01/RESULT.json](files/reference100_01/RESULT.json)
- [reference100_01/PROGRESS.jsonl](files/reference100_01/PROGRESS.jsonl)
- [reference100_01/EVENTS.json](files/reference100_01/EVENTS.json)
- [candidate100_fine_01/RESULT.json](files/candidate100_fine_01/RESULT.json)
- [candidate100_fine_01/PROGRESS.jsonl](files/candidate100_fine_01/PROGRESS.jsonl)
- [candidate100_fine_01/EVENTS.json](files/candidate100_fine_01/EVENTS.json)
- [reference100_fine_01/RESULT.json](files/reference100_fine_01/RESULT.json)
- [reference100_fine_01/PROGRESS.jsonl](files/reference100_fine_01/PROGRESS.jsonl)
- [reference100_fine_01/EVENTS.json](files/reference100_fine_01/EVENTS.json)
- [reference/graph_core.py](files/reference/graph_core.py)
- [reference/organism_adapter.py](files/reference/organism_adapter.py)
- [reference/event_ports.py](files/reference/event_ports.py)
- [reference/event_waveform.py](files/reference/event_waveform.py)
- [reference/event_coupling.py](files/reference/event_coupling.py)
- [reference/runtime_session.py](files/reference/runtime_session.py)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
