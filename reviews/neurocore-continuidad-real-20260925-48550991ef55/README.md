# neurocore-continuidad-real-20260925

CNS residente C++/CUDA compatible: punto medio exponencial con cinco evaluaciones por intento, pareja real de 100 ms exacta en los campos guardados, sin modificar umbrales. Reproducción del núcleo y recálculo local verificados. Snapshot público de fuentes/resultados; no contiene trayectorias grandes ni checkpoints, y no certifica aceleración con hardware compartido.

Snapshot inmutable: `neurocore-continuidad-real-20260925-48550991ef55`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [ENVIRONMENT.json](files/ENVIRONMENT.json)
- [GUARDS.json](files/GUARDS.json)
- [GUARDS_O.json](files/GUARDS_O.json)
- [PAIR100.json](files/PAIR100.json)
- [PAIR_PROCESS.log](files/PAIR_PROCESS.log)
- [PLAN.md](files/PLAN.md)
- [PUBLIC_SCOPE.md](files/PUBLIC_SCOPE.md)
- [README.md](files/README.md)
- [REPRODUCTION.json](files/REPRODUCTION.json)
- [RESULTADOS.md](files/RESULTADOS.md)
- [REVIEW.md](files/REVIEW.md)
- [RUNS.json](files/RUNS.json)
- [build.py](files/build.py)
- [candidate100_01.log](files/candidate100_01.log)
- [candidate100_01/EVENTS.json](files/candidate100_01/EVENTS.json)
- [candidate100_01/PROGRESS.jsonl](files/candidate100_01/PROGRESS.jsonl)
- [candidate100_01/RESULT.json](files/candidate100_01/RESULT.json)
- [candidate100_01/final_state/pn_state.json](files/candidate100_01/final_state/pn_state.json)
- [candidate100_01/final_state/prosthesis.json](files/candidate100_01/final_state/prosthesis.json)
- [candidate100_01/final_state/published.json](files/candidate100_01/final_state/published.json)
- [check_runtime.py](files/check_runtime.py)
- [compare_real.py](files/compare_real.py)
- [comparison.log](files/comparison.log)
- [event_projection.py](files/event_projection.py)
- [graph_runtime.py](files/graph_runtime.py)
- [real_model.py](files/real_model.py)
- [reference100_01.log](files/reference100_01.log)
- [reference100_01/EVENTS.json](files/reference100_01/EVENTS.json)
- [reference100_01/PROGRESS.jsonl](files/reference100_01/PROGRESS.jsonl)
- [reference100_01/RESULT.json](files/reference100_01/RESULT.json)
- [reference100_01/final_state/pn_state.json](files/reference100_01/final_state/pn_state.json)
- [reference100_01/final_state/prosthesis.json](files/reference100_01/final_state/prosthesis.json)
- [reference100_01/final_state/published.json](files/reference100_01/final_state/published.json)
- [resident_controller.cu](files/resident_controller.cu)
- [run_pair.py](files/run_pair.py)
- [run_real.py](files/run_real.py)
- [verify_saved.py](files/verify_saved.py)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
