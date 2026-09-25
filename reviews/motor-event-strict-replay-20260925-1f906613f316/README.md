# motor-event-strict-replay-20260925

Auditoría portátil de eventos especulativos versus confirmados y máximo de estado CNS final en 20ms sham. Ejecutar python3 -B code/verify_event_step_audit_strict.py --root . --out verify.json tras extracción. Sin afirmación de trayectoria completa.

Snapshot inmutable: `motor-event-strict-replay-20260925-1f906613f316`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [EVENT_STEP_PAIR_LONG_RESULT.json](files/EVENT_STEP_PAIR_LONG_RESULT.json)
- [event_step_baseline_long_01/EVENT_AUDIT.json](files/event_step_baseline_long_01/EVENT_AUDIT.json)
- [event_step_candidate_long_01/EVENT_AUDIT.json](files/event_step_candidate_long_01/EVENT_AUDIT.json)
- [code/verify_event_step_audit_strict.py](files/code/verify_event_step_audit_strict.py)
- [result/STRICT_EVENT_AUDIT_RESULT_V2.json](files/result/STRICT_EVENT_AUDIT_RESULT_V2.json)
- [result/STRICT_EVENT_AUDIT_RESULT_V2_O.json](files/result/STRICT_EVENT_AUDIT_RESULT_V2_O.json)
- [review/EVENT_STEP_STRICT_ADDENDUM.md](files/review/EVENT_STEP_STRICT_ADDENDUM.md)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
