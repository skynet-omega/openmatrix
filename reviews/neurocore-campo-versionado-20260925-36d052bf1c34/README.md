# neurocore-campo-versionado-20260925

Checkpoint real del organismo con estado versionado del campo externo, control de cache PN negativo, comparaciones exactas A/C, prueba de corrupcion y traza LIF invalidada por interferencia. Fuentes y recibos pequenos con hashes; sin checkpoint de 464 MB.

Snapshot inmutable: `neurocore-campo-versionado-20260925-36d052bf1c34`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).

Fuentes y evidencia legibles:

- [RESULTADOS.md](files/RESULTADOS.md)
- [check_restart.py](files/check_restart.py)
- [review_sources/PROVENANCE.json](files/review_sources/PROVENANCE.json)
- [review_sources/antennal_world.py](files/review_sources/antennal_world.py)
- [review_sources/antennal_runtime.py](files/review_sources/antennal_runtime.py)
- [review_sources/static_field.py](files/review_sources/static_field.py)
- [run_04/PREPARE.json](files/run_04/PREPARE.json)
- [run_05_control/PREPARE.json](files/run_05_control/PREPARE.json)
- [reload_02/RELOAD.json](files/reload_02/RELOAD.json)
- [reload_07/RELOAD.json](files/reload_07/RELOAD.json)
- [run_04/continuous_2ms/events.json](files/run_04/continuous_2ms/events.json)
- [reload_07/resumed_2ms/events.json](files/reload_07/resumed_2ms/events.json)
- [related/real_model.py](files/related/real_model.py)
- [related/graph_runtime.py](files/related/graph_runtime.py)
- [related/pn_execution.py](files/related/pn_execution.py)
- [related/organism_adapter.py](files/related/organism_adapter.py)
- [external_checkpoint.py](files/external_checkpoint.py)
- [static_lateral_checkpoint.py](files/static_lateral_checkpoint.py)
- [trace_first_lif.py](files/trace_first_lif.py)
- [TRACE_BUDGET.json](files/TRACE_BUDGET.json)
- [CHECKPOINT_GUARDS.json](files/CHECKPOINT_GUARDS.json)
- [VERSIONED_COMPARISON.json](files/VERSIONED_COMPARISON.json)
- [run_08_stagegraph_cache/PREPARE.json](files/run_08_stagegraph_cache/PREPARE.json)
- [run_09_versioned_driver/PREPARE.json](files/run_09_versioned_driver/PREPARE.json)
- [run_10_versioned_driver/PREPARE.json](files/run_10_versioned_driver/PREPARE.json)
- [run_10_versioned_driver/RELOAD.json](files/run_10_versioned_driver/RELOAD.json)
- [run_10_versioned_driver/checkpoint_v1/driver.json](files/run_10_versioned_driver/checkpoint_v1/driver.json)
- [run_10_versioned_driver/checkpoint_v1/manifest.json](files/run_10_versioned_driver/checkpoint_v1/manifest.json)
- [trace_lif_continuous_01/TRACE.json](files/trace_lif_continuous_01/TRACE.json)

Descargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.
