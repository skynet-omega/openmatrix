# Reejecución completa en el entorno local

Requiere recursos estáticos históricos presentes en /home/daroch/AXIOMA_FLYWIRE y nuevas salidas. No utiliza una carpeta extraída como sustituto de esos recursos. Recibos/source hashes congelados conservan la ejecución original; `diagnose_domain.py` vigente solo corrige el manejo de un fallo secundario al guardar checkpoint.

```bash
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=14
export NUMBA_CACHE_DIR=/home/daroch/AXIOMA_ASTRA/campanas/etapa3_motor_nuevo_20260922/numba_cache
PY=/home/daroch/miniconda3/envs/GPU/bin/python
R=motor_nuevo/macro_abc_20260922
timeout 240s "$PY" "$R/diagnose_domain.py" --out "$R/replay_domain_01"
timeout 240s "$PY" "$R/run_reset_candidate.py" --variant fine1562 --ms 20 --out "$R/replay_fine_01"
timeout 240s "$PY" "$R/run_reset_candidate.py" --variant guard --ms 20 --out "$R/replay_guard_01"
timeout 240s "$PY" "$R/run_profile.py" --variant guard --ms 1 --out "$R/replay_profile_01"
```

Estas salidas nuevas no sustituyen evidencia de la campaña ni reciben automáticamente el mismo veredicto.
