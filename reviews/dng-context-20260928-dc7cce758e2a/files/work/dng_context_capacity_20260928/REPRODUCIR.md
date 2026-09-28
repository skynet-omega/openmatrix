# Reproducción CPU del paquete

El ZIP contiene los diez registros originales en su codificación sin pérdida, selección congelada, extracción anatómica CSV, fuentes del consumidor/verificador/codec, adaptador, contrato, tablas y revisiones. El Parquet canónico fue verificado por Motor; aquí se incluye su extracción de columnas con el hash de procedencia, no todo el dataset. No contiene el simulador completo ni permite reivindicar reproducción de la adquisición CNS49.

Desde la raíz de una extracción nueva, con Python y NumPy (versiones registradas en ENVIRONMENT.json):

```bash
python src/dng_context_capacity.py INPUTS.json reconstruccion
python - <<'PY'
from pathlib import Path
import hashlib, json
expected=json.loads(Path('EXPECTED_OUTPUTS.json').read_text())
for name,digest in expected.items():
    actual=hashlib.sha256((Path('reconstruccion')/name).read_bytes()).hexdigest()
    if actual != digest: raise SystemExit('Diferencia: '+name)
print('Tablas e informe idénticos; etapas4/5 siguen abiertas.')
PY
```

La ejecución rechaza alteraciones en contrato, entradas declaradas, identidad anatómica, máscaras, dominio y fases. Recalcula todas las tablas desde los operandos; no confía en un booleano del informe. `execution.json` registra el coste de cada lectura y no se compara byte a byte porque el tiempo varía.

Comprobación breve del instrumento, si pytest está disponible:

```bash
PYTHONPATH=src python -m pytest -q tests/test_dng_context_capacity.py
```

La lectura completa tarda segundos en la máquina registrada. No inicia GPU, simulador, cuerpo ni servicios de red. Una reproducción local desde una extracción nueva no es validación neurobiológica ni confirmación en otra semilla o plataforma.
