"""Package frozen sources/inputs and saved evidence for CPU reconstruction.

Run only after both2s arms and both verifiers finish. No Git push, model loading,
GPU allocation, checkpoint transformation or original-file modification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ANCHOR = Path('/home/daroch')
SECRET = re.compile(rb'(?i)(apikey_[0-9a-f]{16,}|sk-(?:proj-)?[a-z0-9_-]{24,}|gh[pousr]_[a-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [a-z0-9_.-]{20,})')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.stage.exists():
        raise FileExistsError('Capsule and staging directory must be new')
    pair = json.loads((HERE/'PAIR2000.json').read_text())
    runtime = json.loads((HERE/'RUNTIME_PAIR2000.json').read_text())
    if pair['status'] not in ('PASS', 'FAIL') or runtime['status'] != 'PASS_RUNTIME_CONTRACT':
        raise ValueError('No valid completed pair and runtime contract to package')
    lock = json.loads((HERE/'SOURCES.json').read_text())
    files = {Path(s) for s in lock}
    # Preserve all new scientific runs, sources and reviews. Ignore Python cache
    # and compiler fixture executables, which are reproducible from source.
    allowed = {'.py','.cu','.cuh','.cpp','.hpp','.h','.so','.md','.json','.jsonl',
               '.npz','.npy','.log','.txt','.diff','.png','.sha256'}
    excluded = {'__pycache__','capsule_stage','capsule_extracted','choice_cache'}
    for path in HERE.rglob('*'):
        if (path.is_file() and path.suffix in allowed and
                not (set(path.relative_to(HERE).parts) & excluded)):
            files.add(path)
    a.stage.mkdir(parents=True)
    manifest = {}
    total = 0
    for source in sorted(files):
        if source.is_symlink() or not source.is_relative_to(ANCHOR):
            raise ValueError('Unexpected source path: '+str(source))
        relative = source.relative_to(ANCHOR)
        if relative.parts[0] not in ('AXIOMA_ASTRA','AXIOMA_FLYWIRE'):
            raise ValueError('Out of scope capsule payload')
        if str(source) in lock and sha(source) != lock[str(source)]:
            raise ValueError('Frozen source changed before delivery')
        if source.suffix not in {'.npz','.npy','.so','.parquet','.png'}:
            if SECRET.search(source.read_bytes()):
                raise ValueError('Credential pattern found; refuse packaging '+str(relative))
        destination = a.stage/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = sha(destination)
        total += destination.stat().st_size
        if total > 10*1024**3:
            raise ValueError('Capsule exceeds its10GiB payload budget')
    readme = a.stage/'README.md'
    readme.write_text('''# Motor12 — evidencia reproducible

Esta cápsula conserva fuentes congeladas, inputs identificados por el lock y
resultados completos de las corridas guardadas. Permite reconstruir las
comparaciones y contratos en CPU; no ejecuta una nueva vida ni certifica un
reinicio genérico. Los assets fisiológicos externos necesarios para volver a
cargar todo el organismo no se presentan como incluidos por guardar sus rutas.
Requiere Python y NumPy compatibles (runtime original: Python3.10/NumPy1.26.4).

Desde la carpeta extraída, sustituyendo `/ruta/capsula` por su ruta real:

```bash
python -I AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12/verify_capsule.py --root /ruta/capsula --out /tmp/MOTOR12_REPLAY.json
```

El verificador impide acceder a los dos árboles originales. Sólo relocaliza
rutas al leer JSON; no modifica payloads ni números. El manifiesto guarda los
hashes de cada archivo. `PAIR100/2000` y `RUNTIME_PAIR100/2000` deben reconstruirse
íntegramente. El modo `python -I -O` verifica que sus guardas no dependen de assert.
''')
    manifest['README.md'] = sha(readme)
    record = dict(schema='motor12_saved_evidence_capsule_v1', files=manifest,
                  payload_bytes=total+readme.stat().st_size, scientific_values_transformed=False,
                  functional_verdict_preserved=pair['status'],
                  new_organism_run_reproduced=False)
    (a.stage/'CAPSULE.json').write_text(json.dumps(record,indent=2)+'\n')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(a.stage.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(a.stage)))
    result = dict(zip=str(a.out.resolve()), bytes=a.out.stat().st_size,
                  sha256=sha(a.out), payload_files=len(manifest),
                  scope='Saved evidence; extraction/CPU verification still required')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
