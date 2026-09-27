"""Publish and independently extract the small, explicit input-design package."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    started, cpu = time.monotonic(), time.process_time()
    def expire(*unused):
        raise TimeoutError('Delivery wall budget exceeded')
    signal.signal(signal.SIGALRM, expire); signal.alarm(300)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    save('DELIVERY.json', dict(status='STARTED', budget=dict(wall_s=300, CPU_s=120, archive_MiB=2), neural_steps=0))
    spec = importlib.util.spec_from_file_location('openmatrix_publication', ROOT / 'instrumentos/openmatrix/publish.py')
    pub = importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)
    pub.REPO = ROOT / 'intercambio/event_memory_20260925_11_publish'
    pub.ALLOWED.append(HERE)
    ordinary_git = pub.git
    def sparse_git(*args):
        return ordinary_git(*(('add', '--sparse', *args[1:]) if args and args[0] == 'add' else args))
    pub.git = sparse_git
    need(not pub.git('status', '--porcelain'), 'Concurrent publication changes')
    pub.git('pull', '--ff-only')
    names = ['RESULTADOS.md', 'CONTRASTE.md', 'PLAN.json', 'PREFLIGHT.py', 'PREFLIGHT_RESULT.json',
        'INPUT_HASHES.json', 'PERFIL_1_HEXANOL.csv', 'CONTRACT.json', 'ANATOMIA_SELECCIONADA.json',
        'PERMUTATION_CONTROL.py', 'PERMUTATION_RESULT.json', 'MATCHED_PROFILES.csv', 'VERIFY.py',
        'REPORT.py', 'CHATGPT_ASTRA_V2_REQUEST.md', 'CHATGPT_ASTRA_V2_RESPONSE.md',
        'CHATGPT_RECEIPT.json', 'DELIVER.py']
    need(sum((HERE / name).stat().st_size for name in names) < 2*1024**2, 'Package budget')
    manifest = dict(label='stage45-sensory-controls-review-20260927',
        description='Reproduced review and exact nominal-rate controls: matched composition, population '
        'and per-antenna permutation; ChatGPT ASTRA_V2 review and local corrections. No neural simulation '
        'or stage admission. Portable standard-library verification: python -O VERIFY.py.',
        files=[dict(source=str(HERE / name), destination=name) for name in names])
    save('PUBLIC_MANIFEST.json', manifest)
    pub.publish(manifest, HERE / 'PUBLICATION_RESULT.json')
    receipt = json.loads((HERE / 'PUBLICATION_RESULT.json').read_text())
    snapshot = pub.REPO / receipt['snapshot']
    archive = json.loads((snapshot / 'ARCHIVE.json').read_text())
    files = json.loads((snapshot / 'MANIFEST.json').read_text())['files']
    need(archive['archive_bytes'] < 2*1024**2, 'Remote archive budget')
    base = f"https://raw.githubusercontent.com/skynet-omega/openmatrix/{receipt['commit']}/{receipt['snapshot']}/"
    with tempfile.TemporaryDirectory(prefix='clean-package-', dir=HERE) as temp:
        temp = Path(temp); bundle = temp / 'evidence.zip'
        with bundle.open('wb') as stream:
            for part in archive['parts']:
                with urllib.request.urlopen(base + part['path'], timeout=30) as response:
                    data = response.read(part['bytes'] + 1)
                need(len(data) == part['bytes'] and hashlib.sha256(data).hexdigest() == part['sha256'], 'Remote part mismatch')
                stream.write(data)
        need(hashlib.sha256(bundle.read_bytes()).hexdigest() == archive['archive_sha256'], 'Archive mismatch')
        extracted = temp / 'extracted'; extracted.mkdir()
        with zipfile.ZipFile(bundle) as z:
            expected = {entry['path'] for entry in files}
            for name in z.namelist():
                path = Path(name)
                need(not path.is_absolute() and '..' not in path.parts, 'Unsafe ZIP path')
            for entry in files:
                need(hashlib.sha256(z.read(entry['path'])).hexdigest() == entry['sha256'], 'Member mismatch')
            z.extractall(extracted)
        check = subprocess.run([sys.executable, '-O', 'VERIFY.py'], cwd=extracted,
            env=dict(os.environ, PYTHONPATH=''), check=True, capture_output=True, text=True, timeout=30)
        destination = Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/REVISION_CONTROLES_ETAPAS45_20260927.zip')
        need(not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() == archive['archive_sha256'], 'Different existing delivery')
        shutil.copy2(bundle, destination)
    signal.alarm(0)
    save('DELIVERY.json', dict(status='COMPLETE', url=receipt['url'], commit=receipt['commit'],
        zip=str(destination), zip_sha256=archive['archive_sha256'], bytes=archive['archive_bytes'],
        members=len(files), remote_hashes_verified=True, clean_extraction_reproduced=True,
        verification=json.loads(check.stdout), neural_steps=0,
        wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu))
    print(json.dumps(json.loads((HERE / 'DELIVERY.json').read_text()), indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('DELIVERY_FAILURE.json', dict(error=repr(exc), neural_steps=0))
        raise
