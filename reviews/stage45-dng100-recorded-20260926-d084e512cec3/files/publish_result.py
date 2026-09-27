"""Publish this finite acquisition only after its successful analytical close."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(value):
    p = HERE/'ENTREGA.json.partial'
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    p.replace(HERE/'ENTREGA.json')


def expired(*args):
    raise TimeoutError('Finite publication wall budget exhausted')


def main():
    save(dict(status='WAITING_FOR_ANALYTICAL_CLOSE', pid=os.getpid(), new_scientific_runs=0,
        publication_limits=dict(wall_s=600, CPU_s=300, archive_MiB=300, working_disk_GiB=3)))
    deadline = read(HERE/'LAUNCH.json')['started_unix_s'] + read(HERE/'PLAN.json')['aggregate_wall_s_max'] + 60
    while True:
        state = read(HERE/'CIERRE.json')
        if state['status'] in ('COMPLETE', 'INCOMPLETE'):
            require(state['status'] == 'COMPLETE', 'Analysis did not complete')
            break
        require(time.time() < deadline, 'Acquisition dependency timed out')
        time.sleep(10)
    signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    began = time.monotonic(); cpu = time.process_time()
    tool = ROOT/'instrumentos/openmatrix/publish.py'
    spec = importlib.util.spec_from_file_location('openmatrix_publication', tool)
    pub = importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)
    pub.REPO = ROOT/'intercambio/event_memory_20260925_11_publish'
    pub.ALLOWED.append(HERE)
    ordinary_git = pub.git
    def sparse_git(*args):
        if args and args[0] == 'add':
            args = ('add', '--sparse', *args[1:])
        return ordinary_git(*args)
    pub.git = sparse_git
    require(not pub.git('status', '--porcelain'), 'Publication checkout has concurrent changes')
    pub.git('pull', '--ff-only')
    names = ['README.md', 'RESULTADOS.md', 'DECISION.md', 'PLAN.json', 'SOURCES.json',
        'run.py', 'launch.py', 'observer.py', 'coefficient_observed.cu', 'observe.cu',
        'analyze.py', 'diagnosis.py', 'finish.py', 'publish_result.py', 'test_observer.py',
        'SOFTWARE_TEST.json', 'CHATGPT_DISENO.md', 'CHATGPT_CODIGO.md', 'CHATGPT_CIENCIA.md',
        'RESULTADOS.json', 'DICTAMEN.json', 'CURVAS.npz', 'ENTRADAS.png', 'CIERRE.json',
        'CONTEXT45.csv', 'CONTEXT45.json', 'CONTEXT45.npz', 'context45.py',
        'QUEUE.json', 'LAUNCH.json', 'jev_spec.json', 'jev_review/response.json',
        'tao_selected/RECEIPT.json']
    for arm in ('sham', 'odor'):
        names.extend([arm+'/RESULT.json', arm+'/EXECUTED_SOURCES.json', arm+'_SUPERVISOR.json'])
        names.extend(str(p.relative_to(HERE)) for p in sorted((HERE/arm/'blocks').glob('*ms/dng100_observed.npz')))
    require(sum((HERE/n).stat().st_size for n in names) < 300*1024**2, 'Publication archive budget')
    manifest = dict(label='stage45-dng100-recorded-20260926',
        description='Complete paired47: actual DNg100 coefficient records, corrected classification and '
            'analysis. Recompute with analyze.py and diagnosis.py from the extracted root. '
            'Not a whole-organism restart package; original45 hidden states were not recorded. '
            'Tao receipt describes selected independent source files, which are not redistributed here.',
        files=[dict(source=str(HERE/n), destination=n) for n in names])
    (HERE/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pub.publish(manifest, HERE/'PUBLICATION_RESULT.json')
    receipt = read(HERE/'PUBLICATION_RESULT.json')
    published = pub.REPO/receipt['snapshot']
    archive = read(published/'ARCHIVE.json')
    require(archive['archive_bytes'] < 300*1024**2, 'Published archive size budget')
    raw = 'https://raw.githubusercontent.com/skynet-omega/openmatrix/'+receipt['commit']+'/'+receipt['snapshot']+'/'
    with tempfile.TemporaryDirectory(prefix='dng47-delivery-') as temporary:
        folder = Path(temporary); bundle = folder/'evidence.zip'
        with bundle.open('wb') as f:
            for part in archive['parts']:
                with urllib.request.urlopen(raw+part['path'], timeout=45) as response:
                    data = response.read(part['bytes']+1)
                require(len(data) == part['bytes'] and sha(data) == part['sha256'], 'Remote part differs')
                f.write(data)
        require(sha(bundle.read_bytes()) == archive['archive_sha256'], 'Remote ZIP differs')
        extracted = folder/'extracted'; extracted.mkdir()
        with zipfile.ZipFile(bundle) as z:
            for entry in read(published/'MANIFEST.json')['files']:
                require(sha(z.read(entry['path'])) == entry['sha256'], 'Remote member differs')
            z.extractall(extracted)
        original = read(extracted/'RESULTADOS.json'); corrected = read(extracted/'DICTAMEN.json')
        env = dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='1')
        for script in ('analyze.py', 'diagnosis.py'):
            subprocess.run([sys.executable, script], cwd=extracted, env=env,
                stdout=subprocess.DEVNULL, check=True, timeout=180)
        recomputed = read(extracted/'RESULTADOS.json'); check = read(extracted/'DICTAMEN.json')
        require(original['arms'] == recomputed['arms'], 'Clean extraction changed observations')
        require(all(check[key] == corrected[key] for key in
            ('classification', 'literal_zero', 'has_negative', 'has_positive',
             'generic_suppression_supported_in_recorded_evaluations', 'paired')),
            'Clean extraction changed diagnosis')
        destination = Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/ETAPAS45_DNG100_20260926.zip')
        require(not destination.exists() or sha(destination.read_bytes()) == archive['archive_sha256'],
                'Different delivery ZIP already exists')
        destination.write_bytes(bundle.read_bytes())
    signal.alarm(0)
    save(dict(status='COMPLETE', url=receipt['url'], commit=receipt['commit'],
        all_binary_downloads_verified=True, clean_extraction_analysis_reproduced=True,
        zip=str(destination), zip_sha256=archive['archive_sha256'], archive_bytes=archive['archive_bytes'],
        new_scientific_runs=0, wall_s=time.monotonic()-began, CPU_s=time.process_time()-cpu,
        original_publication_receipt='PUBLICATION_RESULT.json; its binary flag predates this verification'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(dict(status='INCOMPLETE', error=repr(exc), new_scientific_runs=0))
        raise
