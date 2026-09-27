"""Publish the bounded offline sensory result and verify its remote portable package."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib
import importlib.util
import json
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def need(ok,msg):
    if not ok:raise ValueError(msg)

def read(p):return json.loads(Path(p).read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def save(name,value):(HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    start=time.monotonic();cpu=time.process_time()
    def expired(*args):raise TimeoutError('Publication budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_CPU,(300,300))
    save('DELIVERY.json',dict(status='STARTED',scientific_runs=0,
        budget=dict(wall_s=600,CPU_s=300,archive_MiB=60,working_disk_GiB=1)))
    sources={
        'orn_peripheral_terminal.py':'/home/daroch/AXIOMA_FLYWIRE/matrix/src/orn_peripheral_terminal.py',
        'orn_peripheral_terminal_brain.py':'/home/daroch/AXIOMA_FLYWIRE/matrix/src/orn_peripheral_terminal_brain.py',
        'orn_pn_synaptic_brain.py':'/home/daroch/AXIOMA_FLYWIRE/matrix/src/orn_pn_synaptic_brain.py',
        'olfactory_synapse_candidate.py':'/home/daroch/AXIOMA_FLYWIRE/matrix/src/olfactory_synapse_candidate.py',
        'pn_cns_orn_stages.py':'/home/daroch/AXIOMA_FLYWIRE/matrix/src/pn_cns_orn_stages.py',
        'protocol45.py':str(ROOT/'campanas/iniciacion_olfativa_20260926_45/protocol.py')}
    (HERE/'sources').mkdir(exist_ok=True)
    for dest,source in sources.items():shutil.copy2(source,HERE/'sources'/dest)
    save('SOURCE_EXCERPTS.json',{dest:dict(original=source,sha256=digest(Path(source).read_bytes())) for dest,source in sources.items()})
    spec=importlib.util.spec_from_file_location('openmatrix_publication',ROOT/'instrumentos/openmatrix/publish.py')
    pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
    pub.REPO=ROOT/'intercambio/event_memory_20260925_11_publish';pub.ALLOWED.append(HERE)
    ordinary_git=pub.git
    def sparse_git(*args):
        if args and args[0]=='add':args=('add','--sparse',*args[1:])
        return ordinary_git(*args)
    pub.git=sparse_git
    need(not pub.git('status','--porcelain'),'Concurrent publication changes')
    pub.git('pull','--ff-only')
    names=['RESULTADOS.md','RESULTADOS.json','PLAN.json','EXECUTION.json','CONTRACT.json',
        'PROVENANCE.json','OBSERVATIONS.npz','COMPARACION.png','PERFIL_1_HEXANOL.csv',
        'DOOR_EXTRACTION.json','door_tree.json','compare.py','acquire_door.py','verify.py',
        'deliver.py','SOURCE_EXCERPTS.json','jev_spec.json','jev_review/response.json','jev_review/receipt.json',
        'primary_door/door_mappings.csv','primary_door/door_dataset_info.csv']
    names+=['sources/'+name for name in sources]
    names+=[r['source'] for r in read(HERE/'DOOR_EXTRACTION.json')['profile']]
    manifest=dict(label='sensory-transfer-post47-20260927',
        description='Offline correspondence after47: source-specific1-hexanol receptor means and recorded45 '
        'ORN/PN observations. Reproduce with python compare.py --verify and python verify.py. '
        'No new brain simulation, no physiological calibration or stage4/5 admission. '
        'Includes selected source CSVs and portable projection, not a restartable organism.',
        files=[dict(source=str(HERE/n),destination=n) for n in names])
    need(sum((HERE/n).stat().st_size for n in names)<60*1024**2,'Archive budget')
    save('PUBLIC_MANIFEST.json',manifest)
    print('Publishing verified offline evidence',flush=True)
    pub.publish(manifest,HERE/'PUBLICATION_RESULT.json')
    receipt=read(HERE/'PUBLICATION_RESULT.json');published=pub.REPO/receipt['snapshot']
    archive=read(published/'ARCHIVE.json');need(archive['archive_bytes']<60*1024**2,'Archive size')
    url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+receipt['commit']+'/'+receipt['snapshot']+'/'
    print('Downloading and recalculating the published package',flush=True)
    with tempfile.TemporaryDirectory(prefix='sensory-delivery-') as tmp:
        tmp=Path(tmp);bundle=tmp/'evidence.zip'
        with bundle.open('wb') as f:
            for part in archive['parts']:
                with urllib.request.urlopen(url+part['path'],timeout=45) as response:data=response.read(part['bytes']+1)
                need(len(data)==part['bytes'] and digest(data)==part['sha256'],'Remote part differs')
                f.write(data)
        need(digest(bundle.read_bytes())==archive['archive_sha256'],'ZIP differs')
        extract=tmp/'extracted';extract.mkdir()
        with zipfile.ZipFile(bundle) as z:
            for entry in read(published/'MANIFEST.json')['files']:
                need(digest(z.read(entry['path']))==entry['sha256'],'Remote member differs')
            z.extractall(extract)
        env=dict(os.environ,PYTHONPATH='',OPENBLAS_NUM_THREADS='1')
        verified=subprocess.run([sys.executable,'-O','verify.py'],cwd=extract,env=env,
            check=True,capture_output=True,text=True,timeout=120)
        destination=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/TRANSFERENCIA_OLFATIVA_20260927.zip')
        need(not destination.exists() or digest(destination.read_bytes())==archive['archive_sha256'],'Different prior delivery')
        shutil.copy2(bundle,destination)
    signal.alarm(0)
    save('DELIVERY.json',dict(status='COMPLETE',url=receipt['url'],commit=receipt['commit'],
        zip=str(destination),zip_sha256=archive['archive_sha256'],bytes=archive['archive_bytes'],
        all_binary_downloads_verified=True,clean_extraction_reproduced=True,
        corruption_checks=json.loads(verified.stdout),scientific_runs=0,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu))
    print(json.dumps(read(HERE/'DELIVERY.json'),indent=2))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('DELIVERY_FAILURE.json',dict(error=repr(e),scientific_runs=0))
        raise
