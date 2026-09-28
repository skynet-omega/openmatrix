"""Bounded lossless packaging recovery; original scientific evidence is immutable."""
from pathlib import Path
import hashlib, json, resource, time, zipfile

H = Path(__file__).resolve().parent
EX = Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
STEM = 'ETAPA45_REPARACION_20260927_52'
A = EX/'salida'/(STEM+'_COMPLETO.zip')
resource.setrlimit(resource.RLIMIT_CPU, (45, 46))
started = time.process_time()
receipt = H/'CAPSULE_ATTEMPT1_FAILED.json'
if receipt.exists():
    raise ValueError('Recovery is one-shot')
sha = hashlib.sha256()
with A.open('rb') as f:
    for block in iter(lambda: f.read(4*1024**2), b''):
        sha.update(block)
count = 0
with zipfile.ZipFile(A) as z:
    if 'MANIFEST.json' in z.namelist():
        raise ValueError('Not the expected incomplete packaging attempt')
    for info in z.infolist():
        expected = Path(info.filename).name.split('.')[0]
        actual = hashlib.sha256()
        with z.open(info) as f:
            for block in iter(lambda: f.read(4*1024**2), b''):
                actual.update(block)
        if len(expected) != 64 or actual.hexdigest() != expected:
            raise ValueError('Partial payload failed integrity check')
        count += 1
result = dict(reason='Prospective archive space bound reached; no scientific run failed',
    partial_zip_sha256=sha.hexdigest(), partial_zip_bytes=A.stat().st_size,
    payloads_verified=count, original_inputs_retained=True,
    action='Remove only verified redundant incomplete transport copy; retain failure log and sources',
    attempt1_CPU_s_upper_bound=1000, cleanup_CPU_s=time.process_time()-started,
    no_new_neural_ms=0)
receipt.write_text(json.dumps(result, indent=2)+'\n')
A.unlink()

source = (H/'build_full_capsule52.py').read_text()
source = source.replace('import argparse', 'import itertools\nimport argparse')
source = source.replace("'full_capsule.log','QUEUE_STATUS.json'", "'full_capsule.log','full_capsule_v2.log','QUEUE_STATUS.json'")
needle = "            if source.suffix=='.npz':"
replacement = """            if source.suffix=='.json' and source.stat().st_size>1024**2:
                # Fixed line-count blocks align repeated session histories even
                # when numeric string lengths differ. Every newline is retained.
                segments=[]
                with source.open('rb') as f:
                    while True:
                        data=b''.join(itertools.islice(f,4096))
                        if not data:break
                        segments.append(payload(data,'.json_segment',True))
                entries.append(dict(path=dest.as_posix(),segments=segments,bytes=source.stat().st_size,sha256=h))
            elif source.suffix=='.npz':"""
if source.count(needle) != 1:
    raise ValueError('Unexpected builder source')
source = source.replace(needle, replacement)
source = source.replace("NPZ_files_reconstructed_exact=reconstructed", "segmented_files_reconstructed_exact=reconstructed,NPZ_files_reconstructed_exact=sum('segments' in e and e['path'].endswith('.npz') for e in entries)")
source = source.replace("schema='repair52_full_evidence_capsule_chunked_v1'", "schema='repair52_full_evidence_capsule_chunked_v2'")
(H/'build_full_capsule52_v2.py').write_text(source)
print(json.dumps(result))
