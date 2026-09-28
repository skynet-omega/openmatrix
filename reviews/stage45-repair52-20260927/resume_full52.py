"""Finish cached payload assembly once, after verified transport-copy cleanup.

This does not rerun either scientific experiments or compression of cached data.
Original aggregate5000CPU/8GiB limits remain in force.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,time,zipfile

H=Path(__file__).resolve().parent
EX=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
STEM='ETAPA45_REPARACION_20260927_52'
A=EX/'salida'/(STEM+'_COMPLETO.zip')
started=time.monotonic();cpu=time.process_time()
resource.setrlimit(resource.RLIMIT_CPU,(160,165))
for name in ['REMOTE_DELIVERY.json','CLEANUP_REMOTE.json']:
    if not (H/name).is_file():raise ValueError('Require verified publication and cleanup')
if (H/'ARCHIVE_APPEND_PLAN.json').exists():raise ValueError('Only one bounded continuation')
digest=hashlib.sha256()
with A.open('rb') as f:
    for b in iter(lambda:f.read(4*1024**2),b''):digest.update(b)
with zipfile.ZipFile(A) as z:
    if 'MANIFEST.json' in z.namelist():raise ValueError('Already complete')
    existing_payloads={Path(i.filename).name.split('.')[0]:i.filename for i in z.infolist()}
    if len(existing_payloads)!=len(z.infolist()):raise ValueError('Duplicate payload identity')

source=(H/'build_full_capsule52_v2.py').read_text()
replacements={
    "need(not a.zip.exists(),'Immutable capsule')":"need(a.zip.is_file(),'Existing interrupted archive required')",
    "entries=[];seen={}":"entries=[]\n    with zipfile.ZipFile(a.zip) as previous:\n        seen={Path(i.filename).name.split('.')[0]:i.filename for i in previous.infolist()}",
    "remote_reserve=(compact.stat().st_size if compact.exists() else 95*1024**2)+sum(v['bytes'] for v in compact_manifest['files'].values())+2*1024**2":"remote_reserve=0  # Download/extraction complete and identical duplicates retired with receipts.",
    "remaining=8589934592-campaign_bytes-delivery_bytes-remote_reserve":"remaining=8589934592-campaign_bytes-delivery_bytes+a.zip.stat().st_size-json.loads((HERE/'REMOTE_DELIVERY.json').read_text())['publication_blob_disk_bytes']-20*1024**2",
    "zipfile.ZipFile(a.zip,'x',allowZip64=True)":"zipfile.ZipFile(a.zip,'a',allowZip64=True)",
    "'full_capsule_v2.log','QUEUE_STATUS.json'":"'full_capsule_v2.log','archive_append.log','QUEUE_STATUS.json'",
}
for old,new in replacements.items():
    if source.count(old)!=1:raise ValueError('Unexpected builder '+old)
    source=source.replace(old,new)
target=H/'resume_builder52.py'
if target.exists():raise ValueError('Preserve continuation builder')
target.write_text(source)
plan=dict(partial_sha256=digest.hexdigest(),partial_bytes=A.stat().st_size,
    cached_payloads=len(existing_payloads),CPU_cap_s=160,
    original_space_cap_bytes=8589934592,central_directory_and_closure_reserve_bytes=20*1024**2,
    source_changes='Only manifest assembly reuses existing payloads and accounts for removed verified duplicates and retained Git objects',
    no_neural_ms=0,created_unix_s=time.time())
(H/'ARCHIVE_APPEND_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('resume_builder52',target)
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
original=zipfile.ZipFile.writestr
def write(self,zinfo,data,compress_type=None,compresslevel=None):
    if compress_type==zipfile.ZIP_DEFLATED:compresslevel=6
    return original(self,zinfo,data,compress_type=compress_type,compresslevel=compresslevel)
zipfile.ZipFile.writestr=write
try:
    pack.main()
finally:
    used=time.process_time()-cpu
    (H/'ARCHIVE_APPEND_EXECUTION.json').write_text(json.dumps(dict(CPU_s=used,
        wall_s=time.monotonic()-started,CPU_cap_s=160,
        full_receipt_exists=(H/'FULL_CAPSULE.json').exists()),indent=2)+'\n')
    if used>160:raise RuntimeError('Bounded continuation CPU exceeded')
