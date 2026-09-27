"""Remove only a proven redundant incomplete delivery copy; preserve all experiments."""
from pathlib import Path
import hashlib,json,time,zipfile,resource,signal
from build_full_capsule51 import inventory,need,digest,HERE
resource.setrlimit(resource.RLIMIT_CPU,(240,250))
signal.alarm(300)
p=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/ETAPA45_ALTERNATIVAS_20260927_51_COMPLETO.zip')
start=time.monotonic();cpu=time.process_time();archive_sha=digest(p);wanted=set();sizes={}
with zipfile.ZipFile(p) as z:
 need('MANIFEST.json' not in z.namelist(),'Never discard a completed capsule')
 for i in z.infolist():
  need(i.filename.startswith('payload/'),'Unexpected incomplete archive entry')
  expected=Path(i.filename).name.split('.')[0];d=hashlib.sha256()
  with z.open(i) as f:
   for b in iter(lambda:f.read(4*1024**2),b''):d.update(b)
  need(d.hexdigest()==expected,'Incomplete payload damaged');wanted.add(expected);sizes[expected]=i.file_size
original_count=len(wanted);original_sources=[]
for source,expected in inventory().items():
 if source.suffix=='.npz':
  with zipfile.ZipFile(source) as z:offsets=sorted({0,*[i.header_offset for i in z.infolist()],z.start_dir,source.stat().st_size})
  with source.open('rb') as f:
   for lo,hi in zip(offsets[:-1],offsets[1:]):
    f.seek(lo);d=hashlib.sha256();remaining=hi-lo
    while remaining:
     b=f.read(min(4*1024**2,remaining));need(b,'Truncated original');d.update(b);remaining-=len(b)
    found=d.hexdigest()
    if found in wanted:need(sizes[found]==hi-lo,'Size identity');wanted.remove(found)
 else:
  found=digest(source)
  if expected is not None:need(found==expected,'Frozen original changed')
  if found in wanted:need(sizes[found]==source.stat().st_size,'Size identity');wanted.remove(found)
 original_sources.append(str(source))
need(not wanted,'An incomplete payload has no verified original source')
receipt=dict(status='REDUNDANT_INCOMPLETE_COPY_VERIFIED',incomplete_archive_sha256=archive_sha,incomplete_archive_bytes=p.stat().st_size,payloads_verified=original_count,all_payloads_found_in_original_sources=True,original_sources_checked=original_sources,scientific_originals_modified=False,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,repair='Split original raw NPZ members into256KiB pieces to share long unchanged history prefixes. Original bytes remain exactly reconstructible. Same8GiB cap and same scientific916ms; no new neural run.')
(HERE/'PACKING_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\n')
p.unlink()  # Only this incomplete, manifest-less, fully duplicated delivery file.
print(json.dumps({k:v for k,v in receipt.items() if k!='original_sources_checked'}))
