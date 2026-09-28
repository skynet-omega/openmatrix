"""Second and final packaging attempt within the original CPU/storage budget."""
from pathlib import Path
import json,resource,time,zipfile
import build_full_capsule52_v2 as pack
H=Path(__file__).resolve().parent;EX=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO');stem='ETAPA45_REPARACION_20260927_52'
manifest=json.loads((H/'MANIFEST.json').read_text());local=json.loads((H/'LOCAL_DELIVERY.json').read_text())
existing=sum(f.stat().st_size for f in H.rglob('*') if f.is_file())
existing+=sum(f.stat().st_size for d in ['salida','recibos'] for top in (EX/d).glob(stem+'*') for f in ([top] if top.is_file() else top.rglob('*')) if f.is_file())
reserve=3*local['bytes']+sum(v['bytes'] for v in manifest['files'].values())+4*1024**2
limit=8589934592-existing-reserve
if limit<=0:raise ValueError('No aggregate space budget')
plan=dict(CPU_cap_s=600,attempt1_CPU_charged_s=1000,archive_cap_bytes=limit,
    future_delivery_reserve_bytes=reserve,existing_new_bytes=existing,created_unix_s=time.time(),
    change='Lossless4096-line JSON segmentation and exact hash dedup; NPZ recipes unchanged; no scientific changes')
receipt=H/'CAPSULE_ENCODING_V2.json'
if receipt.exists():raise ValueError('One-shot second attempt')
receipt.write_text(json.dumps(plan,indent=2)+'\n')
resource.setrlimit(resource.RLIMIT_CPU,(600,610))
original=zipfile.ZipFile.writestr
def bounded_write(self,zinfo,data,compress_type=None,compresslevel=None):
    if compress_type==zipfile.ZIP_DEFLATED:compresslevel=6
    result=original(self,zinfo,data,compress_type=compress_type,compresslevel=compresslevel)
    if Path(self.filename).stat().st_size>limit:raise RuntimeError('Original storage bound reached; preserve partial')
    return result
zipfile.ZipFile.writestr=bounded_write
if __name__=='__main__':pack.main()
