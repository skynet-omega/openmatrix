"""Space-budget wrapper: stronger lossless compression, no evidence alteration."""
from pathlib import Path
import json,time,resource,sys,zipfile
import build_full_capsule52 as pack
H=Path(__file__).resolve().parent;EX=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO');stem='ETAPA45_REPARACION_20260927_52'
manifest=json.loads((H/'MANIFEST.json').read_text());local=json.loads((H/'LOCAL_DELIVERY.json').read_text())
existing=sum(f.stat().st_size for f in H.rglob('*') if f.is_file())
existing+=sum(f.stat().st_size for d in ['salida','recibos'] for top in (EX/d).glob(stem+'*') for f in ([top] if top.is_file() else top.rglob('*')) if f.is_file())
# Remote assembled ZIP, remote extraction, transport pieces and conservative Git
# object reservation. Include generated independent extraction receipt overhead.
reserve=3*local['bytes']+sum(v['bytes'] for v in manifest['files'].values())+4*1024**2
limit=8589934592-existing-reserve
if limit<=0:raise ValueError('No aggregate space budget')
plan={'lossless_non_npz_deflate_level':6,'npz_segments_unchanged':True,'archive_cap_bytes':limit,'future_delivery_reserve_bytes':reserve,'existing_new_bytes':existing,'CPU_cap_s':1000,'created_unix_s':time.time(),'reason':'Compact ZIP128MB requires two transport pieces; include remote/Git copies within8GiB.'}
(H/'CAPSULE_ENCODING.json').write_text(json.dumps(plan,indent=2)+'\n')
resource.setrlimit(resource.RLIMIT_CPU,(1000,1010))
original=zipfile.ZipFile.writestr

def bounded_write(self,zinfo,data,compress_type=None,compresslevel=None):
 if compress_type==zipfile.ZIP_DEFLATED:compresslevel=6
 r=original(self,zinfo,data,compress_type=compress_type,compresslevel=compresslevel)
 if Path(self.filename).stat().st_size>limit:raise RuntimeError('Prospective full capsule space bound; preserve partial')
 return r
zipfile.ZipFile.writestr=bounded_write
if __name__=='__main__':pack.main()
