"""Exclusive run ownership and atomic snapshot publication, not resume claims."""
from pathlib import Path
import os,json,hashlib,uuid

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
 return h.hexdigest()

def atomic_json(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with temporary.open('x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)

class RunStorage:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
  atomic_json(self.out/'OWNER.json',{'run_id':uuid.uuid4().hex,'pid':os.getpid()})
 def snapshot(self,name,writer):
  if name not in ('prepared_state','state_100ms','final_state'):raise ValueError('Unregistered snapshot name')
  final=self.out/name
  if final.exists():raise FileExistsError(final)
  staging=self.out/(name+'.partial-'+uuid.uuid4().hex);staging.mkdir()
  try:
   writer(staging)
   files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in staging.iterdir() if f.is_file()}
   if not files:raise ValueError('Empty snapshot')
   atomic_json(staging/'MANIFEST.json',{'status':'COMPLETE_SERIALIZED_NOT_RESUME_VALIDATED','full_organism_resume_tested':False,'files':files})
   staging.rename(final)
   return {'status':'COMPLETE_SERIALIZED_NOT_RESUME_VALIDATED','path':str(final)}
  except BaseException as exc:
   atomic_json(staging/'FAILURE.json',{'type':type(exc).__name__,'message':str(exc)})
   raise

def verify_snapshot(folder):
 folder=Path(folder);m=json.loads((folder/'MANIFEST.json').read_text())
 if m.get('status')!='COMPLETE_SERIALIZED_NOT_RESUME_VALIDATED' or m.get('full_organism_resume_tested') is not False:raise ValueError('Unexpected checkpoint claim')
 if set(m['files'])!={p.name for p in folder.iterdir() if p.name!='MANIFEST.json'}:raise ValueError('Snapshot members differ')
 for name,row in m['files'].items():
  if Path(name).name!=name or (folder/name).is_symlink() or not (folder/name).is_file():raise ValueError('Invalid snapshot member')
  if (folder/name).stat().st_size!=row['bytes'] or digest(folder/name)!=row['sha256']:raise ValueError('Snapshot corrupted')
 return m
