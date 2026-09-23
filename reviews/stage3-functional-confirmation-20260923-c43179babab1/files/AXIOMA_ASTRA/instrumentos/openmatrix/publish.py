"""Explicit-manifest, immutable evidence publication to the user's OpenMatrix repository."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,re,zipfile,tarfile,subprocess,tempfile,shutil,urllib.request
ROOT=Path('/home/daroch/AXIOMA_ASTRA')
REPO=ROOT/'intercambio/github_20260922_01/repository'
REMOTE='git@github.com:skynet-omega/openmatrix.git'
ALLOWED=[ROOT/'campanas/etapa4_diseno_20260923_17',ROOT/'campanas/etapa3_funcional_20260923_16',ROOT/'campanas/etapa3_pn629_intervention_20260923_15',ROOT/'campanas/etapa3_kc_causal_20260923_14',ROOT/'motor_nuevo',ROOT/'instrumentos/openmatrix',ROOT/'instrumentos/jev_workflow',ROOT/'campanas/etapa3_motor_nuevo_20260922',ROOT/'campanas/etapa3_datos_reales_motor_20260922_01',ROOT/'campanas/etapa3_polaridad_20260923_01',ROOT/'campanas/etapa3_flujo_20260923_01',ROOT/'campanas/etapa3_flujo_directo_20260923_02',ROOT/'campanas/etapa3_flujo_kernel_20260923_03',ROOT/'campanas/etapa3_flujo_kernel_20260923_04',ROOT/'campanas/etapa3_evento_20260923_05',ROOT/'campanas/etapa3_evento_20260923_06',ROOT/'campanas/etapa3_event_set_20260923_07',ROOT/'campanas/etapa3_event_contract_20260923_08',ROOT/'campanas/etapa3_event_trace_20260923_09',ROOT/'campanas/etapa3_largo_diagnostico_20260923_10',ROOT/'campanas/etapa3_causal_controls_20260923_11',ROOT/'campanas/etapa3_dnb05_native_20260923_12',ROOT/'campanas/etapa3_kc_boundary_20260923_13',Path('/home/daroch/AXIOMA_FLYWIRE/matrix/src/rh_tarsal_body.py'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/src/session_io.py'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage2_contact_prosthesis_20260915/controller.py'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage2_contact_cns_20260915/contact_runtime.py'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage2_prosthesis_cns_20260915/runtime.py'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor13_20260922'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/motor14_20260922'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/runs/motor14_20260922'),Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live')]
ALLOWED.append(ROOT/'campanas/etapa3_funcional_repair_20260923_17')
ALLOWED.extend([ROOT/'campanas/etapa3_continuation_repair_20260923_19',ROOT/'campanas/etapa3_postclose_20260923_20'])
EXT={'.py','.cu','.cuh','.cpp','.hpp','.md','.json','.jsonl','.csv','.npz','.npy','.sha256','.log','.txt','.gz'}
TEXT=EXT-{'.npz','.npy','.gz'}
SECRET=re.compile(rb'(?i)(apikey_[0-9a-f]{16,}|sk-(?:proj-)?[a-z0-9_-]{24,}|gh[pousr]_[a-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [a-z0-9_.-]{20,})')
CHUNK=40*1024*1024

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(obj):return (json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def relative(value):
 p=PurePosixPath(value);require(bool(value) and not p.is_absolute() and '..' not in p.parts and '\\' not in value,'Unsafe archive destination');return p

def inspect_entries(spec):
 require(set(spec)=={'label','description','files'},'Unexpected manifest keys')
 require(not SECRET.search(dump({'label':spec['label'],'description':spec['description']})),'Possible credential in publication description')
 require(re.fullmatch(r'[a-z0-9-]{1,50}',spec['label']),'Invalid label')
 seen=set();items=[]
 for row in spec['files']:
  require(set(row)=={'source','destination'},'Unexpected file keys')
  source=Path(row['source']);dest=str(relative(row['destination']))
  require('..' not in source.parts,'Source traversal rejected')
  require(dest!='MANIFEST.json','Reserved archive destination')
  require(source.is_absolute() and source.is_file(),'Missing source')
  require(not any(p.is_symlink() for p in [source,*source.parents]),'Symlink source rejected')
  require(any(source.resolve().is_relative_to(p.resolve()) for p in ALLOWED),'Source outside approved evidence roots')
  require(dest not in seen,'Duplicate destination');seen.add(dest)
  require(source.suffix in EXT and not any(p.startswith('.') for p in PurePosixPath(dest).parts),'Unsupported or hidden evidence')
  data=source.read_bytes();require(not SECRET.search(data),'Possible credential; publication stopped')
  if source.suffix=='.gz':
   require(source.name.endswith('.tar.gz'),'Only inspectable source TAR.GZ archives are supported')
   with tarfile.open(source,'r:gz') as t:
    members=t.getmembers();require(len(members)<=20000 and sum(m.size for m in members)<=200*1024**2,'Source archive budget exceeded')
    names=set()
    for member in members:
     relative(member.name);require(member.name not in names,'Duplicate source archive member');names.add(member.name)
     require(member.isfile() or member.isdir(),'Source archive links/special members rejected')
     if member.isfile():require(not SECRET.search(t.extractfile(member).read()),'Possible credential inside source archive')
  if source.suffix=='.npy':
   import io,numpy as np
   array=np.load(io.BytesIO(data),allow_pickle=False)
   require(not array.dtype.hasobject,'Object arrays are not scientific numeric evidence')
  if source.suffix=='.sha256':
   require(re.fullmatch(rb'[a-f0-9]{64}\n?',data) is not None,'Invalid SHA256 sidecar')
  if source.suffix=='.npz':
   with zipfile.ZipFile(source) as z:
    for member in z.infolist():
     relative(member.filename);require(member.filename.endswith('.npy'),'Non-array NPZ member')
     with z.open(member) as f:
      carry=b''
      while True:
       b=f.read(1024*1024)
       if not b:break
       require(not SECRET.search(carry+b),'Possible credential inside NPZ');carry=b[-512:]
  items.append((source,dest,data))
 require(bool(items),'Empty publication')
 return items

def prepare(spec,root):
 items=inspect_entries(spec)
 public={'format_version':2,'label':spec['label'],'description':spec['description'],'files':[{'path':d,'bytes':len(b),'sha256':sha(b)} for _,d,b in items]}
 content=dump(public);snapshot=spec['label']+'-'+sha(content)[:12];folder=Path(root)/'reviews'/snapshot
 expected={}
 with tempfile.TemporaryDirectory() as td:
  archive=Path(td)/'evidence.zip'
  with zipfile.ZipFile(archive,'w') as z:
   for source,dest,data in items:
    info=zipfile.ZipInfo(dest,date_time=(2026,9,22,0,0,0));info.compress_type=zipfile.ZIP_STORED if source.suffix=='.npz' else zipfile.ZIP_DEFLATED;z.writestr(info,data)
    if source.suffix in TEXT and len(data)<=2*1024*1024:expected['files/'+dest]=data
   info=zipfile.ZipInfo('MANIFEST.json',date_time=(2026,9,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
  full=archive.read_bytes();parts=[]
  for i,start in enumerate(range(0,len(full),CHUNK)):
   name=f'evidence.zip.part{i:03d}';b=full[start:start+CHUNK];expected[name]=b;parts.append({'path':name,'bytes':len(b),'sha256':sha(b)})
  expected['ARCHIVE.json']=dump({'archive_sha256':sha(full),'archive_bytes':len(full),'parts':parts,'join':'Concatenate parts in listed order to evidence.zip; verify SHA256 before extraction.'})
 expected['MANIFEST.json']=content
 links='\n'.join(f'- [{p}](files/{p})' for _,p,b in items if 'files/'+p in expected)
 expected['README.md']=(f'# {spec["label"]}\n\n{spec["description"]}\n\nSnapshot inmutable: `{snapshot}`. [Manifiesto](MANIFEST.json), [ZIP dividido y hashes](ARCHIVE.json).\n\nFuentes y evidencia legibles:\n\n'+links+'\n\nDescargar las partes, concatenar por número y comprobar SHA256. El manifiesto enumera todos los archivos del ZIP. Revisar archivos no equivale a ejecutarlos.\n').encode()
 if folder.exists():
  actual={str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
  require(actual==expected,'Existing immutable snapshot differs')
 else:
  folder.mkdir(parents=True)
  for name,b in expected.items():
   p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 return folder,public

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
def publish(spec,receipt):
 require(git('remote','get-url','origin')==REMOTE,'Wrong publication remote')
 require(git('branch','--show-current')=='main','Wrong branch')
 require(not git('status','--porcelain'),'Publication checkout must be clean')
 folder,public=prepare(spec,REPO);relative_folder=folder.relative_to(REPO).as_posix()
 latest=f'# Última evidencia\n\n[{spec["label"]}]({relative_folder}/README.md)\n\n{spec["description"]}\n'
 (REPO/'LATEST.md').write_text(latest)
 git('add','--',relative_folder,'LATEST.md')
 if git('diff','--cached','--name-only'):
  git('-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com','commit','-m','Publish '+spec['label']+' evidence snapshot')
 commit=git('rev-parse','HEAD')
 subprocess.run(['git','-C',str(REPO),'push','origin','main'],check=True)
 require(git('ls-remote','origin','refs/heads/main').split()[0]==commit,'Remote commit mismatch')
 raw=f'https://raw.githubusercontent.com/skynet-omega/openmatrix/{commit}/{relative_folder}/MANIFEST.json'
 with urllib.request.urlopen(raw,timeout=30) as response:download=response.read()
 require(download==(folder/'MANIFEST.json').read_bytes(),'Public manifest readback mismatch')
 value={'commit':commit,'snapshot':relative_folder,'manifest_public_verified':True,'all_binary_downloads_verified':False,'url':f'https://github.com/skynet-omega/openmatrix/tree/{commit}/{relative_folder}','raw_index':raw.rsplit('/',1)[0]+'/README.md','files':public['files']}
 Path(receipt).write_bytes(dump(value));print(json.dumps({k:v for k,v in value.items() if k!='files'}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--publish',action='store_true');p.add_argument('--receipt',required=True);args=p.parse_args();spec=json.loads(Path(args.manifest).read_text())
 if args.publish:publish(spec,args.receipt)
 else:
  folder,public=prepare(spec,Path(args.receipt).parent/'prepared');Path(args.receipt).write_bytes(dump({'prepared':str(folder),'files':public['files'],'published':False}));print(folder)
