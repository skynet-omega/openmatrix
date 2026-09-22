from pathlib import Path
import json,hashlib,zipfile,subprocess,urllib.request,sys
H=Path(__file__).resolve().parent;root=H.parents[1];r=json.loads((H/'RESULTS_PUBLICATION.json').read_text());repo=root/'intercambio/github_20260922_01/repository';folder=repo/r['snapshot'];a=json.loads((folder/'ARCHIVE.json').read_text())
exchange=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO');dest=exchange/'salida/MATRIX_PN_ABC_2026-09-22.zip';digest=hashlib.sha256();count=0
if dest.exists():raise ValueError('Preserve existing ZIP')
with dest.open('xb') as f:
 for part in a['parts']:
  url=f"https://raw.githubusercontent.com/skynet-omega/openmatrix/{r['commit']}/{r['snapshot']}/{part['path']}"
  with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
  if hashlib.sha256(data).hexdigest()!=part['sha256']:raise ValueError('Public binary hash mismatch')
  digest.update(data);count+=len(data);f.write(data)
if digest.hexdigest()!=a['archive_sha256'] or count!=a['archive_bytes']:raise ValueError('Archive differs')
out=root/'intercambio/pn_abc_clean_20260922';out.mkdir(exist_ok=False)
with zipfile.ZipFile(dest) as z:z.extractall(out)
m=json.loads((out/'MANIFEST.json').read_text())
for f in m['files']:
 p=out/f['path']
 if p.stat().st_size!=f['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Extracted file differs')
python='/home/daroch/miniconda3/envs/GPU/bin/python';commands=[['-I','-B','-O','verify.py'],['-I','-B','validate.py','clean_validation'],['-I','-B','compare_backend.py','--route','B','--out','clean_backend']];results=[]
for i,args in enumerate(commands):
 with (out/f'clean_command_{i}.log').open('w') as log:proc=subprocess.run([python,*args],cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 results.append({'command':args,'returncode':proc.returncode})
 if proc.returncode:raise ValueError('Clean execution failed')
result={'zip':str(dest),'sha256':digest.hexdigest(),'bytes':count,'commit':r['commit'],'all_public_parts_verified':True,'manifest_files_verified':len(m['files']),'clean_directory':str(out),'commands':results,'scope':'PN1ms full numerical conditional replay; not organism or base1s reproduction'}
(H/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n');(exchange/'recibos/PN_ABC_20260922.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
