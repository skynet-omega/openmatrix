"""Recompute the natural spatial diagnostic from a clean extraction; no working-tree imports."""
from pathlib import Path
import json,hashlib,subprocess,sys,os,argparse,time
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--corruptions',action='store_true');a=p.parse_args();start=time.process_time()
 m=json.loads((H/'MANIFEST.json').read_text(encoding='utf-8'))
 for name,v in m['files'].items():
  rel=Path(name);f=H/rel
  need(not rel.is_absolute() and '..' not in rel.parts and f.is_file(),'manifest path '+name)
  need(f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],'manifest content '+name)
 results={}
 for key,script in [('A','verify57.py'),('C','dng_context57.py')]:
  args=[sys.executable,'-B','-O',script]
  if key=='A' and a.corruptions:args.append('--corruptions')
  done=subprocess.run(args,cwd=H,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,encoding='utf-8',timeout=120)
  need(done.returncode==0,script+'\n'+done.stdout+'\n'+done.stderr);results[key]=json.loads(done.stdout)
 print(json.dumps({'manifest_files_checked':len(m['files']),'A':results['A'],'C':results['C'],'new_CNS_ms':0,'parent_CPU_s':time.process_time()-start,'scope':'Reconstruction of retained arrays and CPU protocols, not a new GPU brain simulation.'}))
if __name__=='__main__':main()
