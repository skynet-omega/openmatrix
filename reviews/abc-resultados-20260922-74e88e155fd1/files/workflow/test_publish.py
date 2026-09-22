from pathlib import Path
import tempfile,sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
import publish as p
with tempfile.TemporaryDirectory() as td:
 root=Path(td);p.ALLOWED=[root];src=root/'source.py';src.write_text('print(42)\n')
 spec={'label':'test','description':'Fixture','files':[{'source':str(src),'destination':'source.py'}]}
 a,_=p.prepare(spec,root/'repo');b,_=p.prepare(spec,root/'repo')
 if a!=b:raise RuntimeError('Idempotence failed')
 failures=0
 for action in ('traversal','symlink','credential','immutable','outside','source_traversal'):
  bad=copy.deepcopy(spec)
  if action=='traversal':bad['files'][0]['destination']='../escape.py'
  if action=='symlink':
   link=root/'link.py';link.symlink_to(src);bad['files'][0]['source']=str(link)
  if action=='credential':src.write_text('apikey_'+'a'*32)
  if action=='immutable':src.write_text('print(42)\n');(a/'README.md').write_text('changed')
  if action=='source_traversal':bad['files'][0]['source']=str(root/'..'/'..'/'etc'/'hosts')
  if action=='outside':bad['files'][0]['source']='/etc/hosts'
  try:p.prepare(bad,root/'repo')
  except ValueError:failures+=1
  else:raise RuntimeError('Did not reject '+action)
 if failures!=6:raise RuntimeError('Missing check')
 print('PASS: deterministic archive/idempotence and 6 deliberate rejection cases')
