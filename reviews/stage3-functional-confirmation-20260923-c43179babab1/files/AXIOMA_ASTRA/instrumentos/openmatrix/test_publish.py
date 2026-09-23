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
# Numeric checkpoints are accepted; pickle-backed object arrays and malformed digests are rejected.
with tempfile.TemporaryDirectory() as td:
 import numpy as np
 root=Path(td);p.ALLOWED=[root]
 path=root/'state.npy';np.save(path,np.array([1.,2.]))
 spec={'label':'numeric','description':'Numeric checkpoint','files':[{'source':str(path),'destination':'state.npy'}]}
 p.inspect_entries(spec)
 np.save(path,np.array([{'not':'numeric'}],dtype=object))
 try:p.inspect_entries(spec)
 except ValueError:pass
 else:raise RuntimeError('Object array escaped numeric-only publication')
 path=root/'checkpoint.sha256';path.write_text('a'*64+'\n');spec['files']=[{'source':str(path),'destination':path.name}];p.inspect_entries(spec)
 path.write_text('unrelated content')
 try:p.inspect_entries(spec)
 except ValueError:pass
 else:raise RuntimeError('Invalid hash sidecar accepted')
 print('PASS: numeric NPY and SHA sidecars; object/pickle and malformed hash rejected')
# Dependency archives are source-only, traversals/links and embedded credentials reject.
with tempfile.TemporaryDirectory() as td:
 import tarfile,io
 root=Path(td);p.ALLOWED=[root];path=root/'source.tar.gz'
 spec={'label':'upstream','description':'Dependency source','files':[{'source':str(path),'destination':'vendor/source.tar.gz'}]}
 for case in ['valid','traversal','link','embedded_credential']:
  with tarfile.open(path,'w:gz') as t:
   member=tarfile.TarInfo('../escape' if case=='traversal' else 'LICENSE');data=b'License text'
   if case=='embedded_credential':data=('apikey_'+'b'*32).encode()
   if case=='link':member.type=tarfile.SYMTYPE;member.linkname='/etc/passwd';t.addfile(member)
   else:member.size=len(data);t.addfile(member,io.BytesIO(data))
  try:p.inspect_entries(spec)
  except ValueError:
   if case=='valid':raise
  else:
   if case!='valid':raise RuntimeError('Archive rejection escaped: '+case)
 print('PASS: source TAR.GZ; traversal, links and embedded credential rejected')
