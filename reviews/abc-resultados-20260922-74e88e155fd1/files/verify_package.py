"""Authenticate all manifest-listed extracted files before loading arrays or code."""
from pathlib import Path,PurePosixPath
import json,hashlib
H=Path(__file__).resolve().parent
m=json.loads((H/'MANIFEST.json').read_text())
for item in m['files']:
 p=PurePosixPath(item['path'])
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
 f=H/p
 if f.is_symlink() or not f.is_file():raise ValueError('Missing or linked file '+str(p))
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 if f.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('Corrupt file '+str(p))
print(json.dumps({'verified_files':len(m['files']),'all_hashes_match':True}))
