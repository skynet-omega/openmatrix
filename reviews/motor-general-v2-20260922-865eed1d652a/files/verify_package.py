from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
m=json.loads((H/'MANIFEST.json').read_text())
for f in m['files']:
    p=H/f['path']
    if p.is_symlink() or not p.resolve().is_relative_to(H):raise ValueError('unsafe manifest source')
    b=p.read_bytes()
    if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:raise ValueError('hash mismatch: '+f['path'])
print('All '+str(len(m['files']))+' packaged file hashes verified')
