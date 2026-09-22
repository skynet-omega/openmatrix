from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent
j=json.loads((h/'MANIFEST.json').read_text())
for item in j['files']:
 p=(h/item['path']).resolve()
 if h not in p.parents:raise ValueError('manifest path escapes package')
 b=p.read_bytes()
 if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('file differs: '+item['path'])
print(json.dumps({'verified_files':len(j['files'])}))
