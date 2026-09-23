"""Download and hash every published evidence part against the local archive."""
from pathlib import Path
import json,urllib.request,hashlib,concurrent.futures
R=Path(__file__).resolve().parent
r=json.loads((R/'PUBLICATION.json').read_text());local=json.loads((R/'DELIVERY.json').read_text());base=r['raw_index'].rsplit('/',1)[0]+'/'
with urllib.request.urlopen(base+'ARCHIVE.json',timeout=30) as response:a=json.load(response)
if a['archive_sha256']!=local['sha256']:raise ValueError('Remote archive identity')
def read(p):
 h=hashlib.sha256();size=0
 with urllib.request.urlopen(base+p['path'],timeout=60) as response:
  for data in iter(lambda:response.read(1024*1024),b''):h.update(data);size+=len(data)
 if h.hexdigest()!=p['sha256'] or size!=p['bytes']:raise ValueError('Remote part mismatch')
 return {'path':p['path'],'bytes':size,'sha256':h.hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:parts=list(pool.map(read,a['parts']))
result={'commit':r['commit'],'archive_sha256':a['archive_sha256'],'verified_bytes':sum(x['bytes'] for x in parts),'parts':parts,'all_binary_downloads_verified':True}
(R/'REMOTE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
