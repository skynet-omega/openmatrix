"""Verify a clean extraction without importing the working organism tree."""
from pathlib import Path
import json,hashlib,sys
H=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 m=json.loads((H/'MANIFEST.json').read_text())
 for name,v in m['files'].items():
  p=H/name
  if Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('manifest '+name)
 import verify55
 verify55.main()
if __name__=='__main__':main()
