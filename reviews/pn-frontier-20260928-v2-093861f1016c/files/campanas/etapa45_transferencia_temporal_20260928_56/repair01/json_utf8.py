"""Explicit encoding for new diagnostic metadata; scientific objects unchanged."""
from pathlib import Path
import json,os
def save(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp')
 with temporary.open('w',encoding='utf-8') as f:
  json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 temporary.replace(path)
