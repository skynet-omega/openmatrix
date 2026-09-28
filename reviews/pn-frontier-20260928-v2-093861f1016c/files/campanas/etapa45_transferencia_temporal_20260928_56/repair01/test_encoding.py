"""Reproduce the original serializer failure under C locale and its repair."""
from pathlib import Path
import locale,json,time,tempfile,importlib.util,sys
H=Path(__file__).resolve().parent;start=time.process_time();old=locale.setlocale(locale.LC_CTYPE)
spec=importlib.util.spec_from_file_location('utf8_writer',H/'json_utf8.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
 locale.setlocale(locale.LC_CTYPE,'C');value={'scope':'PN→KC, señal, 128ms','finite':1.25}
 with tempfile.TemporaryDirectory(prefix='encoding_test_',dir=H) as d:
  p=Path(d)/'sample.json';failed=False
  try:
   with p.open('w') as f:json.dump(value,f,ensure_ascii=False)
  except UnicodeEncodeError:failed=True
  if not failed:raise ValueError('Original C-locale failure not reproduced')
  m.save(p,value)
  if json.loads(p.read_text(encoding='utf-8'))!=value:raise ValueError('UTF8 round trip')
  before=p.read_bytes();rejected=False
  try:m.save(p,{'bad':float('nan')})
  except ValueError:rejected=True
  if not rejected or p.read_bytes()!=before:raise ValueError('Nonfinite/atomic guard')
finally:locale.setlocale(locale.LC_CTYPE,old)
print(json.dumps({'original_failure_reproduced':True,'UTF8_round_trip':True,'nonfinite_rejected_and_previous_preserved':True,'optimized':not __debug__,'CPU_s':time.process_time()-start}))
