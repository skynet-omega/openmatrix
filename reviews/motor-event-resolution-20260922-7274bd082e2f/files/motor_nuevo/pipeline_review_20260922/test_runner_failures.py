from pathlib import Path
import tempfile,json,hashlib
from run_pipeline import main
checks=[]
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 for label,exception in [('interrupt',KeyboardInterrupt()),('load_failure',RuntimeError('load failed'))]:
  out=root/label
  def loader(path):raise exception
  code=main(['--out',str(out),'--odor','sham','--engine','causal_cuda','--ms','1','--smoke-test'],loader=loader)
  result=json.loads((out/'RESULT.json').read_text());expected='INTERRUPTED' if label=='interrupt' else 'INCOMPLETE'
  if code!=2 or result['status']!=expected or (out/'final_state').exists():raise RuntimeError('Wrong failed-run receipt')
  before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
  try:main(['--out',str(out),'--odor','sham','--engine','causal_cuda','--ms','1','--smoke-test'],loader=loader)
  except FileExistsError:pass
  else:raise RuntimeError('Existing run accepted')
  after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
  if before!=after:raise RuntimeError('Failed preflight overwrote evidence')
  checks.append(label)
report={'status':'PASS','checks':checks,'existing_evidence_unchanged':True}
(Path(__file__).parent/'RUNNER_FAILURE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
