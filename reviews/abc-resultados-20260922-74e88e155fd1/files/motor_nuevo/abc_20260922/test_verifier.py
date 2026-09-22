"""Deliberate corruption of frozen criterion, condition, and finite flag."""
from pathlib import Path
import tempfile,shutil,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify as v
original=v.HERE
with tempfile.TemporaryDirectory() as td:
 root=Path(td)/'motor_nuevo';here=root/'abc_20260922';here.mkdir(parents=True)
 shutil.copyfile(original.parent/'connections.cu',root/'connections.cu')
 for name in ('contract.json','recurrence.py','recurrence.cu','initial.npz','initial_provenance.json'):shutil.copyfile(original/name,here/name)
 run=here/'runs_01';run.mkdir();shutil.copyfile(original/'runs_01/freeze.json',run/'freeze.json')
 # The first damaged metadata must be rejected before reading absent trajectory data.
 first=run/'tonic_R0';first.mkdir();source=original/'runs_01/tonic_R0/timing.json';raw=source.read_text()
 v.HERE=here;detected=[]
 for kind in ('criterion','context','scope'):
  shutil.copyfile(original/'contract.json',here/'contract.json');(first/'timing.json').write_text(raw)
  if kind=='criterion':
   c=json.loads((here/'contract.json').read_text());c['bounds']['candidate_max_qs']*=2;(here/'contract.json').write_text(json.dumps(c))
  else:
   m=json.loads(raw);m['condition' if kind=='context' else 'scope']='corrupted';(first/'timing.json').write_text(json.dumps(m))
  try:v.verify(run)
  except ValueError:detected.append(kind)
  else:raise RuntimeError('Did not detect '+kind)
 # Explicit finite boolean is checked against numeric trajectories by load().
 m=json.loads(raw);m['finite']=False;(first/'timing.json').write_text(json.dumps(m));(first/'states.npz').symlink_to(original/'runs_01/tonic_R0/states.npz')
 c=json.loads((original/'contract.json').read_text())
 try:v.load(run,'tonic',{'id':'R0','method':'rk4','h_s':c['reference_steps_s'][0]},c)
 except ValueError:detected.append('finite_flag')
 else:raise RuntimeError('Did not detect finite flag')
 v.HERE=original
 print(json.dumps({'detected':detected,'source_tree_untouched':True}))
