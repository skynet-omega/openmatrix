"""Short whole-organism transport against the conserved corrected reference."""
from pathlib import Path
import sys,argparse,json,time,hashlib,shutil
HERE=Path(__file__).resolve().parent
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work')
sys.path[:0]=[str(OLD/'motor14_20260922'),str(OLD/'motor13_20260922')]
import run_engine
from motor_runtime import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--ms',type=int,default=5);p.add_argument('--odor',default='sham');p.add_argument('--reference',action='store_true');p.add_argument('--fine',action='store_true');p.add_argument('--resident-pn',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--batched-membrane',action='store_true');a=p.parse_args()
 saved=run_engine.load;restore=[];context={}
 def load(out):
  obj,*rest=saved(out);b=obj.core.hybrid
  import event_coupling
  events,undo=event_coupling.install(b);restore.append(undo);context['events']=events
  if not a.reference:
   from organism_adapter import install
   adapter,undo=install(b,events);restore.append(undo);context['adapter']=adapter
  if a.resident_pn:
   import pn_execution
   report,undo=pn_execution.install(b);restore.append(undo);context['pn']=type('Report',(),{'report':report})()
  if a.batched_membrane:
   import membrane_batch
   restore.append(membrane_batch.install(b))
  dump(out/'FROZEN.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*') if f.suffix in ('.py','.cpp','.json')})
  frozen=out/'native_sources';frozen.mkdir()
  for file in HERE.glob('*'):
   if file.suffix in ('.py','.cpp'):shutil.copy2(file,frozen/file.name)
  if a.profile:
   import cProfile,pstats
   old_step=obj.step;counter=[0]
   def step():
    counter[0]+=1
    if counter[0]!=a.ms:return old_step()
    pr=cProfile.Profile()
    try:return pr.runcall(old_step)
    finally:
     pr.dump_stats(str(out/'PROFILE.pstats'))
     with (out/'PROFILE.txt').open('w') as f:pstats.Stats(pr,stream=f).sort_stats('cumtime').print_stats(70)
   obj.step=step
  return (obj,*rest)
 run_engine.load=load
 try:run_engine.run(a.out,a.ms,odor=a.odor,fine=a.fine,block_midpoint=None if a.fine else 125000,kc_adaptive=True)
 finally:
  a.out.mkdir(exist_ok=True,parents=True)
  dump(a.out/'NATIVE_REPORT.json',{k:v.report for k,v in context.items()})
  for f in reversed(restore):f()
  run_engine.load=saved
if __name__=='__main__':main()
