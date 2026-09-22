"""Campaign closure: reconstruct decisions, build explicit evidence package, optionally publish."""
from pathlib import Path
import sys,json,subprocess,argparse
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--receipt',required=True);a=p.parse_args()
subprocess.run([sys.executable,'-I','-B','-O',str(H/'report.py'),'--check'],check=True)
entries=[]
def add(path,dest):entries.append({'source':str(path),'destination':dest})
for name in ['recurrence.py','recurrence.cu','contract.json','initial.npz','initial_provenance.json','verify.py','test_numerics.py','test_verifier.py','report.py','long_run.py','README.md','RESULT.json','INDEPENDENT_TESTS.json','CORRUPTION_TESTS.json','jev_tasks.json','execution_01.log','long_execution_01.log','guarded_checks.py','guarded_long.py','safe_engine.py','cache_checks.py','CACHE_TESTS.json','CHATGPT_REVIEW.md','VERDICT.json','finalize.py','guarded_01.log','guarded_02.log','long_guarded_01.log']:
 add(H/name,'motor_nuevo/abc_20260922/'+name)
for folder in ['runs_01','long_01','jev_live','guarded_01','guarded_02','long_guarded_01']:
 for file in sorted((H/folder).rglob('*')):
  if folder=='long_guarded_01' and file.name=='states.npz':continue
  if file.is_file() and file.suffix in {'.npz','.json','.py'}:add(file,'motor_nuevo/abc_20260922/'+file.relative_to(H).as_posix())
for name in ['PLAN_ABC.md','connections.cu','SOURCE.json']:add(H.parent/name,'motor_nuevo/'+name)
add(H/'verify_package.py','verify_package.py')
for name in ['publish.py','test_publish.py']:add(ROOT/'instrumentos/openmatrix'/name,'workflow/'+name)
add(Path('/home/daroch/AXIOMA_FLYWIRE/matrix/runs/motor14_20260922/architecture_input/connections.npz'),'data/connections.npz')
spec={'label':'abc-resultados-20260922','description':'Tres hipótesis ejecutadas sobre la red recurrente base real. Dieciséis prefijos5ms y un segundo continuo base. Fuentes, estados y verificación incluidos. Faltan subsistemas especializados y cuerpo; no prueba fidelidad de1s ni cierra etapa3.','files':entries}
manifest=H/'publication_results.json';manifest.write_text(json.dumps(spec,indent=2)+'\n')
command=[sys.executable,'-I','-B',str(ROOT/'instrumentos/openmatrix/publish.py'),str(manifest),'--receipt',a.receipt]
if a.publish:command.append('--publish')
subprocess.run(command,check=True)
