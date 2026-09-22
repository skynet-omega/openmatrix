"""Verification replay of the original long run with persistent intermediate guards."""
from pathlib import Path
import sys,runpy,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import recurrence
from safe_engine import SafeEngine
recurrence.Engine=SafeEngine
H=Path(__file__).resolve().parent
namespace=runpy.run_path(str(H/'long_run.py'),run_name='__main__')
e=namespace['e'];e.require_valid();out=Path(sys.argv[2])
original=json.loads((H/'long_01/RESULT.json').read_text());actual=json.loads((out/'RESULT.json').read_text())
if actual['state_file_sha256']!=original['state_file_sha256']:raise ValueError('Long guarded replay differs from saved complete trajectory')
recurrence.write(out/'VALIDATION.json',{'persistent_flags':int(e.flags.get()[0]),'full_saved_trajectory_byte_identical':True,'state_sha256':actual['state_file_sha256'],'replay_kind':'instrumentation-only verification, no new hypothesis or accuracy reference','source_sha256':{p.name:recurrence.digest(p) for p in [H/'guarded_long.py',H/'safe_engine.py',H/'guarded_checks.py']}})
