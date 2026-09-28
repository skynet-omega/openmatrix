"""Run frozen A56 with only an explicit UTF-8 metadata writer."""
from pathlib import Path
import sys,importlib.util
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('frozen56_runner',P/'run_hold56.py');runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
spec=importlib.util.spec_from_file_location('repair56_json',H/'json_utf8.py');writer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=writer;spec.loader.exec_module(writer)
runner.H=H;runner.save=writer.save
if __name__=='__main__':runner.main()
