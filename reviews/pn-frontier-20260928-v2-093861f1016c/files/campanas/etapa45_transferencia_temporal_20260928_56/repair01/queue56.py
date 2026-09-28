"""Reuse the frozen supervisor and controls in a new immutable directory."""
from pathlib import Path
import importlib.util,sys
H=Path(__file__).resolve().parent;P=H.parent
spec=importlib.util.spec_from_file_location('frozen56_queue',P/'queue56.py');queue=importlib.util.module_from_spec(spec);sys.modules[spec.name]=queue;spec.loader.exec_module(queue);queue.H=H
if __name__=='__main__':queue.main()
