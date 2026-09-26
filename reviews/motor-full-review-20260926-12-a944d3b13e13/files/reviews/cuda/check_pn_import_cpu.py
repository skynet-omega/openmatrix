"""Verify overlay binding despite legacy sys.path insertion, without real CuPy."""
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.modules['cupy'] = types.ModuleType('cupy')
path = ROOT / 'pn' / 'install_pn.py'
spec = importlib.util.spec_from_file_location('install_pn_review', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sources = module.install()
# These are the insertions performed by pn_execution and resident.
sys.path[:0] = [str(ROOT.parent / 'resident_pn_20260922'),
                str(ROOT.parent / 'pn_abc_20260922')]
for name in module.NAMES:
    actual = Path(importlib.import_module(name).__file__).resolve()
    if actual != ROOT / 'pn' / (name + '.py'):
        raise RuntimeError('Legacy path insertion bypassed overlay: ' + name)
try:
    module.install()
except RuntimeError:
    repeated_install_rejected = True
else:
    raise RuntimeError('Repeated install was not rejected')
for name in module.NAMES:
    del sys.modules[name]
sys.modules['graph_stage'] = types.ModuleType('graph_stage')
try:
    module.install()
except RuntimeError:
    late_install_rejected = True
else:
    raise RuntimeError('Late install was not rejected')
if any(name in sys.modules for name in module.NAMES):
    raise RuntimeError('Late installation changed bindings before rejecting')
result = {'scope': 'CPU import identity check; CuPy replaced with an empty module',
          'sources': sources, 'legacy_path_insertions_preserve_overrides': True,
          'repeated_install_rejected': repeated_install_rejected,
          'late_install_rejected_without_partial_binding': late_install_rejected}
destination = Path(__file__).with_name('PN_IMPORT_CPU.json')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
