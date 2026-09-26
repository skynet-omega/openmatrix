"""Bind the two corrected PN trees before the legacy PN adapter imports them.

One installation per fresh runner process. Call before RuntimeSession, and
before resident/graph_step/graph_stage. Kernel sources remain frozen upstream.
No model construction, CUDA allocation or simulation is performed here.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ('cyclic_tree', 'resident_cyclic')
CONSUMERS = ('resident', 'graph_step', 'graph_stage')


def install():
    occupied = [name for name in NAMES + CONSUMERS if name in sys.modules]
    if occupied:
        raise RuntimeError('Install PN overlay before importing: ' + ', '.join(occupied))
    loaded = {}
    try:
        for name in NAMES:
            path = HERE / (name + '.py')
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError('Cannot load PN overlay: ' + str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            loaded[name] = module
            spec.loader.exec_module(module)
    except BaseException:
        for name, module in loaded.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise
    return {name: {'source': str(Path(module.__file__).resolve()),
                   'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in loaded.items()}
