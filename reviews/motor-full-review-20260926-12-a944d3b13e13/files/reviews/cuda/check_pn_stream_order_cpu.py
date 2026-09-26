"""CPU-only call-order witness; never imports real CuPy or opens a CUDA context.

Executes the actual CyclicTree constructor under a stream spy with asynchronous
producers. It establishes missing explicit ordering, not an observed GPU fault.
An in-memory synchronization control is compared without editing the source.
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

SOURCE = Path('/home/daroch/AXIOMA_ASTRA/motor_nuevo/resident_pn_20260922/resident_cyclic.py')


def witness(code, source=SOURCE):
    queued = []
    hazards = []
    complete = set()
    current = ['caller']

    class Array:
        def __init__(self, shape, operation):
            self.shape = np.shape(shape) if not isinstance(shape, tuple) else shape
            self.producer = current[0]
            self.ticket = len(queued)
            queued.append({'operation': operation, 'stream': current[0]})

        def __len__(self):
            return self.shape[0]

        def fill(self, _):
            queued.append({'operation': 'fill', 'stream': current[0]})

    def read(value, operation):
        if isinstance(value, Array) and value.producer != current[0] and value.ticket not in complete:
            hazards.append({'operation': operation, 'producer': value.producer,
                            'consumer': current[0], 'allocation_ticket': value.ticket})

    class Stream:
        def __init__(self, non_blocking=False, name=None):
            self.name = name or 'tree_nonblocking'

        def __enter__(self):
            self.previous = current[0]
            current[0] = self.name

        def __exit__(self, *_):
            current[0] = self.previous

        def synchronize(self):
            complete.update(i for i, row in enumerate(queued) if row['stream'] == self.name)
            queued.append({'operation': 'synchronize', 'stream': self.name})

        def begin_capture(self):
            queued.append({'operation': 'begin_capture', 'stream': self.name})

        def end_capture(self):
            return object()

    class RawModule:
        def __init__(self, **_):
            pass

        def get_function(self, name):
            def call(grid, block, args, **_):
                for a in args:
                    read(a, name)
                queued.append({'operation': name, 'stream': current[0]})
            return call

    def add(a, b, *, out):
        read(a, 'add'); read(b, 'add')
        queued.append({'operation': 'add', 'stream': current[0]})

    def copyto(dst, src):
        read(src, 'copyto')
        queued.append({'operation': 'copyto', 'stream': current[0]})

    cupy = types.ModuleType('cupy')
    cupy.int64, cupy.int32 = np.int64, np.int32
    cupy.asarray = lambda a, **_: Array(np.shape(a), 'asarray')
    cupy.arange = lambda n, **_: Array((n,), 'arange')
    cupy.zeros = lambda shape, **_: Array((shape,) if isinstance(shape, int) else shape, 'zeros')
    cupy.zeros_like = lambda a: Array(a.shape, 'zeros_like')
    cupy.add, cupy.copyto, cupy.RawModule = add, copyto, RawModule
    cupy.cuda = types.SimpleNamespace(Stream=Stream,
                                     get_current_stream=lambda: Stream(name=current[0]))
    previous = sys.modules.get('cupy')
    sys.modules['cupy'] = cupy
    try:
        namespace = {'__file__': str(source), '__name__': 'cpu_stream_witness'}
        exec(compile(code, str(source), 'exec'), namespace)
        plan = types.SimpleNamespace(
            rounds=[(0, 1)], steps=np.zeros((1, 6), np.int64),
            pairs=np.zeros((1, 2), np.int64), core=np.array([0]),
            core_edges=np.empty((0, 3), np.int64), dg=np.ones(1), dm=np.ones(1),
            base=np.zeros((1, 2)), mass=np.zeros((1, 2)),
            gathers=[(np.array([0]), np.array([0, 1]), np.array([0]))])
        tree = namespace['CyclicTree'](plan, 1.0)
        names = {v.ticket: k for k, v in vars(tree).items() if isinstance(v, Array)}
        for row in hazards:
            row['array'] = names.get(row['allocation_ticket'], 'gathers')
        return {'hazards': hazards, 'calls': queued}
    finally:
        if previous is None:
            sys.modules.pop('cupy', None)
        else:
            sys.modules['cupy'] = previous


raw = SOURCE.read_text()
needle = '  with self.stream:\n   launch();'
if raw.count(needle) != 1:
    raise ValueError('Source layout changed; reassess the witness')
original = witness(raw)
control = witness(raw.replace(needle, '  cp.cuda.get_current_stream().synchronize()\n' + needle))
if not original['hazards'] or control['hazards']:
    raise RuntimeError('Expected ordering distinction was not observed')
repaired = {}
for name in ('resident_cyclic.py', 'cyclic_tree.py'):
    path = Path(__file__).resolve().parents[2] / 'pn' / name
    code = path.read_text()
    evidence = witness(code, source=path)
    if evidence['hazards']:
        raise RuntimeError('Repaired file has an unordered read: ' + str(path))
    repaired[name] = {'path': str(path), 'sha256': hashlib.sha256(code.encode()).hexdigest(),
                      'evidence': evidence}
result = {
    'source': str(SOURCE), 'sha256': hashlib.sha256(raw.encode()).hexdigest(),
    'scope': 'CPU spy of actual source; explicit ordering only, not GPU reproduction',
    'assumption': 'RawModule compilation is not relied upon as a stream barrier',
    'original': original, 'in_memory_synchronization_control': control,
    'repaired_files': repaired,
    'conclusion': 'Missing explicit producer-to-consumer stream dependency',
}
destination = Path(__file__).with_name('PN_STREAM_ORDER_CPU.json')
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'original_unsynchronized_reads': len(original['hazards']),
                  'control_unsynchronized_reads': len(control['hazards']),
                  'repaired_unsynchronized_reads': {k: len(v['evidence']['hazards']) for k, v in repaired.items()},
                  'first_original': original['hazards'][0], 'output': str(destination)}))
