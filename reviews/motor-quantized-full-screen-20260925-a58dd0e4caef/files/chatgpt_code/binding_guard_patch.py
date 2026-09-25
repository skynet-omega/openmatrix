"""Repair of an observer binding; not a model or integrator change.
Use in an isolated worker: dictionary patching is not thread isolation.
No CUDA imports/execution in this module. Recorder preserves the guard's result.
"""
from contextlib import contextmanager
import dis
import inspect
import time


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _functions(brain):
    found = {}
    values = [getattr(brain, 'coefficients_gpu')]
    values += [vars(c).get('coefficients_gpu') for c in type(brain).__mro__]
    for value in values:
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        value = getattr(value, '__func__', value)
        chain = set()
        while inspect.isfunction(value) and id(value) not in chain:
            chain.add(id(value)); found[id(value)] = value
            value = getattr(value, '__wrapped__', None)
    return list(found.values())


@contextmanager
def private_assemble(brain, layout, replacement):
    """Patch actual function global dictionaries, including imported aliases.
    Refuse captured assembler defaults/closures; do not silently rewrite them.
    tap.rounds==3 and same-run output comparison remain mandatory afterwards.
    """
    old = layout.assemble
    need(callable(old) and callable(replacement) and old is not replacement,
         'Invalid assembler replacement')
    sites = {(id(vars(layout)), 'assemble'): (vars(layout), 'assemble')}
    evidence = []
    for fn in _functions(brain):
        defaults = list(fn.__defaults__ or ()) + list((fn.__kwdefaults__ or {}).values())
        cells = []
        for cell in fn.__closure__ or ():
            try:
                cells.append(cell.cell_contents)
            except ValueError:
                pass
        need(not any(v is old for v in defaults + cells),
             'Assembler captured in defaults/closure: explicit adapter required')
        for ins in dis.get_instructions(fn):
            if ins.opname != 'LOAD_GLOBAL':
                continue
            name = ins.argval
            value = fn.__globals__.get(name)
            if name == 'assemble':
                need(value is old, 'Actual method uses another assembler/module')
            if value is old:
                sites[(id(fn.__globals__), name)] = (fn.__globals__, name)
                evidence.append({'function': fn.__qualname__,
                                 'module': fn.__globals__.get('__name__'),
                                 'file': fn.__code__.co_filename, 'binding': name})
    need(evidence, 'No direct assembler global found in actual coefficients_gpu chain')
    saved = [(namespace, name, namespace[name]) for namespace, name in sites.values()]
    try:
        for namespace, name, expected in saved:
            need(namespace[name] is expected, 'Binding changed before patch')
            namespace[name] = replacement
        yield evidence
    finally:
        altered = []
        for namespace, name, original in reversed(saved):
            if namespace.get(name) is not replacement:
                altered.append(namespace.get('__name__', '?') + '.' + name)
            namespace[name] = original
        need(not altered, 'Concurrent binding modification, restored: ' + repr(altered))


def meter_guard(guard):
    """Install timing only. Returns (records, restore). Same trees and decisions.
    fingerprint bytes exclude transfers hidden inside state_dict/reader.
    Record all failures; do not report an incomplete byte count as total traffic.
    """
    records = []
    original_read, original_reader = guard.read, guard.reader
    missing = object()
    prior_read = vars(guard).get('read', missing)
    elapsed_reader = [0.0]

    def reader():
        start = time.perf_counter()
        try:
            return original_reader()
        finally:
            elapsed_reader[0] += time.perf_counter() - start

    def read():
        elapsed_reader[0] = 0.0
        start = time.perf_counter(); result = None
        status = 'FAILED'
        try:
            result = original_read()
            status = 'COMPLETE'
            return result
        finally:
            wall = time.perf_counter() - start
            records.append({'read': len(records) + 1, 'status': status,
                'wall_s_including_wait': wall,
                'reader_s_including_serializers': elapsed_reader[0],
                'other_guard_s_including_wait_hash_copy': wall - elapsed_reader[0],
                'fingerprint_D2H_bytes_only': None if result is None else sum(
                    v['gpu_read_bytes'] for v in result.values()),
                'serialized_leaf_bytes_hashed': None if result is None else sum(
                    leaf.get('bytes', 0) for v in result.values()
                    for leaf in v['leaves'].values()),
                'reader_internal_D2H_bytes': 'NOT_MEASURED'})

    guard.reader, guard.read = reader, read

    def restore():
        guard.reader = original_reader
        if prior_read is missing:
            vars(guard).pop('read', None)
        else:
            guard.read = prior_read
    return records, restore
