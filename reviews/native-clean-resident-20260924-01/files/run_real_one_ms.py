"""One settled-sham organism millisecond; process-local resident CNS swap.

The historical loader/model is read-only. Python still owns the millisecond
coupling in this pilot; only the adaptive CNS trials move to device control.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
EPOCH = ROOT/'motor_nuevo/epoch_cost_20260923'
sys.path[:0] = [str(OLD/'work/motor14_20260922'),
                str(OLD/'work/motor13_20260922'),
                str(ROOT/'motor_nuevo/pipeline_review_20260922'), str(EPOCH)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def resident_class(parent):
    import cupy as cp
    dll = ct.CDLL(str(HERE/'libresident_controller.so'))
    ptr = ct.c_void_p
    lng = ct.c_long
    dll.resident_create.argtypes = [ptr]*6 + [lng, lng]
    dll.resident_create.restype = ptr
    dll.resident_advance.argtypes = [ptr, lng, ct.POINTER(lng), lng, lng,
                                    ct.POINTER(ct.c_double), lng, lng,
                                    ct.POINTER(lng), ct.POINTER(ct.c_double)]
    dll.resident_advance.restype = ct.c_int
    dll.resident_destroy.argtypes = [ptr]
    dll.resident_error.restype = ct.c_char_p

    class ResidentGraph(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # CuPy 13.6 destroys its source cudaGraph_t immediately after
            # graphExec instantiation, leaving self.graph.graph == 0. Capture
            # one fresh source graph through the runtime API and hand its
            # living handle to the native scheduler before destroying it.
            raw_graph = 0
            with cp.cuda.using_allocator(self.pool.malloc), self.stream:
                cp.cuda.runtime.streamBeginCapture(self.stream.ptr)
                try:
                    self.trial()
                finally:
                    raw_graph = cp.cuda.runtime.streamEndCapture(self.stream.ptr)
            try:
                self.resident = dll.resident_create(raw_graph, self.stream.ptr,
                    self.clock.data.ptr, self.status.data.ptr, self.x.data.ptr,
                    self.fine.data.ptr, self.n, 4096)
            finally:
                cp.cuda.runtime.graphDestroy(raw_graph)
            if not self.resident:
                raise RuntimeError('Resident graph setup: '+dll.resident_error().decode())
            self.resident_wall_s = 0.0

        def advance(self, ns, next_ns, min_ns, max_ns, budget=30, boundaries=None):
            with self.stream:
                cp.copyto(self.backup, self.x)
            times = np.ascontiguousarray(np.unique(boundaries)
                if boundaries is not None else [], dtype=np.float64)
            nxt = lng(next_ns)
            counts = (lng*3)()
            error = ct.c_double()
            start = time.perf_counter()
            code = dll.resident_advance(self.resident, ns, ct.byref(nxt),
                min_ns, max_ns,
                times.ctypes.data_as(ct.POINTER(ct.c_double)), len(times),
                10000, counts, ct.byref(error))
            self.resident_wall_s += time.perf_counter() - start
            if code:
                message = dll.resident_error().decode()
                with self.stream:
                    cp.copyto(self.x, self.backup)
                self.stream.synchronize()
                raise RuntimeError(message)
            self.calls += 1
            self.steps += counts[0]
            self.rejected += counts[1]
            return nxt.value, list(counts), error.value

        def close(self):
            if getattr(self, 'resident', None):
                dll.resident_destroy(self.resident)
                self.resident = None
            super().close()

    return ResidentGraph


def main() -> int:
    if not __debug__:
        raise RuntimeError('Historical loader requires normal Python')
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('reference', 'resident'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    obj = session = None
    original_class = None
    result = {'schema':'clean_resident_one_ms_v1','mode':args.mode,
              'status':'STARTING','checkpoint':'settled_700ms',
              'stage_admission':False}
    try:
        import cupy as cp
        from motor_runtime import load
        from runtime_session import RuntimeSession
        from session_io import write_state
        from operator_state import OperatorState, LEGACY_BINDINGS
        obj, _diagnostic, plan, Field, field, _ports = load(args.out/'preparation_inputs')
        if plan['checkpoint'] != 'work/stage234_settling_extension_20260915/settled_700ms':
            raise RuntimeError('Unexpected organism checkpoint')
        h = obj.core.hybrid
        initial_ns = int(h.time_ns)
        # Match the verified real-epoch host profile's physical event owners.
        for name in ('event_waveform','event_ports','event_coupling',
                     'native_cell','device_cell'):
            if name in sys.modules:
                raise RuntimeError('Unexpected event owner preloaded: '+name)
        sys.path.insert(0, str(EPOCH))
        import event_waveform,event_ports,event_coupling,native_cell,device_cell
        for module in (event_waveform,event_ports,event_coupling,native_cell,device_cell):
            if Path(module.__file__).resolve().parent != EPOCH:
                raise RuntimeError('Wrong event owner: '+module.__name__)
        import organism_adapter
        if args.mode == 'resident':
            original_class = organism_adapter.NativeGraph
            organism_adapter.NativeGraph = resident_class(original_class)
        session = RuntimeSession(h,'causal_cuda')
        _diagnostic.instalar_campo(obj, Field, field, 'sham', 0.)
        used = obj.core.pending_sensors.copy()
        step_start = time.perf_counter()
        obj.step()
        cp.cuda.runtime.deviceSynchronize()
        result['step_wall_s'] = time.perf_counter() - step_start
        result['used_sensors_sha256'] = hashlib.sha256(np.ascontiguousarray(used).tobytes()).hexdigest()
        result['time_ns'] = int(h.time_ns)
        if result['time_ns'] != initial_ns + 1000000:
            raise RuntimeError('One-ms physical clock mismatch')
        state_dir = args.out/'state'
        state_dir.mkdir()
        write_state(state_dir/'session',obj.core.state_dict())
        write_state(state_dir/'prosthesis',obj.state())
        write_state(state_dir/'published',{'rates':obj.core.brain.rates,
            'time_ns':obj.core.brain.time_ns,
            'rng':obj.core.brain.rng.bit_generator.state})
        write_state(state_dir/'effective_operator',
                    OperatorState(h,LEGACY_BINDINGS).state_dict())
        save_json(state_dir/'boundary.json',obj.core.world.boundary.metadata())
        result['runtime'] = session.report()
        if args.mode == 'resident':
            result['resident_cns_wall_s'] = session.adapter.core.resident_wall_s
        result['status'] = 'COMPLETE'
    except BaseException as exc:
        result['status'] = 'INCOMPLETE'
        result['error'] = {'type':type(exc).__name__, 'message':str(exc),
                           'traceback':traceback.format_exc()}
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException as exc:
                result.setdefault('cleanup_errors',[]).append(repr(exc))
        if original_class is not None:
            import organism_adapter
            organism_adapter.NativeGraph = original_class
        if obj is not None:
            try:
                obj.close()
            except BaseException as exc:
                result.setdefault('cleanup_errors',[]).append(repr(exc))
        result['wall_total_s'] = time.perf_counter() - start
        result['rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        result['source_hashes'] = {str(p):sha(p) for p in (
            HERE/'run_real_one_ms.py',HERE/'resident_controller.cu',
            HERE/'libresident_controller.so',
            ROOT/'campanas/etapa3_motor_nuevo_20260922/graph_core.py',
            ROOT/'campanas/etapa3_motor_nuevo_20260922/organism_adapter.py',
            ROOT/'motor_nuevo/native_hybrid_20260922/graph_control_v2.cpp')}
        if result.get('cleanup_errors'):
            result['status'] = 'INCOMPLETE_CLEANUP'
        save_json(args.out/'RESULT.json',result)
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
