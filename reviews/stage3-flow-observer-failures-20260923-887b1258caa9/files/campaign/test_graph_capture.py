"""Native CUDA-graph buffer-retention test, independent of the fly model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cupy as cp
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph


def main() -> None:
    seen = []
    def coefficient(z):
        target = 0.2 + 0.5*z
        rate = cp.full_like(z, 1000.)
        seen.append((z, target, rate))
        return target, rate
    graph = NativeGraph(np.array([0.1],dtype=np.float64), coefficient,
                        rtol=1e-6, atol=1e-9, norm_size=1,
                        native_library=ROOT/'motor_nuevo/native_hybrid_20260922/libgraph_control_v2.so')
    try:
        if len(seen) != 18:
            raise ValueError(f'Graph build made {len(seen)} callbacks, expected 18')
        last_state, last_target, _ = seen[-1]
        target_before = float(cp.asnumpy(last_target)[0])
        nxt, counts, error = graph.advance(1_000_000,125_000,1000,125_000)
        state_after = float(cp.asnumpy(last_state)[0])
        target_after = float(cp.asnumpy(last_target)[0])
        if len(seen) != 18 or counts[0] <= 0 or counts[1] < 0:
            raise ValueError('Replay or acceptance count changed callback contract')
        if abs(target_after-(0.2+0.5*state_after)) > 1e-12:
            raise ValueError('Captured target/state no longer match live CUDA graph')
        if abs(target_after-target_before) < 1e-7:
            raise ValueError('Retained buffer did not update on CUDA graph replay')
        result = dict(schema='native_graph_capture_fixture_v1',callbacks=len(seen),
                      accepted=int(counts[0]),rejected=int(counts[1]),
                      target_before=target_before,target_after=target_after,
                      state_stage_after=state_after,final_state=float(cp.asnumpy(graph.x)[0]),
                      next_step_ns=nxt,max_error=error,
                      scope='One-state synthetic CUDA graph; verifies pointer replay only, not fly equation coverage.')
        (HERE/'GRAPH_CAPTURE_FIXTURE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    finally:
        graph.close()


if __name__ == '__main__':
    main()
