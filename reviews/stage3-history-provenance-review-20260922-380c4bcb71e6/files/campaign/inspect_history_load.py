"""Inspect the loaded organism's pending first sensory interval without stepping."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path.insert(0, str(OLD / 'work/motor14_20260922'))
from motor_runtime import load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    obj = None
    result = {'status': 'INCOMPLETE', 'organism_steps': 0}
    try:
        obj, diagnostic, plan, _, _, _ = load(args.out / 'inputs')
        pending = np.asarray(obj.core.pending_sensors, dtype=np.float64)
        excitation = np.asarray(obj.core.pending_excitation, dtype=np.float64)
        if pending.shape != (3,) or not np.isfinite(pending).all() or not np.isfinite(excitation).all():
            raise ValueError('Unexpected pending port shape or nonfinite state')
        manifest = OLD / plan['checkpoint'] / 'manifest.json'
        result.update({
            'status': 'COMPLETE_NO_STEP',
            'pending_sensors': pending.tolist(),
            'pending_excitation_shape': list(excitation.shape),
            'pending_excitation_nonzero': int(np.count_nonzero(excitation)),
            'hybrid_time_ns': int(obj.core.hybrid.time_ns),
            'checkpoint_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'checkpoint_path': str(OLD / plan['checkpoint']),
            'field_after_load': type(obj.core.world.boundary).__name__,
            'intervention': 'none: inspected only',
        })
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if obj is not None:
            obj.close()
        result['wall_s'] = time.perf_counter() - start
        result['rss_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        (args.out / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
