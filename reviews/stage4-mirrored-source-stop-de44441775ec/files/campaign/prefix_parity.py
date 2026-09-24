"""Describe valid 391-ms numerical prefix; never certify the 400-ms gate."""
from pathlib import Path
import hashlib
import json
import numpy as np
from verify_mirror import check_trace

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    native=HERE/'native_plus_01/traces.npz'
    reference=HERE/'reference_plus_01/traces.npz'
    with np.load(native,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(reference,allow_pickle=False) as z:b={k:z[k] for k in z.files}
    ma=check_trace(a,'plus');mb=check_trace(b,'plus')
    if ma['steps']!=400 or mb['steps']!=391 or not np.array_equal(a['CNS_time_ns'][:len(b['CNS_time_ns'])],b['CNS_time_ns']):
        raise ValueError('Unexpected prefix identity or length')
    n=mb['steps']
    yaw_error=np.abs(ma['yaw'][:n]-mb['yaw'])
    command_error=np.abs(ma['command'][:n]-mb['command'])
    out={'schema':'stage4_mirrored_partial_reference_prefix_v1',
         'scope':'Preliminary comparison only; reference ended by predeclared wall budget at 391/400 ms, so the full pair is not numerically confirmed.',
         'native_trace_sha256':sha(native),'reference_trace_sha256':sha(reference),
         'reference_result_sha256':sha(HERE/'reference_plus_01/RESULT.json'),
         'source_sha256':sha(__file__),'aligned_trial_ms':n,
         'yaw_sup_deg':float(yaw_error.max()),
         'command_L1_deg':float(np.rad2deg(command_error.sum()*.001)),
         'all_numeric_finite':bool(np.isfinite(yaw_error).all() and np.isfinite(command_error).all()),
         'full_gate_admission':False}
    with (HERE/'PREFIX_PARITY_01.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))

if __name__=='__main__':main()
