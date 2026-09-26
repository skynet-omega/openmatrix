"""ChatGPT-proposed signed-error identity; calculation by Codex on real data."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify_clock import verify,raw,geometry


def main():
    start=time.process_time()
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    contract=json.loads((HERE/'ACCOUNTING_PLAN.json').read_text())
    verified=verify(HERE/'run_02')
    source=np.asarray(json.loads((HERE/'inputs/donor_GAUSSIAN_SPEC.json').read_text())['minus']['source_mm'])
    results={}
    for name in verified['arms']:
        trace=raw(HERE/'run_02'/name/'trace.npz');g=geometry(trace['qpos'],source)
        e=np.deg2rad(g['signed_error_deg'][1019:]);b=np.deg2rad(g['bearing_deg'][1019:]);y=np.deg2rad(g['yaw_deg'][1019:])
        U=trace['yaw_rad_s'][1020:]*.001
        if len(U)!=980 or not np.array_equal(trace['step'][1020:],np.arange(1021,2001)):
            raise ValueError('Wrong command intervals')
        eb=(e[:-1]+e[1:])*.5
        G=eb*np.diff(b);C=-eb*U;M=eb*(U-np.diff(y));dv=(e[-1]**2-e[0]**2)*.5
        residual=float(abs(np.sum(G+C+M)-dv))
        if not np.isfinite([G,C,M]).all() or residual>contract['tolerance_rad2']:
            raise ValueError('Accounting identity failed')
        results[name]=dict(G_geometry_rad2=float(np.sum(G)),C_command_rad2=float(np.sum(C)),
            M_command_minus_actual_rad2=float(np.sum(M)),delta_error_squared_over2_rad2=float(dv),
            residual_rad2=residual,command_adverse_ms=int(np.count_nonzero(C>0)),
            command_corrective_ms=int(np.count_nonzero(C<0)),command_zero_ms=int(np.count_nonzero(C==0)),
            classification='DELIVERED_COMMAND_NET_ADVERSE' if np.sum(C)>0 else ('DELIVERED_COMMAND_NET_CORRECTIVE' if np.sum(C)<0 else 'ZERO_COMMAND_CONTRIBUTION'))
    if time.process_time()-start>contract['CPU_s_max']:raise ValueError('Diagnostic CPU budget')
    r=dict(schema='signed_error_command_accounting_v1',results=results,
           contract_sha256=hashlib.sha256((HERE/'ACCOUNTING_PLAN.json').read_bytes()).hexdigest(),
           code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           first_command_row=1021,initial_completed_state_ms=1020,final_completed_state_ms=2000,
           interval_count=980,interval_seconds=.001,
           interpretation='Accounting identity only. G/C/M are not independent causal effects. M includes interface/timing/tracking, not pure mechanics. Closing identity cannot itself validate U clock because U cancels; raw-clock/context checks are separate in verifier.',
           reviewer_origin='ChatGPT response e767b18f-b9db-4855-b066-02738c24fa22; no actual file access or execution by reviewer',
           stage4_admission=False,stage5_admission=False,new_physical_or_neural_steps=0)
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))


if __name__=='__main__':main()
