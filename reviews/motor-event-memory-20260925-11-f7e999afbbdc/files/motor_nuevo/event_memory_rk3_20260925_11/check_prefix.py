"""Read an already saved prefix; never advances or changes the organism."""
from pathlib import Path
import argparse,json
import numpy as np
import compare_stable

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'candidate_1000ms_01')
    a=p.parse_args();run=a.run.resolve()
    r=compare_stable.compare(run,prefix=True)
    z=np.load(run/'traces_prefix.npz');old=np.load(HERE.parent/'equivalence_1s_20260925_09/optimized_1000ms_01/traces.npz')
    n=r['completed_ms']
    parent_commands=all(np.array_equal(z[k],old[k][:n]) for k in ('command_forward_mm_s','motor_filter_applied_rad_s'))
    out={'completed_prefix_ms':n,'stable_screen_pass':all(r['engineering_screen'].values()),
         'same_parent_commands':parent_commands,'stable_exact_trace_fields':sum(v['exact'] for v in r['fields'].values()),
         'first_differences':r['first_differences'],'motor_commands':r['motor_commands']}
    # Replace only this live read-only monitoring summary, not trial evidence.
    (HERE/'LIVE_COMPARISON.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2))
    if not out['stable_screen_pass'] or not parent_commands:raise SystemExit(2)

if __name__=='__main__':main()
