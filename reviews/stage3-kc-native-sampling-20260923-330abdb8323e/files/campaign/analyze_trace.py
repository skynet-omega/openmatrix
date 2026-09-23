"""Reconstruct per-sample comparisons; no threshold tuning or time alignment fit."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok: raise ValueError(msg)

def load(name):
    root=HERE/name
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='COMPLETE',name+' incomplete')
    z=np.load(root/'kc_native_0/trace.npz',allow_pickle=False)
    return root,z,result

def main():
    ar,a,ra=load('native_on_01');br,b,rb=load('reference_on_01')
    require(np.array_equal(a['selected_ids'],b['selected_ids']),'identity mismatch')
    require(np.all(np.isfinite(a['values'])) and np.all(np.isfinite(b['values'])),'nonfinite capture')
    rows=[]
    for channel in range(len(a['counts'])):
        am=a['meta'][channel,:a['counts'][channel]];av=a['values'][channel,:a['counts'][channel]]
        bm=b['meta'][channel,:b['counts'][channel]];bv=b['values'][channel,:b['counts'][channel]]
        for call in range(4):
            ai=np.flatnonzero(am[:,0]==call);bi=np.flatnonzero(bm[:,0]==call)
            require(len(ai)>0 and len(bi)>0,'missing capture')
            _,ix,iy=np.intersect1d(am[ai,1],bm[bi,1],assume_unique=True,return_indices=True)
            require(len(ix)>0,'no common timestamps')
            aa=av[ai[ix]];bb=bv[bi[iy]]
            row={'neuron_id':int(a['selected_ids'][channel//13]),'compartment':channel%13,
                 'call':call,'phase':'accepted' if call%2 else 'predictor',
                 'samples_native_reference':[len(ai),len(bi)],'exact_common_times':len(ix),
                 'max_abs_diff_at_common_times':{
                    name:float(np.max(np.abs(aa[:,column]-bb[:,column])))
                    for name,column in [('voltage',0),('raw_increment',2),('pre_trough',4),('q',5),('s',6)]},
                 'accepted_h_ns_native':np.unique(am[ai,2]).tolist(),
                 'accepted_h_ns_reference':np.unique(bm[bi,2]).tolist(),
                 'event_times_ns_native':am[ai[av[ai,8]!=0],1].tolist(),
                 'event_times_ns_reference':bm[bi[bv[bi,8]!=0],1].tolist()}
            rows.append(row)
    inputs=[]
    for call in range(4):
        with np.load(ar/f'kc_native_0/input_{call:02d}.npz') as x,np.load(br/f'kc_native_0/input_{call:02d}.npz') as y:
            require(x.files==y.files,'input schema')
            diffs={k:float(np.max(np.abs(x[k]-y[k]))) for k in x.files}
            inputs.append({'call':call,'all_exact':all(v==0 for v in diffs.values()),'max_absolute_differences':diffs})
    result={'schema':'kc_common_timestamp_trace_comparison_v1','rows':rows,'inputs':inputs,
            'restore_checks':{name:r['runtime']['partition'].get('restore_checks') for name,r in [('native',ra),('reference',rb)]},
            'scope':'Selected cells, first two epochs, exact common timestamps only; raw increments and trough retain different sample histories. No trajectory interpolation or causal motor certificate.',
            'stage3_admission':False}
    (HERE/'TRACE_COMPARISON.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'first_input_exact':inputs[0]['all_exact'],'restore_checks':result['restore_checks'],
        'maximum_voltage_difference_common_times':max(r['max_abs_diff_at_common_times']['voltage'] for r in rows),
        'maximum_q_difference_common_times':max(r['max_abs_diff_at_common_times']['q'] for r in rows),
        'rows_with_different_step_grid':sum(r['accepted_h_ns_native']!=r['accepted_h_ns_reference'] for r in rows)}))

if __name__=='__main__':main()
