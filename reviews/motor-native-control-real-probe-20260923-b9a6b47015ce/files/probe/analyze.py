"""Reconstruct native stream attribution only after scientific/decision neutrality."""
from pathlib import Path
import json,hashlib
import numpy as np
from compare_state import FIELDS,compare

HERE=Path(__file__).resolve().parent
CASES=('parent_01','mode0_01','mode1_01')
def get(p):return json.loads(p.read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=get(HERE/'PLAN.json');neutral={};wall=0.
    parent=HERE/CASES[0]/'final_state';a=get(parent/'session.json')
    for name in CASES:
        root=HERE/name;r=get(root/'RESULT.json')
        require(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Incomplete '+name)
        require(r['wall_total_s']<=plan['budget']['wall_each_s_max'],'Individual budget')
        wall+=r['wall_total_s'];b=get(root/'final_state/session.json');different=[]
        with np.load(parent/'session.npz',allow_pickle=False) as za,np.load(root/'final_state/session.npz',allow_pickle=False) as zb:
            for field in FIELDS:compare(a[field],b[field],za,zb,field,different)
        aux={k:sha(parent/k)==sha(root/'final_state'/k) for k in ('published.npz','prosthesis.npz','boundary.json')}
        require(not different and all(aux.values()),'Scientific state changed '+name)
        require(get(root/'EVENT_AUDIT.json')==get(HERE/CASES[0]/'EVENT_AUDIT.json'),'Physical event log changed '+name)
        neutral[name]={'science_exact':True,'auxiliary_exact':aux,'events_exact':True}
    require(wall<=plan['budget']['wall_total_s_max'],'Aggregate budget')
    probes={mode:get(HERE/f'mode{mode}_01/SONDA.json') for mode in (0,1)}
    for mode,p in probes.items():
        require(p['mode']==mode and not p['overflow'] and not p['timing_query_failed'],'Invalid probe')
        require(all(x['done'] for x in p['rows']),'Incomplete trial')
        require(all(x['return_code']==0 for x in p['epochs']),'Native call failed')
    decision=('epoch','t_hex','h_hex','status_hex','done','committed')
    require([{k:x[k] for k in decision} for x in probes[0]['rows']]==[{k:x[k] for k in decision} for x in probes[1]['rows']],'Native trial decisions differ')
    fields=('duration_ns','event_boundaries','next_in','next_out','return_code','counts_if_success','maxerr_hex_if_success')
    require([{k:x[k] for k in fields} for x in probes[0]['epochs']]==[{k:x[k] for k in fields} for x in probes[1]['epochs']],'Native epoch decisions differ')
    rows=probes[1]['rows'];stream=np.asarray([x['stream_ms'] for x in rows]);host=np.asarray([x['host_api_ms'] for x in rows])
    require(np.isfinite(stream).all() and np.isfinite(host).all(),'Nonfinite timing')
    require(np.all(stream[:,:3]>=0),'Missing mandatory timing markers')
    stream_sums={k:float(np.sum(np.where(stream[:,i]>=0,stream[:,i],0))) for i,k in enumerate(probes[1]['stream_columns'])}
    host_sums={k:float(np.sum(host[:,i])) for i,k in enumerate(probes[1]['host_columns'])}
    epoch_wall={str(k):sum(e['wall_ms'] for e in p['epochs']) for k,p in probes.items()}
    native=epoch_wall['1'];fraction=stream_sums['graph_interval']/native
    output={'scientific_neutrality':neutral,'native_decisions_exact':True,'trials':len(rows),'native_calls':len(probes[1]['epochs']),
            'accepted':sum(x['committed'] for x in rows),'stream_intervals_ms':stream_sums,'host_api_ms':host_sums,
            'native_epoch_wall_ms':epoch_wall,'instrumentation_wall_ratio_mode1_mode0':native/epoch_wall['0'],
            'graph_interval_fraction_native_mode1':fraction,'runner_wall_s':wall,
            'probe_setup':{name:get(HERE/name/'PROBE_SETUP.json') for name in CASES},
            'stage3_admission':False,'optimization_promoted':False,
            'limitations':['One physical ms including discarded predictor work; not 400ms performance',
                          'Graph interval is not exclusive kernel time; host and stream clocks overlap',
                          'Instrumented timing may differ from the parent; no extrapolation to whole-engine speedup']}
    (HERE/'RESULT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('trials','native_calls','stream_intervals_ms','native_epoch_wall_ms','instrumentation_wall_ratio_mode1_mode0','graph_interval_fraction_native_mode1','runner_wall_s')}))
if __name__=='__main__':main()
