"""Reconstruct comparisons from arrays; no runner PASS flags are trusted."""
from pathlib import Path
import argparse,json
import numpy as np

def read_state(stem):
    stem=Path(stem);descriptor=json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:
        def decode(v):
            if isinstance(v,dict):
                if set(v)=={'__array__'}:return z[v['__array__']].copy()
                return {k:decode(x) for k,x in v.items()}
            if isinstance(v,list):return [decode(x) for x in v]
            return v
        return decode(descriptor)

def flatten(v,path=''):
    if isinstance(v,dict):
        for k,x in v.items():yield from flatten(x,path+'/'+k)
    elif isinstance(v,list):
        for k,x in enumerate(v):yield from flatten(x,path+'/'+str(k))
    else:yield path,v

def leaves(run):
    result=dict(flatten(read_state(Path(run)/'brain_final'),'/brain'))
    for file in ('traces','body_final'):
        with np.load(Path(run)/(file+'.npz'),allow_pickle=False) as z:
            for k in z.files:result['/'+file+'/'+k]=z[k].copy()
    return result

def compare(reference,candidate):
    a,b=leaves(reference),leaves(candidate);errors={};different=[];nonfinite=[];shape=[]
    keys=sorted(set(a)|set(b));exact=True
    for k in keys:
        if k not in a or k not in b:shape.append(k);exact=False;continue
        x,y=a[k],b[k]
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            if x.dtype.kind in 'fciu' and (not np.isfinite(x).all() or not np.isfinite(y).all()):nonfinite.append(k)
            same=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            if x.shape!=y.shape or x.dtype!=y.dtype:shape.append(k)
            elif x.dtype.kind in 'fciu':errors[k]=float(np.max(np.abs(x.astype(float)-y.astype(float)),initial=0))
        else:
            same=type(x)==type(y) and x==y
            if type(x) in (float,int) and type(y) in (float,int):
                if not np.isfinite([x,y]).all():nonfinite.append(k)
                errors[k]=abs(float(x)-float(y))
        if not same:exact=False;different.append(k)
    def maximum(predicate):return max((e for k,e in errors.items() if predicate(k)),default=None)
    count_paths=[k for k in keys if k.endswith(('/counts','/clipped','/spike_count','/clipped_spike_events'))]
    event_differences=[k for k in count_paths if k in different or k in shape]
    static_roots={'parameters','photo_ids','visual_ids','schema','visual_output_connected'}
    static_differences=[k for k in different if k.startswith('/brain/') and
        (k.split('/')[2].endswith(('_manifest','_migration')) or k.split('/')[2] in static_roots)]
    metrics={'PN_voltage_max_abs_mV':maximum(lambda k:k.endswith('/voltage_delta_mV')),
        'normalized_state_max_abs':errors.get('/brain/state'),
        'gates_max_abs':maximum(lambda k:k.endswith('/gates')),
        'yaw_trace_max_abs_deg':errors.get('/traces/yaw_delta_deg'),
        'root_position_max_abs_m':float(np.max(np.abs(a['/body_final/qpos'][:3]-b['/body_final/qpos'][:3]))),
        'final_event_counts_exact':not event_differences,'finite':not nonfinite,
        'endpoint_clock_exact':a['/brain/time_ns']==b['/brain/time_ns'],
        'model_metadata_exact':not static_differences}
    # Read the frozen contract; package hashes authenticate this exact criterion.
    contract=json.loads((Path(__file__).resolve().parent/'INTEGRATED_ADAPTIVE_CONTRACT.json').read_text())['predeclared_screen_bounds']
    limits={metric:contract[key] for metric,key in {
        'PN_voltage_max_abs_mV':'PN_voltage_mV','normalized_state_max_abs':'normalized_state',
        'gates_max_abs':'gates','yaw_trace_max_abs_deg':'yaw_degrees','root_position_max_abs_m':'root_position_m'}.items()}
    if any(not np.isfinite(v) or v<=0 for v in limits.values()):raise ValueError('Invalid frozen bounds')
    numerical=all(metrics[k] is not None and metrics[k]<=v for k,v in limits.items()) and all(metrics[k] for k in ('finite','final_event_counts_exact','endpoint_clock_exact','model_metadata_exact'))
    return {'reference':str(reference),'candidate':str(candidate),'exact_all_leaves':exact,'leaf_count':len(keys),
        'different_paths':different,'layout_or_missing_paths':shape,'nonfinite':nonfinite,
        'metrics':metrics,'limits':limits,'screen_pass':numerical,'event_differences':event_differences,'static_differences':static_differences,'all_numeric_errors':errors}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--out',required=True)
    a=p.parse_args();result=compare(a.reference,a.candidate)
    Path(a.out).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['exact_all_leaves','leaf_count','metrics','screen_pass','event_differences']}))
