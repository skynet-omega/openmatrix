"""Independent comparisons of complete saved trajectories and event ledgers."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def need(ok,message):
    if not ok:raise ValueError(message)


def final_pn(path):
    # Keep only the physical PN subtree; the full checkpoint also contains a
    # large anatomical manifest and integrator counters that are not state.
    compact=path/'final_state/pn_state.json'
    if compact.exists():
        state=json.loads(compact.read_text())
        archive=path/'final_state/pn_state.npz'
    else:
        state=json.loads((path/'final_state/session.json').read_text())['hybrid']['pn_online_state']
        archive=path/'final_state/session.npz'
    with np.load(archive,allow_pickle=False) as arrays:
        def unpack(x):
            if isinstance(x,dict):
                if set(x)=={'__array__'}:return arrays[x['__array__']]
                return {k:unpack(v) for k,v in x.items()}
            if isinstance(x,list):return [unpack(v) for v in x]
            return x
        return unpack(state)


def compare_pn(reference,candidate):
    records=[]
    def visit(x,y,path='PN'):
        if isinstance(x,dict):
            need(isinstance(y,dict) and set(x)==set(y),'different PN fields: '+path)
            for k in x:visit(x[k],y[k],path+'/'+k)
        elif isinstance(x,list):
            need(isinstance(y,list) and len(x)==len(y),'different PN history: '+path)
            for k,(a,b) in enumerate(zip(x,y)):visit(a,b,path+'/'+str(k))
        elif isinstance(x,np.ndarray) or isinstance(x,float):
            a,b=np.asarray(x),np.asarray(y)
            need(a.shape==b.shape and a.dtype==b.dtype,'different PN layout: '+path)
            need(np.isfinite(a).all() and np.isfinite(b).all(),'nonfinite PN: '+path)
            diff=abs(a-b)
            if a.dtype.kind in 'iub':value=float(np.max(diff,initial=0));limit=0.
            elif path.endswith('/voltage_delta_mV'):value=float(np.max(diff,initial=0));limit=2e-5
            elif path.endswith('/gates'):value=float(np.max(diff,initial=0));limit=2e-7
            else:
                value=float(np.max(diff/np.maximum(1,np.maximum(abs(a),abs(b))),initial=0));limit=1e-4
            records.append({'state':path,'value':value,'limit':limit,'pass':bool(value<=limit)})
        else:need(type(x) is type(y) and x==y,'different PN identity/clock: '+path)
    visit(final_pn(reference),final_pn(candidate))
    return {'pass':all(r['pass'] for r in records),'comparisons':records,
            'scope':'All saved PN physical states, release, receptors and delay queues at endpoint.'}


def compare(reference,candidate):
    a=json.loads((reference/'RESULT.json').read_text())
    b=json.loads((candidate/'RESULT.json').read_text())
    need(a['status']==b['status']=='COMPLETE','incomplete run')
    need(a['completed_ms']==b['completed_ms'] and a['field']==b['field'],'different experiment')
    for key in ('checkpoint','parameters','initial_cns_sha256','weights_sha256','weights_count'):
        need(a[key]==b[key],'different preparation: '+key)
    # These RNG states contain only scalar integers in the conserved loader.
    published_a=json.loads((reference/'final_state/published.json').read_text())
    published_b=json.loads((candidate/'final_state/published.json').read_text())
    need(published_a['rng']==published_b['rng'],'different final RNG state')
    comparisons=[];passed=True
    with np.load(reference/'trajectory.npz',allow_pickle=False) as za,np.load(candidate/'trajectory.npz',allow_pickle=False) as zb:
        need(set(za.files)==set(zb.files),'different recorded states')
        for key in za.files:
            x=za[key];y=zb[key]
            need(x.shape==y.shape and x.dtype==y.dtype,'different state layout: '+key)
            need(np.array_equal(x[0],y[0]),'different initial state: '+key)
            if x.dtype.kind in 'iu':
                ok=np.array_equal(x,y);value=float(np.max(abs(x-y)));limit=0.
            else:
                need(np.isfinite(x).all() and np.isfinite(y).all(),'nonfinite state: '+key)
                difference=abs(x-y)
                if key=='cns':
                    scaled=difference/(1e-7+1e-5*np.maximum(abs(x),abs(y)))
                    value=float(np.max(scaled));limit=1.
                elif key=='cell_delta':value=float(np.max(difference));limit=2e-5
                elif key=='cell_gates':value=float(np.max(difference));limit=2e-7
                else:value=float(np.max(difference/np.maximum(1,np.maximum(abs(x),abs(y)))));limit=1e-4
                ok=value<=limit
            item={'state':key,'value':value,'limit':limit,'pass':bool(ok)}
            if key=='cns':
                index=np.unravel_index(np.argmax(scaled),scaled.shape)
                item.update(max_sample=int(index[0]),max_coordinate=int(index[1]),
                            raw_max=float(np.max(difference)),p99=float(np.quantile(scaled,.99)))
            comparisons.append(item);passed &= bool(ok)
    ea=json.loads((reference/'EVENTS.json').read_text());eb=json.loads((candidate/'EVENTS.json').read_text())
    need(len(ea)==len(eb),'different physical epoch count')
    event_pass=True;event_max=0.;event_count=0;event_failures=[]
    for i,(pa,pb) in enumerate(zip(ea,eb)):
        need(all(pa[k]==pb[k] for k in ('block','start_elapsed_ns','duration_ns')),'different physical epoch')
        xa,xb=pa['events'],pb['events'];event_count+=len(xa)
        if len(xa)!=len(xb):
            event_pass=False;event_failures.append({'block':i,'counts':[len(xa),len(xb)]});continue
        for j,(x,y) in enumerate(zip(xa,xb)):
            structure=all(x[k]==y[k] for k in ('row','neuron_id','producer'))
            dt=abs(x['time_s']-y['time_s']);event_max=max(event_max,dt)
            ok=structure and dt<=1e-9
            for key in ('jump','post_q'):
                if x[key] is None or y[key] is None:ok &= x[key] is y[key]
                else:ok &= abs(x[key]-y[key])<=1e-4*max(1,abs(x[key]),abs(y[key]))
            if not ok:
                event_pass=False
                if len(event_failures)<12:event_failures.append({'block':i,'event':j,'structure_equal':structure,'time_difference_s':dt})
    passed &= event_pass
    pn=compare_pn(reference,candidate)
    passed &= pn['pass']
    return {'status':'PASS_COUPLED_NUMERICAL' if passed else 'FAIL_COUPLED_NUMERICAL',
            'reference':str(reference),'candidate':str(candidate),'simulated_ms':a['completed_ms'],
            'comparisons':comparisons,'events':{'pass':event_pass,'records_including_predictors':event_count,
                 'max_time_difference_s':event_max,'failures':event_failures},
            'final_PN':pn,
            'reference_advance_s':a['advance_total_s'],'candidate_advance_s':b['advance_total_s'],
            'advance_ratio':b['advance_total_s']/a['advance_total_s'],
            'candidate_CNS_rhs_calls':b.get('rhs_evaluations'),
            'parameters_changed':False,
            'scope':'Paired coupled trajectories, shared GPU. Not full rewrite, long-horizon or biological equivalence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'unique result')
    try:result=compare(a.reference,a.candidate)
    except ValueError as exc:result={'status':'FAIL_CONTRACT','message':str(exc)}
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))
    raise SystemExit(0 if result['status']=='PASS_COUPLED_NUMERICAL' else 1)
