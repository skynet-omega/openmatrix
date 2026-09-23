"""C++ source-filter discrepancy on real accepted KC SET events, with quadrature controls."""
from pathlib import Path
import json,subprocess,hashlib
import numpy as np
from scipy.integrate import quad

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'etapa3_dnb05_native_20260923_12'
COLS=('max_q_error','max_s_error','l1_q_s','l1_s_s','final_q_error','final_s_error')

def events(path,ids):
    blocks=json.loads(path.read_text())['blocks']
    origin=blocks[0]['start_elapsed_ns']
    out={int(i):[] for i in ids}
    for b in blocks:
        if b['duration_ns']!=125000:continue
        for e in b['events']:
            if e['producer']=='gamma_cuda' and e['neuron_id'] in out:
                if e['post_q'] is None:raise ValueError('Missing physical SET post')
                out[e['neuron_id']].append(((b['start_elapsed_ns']-origin)*1e-9+e['time_s'],e['post_q']))
    return out

def main():
    out=HERE/'filter_error_01';out.mkdir(exist_ok=False)
    plan={'scope':'Linear q/s source discrepancy only; does not bound recurrent CNS or body',
          'real_selected_cells':4,'controls':4,'quadrature_atol':1e-11,
          'operator':'q decays exponentially and is SET at published physical events; s_dot=(q-s)/ts; common initial s cancels',
          'horizon_s':0.02,'gain':1.,'no_new_stage3_or_numeric_tolerance':True}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    exe=out/'filter_error'
    subprocess.run(['g++','-O2','-std=c++17','-ffp-contract=off',str(HERE/'filter_error.cpp'),'-o',str(exe)],check=True)
    def run(tq,ts,q0,end,a,b):
        lines=[f'{tq:.17g} {ts:.17g} {q0:.17g} {end:.17g} {len(a)} {len(b)}']
        lines += [f'{t:.17g} {q:.17g}' for t,q in a+b]
        values=np.fromstring(subprocess.check_output([str(exe)],input='\n'.join(lines)+'\n',text=True),sep=' ')
        if len(values)!=6 or not np.isfinite(values).all():raise ValueError('Invalid C++ result')
        return dict(zip(COLS,map(float,values)))
    controls=[]
    # Independent quadrature evaluates the closed-form individual filter responses,
    # rather than the C++ difference recurrence or its extrema roots.
    for tq,ts,delay,amp in ((.01,.02,.00001,.3),(.02,.01,.00002,.7),(.01,.01,.000015,.2),(.005,.03,0.,.4)):
        end=.05;t0=.01;a=[(t0,amp)];b=[(t0+delay,amp)];r=run(tq,ts,0,end,a,b)
        def response(t,ev,filtered):
            u=t-ev
            if u<0:return 0.
            if not filtered:return amp*np.exp(-u/tq)
            return amp*u/ts*np.exp(-u/ts) if tq==ts else amp*tq/(tq-ts)*(np.exp(-u/tq)-np.exp(-u/ts))
        breaks=sorted(set([0.,t0,t0+delay,end]))
        errors={}
        for filtered,key in ((False,'l1_q_s'),(True,'l1_s_s')):
            value=sum(quad(lambda t:abs(response(t,t0,filtered)-response(t,t0+delay,filtered)),l,u,epsabs=1e-13,epsrel=1e-10)[0]
                      for l,u in zip(breaks,breaks[1:]))
            errors[key]=abs(value-r[key])
        if max(errors.values())>plan['quadrature_atol']:raise ValueError('Quadrature control failed')
        controls.append({'tq':tq,'ts':ts,'delay':delay,'errors':errors})
    z=np.load(HERE/'native_on_01/kc_native_0/input_00.npz',allow_pickle=False)
    ids=z['selected_ids'];aa=events(PARENT/'native_sham_20_01/EVENT_AUDIT.json',ids)
    bb=events(PARENT/'reference_sham_20_01/EVENT_AUDIT.json',ids)
    rows=[]
    for k,body in enumerate(ids):
        a=aa[int(body)];b=bb[int(body)]
        rows.append({'neuron_id':int(body),'events_each':[len(a),len(b)],'tau_s':float(z['tau'][k]),
            'synaptic_tau_s':float(z['ts']),**run(float(z['tau'][k]),float(z['ts']),float(z['state_q'][k]),.02,a,b)})
    result={'plan':plan,'rows':rows,'controls':controls,'source_sha256':hashlib.sha256((HERE/'filter_error.cpp').read_bytes()).hexdigest(),
            'stage3_admission':False,'limitation':'Unit-gain linear source filter; does not include axonal gain dynamics or recurrent amplification. No biological or whole-motor certificate.'}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':rows,'controls_pass':True}))

if __name__=='__main__':main()
