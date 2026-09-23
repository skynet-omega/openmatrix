"""Reconstruct recorded control using real event boundaries; no alternative simulation."""
from pathlib import Path
import json,math
HERE=Path(__file__).resolve().parent
def main():
    p=json.loads((HERE/'mode1_01/SONDA.json').read_text())
    events=json.loads((HERE/'mode1_01/EVENT_AUDIT.json').read_text())['blocks']
    values=json.loads((HERE/'CONTROL_PARAMETERS.json').read_text())['values']
    minimum=values['minimum_step_ns'];maximum=values['maximum_step_ns'];out=[]
    for ix,epoch in enumerate(p['epochs']):
        e=events[ix];stop_times=sorted(set(float(z['time_s']) for z in e['events']));end=epoch['duration_ns']*1e-9
        if len(e['events'])!=epoch['event_boundaries'] or e['duration_ns']!=epoch['duration_ns']:raise ValueError('Native/physical correspondence')
        nxt=epoch['next_in'];used=0.;tails=low_errors=interior=0;min_h=maximum;maxerr=0.
        rows=[r for r in p['rows'] if r['epoch']==ix]
        for row in rows:
            t=float.fromhex(row['t_hex']);dt=float.fromhex(row['h_hex']);err=float.fromhex(row['status_hex'][0]);maxerr=max(maxerr,err)
            bound=next((x for x in stop_times if x>used),end);available=bound-used;requested=min(nxt,maximum)*1e-9
            if t!=used or dt!=min(available,requested):raise ValueError('Controller readback mismatch')
            tails+=int(dt<requested);interior+=int(dt<available);low_errors+=int(err<.1)
            if row['committed']:
                used=bound if dt==available else used+dt
                nxt=min(maximum,max(minimum,math.floor(dt*1e9*(2 if err<.1 else 1))))
            else:nxt=math.floor(dt*1e9*.5)
            min_h=min(min_h,dt*1e9)
        if used!=end or nxt!=epoch['next_out']:raise ValueError('Final control mismatch')
        out.append({'epoch':ix,'trials':len(rows),'events':len(stop_times),
                    'minimum_segments':len([x for x in stop_times if 0<x<end])+1,
                    'event_or_epoch_tails':tails,'steps_inside_segments':interior,
                    'errors_below_0_1':low_errors,'max_error_ratio':maxerr,'min_h_ns':min_h})
    result={'same_recorded_controller_reconstructed_exactly':True,
      'scope':'Actual recorded decisions and physical event times; segment count is not an admissible alternative simulation or speedup.',
      'minimum_step_ns':minimum,'maximum_step_ns':maximum,'epochs':out,
      'trials':sum(x['trials'] for x in out),'segments':sum(x['minimum_segments'] for x in out),
      'event_or_epoch_tails':sum(x['event_or_epoch_tails'] for x in out),
      'steps_inside_segments':sum(x['steps_inside_segments'] for x in out),
      'max_error_ratio':max(x['max_error_ratio'] for x in out)}
    (HERE/'CONTROL_READBACK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='epochs'}))
if __name__=='__main__':main()
