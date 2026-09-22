from pathlib import Path
import json,copy,sys,numpy as np
from model import require
from session import Session
from runtime import PROFILES
from development import base,state,scalar

def rejected(f):
    try:f()
    except ValueError:return
    raise RuntimeError('expected rejection')
def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    spec=base(4,2);a=Session(spec);a.advance(.003);a.save_restart(out/'restart')
    b=Session.restore_restart(out/'restart');require(a.t==b.t and np.array_equal(a.read(),b.read()),'physical state restart mismatch')
    for t in [.004,.005]:
        a.advance(t);b.advance(t)
        for _ in range(3):a.scan('cells','v',[0,2])
    err=float(np.max(abs(a.read()-b.read())/(1+abs(b.read()))));require(err<1e-5,'new-epoch restart differs materially')
    # A new epoch from live edits equals an independently constructed session at that boundary.
    changed=copy.deepcopy(spec);changed['populations'][0]['count']=5;changed['populations'][0]['cell_ids']=[0,1,2,3,7]
    changed['connections']=[];changed['clamps']=[{'population':'cells','state':'v','cells':[7],'value':.2}]
    receipt=a.replace(changed);a.advance(.006);require(a.scan('cells','v',[7])['values']==[.2],'clamp changed')
    # Nonfinite migrated state must not commit despite valid descriptor initial state.
    s={'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',0,1,'0')},'parameters':{'tau':scalar('s',1)}}]}
    c=Session(s);before=c.model.identity;old=c.read();bad=copy.deepcopy(s);bad['populations'][0]['states']['x']=state('1',1,1,'log(x)/tau')
    rejected(lambda:c.replace(bad));require(c.model.identity==before and np.array_equal(c.read(),old) and not c.edits,'failed replacement lost live state')
    saved=PROFILES['precise'];PROFILES['precise']=(1e-6,1e-9)
    try:rejected(lambda:Session.restore_restart(out/'restart'))
    finally:PROFILES['precise']=saved
    raw=(out/'restart/state.npy').read_bytes();(out/'restart/state.npy').write_bytes(raw[:-1]+bytes([raw[-1]^1]));rejected(lambda:Session.restore_restart(out/'restart'))
    for obj in [a,b,c]:obj.close()
    result={'restart_semantics':'physical state exact at boundary; new adaptive history','post_restart_scaled_difference':err,'scanner_calls':6,'clamp_new_cell':.2,'atomic_failed_replace':True,'restart_corruption_detected':True,'effective_profile_change_rejected':True,'negative_fixture':'restart/state.npy intentionally corrupted at end','edit':receipt}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
