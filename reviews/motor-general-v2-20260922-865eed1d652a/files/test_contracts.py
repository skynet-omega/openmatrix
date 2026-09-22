from pathlib import Path
import sys,json,copy,hashlib,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import runtime
from runtime import Engine
from model import Model,require,unit,digest
from development import base,state,scalar
H=Path(__file__).resolve().parent

def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise RuntimeError('expected rejection')

def literal_spec(rhs):
    return {'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',0,1,rhs)},'parameters':{'tau':scalar('s',1)}}]}

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);result={}
    cases=['(9007199254740993-9007199254740992)/tau','((3**34+1)-(3**34))/tau','(1/10)/tau']
    values=[]
    for expr in cases:
        m=Model(literal_spec(expr));e=Engine(m.spec);a=m.rhs(0,m.initial);b=e.gpu.rhs_read(0,m.initial)
        require(np.allclose(a,b,rtol=1e-14,atol=0),'literal CPU/GPU mismatch');values.append({'expression':expr,'cpu':a.tolist(),'gpu':b.tolist()})
    result['FP64_literals']=values
    a={'version':1,'populations':[{'id':'p','count':1,'states':{'x':state('1',1,1,'-rate*x'),'y':state('1',1,1,'-rate*y')},'parameters':{'rate':scalar('1/s',1)}}],'mass':{'row':[0,1],'col':[0,1],'values':[1,2]}}
    b=copy.deepcopy(a);b['populations'][0]['states']=dict(reversed(list(b['populations'][0]['states'].items())))
    ma,mb=Model(a),Model(b);require(digest(a)==digest(b) and ma.identity!=mb.identity,'executable identity collision')
    result['ordered_identity']={'same_source_digest':True,'different_execution_identity':True,'before':ma.identity,'after':mb.identity}
    for backend in ['cpu','gpu']:
        e=Engine(literal_spec('0'),backend=backend);e.advance(.01);old=e.read();before=e.model.identity;h=e.next_h;edits=copy.deepcopy(e.edits)
        bad=literal_spec('log(x)/tau');bad['populations'][0]['states']['x']['initial']=1
        rejects(lambda:e.replace(bad));require(np.array_equal(e.read(),old) and e.model.identity==before and e.t==.01 and e.next_h==h and e.edits==edits,'migrated invalid RHS committed')
    result['transaction_boundary_rejects_CPU_GPU']=True
    for value in [-.2,1.5,float('inf'),2**63,True]:
        bad=base(2,1);bad['populations'][0]['cell_ids']=[0,value];rejects(lambda:Model(bad))
        bad=base(2,1);bad['connections'][0]={'source':['cells','release'],'target':['cells','w'],'pattern':'coo','row':[value],'col':[0],'weights':[1]};rejects(lambda:Model(bad))
        bad=base(2,1);bad['connections'][0]['offsets']=[value];rejects(lambda:Model(bad))
    for text in ['1000*mV','mV/1000','0*mV']:rejects(lambda:unit(text))
    require(unit('mV/s')=={'mV':1,'s':-1},'ordinary units')
    result['fractional_indices_scaled_units_rejected']=True
    e=Engine(base());e.advance(.01);e.checkpoint(out/'snapshot');r=Engine.restore(out/'snapshot');e.advance(.02);r.advance(.02)
    require(np.array_equal(e.read(),r.read()) and e.next_h==r.next_h,'same contract replay changed')
    saved=runtime.PROFILES['precise'];runtime.PROFILES['precise']=(1e-6,1e-9)
    rejects(lambda:Engine.restore(out/'snapshot'));runtime.PROFILES['precise']=saved
    meta=json.loads((out/'snapshot/checkpoint.json').read_text());meta['execution_contract']['source_hashes']['model.py']='0'*64
    raw=(json.dumps(meta,indent=2)+'\n').encode();(out/'snapshot/checkpoint.json').write_bytes(raw);(out/'snapshot/checkpoint.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    rejects(lambda:Engine.restore(out/'snapshot'))
    result['snapshot_profile_and_compiler_changes_rejected']=True
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
