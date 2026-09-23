"""Different dimensional equations through existing IR and unchanged scheduler."""
from pathlib import Path
import sys,json,copy,time,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'general_v2_20260922'))
from model import Model
from trayectorias_cpu import Sesion,Evento,PLAN,exigir
from ir_waveform_bridge import from_ir

def descriptor(name,unit,initial):
 scalar=lambda u,v:{'unit':u,'value':v}
 state=lambda i,r:{'unit':unit,'initial':i,'scale':1.,'rhs':r}
 if name=='voltage':
  eq=['k*(-x-0.2*y+ref*tanh(u))','k*(0.3*x-1.5*y)'];output='x/ref'
 else:
  eq=['k*(ref/(1+u*u)-x-0.1*x*y/ref)','k*(x-y)'];output='y/ref'
 return {'version':1,'populations':[{'id':name,'count':1,'states':{'x':state(initial[0],eq[0]),'y':state(initial[1],eq[1])},'parameters':{'k':scalar('1/s',2.),'ref':scalar(unit,1.)},'inputs':{'u':scalar('1',0.)},'outputs':{'out':{'unit':'1','expr':output}}}],
         'mass':{'row':[0,0,1,1],'col':[0,1,0,1],'values':[1.,.2,.2,1.4]}}

def main():
 out=Path(sys.argv[1]) if len(sys.argv)>1 else R/'ir_bridge_01';out.mkdir(exist_ok=False);start=time.perf_counter()
 specs=[descriptor('voltage','mV',[.2,-.1]),descriptor('chemistry','mM',[.4,.2])]
 (out/'DESCRIPTORS.json').write_text(json.dumps(specs,indent=2)+'\n')
 broken=copy.deepcopy(specs[0]);broken['populations'][0]['states']['x']['rhs']='k*u'
 try:Model(broken)
 except ValueError:pass
 else:raise RuntimeError('Dimensional mismatch accepted')
 models=[Model(x) for x in specs];baseline=[(m.initial.copy(),m.port_bias.copy(),m.identity) for m in models]
 for m in models:
  block=from_ir(m);view=copy.copy(m);view.port_bias=m.port_bias+.2
  require_error=float(np.max(abs(np.linalg.solve(block.masa,block.rhs(0.,m.initial,.2))-view.rhs(0.,m.initial))))
  exigir(require_error<=1e-12,'Mass applied twice or omitted in bridge')
 data={};stats={};events=[Evento(.17,0,0,'ADD',.2),Evento(.31,1,0,'SET',.3)]
 for mode in ('referencia','global','iterado'):
  blocks=[from_ir(models[0]),from_ir(models[1],(0.,10.))]
  s=Sesion(blocks,[[0.,.4],[.2,0.]],lambda t:np.zeros(2),events)
  ts=[];ys=[];tic=time.perf_counter()
  for end in (.125,.25,.375,.5):
   t,y=s.avanzar(end,mode,PLAN['iteraciones']);ts.extend(t);ys.extend(y)
  data[mode]=np.asarray(ys);stats[mode]={'wall_s':time.perf_counter()-tic,**s.contadores}
  np.savez_compressed(out/(mode+'.npz'),time=ts,state=data[mode])
  if mode=='referencia':times=np.asarray(ts)
  else:exigir(np.array_equal(ts,times),'Different comparison knots')
  exigir(s.publicado[2]==(0,1),'Lost event')
 for m,(initial,bias,identity) in zip(models,baseline):
  exigir(np.array_equal(m.initial,initial) and np.array_equal(m.port_bias,bias) and m.identity==identity,'Mutated descriptor')
 errors={k:float(np.max(abs(v-data['referencia']))) for k,v in data.items()}
 ok=all(np.isfinite(list(errors.values()))) and max(errors.values())<=1e-4
 result={'status':'PASS' if ok else 'FAIL','limit':1e-4,'errors':errors,'statistics':stats,'wall_s':time.perf_counter()-start,'invalid_units_rejected':True,'descriptor_unmodified':True,'mass_applied_once':True,'external_scheduler_unmodified':True,'scope':'CPU bridge,4states,0.5s,one scalar port,block-local mass. Not whole organism or speed claim.'}
 (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result));exigir(ok,'IR trajectory discrepancy')
if __name__=='__main__':main()
