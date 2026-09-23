"""Reconstruct external-prototype verdicts from locally generated arrays."""
from pathlib import Path
import json,hashlib,copy,numpy as np
R=Path(__file__).resolve().parent
SHA='a43e828372091f7895992bf638e29a2d4ef75927d4bc875e2ec62103aa8ec287'

def require(x,m):
 if not x:raise ValueError(m)

def check(report,arrays,plan):
 require(report['plan']==plan,'Changed criteria')
 out=[]
 for k,(row,data) in enumerate(zip(report['modelos'],arrays)):
  require(row['caso']==k and data['referencia'].shape==data['iterado'].shape==data['global'].shape==data['congelado'].shape,'Layout changed')
  require(all(np.isfinite(x).all() for x in data.values()),'Nonfinite arrays')
  errors={m:float(np.max(abs(data[m]-data['referencia']))) for m in ('referencia','global','iterado','congelado')}
  require(errors==row['error_max'],'Errors not reconstructed')
  ok=errors['iterado']<=plan['error_externo'] and errors['global']<=plan['error_externo'] and errors['congelado']>plan['error_externo']
  require(type(row['contraste_correcto']) is bool and row['contraste_correcto']==ok,'Unjustified emitted flag')
  out.append({'case':k,'errors':errors,'contrast_pass':ok,'serial_B_over_global_time':row['medidas']['iterado']['pared_s']/row['medidas']['global']['pared_s'],'rhs_counts':{m:row['medidas'][m]['rhs'] for m in ('iterado','global','referencia','congelado')}})
 require(report['estado']==('COMPLETO' if all(x['contrast_pass'] for x in out) else 'FALLO_CONSERVADO'),'Invalid status')
 return out

def main():
 require(hashlib.sha256((R/'trayectorias_cpu.py').read_bytes()).hexdigest()==SHA,'Received code changed')
 from trayectorias_cpu import PLAN
 folder=R/'external_cpu_01';report=json.loads((folder/'RESULTADO.json').read_text());arrays=[]
 for k in (0,1):
  combined=dict(np.load(folder/f'modelo_{k}.npz',allow_pickle=False));arrays.append(combined)
  for mode in ('referencia','global','iterado','congelado'):
   one=np.load(folder/f'modelo_{k}_{mode}.npz',allow_pickle=False)
   require(np.array_equal(one['tiempo'],combined['tiempo']) and np.array_equal(one['estado'],combined[mode]),'Per-mode data mismatch')
 evidence=check(report,arrays,PLAN);corruption=[]
 for name in ('criterion','flag','raw_state'):
  r=copy.deepcopy(report);a=[dict(x) for x in arrays]
  if name=='criterion':r['plan']['error_externo']=1.
  elif name=='flag':r['modelos'][0]['contraste_correcto']=False
  else:a[0]['iterado']=a[0]['iterado'].copy();a[0]['iterado'][0,0]+=.1
  try:check(r,a,PLAN)
  except ValueError:corruption.append(name)
  else:raise RuntimeError('Missed deliberate corruption: '+name)
 bridge=R/'ir_bridge_01';br=json.loads((bridge/'RESULT.json').read_text());reference=np.load(bridge/'referencia.npz',allow_pickle=False);be={}
 for mode in ('referencia','global','iterado'):
  data=np.load(bridge/(mode+'.npz'),allow_pickle=False)
  require(np.array_equal(data['time'],reference['time']),'IR grid mismatch');be[mode]=float(np.max(abs(data['state']-reference['state'])))
 require(be==br['errors'] and br['limit']==1e-4 and max(be.values())<=1e-4 and br['status']=='PASS','IR bridge report mismatch')
 result={'code_sha256':SHA,'local_CPU_evidence':evidence,'IR_bridge_errors':be,'corruptions_detected':corruption,'scope':'Local raw arrays and criteria verified; external remote arrays not downloaded. No organism execution by ChatGPT.'}
 (R/'EXTERNAL_VERIFIED.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
