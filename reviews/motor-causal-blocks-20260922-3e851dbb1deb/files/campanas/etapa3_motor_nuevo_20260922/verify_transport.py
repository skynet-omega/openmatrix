"""Reconstruct transport evidence; compare delay functions on common knots.

Does not infer a long-time error bound, biology, or stage3 admission. Different
delay-history segment counts are expected under temporal refinement, but their
coverage, clocks, continuity and actual interpolated values must be checked.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'verification_vendor'))
from compare import compare,read_state,flatten,leaves

def histories(value,path=''):
 if isinstance(value,dict):
  if isinstance(value.get('history'),list):yield path,value
  for k,v in value.items():
   if k!='history':yield from histories(v,path+'/'+k)

def history_difference(a,b):
 def validate(x):
  h=x['history']
  if not h:return None
  for i,s in enumerate(h):
   if set(s)!={'start_ns','end_ns','left','right'} or type(s['start_ns']) is not int or type(s['end_ns']) is not int or s['end_ns']<=s['start_ns']:raise ValueError('Malformed delay segment')
   if not all(isinstance(s[k],np.ndarray) for k in ('left','right')):raise ValueError('Delay arrays required')
   if s['left'].shape!=s['right'].shape or s['left'].dtype!=s['right'].dtype or not np.isfinite(s['left']).all() or not np.isfinite(s['right']).all():raise ValueError('Invalid delay driver')
   if s['left'].shape!=h[0]['left'].shape or s['left'].dtype!=h[0]['left'].dtype:raise ValueError('Delay channel layout changed within history')
   if i and (h[i-1]['end_ns']!=s['start_ns'] or not np.array_equal(h[i-1]['right'],s['left'])):raise ValueError('Discontinuous delay history')
  return h
 ah,bh=validate(a),validate(b)
 if a['time_ns']!=b['time_ns'] or a['identity']!=b['identity']:raise ValueError('Delay clock/identity mismatch')
 if ah is None or bh is None:
  if ah!=bh:raise ValueError('Missing history')
  return {'max_abs':0.,'segments':[0,0]}
 if ah[0]['left'].shape!=bh[0]['left'].shape or ah[0]['left'].dtype!=bh[0]['left'].dtype:raise ValueError('Delay channel shape/dtype mismatch')
 # Future delayed inputs start at the current physical clock. A retained old
 # segment may start earlier; only its unconsumed suffix affects future state.
 left=a['time_ns'];right=ah[-1]['end_ns']
 if right!=bh[-1]['end_ns'] or ah[0]['start_ns']>left or bh[0]['start_ns']>left:raise ValueError('Incomplete delay coverage')
 knots=sorted({left,right}|{s[k] for h in (ah,bh) for s in h for k in ('start_ns','end_ns') if left<=s[k]<=right})
 def at(h,t):
  for s in h:
   if s['start_ns']<=t<=s['end_ns']:return s['left']+(s['right']-s['left'])*((t-s['start_ns'])/(s['end_ns']-s['start_ns']))
  raise ValueError('Gap in future history')
 error=max(float(np.max(abs(at(ah,t)-at(bh,t)),initial=0)) for t in knots)
 return {'max_abs':error,'segments':[len(ah),len(bh)],'future_interval_ns':[left,right],'knots':len(knots)}

def exact_fields(reference,candidate):
 """Discrete state, identities and physical clocks are not error tolerances.

 Only numerical integrator statistics and its proposed next step are exempt.
 Piecewise history boundaries are checked separately as functions.
 """
 a,b=leaves(reference),leaves(candidate);changed=[]
 for path in sorted(set(a)|set(b)):
  if '/history/' in path or '/statistics/' in path or path in ('/brain/next_step_ns','/brain/kc_apl_dynamic_state/coupling_steps'):continue
  if path not in a or path not in b:changed.append(path);continue
  x,y=a[path],b[path];leaf=path.rsplit('/',1)[-1]
  clock=leaf.endswith(('_time_ns','_elapsed_ns')) or leaf in ('time_ns','elapsed_ns','time_s')
  exact=clock or type(x) in (str,bool,int,type(None)) or type(y) in (str,bool,int,type(None))
  if isinstance(x,np.ndarray):exact=exact or x.dtype.kind not in 'fc'
  if exact:
   same=(isinstance(y,np.ndarray) and x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()) if isinstance(x,np.ndarray) else (type(x) is type(y) and x==y)
   if not same:changed.append(path)
 return changed

def verify(reference,candidate):
 contract=HERE/'verification_vendor/INTEGRATED_ADAPTIVE_CONTRACT.json'
 if hashlib.sha256(contract.read_bytes()).hexdigest()!='c9d284d89c33d620972f5ec282c72cb3626ecf105345d00034125adac53e821d':raise ValueError('Frozen criterion changed')
 r=compare(reference,candidate)
 a=dict(histories(read_state(Path(reference)/'brain_final')));b=dict(histories(read_state(Path(candidate)/'brain_final')))
 if set(a)!=set(b):raise ValueError('History owners differ')
 history={k:history_difference(a[k],b[k]) for k in a}
 unexpected=[p for p in r['layout_or_missing_paths'] if '/history/' not in p]
 flags=[p for p,x in flatten(read_state(Path(reference)/'brain_final')) if type(x) is bool]
 aa=dict(flatten(read_state(Path(reference)/'brain_final')));bb=dict(flatten(read_state(Path(candidate)/'brain_final')))
 changed_flags=[p for p in flags if p not in bb or type(bb[p]) is not bool or aa[p]!=bb[p]]
 changed_exact=exact_fields(reference,candidate)
 passed=r['screen_pass'] and not unexpected and not changed_flags and not changed_exact and all(x['max_abs']<=1e-4 for x in history.values())
 return {'reference':str(reference),'candidate':str(candidate),'verifier_revision':2,'metrics':r['metrics'],'limits':r['limits'],'exact_all_leaves':r['exact_all_leaves'],'histories':history,'history_normalized_limit':1e-4,'unexpected_layouts':unexpected,'changed_flags':changed_flags,'changed_exact_fields':changed_exact,'screen_pass':bool(passed),'scope':'Short numerical transport screen only. History channel layout, all discrete identities and physical clocks exact; no long-time certification.'}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--out',required=True);v=p.parse_args();r=verify(v.reference,v.candidate);Path(v.out).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r));raise SystemExit(0 if r['screen_pass'] else 1)
