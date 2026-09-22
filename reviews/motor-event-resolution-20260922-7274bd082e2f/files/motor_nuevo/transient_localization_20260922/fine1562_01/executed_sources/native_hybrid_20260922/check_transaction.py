"""Fail after a native accepted substep, recover private state, then continue."""
from pathlib import Path
import sys,json
import numpy as np,cupy as cp
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph
holder={}
def freeze(core):holder['core']=core;holder['inject']=cp.asarray([1.]);holder['rate']=cp.asarray([1000.])
def coefficient(z):
 core=holder['core'];bad=(core.clock[0]>=1e-4)&(holder['inject'][0]>0)
 return cp.where(bad,cp.nan,cp.ones_like(z)),holder['rate']
core=NativeGraph(np.array([0.]),coefficient,rtol=1e-5,atol=1e-7,norm_size=1,freeze=freeze)
try:
 core.advance(300000,100000,100,100000)
except RuntimeError as exc:failure=str(exc)
else:raise RuntimeError('Expected injected second-substep failure')
rolled=core.x.get(stream=core.stream)
if rolled[0]!=0:raise RuntimeError('Partial accepted state leaked')
with core.stream:holder['inject'].fill(0)
core.advance(300000,100000,100,100000);continued=core.x.get(stream=core.stream);core.close()
control=NativeGraph(np.array([0.]),lambda z:(cp.ones_like(z),cp.ones_like(z)*1000),rtol=1e-5,atol=1e-7,norm_size=1)
control.advance(300000,100000,100,100000);expected=control.x.get(stream=control.stream);control.close()
if not np.array_equal(continued,expected):raise RuntimeError('Continuation differs after rollback')
result=dict(injected_failure=failure,private_state_rollback_exact=True,continuation_exact=True,analytic_error=float(abs(continued[0]-(1-np.exp(-.3)))),scope='Native graph transaction only. Whole-organism cold loader replaces owners and requires runtime reconstruction.')
(HERE/'TRANSACTION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
