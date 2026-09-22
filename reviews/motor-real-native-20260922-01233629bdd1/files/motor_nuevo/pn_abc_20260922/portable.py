"""Portable numerical PN capsule: full G/M, membrane, gates and calcium history."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.sparse import load_npz
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'vendor'))
from pn_mass_backend import FullMassBackend,immutable
from pn_mass_krylov_backend import FullMassKrylovBackend
from pn_graph_elimination_backend import FullMassGraphBackend
from pn_calcium_port import CalciumPort
from pn_calcium_release import CalciumReleaseSites
from pn_mass_coupled_step import advance_full_mass

def read(stem):
 stem=Path(stem);d=json.loads(stem.with_suffix('.json').read_text())
 with np.load(stem.with_suffix('.npz'),allow_pickle=False) as z:
  def decode(v):
   if isinstance(v,dict):
    if set(v)=={'__array__'}:return z[v['__array__']].copy()
    return {k:decode(x) for k,x in v.items()}
   if isinstance(v,list):return [decode(x) for x in v]
   return v
  return decode(d)
class PortablePN:
 def __init__(self,folder):
  folder=Path(folder);d=read(folder/'static');b=FullMassBackend(load_npz(folder/'G.npz'),load_npz(folder/'M.npz'),d['C'],provenance='PN numerical capsule of recorded unchanged G/M')
  self.backend=FullMassGraphBackend.adopt(FullMassKrylovBackend.adopt(b));self.cp=b.cp
  self.active_nodes=immutable(d['active']);self._nodes_gpu=self.active_nodes;self.gbar_nS=immutable(d['gbar']);self.reversal_mV=immutable(d['reversal']);self.leak_reversal_mV=d['leak']
  chem=CalciumReleaseSites(d['chem_ids'],d['chem_parameters'],provenance=d['chem_provenance'])
  self.calcium_port=CalciumPort(d['ca_nodes'],d['ca_gbar'],d['ca_reversal'],d['ca_half'],d['ca_slope'],d['ca_tau'],d['ca_power'],d['ca_sites'],d['ca_fractions'],chem,provenance=d['ca_provenance'],enabled=d['ca_enabled'],release_enabled=d['release_enabled'])
  self.definition=(self.active_nodes,self.gbar_nS,self.reversal_mV,self.calcium_port,self.leak_reversal_mV)
  self.restore(read(folder/'initial'))
 def _assert_model(self):
  self.backend.assert_model();self.calcium_port.assert_model()
  if any(a is not b for a,b in zip((self.active_nodes,self.gbar_nS,self.reversal_mV,self.calcium_port),self.definition[:4])) or self.leak_reversal_mV!=self.definition[4]:raise ValueError('PN model changed')
 def restore(self,state):
  self.voltage=state['voltage'].copy();self.gates=state['gates'].copy();self.ionic_charge_pC=state['charge'].copy();self.time_ns=state['time_ns'];self.calcium_port.load_state_dict(state['calcium'])
 def state(self):return dict(voltage=self.voltage.copy(),gates=self.gates.copy(),charge=self.ionic_charge_pC.copy(),time_ns=self.time_ns,calcium=self.calcium_port.state_dict())
 def advance(self,dt_ns,current,**options):return advance_full_mass(self,dt_ns,current,_local_channel=self.calcium_port,**options)

def errors(a,b):
 result={}
 def visit(x,y,key):
  if isinstance(x,dict):
   if set(x)!=set(y):raise ValueError('State layout differs')
   for k in x:visit(x[k],y[k],key+'/'+k)
  elif isinstance(x,np.ndarray):
   if x.shape!=y.shape or x.dtype!=y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('State array invalid')
   result[key]=float(np.max(abs(x-y),initial=0))
  elif x!=y:raise ValueError('Nonarray state differs '+key)
 visit(a,b,'');return result
