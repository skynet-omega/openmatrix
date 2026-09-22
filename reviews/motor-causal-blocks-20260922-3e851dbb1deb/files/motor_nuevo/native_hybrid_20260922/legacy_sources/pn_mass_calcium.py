"""Candidate PN with explicit full mass and unchanged local Na/K/Ca equations.

Exterior coordinates retain physical voltages; extra coordinates describe
internal charge modes. Constructing this session does not qualify a reduction.
"""
import hashlib,inspect
from pn_calcium_coupled import CalciumCoupledFinePN
from pn_mass_backend import FullMassBackend
from pn_mass_coupled_step import advance_full_mass

SCHEMA='pn_full_mass_voltage_calcium_v1'

class MassCalciumPN(CalciumCoupledFinePN):
    def __init__(self,backend,*,node_identity,**kwargs):
        if not isinstance(backend,FullMassBackend):raise ValueError('Explicit full-mass backend required')
        token=hashlib.sha256((SCHEMA+backend.identity+inspect.getsource(advance_full_mass)).encode()).hexdigest()
        super().__init__(backend,node_identity=node_identity+':mass:'+token,**kwargs)
    def advance(self,dt_ns,current_pA,**options):
        if '_local_channel' in options:raise ValueError('Session owns local calcium state')
        return advance_full_mass(self,dt_ns,current_pA,_local_channel=self.calcium_port,**options)
    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,coordinate_scope='Exterior voltage plus internal generalized charge coordinates',base=super().state_dict())
    def load_state_dict(self,saved):
        if (set(saved)!={'schema','identity','coordinate_scope','base'} or saved['schema']!=SCHEMA
            or saved['identity']!=self.identity or saved['coordinate_scope']!='Exterior voltage plus internal generalized charge coordinates'):
            raise ValueError('Wrong full-mass state or coordinate contract')
        super().load_state_dict(saved['base'])
