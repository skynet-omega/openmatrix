"""Fine PN with explicit local Ca gates and a joint, restartable state.

Adding previously absent chemistry is a preparation, not recovery of its
unknown history. An uncoupled import therefore requires explicit Ca state at
the parent's clock and a preparation description hashed into the identity.
"""
import copy
import hashlib
import inspect
from pn_coupled_ionic import CoupledFinePNIonicSession
from pn_fine_ionic import FinePNIonicSession
from pn_calcium_port import CalciumPort, chemistry_stages, METHOD

SCHEMA='pn_voltage_calcium_coupled_v1'

class CalciumCoupledFinePN(CoupledFinePNIonicSession):
    def __init__(self,*args,calcium_port,preparation,**kwargs):
        if not isinstance(calcium_port,CalciumPort) or not isinstance(preparation,str) or not preparation:
            raise ValueError('Explicit calcium port and preparation required')
        calcium_port.assert_state()
        super().__init__(*args,**kwargs)
        self.uncoupled_identity=self.identity
        self.calcium_port=calcium_port;self.preparation=preparation
        C=self.backend.cpu.levels[0].C
        if (calcium_port.time_ns!=0 or (calcium_port.nodes>=len(C)).any()
                or (C[calcium_port.nodes]<=0).any()):
            raise ValueError('Fresh Ca port on physical membrane nodes required')
        self.identity=hashlib.sha256((self.identity+SCHEMA+METHOD+calcium_port.identity+preparation+
            inspect.getsource(chemistry_stages)+inspect.getsource(CalciumPort.stage)).encode()).hexdigest()
        self._calcium_definition=(calcium_port,preparation,self.identity)

    def _assert_model(self):
        super()._assert_model()
        if hasattr(self,'_calcium_definition'):
            if (self.calcium_port is not self._calcium_definition[0]
                    or (self.preparation,self.identity)!=self._calcium_definition[1:]):
                raise ValueError('Joint Ca preparation changed')
            self.calcium_port.assert_model()

    def advance(self,dt_ns,current_pA,**options):
        if '_local_channel' in options:raise ValueError('Session owns its calcium current')
        return super().advance(dt_ns,current_pA,_local_channel=self.calcium_port,**options)

    def state_dict(self):
        if self.calcium_port.time_ns!=self.time_ns:raise ValueError('PN/Ca clock mismatch')
        return dict(schema=SCHEMA,identity=self.identity,pn=super().state_dict(),calcium=self.calcium_port.state_dict())

    def load_state_dict(self,state):
        self._assert_model()
        if (set(state)!={'schema','identity','pn','calcium'} or state['schema']!=SCHEMA
                or state['identity']!=self.identity or state['pn']['time_ns']!=state['calcium']['time_ns']):
            raise ValueError('Joint identity or clocks mismatch')
        # Validate chemistry and allocate the replacement voltage before any
        # live state changes. The shallow shadow reuses the immutable operator.
        calcium=self.calcium_port.validated_state(state['calcium'])
        shadow=copy.copy(self)
        FinePNIonicSession.load_state_dict(shadow,state['pn'])
        self.calcium_port.apply_validated(calcium)
        self.voltage=shadow.voltage;self.gates=shadow.gates
        self.ionic_charge_pC=shadow.ionic_charge_pC;self.time_ns=shadow.time_ns

    def import_uncoupled_state(self,parent,*,calcium_state):
        if self.time_ns!=0 or self.cp.any(self.voltage) or self.ionic_charge_pC.any():
            raise ValueError('Import requires a fresh target')
        if parent.get('identity')!=self.uncoupled_identity:
            raise ValueError('Uncoupled parent anatomy/channels do not match')
        self.load_state_dict(dict(schema=SCHEMA,identity=self.identity,
            pn=dict(parent,identity=self.identity),calcium=calcium_state))
        return dict(source_identity=self.uncoupled_identity,target_identity=self.identity,time_ns=self.time_ns,
                    existing_physical_state_preserved=True,calcium_history_reconstructed=False,
                    preparation=self.preparation)
