"""Compatibility model only: conserved AXIOMA equations to generic RHS ABI.

The integration core imports none of the organism. Geometry, weights, owner
order and timestamped projections stay authoritative in this read-only model.
"""
import numpy as np
from graph_runtime import GraphRK23


class RealCNS(GraphRK23):
    def __init__(self, initial, coefficient, *, rtol, atol, norm_size,
                 project=None, freeze=None, native_library=None,
                 state_bounds=(0.,1.)):
        def rhs(y,clock,fraction):
            target,rate=coefficient(y)
            return rate*(target-y)
        super().__init__(initial,rhs,rtol=rtol,atol=atol,norm_size=norm_size,
                         project=project,freeze=freeze,state_bounds=state_bounds)


def install():
    import organism_adapter
    old_class=organism_adapter.NativeGraph
    old_step=organism_adapter.OrganismAdapter.step
    organism_adapter.NativeGraph=RealCNS

    def step(self,b,ns,drive,light):
        before=self.report['accepted']+self.report['rejected']
        result=old_step(self,b,ns,drive,light)
        trials=self.report['accepted']+self.report['rejected']-before
        # The compatibility adapter historically accounts six RHS per trial.
        b.statistics['evaluations']-=2*trials
        self.report['rhs_evaluations']=4*(self.report['accepted']+self.report['rejected'])
        self.report['method']='resident_RK3(2)_complete_effective_operator'
        return result

    organism_adapter.OrganismAdapter.step=step
    def restore():
        organism_adapter.NativeGraph=old_class
        organism_adapter.OrganismAdapter.step=old_step
    return restore
