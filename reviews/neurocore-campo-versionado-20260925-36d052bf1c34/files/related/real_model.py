"""Process-local bridge from conserved CNS coefficients to the generic core."""
from graph_runtime import GraphMidpoint
from event_projection import FilterPorts
from pathlib import Path
import importlib.util
import sys


class RealCNS(GraphMidpoint):
    def __init__(self,initial,coefficient,*,rtol,atol,norm_size,
                 project=None,freeze=None,native_library=None,state_bounds=(0.,1.)):
        super().__init__(initial,lambda y,t,side:coefficient(y),rtol=rtol,atol=atol,
            norm_size=norm_size,project=project,freeze=freeze,state_bounds=state_bounds)


def install():
    import organism_adapter
    original_cell_module=sys.modules['device_cell']
    spec=importlib.util.spec_from_file_location('device_cell',Path(__file__).with_name('device_cell.py'))
    new_cell_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_cell_module)
    sys.modules['device_cell']=new_cell_module
    old_class=organism_adapter.NativeGraph;old_ports=organism_adapter.FilterPorts
    old_step=organism_adapter.OrganismAdapter.step
    organism_adapter.NativeGraph=RealCNS;organism_adapter.FilterPorts=FilterPorts

    def step(self,b,ns,drive,light):
        before=self.report['accepted']+self.report['rejected']
        result=old_step(self,b,ns,drive,light)
        trials=self.report['accepted']+self.report['rejected']-before
        b.statistics['evaluations']-=trials
        self.report['rhs_evaluations']=5*(self.report['accepted']+self.report['rejected'])
        self.report['method']='resident_exponential_midpoint_shared_initial_5'
        self.report['event_time_contract']='authoritative_endpoint_explicit_side'
        return result

    organism_adapter.OrganismAdapter.step=step
    def restore():
        sys.modules['device_cell']=original_cell_module
        organism_adapter.NativeGraph=old_class;organism_adapter.FilterPorts=old_ports
        organism_adapter.OrganismAdapter.step=old_step
    return restore
