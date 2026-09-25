"""Process-local bridge from conserved CNS coefficients to the generic core."""
from graph_runtime import GraphMidpoint
from event_projection import FilterPorts


class RealCNS(GraphMidpoint):
    def __init__(self,initial,coefficient,*,rtol,atol,norm_size,
                 project=None,freeze=None,native_library=None,state_bounds=(0.,1.)):
        super().__init__(initial,lambda y,t,side:coefficient(y),rtol=rtol,atol=atol,
            norm_size=norm_size,project=project,freeze=freeze,state_bounds=state_bounds)


def install():
    import organism_adapter
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
        organism_adapter.NativeGraph=old_class;organism_adapter.FilterPorts=old_ports
        organism_adapter.OrganismAdapter.step=old_step
    return restore
