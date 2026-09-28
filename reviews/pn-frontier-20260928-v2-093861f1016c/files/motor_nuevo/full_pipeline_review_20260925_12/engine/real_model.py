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


def install(precision='fp64', *, audit_weights=False):
    import organism_adapter
    if precision not in ('fp64', 'fp32', 'persistent_fp32'):
        raise ValueError('unknown CNS precision')
    old_class=organism_adapter.NativeGraph
    old_step=organism_adapter.OrganismAdapter.step
    old_build=organism_adapter.OrganismAdapter.build
    old_close=organism_adapter.OrganismAdapter.close
    organism_adapter.NativeGraph=RealCNS

    def build(self,drive,light):
        if precision in ('fp32', 'persistent_fp32'):
            from fp32_operator import FastCSR
            self._fp32_original_kernel=self.brain.kernel
            self._fp32_operator=FastCSR(self.brain, persistent=precision=='persistent_fp32', audit=audit_weights)
            self.brain.kernel=self._fp32_operator
            if self._fp32_operator.mirror is not None:
                brain=self.brain
                from model_weight_writers import writer_layout
                self._writer_layout=writer_layout(brain)
                self._plastic_original=brain.sync_plastic_weights
                self._plastic_was_local='sync_plastic_weights' in brain.__dict__
                def sync_plastic_weights(*args,**kwargs):
                    import cupy as cp
                    if self.core is not None:self.core.stream.synchronize()
                    mirror=self._fp32_operator.mirror
                    try:
                        result=self._plastic_original(*args,**kwargs)
                        mirror.validate_source(brain.cuda['weights'])
                        mirror.refresh_all()
                        # Include later in-place rollback of unregistered plastic edges.
                        mirror.refresh_at_boundary=True
                        cp.cuda.get_current_stream().synchronize()
                        return result
                    except BaseException:
                        mirror.invalid=True
                        raise
                brain.sync_plastic_weights=sync_plastic_weights
        try:
            return old_build(self,drive,light)
        except BaseException:
            restore_operator(self)
            raise

    def restore_operator(self):
        if hasattr(self,'_fp32_original_kernel'):
            self.brain.kernel=self._fp32_original_kernel
        if hasattr(self,'_plastic_original'):
            if self._plastic_was_local:self.brain.sync_plastic_weights=self._plastic_original
            else:self.brain.__dict__.pop('sync_plastic_weights',None)

    def close(self):
        try:return old_close(self)
        finally:restore_operator(self)

    def step(self,b,ns,drive,light):
        if hasattr(self,'_fp32_operator') and self._fp32_operator.mirror is not None:
            from model_weight_writers import writer_layout
            if writer_layout(b)!=self._writer_layout:
                raise RuntimeError('Effective-weight writers changed: rebuild the captured operator')
            mirror=self._fp32_operator.mirror
            mirror.validate_source(b.cuda['weights'])
            if mirror.refresh_at_boundary:mirror.refresh_all()
        before=self.report['accepted']+self.report['rejected']
        result=old_step(self,b,ns,drive,light)
        trials=self.report['accepted']+self.report['rejected']-before
        # The compatibility adapter historically accounts six RHS per trial.
        b.statistics['evaluations']-=2*trials
        self.report['rhs_evaluations']=4*(self.report['accepted']+self.report['rejected'])
        self.report['method']='resident_RK3(2)_CSR_'+precision.upper()
        return result

    organism_adapter.OrganismAdapter.build=build
    organism_adapter.OrganismAdapter.close=close
    organism_adapter.OrganismAdapter.step=step
    def restore():
        organism_adapter.NativeGraph=old_class
        organism_adapter.OrganismAdapter.build=old_build
        organism_adapter.OrganismAdapter.close=old_close
        organism_adapter.OrganismAdapter.step=old_step
    return restore
