"""Reference refinement: alternating integer halves, no end-of-ms tiny step."""
import inspect,textwrap
from snapshot_execution_brain import GpuSnapshotExecutionBrain

def install(brain):
    old=GpuSnapshotExecutionBrain.advance
    source=textwrap.dedent(inspect.getsource(old))
    needle="ns = min(left, m['coupling_ns'])"
    if source.count(needle)!=1:raise ValueError('Reference source changed')
    source=source.replace(needle,"ns = min(left, 7812 if (dt_ns-left)%15625 == 0 else 7813)")
    scope=dict(old.__globals__);exec(compile(source,__file__,'exec'),scope)
    refined=scope['advance']
    def advance(self,dt_ns,drive,light):
        if self is not brain:return old(self,dt_ns,drive,light)
        if dt_ns%15625:raise ValueError('Refined reference needs whole original steps')
        return refined(self,dt_ns,drive,light)
    GpuSnapshotExecutionBrain.advance=advance
    def restore():GpuSnapshotExecutionBrain.advance=old
    return restore
