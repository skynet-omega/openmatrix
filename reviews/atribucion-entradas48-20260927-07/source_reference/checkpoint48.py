"""Save scientific owners and the new input owner at a committed boundary."""
from pathlib import Path
import json
import numpy as np


def save_checkpoint(folder,obj,motor,stimulus,auditor):
    from session_io import write_state
    from operator_state import OperatorState,LEGACY_BINDINGS
    from observations import sha,save,need,digest
    folder=Path(folder);temp=folder.with_name(folder.name+'.partial')
    need(not folder.exists(),'Preserve existing checkpoint')
    temp.mkdir(exist_ok=False)
    h=obj.core.hybrid
    need(not getattr(obj.core,'failed',False) and not getattr(h,'_native_rebuild_required',False),
         'Cannot save invalidated organism')
    before=digest(dict(state=h.state,qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,
        time_ns=h.time_ns,rng=obj.core.brain.rng.bit_generator.state,input=stimulus.state()))
    write_state(temp/'session',obj.core.state_dict())
    write_state(temp/'prosthesis',obj.state())
    write_state(temp/'published',dict(rates=obj.core.brain.rates,time_ns=obj.core.brain.time_ns,
        rng=obj.core.brain.rng.bit_generator.state))
    write_state(temp/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
    write_state(temp/'input_owner',stimulus.state())
    write_state(temp/'interval_owner',dict(origin_ns=auditor.origin_ns,pending=auditor.pending,
        previous_dn=auditor.previous_dn,baseline=auditor.baseline))
    write_state(temp/'motor_owner',{k:getattr(motor,k) for k in
        ('substeps','body_calls','trial_step','raw_forward','forward','raw_yaw')})
    write_state(temp/'stored_operator',dict(weights=obj.core.brain.W.data))
    save(temp/'boundary.json',dict(schema='zero_legacy_odor_prescribed_ORN_v1',
        legacy_origin_ns=obj.core.world.boundary.origin_ns,plan=stimulus.plan,arm=stimulus.arm))
    after=digest(dict(state=h.state,qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,
        time_ns=h.time_ns,rng=obj.core.brain.rng.bit_generator.state,input=stimulus.state()))
    need(before==after,'Checkpoint changed an owner')
    save(temp/'MANIFEST.json',dict(schema='composition48_scientific_state_v1',time_ns=int(h.time_ns),
        read_only_witness=before,restart_tested=False,
        scope='Scientific state, RNG, effective and stored weights, body, input and motor owner; resume harness not qualified.',
        files={p.name:sha(p) for p in temp.iterdir() if p.is_file()}))
    temp.rename(folder)
